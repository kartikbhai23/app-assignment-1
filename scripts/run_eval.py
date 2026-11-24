"""Run the offline evaluation and score the results.

    # One model called directly: real vLLM (needs VLLM_API_KEY), or the mock
    uv run python scripts/run_eval.py run --target small \\
        --base-url http://GPU_PRIVATE_IP:8001 --out eval/test-small.jsonl
    uv run python scripts/run_eval.py run --target large --mock \\
        --out eval/mock-large.jsonl

    # The whole gateway, through POST /v1/chat on a stack you run yourself
    uv run python scripts/run_eval.py run --target gateway --out eval/test-gateway.jsonl

    # Score one or more results files; combine small and large into the oracle
    uv run python scripts/run_eval.py score eval/test-small.jsonl eval/test-large.jsonl
    uv run python scripts/run_eval.py oracle eval/test-small.jsonl eval/test-large.jsonl

Datasets: `--dataset banking77-test` (the default) is the official Banking77
test set, 3,080 messages, downloaded at a pinned commit and checked against its
SHA-256, then cached in data/cache/ (ignored by git). `--dataset-csv PATH` is any
CSV with `text` and `intent` columns, such as the fresh set. Add `--split dev`
for a CSV with a `split` column, such as the dev split, to keep only those rows.

Results are JSONL: one `run` line with the settings, then one `item` line per
message, keyed by its row in the dataset. They hold no message text and no
reply text. A run that stops can be started again with the same command, from
the same git commit; it skips the rows already in the file.

Failed calls come in two kinds, and both count as wrong:
- Final: what a customer would have got. Invalid output from the model
  (`BackendOutputError`), or an error status from the gateway (`unavailable`
  for a 503, `HTTP422` and so on). A rerun keeps these.
- Retried: a problem with the harness, such as a timeout or a refused
  connection. A rerun sends these messages again.

The gateway target changes the stack it runs against (the monitor counts every
reply, and requests are stored), so it refuses any host but localhost unless
--allow-remote is given. Never run it on the deployment testers use.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import TypeAdapter

from app_assignment_1.backends import MockBackend
from app_assignment_1.evaluation import (
    BANKING77_SHA256,
    BANKING77_URL,
    CACHE_DIR,
    Dataset,
    Message,
    oracle,
    parse_banking77,
    parse_labelled_csv,
    reply_format_errors,
    score,
    sha256,
)
from app_assignment_1.intents import load_intents
from app_assignment_1.labels import UNKNOWN, LabelSet, load_labels
from app_assignment_1.schemas import BackendConfig, BackendReply, PolicyCard
from app_assignment_1.vllm import BackendOutputError, VLLMBackend, build_system_prompt

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
ROUTES = {"small": "standard", "large": "complex"}
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

Item = dict[str, Any]
Call = Callable[[Message], Awaitable[Item]]


def load_config() -> tuple[LabelSet, dict[str, PolicyCard], list[BackendConfig]]:
    labels = load_labels(CONFIG / "labels.json")
    cards = TypeAdapter(dict[str, PolicyCard]).validate_json(
        (CONFIG / "policy_cards.json").read_text()
    )
    backends = TypeAdapter(list[BackendConfig]).validate_json(
        (CONFIG / "backends.json").read_text()
    )
    return labels, cards, backends


def banking77_test(labels: LabelSet, cache: Path = ROOT / CACHE_DIR) -> Dataset:
    expected = BANKING77_SHA256["test"]
    path = cache / f"banking77-test-{expected[:12]}.csv"
    if path.exists() and sha256(path.read_bytes()) == expected:
        body = path.read_bytes()
    else:
        response = httpx.get(
            BANKING77_URL.format(name="test"), timeout=30, follow_redirects=True
        )
        body = response.raise_for_status().content
        if sha256(body) != expected:
            raise SystemExit("Banking77 test.csv does not match the pinned SHA-256")
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return Dataset("banking77-test", expected, tuple(parse_banking77(body, labels)))


def csv_dataset(path: Path, labels: LabelSet, split: str | None = None) -> Dataset:
    body = path.read_bytes()
    try:
        messages = parse_labelled_csv(body, labels, split)
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from None
    name = path.stem if split is None else f"{path.stem}:{split}"
    return Dataset(name, sha256(body), tuple(messages))


def git_commit() -> str:
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=True
            )
            .stdout.decode()
            .strip()
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def backend_call(
    backend: MockBackend | VLLMBackend, labels: LabelSet, cards: dict[str, PolicyCard]
) -> Call:
    async def call(message: Message) -> Item:
        start = time.perf_counter()
        reply = BackendReply.model_validate(await backend.generate(message.text))
        errors = reply_format_errors(reply, labels, cards)
        return {
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "model": backend.config.name,
            "raw_intent": reply.intent if reply.intent in labels.names else None,
            "format_ok": not errors,
            "format_errors": errors,
            # What the customer would see: a failed check gets the fallback
            # apology, which carries `unknown`.
            "predicted": UNKNOWN if errors else reply.intent,
        }

    return call


def gateway_call(client: httpx.AsyncClient) -> Call:
    async def call(message: Message) -> Item:
        start = time.perf_counter()
        response = await client.post("/v1/chat", json={"message": message.text})
        if response.status_code == 503 or 400 <= response.status_code < 500:
            raise GatewayAnswered(response.status_code)
        response.raise_for_status()
        body = response.json()
        trace = body["trace"]
        return {
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "model": trace["model"],
            "route": trace["route"],
            "router": trace["router"],
            "reply_source": trace["reply_source"],
            "format_ok": trace["format_check"] == "passed",
            "format_errors": trace["format_errors"],
            "fallback_reason": trace["fallback_reason"],
            "controller_action": trace["controller_action"],
            "predicted": body["intent"],
        }

    return call


class GatewayAnswered(Exception):
    """The gateway answered with an error status: what the customer got.

    A 503 means no backend could answer; a 4xx means it refused the message
    (empty, or too long). Other 5xx come from the proxy or a crash and are
    retried, like a refused connection."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.code = "unavailable" if status == 503 else f"HTTP{status}"


# Failures that belong to the system being measured, not to the harness. They
# are kept as results; every other exception is retried on the next run.
FINAL_ERRORS: tuple[type[Exception], ...] = (GatewayAnswered, BackendOutputError)


def error_code(exc: Exception) -> str:
    # The type or status only: provider messages can echo the input.
    return exc.code if isinstance(exc, GatewayAnswered) else type(exc).__name__


def read_results(path: Path) -> tuple[dict[str, Any], list[Item]]:
    header: dict[str, Any] | None = None
    items: dict[int, Item] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                # Cut off by a kill or a full disk; a rerun sends that row again.
                print(f"{path}: skipping a cut-off last line", file=sys.stderr)
                continue
            raise SystemExit(f"{path} line {number} is not valid JSON") from None
        if record["type"] == "run":
            header = header or record
        else:
            items[record["row"]] = record  # a rerun row replaces the earlier one
    if header is None:
        raise SystemExit(f"{path} has no run line")
    return header, [items[row] for row in sorted(items)]


async def run_items(
    messages: list[Message],
    call: Call,
    write: Callable[[Item], None],
    concurrency: int,
    model: str | None = None,
) -> None:
    limit = asyncio.Semaphore(concurrency)
    done = 0

    async def one(message: Message) -> None:
        nonlocal done
        async with limit:
            item: Item = {"type": "item", "row": message.row, "gold": message.gold}
            try:
                item |= await call(message)
                item["error"] = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                item |= {
                    "model": model,
                    "predicted": None,
                    "error": error_code(exc),
                    "final": isinstance(exc, FINAL_ERRORS),
                }
            write(item)
            done += 1
            if done % 100 == 0 or done == len(messages):
                print(f"  {done}/{len(messages)}", file=sys.stderr)

    await asyncio.gather(*(one(message) for message in messages))


async def run(args: argparse.Namespace) -> int:
    labels, cards, backends = load_config()
    dataset = (
        csv_dataset(args.dataset_csv, labels, args.split)
        if args.dataset_csv
        else banking77_test(labels)
    )
    messages = list(dataset.messages)
    if args.limit:
        messages = messages[: args.limit]

    header: dict[str, Any] = {
        "type": "run",
        "dataset": dataset.name,
        "dataset_sha256": dataset.sha256,
        "dataset_size": len(dataset.messages),
        "target": args.target,
        "option_set_version": labels.option_set_version,
    }
    client: httpx.AsyncClient | None = None
    if args.target == "gateway":
        host = urlparse(args.base_url).hostname
        if host not in LOCAL_HOSTS and not args.allow_remote:
            raise SystemExit(
                f"Refusing {host}: the gateway target changes the stack it runs "
                "against. Pass --allow-remote for a stack no one else uses."
            )
        client = httpx.AsyncClient(base_url=args.base_url, timeout=60)
        call = gateway_call(client)
        header |= {"model": "routed", "mocked": None}
    else:
        config = next(b for b in backends if b.route == ROUTES[args.target])
        if args.mock:
            backend: MockBackend | VLLMBackend = MockBackend(
                config, load_intents(CONFIG / "intents.json"), cards
            )
        else:
            if not args.base_url:
                raise SystemExit("--base-url is required for a real model")
            key = os.environ.get("VLLM_API_KEY")
            if not key:
                raise SystemExit(
                    "Set VLLM_API_KEY (on the CPU instance: "
                    "export VLLM_API_KEY=$(cat ~/.vllm_api_key))"
                )
            client = httpx.AsyncClient(
                base_url=args.base_url,
                timeout=60,
                headers={"Authorization": f"Bearer {key}"},
            )
            backend = VLLMBackend(
                config,
                client,
                model=config.name,
                system_prompt=build_system_prompt(labels, cards),
                labels=labels,
            )
        call = backend_call(backend, labels, cards)
        header |= {"model": config.name, "mocked": args.mock}

    out: Path = args.out
    done_rows: set[int] = set()
    if out.exists():
        old, items = read_results(out)
        keys = (
            "dataset",
            "dataset_sha256",
            "target",
            "model",
            "mocked",
            "option_set_version",
        )
        if any(old.get(key) != header[key] for key in keys):
            raise SystemExit(
                f"{out} holds a different run; use another --out to start a new one"
            )
        commit = git_commit()
        if old.get("git_commit") != commit:
            raise SystemExit(
                f"{out} was started at commit {old.get('git_commit')}, and this "
                f"checkout is at {commit}. Check out that commit to finish it, or "
                "use another --out to start a new run."
            )
        done_rows = {
            item["row"] for item in items if item["error"] is None or item.get("final")
        }
    todo = [m for m in messages if m.row not in done_rows]
    print(
        f"{dataset.name}: {len(messages)} messages, {len(messages) - len(todo)} "
        f"already done, target {args.target} ({header['model']})",
        file=sys.stderr,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        body = out.read_bytes()
        if body and not body.endswith(b"\n"):
            # Drop the cut-off last line; its row is in `todo` and runs again.
            out.write_bytes(body[: body.rfind(b"\n") + 1])
    with out.open("a", encoding="utf-8") as file:
        if not done_rows and out.stat().st_size == 0:
            header |= {
                "git_commit": git_commit(),
                "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            file.write(json.dumps(header) + "\n")

        def write(item: Item) -> None:
            file.write(json.dumps(item) + "\n")
            file.flush()

        try:
            model = None if args.target == "gateway" else header["model"]
            await run_items(todo, call, write, args.concurrency, model)
        finally:
            if client is not None:
                await client.aclose()
    print_scores([out])
    return 0


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_scores(paths: list[Path]) -> None:
    labels, _, _ = load_config()
    for path in paths:
        header, items = read_results(path)
        result = score(items, labels)
        low, high = result["accuracy_95ci"]
        coverage = f"{result['messages']}/{header['dataset_size']}"
        print(
            f"{path.name}: {header['dataset']}, target {header['target']} "
            f"({header['model']}{', mock' if header.get('mocked') else ''})\n"
            f"  messages {coverage}  accuracy {_fmt(result['accuracy'])} "
            f"(95% CI {_fmt(low)}-{_fmt(high)})  macro F1 {_fmt(result['macro_f1'])}\n"
            f"  errors {result['errors'] or 0}  format failures "
            f"{result['format_failures']}  predicted unknown "
            f"{result['predicted_unknown']}  models {result['models']}\n"
            f"  latency p50 {_fmt(result['latency_ms_p50'], 0)} ms  "
            f"p95 {_fmt(result['latency_ms_p95'], 0)} ms"
        )
        retry = sum(bool(i.get("error")) and not i.get("final") for i in items)
        if result["messages"] < header["dataset_size"] or retry:
            print(
                f"  incomplete ({retry} failed calls to retry): rerun the same "
                "command to finish it"
            )


def cmd_score(args: argparse.Namespace) -> int:
    if args.json:
        labels, _, _ = load_config()
        report = {}
        for path in args.files:
            header, items = read_results(path)
            report[str(path)] = {"run": header, "score": score(items, labels)}
        print(json.dumps(report, indent=2))
    else:
        print_scores(args.files)
    return 0


def cmd_oracle(args: argparse.Namespace) -> int:
    labels, _, _ = load_config()
    small_header, small = read_results(args.small)
    large_header, large = read_results(args.large)
    if (small_header["target"], large_header["target"]) != ("small", "large"):
        raise SystemExit("Pass the small results first, then the large results")
    for key in ("dataset", "dataset_sha256", "option_set_version", "mocked"):
        if small_header.get(key) != large_header.get(key):
            raise SystemExit(f"The two results differ in {key}")
    combined = oracle(small, large)
    result = score(combined, labels)
    large_share = sum(item.get("model") == large_header["model"] for item in combined)
    print(
        f"oracle over {small_header['model']} and {large_header['model']}: "
        f"accuracy {_fmt(result['accuracy'])}  macro F1 {_fmt(result['macro_f1'])}  "
        f"large model on {large_share}/{len(combined)} messages"
    )
    retry = sum(bool(i.get("error")) and not i.get("final") for i in [*small, *large])
    if retry:
        print(f"  {retry} failed calls to retry: rerun both targets first")
    if args.out:
        header = {
            **small_header,
            "target": "oracle",
            "model": "oracle",
            # The two runs can come from different commits; keep both.
            "git_commit": {
                "small": small_header.get("git_commit"),
                "large": large_header.get("git_commit"),
            },
        }
        lines = [json.dumps(header)] + [json.dumps(item) for item in combined]
        args.out.write_text("\n".join(lines) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="send each message to one target")
    p.add_argument("--target", choices=("small", "large", "gateway"), required=True)
    data = p.add_mutually_exclusive_group()
    data.add_argument(
        "--dataset", choices=("banking77-test",), default="banking77-test"
    )
    data.add_argument("--dataset-csv", type=Path, help="CSV with text,intent")
    p.add_argument(
        "--split", help="with --dataset-csv: only rows whose split column is this"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-url", help="vLLM server, or the gateway's public proxy")
    p.add_argument("--mock", action="store_true", help="use the mock model")
    p.add_argument("--limit", type=int, help="only the first N messages")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--allow-remote", action="store_true")

    s = sub.add_parser("score", help="score results files")
    s.add_argument("files", type=Path, nargs="+")
    s.add_argument("--json", action="store_true", help="full scores, per class too")

    o = sub.add_parser("oracle", help="combine small and large results")
    o.add_argument("small", type=Path)
    o.add_argument("large", type=Path)
    o.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        if args.split and not args.dataset_csv:
            parser.error("--split needs --dataset-csv")
        if args.target == "gateway":
            args.base_url = args.base_url or "http://127.0.0.1:8080"
            if args.mock:
                parser.error("--mock is for the small and large targets")
        return asyncio.run(run(args))
    if args.command == "score":
        return cmd_score(args)
    return cmd_oracle(args)


if __name__ == "__main__":
    sys.exit(main())
