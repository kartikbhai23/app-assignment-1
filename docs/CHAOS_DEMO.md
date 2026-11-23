# Chaos demo runbook

A live run of the three simulated faults on one stack, with no restarts between them. For each fault the script shows the monitor noticing, the controller disabling the small model (`qwen3-1.7b`), traffic moving to `qwen3-4b`, and the small model coming back after the cooldown.

Everything here uses the mock backends. It shows that the loop reacts; it says nothing about how real Qwen models fail. Do not run it on the deployment testers use: it replaces the running stack and resets its history, and it turns on the operator routes with a public token.

## Before the demo

```bash
docker compose -f compose.yml -f compose.demo.yml up -d --build --wait
python3 scripts/smoke_compose.py http://localhost:8080   # optional check
```

`compose.demo.yml` sets a public throwaway operator token (the script knows it), the mock semantic baseline of 0, semantic actions on, and a 20 second cooldown. Open the Health screen at `http://localhost:8080/#/health` on the projector and keep it visible; it refreshes every 5 seconds.

## Run it

```bash
python3 scripts/chaos_demo.py                  # all three scenarios; each waits out the 20 second cooldown
python3 scripts/chaos_demo.py --scenario outage
python3 scripts/chaos_demo.py --prometheus     # also query Prometheus after each scenario
```

Scenarios: `bad_format`, `outage`, `label_swap`. Each one clears every fault and resets both models first, so you can rerun any of them in any order. The script exits with a non-zero status and `DEMO FAILED` if the stack does not behave as described below.

## What to say, per scenario

| Scenario | Fault | What you see | Monitor that catches it |
|---|---|---|---|
| `bad_format` | Small model corrupts every reply | Five replies fail the format check; the fifth line shows `CONTROLLER: disabled`; the next reply comes from `qwen3-4b` | Failure monitor (format failures) |
| `outage` | Every call to the small model raises an error | Every request is still answered, by `qwen3-4b`, with `selected_backend_error`; the controller disables the small model within about five calls | Failure monitor (call errors) |
| `label_swap` | Small model answers well-formed replies for the wrong intent | Every format check passes, so the failure monitor sees nothing; after about ten replies the semantic monitor sees the intents disagree with Jev's, and the controller disables the model | Semantic monitor |

After the disable, the script clears the fault and keeps sending the same message every two seconds. Replies keep coming from `qwen3-4b` until the cooldown ends, then `qwen3-1.7b` answers again (the controller re-enables a model when it next checks, so it takes a request to notice).

At the end the script prints the messages and seconds it measured in that run. They come from one run on mocks with a 20 second cooldown. Quote them as that, not as averages or as real-model results.

## Prometheus views

Prometheus publishes no port, by design. `--prometheus` waits one 15 second scrape and then prints these queries through `docker compose exec`. You can run any of them by hand:

```bash
docker compose exec prometheus wget -qO- 'http://127.0.0.1:9090/api/v1/query?query=gateway_model_disabled'
```

| Query | Shows |
|---|---|
| `gateway_model_disabled` | 1 while the controller holds a model out |
| `gateway_monitor_failure_rate` | Format and call failure rate in the window |
| `gateway_semantic_disagreement_rate` | Share of replies whose intent differs from Jev's |
| `sum by (action, model) (gateway_controller_interventions_total)` | Disables and re-enables so far |
| `sum by (model) (gateway_chat_requests_total)` | Which model answered |

Metrics are scraped every 15 seconds, so they trail the live screen by up to that long. Counters reset when the gateway restarts.

## After the demo

```bash
docker compose up -d --wait
```

This returns to the normal stack without the demo override (and without its public token).

## If something goes wrong

- `answered 403`: the stack was not started with `compose.demo.yml`, so the operator routes are off.
- `answered 401`: a different `APP_OPERATOR_TOKEN` is set. Export it in your shell before running the script.
- `was not disabled after 30 messages`: a baseline or threshold was changed by `env/.env`. Remove those lines and restart.
- Health shows old interventions: they come from Postgres and survive restarts. That is expected.
