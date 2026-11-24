# Deployment

How the two Jetstream2 instances are set up right now, and a log of the changes made to them. The README has the general steps; this file says what is actually running. It names files and settings, never their values. It has no IP addresses, because this repository is public: ask Pavlos for them.

Owner: Pavlos (GPU host, Compose, ports and secrets). Update this file in the same PR as a change to either instance, or in a follow-up PR straight after a change made by hand.

## CPU instance (`team-3`)

The stack testers use.

| | |
| --- | --- |
| Checkout | `~/app-assignment-1`, on `main` |
| Compose project | `app-assignment-1`, **not** the README's `smart-gateway-prototype`. Use `docker compose -p app-assignment-1 ...` |
| Services | `gateway`, `postgres`, `proxy`, `prometheus`. Only the proxy publishes a port, 8080 |
| Settings | `env/.env`: `APP_ENVIRONMENT=production` and `APP_DATABASE_URL`. `env/postgres.env`: `POSTGRES_PASSWORD`. Both are mode 600 and ignored by Git. `env/compose.defaults.env` still supplies the non-secret defaults |
| Not set yet | `APP_OPERATOR_TOKEN`, so the operator routes answer 403. `APP_VLLM_SMALL_BASE_URL`, `APP_VLLM_LARGE_BASE_URL`, `APP_VLLM_API_KEY`, because the gateway does not call vLLM yet. `APP_JEV_MODE`, so Jev is the mock |
| Start at boot | `smart-gateway-cpu.service` (`deploy/autostart/`), installed with `PROJECT_NAME=app-assignment-1` |
| Other stacks | `guacamole` is Jetstream2's web desktop. Leave it alone |

Separate clones for branch work, so the deployed checkout stays on `main`:
- `~/eval-runner`: the evaluation runner
- `~/gpu-smoke`: vLLM smoke tests, a single-branch clone

The vLLM API key is in `~/.vllm_api_key`, mode 600.

**To check** the public address testers use: `USER_BRIEF.md` gives it, and a Jetstream2 public IP can change when an instance is shelved and unshelved. After an unshelve, compare the address in Exosphere with `USER_BRIEF.md`.

## GPU instance (`gateway-gpu`)

The two vLLM servers. Only the CPU instance can reach them, over the private network.

| | |
| --- | --- |
| Checkout | `~/app-assignment-1`, on `feat/autostart` until #113 merges, then `main` |
| Compose project | `smart-gateway-gpu` (`compose.gpu.yml`) |
| Servers | `vllm-small` (Qwen3-1.7B) on 8001 and `vllm-large` (Qwen3-4B) on 8002, bound to the private IP only |
| Settings | `env/gpu.env`: `VLLM_BIND_IP`, `MODEL_CACHE_DIR`, `VLLM_API_KEY`, and `HF_HUB_OFFLINE=1`, so starts never contact Hugging Face. Ignored by Git |
| Model weights | The attached volume at `/media/volume/gpu-models` (`MODEL_CACHE_DIR`) |
| Start at boot | `smart-gateway-gpu.service`. It checks the bind IP, the volume and the GPU, then starts `vllm-small`, then `vllm-large`. About 3 minutes |

## Change log

Newest first. Times are UTC.

### 2026-11-24

- **CPU: off the public settings.**
  - **Before:** the stack had run on `env/compose.defaults.env` alone since it was deployed, with the public throwaway database password and `APP_ENVIRONMENT` left at `local`. The database port was never published, so only containers on the Compose network could use that password.
  - **The change:** the password was changed with `ALTER USER` inside Postgres, then written to new `env/postgres.env` and `env/.env` files, with `APP_ENVIRONMENT=production`. The stack was recreated.
  - **Result:** the gateway now refuses to start with the public database password or the demo operator token. The stored history was kept, because the volume was never deleted. The smoke test passes.
- **CPU: updated from `78a52d9` to `main` at `1ed9c0b`** (#96 and #106). The only change to the running gateway is in `config.py`: optional vLLM settings, unset, and the demo-token refusal. The smoke test passes.
- **Both: start at boot (#113).** The units were installed on both instances, then the instances were shelved and unshelved. Both stacks came back in order with nobody logged in, and the GPU private IP did not change.
- **GPU: `HF_HUB_OFFLINE=1`** was confirmed in `env/gpu.env` and in both running containers.

### 2026-11-23

- **GPU: instance created** (GRID A100X-20C), and both vLLM servers started with an API key, bound to the private IP (#96). The model weights went onto an attached 50 GB volume. The API key was copied to the CPU instance's `~/.vllm_api_key`.

## Next changes, and who makes them

- **Verness, connecting the gateway to vLLM:** set `APP_VLLM_SMALL_BASE_URL`, `APP_VLLM_LARGE_BASE_URL` and `APP_VLLM_API_KEY` in the CPU instance's `env/.env`. Pavlos does this, because he holds the key and the addresses. The deployed gateway accepts these settings now, since #96 is deployed.
- **The operator token:** if operators need the routes on the deployment, set a long random `APP_OPERATOR_TOKEN` in `env/.env`. Never use the demo token; the gateway refuses it in production.
- **README deploy steps:** they still name `~/smart-ai-gateway-prototype` and project `smart-gateway-prototype`. Either the steps change to match the instance, or the instance moves to match them.
