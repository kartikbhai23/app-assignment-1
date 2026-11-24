#!/bin/sh
# Checks the GPU host is ready before smart-gateway-gpu.service starts vLLM.
# Run from the checkout. Exits non-zero, with the reason, if something is not
# ready yet; systemd then tries again (see the unit's RestartSec).
# Prints no values from env/gpu.env except the IP and the cache path.
set -eu

# The last `KEY=value` line wins, as in Compose; surrounding quotes are dropped.
setting() { sed -n "s/^$1=//p" env/gpu.env | tail -n 1 | sed 's/^"\(.*\)"$/\1/'; }
ip=$(setting VLLM_BIND_IP)
cache=$(setting MODEL_CACHE_DIR)

# The private IP the ports bind to must belong to this host. An unshelve can
# hand out a new one; then update VLLM_BIND_IP and the CPU side's URLs.
if ! ip -4 -o addr show | grep -qwF "inet $ip"; then
    echo "VLLM_BIND_IP=$ip is not an address of this host; update env/gpu.env" >&2
    exit 1
fi

# The model cache must be the attached volume, mounted and holding the
# weights. An empty directory would make vLLM download about 12 GB onto the
# root disk, which has no room for it.
if [ ! -d "$cache/hub" ]; then
    echo "MODEL_CACHE_DIR=$cache has no hub/ folder: is the volume mounted?" >&2
    exit 1
fi

# The GPU and its driver must be up.
if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "nvidia-smi cannot see the GPU yet" >&2
    exit 1
fi
