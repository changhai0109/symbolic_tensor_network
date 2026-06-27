#!/bin/bash

SCRIPT_DIR="$PWD"
ASTRASIM_BIN=$SCRIPT_DIR/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware
WORKLOAD_DIR=$SCRIPT_DIR/workloads

SYSTEM=$SCRIPT_DIR/system.json
NETWORK=$SCRIPT_DIR/network.yml
MEMORY=$SCRIPT_DIR/memory.json

## build astrasim
bash $SCRIPT_DIR/astra-sim/build/astra_analytical/build.sh

WORKLOAD=$WORKLOAD_DIR/llama_dp8
"$ASTRASIM_BIN" \
    --workload-configuration="${WORKLOAD}" \
    --comm-group-configuration="${WORKLOAD}.json" \
    --system-configuration="${SYSTEM}" \
    --remote-memory-configuration="${MEMORY}" \
    --network-configuration="${NETWORK}" > llama_dp8.stdout &

WORKLOAD=$WORKLOAD_DIR/llama_tp8
"$ASTRASIM_BIN" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM}" \
    --remote-memory-configuration="${MEMORY}" \
    --comm-group-configuration="${WORKLOAD}.json" \
    --network-configuration="${NETWORK}" > llama_tp8.stdout &

WORKLOAD=$WORKLOAD_DIR/llama_dp4tp2_wsar
"$ASTRASIM_BIN" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM}" \
    --remote-memory-configuration="${MEMORY}" \
    --comm-group-configuration="${WORKLOAD}.json" \
    --network-configuration="${NETWORK}" > llama_dp4tp2_wsar.stdout &

wait
