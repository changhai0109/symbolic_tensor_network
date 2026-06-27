#!/bin/bash

RUN_DIR="$PWD"
WORKLOAD_DIR="$RUN_DIR/workloads"
mkdir -p $WORKLOAD_DIR

cd ./stage
source .venv/bin/activate

# generate toy llama dp=2 tp=2 with WS
python main.py \
    --output_dir $WORKLOAD_DIR \
    --output_name toy_ws \
    --model_type dense \
    --dp 2 --tp 2 --sp 1 --pp 1 --tpsp 1\
    --weight_sharded 1  --activation_recompute 0 \
    --head 32 --kvhead 8 \
    --batch 32 --micro_batch 32 --seq 1024 --num_stacks 2 &

# generate llama dp=8
python main.py \
    --output_dir $WORKLOAD_DIR \
    --output_name llama_dp8 \
    --model_type dense \
    --dp 8 --tp 1 --sp 1 --pp 1 --tpsp 1\
    --weight_sharded 0  --activation_recompute 0 \
    --head 32 --kvhead 8 \
    --batch 32 --micro_batch 32 --seq 1024 &

# generate llama tp=8
python main.py \
    --output_dir $WORKLOAD_DIR \
    --output_name llama_tp8 \
    --model_type dense \
    --dp 1 --tp 8 --sp 1 --pp 1 --tpsp 1\
    --weight_sharded 0  --activation_recompute 0 \
    --head 32 --kvhead 8 \
    --batch 32 --micro_batch 32 --seq 1024 &

# generate llama dp4tp2_wsar
python main.py \
    --output_dir $WORKLOAD_DIR \
    --output_name llama_dp4tp2_wsar \
    --model_type dense \
    --dp 4 --tp 2 --sp 1 --pp 1 --tpsp 1\
    --weight_sharded 1  --activation_recompute 1 \
    --head 32 --kvhead 8 \
    --batch 32 --micro_batch 32 --seq 1024 
