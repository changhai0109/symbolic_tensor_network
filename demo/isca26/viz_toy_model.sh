#!/bin/bash

SCRIPT_DIR="$PWD"
WORKLOAD_DIR=$SCRIPT_DIR/workloads

WORKLOAD=$WORKLOAD_DIR/toy_arws.0
source $SCRIPT_DIR/.venv/bin/activate

chakra_visualizer --input_filename $WORKLOAD.et --output_filename $WORKLOAD.pdf

wait
