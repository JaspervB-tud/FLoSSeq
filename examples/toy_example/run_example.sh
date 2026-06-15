#!/usr/bin/env bash
# Run ReSeT on the toy example.

# Parameters:
#   --selection_cost 1e-1   cost for including each genome in the reference set
#   --scale          1e-5   small penalty on inter-taxon similarity (keeps representatives distinct across taxa but lets intra-taxon coverage dominate the objective)
#   --num_processes  1      single core 

SCRIPT_DIR="examples/toy_example"

python3 -m reset.solution \
    --clusters        "${SCRIPT_DIR}/clusters.csv" \
    --distances       "${SCRIPT_DIR}/distances.tsv" \
    --distance_format mash \
    --selection_cost  1e-1 \
    --scale           1e-5 \
    --num_processes   1 \
    --seed            42 \
    --output          "${SCRIPT_DIR}/selected.txt"
