# Toy example — fictional viral genome dataset

> **All data in this example are entirely fictional** and were generated
> programmatically for illustration purposes only. The sequences, taxa, and
> distances do not correspond to any real organism or database entry.

This example walks through a complete ReSeT run on a small synthetic dataset
that mimics a reference genome selection task. It uses four fictional
taxa and 34 fictional genome sequences with Mash-format pairwise distances.

## Dataset overview

| Taxon | Sequences | Description |
|---|---|---|
| Alpha | 10 | Tight cluster; all genomes are closely related |
| Beta  |  8 | Moderate within-taxon diversity |
| Gamma |  9 | Moderate within-taxon diversity |
| Delta |  7 | Moderate within-taxon diversity; positioned between Alpha–Beta–Gamma |

Each taxon has one or two "outlier" sequences placed further from the cluster
centre to give the optimiser a non-trivial coverage problem.

**Distance structure (mean ± std):**

| Pair | Mean distance |
|---|---|
| Within Alpha  | 0.047 ± 0.019 |
| Within Beta   | 0.058 ± 0.039 |
| Within Gamma  | 0.060 ± 0.046 |
| Within Delta  | 0.055 ± 0.036 |
| Alpha – Beta  | 0.610 ± 0.046 |
| Alpha – Gamma | 0.588 ± 0.044 |
| Alpha – Delta | 0.368 ± 0.050 |
| Beta – Gamma  | 0.598 ± 0.057 |
| Beta – Delta  | 0.366 ± 0.041 |
| Gamma – Delta | 0.309 ± 0.054 |

## Running the example

From the repository root:

```bash
python -m reset.solution \
    --clusters        examples/toy_example/clusters.csv \
    --distances       examples/toy_example/distances.tsv \
    --distance_format mash \
    --selection_cost  1e-1 \
    --scale           1e-5 \
    --num_processes   1 \
    --seed            42 \
    --output          examples/toy_example/selected.txt
```

Or simply:

```bash
bash examples/toy_example/run_example.sh
```

## Parameters and expected output

| Parameter | Value | Rationale |
|---|---|---|
| `--selection_cost` | `0.1` | Moderate per-genome cost; the optimiser will select a few representatives per taxon rather than one or all |
| `--scale` | `1e-5` | Very small inter-taxon similarity penalty; intra-taxon coverage dominates the objective |
| `--num_processes` | `1` | Single process is sufficient for a 34-genome dataset |

**Expected selected genomes** (7 total):

```
Alpha_003
Beta_001
Beta_005
Gamma_001
Gamma_004
Delta_002
Delta_006
```

## Regenerating the data

```bash
cd examples/toy_example
python generate_data.py
```

The script uses a fixed seed (`SEED = 42`) so the output is fully
reproducible. Modify `TAXON_COUNTS`, `SPREAD`, or `TAXON_CENTRES` in
`generate_data.py` to experiment with different dataset configurations.
