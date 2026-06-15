# ReSeT — Reference genome Selection Tool

**ReSeT** selects a minimal, representative subset of sequences from a pre-clustered collection. Given a pairwise distance matrix and cluster labels, it minimises an objective that penalises (1) the cost of selecting each sequence, (2) within-cluster coverage gaps, and (3) cross-cluster similarity between selected sequences. The result is a compact reference set that covers every taxon while avoiding redundancy.

## Installation

```bash
pip install reset-bio
```

## Quick start

```python
import numpy as np
from reset import Solution

# Distance matrix (n × n) and cluster assignments (integer, length n)
D = np.array([[0.0, 0.2, 0.8], [0.2, 0.0, 0.7], [0.8, 0.7, 0.0]])
clusters = np.array([0, 0, 1])

sol = Solution(D, clusters, selection_cost=0.1)
sol.local_search(max_runtime=10)
print(np.flatnonzero(sol.selection))   # indices of selected sequences
```

## Command-line interface

```bash
reset \
  --clusters     clusters.csv \
  --distances    distances.tsv \
  --distance_format mash \
  --selection_cost 1e-1 \
  --scale 1e-5 \
  --output selected.txt
```

Supported distance formats: `mash`, `sourmash_cosine`, `sourmash_jaccard`, `generic`.

## Distance formats

| Format | Description |
|--------|-------------|
| `mash` | Mash lower-triangular output (`mash triangle`) |
| `sourmash_cosine` | Sourmash CSV (column 12 = cosine distance) |
| `sourmash_jaccard` | Sourmash CSV (column 6 = Jaccard distance) |
| `generic` | Any delimited `(i, j, distance)` file |

## Multiprocessing

Use `Solution_shm` for parallel local search:

```python
from reset import Solution_shm
sol = Solution_shm(D, clusters, selection_cost=0.1)
sol.local_search(num_processes=4, max_runtime=60)
```

## Dashboard

An interactive Streamlit dashboard is included. Install with dashboard extras and launch:

```bash
pip install "reset-bio[dashboard]"
streamlit run $(python -c "import reset.dashboard.app as a; import os; print(os.path.abspath(a.__file__))")
```

The dashboard defaults to the bundled toy example — no configuration needed.

## License

MIT — see [LICENSE](https://github.com/JaspervB-tud/ReSeT/blob/main/LICENSE).
