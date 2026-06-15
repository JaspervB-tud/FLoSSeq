# ReSeT — Reference genome Selection Tool

ReSeT selects a **representative subset** of items from a collection that is partitioned into clusters (e.g. taxonomic lineages, clades, or any user-defined groups). Given pairwise distances between items, it minimises an objective that balances three competing costs:

1. **Selection cost** — a fixed cost for every selected item.
2. **Intra-cluster coverage** — for every unselected item, the distance to the nearest selected item in the same cluster.
3. **Inter-cluster diversity** — for every pair of clusters, the similarity between the two closest selected items across the pair (penalising ambiguity across clusters).

The optimization is solved with a **local search** algorithm (add / swap / double-swap / remove moves) that supports both single-process and parallel multi-process execution via shared memory.

---

## Installation

```bash
pip install reset-bio
```

ReSeT requires Python ≥ 3.10 and depends only on NumPy and SciPy.

### Optional dashboard dependencies

```bash
pip install "reset-bio[dashboard]"
```

---

## Quick start (Python API)

```python
import numpy as np
from reset import Solution

# Distance matrix (values in [0, 1]) and cluster assignments
distances = np.array([[0.0, 0.1, 0.9],
                      [0.1, 0.0, 0.8],
                      [0.9, 0.8, 0.0]])
clusters  = np.array([0, 0, 1])   # two clusters

# Generate an initial solution and run local search
sol = Solution.generate_medoid_solution(distances, clusters, selection_cost=1.0)
sol.local_search(max_runtime=60)

print("Selected indices:", np.flatnonzero(sol.selection))
print("Objective value:", sol.objective)
```

For large datasets, use `Solution_shm` (shared-memory backend) to parallelise the search across multiple CPU cores:

```python
from reset import Solution_shm

with Solution_shm.generate_random_solution(distances, clusters, seed=42) as sol:
    sol.local_search(num_processes=8, max_runtime=3600)
    print("Selected indices:", np.flatnonzero(sol.selection))
```

### Streaming distances

For very large datasets where the full distance matrix does not fit in memory, you can pass a **generator** that yields `(i, j, distance)` tuples:

```python
def my_distance_stream():
    # yield pairwise (i, j, dist) — only lower triangle needed
    for i, j, dist in ...:
        yield i, j, dist

sol = Solution.generate_random_solution(
    distances=my_distance_stream(),
    clusters=clusters,
)
```

---

## Command-line interface

ReSeT ships a CLI entry point for running the full pipeline from pre-computed distance files.

```
python -m reset.solution [OPTIONS]
```

### Required arguments

| Argument | Description |
|---|---|
| `--clusters PATH` | Clusters/labels file (see [format](#clusters-file) below). |
| `--distances PATH` | Pairwise distances file in Mash or Sourmash format. |
| `--sequences_mapping PATH` | *(Optional)* Index-to-ID mapping file (see [format](#sequences-mapping-file) below). |

### Clusters file

A delimited text file where each row maps an item ID to its cluster label:

```
seq001,Cluster_A
seq002,Cluster_A
seq003,Cluster_B
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--id_col` | `0` | Column index (0-based) of the item ID. |
| `--cluster_col` | `1` | Column index (0-based) of the cluster label. |
| `--delimiter` | `,` | Field delimiter character. |
| `--header` | *(flag)* | Skip the first line if a header is present. |

### Sequences mapping file *(optional)*

When the distance file uses integer indices that differ from the IDs in the clusters file (e.g. sequences were renamed to shorter labels before running Mash/Sourmash), provide a tab-separated mapping:

```
0	seq001
1	seq002
2	seq003
```

When `--sequences_mapping` is omitted, the row order of the clusters file is used directly as the index — row 0 maps to index 0, row 1 to index 1, and so on. This requires that the clusters file rows are in the same order as the items in the distance file.

### Distance file formats

Select the format with `--distance_format`:

| Value | Description |
|---|---|
| `sourmash_cosine` | Sourmash `compare --csv` output, cosine similarity (column 12) |
| `sourmash_jaccard` | Sourmash `compare --csv` output, Jaccard similarity (column 6) |
| `mash` *default* | Mash `dist` lower-triangular output |
| `generic` | Any delimited file with `(index1, index2, distance)` columns |

The `generic` format accepts any pairwise distance file where each row contains two integer indices and a distance value. Only the lower triangle (or any subset of pairs) needs to be present. Configure it with:

| Argument | Default | Description |
|---|---|---|
| `--dist_idx1_col` | `0` | Column index of the first item's integer index. |
| `--dist_idx2_col` | `1` | Column index of the second item's integer index. |
| `--dist_dist_col` | `2` | Column index of the distance value. |
| `--dist_delimiter` | `\t` | Field delimiter. |
| `--dist_header` | *(flag)* | Skip the first line if a header is present. |

When using the Python API directly, any generator yielding `(i, j, distance)` tuples can be passed as the `distances` argument to `Solution` or `Solution_shm`, giving full flexibility over how distances are computed or retrieved.

### Objective options

| Argument | Default | Description |
|---|---|---|
| `--selection_cost` | `1.0` | Fixed cost per selected item. |
| `--scale` | *(none)* | Scaling factor applied to the inter-cluster similarity term. If omitted, the term is included unscaled. |

### Solver options

| Argument | Default | Description |
|---|---|---|
| `--seed` | `12345` | Random seed for initial solution generation. |
| `--max_fraction` | `0.5` | Maximum fraction of items in the random initial solution. |
| `--max_iterations` | `10000000` | Maximum number of accepted local search moves. |
| `--max_runtime` | `3600` | Wall-clock time limit in seconds. |
| `--doubleswap_time_threshold` | `60.0` | Per-iteration time (seconds) beyond which double-swap moves are skipped. |
| `--num_processes` | `8` | Number of parallel worker processes. Set to `1` for single-process mode. |

### Output

| Argument | Default | Description |
|---|---|---|
| `--output PATH` | *(none)* | Write selected item IDs (one per line) to this file. If omitted, results are printed to stdout only. |

### Example

See [examples/toy_example/](examples/toy_example/) for a working end-to-end example with fictional data. The command used there is:

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

---

## Examples

| Example | Description |
|---|---|
| [`examples/toy_example/`](examples/toy_example/) | **Fictional** 34-genome dataset across 4 viral taxa in Mash format. Demonstrates the full CLI pipeline including cluster file, distance file, and output. |

---

## Python API reference

### `Solution`

Single-process solution class. Suitable for small-to-medium datasets.

```python
Solution(distances, clusters, selection=None, selection_cost=1.0,
         cost_per_cluster=0, scale=None, seed=None)
```

**Class methods for initialisation:**

| Method | Description |
|---|---|
| `Solution.generate_random_solution(distances, clusters, ...)` | Random initial solution. |
| `Solution.generate_medoid_solution(distances, clusters, ...)` | Selects the medoid of each cluster as the initial solution. |

**Key methods:**

| Method | Description |
|---|---|
| `local_search(max_iterations, max_runtime, ...)` | Run local search. Returns `(time_per_iteration, objectives, components)`. |
| `determine_feasibility()` | Check whether every cluster has at least one selected item. |
| `calculate_objective()` | Recompute and return the objective value from scratch. |

**Key attributes:**

| Attribute | Description |
|---|---|
| `selection` | Boolean array — `True` for selected items. |
| `objective` | Current objective value. |
| `components` | Array `[selection_cost, intra_cost, inter_cost]`. |

### `Solution_shm`

Shared-memory variant of `Solution` for parallel local search. Same API as `Solution`, with an additional `num_processes` argument in `local_search`. Use as a context manager to ensure shared memory is released:

```python
with Solution_shm.generate_random_solution(distances, clusters) as sol:
    sol.local_search(num_processes=8, max_runtime=3600)
```

Or call `sol.cleanup()` explicitly when done.

---

## Objective function

$$
\text{Minimise} \quad \underbrace{\sum_{i \in S} f(i)}_{\text{selection cost}} + \underbrace{\sum_{c} \sum_{i \notin S,\, i \in c} \min_{j \in S \cap c} d(i,j)}_{\text{intra-cluster coverage}} + \lambda \underbrace{\sum_{c \neq c'} \max_{i \in S \cap c,\, j \in S \cap c'} s(i,j)}_{\text{inter-cluster similarity}}
$$

where $S$ is the selected set, $d(i,j) \in [0,1]$ is the distance between items $i$ and $j$, $s(i,j) = 1 - d(i,j)$ is their similarity, $f(i)$ is the selection cost, and $\lambda$ (`--scale`) controls the weight of the inter-cluster term.
