# ReSeT — Reference (genome) Selection Tool

ReSeT selects a **representative subset** of items from a collection that is partitioned into clusters (e.g. taxonomic lineages, clades, or any user-defined groups). Given pairwise distances in [0,1] between items, it minimises an objective that balances three competing costs:

1. **Selection cost** — a fixed cost for every selected item.
2. **Intra-cluster coverage** — for every unselected item, the distance to the nearest selected item in the same cluster.
3. **Inter-cluster diversity** — for every pair of clusters, the similarity between the two closest selected items across the pair (penalising ambiguity across clusters).

The optimization is solved with a **local search** algorithm (add / swap / double-swap / remove moves) that supports both single-process and parallel multi-process execution via shared memory.

If you use our tool, or want to refer to it for any reason, please cite our paper: "ReSeT: a taxonomy-aware reference genome selection tool" which is available on [bioRxiv](https://doi.org/10.64898/2026.06.17.732946)

## Manuscript
To obtain the results in our [manuscript](https://doi.org/10.64898/2026.06.17.732946), follow the steps outlined in the README found in `manuscript`!

## Installation

```bash
pip install reset-bio
```

ReSeT requires Python ≥ 3.10 and depends only on NumPy and SciPy.

### Optional dashboard dependencies

```bash
pip install "reset-bio[dashboard]"
```

## Command line usage

ReSeT has a command line interface for running the tool with pre-computed distance files.

```
python -m reset.solution [OPTIONS]
```

### Required arguments

| Argument | Description |
|---|---|
| `--clusters PATH` | Clusters/labels file (see [format](#clusters-file) below) mapping every item to a label. |
| `--distances PATH` | Pairwise distances file in Mash or Sourmash format. |
| `--sequences_mapping PATH` | *(Optional)* Index-to-ID mapping file (see [format](#sequences-mapping-file) below). |

### Clusters file

A delimited text file where each row maps an item ID to its cluster label:

```
seq001,Cluster_A
seq002,Cluster_A
seq003,Cluster_B
```

**NOTE**: with cluster we refer to a labeling of the items, rather than specific clusters.

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--id_col` | `0` | Column index (0-based) of the item ID. |
| `--cluster_col` | `1` | Column index (0-based) of the label. |
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

The `generic` format accepts any pairwise distance file where each row contains two integer indices and a distance value. Only the lower triangle (or any subset of pairs) needs to be present. It can be configured using:

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
| `--selection_cost` | `0.1` | Fixed cost per selected item. |
| `--scale` | `0.00001` | Scaling factor applied to the inter-cluster similarity term. If omitted, the term is included unscaled. If set to 0, inter-cluster similarity is omitted. |

### Solver options

| Argument | Default | Description |
|---|---|---|
| `--seed` | `12345` | Random seed for initial solution generation. |
| `--max_fraction` | `0.5` | Maximum fraction of items in the random initial solution. |
| `--max_iterations` | `10000000` | Maximum number of accepted local search moves. |
| `--max_runtime` | `3600` | Wall-clock time limit in seconds. |
| `--doubleswap_time_threshold` | `60.0` | Per-iteration time (seconds) beyond which double-swap moves are skipped. |

### Control over multiprocessing

| Argument | Default | Description |
|---|---|---|
| `--num_processes` | `1` | Number of parallel worker processes. Set to `1` for single-process mode. |
| `--mp_switch_threshold` | `15.0` | Time threshold (seconds) for switching from single-process to multiprocessing mode. |

Running ReSeT with multiple cores relies on the `Solution_shm` set-up which stores most information in shared memory in order to share across processes. \
**WARNING:** Using multiple processes may create race conditions, leading to potential non-deterministic behaviour of the local search!

### Logging frequency
| Argument | Default | Description |
|---|---|---|
| `--logging_frequency` | `100` | Frequency of logging progress during local search (every N iterations). |

### Output

| Argument | Default | Description |
|---|---|---|
| `--output PATH` | *(none)* | Write selected item IDs (one per line) to this file. If omitted, results are printed to stdout only. |

## Dashboard

ReSeT includes an interactive Streamlit dashboard for exploring and running the optimiser without writing code.

### Install dashboard dependencies

```bash
pip install "reset-bio[dashboard]"
```

### Run

```bash
streamlit run $(python -c "import reset.dashboard.app as a; import os; print(os.path.abspath(a.__file__))")
```

Or, if you have the repository cloned:

```bash
streamlit run src/reset/dashboard/app.py
```

The dashboard opens in your browser automatically. By default it loads the bundled toy example — no configuration needed. To use your own data, enter the path to a folder containing a clusters file and a distance file in the sidebar.

The dashboard supports the full optimiser configuration: distance format, downsampling, initialisation method, `selection_cost`, `scale` (inter-cluster penalty), move types, number of processes, and runtime/iteration limits.

## Example

See [src/reset/examples/toy_example/](src/reset/examples/toy_example/) for a working end-to-end example with fictional data.

## Python usage

### `Solution`

Single-process solution class. Suitable for small-to-medium datasets.

```python
Solution(distances, clusters, selection=None, selection_cost=1.0,
         cost_per_cluster=0, scale=None, seed=None)
```

### Streaming distances

When the full distance matrix is big, you can pass a **generator** that yields `(i, j, distance)` tuples instead of directly providing a distance matrix:

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

This enables streaming the distances into the Solution object which prevents simultaneously having multiple copies of the distance matrix in memory.

**Class methods for initialisation:**

| Method | Description |
|---|---|
| `Solution.generate_random_solution(distances, clusters, ...)` | Random initial solution. |
| `Solution.generate_medoid_solution(distances, clusters, ...)` | Selects the medoid of each cluster as the initial solution (not used in manuscript). |

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

Shared-memory variant of `Solution` for parallel local search. Same API as `Solution`, with an additional `num_processes` argument in `local_search`. **Use as a context manager to ensure shared memory is released:**

```python
with Solution_shm.generate_random_solution(distances, clusters) as sol:
    sol.local_search(num_processes=8, max_runtime=3600)
```

**or call** `sol.cleanup()` **explicitly!**

## MILP model

### Parameters and variables

**Sets**

- $\mathcal{G}$ — candidate reference items (genomes)
- $T$ — clusters (taxa)
- $\mathcal{G}_t := \{g \in \mathcal{G} : \tau(g) = t\}$ — items belonging to cluster $t$

**Functions**

- $\tau: \mathcal{G} \to T$ — maps each item to its cluster
- $d: \mathcal{G} \times \mathcal{G} \to [0,1]$ — pairwise distance; symmetric, $d(g,g)=0$

**Decision Variables**

- $x_g \in \{0,1\}$ — 1 if item $g$ is selected $\quad \forall g \in \mathcal{G}$

**Auxiliary Variables**

- $y_{g,g'} \in \{0,1\}$ — 1 if $g'$ represents $g$ $\quad \forall g,g' \in \mathcal{G} : \tau(g) = \tau(g')$
- $z_{g,g'} \in \{0,1\}$ — 1 if both $g$ and $g'$ are selected $\quad \forall g < g' \in \mathcal{G} : \tau(g) < \tau(g')$
- $q_{t,t'} \geq 0$ - equal to the maximum similarity of selected items from different clusters $t$ and $t'$ $\quad \forall t < t' \in T$

### Full model with constraints

$$
\begin{align*}
\min \quad
    & c\sum_{g \in \mathcal{G}} x_g
    + \sum_{\substack{g,g' \in \mathcal{G} \\ \tau(g)=\tau(g')}} d(g,g') y_{g,g'}
    + \lambda \frac{2(|\mathcal{G}|-|T|)}{|T|(|T|-1)} \sum_{t<t'} q_{t,t'} \\
\text{s.t.} \quad
    & \sum_{g' \in \mathcal{G}_{\tau(g)}} y_{g,g'} = 1
    && \forall g \in \mathcal{G}, \\
    & y_{g,g'} \leq x_{g'}
    && \forall g,g' \in \mathcal{G} : \tau(g)=\tau(g'), \\
    & z_{g,g'} \geq x_g + x_{g'} - 1
    && \forall g,g' \in \mathcal{G} : g<g',\ \tau(g)<\tau(g'), \\
    & q_{t,t'} \geq (1-d(g,g')) z_{g,g'}
    && \forall t,t' \in T : t<t',\ \forall g,g' \in \mathcal{G} : g<g',\ \tau(g)=t,\ \tau(g')=t', \\
    & x_g \in \{0,1\}
    && \forall g \in \mathcal{G}, \\
    & y_{g,g'} \in \{0,1\}
    && \forall g,g' \in \mathcal{G} : \tau(g)=\tau(g'), \\
    & z_{g,g'} \in \{0,1\}
    && \forall g,g' \in \mathcal{G} : g<g',\ \tau(g)<\tau(g'), \\
    & q_{t,t'} \geq 0
    && \forall t,t' \in T : t<t'.
\end{align*}
$$

