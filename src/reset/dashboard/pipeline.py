from __future__ import annotations
import numpy as np
from ..solution import (
    generate_distances_mash,
    generate_distances_sourmash,
    generate_distances_generic,
)

DISTANCE_FORMATS = ["mash", "sourmash_cosine", "sourmash_jaccard", "generic"]


def read_clusters_for_dashboard(path, id_col=0, cluster_col=1, delimiter=",", header=False):
    """
    Read a delimited clusters/labels file.

    Parameters:
    -----------
    path: str
        Path to the clusters file.
    id_col: int
        Column index (0-based) of the item ID. Default is 0.
    cluster_col: int
        Column index (0-based) of the cluster label. Default is 1.
    delimiter: str
        Field delimiter. Default is comma.
    header: bool
        If True, skip the first line. Default is False.

    Returns:
    --------
    genomes: dict[str, dict]
        {seq_id: {"cluster": label}} in file order.
    ordered_seq_ids: list[str]
        Sequence IDs in the order they appear in the file.
        The position of each ID in this list is its integer index
        in the distance file.
    """
    genomes = {}
    ordered_seq_ids = []
    with open(path, "r") as f_in:
        if header:
            next(f_in)
        for line in f_in:
            parts = line.strip().split(delimiter)
            seq_id  = parts[id_col]
            cluster = parts[cluster_col]
            genomes[seq_id] = {"cluster": cluster}
            ordered_seq_ids.append(seq_id)
    return genomes, ordered_seq_ids


def load_distance_matrix(path, n, distance_format="mash"):
    """
    Load the full pairwise distance matrix from a pre-computed distance file.

    The integer indices in the distance file must correspond to the row order
    of the clusters file (index 0 = first row, index 1 = second row, …).

    Parameters:
    -----------
    path: str
        Path to the distance file.
    n: int
        Total number of sequences (size of the square distance matrix).
    distance_format: str
        One of 'mash', 'sourmash_cosine', 'sourmash_jaccard', 'generic'.

    Returns:
    --------
    D: np.ndarray, shape (n, n)
        Full pairwise distance matrix.
    """
    D = np.zeros((n, n), dtype=np.float64)

    if distance_format == "mash":
        gen = generate_distances_mash(path)
    elif distance_format == "sourmash_cosine":
        gen = generate_distances_sourmash(path, dist_col=12)
    elif distance_format == "sourmash_jaccard":
        gen = generate_distances_sourmash(path, dist_col=6)
    elif distance_format == "generic":
        gen = generate_distances_generic(path)
    else:
        raise ValueError(f"Unknown distance format '{distance_format}'. "
                         f"Choose from: {DISTANCE_FORMATS}")

    for i, j, d in gen:
        D[i, j] = d
        D[j, i] = d
    return D


def downsample(genomes, max_genomes=2**31, random_state=None):
    """
    Downsample genomes within each cluster to at most max_genomes sequences.

    Parameters:
    -----------
    genomes: dict[str, dict]
        {seq_id: {"cluster": label, ...}}
    max_genomes: int
        Maximum number of sequences to retain per cluster.
    random_state: int or np.random.RandomState or None
        Controls shuffling before downsampling. When None, the first
        max_genomes sequences per cluster are kept without shuffling.

    Returns:
    --------
    dict[str, dict]
        Downsampled subset of genomes, preserving original values.
    """
    if isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = None

    per_cluster = {}
    for seq_id, g in genomes.items():
        cluster = g["cluster"]
        per_cluster.setdefault(cluster, []).append(seq_id)

    downsampled = {}
    for cluster, seq_ids in per_cluster.items():
        if len(seq_ids) <= max_genomes:
            for seq_id in seq_ids:
                downsampled[seq_id] = genomes[seq_id]
        else:
            pool = seq_ids[:]
            if rng is not None:
                rng.shuffle(pool)
            for seq_id in pool[:max_genomes]:
                downsampled[seq_id] = genomes[seq_id]

    return downsampled


def extract_submatrix(D_full, global_indices):
    """
    Extract a square submatrix for a subset of sequences.

    Parameters:
    -----------
    D_full: np.ndarray, shape (n, n)
        Full pairwise distance matrix.
    global_indices: list[int]
        Row/column indices of the sequences to retain,
        in the desired local order.

    Returns:
    --------
    np.ndarray, shape (k, k)  where k = len(global_indices)
    """
    idx = np.array(global_indices)
    return D_full[np.ix_(idx, idx)]
