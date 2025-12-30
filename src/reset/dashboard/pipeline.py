from __future__ import annotations
from typing import List, Tuple, Literal
import numpy as np
from Bio import SeqIO
import sourmash
from scipy.spatial.distance import squareform
from multiprocessing import Pool, get_context
from itertools import combinations
from ..solution import Solution

SelectionMethod = Literal["random", "centroid"]
SeqPath = "/Users/jaspervanbemmelen/Documents/Projects/Reference Optimization/GISAID_downloaded-23-05-2025_dates-01-07-2024_31-12-2024/sequences.fasta"
ClustPath = "/Users/jaspervanbemmelen/Documents/Projects/Reference Optimization/GISAID_downloaded-23-05-2025_dates-01-07-2024_31-12-2024/clusters.tsv"

def read_fasta(filepath: str, min_length: int = 0):
    """
    Reads a multi-FASTA file and returns a dictionary mapping sequence IDs to sequences.

    Parameters:
    -----------
    filepath: str
        Path to the multi-FASTA file.
    min_length: int
        Minimum length of sequences to include. Sequences shorter than this length will be skipped.
        Default is 0 (no minimum length).

    Returns:
    --------
    genomes: dict [str] -> (SeqRecord, str, sourmash.MinHash)
        Dictionary mapping sequence IDs in the FASTA file to their corresponding sequences.
    """
    genomes = {}
    for record in SeqIO.parse(filepath, "fasta"):
        cur_seq = str(record.seq)
        if len(cur_seq) >= min_length:
            mh = sourmash.MinHash(n=0, ksize=31, scaled=10, track_abundance=True)
            mh.add_sequence(cur_seq, force=True) #force=True to allow short sequences and non-ACGT chars
            genomes[record.id] = {
                "record": record,
                "sequence": cur_seq,
                "minhash": mh
            }
    return genomes

def determine_clusters(filepath: str, genomes: dict):
    """
    Reads a clustering file and returns an updated version of the genomes dictionary with cluster information.
    NOTE: For now this assumes that the clustering file is in tab-separated format with two columns:
        - sequence ID
        - cluster name (to be converted into ID later)

    Parameters:
    -----------
    filepath: str
        Path to the clustering file.
    genomes: dict[str]
        Dictionary mapping sequence IDs to their corresponding sequences.

    Returns:
    --------
    genomes: dict [str] -> (SeqRecord, str, sourmash.MinHash, str)
        Updated dictionary mapping sequence IDs to their corresponding sequences and cluster information.
    """
    with open(filepath, "r") as f_in:
        for line in f_in:
            seq_id, cluster_name = line.strip().split("\t")
            if seq_id in genomes:
                genomes[seq_id]["cluster"] = cluster_name
            else:
                print(f"Warning: sequence ID {seq_id} in clustering file not found in genomes.")

    return genomes

def downsample(genomes: dict, max_genomes: int = 2**31, random_state: int = None):
    """
    Downsamples genomes within each cluster to a maximum number of genomes.

    Parameters:
    -----------
    genomes: dict[str]
        Dictionary mapping sequence IDs to their corresponding sequences and cluster information.
    max_genomes: int
        Maximum number of genomes to retain per cluster. Default is 2^31.
    random_state: int 
        Random seed or RandomState for reproducibility. Default is None.
        NOTE: When random_state is None or an unrecognized type, no shuffling is performed and 
        the first max_genomes sequences per cluster are retained.

    Returns:
    --------
    downsampled_genomes: dict[str]
        Downsampled dictionary mapping sequence IDs to their corresponding sequences and cluster information.
    """
    # Check for random state
    if isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = None
    # First organize in dict (cluster -> list of seq_ids)
    sequences_per_cluster = {}
    for seq_id in genomes:
        cur_cluster = genomes[seq_id]["cluster"]
        if cur_cluster not in sequences_per_cluster:
            sequences_per_cluster[cur_cluster] = []
        sequences_per_cluster[cur_cluster].append(seq_id)
    # Now downsample
    downsampled_genomes = {}
    for cluster in sequences_per_cluster:
        if len(sequences_per_cluster[cluster]) <= max_genomes: #can simply use all
            for seq_id in sequences_per_cluster[cluster]:
                downsampled_genomes[seq_id] = genomes[seq_id]
        else:
            if rng is not None:
                sequences_copy = sequences_per_cluster[cluster][:]
                rng.shuffle(sequences_copy)
                for seq_id in sequences_copy[:max_genomes]:
                    downsampled_genomes[seq_id] = genomes[seq_id]
            else:
                for seq_id in sequences_per_cluster[cluster][:max_genomes]:
                    downsampled_genomes[seq_id] = genomes[seq_id]

    return downsampled_genomes


_minhashes = None
_index2id = None
def _init_pool(minhashes, index2id):
    global _minhashes, _index2id
    _minhashes = minhashes
    _index2id = index2id

def _compute_distance(pair):
    i, j = pair
    d = _minhashes[_index2id[i]].similarity(_minhashes[_index2id[j]])
    return i, j, 1.0-d

def compute_distances(genomes: dict, index2seq: list, seq2index: dict, cores: int = 1):
    """
    Computes the pairwise distance matrix between genomes based on their MinHash sketches.

    Parameters:
    -----------
    genomes: dict[str]
        Dictionary mapping sequence IDs to their corresponding sequences and MinHash sketches.
    index2seq: list[str]
        List mapping indices to sequence IDs.
    seq2index: dict[str, int]
        Dictionary mapping sequence IDs to their corresponding indices.
    cores: int
        Number of CPU cores to use for parallel computation. Default is 1 (single-core).

    Returns:
    --------
    D: np.ndarray
        Pairwise distance matrix between genomes.
    """
    n = len(index2seq)
    D = np.zeros((n, n), dtype=np.float32)
    if n <= 1:
        return D
    else:
        if cores == 1: #single core
            for i in range(n):
                for j in range(i):
                    d = 1.0 - genomes[index2seq[i]]["minhash"].similarity(genomes[index2seq[j]]["minhash"])
                    D[i,j] = d
                    D[j,i] = d
        else:   #multi-core
            pairs = combinations(range(n), 2)
            minhashes = {seq_id: genomes[seq_id]["minhash"] for seq_id in index2seq}
            ctx = get_context("spawn")
            with ctx.Pool(processes=cores, initializer=_init_pool, initargs=(minhashes, index2seq)) as pool:
                for i, j, d in pool.imap_unordered(_compute_distance, pairs, chunksize=2048):
                    D[i,j] = d
                    D[j,i] = d
    return D


def main():
    print()

if __name__ == "__main__":
    main()