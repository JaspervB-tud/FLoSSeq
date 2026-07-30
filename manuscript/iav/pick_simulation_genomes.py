import numpy as np
import itertools
import networkx as nx
import argparse
from Bio import SeqIO
import os
import copy

def read_metadata(path):
    seq2clade = {}
    with open(path, "r") as f_in:
        next(f_in)  #skip header
        for line in f_in: 
            fields = line.strip().split("\t")
            seq_id = fields[0]
            clade = fields[15]
            seq2clade[seq_id] = clade
        
    return seq2clade

def read_sequences(path):
    # Use the concatenated_ns.fasta file!
    sequences = {}
    for record in SeqIO.parse(path, "fasta"):
        sequences[record.id] = record
    return sequences

def read_distances(path, seq2idx, num_seqs):
    D = np.zeros((num_seqs, num_seqs), dtype=np.float64)
    with open(path, "r") as f_in:
        next(f_in)  # skip header
        for line in f_in:
            seq_id1, seq_id2, dist = line.strip().split("\t")
            idx1 = seq2idx[seq_id1]
            idx2 = seq2idx[seq_id2]
            dist = np.float64(dist)
            D[idx1, idx2] = dist
            D[idx2, idx1] = dist
    return D

def find_duplicate_groups(D):
    """
    Find groups of duplicate sequences based on the distance matrix D. 
    Two sequences are considered duplicates if their distance is zero.

    Parameters:
    -----------
    D : np.ndarray
        A square distance matrix where D[i, j] is the distance between sequence i and sequence j.

    Returns:
    --------
    duplicate_groups : list of lists
        A list where each element is a list of indices representing a group of duplicate sequences.
    """
    n = D.shape[0]
    visited = np.zeros(n, dtype=bool)
    duplicate_groups = []

    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        stack = [i]
        # Use a stack to perform depth-first search (DFS) to find all duplicates of the current sequence
        while stack:
            current = stack.pop()
            zeros = np.where(D[current] == 0)[0]
            for z in zeros:
                if not visited[z]:
                    visited[z] = True
                    group.append(z)
                    stack.append(z)
        duplicate_groups.append(group)

    return duplicate_groups

def main():
    parser = argparse.ArgumentParser(description="Select IAV genomes based on difficulty criteria.")
    parser.add_argument("--fasta_HA", required=True, type=str, help="Path to input FASTA file with sequences (HA segment).")
    parser.add_argument("--fasta_NA", required=True, type=str, help="Path to input FASTA file with sequences (NA segment).")
    parser.add_argument("--distance_matrix", required=True, help="Path to the pairwise distances file.")
    parser.add_argument("--metadata", required=True, help="Path to the metadata TSV file.")
    parser.add_argument("--output", type=str, help="Path to output folder.")
    parser.add_argument("--max_sequences", type=int, default=5, help="Maximum number of sequences to select per clade.")
    args = parser.parse_args()

    # Read input data
    seq2clade = read_metadata(args.metadata)
    sequences = sorted(list(seq2clade.keys()))
    num_sequences = len(sequences)

    # Read sequences from FASTA files
    HA_sequences = read_sequences(args.fasta_HA)
    NA_sequences = read_sequences(args.fasta_NA)

    # Create indices
    idx2clade = []
    clade2idx = {}
    idx2seq = []
    seq2idx = {}
    for idx, seq_id in enumerate(sequences):
        clade = seq2clade[seq_id]
        if clade not in clade2idx:
            clade2idx[clade] = len(idx2clade)
            idx2clade.append(clade)
        seq2idx[seq_id] = idx
        idx2seq.append(seq_id)

    # Read distance matrix
    D = read_distances(args.distance_matrix, seq2idx, num_sequences)

    # Find and remove duplicates
    groups = find_duplicate_groups(D)
    to_keep = [
        min(group) for group in groups if len(group) > 0 # keep the first sequence in each group
    ]
    D = D[np.ix_(to_keep, to_keep)]

    # Create new indices
    idx2seq = [idx2seq[i] for i in to_keep]
    seq2idx = {seq_id: idx for idx, seq_id in enumerate(idx2seq)}

    # Fix clade ordering
    clades = sorted(list(set(seq2clade[seq_id] for seq_id in idx2seq))) # sorted for consistency
    medoid_per_clade = {clade: [] for clade in clades}
    for clade in clades:
        member_indices = [
            idx for idx in range(D.shape[0])
            if seq2clade[idx2seq[idx]] == clade
        ]
        if not member_indices:
            continue
        sub_D = D[np.ix_(member_indices, member_indices)]
        medoid_idx = np.argmin(np.sum(sub_D, axis=1))
        medoid_per_clade[clade] = member_indices[medoid_idx]

    # Score sequences based on distance to other clade medoids
    score_per_sequence = {}
    for s_idx, seq_id in enumerate(idx2seq):
        clade = seq2clade[seq_id]
        dist_to_own_medoid = D[s_idx, medoid_per_clade[clade]]
        cur_dist = 1.0
        for other_clade in clades:
            if other_clade == clade:
                continue
            other_medoid_idx = medoid_per_clade[other_clade]
            dist_to_other_medoid = D[s_idx, other_medoid_idx]
            cur_dist = min(cur_dist, dist_to_other_medoid)
        cur_dist = dist_to_own_medoid - cur_dist # set to difference between distance to own medoid and closest other medoid
        score_per_sequence[seq_id] = cur_dist

    max_sequences = args.max_sequences
    selection = {
        "Easy": {},
        "Hard": {},
    } # no medium since there is no non-arbitrary way of defining them for IAV.
    for clade in clades:
        clade_member_sequences = [
            (seq_id, score_per_sequence[seq_id]) for seq_id in idx2seq if seq2clade[seq_id] == clade
        ]
        clade_member_sequences = sorted(clade_member_sequences, key=lambda x: x[1], reverse=True) # high score = hard, low score = easy

        cutoff = 0.0 # cutoff between hard and easy (0 means that sequence is as close to its own medoid as it is to the closest other medoid)
        hard_sequences = [item for item in clade_member_sequences if item[1] >= cutoff]
        easy_sequences = [item for item in clade_member_sequences if item[1] < cutoff]

        # Hard sequences
        if len(hard_sequences) > max_sequences:
            selection["Hard"][clade] = []
            for seq_id, dist in hard_sequences[:max_sequences]:
                selection["Hard"][clade].append(seq_id)
        
        # Easy sequences
        if len(easy_sequences) > max_sequences:
            selection["Easy"][clade] = []
            for seq_id, dist in easy_sequences[:max_sequences]:
                selection["Easy"][clade].append(seq_id)

        # Write output
        for difficulty in ["Easy", "Hard"]:
            if len(selection[difficulty].get(clade, [])) > 0:
                os.makedirs(f"{args.output}/{difficulty}/{clade}", exist_ok=True)
                for seq_id, _ in selection[difficulty][clade]:
                    # Create multi-fasta with both segments, but change ids to original id + _{HA,NA}
                    records_to_write = []
                    record_HA = copy.deepcopy(HA_sequences[seq_id])
                    record_HA.id = f"{seq_id}_HA"
                    record_HA.description = ""
                    records_to_write.append(record_HA)
                    record_NA = copy.deepcopy(NA_sequences[seq_id])
                    record_NA.id = f"{seq_id}_NA"
                    record_NA.description = ""
                    records_to_write.append(record_NA)
                    SeqIO.write(records_to_write, f"{args.output}/{difficulty}/{clade}/{seq_id}.fasta", "fasta")