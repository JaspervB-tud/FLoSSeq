import numpy as np
from scipy.spatial.distance import squareform
import hdbscan
import networkx as nx
import argparse
import matplotlib.pyplot as plt
from Bio import SeqIO
import os

def read_metadata(path):
    seq2lin = {}
    seq2code = {}
    with open(path, "r") as f_in:
        next(f_in)  # Skip header
        for line in f_in:
            fields = line.strip().split("\t")
            seq_id = fields[0]
            epi_isl = fields[4]
            lineage = fields[13]

            seq2lin[seq_id] = lineage
            seq2code[seq_id] = epi_isl

    return seq2lin, seq2code

def read_sequences(path):
    sequences = {}
    for record in SeqIO.parse(path, "fasta"):
        sequences[record.id] = record

    return sequences

def read_distances(path, seq2idx, num_seqs):
    D = np.zeros((num_seqs, num_seqs), dtype=np.float64)
    with open(path, "r") as f_in:
        next(f_in) # Skip header
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

def create_graph(D, clusters, idx2seq, seq2lin, idx2lin, lin2idx):
    """
    Create a "difficulty graph" based on the distance matrix D, clustering results, and lineage information.
    A sequence is considered "difficult" if it is closer to the medoid of a different lineage than to the medoid of its own lineage,
    and the corresponding lineages are connected in the graph.

    Parameters:
    -----------
    D : np.ndarray
        A square distance matrix where D[i, j] is the distance between sequence i and sequence j.
    clusters : np.ndarray
        An array of cluster labels for each sequence.
    idx2seq : dict
        A mapping from index to sequence ID.
    seq2lin : dict
        A mapping from sequence ID to lineage.
    idx2lin : dict
        A mapping from index to lineage.
    lin2idx : dict
        A mapping from lineage to index.

    Returns:
    --------
    G : networkx.Graph
        A graph where nodes represent lineages and edges represent "difficulty" connections between lineages.
    difficult_sequences : dict
        A dictionary where keys are sequence IDs and values are lists of tuples representing the lineages that the sequence is difficult with respect to.
    distances_to_medoids : np.ndarray
        An array of distances from each sequence to the medoid of its own lineage.
    """
    # Create lineage graph
    N = len(idx2lin)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Process each cluster
    difficult_sequences = {}
    distances_to_medoids = np.zeros(D.shape[0], dtype=np.float64)
    for cluster_id in np.unique(clusters):
        cluster_member_indices = np.where(clusters == cluster_id)[0]

        # Determine medoids per lineage in this cluster
        lineage_medoids = {}
        for lineage in set(seq2lin[idx2seq[idx]] for idx in cluster_member_indices):
            member_indices = [
                idx for idx in cluster_member_indices
                if seq2lin[idx2seq[idx]] == lineage
            ]
            if not member_indices:
                continue
            sub_D = D[np.ix_(member_indices, member_indices)]
            medoid_idx = member_indices[np.argmin(sub_D.sum(axis=1))]
            lineage_medoids[lineage] = medoid_idx

        # Find difficult sequences in this cluster
        for idx in cluster_member_indices:
            seq_id = idx2seq[idx]
            true_lineage = seq2lin[seq_id]
            true_lineage_medoid_idx = lineage_medoids[true_lineage]
            distances_to_medoids[idx] = D[idx, true_lineage_medoid_idx]
            for lineage, medoid_idx in lineage_medoids.items():
                if lineage == true_lineage: # If closest medoid is of the same lineage, skip
                    continue
                # Add edge if closer to other lineage's medoid
                if D[idx, medoid_idx] < D[idx, true_lineage_medoid_idx]:
                    G.add_edge(
                        lin2idx[true_lineage],
                        lin2idx[lineage]
                    )
                    if seq_id not in difficult_sequences:
                        difficult_sequences[seq_id] = []
                    difficult_sequences[seq_id].append((lin2idx[true_lineage], lin2idx[lineage]))

    return G, difficult_sequences, distances_to_medoids

def find_MWIS(G, num_difficult_per_lineage):
    """
    Greedily find a maximium weight independent set (MWIS) in the graph G, 
    where weights are assigned based on the number of difficult sequences per lineage and
    normalized by the degree of the node. The goal is to select high weight lineages that 
    are less connected and have fewer difficult sequences.

    Parameters:
    -----------
    G : networkx.Graph
        A graph where nodes represent lineages and edges represent "difficulty" connections between lineages.
    num_difficult_per_lineage : dict
        A dictionary where keys are lineage indices and values are the number of difficult sequences associated with that lineage.

    Returns:
    --------
    selection : set
        A set of selected lineage indices that form a maximum weight independent set.
    unselection : set
        A set of lineage indices that were not selected because they were isolated.
    """

    # Determine node weights based on number of sequences
    weights = {}
    degrees = {}

    # Assign weights to lineages based on difficult sequences
    for node in G.nodes():
        lineage = int(node)
        num_difficult_sequences = num_difficult_per_lineage.get(lineage, 0)

        if num_difficult_sequences == 0:
            weights[node] = 0
        elif 1 <= num_difficult_sequences <= 5:
            weights[node] = 0.5 # Lowly connected lineages get weight 0.5
        elif 5 < num_difficult_sequences <= 15:
            weights[node] = 1 # Moderately connected lineages get weight 1
        else:
            weights[node] = 3 # Highly connected lineages get weight 3
        degrees[lineage] = G.degree(lineage)

    # Create a copy of the graph
    G_copy = G.copy()
    selection = set()
    unselection = set()

    # Remove isolated nodes (singletons)
    for node in list(G_copy.nodes()):
        if degrees[node] == 0:
            unselection.add(node)
            G_copy.remove_node(node)

    # While nodes remain, greedily select highest score node and remove its neighbors
    while G_copy.number_of_nodes() > 0:
        # Set scores by normalizing weights by degree (we want high weight and low degree)
        scores = {
            node: weights[node] / (G_copy.degree[node] + 1)
            for node in G_copy.nodes()
        }

        # Select node with highest score
        max_node = max(scores, key=scores.get)
        if scores[max_node] > 0:
            selection.add(max_node)

        # Remove selected node and its neighbors from the graph
        neighbors = list(G_copy.neighbors(max_node))
        G_copy.remove_node(max_node)
        for neighbor in neighbors:
            G_copy.remove_node(neighbor)

    return selection, unselection

def main():
    parser = argparse.ArgumentParser(description="Select SARS-CoV-2 genomes based on difficulty criteria.")
    parser.add_argument("--fasta", required=True, type=str, help="Path to input FASTA file with sequences.")
    parser.add_argument("--distance_matrix", required=True, help="Path to the pairwise distances file.")
    parser.add_argument("--metadata", required=True, help="Path to the metadata TSV file.")
    parser.add_argument("--output", type=str, help="Path to output folder.")
    parser.add_argument("--min_cluster_size", type=int, default=5, help="Minimum cluster size when using HDBSCAN.")
    parser.add_argument("--max_sequences", type=int, default=7, help="Maximum number of sequences to select per lineage.")
    parser.add_argument("--max_lineages", type=int, default=10, help="Maximum number of lineages to select sequences for.")
    args = parser.parse_args()

    # Read input data
    seq2lin, seq2code = read_metadata(args.metadata)
    sequences = sorted(list(seq2lin.keys()))
    num_sequences = len(sequences)

    fasta_sequences = read_sequences(args.fasta)
    for seq_id in sequences:
        if seq_id not in fasta_sequences:
            raise ValueError(f"Sequence ID {seq_id} from metadata not found in FASTA file.")
        
    ## Create indices
    idx2lin = []
    lin2idx = {}
    idx2seq = []
    seq2idx = {}
    for idx, seq_id in enumerate(sequences):
        lineage = seq2lin[seq_id]
        if lineage not in lin2idx:
            lin2idx[lineage] = len(idx2lin)
            idx2lin.append(lineage)
        seq2idx[seq_id] = idx
        idx2seq.append(seq_id)

    ## Read distance matrix
    D = read_distances(args.distance_matrix, seq2idx, num_sequences)

    # Find and remove duplicates
    groups = find_duplicate_groups(D)
    to_keep = [
        min(group) for group in groups if len(group) > 0 # keep the first sequence in each group
    ]
    D = D[np.ix_(to_keep, to_keep)] #only keep unique sequences

    # Create new indices
    idx2seq = [sequences[i] for i in to_keep]
    seq2idx = {seq_id: idx for idx, seq_id in enumerate(idx2seq)}

    # Cluster sequences with HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        metric='precomputed',
    )
    clusters = clusterer.fit_predict(D)

    # HDBSCAN assigns -1 to noise points, we can treat them as their own clusters
    next_cluster_id = max(clusters) + 1
    for idx, cluster in enumerate(clusters):
        if cluster == -1:
            clusters[idx] = next_cluster_id
            next_cluster_id += 1

    # Create difficulty graph
    G, difficult_sequences, distances_to_medoids = create_graph(D, clusters, idx2seq, seq2lin, idx2lin, lin2idx)

    # Count difficult sequences per lineage
    num_difficult_per_lineage = {
        lineage: 0 for lineage in range(len(idx2lin))
    }
    for seq_id in difficult_sequences:
        lineage = lin2idx[seq2lin[seq_id]]
        if len(difficult_sequences[seq_id]) > 0:
            num_difficult_per_lineage[lineage] += 1

    # Find MWIS
    hard_lineages, easy_lineages = find_MWIS(G, num_difficult_per_lineage)

    # Sort by number of (difficult) sequences and limit to max_lineages
    hard_lineages = sorted(hard_lineages, key=lambda x: -num_difficult_per_lineage.get(x, 0))[:args.max_lineages]
    easy_lineages = sorted(easy_lineages, key=lambda x: -len([idx for idx in range(D.shape[0]) if seq2lin[idx2seq[idx]] == idx2lin[x]]))[:args.max_lineages]
    
    # Determine sequence scores
    sequence_scores = []
    score_per_sequence = {}
    for seq_id, idx in seq2idx.items():
        lineage = lin2idx[seq2lin[seq_id]]

        if lineage in hard_lineages:
            count = 0 # num sequences from other lineages closer to this than their own medoid
            tot = 0 # total number of sequences that are closer to this than their own medoid
            cluster = clusters[idx]
            other_indices = np.where(clusters == cluster)[0]

            for other_idx in other_indices:
                if other_idx == idx:
                    continue
                other_seq_id = idx2seq[other_idx]
                other_lineage = lin2idx[seq2lin[other_seq_id]]

                if D[other_idx, idx] < distances_to_medoids[other_idx]:
                    tot += 1
                    if other_lineage != lineage:
                        count += 1

            if tot > 0:
                sequence_scores.append((seq_id, count / tot))
                score_per_sequence[seq_id] = count / tot
            else:
                sequence_scores.append((seq_id, 0.0))
                score_per_sequence[seq_id] = 0.0
        else:
            sequence_scores.append((seq_id, 0.0))
            score_per_sequence[seq_id] = 0.0

    # Select Hard and Medium sequences from "difficult" lineages
    selected_hard_sequences = {}
    selected_hard_sequences_set = set()
    selected_medium_sequences = {}
    selected_medium_sequences_set = set()

    for lineage in hard_lineages:
        selected_hard_sequences[lineage] = []
        selected_medium_sequences[lineage] = []

        cur_sequences = [
            (seq_id, score_per_sequence[seq_id])
            for seq_id in seq2idx
            if lin2idx[seq2lin[seq_id]] == lineage
        ]

        # Sort sequences by score and prioritize difficult sequences
        cur_sequences = sorted(cur_sequences, key=lambda x: (int(x[0] in difficult_sequences), x[1]), reverse=True)

        # Select Hard sequences
        top = min(num_difficult_per_lineage.get(lineage, 0), args.max_sequences)
        for seq_id, score in cur_sequences[:top]:
            selected_hard_sequences[lineage].append((seq_id, score))
            selected_hard_sequences_set.add(seq_id)

        # Select Medium sequences
        top = min(len(cur_sequences), args.max_sequences)
        for seq_id, score in cur_sequences[-top:]:
            selected_medium_sequences[lineage].append((seq_id, score))
            selected_medium_sequences_set.add(seq_id)

    # Select Easy sequences from "easy" lineages
    selected_easy_sequences = {}
    selected_easy_sequences_set = set()

    for lineage in easy_lineages:
        selected_easy_sequences[lineage] = []
        member_indices = [
            idx for idx in range(D.shape[0])
            if seq2lin[idx2seq[idx]] == idx2lin[lineage]
        ]

        # Calculate and sort by distance to nearest other lineage
        member_distances = []
        for idx in member_indices:
            seq_id = idx2seq[idx]
            min_dist = np.inf
            for other_idx in range(D.shape[0]):
                other_seq_id = idx2seq[other_idx]
                if seq2lin[other_seq_id] != seq2lin[seq_id]:
                    if D[idx, other_idx] < min_dist:
                        min_dist = D[idx, other_idx]
            member_distances.append((seq_id, min_dist))
        member_distances = sorted(member_distances, key=lambda x: x[1], reverse=True)

        # Select Easy sequences
        top = min(len(member_distances), args.max_sequences)
        for i in range(top):
            seq_id, min_dist = member_distances[i]
            selected_easy_sequences[lineage].append((seq_id, min_dist))
            selected_easy_sequences_set.add(seq_id)

    # Write output
    ## Hard sequences
    for lineage in sorted(selected_hard_sequences, key=lambda x: idx2lin[x]):
        if G.degree[lineage] == 0:
            continue

        os.makedirs(f"{args.output}/Hard/{idx2lin[lineage]}", exist_ok=True)
        for seq_id, _ in selected_hard_sequences[lineage]:
            if seq_id in fasta_sequences:
                SeqIO.write(
                    fasta_sequences[seq_id],
                    f"{args.output}/Hard/{idx2lin[lineage]}/{seq2code[seq_id]}.fasta",
                    "fasta"
                )

    ## Medium sequences
    for lineage in sorted(selected_medium_sequences, key=lambda x: idx2lin[x]):
        if G.degree[lineage] == 0:
            continue

        os.makedirs(f"{args.output}/Medium/{idx2lin[lineage]}", exist_ok=True)
        for seq_id, _ in selected_medium_sequences[lineage]:
            if seq_id in fasta_sequences:
                SeqIO.write(
                    fasta_sequences[seq_id],
                    f"{args.output}/Medium/{idx2lin[lineage]}/{seq2code[seq_id]}.fasta",
                    "fasta"
                )

    ## Easy sequences
    for lineage in sorted(selected_easy_sequences, key=lambda x: idx2lin[x]):
        # Do not skip 0-degree nodes, because those are exactly the easy lineages
        os.makedirs(f"{args.output}/Easy/{idx2lin[lineage]}", exist_ok=True)
        for seq_id, _ in selected_easy_sequences[lineage]:
            if seq_id in fasta_sequences:
                SeqIO.write(
                    fasta_sequences[seq_id],
                    f"{args.output}/Easy/{idx2lin[lineage]}/{seq2code[seq_id]}.fasta",
                    "fasta"
                )

if __name__ == "__main__":
    main()