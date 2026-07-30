import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import argparse
import os

def parse_sequence_mapping(path):
    idx2seq = []
    seq2idx = {}
    with open(path, "r") as f_in:
        for line in f_in:
            idx, seq_id = line.strip().split("\t")
            idx = int(idx)
            seq2idx[seq_id] = idx
            idx2seq.append(seq_id)

    return idx2seq, seq2idx

# NOTE: Sequence ids are indices from 0 to N-1
def read_MASH(path):
    with open(path, "r") as f_in:
        header = next(f_in).strip()
        num_sequences = int(header)

        D = np.zeros((num_sequences, num_sequences), dtype=np.float64)
        indices = []

        for line in f_in:
            parts = line.strip().split()
            idx1 = int(parts[0])
            indices.append(idx1)

            for j, dist_str in enumerate(parts[1:]):
                idx2 = indices[j]
                dist = float(dist_str)
                D[idx1, idx2] = float(dist)
                D[idx2, idx1] = float(dist)

    return D, indices

def read_SOURMASH(path, num_sequences):
    QUERY_NAME_POS=0
    MATCH_NAME_POS=2
    JACCARD_POS=6
    COSINE_POS=12

    D_jaccard = np.zeros((num_sequences, num_sequences), dtype=np.float64)
    D_cosine = np.zeros((num_sequences, num_sequences), dtype=np.float64)

    with open(path, "r") as f_in:
        next(f_in)
        for line in f_in:
            parts = line.strip().split(",")
            idx1 = int(parts[QUERY_NAME_POS])
            idx2 = int(parts[MATCH_NAME_POS])
            jaccard = float(parts[JACCARD_POS])
            cosine = float(parts[COSINE_POS])

            D_jaccard[idx1, idx2] = 1.0 - jaccard
            D_jaccard[idx2, idx1] = 1.0 - jaccard
            D_cosine[idx1, idx2] = 1.0 - cosine
            D_cosine[idx2, idx1] = 1.0 - cosine

    return D_jaccard, D_cosine

def main():
    parser = argparse.ArgumentParser(description="Compute medoid sequence.")
    parser.add_argument("--input_path", required=True, help="Path to the file containing the distance information")
    parser.add_argument("--output_path", required=True, help="Path to the file where the medoid sequence index will be stored")
    parser.add_argument("--mapping_path", required=True, help="Path to the file containing the mapping from sequence indices to sequence identifiers")
    
    parser.add_argument("--threshold", type=float, help="Similarity threshold to run clustering.")
    parser.add_argument("--dynamic", action="store_true", help="Indicates that the threshold is dynamic and should be applied as a percentile.")
    
    parser.add_argument("--sourmash_jaccard", action="store_true", help="Indicates that the input file is a sourmash output containing Jaccard similarities")
    parser.add_argument("--sourmash_cosine", action="store_true", help="Indicates that the input file is a sourmash output containing Cosine similarities")
    args = parser.parse_args()

    idx2seq, seq2idx = parse_sequence_mapping(args.mapping_path)

    if args.sourmash_jaccard or args.sourmash_cosine:
        num_sequences = len(idx2seq)
        D_jaccard, D_cosine = read_SOURMASH(args.input_path, num_sequences)
        D = D_jaccard if args.sourmash_jaccard else D_cosine
    else:
        D, _ = read_MASH(args.input_path)

    threshold = 0.0
    if args.dynamic:
        nonzero_distances = np.array([D[i,j] for i in range(D.shape[0]) for j in range(i) if D[i,j] != 0.0])
        threshold = np.percentile(nonzero_distances, args.threshold)
    else:
        threshold = 1.0 - args.threshold

    # Perform hierarchical clustering
    Z = linkage(squareform(D), method="complete")
    clusters = fcluster(Z, t=threshold, criterion="distance")
    representatives = {}
    for cluster in np.unique(clusters):
        members = np.where(clusters == cluster)[0]
        if len(members) == 1:
            representatives[cluster] = members[0]
        else:
            intra_distances = D[np.ix_(members, members)]
            medoid_idx = np.argmin(np.sum(intra_distances, axis=1))
            representatives[cluster] = members[medoid_idx]

    with open(args.output_path, "w") as f_out:
        for cluster in np.unique(clusters):
            f_out.write(f"{idx2seq[representatives[cluster]]}\n")

if __name__ == "__main__":
    main()