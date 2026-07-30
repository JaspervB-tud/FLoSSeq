import numpy as np
from sklearn.metrics import f1_score
import os
from Bio import SeqIO
import itertools
import argparse

# Constants
ID_IDX = 0
CLADE_IDX = 15
DIFFICULTIES = ["Easy", "Hard"]
SAMPLES = [f"{i:02d}" for i in range(1, 21)]

def read_metadata(path):
    metadata = {}
    with open(f"{path}/metadata.tsv", "r") as f_in:
        next(f_in)  # skip header
        for line in f_in:
            fields = line.strip().split("\t")
            seq_id = fields[ID_IDX]
            clade = fields[CLADE_IDX]
            metadata[seq_id] = {
                "clade": clade
            }
    seq_ids = set(metadata.keys())

    # Since GISAID does not store sequence lengths for IAV segments, compute them from fasta
    for record in SeqIO.parse(f"{path}/sequences_concatenated.fasta", "fasta"):
        seq_id = record.id
        if seq_id in metadata:
            metadata[seq_id]["length"] = len(record.seq)
            seq_ids.remove(seq_id)
        else:
            raise ValueError(f"Sequence ID {seq_id} found in FASTA but not in metadata.")

    if seq_ids:
        raise ValueError(f"Some sequence IDs from metadata not found in FASTA: {seq_ids}")
    
    return metadata

def read_reference(path):
    reference_set = set()
    for record in SeqIO.parse(path, "fasta"):
        seq_id = record.id.split("_")
        seq_id = "_".join(seq_id[:-1])  # remove segment information
        reference_set.add(seq_id)

    return reference_set

def read_groundtruth(path, sample, metadata, min_abundance=0.001):
    nucleotide_counts = {}
    
    def read_fastq(path, part):
        with open(f"{path}/sample_{sample}_{part}.fastq", "r") as f_in:
            for idx, line in enumerate(f_in):
                if idx % 4 == 0: # header -> sequence line
                    seq_id = line.strip()[1:]
                    parts = seq_id.split("_")
                    seq_id = "_".join(parts[:-1]) # remove segment information and number information
                    if seq_id not in nucleotide_counts:
                        nucleotide_counts[seq_id] = 0
                elif idx % 4 == 1: # sequence line
                    nucleotide_counts[seq_id] += len(line.strip())

    # Forward reads
    read_fastq(path, part="R1")
    # Reverse reads
    read_fastq(path, part="R2")

    # Calculate clade abundances
    clades = set(metadata[seq_id]["clade"] for seq_id in nucleotide_counts)
    clade_counts = {clade: 0 for clade in clades}
    for seq_id in nucleotide_counts:
        clade = metadata[seq_id]["clade"]
        clade_counts[clade] += nucleotide_counts[seq_id] / metadata[seq_id]["length"] # normalize by length similar to kallisto
    total_counts = sum(clade_counts.values())

    # Filter by min_abundance and normalize
    abundances = {}
    for clade in clade_counts:
        abundance = clade_counts[clade] / total_counts
        if abundance >= min_abundance:
            abundances[clade] = abundance
    total_abundance = sum(abundances.values())
    for clade in abundances:
        abundances[clade] /= total_abundance # normalize to sum up to 1

    return abundances

def read_kallisto_output(path, metadata, min_abundance=0.001):
    id_idx = 0
    tpm_idx = 4

    abundances = {}
    total = 0.0

    with open(path, "r") as f_in:
        next(f_in)  # skip header
        for line in f_in:
            fields = line.strip().split("\t")
            seq_id = fields[id_idx].split("_")
            seq_id = "_".join(seq_id[:-1])  # remove segment information
            clade = metadata[seq_id]["clade"]
            # Add abundance to clade
            tpm = float(fields[tpm_idx])
            if clade not in abundances:
                abundances[clade] = 0.0
            abundances[clade] += tpm
            total += tpm

    # Normalize abundances and filter
    filtered_abundances = {}
    for clade in abundances:
        normalized_abundance = abundances[clade] / total
        if normalized_abundance >= min_abundance:
            filtered_abundances[clade] = normalized_abundance
    total_filtered = sum(filtered_abundances.values())
    for clade in filtered_abundances:
        filtered_abundances[clade] /= total_filtered  # normalize to sum up to 1

    return filtered_abundances

def main():
    parser = argparse.ArgumentParser(description="Analyze tuning results for IAV samples.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the analysis results.")
    parser.add_argument("--min_abundance", type=float, default=0.001, help="Minimum abundance threshold for filtering.")
    args = parser.parse_args()

    min_abundance = args.min_abundance
    os.makedirs(args.output_dir, exist_ok=True)

    # Fetch all indices
    index_folder = "data/IAV/Reference/Test/estimations"
    indices = [folder for folder in os.listdir(index_folder) if os.path.isdir(os.path.join(index_folder, folder))]
    indices = sorted(indices) # sort for consistent ordering
    num_indices = len(indices)

    # Create categories
    method_categories = []
    categories = np.zeros(len(indices), dtype=np.int8)
    for m_idx, method in enumerate(indices):
        if "medoid" in method:
            method_categories.append("MEDOID")
            categories[m_idx] = 0
        elif "hierarchical" in method:
            method_categories.append("HIERARCHICAL")
            categories[m_idx] = 1
        elif "vsearch" in method:
            method_categories.append("VSEARCH")
            categories[m_idx] = 2
        elif "reset" in method:
            method_categories.append("RESET")
            categories[m_idx] = 3
        else:
            raise ValueError(f"Unknown method: {method}")
    idx2cat = {
        0: "MEDOID",
        1: "HIERARCHICAL",
        2: "VSEARCH",
        3: "RESET"
    }

    # Create variables for storing results
    l1_scores = np.zeros((len(SAMPLES), num_indices, len(DIFFICULTIES)), dtype=np.float64)
    f1_scores = np.zeros((len(SAMPLES), num_indices, len(DIFFICULTIES)), dtype=np.float64)

    # Parse metadata
    metadata = read_metadata("data/IAV/Reference/metadata.tsv")
    clades = sorted(list(set(metadata[seq_id]["clade"] for seq_id in metadata))) # sort for consistent ordering
    clade2idx = {clade: idx for idx, clade in enumerate(clades)}

    # Fetch groundtruths and estimated abundances
    l1_errors = np.zeros((len(SAMPLES), num_indices, len(DIFFICULTIES)), dtype=np.float64)

    for d_idx, difficulty in enumerate(DIFFICULTIES):
        # Fetch groundtruths
        ground_truths = np.zeros((len(SAMPLES), len(clades)), dtype=np.float64)
        for s_idx, sample in enumerate(SAMPLES):
            gt_abundances = read_groundtruth(
                f"data/IAV/Reference/Test/{difficulty}",
                sample, metadata, min_abundance=min_abundance
            )
            for clade in gt_abundances:
                clade_idx = clade2idx[clade]
                ground_truths[s_idx, clade_idx] = gt_abundances[clade]

        # Fetch estimated abundances
        estimations = np.zeros((len(SAMPLES), len(clades), num_indices), dtype=np.float64)
        combinations = itertools.product(enumerate(indices), enumerate(SAMPLES))
        for (m_idx, method), (s_idx, sample) in combinations:
            kallisto_path = f"data/IAV/Reference/Test/estimations/{method}/{difficulty}_{sample}/abundance.tsv"
            est_abundances = read_kallisto_output(kallisto_path, metadata, min_abundance=min_abundance)
            for clade in est_abundances:
                clade_idx = clade2idx[clade]
                estimations[s_idx, clade_idx, m_idx] = est_abundances[clade]

            # Calculate L1-score (abundance accuracy)
            l1_error = np.sum(np.abs(ground_truths[s_idx] - estimations[s_idx, :, m_idx]))
            l1_errors[s_idx, m_idx, d_idx] = l1_error # no normalization
            l1_scores[s_idx, m_idx, d_idx] = 1.0 - (l1_error / 2.0) # L1-score is 1 - (L1-error / 2)

            # Calculate F1-score
            gt_binary = (ground_truths[s_idx] > 0).astype(int)
            est_binary = (estimations[s_idx, :, m_idx] > 0).astype(int)
            f1 = f1_score(gt_binary, est_binary, zero_division=0)
            f1_scores[s_idx, m_idx, d_idx] = f1

    # Set weighting for difficulties (more weight for difficult samples)
    weights = {
        "Easy": 0.4,
        "Hard": 0.6
    }

    for category_idx in range(len(idx2cat)):
        category = idx2cat[category_idx]
        method_indices = [m_idx for m_idx, _ in enumerate(indices) if method_categories[m_idx] == category]

        weighted_l1_scores = np.zeros(num_indices, dtype=np.float64)
        weighted_f1_scores = np.zeros(num_indices, dtype=np.float64)

        for local_idx, m_idx in enumerate(method_indices):
            method = indices[m_idx]
            weighted_l1 = 0.0
            weighted_f1 = 0.0
            for d_idx, difficulty in enumerate(DIFFICULTIES):
                weight = weights[difficulty]

                median_l1 = np.median(l1_scores[:, m_idx, d_idx])
                median_f1 = np.median(f1_scores[:, m_idx, d_idx])

                weighted_l1 += weight * median_l1
                weighted_f1 += weight * median_f1
            weighted_l1_scores[local_idx] = weighted_l1
            weighted_f1_scores[local_idx] = weighted_f1

        # Sort methods based on weighted L1 and write to file
        sorted_l1_indices = np.argsort(-weighted_l1_scores)  # sort by L1 descending
        sorted_l1_indices = sorted(sorted_l1_indices, key = lambda idx: (-weighted_l1_scores[idx], -weighted_f1_scores[idx])) # sort by L1 first, then F1
        with open(f"{args.output_dir}/l1_{category}.tsv", "w") as f_out:
            f_out.write("Method\tWeighted_L1\n")
            for rank, local_idx in enumerate(sorted_l1_indices):
                m_idx = method_indices[local_idx]
                method = indices[m_idx]
                score = weighted_l1_scores[local_idx]
                f_out.write(f"{method}\t{score:.8f}\n")

        # Sort methods based on weighted F1 and write to file
        sorted_f1_indices = np.argsort(-weighted_f1_scores)  # sort by F1 descending
        sorted_f1_indices = sorted(sorted_f1_indices, key = lambda idx: (-weighted_f1_scores[idx], -weighted_l1_scores[idx])) # sort by F1 first, then L1
        with open(f"{args.output_dir}/f1_{category}.tsv", "w") as f_out:
            f_out.write("Method\tWeighted_F1\n")
            for rank, local_idx in enumerate(sorted_f1_indices):
                m_idx = method_indices[local_idx]
                method = indices[m_idx]
                score = weighted_f1_scores[local_idx]
                f_out.write(f"{method}\t{score:.8f}\n")

        # Sort methods based on minimax of ranks and write to file
        l1_ranks = np.argsort(np.argsort(-weighted_l1_scores))  # rank 0 is best
        f1_ranks = np.argsort(np.argsort(-weighted_f1_scores))
        minimax_ranks = np.maximum(l1_ranks, f1_ranks) # take worst of both ranks
        sorted_minimax_indices = np.argsort(minimax_ranks)  # sort based on minimax ranks (lower is better)
        with open(f"{args.output_dir}/minimax_{category}.tsv", "w") as f_out:
            f_out.write("Method\L1_Rank\tF1_Rank\tMinimax_rank\n")
            for rank, local_idx in enumerate(sorted_minimax_indices):
                m_idx = method_indices[local_idx]
                method = indices[m_idx]
                l1_rank = l1_ranks[local_idx]
                f1_rank = f1_ranks[local_idx]
                minimax_rank = minimax_ranks[local_idx]
                f_out.write(
                    f"{method}\t{l1_rank}\t{f1_rank}\t{minimax_rank}\n"
                )

if __name__ == "__main__":
    main()