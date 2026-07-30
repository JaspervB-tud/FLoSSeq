import numpy as np
from Bio import SeqIO
import os
import argparse
import subprocess
import shutil
import hashlib

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

def main():
    parser = argparse.ArgumentParser(description="Create SARS-CoV-2 samples.")
    parser.add_argument("--distance_matrix", required=True, help="Path to the pairwise distances file.")
    parser.add_argument("--metadata", required=True, help="Path to the metadata TSV file.")
    parser.add_argument("--max_sequences", type=int, default=10, help="Maximum number of sequences to select per lineage.")
    parser.add_argument("--max_lineages", type=int, default=20, help="Maximum number of lineages to select sequences for.")
    parser.add_argument("--num_replicates", type=int, default=10, help="Number of sample replicates to create.")
    parser.add_argument("--num_configs", type=int, default=10, help="Number of different configurations to create.")
    parser.add_argument("--coverage", type=int, default=2000, help="Total coverage for each sample.")
    parser.add_argument("--min_abundance", type=float, default=0.001, help="Minimum abundance per lineage in the sample.")
    parser.add_argument("--fasta", required=True, type=str, help="Path to input FASTA file with sequences. If not provided, sequences will not be outputted.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility.")
    parser.add_argument("--output", required=True, type=str, help="Path to output folder. If not provided, only prints to stdout.")
    # Read parameters
    parser.add_argument("--read_len", type=int, default=150, help="Read length.")
    parser.add_argument("--frag_mean", type=int, default=250, help="Fragment mean size.")
    parser.add_argument("--frag_sd", type=int, default=10, help="Fragment size standard deviation.")
    parser.add_argument("--profile", choices=["HS10", "HS20", "HS25", "HSXn", "HSXt"], default="HS25", help="Illumina error profile to use.")
    args = parser.parse_args()

    # Fetch sequence to lineage mapping
    seq2lin, seq2code = read_metadata(args.metadata)
    sequences = sorted(list(seq2lin.keys()))
    num_sequences = len(sequences)

    fasta_sequences = read_sequences(args.fasta)

    # Total coverage for each sample
    coverage = args.coverage

    # Create indices
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

    # Read distance matrix
    D = read_distances(args.distance_matrix, seq2idx, num_sequences)

    # Remove exact duplicates
    groups = find_duplicate_groups(D)
    to_keep = [
        min(group) for group in groups if len(group) > 0
    ]
    D = D[np.ix_(to_keep, to_keep)]

    # Create updated indices
    idx2seq = [sequences[i] for i in to_keep]
    seq2idx = {seq_id: i for i, seq_id in enumerate(idx2seq)}

    # Pre-processing: select up to X sequences per lineage (greedy maxmin distance)
    selection_per_lineage = {}
    rng = np.random.default_rng(args.seed)
    for lineage in idx2lin:
        seqs = [seq_id for seq_id in idx2seq if seq2lin[seq_id] == lineage]
        D_lineage = D[np.ix_([seq2idx[seq_id] for seq_id in seqs], [seq2idx[seq_id] for seq_id in seqs])]

        medoid = np.argmin(np.sum(D_lineage, axis=1))
        selected_idx = [medoid]
        remaining = set(range(len(seqs))) - {medoid}

        # Select sequences based on maxmin distance
        while remaining and len(selected_idx) < args.max_sequences:
            next_i = max(remaining, key=lambda i: min(D_lineage[i, j] for j in selected_idx))
            selected_idx.append(next_i)
            remaining.remove(next_i)

        pool = [seqs[i] for i in selected_idx]
        rng.shuffle(pool)
        selection_per_lineage[lineage] = pool

    # Actual selection
    final_selections = []
    for config_idx in range(args.num_configs):
        cur_lineages = rng.choice(idx2lin, size=min(args.max_lineages, len(idx2lin)), replace=False)
        cur_selections = []
        for rep in range(args.num_replicates):
            cur_sequences = {}
            for lineage in cur_lineages:
                pool = selection_per_lineage[lineage] # already shuffled
                genome = pool[rep % len(pool)] # rotate through pool
                cur_sequences[lineage] = genome
            cur_selections.append(cur_sequences)

            # Write output
            for lineage in cur_sequences:
                output_dir = f"{args.output}/config_{config_idx+1}/replicate_{rep+1}"
                os.makedirs(output_dir, exist_ok=True)

                seq_id = cur_sequences[lineage]
                record = fasta_sequences[seq_id]
                SeqIO.write(record, f"{output_dir}/{lineage}.fasta", "fasta") # one genome per lineage -> use lineage as filename

        final_selections.append(cur_selections)

    # Create samples with Dirichlet distribution using ART
    alphas = [0.1, 1.0, 10.0] # concentration parameters for Dirichlet distribution

    for config_idx in range(args.num_configs):
        cur_selections = final_selections[config_idx]
        cur_lineages = sorted(cur_selections[0].keys()) # sort for consistency
        num_lineages = len(cur_lineages)

        for alpha in alphas:
            p = rng.dirichlet([alpha] * num_lineages)
            p = args.min_abundance + (1.0 - num_lineages * args.min_abundance) * p # ensure minimum abundance
            p /= p.sum() # normalize
            abundance_per_lineage = {
                lineage: p[idx] for idx, lineage in enumerate(cur_lineages)
            }

            for rep_idx, cur_sequences in enumerate(cur_selections):
                r1_parts = []
                r2_parts = []

                base_dir = f"{args.output}/config_{config_idx+1}/alpha_{alpha}/replicate_{rep_idx+1}"
                os.makedirs(base_dir, exist_ok=True)

                manifest_rows = [] # stores manifest information

                for lineage in cur_lineages:
                    seq_id = cur_sequences[lineage]
                    abundance = abundance_per_lineage[lineage]

                    lineage_coverage = coverage * abundance
                    prefix = f"{base_dir}/{lineage}_"
                    reads_seed = int(hashlib.sha256(f"{config_idx}_{alpha}_{rep_idx}_{lineage}_{seq_id}_{coverage}".encode("utf-8")).hexdigest(), 16) % (2**31 - 1) #create a stable 32bit seed per lineage (1 genome per lineage) for ART
                    cmd = [
                        "art_illumina",
                        "-ss", args.profile,
                        "-p",
                        "-i", f"{args.output}/config_{config_idx+1}/replicate_{rep_idx+1}/{lineage}.fasta",
                        "-l", str(args.read_len),
                        "-f", str(lineage_coverage),
                        "-m", str(args.frag_mean),
                        "-s", str(args.frag_sd),
                        "-rs", str(reads_seed),
                        "-o", prefix,
                        "-na"
                    ]
                    subprocess.run(cmd, check=True)

                    r1_parts.append(f"{prefix}1.fq")
                    r2_parts.append(f"{prefix}2.fq")

                    # Record in manifest
                    manifest_rows.append((
                        config_idx+1, alpha, rep_idx+1, lineage, seq_id, seq2code[seq_id],
                        abundance, lineage_coverage, reads_seed
                    ))

                # Create manifest file
                manifest_path = f"{base_dir}/manifest.tsv"
                with open(manifest_path, "w") as f_out:
                    f_out.write("configuration\talpha\treplicate\tlineage\tsequence_id\tepi_isl_code\tabundance\tlineage_coverage\tseed\n")
                    for row in manifest_rows:
                        f_out.write("\t".join(map(str, row)) + "\n")

                # Merge parts
                merged_r1 = f"{base_dir}/sample_1.fq"
                merged_r2 = f"{base_dir}/sample_2.fq"
                with open(merged_r1, "wb") as f_out:
                    for part in r1_parts:
                        with open(part, "rb") as f_in:
                            shutil.copyfileobj(f_in, f_out)
                with open(merged_r2, "wb") as f_out:
                    for part in r2_parts:
                        with open(part, "rb") as f_in:
                            shutil.copyfileobj(f_in, f_out)
                # Delete intermediate files
                for lineage in cur_lineages:
                    prefix = f"{base_dir}/{lineage}_"
                    os.remove(f"{prefix}1.fq")
                    os.remove(f"{prefix}2.fq")

if __name__ == "__main__":
    main()