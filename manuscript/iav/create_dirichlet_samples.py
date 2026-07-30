import numpy as np
from Bio import SeqIO
import os
import argparse
import subprocess
import shutil
import hashlib
import copy

def read_metadata(path):
    seq2clade = {}
    seq2code = {}
    with open(path, "r") as f_in:
        next(f_in)  #skip header
        for line in f_in: 
            fields = line.strip().split("\t")
            seq_id = fields[0]
            clade = fields[15]

            seq2clade[seq_id] = clade
            seq2code[seq_id] = fields[11].strip() #GISAID EPI_ISL code
        
    return seq2clade, seq2code

def read_sequences(path):
    # Use the concatenated_ns.fasta file!
    sequences = {}
    for record in SeqIO.parse(path, "fasta"):
        sequences[record.id] = record
    return sequences

def read_distances(distance_matrix_path, seq2idx, num_seqs):
    D = np.zeros((num_seqs, num_seqs), dtype=np.float64)
    with open(distance_matrix_path, "r") as f_in:
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
    Recursively find all groups of exact duplicates (entries with distance 0).
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
    parser = argparse.ArgumentParser(description="Create IAV samples.")
    parser.add_argument("--distance_matrix", required=True, help="Path to the pairwise distances file.")
    parser.add_argument("--metadata", required=True, help="Path to the metadata TSV file.")
    parser.add_argument("--max_sequences", type=int, default=10, help="Maximum number of sequences to select per clade.")
    parser.add_argument("--max_lineages", type=int, default=20, help="Maximum number of clades to select sequences for.")
    parser.add_argument("--num_replicates", type=int, default=10, help="Number of sample replicates to create.")
    parser.add_argument("--num_configs", type=int, default=10, help="Number of different configurations to create.")
    parser.add_argument("--coverage", type=int, default=2000, help="Total coverage for each sample.")
    parser.add_argument("--min_abundance", type=float, default=0.001, help="Minimum abundance per clade in the sample.")
    parser.add_argument("--fasta_HA", required=True, type=str, help="Path to input FASTA file HA with sequences.")
    parser.add_argument("--fasta_NA", required=True, type=str, help="Path to input FASTA file NA with sequences.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility.")
    parser.add_argument("--output", required=True, type=str, help="Path to output folder.")
    # Read parameters
    parser.add_argument("--read_len", type=int, default=150, help="Read length.")
    parser.add_argument("--frag_mean", type=int, default=250, help="Fragment mean size.")
    parser.add_argument("--frag_sd", type=int, default=10, help="Fragment size standard deviation.")
    parser.add_argument("--profile", choices=["HS10", "HS20", "HS25", "HSXn", "HSXt"], default="HS25", help="Illumina error profile to use.")
    args = parser.parse_args()

    # Fetch sequence to clade mapping
    seq2clade, seq2code = read_metadata(args.metadata)
    sequences = sorted(list(seq2clade.keys()))
    num_sequences = len(sequences)

    fasta_HA_sequences = read_sequences(args.fasta_HA)
    fasta_NA_sequences = read_sequences(args.fasta_NA)

    # Total coverage for each sample
    coverage = args.coverage

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

    # Remove exact duplicates
    groups = find_duplicate_groups(D)
    to_keep = [
        min(group) for group in groups if len(group) > 0 # keep the first sequence in each group
    ]
    D = D[np.ix_(to_keep, to_keep)]

    # Create updated indices
    idx2seq = [idx2seq[i] for i in to_keep]
    seq2idx = {seq_id: idx for idx, seq_id in enumerate(idx2seq)}

    # Pre-processing: select up to X sequences per clade (greedy maxmin distance)
    selection_per_clade = {}
    rng = np.random.default_rng(args.seed)
    for clade in idx2clade:
        seqs = [seq_id for seq_id in idx2seq if seq2clade[seq_id] == clade]
        D_clade = D[np.ix_([seq2idx[seq_id] for seq_id in seqs], [seq2idx[seq_id] for seq_id in seqs])]

        medoid = np.argmin(np.sum(D_clade, axis=1))
        selected_idx = [medoid]
        remaining = set(range(len(seqs))) - {medoid}

        # Select sequences based on maxmin distance
        while remaining and len(selected_idx) < args.max_sequences:
            next_i = max(remaining, key=lambda i: min(D_clade[i,j] for j in selected_idx))
            selected_idx.append(next_i)
            remaining.remove(next_i)

        pool = [seqs[i] for i in selected_idx]
        rng.shuffle(pool)
        selection_per_clade[clade] = pool

    # Actual selection
    final_selections = []
    for config_idx in range(args.num_configs):
        cur_clades = rng.choice(idx2clade, size=min(args.max_lineages, len(idx2clade)), replace=False)
        cur_selections = []
        for rep in range(args.num_replicates):
            cur_sequences = {}
            for clade in cur_clades:
                pool = selection_per_clade[clade]
                genome = pool[rep % len(pool)] # rotate through pool
                cur_sequences[clade] = genome
            cur_selections.append(cur_sequences)

            # Write output
            for clade in cur_sequences:
                output_dir = f"{args.output}/config_{config_idx+1}/replicate_{rep+1}"
                os.makedirs(output_dir, exist_ok=True)

                seq_id = cur_sequences[clade]
                record_HA = copy.deepcopy(fasta_HA_sequences[seq_id])
                record_HA.id = f"{seq_id}_HA"
                record_HA.description = ""
                record_NA = copy.deepcopy(fasta_NA_sequences[seq_id])
                record_NA.id = f"{seq_id}_NA"
                record_NA.description = ""

                # Write both segments to the same file
                with open(f"{output_dir}/{clade}.fasta", "w") as f_out:
                    SeqIO.write([record_HA, record_NA], f_out, "fasta")

        final_selections.append(cur_selections)

    # Create samples with Dirichlet distribution using ART
    alphas = [0.1, 1.0, 10.0] # concentration parameters for Dirichlet distribution

    for config_idx in range(args.num_configs):
        cur_selections = final_selections[config_idx]
        cur_clades = sorted(cur_selections[0].keys())
        num_clades = len(cur_clades)

        for alpha in alphas:
            p = rng.dirichlet([alpha] * num_clades)
            p = args.min_abundance + (1.0 - num_clades * args.min_abundance) * p # ensure minimum abundance
            p /= p.sum() # normalize
            abundance_per_clade = {
                clade: p[idx] for idx, clade in enumerate(cur_clades)
            }

            for rep_idx, cur_sequences in enumerate(cur_selections):
                r1_parts = []
                r2_parts = []

                base_dir = f"{args.output}/config_{config_idx+1}/alpha_{alpha}/replicate_{rep_idx+1}"
                os.makedirs(base_dir, exist_ok=True)

                manifest_rows = [] # stores manifest information

                for clade in cur_clades:
                    seq_id = cur_sequences[clade]
                    abundance = abundance_per_clade[clade]

                    clade_coverage = coverage * abundance
                    prefix = f"{base_dir}/{clade}_"
                    reads_seed = int(hashlib.sha256(f"{config_idx}_{alpha}_{rep_idx}_{clade}_{seq_id}_{coverage}".encode("utf-8")).hexdigest(), 16) % (2**31 - 1) #create a stable 32bit seed per lineage (1 genome per lineage) for ART
                    cmd = [
                        "art_illumina",
                        "-ss", args.profile,
                        "-p",
                        "-i", f"{args.output}/config_{config_idx+1}/replicate_{rep_idx+1}/{clade}.fasta",
                        "-l", str(args.read_len),
                        "-f", str(clade_coverage),
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
                        config_idx+1, alpha, rep_idx+1, clade, seq_id, seq2code[seq_id],
                        abundance, clade_coverage, reads_seed
                    ))

                # Create manifest file
                manifest_path = f"{base_dir}/manifest.tsv"
                with open(manifest_path, "w") as f_out:
                    f_out.write("configuration\talpha\treplicate\tclade\tsequence_id\tepi_isl_code\tabundance\tclade_coverage\tseed\n")
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
                for clade in cur_clades:
                    prefix = f"{base_dir}/{clade}_"
                    os.remove(f"{prefix}1.fq")
                    os.remove(f"{prefix}2.fq")

if __name__ == "__main__":
    main()