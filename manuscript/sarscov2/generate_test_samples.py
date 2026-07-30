import argparse
from pathlib import Path
import os
import random
import shutil
import subprocess
import hashlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", required=True, help="Root folder containing the lineages and genome files.")
    parser.add_argument("--output_folder", required=True, help="Output folder for the generated sample")
    parser.add_argument("--sample", type=int, default=1, help="Sample number (used for ART seed derivation).")
    # Lineage parameters
    parser.add_argument("--coverage", type=float, default=100.0, help="Target fold coverage per lineage.")
    # Read parameters
    parser.add_argument("--read_len", type=int, default=150, help="Read length.")
    parser.add_argument("--frag_mean", type=int, default=250, help="Fragment mean size.")
    parser.add_argument("--frag_sd", type=int, default=10, help="Fragment size standard deviation.")
    parser.add_argument("--profile", choices=["HS10", "HS20", "HS25", "HSXn", "HSXt"], default="HS25", help="Illumina error profile to use.")
    # Sequence selection parameters
    parser.add_argument("--num_seqs", type=int, default=3, help="Number of sequences to select per lineage.")
    args = parser.parse_args()

    sample_num = args.sample
    root_folder = Path(args.root_folder)
    output_folder = Path(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Create temporary per-genome directory to store reads for each genome before merging
    per_genome_dir = output_folder / "per_genome"
    os.makedirs(per_genome_dir, exist_ok=True)

    # Discover lineages and genomes
    lineages = []
    sequences_per_lineage = {}
    for sub in sorted(root_folder.iterdir()):
        if not sub.is_dir():
            continue
        if "sample" in sub.name or "per_genome" in sub.name:
            continue

        seqs = sorted(sub.glob("*.fasta")) # retrieve all fasta files in lineage dir
        if seqs:
            lineages.append(sub.name)
            sequences_per_lineage[sub.name] = seqs

    if not lineages:
        raise ValueError(f"No lineages found in root folder: {root_folder}")

    # Store read parts (1 and 2) to merge later
    r1_parts = []
    r2_parts = []

    # Simulate per-genome reads, then merge
    for lineage_idx, lineage in enumerate(lineages):
        seq_files = sequences_per_lineage[lineage]

        # Selection per lineage
        sequences_seed = int(hashlib.sha256(f"{sample_num}_{lineage}_{root_folder.name}".encode("utf-8")).hexdigest(), 16) % (2**31 - 1) # create a stable 32bit seed per lineage for selecting genomes
        rng_select = random.Random(sequences_seed)
        selected = rng_select.sample(seq_files, k=min(args.num_seqs, len(seq_files))) # randomly select sequences

        k_sel = len(selected)
        cov_per_genome = args.coverage / float(k_sel) # equal coverage per genome in lineage

        for g_idx, genome_path in enumerate(selected):
            # Create unique prefix
            prefix = f"s{sample_num:02d}_{lineage}_g{g_idx+1}"
            out_prefix = per_genome_dir / prefix

            # ART names: <prefix>1.fq and <prefix>2.fq
            r1 = Path(str(out_prefix) + "1.fq")
            r2 = Path(str(out_prefix) + "2.fq")

            reads_seed = int(hashlib.sha256(f"{sample_num}_{lineage}_{root_folder.name}_{g_idx}_{genome_path.name}".encode("utf-8")).hexdigest(), 16) % (2**31 - 1) #create a stable 32bit seed per genome for ART

            ref = genome_path
            cmd = [
                "art_illumina",
                "-ss", args.profile,
                "-p",
                "-i", str(ref),
                "-l", str(args.read_len),
                "-f", f"{cov_per_genome:.6f}",
                "-m", str(args.frag_mean),
                "-s", str(args.frag_sd),
                "-rs", str(reads_seed),
                "-o", str(out_prefix),
                "-na", # do not generate alignment file to save space
            ]
            subprocess.run(cmd, check=True)

            r1_parts.append(r1)
            r2_parts.append(r2)

    # Merge all parts
    merged_r1 = output_folder / f"sample_{sample_num:02d}_R1.fastq"
    merged_r2 = output_folder / f"sample_{sample_num:02d}_R2.fastq"

    with open(merged_r1, "wb") as f_out:
        for p in r1_parts:
            with open(p, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
    with open(merged_r2, "wb") as f_out:
        for p in r2_parts:
            with open(p, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)

    # Remove intermediate files and folder
    shutil.rmtree(per_genome_dir)

if __name__ == "__main__":
    main()