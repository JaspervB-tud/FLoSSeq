import argparse
import os
from Bio import SeqIO

ID_IDX = 0 # index for sequence ID in metadata file
LIN_IDX = 13 # index for lineage in metadata file

def read_metadata(path):
    seq2lin = {}
    with open(path, "r") as f_in:
        next(f_in) # skip header
        for line in f_in:
            fields = line.strip().split("\t")
            seq_id = fields[ID_IDX].strip()
            lineage = fields[LIN_IDX].strip()
            seq2lin[seq_id] = lineage
    
    return seq2lin

def read_fasta(path):
    sequences = {}
    for record in SeqIO.parse(path, "fasta"):
        sequences[record.id] = record

    return sequences

# NOTE: The following functions are effectively the same, but kept separate for clarity
def process_medoid(path):
    selection = set()
    with open(path, "r") as f_in:
        for line in f_in:
            seq_id = line.strip()
            selection.add(seq_id)

    return sorted(list(selection))

def process_hierarchical(path):
    selection = set()
    with open(path, "r") as f_in:
        for line in f_in:
            seq_id = line.strip()
            selection.add(seq_id)

    return sorted(list(selection))

def process_vsearch(path):
    # NOTE: vsearch saves output as .fasta
    selection = set()
    for record in SeqIO.parse(path, "fasta"):
        selection.add(record.id)

    return sorted(list(selection))

def process_reset(path):
    selection = set()
    with open(path, "r") as f_in:
        for line in f_in:
            seq_id = line.strip()
            selection.add(seq_id)
            
    return sorted(list(selection))

def main():
    parser = argparse.ArgumentParser(description="Gather reference genomes for SARS-CoV-2 parameter tuning")
    parser.add_argument("--metadata", required=True, help="Path to metadata file")
    parser.add_argument("--fasta", required=True, help="Path to fasta file")
    parser.add_argument("--input_dir", required=True, help="Directory containing selections")
    parser.add_argument("--output_dir", required=True, help="Directory to save output fasta files")
    args = parser.parse_args()

    seq2lin = read_metadata(args.metadata)
    sequences = read_fasta(args.fasta)

    os.makedirs(args.output_dir, exist_ok=True)

    lineages = set(seq2lin.values())

    # Determine singleton lineages (always include these)
    singleton_lineages = {}
    for lineage in lineages:
        seq_ids = [seq_id for seq_id, lin in seq2lin.items() if lin == lineage]
        if len(seq_ids) == 1:
            singleton_lineages[lineage] = seq_ids[0]

    # Medoid selections (may crash on singletons)
    parameters = [
        "mash_s5000",
        "mash_s10000",
        "sourmash-jaccard_s6",
        "sourmash-jaccard_s3",
        "sourmash-cosine_s6",
        "sourmash-cosine_s3"
    ]
    for param in parameters:
        selection = []
        # Fetch selected sequence IDs
        for lineage in lineages:
            if lineage in singleton_lineages:
                selection.append(singleton_lineages[lineage])
            else:
                base_folder = f"{args.input_dir}/{lineage}"
                selection_path = f"{base_folder}/medoid_{param}.txt"
                selection += process_medoid(selection_path)
        # Create output fasta file
        output_fasta_path = f"{args.output_dir}/selection_medoid_{param}.fasta"
        with open(output_fasta_path, "w") as f_out:
            for seq_id in selection:
                SeqIO.write(sequences[seq_id], f_out, "fasta")

    # Hierarchical clustering selections (may crash on singletons)
    methods = [
        "mash_s5000",
        "mash_s10000",
        "sourmash-jaccard_s6",
        "sourmash-jaccard_s3",
        "sourmash-cosine_s6",
        "sourmash-cosine_s3"
    ]
    for method in methods:
        # Static selections
        thresholds = [0.99, 0.999, 0.9999, 0.99999]
        for t in thresholds:
            # Fetch selected sequence IDs
            selection = []
            for lineage in lineages:
                if lineage in singleton_lineages:
                    selection.append(singleton_lineages[lineage])
                else:
                    base_folder = f"{args.input_dir}/{lineage}"
                    selection_path = f"{base_folder}/hierarchical_{method}_t{t}.txt"
                    selection += process_hierarchical(selection_path)
            # Create output fasta file
            output_fasta_path = f"{args.output_dir}/selection_hierarchical_{method}_t{t}.fasta"
            with open(output_fasta_path, "w") as f_out:
                for seq_id in selection:
                    SeqIO.write(sequences[seq_id], f_out, "fasta")
        # Dynamic selections
        percentiles = [1, 25, 50, 75, 90]
        for p in percentiles:
            # Fetch selected sequence IDs
            selection = []
            for lineage in lineages:
                if lineage in singleton_lineages:
                    selection.append(singleton_lineages[lineage])
                else:
                    base_folder = f"{args.input_dir}/{lineage}"
                    selection_path = f"{base_folder}/hierarchical_{method}_p{p}.txt"
                    selection += process_hierarchical(selection_path)
            # Create output fasta file
            output_fasta_path = f"{args.output_dir}/selection_hierarchical_{method}_p{p}.fasta"
            with open(output_fasta_path, "w") as f_out:
                for seq_id in selection:
                    SeqIO.write(sequences[seq_id], f_out, "fasta")

    # VSEARCH selections (may crash on singletons)
    # Static selections
    thresholds = [0.99, 0.999, 0.9999, 0.99999]
    for t in thresholds:
        # Fetch selected sequence IDs
        selection = []
        for lineage in lineages:
            if lineage in singleton_lineages:
                selection.append(singleton_lineages[lineage])
            else:
                base_folder = f"{args.input_dir}/{lineage}"
                selection_path = f"{base_folder}/vsearch_t{t}.fasta"
                selection += process_vsearch(selection_path)
        # Create output fasta file
        output_fasta_path = f"{args.output_dir}/selection_vsearch_t{t}.fasta"
        with open(output_fasta_path, "w") as f_out:
            for seq_id in selection:
                SeqIO.write(sequences[seq_id], f_out, "fasta")
    # Dynamic selections
    methods = [
        "mash_s5000",
        "mash_s10000",
        "sourmash-jaccard_s6",
        "sourmash-jaccard_s3",
        "sourmash-cosine_s6",
        "sourmash-cosine_s3"
    ]
    percentiles = [1, 25, 50, 75, 90]
    for method in methods:
        for p in percentiles:
            # Fetch selected sequence IDs
            selection = []
            for lineage in lineages:
                if lineage in singleton_lineages:
                    selection.append(singleton_lineages[lineage])
                else:
                    base_folder = f"{args.input_dir}/{lineage}"
                    selection_path = f"{base_folder}/vsearch_{method}_p{p}.fasta"
                    selection += process_vsearch(selection_path)
            # Create output fasta file
            output_fasta_path = f"{args.output_dir}/selection_vsearch_{method}_p{p}.fasta"
            with open(output_fasta_path, "w") as f_out:
                for seq_id in selection:
                    SeqIO.write(sequences[seq_id], f_out, "fasta")

    # ReSeT selections (always runs)
    methods = [
        "mash_s5000",
        "mash_s10000",
        "sourmash-jaccard_s6",
        "sourmash-jaccard_s3",
        "sourmash-cosine_s6",
        "sourmash-cosine_s3"
    ]
    costs = ["0.000000", "0.000001", "0.000010", "0.000100", "0.001000", "0.010000", "0.100000", "1.000000"]
    scales = ["0.00000", "0.00001", "0.00010", "0.00100", "0.01000", "0.10000", "1.00000"]
    for method in methods:
        for cost in costs:
            for scale in scales:
                # Fetch selected sequence IDs
                base_folder = f"{args.input_dir}"
                selection = process_reset(f"{base_folder}/reset_{method}_cost-{cost}_scale-{scale}.txt")
                # Create output fasta file
                output_fasta_path = f"{args.output_dir}/selection_reset_{method}_cost-{cost}_scale-{scale}.fasta"
                with open(output_fasta_path, "w") as f_out:
                    for seq_id in selection:
                        SeqIO.write(sequences[seq_id], f_out, "fasta")

if __name__ == "__main__":
    main()