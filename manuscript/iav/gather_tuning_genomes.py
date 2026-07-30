import argparse
import os
from Bio import SeqIO
import copy

ID_IDX = 0 # index for sequence ID in metadata file
CLADE_IDX = 15 # index for clade in metadata file

def read_metadata(path):
    seq2clade = {}
    with open(path, "r") as f_in:
        next(f_in) # skip header
        for line in f_in:
            fields = line.strip().split("\t")
            seq_id = fields[ID_IDX].strip()
            clade = fields[CLADE_IDX].strip()
            seq2clade[seq_id] = clade
    
    return seq2clade

def read_fasta(path):
    sequences = {}
    for record in SeqIO.parse(path, "fasta"):
        sequences[record.id] = record

    return sequences

# NOTE: The following functions are effectively the same, but kept separate for clarity
def process_medoid(path, mapping):
    selection = set()
    with open(path, "r") as f_in:
        for line in f_in:
            seq_id = mapping[line.strip()] # medoid file stores indices
            selection.add(seq_id)

    return sorted(list(selection))

def process_hierarchical(path, mapping):
    selection = set()
    with open(path, "r") as f_in:
        for line in f_in:
            seq_id = mapping[line.strip()] # hierarchical file stores indices
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
    parser = argparse.ArgumentParser(description="Gather reference genomes for IAV parameter tuning")
    parser.add_argument("--metadata", required=True, help="Path to metadata file")
    parser.add_argument("--fasta_HA", required=True, help="Path to fasta file")
    parser.add_argument("--fasta_NA", required=True, help="Path to fasta file")
    parser.add_argument("--mapping_prefix", required=True, help="Path to mapping file (for medoid and hierarchical selections)")
    parser.add_argument("--input_dir", required=True, help="Directory containing selections")
    parser.add_argument("--output_dir", required=True, help="Directory to save output fasta files")
    args = parser.parse_args()

    seq2clade = read_metadata(args.metadata)
    sequences_HA = read_fasta(args.fasta_HA)
    sequences_NA = read_fasta(args.fasta_NA)

    os.makedirs(args.output_dir, exist_ok=True)

    clades = set(seq2clade.values())
    mapping_per_clade = {clade: {} for clade in clades}
    for clade in clades:
        with open(f"{args.mapping_prefix}/{clade}/sequences_concatenated_ns_mapping.txt", "r") as f_in:
            for line in f_in:
                idx, seq_id = line.strip().split(",")
                mapping_per_clade[clade][idx] = seq_id

    # Determine singleton clades (always include these)
    singleton_clades = {}
    for clade in clades:
        seq_ids = [seq_id for seq_id, c in seq2clade.items() if c == clade]
        if len(seq_ids) == 1:
            singleton_clades[clade] = seq_ids[0]

    # Medoid selections
    parameters = [
        "mash_s500",
        "mash_s1000",
        "sourmash_s6",
        "sourmash_s3",
        "sourmash_cosine_s6",
        "sourmash_cosine_s3",
    ]
    for param in parameters:
        selection = []
        for clade in clades:
            if clade in singleton_clades:
                selection.append(singleton_clades[clade])
            else:
                base_folder = f"{args.input_dir}/{clade}"
                selection_path = f"{base_folder}/medoid_{param}.txt"
                selection += process_medoid(selection_path, mapping_per_clade[clade])
        # Create output fasta files for HA and NA
        output_fasta_path = f"{args.output_dir}/selection_medoid_{param}.fasta"
        with open(output_fasta_path, "w") as f_out:
            for seq_id in selection:
                HA_seq = copy.deepcopy(sequences_HA[seq_id])
                HA_seq.id = f"{seq_id}_HA"
                HA_seq.description = ""
                NA_seq = copy.deepcopy(sequences_NA[seq_id])
                NA_seq.id = f"{seq_id}_NA"
                NA_seq.description = ""
                # Store as separate records in the same fasta file
                SeqIO.write(HA_seq, f_out, "fasta")
                SeqIO.write(NA_seq, f_out, "fasta")

    # Hierarchical selections
    parameters = [
        "mash_s500",
        "mash_s1000",
        "sourmash_s6",
        "sourmash_s3",
        "sourmash_cosine_s6",
        "sourmash_cosine_s3",
    ]
    thresholds = [0.95, 0.97, 0.99, 0.999]
    percentiles = [1, 25, 50, 75, 90]
    for param in parameters:
        # Static selections
        for t in thresholds:
            # Fetch selected sequence IDs
            selection = []
            for clade in clades:
                if clade in singleton_clades:
                    selection.append(singleton_clades[clade])
                else:
                    base_folder = f"{args.input_dir}/{clade}"
                    selection_path = f"{base_folder}/hierarchical_{param}_t{t}.txt"
                    selection += process_hierarchical(selection_path, mapping_per_clade[clade])
            # Create output fasta files for HA and NA
            output_fasta_path = f"{args.output_dir}/selection_hierarchical_{param}_t{t}.fasta"
            with open(output_fasta_path, "w") as f_out:
                for seq_id in selection:
                    HA_seq = copy.deepcopy(sequences_HA[seq_id])
                    HA_seq.id = f"{seq_id}_HA"
                    HA_seq.description = ""
                    NA_seq = copy.deepcopy(sequences_NA[seq_id])
                    NA_seq.id = f"{seq_id}_NA"
                    NA_seq.description = ""
                    # Store as separate records in the same fasta file
                    SeqIO.write(HA_seq, f_out, "fasta")
                    SeqIO.write(NA_seq, f_out, "fasta")

        # Dynamic selections
        for p in percentiles:
            # Fetch selected sequence IDs
            selection = []
            for clade in clades:
                if clade in singleton_clades:
                    selection.append(singleton_clades[clade])
                else:
                    base_folder = f"{args.input_dir}/{clade}"
                    selection_path = f"{base_folder}/hierarchical_{param}_p{p}.txt"
                    selection += process_hierarchical(selection_path, mapping_per_clade[clade])
            # Create output fasta files for HA and NA
            output_fasta_path = f"{args.output_dir}/selection_hierarchical_{param}_p{p}.fasta"
            with open(output_fasta_path, "w") as f_out:
                for seq_id in selection:
                    HA_seq = copy.deepcopy(sequences_HA[seq_id])
                    HA_seq.id = f"{seq_id}_HA"
                    HA_seq.description = ""
                    NA_seq = copy.deepcopy(sequences_NA[seq_id])
                    NA_seq.id = f"{seq_id}_NA"
                    NA_seq.description = ""
                    # Store as separate records in the same fasta file
                    SeqIO.write(HA_seq, f_out, "fasta")
                    SeqIO.write(NA_seq, f_out, "fasta")

    # VSEARCH selections
    # Static selections
    thresholds = [0.95, 0.97, 0.99, 0.999]
    for t in thresholds:
        # Fetch selected sequence IDs
        selection = []
        for clade in clades:
            if clade in singleton_clades:
                selection.append(singleton_clades[clade])
            else:
                base_folder = f"{args.input_dir}/{clade}"
                selection_path = f"{base_folder}/vsearch_t{t}.fasta"
                selection += process_vsearch(selection_path)
        # Create output fasta files for HA and NA
        output_fasta_path = f"{args.output_dir}/selection_vsearch_t{t}.fasta"
        with open(output_fasta_path, "w") as f_out:
            for seq_id in selection:
                HA_seq = copy.deepcopy(sequences_HA[seq_id])
                HA_seq.id = f"{seq_id}_HA"
                HA_seq.description = ""
                NA_seq = copy.deepcopy(sequences_NA[seq_id])
                NA_seq.id = f"{seq_id}_NA"
                NA_seq.description = ""
                # Store as separate records in the same fasta file
                SeqIO.write(HA_seq, f_out, "fasta")
                SeqIO.write(NA_seq, f_out, "fasta")
    # Dynamic selections
    parameters = [
        "mash_s500",
        "mash_s1000",
        "sourmash_s6",
        "sourmash_s3",
        "sourmash_cosine_s6",
        "sourmash_cosine_s3",
    ]
    percentiles = [1, 25, 50, 75, 90]
    for param in parameters:
        for p in percentiles:
            # Fetch selected sequence IDs
            selection = []
            for clade in clades:
                if clade in singleton_clades:
                    selection.append(singleton_clades[clade])
                else:
                    base_folder = f"{args.input_dir}/{clade}"
                    selection_path = f"{base_folder}/vsearch_{param}_p{p}.fasta"
                    selection += process_vsearch(selection_path)
            # Create output fasta files for HA and NA
            output_fasta_path = f"{args.output_dir}/selection_vsearch_{param}_p{p}.fasta"
            with open(output_fasta_path, "w") as f_out:
                for seq_id in selection:
                    HA_seq = copy.deepcopy(sequences_HA[seq_id])
                    HA_seq.id = f"{seq_id}_HA"
                    HA_seq.description = ""
                    NA_seq = copy.deepcopy(sequences_NA[seq_id])
                    NA_seq.id = f"{seq_id}_NA"
                    NA_seq.description = ""
                    # Store as separate records in the same fasta file
                    SeqIO.write(HA_seq, f_out, "fasta")
                    SeqIO.write(NA_seq, f_out, "fasta")

    # ReSeT selections
    parameters = [
        "mash_s500",
        "mash_s1000",
        "sourmash_s6",
        "sourmash_s3",
        "sourmash_cosine_s6",
        "sourmash_cosine_s3",
    ]
    costs = ["0.000000", "0.000001", "0.000010", "0.000100", "0.001000", "0.010000", "0.100000", "1.000000"]
    scales = ["0.00000", "0.00001", "0.00010", "0.00100", "0.01000", "0.10000", "1.00000"]
    for param in parameters:
        for cost in costs:
            for scale in scales:
                # Fetch selected sequence IDs
                base_folder = f"{args.input_dir}"
                selection_path = f"{base_folder}/reset_{param}_cost-{cost}_scale-{scale}.txt"
                selection = process_reset(selection_path)
                # Create output fasta files for HA and NA
                output_fasta_path = f"{args.output_dir}/selection_reset_{param}_cost-{cost}_scale-{scale}.fasta"
                with open(output_fasta_path, "w") as f_out:
                    for seq_id in selection:
                        HA_seq = copy.deepcopy(sequences_HA[seq_id])
                        HA_seq.id = f"{seq_id}_HA"
                        HA_seq.description = ""
                        NA_seq = copy.deepcopy(sequences_NA[seq_id])
                        NA_seq.id = f"{seq_id}_NA"
                        NA_seq.description = ""
                        # Store as separate records in the same fasta file
                        SeqIO.write(HA_seq, f_out, "fasta")
                        SeqIO.write(NA_seq, f_out, "fasta")

if __name__ == "__main__":
    main()