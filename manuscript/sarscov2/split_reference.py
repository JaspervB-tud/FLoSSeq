import argparse
import os
from Bio import SeqIO
import copy

ID_IDX = 0
LINEAGE_IDX = 13

def read_metadata(path):
    seq2lin = {}
    metadata = {}
    sequences_per_lineage = {}

    with open(path, "r") as f_in:
        header = next(f_in)
        for line in f_in:
            fields = line.strip().split("\t")
            seq_id = fields[ID_IDX].strip()
            lineage = fields[LINEAGE_IDX].strip()

            seq2lin[seq_id] = lineage
            metadata[seq_id] = line
            if lineage not in sequences_per_lineage:
                sequences_per_lineage[lineage] = set()
            sequences_per_lineage[lineage].add(seq_id)

    for lineage in sequences_per_lineage:
        sequences_per_lineage[lineage] = sorted(list(sequences_per_lineage[lineage]))

    return header, metadata, seq2lin, sequences_per_lineage

def main():
    parser = argparse.ArgumentParser(description="Split SARS-CoV-2 reference sequences by lineage.")
    parser.add_argument("--fasta", required=True, help="Path to the input FASTA file containing SARS-CoV-2 sequences.")
    parser.add_argument("--metadata", required=True, help="Path to the metadata TSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the split FASTA files.")
    args = parser.parse_args()

    # Read metadata
    header, metadata, seq2lin, sequences_per_lineage = read_metadata(args.metadata)
    records = {record.id: record for record in SeqIO.parse(args.fasta, "fasta")}

    # Split per lineage and save to separate FASTA files
    for lineage, seq_ids in sequences_per_lineage.items():
        lineage_dir = f"{args.output_dir}/{lineage}"
        os.makedirs(lineage_dir, exist_ok=True)

        # Remap sequences to integers
        cur_sequences = []
        mapping = {}
        for s_idx, seq_id in enumerate(seq_ids):
            record = copy.deepcopy(records[seq_id]) # make copy to avoid modifying original
            mapping[record.id] = s_idx
            record.id = str(s_idx)
            record.description = ""  # clear description
            cur_sequences.append(record)

        # Write remapped sequences and mapping file
        SeqIO.write(cur_sequences, f"{lineage_dir}/sequences_remapped.fasta", "fasta")
        with open(f"{lineage_dir}/sequence_mapping.txt", "w") as f_out:
            for record_id, new_id in mapping.items():
                f_out.write(f"{new_id}\t{record_id}\n")

    # Remap sequences in main folder (if remapping doesn't exist)
    cur_sequences = []
    mapping = {}
    seq_ids = sorted(list(records.keys()))
    for s_idx, seq_id in enumerate(seq_ids):
        record = copy.deepcopy(records[seq_id]) # make copy to avoid modifying original
        mapping[record.id] = s_idx
        record.id = str(s_idx)
        record.description = ""  # clear description
        cur_sequences.append(record)

    if not os.path.exists(f"{args.output_dir}/sequences_remapped.fasta") and not os.path.exists(f"{args.output_dir}/sequence_mapping.txt"):
        SeqIO.write(cur_sequences, f"{args.output_dir}/sequences_remapped.fasta", "fasta")
        
        # Write remapping
        with open(f"{args.output_dir}/sequence_mapping.txt", "w") as f_out:
            for record_id, new_id in mapping.items():
                f_out.write(f"{new_id}\t{record_id}\n")

if __name__ == "__main__":
    main()