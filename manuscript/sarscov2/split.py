from Bio import SeqIO
import argparse
import random
import os

# Constants for metadata parsing
COLLECTION_DATE_INDEX = 5
LINEAGE_INDEX = 13

def read_metadata(path, train_start, train_end, test_start, test_end):
    """
    Reads metadata from the given path and filters sequences based on collection date. Returns dictionaries for training and testing sequences, as well as a metadata dictionary.

    Parameters:
    -----------
    path : str
        Path to the metadata file (tab-separated from GISAID).
    train_start : str
        Start date for training sequences (inclusive) in YYYY-MM-DD format.
    train_end : str
        End date for training sequences (inclusive) in YYYY-MM-DD format.
    test_start : str
        Start date for testing sequences (inclusive) in YYYY-MM-DD format.
    test_end : str
        End date for testing sequences (inclusive) in YYYY-MM-DD format.

    Returns:
    --------
    train : dict
        Dictionary mapping lineages to sets of sequence IDs for training.
    test : dict
        Dictionary mapping lineages to sets of sequence IDs for testing.
    metadata : dict
        Dictionary mapping sequence IDs to their corresponding metadata lines.
    """
    train = {}
    test = {}
    metadata = {}

    with open(path, "r") as f_in:
        header = next(f_in) #skip header
        for line in f_in:
            fields = line.strip().split("\t")
            seq_id = fields[0].strip()
            collection_date = fields[COLLECTION_DATE_INDEX].strip()
            lineage = fields[LINEAGE_INDEX].strip()

            # Parse date
            date_parts = collection_date.split("-")
            if len(date_parts) == 2: #sometimes, only year and month are provided
                year = date_parts[0]
                month = date_parts[1].zfill(2)
                collection_date = f"{year}-{month}-01" #set day to 1 for uniformity
            elif len(date_parts) == 3:
                year = date_parts[0]
                month = date_parts[1].zfill(2)
                day = date_parts[2].zfill(2)
                collection_date = f"{year}-{month}-{day}"
            else:
                continue #skip invalid dates

            metadata[seq_id] = line

            # Check date
            if train_start <= collection_date <= train_end: #train sequences
                if lineage not in train:
                    train[lineage] = set()
                train[lineage].add(seq_id)
            if test_start <= collection_date <= test_end: #test sequences
                if lineage not in test:
                    test[lineage] = set()
                test[lineage].add(seq_id)

    return train, test, metadata, header

def read_sequences(path):
    """
    Reads sequences from the given FASTA file and returns a dictionary mapping sequence IDs to SeqRecord objects.

    Parameters:
    -----------
    path : str
        Path to the FASTA file containing the sequences.

    Returns:
    --------
    sequences : dict
        Dictionary mapping sequence IDs to SeqRecord objects.
    """
    sequences = {}
    for record in SeqIO.parse(path, "fasta"):
        sequences[record.id] = record
    return sequences

def main():
    parser = argparse.ArgumentParser(description="Create SARS-CoV-2 train/test splits.")
    parser.add_argument("--input_fasta", required=True, help="Fasta file containing SARS-CoV-2 reference genomes")
    parser.add_argument("--input_metadata", required=True, help="Metadata file for the reference genomes")
    parser.add_argument("--output_folder", required=True, help="Output folder for train/test splits")

    parser.add_argument("--train_start", required=True, help="Start date for training sequences (YYYY-MM-DD)")
    parser.add_argument("--train_end", required=True, help="End date for training sequences (YYYY-MM-DD)")
    parser.add_argument("--test_start", required=True, help="Start date for testing sequences (YYYY-MM-DD)")
    parser.add_argument("--test_end", required=True, help="End date for testing sequences (YYYY-MM-DD)")

    parser.add_argument("--max_genomes_train", type=int, default=200, help="Maximum number of genomes per lineage in training set")
    parser.add_argument("--max_genomes_test", type=int, default=100, help="Maximum number of genomes per lineage in testing set")

    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility of downsampling")
    args = parser.parse_args()

    train, test, metadata, header = read_metadata(
        args.input_metadata,
        args.train_start,
        args.train_end,
        args.test_start,
        args.test_end
    )
    sequences = read_sequences(args.input_fasta)

    # Subsample to $max_genomes_{train,test} per lineage
    lineages = sorted(set(train.keys())) #only consider lineages in training set
    final_train = []
    final_test = []

    random.seed(args.seed)
    for lineage in lineages: #only consider lineages that have training data (based on temporal cut-off)
        # Sort for reproducibility before subsampling
        train_ids = sorted(list(train.get(lineage, [])))
        test_ids = sorted(list(test.get(lineage, [])))

        # Subsample training
        if len(train_ids) > args.max_genomes_train:
            train_ids = random.sample(train_ids, args.max_genomes_train)
        final_train.extend(train_ids)

        # Subsample testing
        if len(test_ids) > args.max_genomes_test:
            test_ids = random.sample(test_ids, args.max_genomes_test)
        final_test.extend(test_ids)

    # Write training data (both a multifasta and per lineage both with metadata)
    os.makedirs(f"{args.output_folder}/Train", exist_ok=True)
    with open(f"{args.output_folder}/Train/sequences.fasta", "w") as train_fasta, \
            open(f"{args.output_folder}/Train/metadata.tsv", "w") as train_meta:
        
        # First, overall multifasta and metadata
        train_meta.write(header) #write header for metadata

        per_lineage = {}
        for seq_id in final_train:
            if seq_id in sequences and seq_id in metadata:
                SeqIO.write(sequences[seq_id], train_fasta, "fasta")
                train_meta.write(metadata[seq_id])
                lineage = metadata[seq_id].strip().split("\t")[LINEAGE_INDEX].strip()
                if lineage not in per_lineage:
                    per_lineage[lineage] = []
                per_lineage[lineage].append(seq_id)
            else:
                print(f"Warning: Sequence ID {seq_id} not found in sequences or metadata.")

        # Then, per lineage
        for lineage, ids in per_lineage.items():
            os.makedirs(f"{args.output_folder}/Train/{lineage}", exist_ok=True)
            with open(f"{args.output_folder}/Train/{lineage}/sequences.fasta", "w") as lineage_fasta, \
                    open(f"{args.output_folder}/Train/{lineage}/metadata.tsv", "w") as lineage_meta:
                
                lineage_meta.write(header) #write header for metadata

                for seq_id in ids:
                    SeqIO.write(sequences[seq_id], lineage_fasta, "fasta")
                    lineage_meta.write(metadata[seq_id])

    # Write testing data (only a single multifasta with metadata)
    os.makedirs(f"{args.output_folder}/Test", exist_ok=True)
    with open(f"{args.output_folder}/Test/sequences.fasta", "w") as test_fasta, \
            open(f"{args.output_folder}/Test/metadata.tsv", "w") as test_meta:
        
        test_meta.write(header) #write header for metadata
        
        for seq_id in final_test:
            if seq_id in sequences and seq_id in metadata:
                SeqIO.write(sequences[seq_id], test_fasta, "fasta")
                test_meta.write(metadata[seq_id])
            else:
                print(f"Warning: Sequence ID {seq_id} not found in sequences or metadata.")

if __name__ == "__main__":
    main()