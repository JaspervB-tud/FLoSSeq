from Bio import SeqIO
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Partition genomes into separate files for troubleshooting sourmash.")
    parser.add_argument("--input_fasta", type=str, required=True, help="Path to input FASTA file containing genomes.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to write partitioned genome files.")
    parser.add_argument("--partition_size", type=int, default=15000, help="Number of genomes per partition.")
    args = parser.parse_args() 

    input_file = args.input_fasta
    output_folder = args.output_folder
    partition_size = args.partition_size

    genomes = list(SeqIO.parse(input_file, "fasta"))
    total_genomes = len(genomes)

    os.makedirs(output_folder, exist_ok=True)
    for part_idx, start_idx in enumerate(range(0, total_genomes, partition_size)):
        end_idx = min(start_idx + partition_size, total_genomes)
        partition_genomes = genomes[start_idx:end_idx]
        output_file = os.path.join(output_folder, f"partition_{part_idx + 1}.fasta")
        SeqIO.write(partition_genomes, output_file, "fasta")
        print(f"Wrote {len(partition_genomes)} genomes to {output_file}")

if __name__ == "__main__":
    main()