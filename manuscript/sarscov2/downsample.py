from Bio import SeqIO
import argparse
import os
import math
import random

# Constants for metadata parsing
COLLECTION_DATE_INDEX = 5
LINEAGE_INDEX = 13

def read_metadata(path, min_date="1990-01-01", max_date="2100-12-31"):
    """
    Reads metadata from the given path and filters sequences based on collection date. Returns a dictionary of references, metadata, dates, and lineages.

    Parameters:
    -----------
    path : str
        Path to the metadata file (tab-separated from GISAID).
    min_date : str
        Minimum collection date (inclusive) in YYYY-MM-DD format.
    max_date : str
        Maximum collection date (inclusive) in YYYY-MM-DD format.

    Returns:
    --------
    references : dict
        Dictionary mapping lineages to sets of sequence IDs that fall within the specified date range.
    metadata : dict
        Dictionary mapping sequence IDs to their corresponding metadata lines.
    dates : dict
        Dictionary mapping sequence IDs to their parsed collection dates in YYYY-MM-DD format.
    lineages : dict
        Dictionary mapping sequence IDs to their corresponding lineages.
    """
    references = {}
    metadata = {}
    dates = {}
    lineages = {}

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
            dates[seq_id] = collection_date
            lineages[seq_id] = lineage

            # Check date
            if min_date <= collection_date <= max_date:
                if lineage not in references:
                    references[lineage] = set()
                references[lineage].add(seq_id)

    return references, metadata, dates, lineages, header

def main():
    parser = argparse.ArgumentParser(description="Downsample SARS-CoV-2 genomes.")
    parser.add_argument("--input_fasta", required=True, help="Fasta file containing SARS-CoV-2 genomes to downsample")
    parser.add_argument("--input_metadata", required=True, help="Metadata file corresponding to the fasta file")
    parser.add_argument("--output_folder", required=True, help="Output folder to write downsampled fasta and metadata")

    parser.add_argument("--ref_start", required=True, help="Start date for the sequences (YYYY-MM-DD)")
    parser.add_argument("--ref_end", required=True, help="End date for the sequences (YYYY-MM-DD)")

    parser.add_argument("--monthly_cap", type=int, default=3500, help="Maximum number of sequences to select per month")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility of downsampling")
    args = parser.parse_args()

    # Fetch data
    references, metadata, dates, lineages, header = read_metadata(
        args.input_metadata,
        args.ref_start,
        args.ref_end,
    )

    # Gather reference sequences
    reference_lineages = set(references.keys())
    reference_sequences = set()
    for lineage in reference_lineages:
        reference_sequences.update(references[lineage])

    # Gather reference sequences per month
    reference_sequences_by_month = {}
    for seq_id in reference_sequences:
        collection_date = dates[seq_id]
        year_month = collection_date[:7] #YYYY-MM
        lineage = lineages[seq_id]
        if year_month not in reference_sequences_by_month:
            reference_sequences_by_month[year_month] = {}
        if lineage not in reference_sequences_by_month[year_month]:
            reference_sequences_by_month[year_month][lineage] = []
        reference_sequences_by_month[year_month][lineage].append(seq_id)

    # Downsample reference sequences
    rng = random.Random(args.seed)
    selection = {}
    selection_flat = set()
    total_selected = 0
    for year_month in sorted(reference_sequences_by_month.keys()):
        lineage_to_sequences = reference_sequences_by_month[year_month]
        cur_lineages = sorted(lineage_to_sequences.keys())
        counts ={l: len(lineage_to_sequences[l]) for l in cur_lineages}

        total_sequences = sum(counts.values())
        total_cap = min(args.monthly_cap, total_sequences)
        total_lineages = len(cur_lineages)

        # Allocate at least 1 sequence per lineage
        alloc = {l: 0 for l in cur_lineages}
        if total_cap >= total_lineages:
            for l in cur_lineages:
                alloc[l] = 1
            remaining = total_cap - total_lineages
        else:
            remaining = total_cap

        # Remaining capacity
        cap_left = {l: counts[l] - alloc[l] for l in cur_lineages}
        total_left = sum(cap_left.values())

        if remaining > 0 and total_left > 0:
            # Proportions for remaining allocation
            proportions = {l: remaining * (cap_left[l] / total_left) for l in cur_lineages}

            # Take floors first for initial allocation
            for l in cur_lineages:
                add = min(cap_left[l], int(math.floor(proportions[l])))
                alloc[l] += add
                cap_left[l] -= add

            # Distribute remaining by largest remainder
            still = total_cap - sum(alloc.values())
            if still > 0:
                # Lineages with spare capacity
                candidates = [l for l in cur_lineages if cap_left[l] > 0]
                candidates.sort(key = lambda l: proportions[l] - math.floor(proportions[l]), reverse=True) #largest remainder first
                # Iterate until full
                idx = 0
                while still > 0 and len(candidates) > 0:
                    l = candidates[idx % len(candidates)]
                    if cap_left[l] > 0:
                        alloc[l] += 1
                        cap_left[l] -= 1
                        still -= 1
                        if cap_left[l] == 0:
                            candidates = [x for x in candidates if cap_left[x] > 0]
                            idx = 0
                            continue
                    idx += 1

        # Sanity check
        assert sum(alloc.values()) <= total_cap

        # Randomly select sequences
        for l in cur_lineages:
            num = alloc[l]
            if num <= 0:
                continue #safety skip
            candidates = lineage_to_sequences[l]
            chosen = rng.sample(candidates, num)
            if l not in selection:
                selection[l] = set()
            selection[l].update(chosen)
            for s in chosen:
                selection_flat.add(s)
            total_selected += len(chosen)

    print("Finished downsampling. Total selected sequences:", total_selected)

    # Define output paths
    ref_output_path = f"{args.output_folder}/{args.ref_start}_{args.ref_end}"

    # Create output directories
    os.makedirs(ref_output_path, exist_ok=True)

    # Create output files while parsing sequences (note that metadata will lose its header here)
    processed_seqids = set() #deal with duplicates
    sequence_num = 0
    with open(f"{ref_output_path}/sequences.fasta", "w") as ref_fasta_out, \
            open(f"{ref_output_path}/metadata.tsv", "w") as ref_meta_out, \
            open(f"{ref_output_path}/sequence_mapping.txt", "w") as ref_map_out:
        
        # Write header for metadata
        ref_meta_out.write(header)

        # Process sequences and write selected ones
        for record in SeqIO.parse(args.input_fasta, "fasta"):
            seq_id = record.id
            if seq_id in processed_seqids:
                continue #skip duplicates
            # Reference sequences
            if seq_id in selection_flat:
                SeqIO.write(record, ref_fasta_out, "fasta")
                ref_meta_out.write(metadata[seq_id])
                ref_map_out.write(f"{sequence_num}\t{seq_id}\n")
                processed_seqids.add(seq_id)
                sequence_num += 1
            if len(processed_seqids) % 5_000 == 0 and len(processed_seqids) > 0:
                print(f"Processed {len(processed_seqids)} sequences...")

if __name__ == "__main__":
    main()
