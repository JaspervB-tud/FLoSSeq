import os
import argparse
from Bio import SeqIO, SeqRecord
import pandas as pd
import glob
from pathlib import Path
import random

def gather(input_dir, min_date="1990-01-01", max_date="2100-12-31", max_n_content=0.001, max_non_acgt_content=0.01):
    """
    This function gathers all IAV metadata and genomes from the provided input directory. For some reason,
    the metadata files were in .xls format and contained end line characters that ruined parsing. This function
    reads all metadata files as panda dataframes and merges them. It checks if all genome files are present,
    and filters based on time, N-content and non-ACGT-content.
    """
    def collapse_newlines(txt):
        """
        Helper function for collapsing newlines in a string.
        """
        if isinstance(txt, str):
            txt = txt.replace("\r\n", "\n").replace("\r", "\n")
            txt = " ".join(txt.splitlines())
        return txt
    
    # Read metadata first
    dfs = []
    for file in sorted(input_dir.glob("*.xls")):
        df = pd.read_excel(file, engine="xlrd")
        dfs.append(df)
    metadata = pd.concat(dfs, ignore_index=True)
    # Collapse newlines
    obj_cols = metadata.select_dtypes(include=["object"]).columns
    metadata[obj_cols] = metadata[obj_cols].map(collapse_newlines)
    metadata = metadata.set_index("Isolate_Id", drop=True) #set sequence id as index

    # Read genomes
    genomes = {}
    for file in sorted(input_dir.glob("*.fasta")):
        for record in SeqIO.parse(file, "fasta"):
            seq_id = record.id.split("|")[0].strip()
            segment = record.id.split("|")[1].strip() #split by segment
            if seq_id not in genomes:
                genomes[seq_id] = {}
            genomes[seq_id][segment] = record

    # Parse genomes
    header = "\t".join(map(str, metadata.columns))
    dates = {}
    clades = {}
    metadata_dict = {}
    start_count = len(genomes)
    for seq_id in list(genomes.keys()):
        if seq_id not in metadata.index:
            print(f"Warning: Sequence ID {seq_id} not found in metadata, skipping...")
            genomes.pop(seq_id)
            continue
        
        meta_row = metadata.loc[seq_id]
        
        # Collection date
        collection_date = str(meta_row["Collection_Date"])
        parts = collection_date.split("-")
        if len(parts) == 1:
            genomes.pop(seq_id)
            continue
        elif len(parts) == 2:
            # Ensure MM padding and add day
            year = parts[0]
            month = parts[1].zfill(2)
            collection_date = f"{year}-{month}-01"
        elif len(parts) == 3:
            year = parts[0]
            month = parts[1].zfill(2)
            day = parts[2].zfill(2)
            collection_date = f"{year}-{month}-{day}"
        else:
            genomes.pop(seq_id)
            continue
        try:
            if not (min_date <= collection_date <= max_date):
                genomes.pop(seq_id)
                continue
        except:
            genomes.pop(seq_id)
            continue

        # Host
        host = str(meta_row["Host"]).strip().lower()
        if host != "human":
            genomes.pop(seq_id)
            continue

        # Clade
        clade = str(meta_row["Clade"]).strip()
        if clade == "" or clade.lower() == "unassigned":
            genomes.pop(seq_id)
            continue

        # N-content and non-ACGT-content
        concatenated_seq = ""
        for segment in sorted(genomes[seq_id].keys()):
            if segment in ["HA", "NA"]:  # Only consider HA and NA segments
                concatenated_seq += str(genomes[seq_id][segment].seq).upper()
        n_content = concatenated_seq.count("N") / len(concatenated_seq)
        non_acgt_content = sum([1 for base in concatenated_seq if base not in "ACGT"]) / len(concatenated_seq)
        if n_content > max_n_content or non_acgt_content > max_non_acgt_content:
            genomes.pop(seq_id)
            continue

        dates[seq_id] = collection_date
        clades[seq_id] = clade
        metadata_dict[seq_id] = "\t".join(map(str, meta_row.values))
    print(f"Number of genomes after filtering: {len(genomes)}/{start_count}")
    return genomes, metadata, metadata_dict, dates, clades, header

def main():
    parser = argparse.ArgumentParser(description="Filter IAV genomes based on date, host, N-content, and clade, and downsample per month and clade.")
    parser.add_argument("--input", type=str, help="Input directory containing IAV fasta and metadata files.", required=True)
    parser.add_argument("--output", type=str, help="Output directory to store filtered and downsampled genomes.", required=True)
    parser.add_argument("--reference_start_date", type=str, default="2024-01-01", help="Start date for reference genomes (YYYY-MM-DD).")
    parser.add_argument("--simulation_start_date", type=str, default="2025-01-01", help="Start date for simulated genomes (YYYY-MM-DD).")
    parser.add_argument("--test_start_date", type=str, default="2024-10-01", help="Start date for test genomes (YYYY-MM-DD).")
    parser.add_argument("--max_monthly_cap", type=int, default=2000, help="Maximum number of genomes to select per month.")
    parser.add_argument("--max_genomes_train", type=int, default=1000, help="Maximum number of genomes per clade in training set.")
    parser.add_argument("--max_genomes_test", type=int, default=500, help="Maximum number of genomes per clade in testing set.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for downsampling.")
    args = parser.parse_args()
    reference_start_date = args.reference_start_date
    simulation_start_date = args.simulation_start_date
    test_start_date = args.test_start_date
    max_monthly_cap = args.max_monthly_cap
    seed = args.seed

    genomes, metadata, metadata_dict, dates, clades, header = gather(Path(f"{args.input}"), min_date=reference_start_date)

    final_date = "0000-00-00"
    # Gather sequences per month
    genomes_per_month = {}
    for seq_id in genomes.keys():
        collection_date = dates[seq_id]
        month = collection_date[:7]  # YYYY-MM
        clade = clades[seq_id]
        if month not in genomes_per_month:
            genomes_per_month[month] = {}
        if clade not in genomes_per_month[month]:
            genomes_per_month[month][clade] = []
        genomes_per_month[month][clade].append(seq_id)
        if collection_date > final_date:
            final_date = collection_date

    # Downsample sequences
    rng = random.Random(seed)
    selection = {}
    selection_flat = set()
    total_selected = 0
    total_selected_ref = 0
    total_selected_sim = 0
    reference_genomes = []
    reference_clades = set()
    simulation_genomes = []
    for month in sorted(genomes_per_month.keys()):
        clade_to_sequences = genomes_per_month[month]
        cur_clades = sorted(clade_to_sequences.keys())
        counts = {c: len(clade_to_sequences[c]) for c in cur_clades}

        total_sequences = sum(counts.values())
        total_cap = min(max_monthly_cap, total_sequences)
        total_clades = len(cur_clades)

        # Allocate at least 1 sequence per clade
        alloc = {c: 0 for c in cur_clades}
        if total_cap >= total_clades:
            for c in cur_clades:
                alloc[c] = 1
            remaining = total_cap - total_clades
        else:
            remaining = total_cap

        # Remaining capacity
        cap_left = {c: counts[c] - alloc[c] for c in cur_clades}
        total_left = sum(cap_left.values())

        if remaining > 0 and total_left > 0:
            # Proportions for remaining allocation
            proportions = {c: remaining * (cap_left[c] / total_left) for c in cur_clades}

            # Take floors first for initial allocation
            for c in cur_clades:
                floor_alloc = int(proportions[c])
                alloc[c] += floor_alloc
            allocated_so_far = sum(alloc.values())
            fractional_parts = {c: proportions[c] - int(proportions[c]) for c in cur_clades}

            # Distribute remaining based on largest fractional parts
            while allocated_so_far < total_cap:
                best_clade = max(fractional_parts, key=fractional_parts.get)
                alloc[best_clade] += 1
                fractional_parts[best_clade] = 0  #don't pick it again immediately
                allocated_so_far += 1

        # Select sequences
        for c in cur_clades:
            selected_seqs = rng.sample(clade_to_sequences[c], alloc[c])
            for seq_id in selected_seqs:
                selection[seq_id] = genomes[seq_id]
                selection_flat.add(seq_id)
                total_selected += 1
                if simulation_start_date <= dates[seq_id]:
                    total_selected_sim += 1
                    simulation_genomes.append(seq_id)
                else:
                    total_selected_ref += 1
                    reference_genomes.append(seq_id)
                    reference_clades.add(c)
    reference_clades = sorted(list(reference_clades))

    # Organize per clade
    reference_genomes_per_clade = {clade: [] for clade in reference_clades}
    simulation_genomes_per_clade = {}
    for seq_id in reference_genomes:
        clade = clades[seq_id]
        reference_genomes_per_clade[clade].append(seq_id)
    for seq_id in simulation_genomes:
        clade = clades[seq_id]
        if clade in reference_clades: #only keep simulation genomes from clades present in reference set
            if clade not in simulation_genomes_per_clade:
                simulation_genomes_per_clade[clade] = []
            simulation_genomes_per_clade[clade].append(seq_id)

    # Process reference genomes into Train and Test
    train_genomes_per_clade = {}
    test_genomes_per_clade = {}
    train_clades = set()
    for seq_id in reference_genomes:
        date = dates[seq_id]
        clade = clades[seq_id]
        if date < test_start_date: #train set
            if clade not in train_genomes_per_clade:
                train_genomes_per_clade[clade] = []
                train_clades.add(clade)
            train_genomes_per_clade[clade].append(seq_id)
        else: #test set
            if clade not in test_genomes_per_clade:
                test_genomes_per_clade[clade] = []
            test_genomes_per_clade[clade].append(seq_id)

    # Sort for consistent ordering
    reference_genomes = []
    simulation_genomes = []
    train_genomes = []
    test_genomes = []
    for clade in reference_genomes_per_clade:
        reference_genomes_per_clade[clade] = sorted(reference_genomes_per_clade[clade])
        reference_genomes.extend(reference_genomes_per_clade[clade])
    for clade in simulation_genomes_per_clade:
        simulation_genomes_per_clade[clade] = sorted(simulation_genomes_per_clade[clade])
        simulation_genomes.extend(simulation_genomes_per_clade[clade])
    for clade in train_genomes_per_clade:
        cur_genomes = sorted(train_genomes_per_clade[clade])
        if len(cur_genomes) > args.max_genomes_train: #downsample training genomes
            cur_genomes = rng.sample(cur_genomes, args.max_genomes_train)
        train_genomes_per_clade[clade] = cur_genomes
        train_genomes.extend(cur_genomes)
    for clade in test_genomes_per_clade:
        cur_genomes = sorted(test_genomes_per_clade[clade])
        if len(cur_genomes) > args.max_genomes_test: #downsample testing genomes
            cur_genomes = rng.sample(cur_genomes, args.max_genomes_test)
        test_genomes_per_clade[clade] = cur_genomes
        test_genomes.extend(cur_genomes)
    train_genomes = sorted(train_genomes)
    test_genomes = sorted(test_genomes)

    # Create output directories and store genomes
    base_output_dir = args.output
    reference_dir = f"{base_output_dir}/Reference"
    simulation_dir = f"{base_output_dir}/Simulation"
    os.makedirs(reference_dir, exist_ok=True)
    os.makedirs(simulation_dir, exist_ok=True)

    genomes_separated_HA = {}
    genomes_separated_NA = {}
    genomes_concatenated = {}
    genomes_concatenated_ns = {}
    # Store reference genomes (overall, per clade, Train/Test)
    for clade in reference_clades:
        os.makedirs(f"{reference_dir}/{clade}", exist_ok=True)
        os.makedirs(f"{reference_dir}/Train/{clade}", exist_ok=True)
        os.makedirs(f"{reference_dir}/Test", exist_ok=True)
        # Create SeqRecord entries for easy writing
        for seq_id in reference_genomes_per_clade[clade]:
            ha_seq = genomes[seq_id]["HA"]
            na_seq = genomes[seq_id]["NA"]
            separated_HA = ha_seq.seq
            separated_NA = na_seq.seq
            concatenated_seq = ha_seq.seq + na_seq.seq
            concatenated_ns_seq = ha_seq.seq + "N" * 100 + na_seq.seq
            genomes_separated_HA[seq_id] = SeqRecord.SeqRecord(separated_HA, id=seq_id, description="")
            genomes_separated_NA[seq_id] = SeqRecord.SeqRecord(separated_NA, id=seq_id, description="")
            genomes_concatenated[seq_id] = SeqRecord.SeqRecord(concatenated_seq, id=seq_id, description="")
            genomes_concatenated_ns[seq_id] = SeqRecord.SeqRecord(concatenated_ns_seq, id=seq_id, description="")
        for seq_id in simulation_genomes_per_clade.get(clade, []):
            ha_seq = genomes[seq_id]["HA"]
            na_seq = genomes[seq_id]["NA"]
            separated_HA = ha_seq.seq
            separated_NA = na_seq.seq
            concatenated_seq = ha_seq.seq + na_seq.seq
            concatenated_ns_seq = ha_seq.seq + "N" * 100 + na_seq.seq
            genomes_separated_HA[seq_id] = SeqRecord.SeqRecord(separated_HA, id=seq_id, description="")
            genomes_separated_NA[seq_id] = SeqRecord.SeqRecord(separated_NA, id=seq_id, description="")
            genomes_concatenated[seq_id] = SeqRecord.SeqRecord(concatenated_seq, id=seq_id, description="")
            genomes_concatenated_ns[seq_id] = SeqRecord.SeqRecord(concatenated_ns_seq, id=seq_id, description="")
        # Write for this clade
        SeqIO.write([genomes_separated_HA[seq_id] for seq_id in reference_genomes_per_clade[clade]],
                    f"{reference_dir}/{clade}/sequences_HA.fasta", "fasta")
        SeqIO.write([genomes_separated_NA[seq_id] for seq_id in reference_genomes_per_clade[clade]],
                    f"{reference_dir}/{clade}/sequences_NA.fasta", "fasta")
        SeqIO.write([genomes_concatenated[seq_id] for seq_id in reference_genomes_per_clade[clade]],
                    f"{reference_dir}/{clade}/sequences_concatenated.fasta", "fasta")
        SeqIO.write([genomes_concatenated_ns[seq_id] for seq_id in reference_genomes_per_clade[clade]],
                    f"{reference_dir}/{clade}/sequences_concatenated_ns.fasta", "fasta")
        with open(f"{reference_dir}/{clade}/metadata.tsv", "w") as f_out:
            f_out.write("Isolate_Id\t" + header + "\n")
            for seq_id in reference_genomes_per_clade[clade]:
                f_out.write(f"{seq_id}\t{metadata_dict[seq_id]}\n")
        
        # Write Train sets
        if clade in train_genomes_per_clade:
            SeqIO.write([genomes_separated_HA[seq_id] for seq_id in train_genomes_per_clade[clade]],
                        f"{reference_dir}/Train/{clade}/sequences_HA.fasta", "fasta")
            SeqIO.write([genomes_separated_NA[seq_id] for seq_id in train_genomes_per_clade[clade]],
                        f"{reference_dir}/Train/{clade}/sequences_NA.fasta", "fasta")
            SeqIO.write([genomes_concatenated[seq_id] for seq_id in train_genomes_per_clade[clade]],
                        f"{reference_dir}/Train/{clade}/sequences_concatenated.fasta", "fasta")
            SeqIO.write([genomes_concatenated_ns[seq_id] for seq_id in train_genomes_per_clade[clade]],
                        f"{reference_dir}/Train/{clade}/sequences_concatenated_ns.fasta", "fasta")
            with open(f"{reference_dir}/Train/{clade}/metadata.tsv", "w") as f_out:
                f_out.write("Isolate_Id\t" + header + "\n")
                for seq_id in train_genomes_per_clade[clade]:
                    f_out.write(f"{seq_id}\t{metadata_dict[seq_id]}\n")
            
    # Write overall Reference set
    SeqIO.write([genomes_separated_HA[seq_id] for seq_id in reference_genomes],
                f"{reference_dir}/sequences_HA.fasta", "fasta")
    SeqIO.write([genomes_separated_NA[seq_id] for seq_id in reference_genomes],
                f"{reference_dir}/sequences_NA.fasta", "fasta")
    SeqIO.write([genomes_concatenated[seq_id] for seq_id in reference_genomes],
                f"{reference_dir}/sequences_concatenated.fasta", "fasta")
    SeqIO.write([genomes_concatenated_ns[seq_id] for seq_id in reference_genomes],
                f"{reference_dir}/sequences_concatenated_ns.fasta", "fasta")
    with open(f"{reference_dir}/metadata.tsv", "w") as f_out:
        f_out.write("Isolate_Id\t" + header + "\n")
        for seq_id in reference_genomes:
            f_out.write(f"{seq_id}\t{metadata_dict[seq_id]}\n")
    
    # Write overall Train set
    SeqIO.write([genomes_separated_HA[seq_id] for seq_id in train_genomes],
                f"{reference_dir}/Train/sequences_HA.fasta", "fasta")
    SeqIO.write([genomes_separated_NA[seq_id] for seq_id in train_genomes],
                f"{reference_dir}/Train/sequences_NA.fasta", "fasta")
    SeqIO.write([genomes_concatenated[seq_id] for seq_id in train_genomes],
                f"{reference_dir}/Train/sequences_concatenated.fasta", "fasta")
    SeqIO.write([genomes_concatenated_ns[seq_id] for seq_id in train_genomes],
                f"{reference_dir}/Train/sequences_concatenated_ns.fasta", "fasta")
    with open(f"{reference_dir}/Train/metadata.tsv", "w") as f_out:
        f_out.write("Isolate_Id\t" + header + "\n")
        for seq_id in train_genomes:
            f_out.write(f"{seq_id}\t{metadata_dict[seq_id]}\n")

    # Write overall Simulation set
    SeqIO.write([genomes_separated_HA[seq_id] for seq_id in simulation_genomes],
                f"{simulation_dir}/sequences_HA.fasta", "fasta")
    SeqIO.write([genomes_separated_NA[seq_id] for seq_id in simulation_genomes],
                f"{simulation_dir}/sequences_NA.fasta", "fasta")
    SeqIO.write([genomes_concatenated[seq_id] for seq_id in simulation_genomes],
                f"{simulation_dir}/sequences_concatenated.fasta", "fasta")
    SeqIO.write([genomes_concatenated_ns[seq_id] for seq_id in simulation_genomes],
                f"{simulation_dir}/sequences_concatenated_ns.fasta", "fasta")
    with open(f"{simulation_dir}/metadata.tsv", "w") as f_out:
        f_out.write("Isolate_Id\t" + header + "\n")
        for seq_id in simulation_genomes:
            f_out.write(f"{seq_id}\t{metadata_dict[seq_id]}\n")
    
    # Write overall Test set
    SeqIO.write([genomes_separated_HA[seq_id] for seq_id in test_genomes],
                f"{reference_dir}/Test/sequences_HA.fasta", "fasta")
    SeqIO.write([genomes_separated_NA[seq_id] for seq_id in test_genomes],
                f"{reference_dir}/Test/sequences_NA.fasta", "fasta")
    SeqIO.write([genomes_concatenated[seq_id] for seq_id in test_genomes],
                f"{reference_dir}/Test/sequences_concatenated.fasta", "fasta")
    SeqIO.write([genomes_concatenated_ns[seq_id] for seq_id in test_genomes],
                f"{reference_dir}/Test/sequences_concatenated_ns.fasta", "fasta")
    with open(f"{reference_dir}/Test/metadata.tsv", "w") as f_out:
        f_out.write("Isolate_Id\t" + header + "\n")
        for seq_id in test_genomes:
            f_out.write(f"{seq_id}\t{metadata_dict[seq_id]}\n")


if __name__ == "__main__":
    main()