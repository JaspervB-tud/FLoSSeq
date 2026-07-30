from multiprocessing import Pool, Queue, Process
import os
import argparse
from Bio import SeqIO
from functools import partial

def read_metadata(path, min_date="1990-01-01", max_date="2100-12-31", allowed_location=None, min_length=29000, max_n_content=0.01):
    """
    Reads metadata from the given path, and filters entries based on collection date, location and quality criteria.

    Parameters:
    -----------
    path : str
        Path to the metadata file (tab-separated from GISAID).
    min_date : str
        Minimum collection date (inclusive) in YYYY-MM-DD format.
    max_date : str
        Maximum collection date (inclusive) in YYYY-MM-DD format.
    allowed_location : str or None
        If specified, only sequences from this country will be included (e.g., "USA" or "China"). If None, no location filtering is applied.
    min_length : int
        Minimum sequence length (in nucleotides) to include.
    max_n_content : float
        Maximum allowed N content (as a fraction) in the sequence to include.

    Returns:
    --------
    metadata : dict
        A dictionary mapping sequence IDs to their metadata fields (as lists).
    header : str
        The header line from the metadata file (without the newline character).
    """
    metadata = {}
    with open(path, "r") as f_in:
        header = next(f_in).rstrip("\n")
        for line in f_in:
            fields = line.rstrip("\n").split("\t")
            seq_id = fields[0].replace(" ", "_")

            # Collection date
            collection_date = fields[5].strip()
            parts = collection_date.split("-")
            if len(parts) == 1:
                continue
            elif len(parts) == 2:
                year = parts[0]
                month = parts[1].zfill(2)
                collection_date = f"{year}-{month}-01"
            elif len(parts) == 3:
                year = parts[0]
                month = parts[1].zfill(2)
                day = parts[2].zfill(2)
                collection_date = f"{year}-{month}-{day}"
            else:
                continue
            try:
                if not (min_date <= collection_date <= max_date):
                    continue
            except:
                continue

            # Location
            if allowed_location is not None:
                location = fields[6].strip()
                split_loc = location.split(" / ")
                country = split_loc[1] if len(split_loc) > 1 else ""
                if country != allowed_location:
                    continue

            # Sequence length
            seq_length = int(fields[8].strip())
            if seq_length < min_length:
                continue

            # Host
            host = fields[9].strip().lower()
            if host != "human":
                continue

            # Completeness
            completeness = fields[19].strip().lower()
            if completeness != "true":
                continue

            # N content
            try:
                n_content = float(fields[22].strip())
            except ValueError:
                continue
            if n_content > max_n_content:
                continue

            metadata[seq_id] = fields
    return metadata, header


"""
Workers for multiprocessing.

These functions and variables are used by the worker processes to filter sequences based on metadata.
"""
worker_metadata = None

def init_worker(metadata):
    """
    Initializes the worker process by setting the global worker_metadata variable to the provided metadata dictionary.

    Parameters:
    -----------
    metadata : dict
        A dictionary mapping sequence IDs to their metadata fields (as lists). This will be used by the worker function to filter sequences based on their metadata.
    """
    global worker_metadata
    worker_metadata = metadata


def worker(batch, max_non_acgt_content):
    """
    Worker function that processes a batch of sequences, filtering them based on the metadata and non-ACGT content.

    Parameters:
    -----------
    batch : list of SeqRecord
        A list of SeqRecord objects representing the sequences in the current batch.
    max_non_acgt_content : float
        Maximum allowed non-ACGT content (as a fraction) in the sequence to include.

    Returns:
    --------
    accepted_seqs : list of SeqRecord
        A list of SeqRecord objects that passed the filtering criteria.
    accepted_meta_lines : list of str
        A list of metadata lines (as strings) corresponding to the accepted sequences, ready to be written to the output metadata file.
    """
    accepted_seqs = []
    accepted_meta_lines = []

    for record in batch:
        meta = worker_metadata.get(record.description.split("|")[0].strip())
        if meta is None:
            continue

        seq = str(record.seq).lower()
        non_acgt = sum(1 for base in seq if base not in ("a", "c", "g", "t"))
        non_acgt_content = non_acgt / len(seq)
        if non_acgt_content > max_non_acgt_content:
            continue
        else:
            accepted_seqs.append(record)
            accepted_meta_lines.append("\t".join(meta) + "\n")

    return (accepted_seqs, accepted_meta_lines)

def task(batch, max_non_acgt_content):
    return worker(batch, max_non_acgt_content)


def writer_process(fasta_path, meta_path, queue):
    """
    Writer process that listens to a multiprocessing queue for batches of accepted sequences and their corresponding metadata lines, and writes them to the output FASTA and metadata files.

    Parameters:
    -----------
    fasta_path : str
        Path to the output FASTA file where accepted sequences will be written.
    meta_path : str
        Path to the output metadata file where corresponding metadata lines will be appended.
    queue : multiprocessing.Queue
        A multiprocessing queue that the worker processes will put their results into. Each item in the queue should be a tuple of 
        (accepted_seqs, accepted_meta_lines) where accepted_seqs is a list of SeqRecord objects and accepted_meta_lines is a list 
        of strings corresponding to the metadata lines for those sequences. The writer process will keep listening to the queue 
        until it receives a None value, which signals it to stop.
    """
    count = 0
    with open(fasta_path, "w") as fasta_out, open(meta_path, "a") as meta_out:
        while True:
            batch = queue.get()
            if batch is None:
                break

            seqs, metalines = batch

            for record in seqs:
                record.id = record.description.split("|")[0].strip().replace(" ", "_")
                record.name = record.id
                record.description = ""
                SeqIO.write(record, fasta_out, "fasta")
                count += 1
                if count % 500 == 0:
                    print(f"Wrote {count} sequences...", flush=True)

            for line in metalines:
                meta_out.write(line)

def stream_batches(fasta_path, batch_size=5000):
    """
    Generator function that reads sequences from a FASTA file in batches and yields them as lists of SeqRecord objects.

    Parameters:
    -----------
    fasta_path : str
        Path to the input FASTA file containing the sequences to be processed.
    batch_size : int
        The number of sequences to include in each batch. The generator will read this many sequences at a time and yield them as a list.
        The last batch may contain fewer than batch_size sequences if the total number of sequences in the FASTA file is not a multiple of batch_size.
    """
    batch = []
    with open(fasta_path, "r") as f_in:
        for record in SeqIO.parse(f_in, "fasta"):
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Filter SARS-CoV-2 genomes based on metadata.")
    parser.add_argument("--input_fasta", required=True, help="Fasta file containing SARS-CoV-2 genomes to filter")
    parser.add_argument("--input_metadata", required=True, help="Metadata file corresponding to the fasta file")
    parser.add_argument("--output_folder", required=True, help="Output folder to write filtered fasta and metadata")

    parser.add_argument("--ref_start", required=True, help="Start date for the reference sequences (YYYY-MM-DD)")
    parser.add_argument("--ref_end", required=True, help="End date for the reference sequences (YYYY-MM-DD)")
    parser.add_argument("--sim_start", required=True, help="Start date for the simulation sequences (YYYY-MM-DD)")
    parser.add_argument("--sim_end", required=True, help="End date for the simulation sequences (YYYY-MM-DD)")

    parser.add_argument("--location", required=True, type=str, help="Country to filter sequences by (e.g. USA or China)")

    parser.add_argument("--min_length", type=int, default=29000, help="Minimum sequence length to include")
    parser.add_argument("--max_n_content", type=float, default=0.01, help="Maximum allowed N content in sequences")
    parser.add_argument("--max_non_acgt_content", type=float, default=1.0, help="Maximum allowed non-ACGT content in sequences")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes to use for filtering")
    args = parser.parse_args()

    # Define output subdirectories
    ref_dir = os.path.join(args.output_folder, "Reference")
    sim_dir = os.path.join(args.output_folder, "Simulation")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(sim_dir, exist_ok=True)

    # Run filtering for reference and simulation periods
    periods = [
        ("Reference", args.ref_start, args.ref_end, ref_dir),
        ("Simulation", args.sim_start, args.sim_end, sim_dir),
    ]

    for period_name, min_date, max_date, out_dir in periods:
        metadata, meta_header = read_metadata(
            args.input_metadata,
            min_date=min_date,
            max_date=max_date,
            allowed_location=args.location,
            min_length=args.min_length,
            max_n_content=args.max_n_content,
        )

        output_fasta = os.path.join(out_dir, "sequences.fasta")
        output_meta = os.path.join(out_dir, "metadata.tsv")

        # Write metadata header
        with open(output_meta, "w") as f:
            f.write(meta_header + "\n")

        # Queue for writer
        queue = Queue(maxsize=50)

        # Start writer process
        writer = Process(target=writer_process, args=(output_fasta, output_meta, queue))
        writer.start()

        # Worker pool
        with Pool(
            processes=args.num_workers,
            initializer=init_worker,
            initargs=(metadata,)
        ) as pool:
            worker_function = partial(task, max_non_acgt_content=args.max_non_acgt_content)
            for results in pool.imap_unordered(worker_function, stream_batches(args.input_fasta)):
                queue.put(results)

        # Stop writer
        queue.put(None)
        writer.join()

if __name__ == "__main__":
    main()