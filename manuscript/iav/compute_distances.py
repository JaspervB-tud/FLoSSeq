import argparse
import multiprocessing as mp
import itertools

def parse_msa(path):
    sequences = {}
    with open(path, "r") as f_in:
        for line in f_in:
            if line.startswith(">"):
                seq_id = line[1:].strip()
                sequences[seq_id] = ""
            else:
                sequences[seq_id] += line.strip()

    return sequences

"""
This is for multiprocessing. Each worker will process some distance pairs.
"""
def init_worker(sequences_HA, sequences_NA):
    global worker_sequences_HA, worker_sequences_NA
    worker_sequences_HA = sequences_HA
    worker_sequences_NA = sequences_NA

def worker(pair):
    id1, id2 = pair
    # First process HA
    seq1 = worker_sequences_HA[id1]
    seq2 = worker_sequences_HA[id2]
    d_HA = 0
    tot_HA = 0
    for char_idx in range(len(seq1)): #MSA -> same length
        if (seq1[char_idx] == "-" and seq2[char_idx] == "-") or (seq1[char_idx] == "n" or seq2[char_idx] == "n"): # require both to be non-gap and non-N
            continue
        tot_HA += 1
        if seq1[char_idx] != seq2[char_idx]:
            d_HA += 1
    # Then process NA
    seq1 = worker_sequences_NA[id1]
    seq2 = worker_sequences_NA[id2]
    d_NA = 0
    tot_NA = 0
    for char_idx in range(len(seq1)): #MSA -> same length
        if (seq1[char_idx] == "-" and seq2[char_idx] == "-") or (seq1[char_idx] == "n" or seq2[char_idx] == "n"): # require both to be non-gap and non-N
            continue
        tot_NA += 1
        if seq1[char_idx] != seq2[char_idx]:
            d_NA += 1
    # Combine results
    d_total = d_HA + d_NA
    tot_total = tot_HA + tot_NA

    return (id1, id2, d_total/tot_total) if tot_total > 0 else (id1, id2, 1.0)

def main():
    parser = argparse.ArgumentParser(descripion="Calculate pairwise distances based on MSA.")
    parser.add_argument("--msa_HA", required=True, help="Path to the MSA FASTA file for HA segments.")
    parser.add_argument("--msa_NA", required=True, help="Path to the MSA FASTA file for NA segments.")
    parser.add_argument("--output", required=True, help="Path ot the output distance file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes.")
    args = parser.parse_args()

    sequences_HA = parse_msa(args.msa_HA)
    sequences_NA = parse_msa(args.msa_NA)
    seq_ids = sorted(list(sequences_HA.keys())) # we use the aggregate assembly sequence IDs, which are the same for HA and NA

    with mp.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(sequences_HA, sequences_NA)
    ) as pool:
        pairs = itertools.combinations(seq_ids, 2)
        results = pool.imap_unordered(worker, pairs, chunksize=2_000)

        with open(args.output, "w") as f_out:
            f_out.write("SeqID1\tSeqID2\tDistance\n")
            for id1, id2, dist in results:
                f_out.write(f"{id1}\t{id2}\t{dist:.6f}\n")

if __name__ == "__main__":
    main()