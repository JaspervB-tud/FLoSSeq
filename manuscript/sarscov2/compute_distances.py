import argparse
import multiprocessing as mp
import itertools

def parse_msa(path):
    sequences = {}
    with open(path, "r") as f_in:
        for line in f_in:
            if line.startswith(">"):
                seq_id = line[1:].strip() #remove '>'
                sequences[seq_id] = ""
            else:
                sequences[seq_id] += line.strip()

    return sequences

"""
This is for multiprocessing. Each worker will process some distance pairs.
"""
def init_worker(sequences):
    global worker_sequences
    worker_sequences = sequences

def worker(pair):
    id1, id2 = pair
    seq1 = worker_sequences[id1]
    seq2 = worker_sequences[id2]
    d = 0
    tot = 0
    for char_idx in range(len(seq1)): #MSA -> same length
        # Only consider positions where both sequences have a non-gap and non-N character
        if (seq1[char_idx] == "-" and seq2[char_idx] == "-") or (seq1[char_idx] == "n" or seq2[char_idx] == "n"):
            continue
        tot += 1
        if seq1[char_idx] != seq2[char_idx]:
            d += 1

    return (id1, id2, d / tot) if tot > 0 else (id1, id2, 1.0) # this returns the distance, not similarity

def main():
    parser = argparse.ArgumentParser(descripion="Calculate pairwise distances based on MSA.")
    parser.add_argument("--msa", required=True, help="Path to the MSA FASTA file.")
    parser.add_argument("--output", required=True, help="Path ot the output distance file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes.")
    args = parser.parse_args()

    sequences = parse_msa(args.msa)
    seq_ids = list(sequences.keys())

    with mp.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(sequences,)
    ) as pool:
        pairs = itertools.combinations(seq_ids, 2)
        results = pool.imap_unordered(worker, pairs, chunksize=2_000)

        with open(args.output, "w") as f_out:
            f_out.write("SeqID1\tSeqID2\tDistance\n")
            for id1, id2, dist in results:
                f_out.write(f"{id1}\t{id2}\t{dist:.8f}\n")

if __name__ == "__main__":
    main()