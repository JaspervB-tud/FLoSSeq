import numpy as np
import argparse
import os

def read_MASH(path):
    names = []
    name2idx = {}

    with open(path, "r") as f_in:
        header = next(f_in).strip()
        num_sequences = int(header) #MASH stores number of sequences in first line
        D = np.zeros((num_sequences, num_sequences), dtype=np.float64) #Storing only lower triangle

        for i, line in enumerate(f_in):
            line = line.strip().split("\t")
            names.append(line[0])
            for j in range(1, len(line)):
                D[i,j-1] = float(line[j])
                D[j-1,i] = D[i,j-1] #symmetric matrix

    return D, names

def read_SOURMASH(path):
    QUERY_NAME_POS=0 #index for query name in sourmash output
    MATCH_NAME_POS=2 #index for subject name in sourmash output
    JACCARD_POS=6 #index for jaccard in sourmash output
    COSINE_POS=12 #index for cosine in sourmash output

    names = []
    name2idx = {}

    jaccard_dict = {}
    cosine_dict = {}

    with open(path, "r") as f_in:
        next(f_in) #skip header
        for line in f_in:
            fields = line.strip().split(",")

            name_1 = fields[QUERY_NAME_POS]
            name_2 = fields[MATCH_NAME_POS]

            jaccard = float(fields[JACCARD_POS])
            cosine = float(fields[COSINE_POS])

            if name_1 not in name2idx:
                name2idx[name_1] = len(names)
                names.append(name_1)
            if name_2 not in name2idx:
                name2idx[name_2] = len(names)
                names.append(name_2)

            idx_1 = name2idx[name_1]
            idx_2 = name2idx[name_2]

            jaccard_dict[(idx_1, idx_2)] = jaccard
            jaccard_dict[(idx_2, idx_1)] = jaccard

            cosine_dict[(idx_1, idx_2)] = cosine
            cosine_dict[(idx_2, idx_1)] = cosine

    num_sequences = len(names)
    D_jaccard = np.zeros((num_sequences, num_sequences), dtype=np.float64)
    D_cosine = np.zeros((num_sequences, num_sequences), dtype=np.float64)

    for i in range(num_sequences):
        for j in range(i, num_sequences):
            if i == j:
                D_jaccard[i, j] = 0.0
                D_cosine[i, j] = 0.0
            else:
                D_jaccard[i, j] = 1.0 - jaccard_dict.get((i, j), 0.0) #convert to distance
                D_jaccard[j, i] = D_jaccard[i, j]

                D_cosine[i, j] = 1.0 - cosine_dict.get((i, j), 0.0) #convert to distance
                D_cosine[j, i] = D_cosine[i, j]

    return D_jaccard, D_cosine, names

def main():
    parser = argparse.ArgumentParser(description="Compute medoid sequence.")
    parser.add_argument("--input_path", required=True, help="Path to the file containing the distance information")
    parser.add_argument("--output_path", required=True, help="Path to the file where the medoid sequence index will be stored")
    parser.add_argument("--sourmash_jaccard", action="store_true", help="Indicates that the input file is a sourmash output containing Jaccard similarities")
    parser.add_argument("--sourmash_cosine", action="store_true", help="Indicates that the input file is a sourmash output containing Cosine similarities")
    args = parser.parse_args()

    if args.sourmash_jaccard or args.sourmash_cosine:
        D_jaccard, D_cosine, names = read_SOURMASH(args.input_path)
        D = D_jaccard if args.sourmash_jaccard else D_cosine
    else:        
        D, names = read_MASH(args.input_path)

    medoid = np.argmin(np.sum(D, axis=1))

    with open(args.output_path, "w") as f_out:
        f_out.write(f"{names[medoid]}\n")

if __name__ == "__main__":
    main()