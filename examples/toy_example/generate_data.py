import numpy as np
from scipy.spatial.distance import cdist

SEED = 42
rng  = np.random.RandomState(SEED)

"""
Define taxa and per-sequence coordinates to construct a synthetic dataset. The taxa are placed in
a 2D space for simplicity, and sequences are generated around centres with some Gaussian noise.
"""
TAXON_CENTRES = {
    "Alpha": np.array([0.00, 0.00]),
    "Beta":  np.array([0.30, 0.00]),
    "Gamma": np.array([0.15, 0.26]),
    "Delta": np.array([0.15, 0.09]),   # inside the triangle formed by the other taxa to create ambiguity
}

TAXON_COUNTS = {
    "Alpha": 10,
    "Beta":   8,
    "Gamma":  9,
    "Delta":  7,
}

SPREAD = 0.012
OUTLIER_FRACTION = 0.15   # specify a fraction of outliers that lie at 3x the normal spread from the centre

# List of sequences (seq_id, taxon) and their coordinates
sequences = []
coords    = []

for taxon in TAXON_CENTRES:
    centre = TAXON_CENTRES[taxon]
    n = TAXON_COUNTS[taxon]
    n_outliers = max(1, int(n * OUTLIER_FRACTION))
    for idx in range(1, n + 1):
        seq_id = f"{taxon}_{idx:03d}" # set name
        spread = SPREAD * 3 if idx <= n_outliers else SPREAD # determine spread for outliers (first few sequences)
        jitter = rng.randn(2) * spread # induce Gaussian noise around centre
        sequences.append((seq_id, taxon))
        coords.append(centre + jitter)

coords = np.array(coords)
n_seqs = len(sequences)

# Compute pairwise (Euclidean) distances between sequences
NORMALISER = 0.50 # this is a fixed normaliser to keep the values "interpretable" across runs
dist_matrix = np.zeros((n_seqs, n_seqs))
for idx_1 in range(n_seqs):
    for idx_2 in range(idx_1):
        dist = np.sqrt(np.sum((coords[idx_1] - coords[idx_2]) ** 2))
        dist = np.clip(dist / NORMALISER, 0.0, 1.0) # scale to [0, 1] using the fixed normaliser
        dist_matrix[idx_1, idx_2] = dist
        dist_matrix[idx_2, idx_1] = dist

# Write output
with open("clusters.csv", "w") as f_out:
    for seq_id, taxon in sequences:
        f_out.write(f"{seq_id},{taxon}\n")

# Write distances in MASH triangle format
with open("distances.tsv", "w") as f_out:
    f_out.write(f"{n_seqs}\n")
    for idx_1 in range(n_seqs):
        row = [str(idx_1)] + [f"{dist_matrix[idx_1, idx_2]:.6f}" for idx_2 in range(idx_1)]
        f_out.write("\t".join(row) + "\n")

"""
# Print summary of distances (commented since it is unnecessary for the example)
print("\nDistance summary (mean ± std):")
taxon_indices = {}
offset = 0
for taxon in TAXON_CENTRES:
    centre = TAXON_CENTRES[taxon]
    n = TAXON_COUNTS[taxon]
    taxon_indices[taxon] = list(range(offset, offset + n))
    offset += n

taxa = list(TAXON_CENTRES.keys())
for idx, tax_1 in enumerate(taxa):
    for tax_2 in taxa[idx:]:
        idx_1 = taxon_indices[tax_1]
        idx_2 = taxon_indices[tax_2]
        sub  = dist_matrix[np.ix_(idx_1, idx_2)]
        if tax_1 == tax_2:
            # exclude diagonal (self-distances)
            mask = ~np.eye(len(idx1), dtype=bool)
            vals = sub[mask]
        else:
            vals = sub.flatten()
        print(f"  {t1:6s} – {t2:6s}: {vals.mean():.4f} ± {vals.std():.4f}")
"""