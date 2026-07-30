import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.metrics import f1_score

# ==============
# Configuration
# ==============
DATA_ROOT = "data/SARS-CoV-2"
OUT_DIR = "results/SARS-CoV-2/figures/heatmaps"
os.makedirs(OUT_DIR, exist_ok=True)

LOCATIONS = ["China", "USA"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]
SAMPLES = [f"{i:02d}" for i in range(1, 21)]  # 20 samples per difficulty
MIN_ABUNDANCE = 0.001

ID_IDX = 0
LENGTH_IDX = 8
LINEAGE_IDX = 13
KALLISTO_ID_IDX = 0
KALLISTO_TPM_IDX = 4

SCALES = ["0.00000", "0.00001", "0.00010", "0.00100", "0.01000", "0.10000", "1.00000"]
COSTS = ["0.000000", "0.000001", "0.000010", "0.000100", "0.001000", "0.010000", "0.100000", "1.000000"]

METRICS = [
    ("MASH_s5000", "mash_s5000"),
    ("MASH_s10000", "mash_s10000"),
    ("SOURMASH_Jaccard_s6", "sourmash-jaccard_s6"),
    ("SOURMASH_Jaccard_s3", "sourmash-jaccard_s3"),
    ("SOURMASH_Cosine_s6", "sourmash-cosine_s6"),
    ("SOURMASH_Cosine_s3", "sourmash-cosine_s3"),
]

TICK_FONTSIZE = 10
LABEL_FONTSIZE = 16
CELL_FONTSIZE = 10
CELL_SIZE = 0.5
FIG_WIDTH = len(SCALES) * CELL_SIZE + 2
FIG_HEIGHT = len(COSTS) * CELL_SIZE + 2


# ==============
# Path builders
# ==============
def metadata_train_path(location):
    return f"{DATA_ROOT}/{location}/Reference/Train/metadata.tsv"


def metadata_test_path(location):
    return f"{DATA_ROOT}/{location}/Reference/Test/metadata.tsv"


def difficulty_reads_dir(location, difficulty):
    return f"{DATA_ROOT}/{location}/Reference/Test/{difficulty}"


def estimations_dir(location):
    return f"{DATA_ROOT}/{location}/Reference/Test/estimations"


def reset_method_name(metric_frag, cost, scale):
    return f"reset_{metric_frag}_cost-{cost}_scale-{scale}"


def abundance_path(location, method, difficulty, sample):
    return f"{estimations_dir(location)}/{method}/{difficulty}_{sample}/abundance.tsv"


# ==============================
# Metadata / read-level parsing
# ==============================
def read_metadata(path):
    metadata = {}
    with open(path, "r") as f_in:
        next(f_in)  # skip header
        for line in f_in:
            fields = line.strip().split("\t")
            seq_id = fields[ID_IDX]
            metadata[seq_id] = {
                "length": int(fields[LENGTH_IDX]),
                "lineage": fields[LINEAGE_IDX],
            }
    return metadata


def _accumulate_fastq_counts(path, metadata, lin2idx, abundances, sample_idx):
    with open(path, "r") as f_in:
        seq_id = None
        lin_idx = None
        for idx, line in enumerate(f_in):
            if idx % 4 == 0:  # header
                seq_id = line.strip()[1:]
                seq_id = "-".join(seq_id.split("-")[:-1])  # strip read number
                try:
                    lin_idx = lin2idx[metadata[seq_id]["lineage"]]
                except KeyError:
                    raise KeyError(
                        f"Sequence '{seq_id}' from {path} not found in metadata or its lineage not in lin2idx"
                    )
            elif idx % 4 == 1:  # sequence
                abundances[sample_idx, lin_idx] += len(line.strip()) / metadata[seq_id]["length"]


def _finalize_abundances(abundances, min_abundance):
    total = np.sum(abundances, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        abundances = np.where(total > 0, abundances / total, 0.0)
    abundances[abundances < min_abundance] = 0.0
    total2 = np.sum(abundances, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        abundances = np.where(total2 > 0, abundances / total2, 0.0)
    return abundances


def _accumulate_kallisto_counts(path, metadata, lin2idx, abundances, sample_idx):
    with open(path, "r") as f_in:
        next(f_in)  # header
        for line in f_in:
            fields = line.strip().split("\t")
            seq_id = fields[KALLISTO_ID_IDX]
            try:
                lin_idx = lin2idx[metadata[seq_id]["lineage"]]
            except KeyError:
                raise KeyError(
                    f"Sequence '{seq_id}' from {path} not found in metadata or its lineage not in lin2idx."
                )
            tpm = float(fields[KALLISTO_TPM_IDX])
            abundances[sample_idx, lin_idx] += tpm


def read_groundtruth(location, difficulty, metadata, lin2idx):
    abundances = np.zeros((len(SAMPLES), len(lin2idx)), dtype=np.float64)
    reads_dir = difficulty_reads_dir(location, difficulty)
    for s_idx, sample in enumerate(SAMPLES):
        for part in ("R1", "R2"):
            fq_path = f"{reads_dir}/sample_{sample}_{part}.fastq"
            _accumulate_fastq_counts(fq_path, metadata, lin2idx, abundances, s_idx)
    return _finalize_abundances(abundances, MIN_ABUNDANCE)


def read_kallisto(location, method, difficulty, metadata, lin2idx):
    abundances = np.zeros((len(SAMPLES), len(lin2idx)), dtype=np.float64)
    for s_idx, sample in enumerate(SAMPLES):
        path = abundance_path(location, method, difficulty, sample)
        _accumulate_kallisto_counts(path, metadata, lin2idx, abundances, s_idx)
    return _finalize_abundances(abundances, MIN_ABUNDANCE)


# ================
# Display helpers
# ================
def convert_to_logscale(s):
    val = float(s)
    if val == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(val))))
    coefficient = val / 10 ** exponent
    if coefficient == 1.0:
        return f"10^{exponent}"
    return f"{coefficient:.2f} x 10^{exponent}"


def plot_heatmap(data, cost_labels, scale_labels, output_path):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.imshow(data, aspect="auto", cmap="Greys", vmin=0, vmax=1)

    ax.set_xticks(range(len(scale_labels)))
    ax.set_xticklabels(scale_labels, rotation=0, ha="center", fontsize=TICK_FONTSIZE)
    ax.set_yticks(range(len(cost_labels)))
    ax.set_yticklabels(cost_labels, fontsize=TICK_FONTSIZE)

    ax.set_xticks(np.arange(-0.5, len(scale_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cost_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    for i in range(len(cost_labels)):
        for j in range(len(scale_labels)):
            text_color = "white" if data[i, j] > 0.5 else "black"
            ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center",
                     color=text_color, fontsize=CELL_FONTSIZE)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_legend(output_path):
    fig = plt.figure(figsize=(4, 11.69))  # narrow, full A4 height
    cbar_ax = fig.add_axes([0.35, 0.08, 0.15, 0.87])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap="Greys", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Median score across samples", fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    fig.text(0.5, 0.02, "Scale (lambda)", ha="center", fontsize=LABEL_FONTSIZE)
    fig.text(0.02, 0.5, "Cost (c)", ha="center", va="center", rotation="vertical", fontsize=LABEL_FONTSIZE)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)





def main():
    for location in LOCATIONS:
        metadata_train = read_metadata(metadata_train_path(location))
        metadata_test = read_metadata(metadata_test_path(location))

        lineages = sorted(set(metadata_train[s]["lineage"] for s in metadata_train))
        lin2idx = {lineage: idx for idx, lineage in enumerate(lineages)}

        for difficulty in DIFFICULTIES:
            groundtruth = read_groundtruth(location, difficulty, metadata_test, lin2idx)
            groundtruth_binary = (groundtruth > 0).astype(int)

            for metric_display, metric_frag in METRICS:
                l1_data = np.zeros((len(COSTS), len(SCALES)), dtype=np.float64)
                f1_data = np.zeros((len(COSTS), len(SCALES)), dtype=np.float64)

                for cost_idx, cost in enumerate(COSTS):
                    for scale_idx, scale in enumerate(SCALES):
                        method = reset_method_name(metric_frag, cost, scale)
                        estimation = read_kallisto(location, method, difficulty, metadata_train, lin2idx)
                        estimation_binary = (estimation > 0).astype(int)

                        l1_errors = np.sum(np.abs(estimation - groundtruth), axis=1)
                        l1_scores = 1.0 - (l1_errors / 2.0)
                        f1_scores = np.array([f1_score(groundtruth_binary[i], estimation_binary[i])
                                              for i in range(len(SAMPLES))])

                        l1_data[cost_idx, scale_idx] = np.median(l1_scores)
                        f1_data[cost_idx, scale_idx] = np.median(f1_scores)

                scale_labels = [convert_to_logscale(s) for s in SCALES]
                cost_labels = [convert_to_logscale(c) for c in COSTS]

                # Reverse cost order so lowest cost sits at the bottom of the plot.
                l1_data = l1_data[::-1, :]
                f1_data = f1_data[::-1, :]
                cost_labels = cost_labels[::-1]

                plot_heatmap(l1_data, cost_labels, scale_labels,
                             f"{OUT_DIR}/{location}_{difficulty}_{metric_display}_L1_heatmap.svg")
                plot_heatmap(f1_data, cost_labels, scale_labels,
                             f"{OUT_DIR}/{location}_{difficulty}_{metric_display}_F1_heatmap.svg")

    plot_legend(f"{OUT_DIR}/legend_heatmap.svg")


if __name__ == "__main__":
    main()