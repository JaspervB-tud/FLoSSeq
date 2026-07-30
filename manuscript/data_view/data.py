import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def read_metadata_sarscov2(path):
    ID_COL = 0
    LINEAGE_COL = 13

    seq2lin = {}
    with open(path, "r") as f_in:
        next(f_in)  # header
        for line in f_in:
            parts = line.strip().split("\t")
            seq2lin[parts[ID_COL]] = parts[LINEAGE_COL]
    return seq2lin

def read_metadata_iav(path):
    ID_COL = 0
    CLADE_COL = 15

    seq2clade = {}
    with open(path, "r") as f_in:
        next(f_in)  # header
        for line in f_in:
            parts = line.strip().split("\t")
            seq2clade[parts[ID_COL]] = parts[CLADE_COL]
    return seq2clade

def read_sequence_mapping(path):
    IDX_COL = 0
    SEQID_COL = 1

    sequence_mapping = {}
    with open(path, "r") as f_in:
        for line in f_in:
            parts = line.strip().split("\t")
            idx = int(parts[IDX_COL])
            seq_id = parts[SEQID_COL]
            sequence_mapping[idx] = seq_id
    return sequence_mapping

def calculate_characteristics(mash_path, seq2tax, sequence_mapping, name=None):
    """
    Stream the pairwise distances from the MASH output file and calculate
    characteristics on the fly.

    Parameters
    ----------
    mash_path : str
        Path to the MASH output file containing pairwise distances.
    seq2tax : dict
        Mapping from sequence IDs to their taxonomic classifications (lineages or clades).
    sequence_mapping : dict
        Mapping from MASH index to sequence IDs.
    name : str, optional
        Dataset name, for reporting purposes.
    """
    taxa = sorted(set(seq2tax.values()))
    tax2idx = {tax: idx for idx, tax in enumerate(taxa)}
    non_singleton_taxa = set()
    num_taxa = len(taxa)
    num_genomes_per_taxon = {}

    for sequence in seq2tax:
        taxon = seq2tax[sequence]
        num_genomes_per_taxon[taxon] = num_genomes_per_taxon.get(taxon, 0) + 1
        if num_genomes_per_taxon[taxon] > 1:
            non_singleton_taxa.add(taxon)

    singleton_taxa = set(taxa) - non_singleton_taxa
    num_singleton = len(singleton_taxa)

    within_taxon_distances = np.zeros((num_taxa, 3), dtype=np.float64)  # (avg, min, max)
    within_taxon_distances[:, 1] = np.inf
    within_taxon_counts = np.zeros(num_taxa, dtype=np.float64)
    between_taxon_distances = np.zeros((num_taxa, num_taxa, 3, 2), dtype=np.float64)  # (avg,min,max) x (with,without singletons)
    between_taxon_distances[:, :, 1, :] = np.inf
    between_taxon_counts = np.zeros((num_taxa, num_taxa, 2), dtype=np.float64)
    ambiguity_indices = np.full((len(sequence_mapping), 2), np.inf, dtype=np.float64)  # (nearest other, nearest same)

    indices = []
    with open(mash_path, "r") as f_in:
        next(f_in)  # header
        for line in f_in:
            parts = line.strip().split("\t")

            cur_idx = int(parts[0])
            cur_taxon = seq2tax[sequence_mapping[cur_idx]]
            taxon_idx_1 = tax2idx[cur_taxon]

            for other_idx, d in enumerate(parts[1:]):  # remainder: distances to previous indices
                other_idx = indices[other_idx]
                other_taxon = seq2tax[sequence_mapping[other_idx]]
                taxon_idx_2 = tax2idx[other_taxon]
                d = float(d)

                if taxon_idx_1 == taxon_idx_2:
                    within_taxon_distances[taxon_idx_1, 0] += d
                    within_taxon_distances[taxon_idx_1, 1] = min(within_taxon_distances[taxon_idx_1, 1], d)
                    within_taxon_distances[taxon_idx_1, 2] = max(within_taxon_distances[taxon_idx_1, 2], d)
                    within_taxon_counts[taxon_idx_1] += 1

                    ambiguity_indices[cur_idx, 1] = min(ambiguity_indices[cur_idx, 1], d)
                    ambiguity_indices[other_idx, 1] = min(ambiguity_indices[other_idx, 1], d)

                if taxon_idx_1 != taxon_idx_2:
                    between_taxon_distances[taxon_idx_1, taxon_idx_2, 0, 0] += d
                    between_taxon_distances[taxon_idx_1, taxon_idx_2, 1, 0] = min(
                        between_taxon_distances[taxon_idx_1, taxon_idx_2, 1, 0], d)
                    between_taxon_distances[taxon_idx_1, taxon_idx_2, 2, 0] = max(
                        between_taxon_distances[taxon_idx_1, taxon_idx_2, 2, 0], d)
                    between_taxon_counts[taxon_idx_1, taxon_idx_2, 0] += 1

                    between_taxon_distances[taxon_idx_2, taxon_idx_1, 0, 0] += d
                    between_taxon_distances[taxon_idx_2, taxon_idx_1, 1, 0] = min(
                        between_taxon_distances[taxon_idx_2, taxon_idx_1, 1, 0], d)
                    between_taxon_distances[taxon_idx_2, taxon_idx_1, 2, 0] = max(
                        between_taxon_distances[taxon_idx_2, taxon_idx_1, 2, 0], d)
                    between_taxon_counts[taxon_idx_2, taxon_idx_1, 0] += 1

                    ambiguity_indices[cur_idx, 0] = min(ambiguity_indices[cur_idx, 0], d)
                    ambiguity_indices[other_idx, 0] = min(ambiguity_indices[other_idx, 0], d)

                    if cur_taxon in non_singleton_taxa and other_taxon in non_singleton_taxa:
                        between_taxon_distances[taxon_idx_1, taxon_idx_2, 0, 1] += d
                        between_taxon_distances[taxon_idx_1, taxon_idx_2, 1, 1] = min(
                            between_taxon_distances[taxon_idx_1, taxon_idx_2, 1, 1], d)
                        between_taxon_distances[taxon_idx_1, taxon_idx_2, 2, 1] = max(
                            between_taxon_distances[taxon_idx_1, taxon_idx_2, 2, 1], d)
                        between_taxon_counts[taxon_idx_1, taxon_idx_2, 1] += 1

                        between_taxon_distances[taxon_idx_2, taxon_idx_1, 0, 1] += d
                        between_taxon_distances[taxon_idx_2, taxon_idx_1, 1, 1] = min(
                            between_taxon_distances[taxon_idx_2, taxon_idx_1, 1, 1], d)
                        between_taxon_distances[taxon_idx_2, taxon_idx_1, 2, 1] = max(
                            between_taxon_distances[taxon_idx_2, taxon_idx_1, 2, 1], d)
                        between_taxon_counts[taxon_idx_2, taxon_idx_1, 1] += 1

            indices.append(cur_idx)

    # Copy nearest distances before the ambiguity index calculation overwrites them.
    nearest_other = ambiguity_indices[:, 0].copy()
    nearest_same = ambiguity_indices[:, 1].copy()

    ambiguity_indices = np.divide(
        ambiguity_indices[:, 0],
        ambiguity_indices[:, 1],
        where=np.isfinite(ambiguity_indices[:, 1]) & (ambiguity_indices[:, 1] > 0),
        out=np.full(len(sequence_mapping), np.nan),
    )

    return {
        "taxa": taxa,
        "num_singleton": num_singleton,
        "within_taxon_distances": within_taxon_distances,
        "within_taxon_counts": within_taxon_counts,
        "between_taxon_distances": between_taxon_distances,
        "between_taxon_counts": between_taxon_counts,
        "ambiguity_indices": ambiguity_indices,
        "nearest_other": nearest_other,
        "nearest_same": nearest_same,
    }

def calculate_averages(characteristics, output_path):
    num_taxa = len(list(characteristics["taxa"]))
    upper = np.triu_indices(num_taxa, k=1)  # avoid double counting pairs and self-pairs

    ambiguity_indices = characteristics["ambiguity_indices"].copy()
    valid = np.isfinite(ambiguity_indices)
    ambiguity_ratio = np.mean(ambiguity_indices[valid] < 1.0)

    within_taxon_distances = characteristics["within_taxon_distances"].copy()
    within_taxon_counts = characteristics["within_taxon_counts"].copy()
    within_taxon_distances[:, 0] = np.divide(
        within_taxon_distances[:, 0], within_taxon_counts,
        where=within_taxon_counts > 0, out=np.full(num_taxa, np.nan),
    )
    mask = within_taxon_counts == 0
    within_taxon_distances[mask, 1] = np.nan
    within_taxon_distances[mask, 2] = np.nan

    between_taxon_distances = characteristics["between_taxon_distances"].copy()
    between_taxon_counts = characteristics["between_taxon_counts"].copy()
    between_taxon_distances[:, :, 0, 0] = np.divide(
        between_taxon_distances[:, :, 0, 0], between_taxon_counts[:, :, 0],
        where=between_taxon_counts[:, :, 0] > 0, out=np.full((num_taxa, num_taxa), np.nan),
    )
    between_taxon_distances[:, :, 0, 1] = np.divide(
        between_taxon_distances[:, :, 0, 1], between_taxon_counts[:, :, 1],
        where=between_taxon_counts[:, :, 1] > 0, out=np.full((num_taxa, num_taxa), np.nan),
    )
    mask = between_taxon_counts[:, :, 1] == 0
    between_taxon_distances[:, :, 1, 1][mask] = np.nan
    between_taxon_distances[:, :, 2, 1][mask] = np.nan

    with open(output_path, "w") as f_out:
        f_out.write(f"Number of taxa: {num_taxa}\n")
        f_out.write(f"Number of singleton taxa: {characteristics['num_singleton']}/{num_taxa} "
                    f"({(characteristics['num_singleton']/num_taxa)*100:.2f}%)\n\n")

        def write_block(title, values):
            cur_str = f"Min: {np.nanmin(values):.7f} -- Median: {np.nanmedian(values):.7f} -- Max: {np.nanmax(values):.7f}"
            f_out.write(title + "\n")
            f_out.write(cur_str + "\n")
            f_out.write("-" * len(cur_str) + "\n")

        f_out.write("Within-taxon distances\n")
        write_block("MINIMUM distance", within_taxon_distances[:, 1])
        write_block("AVERAGE distance", within_taxon_distances[:, 0])
        f_out.write("MAXIMUM distance\n")
        cur = within_taxon_distances[:, 2]
        f_out.write(f"Min: {np.nanmin(cur):.7f} -- Median: {np.nanmedian(cur):.7f} -- Max: {np.nanmax(cur):.7f}\n\n")

        f_out.write("Between-taxon distances (with singletons)\n")
        write_block("MINIMUM distance", between_taxon_distances[:, :, 1, 0][upper])
        write_block("AVERAGE distance", between_taxon_distances[:, :, 0, 0][upper])
        f_out.write("MAXIMUM distance\n")
        cur = between_taxon_distances[:, :, 2, 0][upper]
        f_out.write(f"Min: {np.nanmin(cur):.7f} -- Median: {np.nanmedian(cur):.7f} -- Max: {np.nanmax(cur):.7f}\n\n")

        f_out.write("Between-taxon distances (without singletons)\n")
        write_block("MINIMUM distance", between_taxon_distances[:, :, 1, 1][upper])
        write_block("AVERAGE distance", between_taxon_distances[:, :, 0, 1][upper])
        f_out.write("MAXIMUM distance\n")
        cur = between_taxon_distances[:, :, 2, 1][upper]
        f_out.write(f"Min: {np.nanmin(cur):.7f} -- Median: {np.nanmedian(cur):.7f} -- Max: {np.nanmax(cur):.7f}\n\n")

        f_out.write(f"Ambiguity index (proportion of genomes closer to foreign taxon than own): "
                    f"{ambiguity_ratio*100:.4f}%\n")

def plot_scatter(same, other, out_filepath):
    n_total = len(same)
    n_ambig = int((other < same).sum())
    ambiguity_ratio = n_ambig / n_total if n_total > 0 else 0.0

    is_ambig = other < same
    ambig_color = "#d85a30"
    unambig_color = "#9a9a9a"

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    marker_size = 5
    ax.scatter(
        same[~is_ambig], other[~is_ambig],
        s=marker_size, alpha=0.3, color=unambig_color, label=f"Unambiguous ({(~is_ambig).sum()})",
        zorder=1, rasterized=True,
    )
    ax.scatter(
        same[is_ambig], other[is_ambig],
        s=marker_size, alpha=0.3, color=ambig_color, label=f"Ambiguous ({is_ambig.sum()})",
        zorder=2, rasterized=True,
    )

    lims = [1e-6, 1]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, alpha=0.5, zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Distance to nearest same-taxon genome")
    ax.set_ylabel("Distance to nearest other-taxon genome")
    ax.set_aspect("equal")
    ax.grid(True, which="major", alpha=0.15)

    ax.legend(
        title=f"Ambiguity ratio: {ambiguity_ratio*100:.2f}%",
        loc="upper center", fontsize=10, framealpha=0.9, markerscale=2.5,
    )

    plt.tight_layout()
    plt.savefig(out_filepath, dpi=300)
    plt.close(fig)

def plot_histogram(aggregated_data, out_filepath):
    """
    Overlaid KDE of log2(d_other / d_same) per dataset.
    aggregated_data: list of (name, same, other) tuples with pre-filtered arrays.
    """
    palette_colors = ["#a6611a", "#dfc27d", "#80cdc1"]

    rows = []
    summary = {}
    for i, (name, same, other) in enumerate(aggregated_data):
        log_ratio = np.log2(other / same)
        n_total = len(log_ratio)
        n_ambig = int((log_ratio < 0).sum())
        ambig_frac = n_ambig / n_total if n_total > 0 else 0.0
        summary[name] = (ambig_frac, n_total, palette_colors[i % len(palette_colors)])
        for v in log_ratio:
            rows.append({"dataset": name, "log_ratio": v})

    df = pd.DataFrame(rows)

    df["label"] = df["dataset"].map(
        lambda n: f"{n} (ambiguity = {summary[n][0]*100:.2f}%, n = {summary[n][1]:,})"
    )
    label_order = [
        f"{n} (ambiguity = {summary[n][0]*100:.2f}%, n = {summary[n][1]:,})"
        for n, _, _ in aggregated_data
    ]
    palette = {
        label: summary[name][2]
        for label, (name, _, _) in zip(label_order, aggregated_data)
    }

    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)

    sns.kdeplot(
        data=df, x="log_ratio", hue="label", hue_order=label_order, palette=palette,
        fill=True, alpha=0.35, linewidth=1.5, common_norm=False, bw_adjust=0.8, ax=ax,
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6, zorder=0)
    ax.set_xlabel(r"$\log_{2}(d_\mathrm{inter} \,/\, d_\mathrm{intra})$")
    ax.set_ylabel("Density")

    x_min, x_max = ax.get_xlim()
    ticks = np.arange(np.floor(x_min), np.ceil(x_max) + 1)
    ax.set_xticks(ticks)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, which="major", alpha=0.15)

    leg = ax.get_legend()
    if leg is not None:
        leg.set_title(None)
        leg.set_frame_on(True)
        leg.get_frame().set_alpha(0.9)

    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(out_filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    OUT_DIR = "results/reference_characteristics"
    os.makedirs(OUT_DIR, exist_ok=True)
    FILE_FORMAT = "svg"

    def filtered(characteristics):
        """Filter out singleton genomes (no finite/positive nearest distance)."""
        same = characteristics["nearest_same"]
        other = characteristics["nearest_other"]
        mask = np.isfinite(same) & np.isfinite(other) & (same > 0) & (other > 0)
        return same[mask], other[mask]

    aggregated_data = []

    ############################## SARS-CoV-2 (China + USA) ##############################
    for location in ["China", "USA"]:
        # USA reference lives under Reference_downsampled per the README's downsampling note;
        # China's aggregate remapping/mapping file is produced by split_reference.py directly
        # under the bare Reference/ folder (confirmed: this step also writes sequence_mapping.txt).
        ref_dir = f"data/SARS-CoV-2/{location}/Reference"
        if location == "USA":
            ref_dir = f"data/SARS-CoV-2/{location}/Reference_downsampled"

        metadata_path = f"{ref_dir}/metadata.tsv"
        sequence_mapping_path = f"{ref_dir}/sequence_mapping.txt"
        # Both s5000 and s10000 were computed at the aggregate level for SARS-CoV-2
        # (unlike IAV, where only s500 was); s10000 matches the original script's choice.
        mash_path = f"{ref_dir}/mash_triangle_s10000.dist"

        seq2lin = read_metadata_sarscov2(metadata_path)
        sequence_mapping = read_sequence_mapping(sequence_mapping_path)

        characteristics = calculate_characteristics(
            mash_path, seq2lin, sequence_mapping, name=f"SARS-CoV-2 ({location})"
        )
        calculate_averages(characteristics, f"{OUT_DIR}/characteristics_sarscov2_{location.lower()}.txt")

        same, other = filtered(characteristics)
        plot_scatter(same, other, f"{OUT_DIR}/ambiguity_scatter_sarscov2_{location.lower()}.{FILE_FORMAT}")
        aggregated_data.append((f"SARS-CoV-2 ({location})", same, other))

    ############################## IAV ##############################
    ref_dir = "data/IAV/Reference"
    metadata_path = f"{ref_dir}/metadata.tsv"
    sequence_mapping_path = f"{ref_dir}/sequences_concatenated_ns_mapping.txt"
    # s1000 computed specifically for this script (per the README addition
    # documenting this extra aggregate-level sketch) -- distinct from the
    # s500 aggregate sketch used elsewhere in benchmarking.
    mash_path = f"{ref_dir}/mash_triangle_s1000.dist"

    seq2clade = read_metadata_iav(metadata_path)
    sequence_mapping = read_sequence_mapping(sequence_mapping_path)

    characteristics = calculate_characteristics(
        mash_path, seq2clade, sequence_mapping, name="IAV"
    )
    calculate_averages(characteristics, f"{OUT_DIR}/characteristics_iav.txt")

    same, other = filtered(characteristics)
    plot_scatter(same, other, f"{OUT_DIR}/ambiguity_scatter_iav.{FILE_FORMAT}")
    aggregated_data.append(("IAV", same, other))

    ############################## Aggregated histogram ##############################
    plot_histogram(aggregated_data, f"{OUT_DIR}/ambiguity_histogram_all_datasets.{FILE_FORMAT}")

if __name__ == "__main__":
    main()