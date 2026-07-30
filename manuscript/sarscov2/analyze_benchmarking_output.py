import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import f1_score
from scipy.stats import rankdata, spearmanr, wilcoxon

# ==============
# Configuration
# ==============
DATA_ROOT = "data/SARS-CoV-2"
OUT_DIR = "results/SARS-CoV-2/figures"
os.makedirs(OUT_DIR, exist_ok=True)

LOCATIONS = ["China", "USA"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]
ALPHAS = ["0.1", "1.0", "10.0"]
CONFIGURATIONS = list(range(1, 11))
REPLICATES = list(range(1, 11))
DIRICHLET_COVERAGE = "2000"
MIN_ABUNDANCE = 0.001
FILE_FORMAT = "svg"

ID_IDX = 0
LENGTH_IDX = 8
LINEAGE_IDX = 13
KALLISTO_ID_IDX = 0
KALLISTO_TPM_IDX = 4

MAPPING_SC2 = {
    "all": ["F1-optimized", "L1-optimized", "Rank-optimized"],
    "medoid_mash_s10000": ["F1-optimized", "L1-optimized", "Rank-optimized"],
    "hierarchical_mash_s10000_t0.99": ["F1-optimized", "L1-optimized", "Rank-optimized"],
    "vsearch_sourmash-cosine_s3_p1": ["F1-optimized"],
    "vsearch_mash_s5000_p90": ["L1-optimized"],
    "vsearch_sourmash-jaccard_s6_p25": ["Rank-optimized"],
    "reset_mash_s10000_cost-0.100000_scale-0.00001": ["F1-optimized"],
    "reset_sourmash-cosine_s3_cost-0.100000_scale-0.00001": ["L1-optimized"],
    "reset_mash_s5000_cost-1.000000_scale-0.00000": ["Rank-optimized"],
}
RESET_SC2 = {
    "F1-optimized": "reset_mash_s10000_cost-0.100000_scale-0.00001",
    "L1-optimized": "reset_sourmash-cosine_s3_cost-0.100000_scale-0.00001",
    "Rank-optimized": "reset_mash_s5000_cost-1.000000_scale-0.00000",
}

COLORS = {
    "ALL": "#378ADD",
    "MEDOID": "#888780",
    "HIERARCHICAL": "#EFAF00",
    "VSEARCH": "#534AB7",
    "RESET": "#D85A30",
}
TUNING_COLORS = {
    "F1-optimized": "#D85A30",
    "L1-optimized": "#534AB7",
    "Rank-optimized": "#3AA760",
}
MARKERS = {"F1-optimized": "o", "L1-optimized": "s", "Rank-optimized": "D"}
TUNING_TARGETS = ["F1-optimized", "L1-optimized", "Rank-optimized"]
GROUP_ORDER = ["All", "Medoid", "Hierarchical", "VSEARCH", "ReSeT"]
OPT_ORDER = ["F1/L1/Rank", "F1/Rank", "L1/Rank", "F1/L1", "F1", "L1", "Rank"]

FIGURE_WIDTH = 4.0
FIGURE_HEIGHT = 4.0
MARKER_ALPHA = 0.6
SHAPE_SIZE = 50
STRIP_FIGURE_WIDTH = 9
STRIP_FIGURE_HEIGHT = 14
MEDIAN_MARKER_KW = dict(marker="|", markersize=40, markeredgewidth=2.5, linestyle="None")


# ==============
# Path builders
# ==============
def metadata_reference_path(location):
    return f"{DATA_ROOT}/{location}/Reference/metadata.tsv"

def metadata_simulation_path(location):
    return f"{DATA_ROOT}/{location}/Simulation/metadata.tsv"

def estimations_dir(location):
    return f"{DATA_ROOT}/{location}/Simulation/estimations"

def difficulty_reads_dir(location, difficulty):
    return f"{DATA_ROOT}/{location}/Simulation/{difficulty}"

def difficulty_abundance_path(location, method, difficulty, sample):
    return f"{estimations_dir(location)}/{method}/{difficulty}_{sample}/abundance.tsv"

def dirichlet_reads_dir(location, config, alpha):
    return f"{DATA_ROOT}/{location}/Simulation/Dirichlet/config_{config}/alpha_{alpha}"

def dirichlet_abundance_path(location, method, config, alpha, replicate):
    return (f"{estimations_dir(location)}/{method}/Dirichlet/config_{config}/"
            f"alpha_{alpha}/replicate_{replicate}/abundance.tsv")

def discover_methods(location):
    base = estimations_dir(location)
    methods = [m for m in os.listdir(base) if os.path.isdir(os.path.join(base, m))]
    methods.sort()  # consistent ordering across locations
    return methods


# ==============================
# Metadata / read-level parsing
# ==============================
def read_metadata(path):
    metadata = {}
    with open(path, "r") as f_in:
        next(f_in)  # header
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
                        f"Sequence '{seq_id}' from {path} not found in metadata "
                        f"or its lineage not in lin2idx -- check metadata/read "
                        f"consistency before trusting downstream results."
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
                    f"Sequence '{seq_id}' from {path} not found in metadata "
                    f"or its lineage not in lin2idx."
                )
            tpm = float(fields[KALLISTO_TPM_IDX])
            abundances[sample_idx, lin_idx] += tpm


# =============================================================================
# Difficulty-stratified: build (samples x lineages x methods+1) per difficulty
# =============================================================================
def build_difficulty_arrays(location, metadata_ref, metadata_sim, lin2idx, methods, samples):
    n_lineages = len(lin2idx)
    n_methods = len(methods)
    results = {}

    for difficulty in DIFFICULTIES:
        abundances = np.zeros((len(samples), n_lineages, n_methods + 1), dtype=np.float64)

        # Groundtruth (last index)
        gt = np.zeros((len(samples), n_lineages), dtype=np.float64)
        reads_dir = difficulty_reads_dir(location, difficulty)
        for s_idx, sample in enumerate(samples):
            for part in ("R1", "R2"):
                fq_path = f"{reads_dir}/sample_{sample}_{part}.fastq"
                _accumulate_fastq_counts(fq_path, metadata_sim, lin2idx, gt, s_idx)
        gt = _finalize_abundances(gt, MIN_ABUNDANCE)
        abundances[:, :, -1] = gt

        # Estimated methods
        for m_idx, method in enumerate(methods):
            est = np.zeros((len(samples), n_lineages), dtype=np.float64)
            for s_idx, sample in enumerate(samples):
                path = difficulty_abundance_path(location, method, difficulty, sample)
                _accumulate_kallisto_counts(path, metadata_ref, lin2idx, est, s_idx)
            est = _finalize_abundances(est, MIN_ABUNDANCE)
            abundances[:, :, m_idx] = est

        results[difficulty] = abundances

    method_names = np.array(methods + ["GROUNDTRUTH"])
    return results, method_names


# =====================================================================
# Dirichlet: build (samples x lineages x methods+1 x alphas x configs)
# =====================================================================
def build_dirichlet_arrays(location, metadata_ref, metadata_sim, lin2idx, methods, replicates):
    n_lineages = len(lin2idx)
    n_methods = len(methods)
    n_alphas = len(ALPHAS)
    n_configs = len(CONFIGURATIONS)

    abundances = np.zeros(
        (len(replicates), n_lineages, n_methods + 1, n_alphas, n_configs), dtype=np.float64
    )

    for a_idx, alpha in enumerate(ALPHAS):
        for c_idx, config in enumerate(CONFIGURATIONS):
            reads_dir = dirichlet_reads_dir(location, config, alpha)

            # Groundtruth
            gt = np.zeros((len(replicates), n_lineages), dtype=np.float64)
            for r_idx, rep in enumerate(replicates):
                for part in ("1", "2"):
                    fq_path = f"{reads_dir}/replicate_{rep}/sample_{part}.fq"
                    _accumulate_fastq_counts(fq_path, metadata_sim, lin2idx, gt, r_idx)
            gt = _finalize_abundances(gt, MIN_ABUNDANCE)
            abundances[:, :, -1, a_idx, c_idx] = gt

            # Estimated methods
            for m_idx, method in enumerate(methods):
                est = np.zeros((len(replicates), n_lineages), dtype=np.float64)
                for r_idx, rep in enumerate(replicates):
                    path = dirichlet_abundance_path(location, method, config, alpha, rep)
                    _accumulate_kallisto_counts(path, metadata_ref, lin2idx, est, r_idx)
                est = _finalize_abundances(est, MIN_ABUNDANCE)
                abundances[:, :, m_idx, a_idx, c_idx] = est

    method_names = np.array(methods + ["GROUNDTRUTH"])
    return abundances, method_names


# ============================================================
# Metrics (from the analysis scripts, decoupled from npz I/O)
# ============================================================
def compute_metrics_diff(abundances, groundtruth_idx=-1):
    n_methods = abundances.shape[2] - 1
    n_samples = abundances.shape[0]

    groundtruth = abundances[:, :, groundtruth_idx]
    groundtruth_binary = (groundtruth > 0).astype(int)

    results = []
    for method_idx in range(n_methods):
        cur = abundances[:, :, method_idx]
        cur_binary = (cur > 0).astype(int)
        l1_errors = np.sum(np.abs(cur - groundtruth), axis=1)
        l1_scores = 1.0 - (l1_errors / 2)
        f1_scores = np.array([f1_score(groundtruth_binary[i], cur_binary[i]) for i in range(n_samples)])
        results.append({"l1_scores": l1_scores, "f1_scores": f1_scores})
    return results


# ===============================
# Difficulty-stratified plotting
# ===============================
def plot_panel_diff(abundances, method_names, mapping, virus, difficulty, output_path,
                     l1_min=0, l1_max=1, f1_min=0, f1_max=1):
    metrics = compute_metrics_diff(abundances)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    def classify(method_name):
        name_lower = method_name.lower()
        if name_lower.startswith("medoid"):
            group = "MEDOID/HIERARCHICAL" if virus == "SARS-CoV-2" else "MEDOID"
            return group, COLORS["MEDOID"]
        if name_lower.startswith("hierarchical"):
            return "HIERARCHICAL", COLORS["HIERARCHICAL"]
        if name_lower.startswith("vsearch"):
            return "VSEARCH", COLORS["VSEARCH"]
        if name_lower.startswith("reset"):
            return "RESET", COLORS["RESET"]
        if method_name == "all":
            return "ALL", COLORS["ALL"]
        raise ValueError(f"Unknown method name: {method_name}")

    def marker_and_label(optimization_type):
        if len(optimization_type) == 3:
            return "^", "all"
        if len(optimization_type) == 2:
            return "v", "/".join(t.replace("-optimized", "") for t in optimization_type)
        if len(optimization_type) == 1:
            return MARKERS[optimization_type[0]], optimization_type[0].replace("-optimized", "")
        raise ValueError(f"Invalid optimization type: {optimization_type}")

    legend_handles = {}
    for method_idx, method_name in enumerate(method_names[:-1]):
        method_name = str(method_name)
        if virus == "SARS-CoV-2" and method_name.lower().startswith("hierarchical"):
            continue  # excluded: identical to Medoid for SARS-CoV-2
        if method_name not in mapping:
            print(f"Warning: method '{method_name}' not found in mapping and will be skipped.")
            continue

        group, color = classify(method_name)
        marker, label = marker_and_label(mapping[method_name])

        cur = metrics[method_idx]
        l1_median, f1_median = np.median(cur["l1_scores"]), np.median(cur["f1_scores"])
        l1_q25, l1_q75 = np.percentile(cur["l1_scores"], [25, 75])
        f1_q25, f1_q75 = np.percentile(cur["f1_scores"], [25, 75])

        ax.errorbar(l1_median, f1_median,
                     xerr=[[l1_median - l1_q25], [l1_q75 - l1_median]],
                     yerr=[[f1_median - f1_q25], [f1_q75 - f1_median]],
                     fmt="none", ecolor=color, elinewidth=1, capsize=2, capthick=1,
                     alpha=0.6, zorder=1)
        ax.scatter(l1_median, f1_median, color=color, alpha=MARKER_ALPHA, marker=marker,
                    s=SHAPE_SIZE, linewidths=0.5, zorder=2)

        opt_label = "/".join(t.replace("-optimized", "").replace("L1", "Abund.")
                              for t in mapping[method_name]) if len(mapping[method_name]) <= 2 else "all"
        key = f"{group} ({opt_label})" if opt_label != "all" else group
        if key not in legend_handles:
            legend_handles[key] = mlines.Line2D(
                [], [], color=color, alpha=MARKER_ALPHA, marker=marker, linestyle="None",
                markersize=4, markeredgewidth=0.5, label=key,
            )

    ax.set_xlabel("Abundance accuracy (1 - L1/2)", fontsize=10)
    ax.set_ylabel("F1-score", fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlim(l1_min, l1_max)
    ax.set_ylim(f1_min, f1_max)

    sorted_handles = sorted(
        legend_handles.values(),
        key=lambda h: (
            next((i for i, g in enumerate(["ALL", "MEDOID/HIERARCHICAL", "MEDOID", "HIERARCHICAL", "VSEARCH", "RESET"])
                  if h.get_label().startswith(g)), 99),
            next((i for i, o in enumerate(["all", "F1", "Abund.", "Rank", "F1/Abund.", "F1/Rank", "Abund./Rank"])
                  if o in h.get_label()), 99),
        )
    )
    ax.legend(handles=sorted_handles, fontsize=7, frameon=True, framealpha=0.9, markerscale=1.2, loc="best")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ========================
# Pooled rank correlation
# ========================
def compute_pooled_rank_correlation(abundance_sets, method_names_sets, mapping_per_panel, virus):
    pooled_l1_ranks, pooled_f1_ranks, per_panel = [], [], []

    for panel_label, abundances, method_names, mapping in zip(
        [f"panel_{i}" for i in range(len(abundance_sets))], abundance_sets, method_names_sets, mapping_per_panel
    ):
        metrics = compute_metrics_diff(abundances)
        l1_meds, f1_meds = [], []
        for method_idx, method_name in enumerate(method_names[:-1]):
            method_name = str(method_name)
            if virus == "SARS-CoV-2" and method_name.lower().startswith("hierarchical"):
                continue
            if method_name not in mapping:
                continue
            cur = metrics[method_idx]
            l1_meds.append(np.median(cur["l1_scores"]))
            f1_meds.append(np.median(cur["f1_scores"]))

        l1_meds, f1_meds = np.array(l1_meds), np.array(f1_meds)
        if len(l1_meds) < 2:
            continue

        l1_ranks = rankdata(l1_meds, method="average")
        f1_ranks = rankdata(f1_meds, method="average")
        pooled_l1_ranks.extend(l1_ranks)
        pooled_f1_ranks.extend(f1_ranks)

        panel_rho, _ = spearmanr(l1_meds, f1_meds)
        per_panel.append((panel_label, panel_rho, len(l1_meds)))

    rho, p = spearmanr(pooled_l1_ranks, pooled_f1_ranks)
    return {"rho": rho, "p": p, "n": len(pooled_l1_ranks), "per_panel": per_panel}


# =============================
# Wilcoxon / BH-FDR comparison
# =============================
def benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if not valid.any():
        return q
    pv = p[valid]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    raw = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    q[valid] = out
    return q

def per_sample_metrics(abundances, method_names, mapping, virus):
    metrics = compute_metrics_diff(abundances)
    out = {}
    for i, name in enumerate(method_names[:-1]):
        name = str(name)
        if virus == "SARS-CoV-2" and name.lower().startswith("hierarchical"):
            continue
        if name not in mapping:
            continue
        out[name] = {"l1": metrics[i]["l1_scores"], "f1": metrics[i]["f1_scores"]}
    return out

def wilcoxon_verdict(reset, base, alpha=0.05):
    delta = float(np.median(reset - base))
    try:
        _, p = wilcoxon(reset, base, alternative="two-sided")
    except ValueError:
        return delta, np.nan, "n/a"
    if np.isnan(p) or p >= alpha:
        return delta, p, "ns"
    return delta, p, "better" if delta > 0 else "worse"

def analyse_dataset(label, abundance_sets, difficulty_labels, mapping, virus, alpha=0.05):
    per_file = [per_sample_metrics(a, m, mapping, virus) for a, m in abundance_sets]
    methods = list(per_file[0].keys())
    pooled = {m: {"l1": np.concatenate([f[m]["l1"] for f in per_file]),
                  "f1": np.concatenate([f[m]["f1"] for f in per_file])}
              for m in methods}

    baselines = {m: s for m, s in pooled.items() if not m.lower().startswith("reset")}
    resets = [m for m in methods if m.lower().startswith("reset")]

    bl_names = list(baselines.keys())
    l1_meds = np.array([np.median(baselines[m]["l1"]) for m in bl_names])
    f1_meds = np.array([np.median(baselines[m]["f1"]) for m in bl_names])
    l1_ranks = rankdata(-l1_meds, method="average")
    f1_ranks = rankdata(-f1_meds, method="average")
    mean_ranks = (l1_ranks + f1_ranks) / 2.0
    best = bl_names[int(np.argmin(mean_ranks))]

    def build_records(strata):
        records = []
        for stratum_label, scores in strata:
            for r in resets:
                for metric in ("l1", "f1"):
                    delta, p, _ = wilcoxon_verdict(scores[r][metric], scores[best][metric])
                    records.append({"stratum": stratum_label, "reset": r, "metric": metric,
                                     "n": len(scores[r][metric]), "delta": delta, "p": p})
        qvals = benjamini_hochberg([rec["p"] for rec in records])
        for rec, q in zip(records, qvals):
            rec["q"] = q
            if np.isnan(q):
                rec["verdict"] = "n/a"
            elif q >= alpha:
                rec["verdict"] = "ns"
            else:
                rec["verdict"] = "better" if rec["delta"] > 0 else "worse"
        return records

    pooled_records = build_records([("pooled", pooled)])
    perdiff_records = build_records(list(zip(difficulty_labels, per_file)))

    print(f"\n=== {label}  (BH-FDR within each family, alpha = {alpha}) ===")
    print(f"  Best baseline (mean-rank winner, pooled): {best}")
    print(f"    median L1 = {np.median(baselines[best]['l1']):.3f}, "
          f"median F1 = {np.median(baselines[best]['f1']):.3f}")

    def print_block(title, records):
        print(f"\n  ----- {title} (BH family size = {len(records)}) -----")
        by_stratum = {}
        for rec in records:
            by_stratum.setdefault(rec["stratum"], []).append(rec)
        for stratum_label, rows in by_stratum.items():
            n = rows[0]["n"]
            print(f"\n  {stratum_label} (n = {n})")
            print(f"  {'ReSeT config':70s}   {'Delta L1':>8s}  {'p (L1)':>9s}  {'q (L1)':>9s}  {'L1':<8s}"
                  f"   {'Delta F1':>8s}  {'p (F1)':>9s}  {'q (F1)':>9s}  {'F1':<8s}")
            by_reset = {}
            for rec in rows:
                by_reset.setdefault(rec["reset"], {})[rec["metric"]] = rec
            for r in resets:
                l1, f1 = by_reset[r]["l1"], by_reset[r]["f1"]

                def fmt(rec):
                    p_ = "n/a" if np.isnan(rec["p"]) else f"{rec['p']:.2e}"
                    q_ = "n/a" if np.isnan(rec["q"]) else f"{rec['q']:.2e}"
                    return f"{rec['delta']:+8.3f}  {p_:>9s}  {q_:>9s}  {rec['verdict']:<8s}"

                print(f"  {r:70s}   {fmt(l1)}   {fmt(f1)}")

    print_block("Pooled across difficulties", pooled_records)
    print_block("Per-difficulty", perdiff_records)


# ===================
# Dirichlet analysis
# ===================
def analyze_per_composition(abundances, method_names):
    n_samples, _, n_total_methods, n_alphas, n_configs = abundances.shape
    n_methods = n_total_methods - 1
    per_composition = np.zeros((n_alphas, n_configs, n_methods, 2), dtype=np.float64)
    for a in range(n_alphas):
        for c in range(n_configs):
            gt = abundances[:, :, -1, a, c]
            gt_bin = (gt > 0).astype(int)
            for m in range(n_methods):
                pred = abundances[:, :, m, a, c]
                pred_bin = (pred > 0).astype(int)
                l1_scores = 1.0 - np.sum(np.abs(pred - gt), axis=1) / 2
                f1_scores = np.array([f1_score(gt_bin[s], pred_bin[s]) for s in range(n_samples)])
                per_composition[a, c, m] = [np.mean(l1_scores), np.mean(f1_scores)]
    return per_composition

def get_pair_diffs(per_composition, method_names, reset_method, baseline_method, alpha_idx):
    names = [str(n) for n in method_names[:-1]]
    diffs = (per_composition[alpha_idx, :, names.index(reset_method), :]
             - per_composition[alpha_idx, :, names.index(baseline_method), :])
    return diffs[:, 0], diffs[:, 1]

def get_display_label(method_name, mapping, dataset_name):
    opts = mapping[method_name]
    opt_str = "/".join(o.replace("-optimized", "") for o in opts)
    name_lower = method_name.lower()
    if name_lower.startswith("reset"):
        return f"ReSeT ({opt_str})"
    if name_lower.startswith("medoid"):
        prefix = "Medoid/Hierarchical" if "SARS-CoV-2" in (dataset_name or "") else "Medoid"
        return f"{prefix} ({opt_str})"
    if name_lower.startswith("hierarchical"):
        return f"Hierarchical ({opt_str})"
    if name_lower.startswith("vsearch"):
        return f"VSEARCH ({opt_str})"
    if method_name == "ALL":
        return f"All ({opt_str})"
    return f"{method_name.split('_')[0]} ({opt_str})"

def sort_methods_by_label(indices, labels):
    def key(i):
        label = labels[i]
        group = label.split(" (")[0]
        opt = label.split("(")[1].rstrip(")")
        g_idx = GROUP_ORDER.index(group) if group in GROUP_ORDER else 99
        o_idx = OPT_ORDER.index(opt) if opt in OPT_ORDER else 99
        return (g_idx, o_idx)
    return sorted(indices, key=key)

def rank_baselines(per_composition, method_names, name=None):
    mean_l1 = np.mean(per_composition[:, :, :, 0], axis=(0, 1))
    mean_f1 = np.mean(per_composition[:, :, :, 1], axis=(0, 1))

    names = [str(n) for n in method_names[:-1]]
    bl_idx = [i for i, n in enumerate(names) if not n.lower().startswith("reset")]
    bl_names = [names[i] for i in bl_idx]
    bl_l1, bl_f1 = mean_l1[bl_idx], mean_f1[bl_idx]

    l1_ranks = rankdata(-bl_l1, method="average")
    f1_ranks = rankdata(-bl_f1, method="average")
    avg_ranks = (l1_ranks + f1_ranks) / 2

    preference_order = ["hierarchical", "medoid", "vsearch"]

    def preference(n):
        n_lower = n.lower()
        for i, k in enumerate(preference_order):
            if n_lower.startswith(k):
                return i
        return len(preference_order)

    order = sorted(range(len(bl_names)), key=lambda i: (avg_ranks[i], preference(bl_names[i])))

    print(f"\n=== Baseline Ranking for {name or 'Dataset'} ===")
    print(f"{'Method':<55} {'F1':>8} {'L1':>8} {'F1-rank':>8} {'L1-rank':>8} {'avg rank':>10}")
    for i in order:
        print(f"{bl_names[i]:<55} {bl_f1[i]:>8.4f} {bl_l1[i]:>8.4f} "
              f"{f1_ranks[i]:>8.1f} {l1_ranks[i]:>8.1f} {avg_ranks[i]:>10.2f}")
    print(f"\n  -> Strongest baseline: {bl_names[order[0]]}")
    return bl_names[order[0]]

def _print_rank_table(labels, order, l1_ranks, f1_ranks, avg_ranks, total, label_width, title):
    header = (f"  {'Method':<{label_width}}"
              f"  {'F1 rank':>8}  {'L1 rank':>8}  {'Avg rank':>9}  {'Std':>6}"
              f"  {'#1 F1':>8}  {'#1 L1':>8}  {'#1 avg':>8}")
    print(f"\n  {title}  ({total} comparisons)")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in order:
        rk_avg = avg_ranks[:, :, i]
        std_avg = np.std(rk_avg, ddof=1) if rk_avg.size > 1 else 0.0
        print(f"  {labels[i]:<{label_width}}"
              f"  {np.mean(f1_ranks[:, :, i]):>8.2f}"
              f"  {np.mean(l1_ranks[:, :, i]):>8.2f}"
              f"  {np.mean(rk_avg):>9.2f}"
              f"  {std_avg:>6.2f}"
              f"  {int(np.sum(f1_ranks[:, :, i] == 1)):>3}/{total:<4}"
              f"  {int(np.sum(l1_ranks[:, :, i] == 1)):>3}/{total:<4}"
              f"  {int(np.sum(rk_avg == 1)):>3}/{total:<4}")

def compute_rank_summary(per_composition, method_names, mapping, name=None):
    names = [str(n) for n in method_names[:-1]]
    n_alphas, n_configs, _, _ = per_composition.shape

    is_sc2 = "SARS-CoV-2" in (name or "")
    valid_idx = [i for i, m in enumerate(names)
                 if m in mapping and not (is_sc2 and m.lower().startswith("hierarchical"))]
    valid_names = [names[i] for i in valid_idx]
    valid_pc = per_composition[:, :, valid_idx, :]

    n_valid = len(valid_idx)
    l1_ranks = np.zeros((n_alphas, n_configs, n_valid))
    f1_ranks = np.zeros((n_alphas, n_configs, n_valid))
    for a in range(n_alphas):
        for c in range(n_configs):
            l1_ranks[a, c] = rankdata(-valid_pc[a, c, :, 0], method="average")
            f1_ranks[a, c] = rankdata(-valid_pc[a, c, :, 1], method="average")
    avg_ranks = (l1_ranks + f1_ranks) / 2

    labels = {i: get_display_label(valid_names[i], mapping, name) for i in range(n_valid)}
    order = sort_methods_by_label(range(n_valid), labels)
    label_width = max(len(labels[i]) for i in range(n_valid)) + 2

    print(f"\n=== Rank Summary for {name or 'Dataset'} ===")
    print(f"    {n_alphas} alphas x {n_configs} compositions = {n_alphas * n_configs} total comparisons")

    _print_rank_table(labels, order, l1_ranks, f1_ranks, avg_ranks,
                       total=n_alphas * n_configs, label_width=label_width,
                       title="Pooled across all alpha (main table)")
    for a, alpha in enumerate(ALPHAS):
        _print_rank_table(labels, order, l1_ranks[a:a + 1], f1_ranks[a:a + 1], avg_ranks[a:a + 1],
                           total=n_configs, label_width=label_width,
                           title=f"alpha = {alpha} (supplementary)")

def plot_strip_row(ax, diffs, color, y_pos, jitter_amount=0.18, seed=0):
    rng = np.random.default_rng(seed)
    y_jit = y_pos + rng.uniform(-jitter_amount, jitter_amount, size=len(diffs))
    ax.scatter(diffs, y_jit, s=40, alpha=0.35, color=color, edgecolors="white", linewidths=0.4, zorder=2)
    ax.plot([np.median(diffs)], [y_pos], color=color, **MEDIAN_MARKER_KW, zorder=3)
    ax.text(0.98, y_pos, f"{int(np.sum(diffs > 0))}/{len(diffs)}",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=10, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

def plot_comparison(per_composition, method_names, reset_pairs, baseline_pairs, metric, output_path, xlims):
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(STRIP_FIGURE_WIDTH, STRIP_FIGURE_HEIGHT / len(ALPHAS)),
                              gridspec_kw=dict(wspace=0.15))
    y_pos = {t: i for i, t in enumerate(TUNING_TARGETS)}

    for a, alpha in enumerate(ALPHAS):
        ax = axes[a]
        for tuning in TUNING_TARGETS:
            l1_d, f1_d = get_pair_diffs(per_composition, method_names, reset_pairs[tuning], baseline_pairs[tuning], a)
            diffs = f1_d if metric == "F1" else l1_d
            plot_strip_row(ax, diffs, TUNING_COLORS[tuning], y_pos[tuning])

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
        ax.set_xlim(*xlims)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks(list(y_pos.values()))
        if a == 0:
            ax.set_yticklabels([f"{t.split('-')[0]:^9}\n{t.split('-')[1]:^9}" for t in TUNING_TARGETS], fontsize=12)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="x", labelsize=10)
        ax.set_xlabel(f"D {metric}", fontsize=12)
        ax.grid(True, axis="x", alpha=0.15, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"alpha = {alpha}", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format=FILE_FORMAT)
    plt.close(fig)

def compute_axis_limits(all_diffs_per_metric, padding=0.05):
    limits = {}
    for metric, arrays in all_diffs_per_metric.items():
        max_abs = np.max(np.abs(np.concatenate(arrays)))
        lim = max_abs * (1 + padding)
        limits[metric] = (-lim, lim)
    return limits


def main():
    difficulty_samples = [f"{i:02d}" for i in range(1, 21)]  # 20 samples per difficulty

    per_location_difficulty = {}   # location -> {difficulty: abundances}
    per_location_difficulty_methods = {}
    per_location_dirichlet = {}    # location -> abundances
    per_location_dirichlet_methods = {}

    for location in LOCATIONS:
        metadata_ref = read_metadata(metadata_reference_path(location))
        metadata_sim = read_metadata(metadata_simulation_path(location))

        lineages = sorted(set(metadata_ref[s]["lineage"] for s in metadata_ref))
        lin2idx = {lineage: idx for idx, lineage in enumerate(lineages)}

        methods = discover_methods(location)

        diff_arrays, diff_method_names = build_difficulty_arrays(
            location, metadata_ref, metadata_sim, lin2idx, methods, difficulty_samples
        )
        per_location_difficulty[location] = diff_arrays
        per_location_difficulty_methods[location] = diff_method_names

        dir_arrays, dir_method_names = build_dirichlet_arrays(
            location, metadata_ref, metadata_sim, lin2idx, methods, REPLICATES
        )
        per_location_dirichlet[location] = dir_arrays
        per_location_dirichlet_methods[location] = dir_method_names

    # ---------- Difficulty-stratified: per-dataset axis limits ----------
    min_l1_per_dataset, max_l1_per_dataset = {}, {}
    min_f1_per_dataset, max_f1_per_dataset = {}, {}

    def update_ranges(key, abundances):
        for m in compute_metrics_diff(abundances):
            min_l1_per_dataset[key] = min(min_l1_per_dataset.get(key, 1.0), np.min(m["l1_scores"]))
            max_l1_per_dataset[key] = max(max_l1_per_dataset.get(key, 0.0), np.max(m["l1_scores"]))
            min_f1_per_dataset[key] = min(min_f1_per_dataset.get(key, 1.0), np.min(m["f1_scores"]))
            max_f1_per_dataset[key] = max(max_f1_per_dataset.get(key, 0.0), np.max(m["f1_scores"]))

    for location in LOCATIONS:
        for difficulty in DIFFICULTIES:
            update_ranges(location, per_location_difficulty[location][difficulty])

    def lims(key):
        return (max(0.0, min_l1_per_dataset[key] - 0.05), min(1.05, max_l1_per_dataset[key] + 0.05),
                max(0.0, min_f1_per_dataset[key] - 0.05), min(1.05, max_f1_per_dataset[key] + 0.05))

    # ---------- Difficulty-stratified plots ----------
    for location in LOCATIONS:
        l1_lo, l1_hi, f1_lo, f1_hi = lims(location)
        for difficulty in DIFFICULTIES:
            print(f"Plotting SARS-CoV-2 {location} - {difficulty}...")
            plot_panel_diff(
                per_location_difficulty[location][difficulty],
                per_location_difficulty_methods[location],
                MAPPING_SC2, "SARS-CoV-2", difficulty,
                f"{OUT_DIR}/difficulty_sarscov2_{location.lower()}_{difficulty.lower()}.{FILE_FORMAT}",
                l1_lo, l1_hi, f1_lo, f1_hi,
            )

    # ---------- Pooled rank correlation ----------
    print("\n========== Pooled L1 vs F1 rank correlations ==========")
    sc2_abundance_sets = [per_location_difficulty[loc][diff] for loc in LOCATIONS for diff in DIFFICULTIES]
    sc2_method_sets = [per_location_difficulty_methods[loc] for loc in LOCATIONS for diff in DIFFICULTIES]
    sc2_result = compute_pooled_rank_correlation(
        sc2_abundance_sets, sc2_method_sets, [MAPPING_SC2] * len(sc2_abundance_sets), "SARS-CoV-2"
    )
    print(f"\nSARS-CoV-2: rho = {sc2_result['rho']:.3f}, p = {sc2_result['p']:.2e}, n = {sc2_result['n']}")
    for name, rho, n in sc2_result["per_panel"]:
        print(f"  {name}: rho = {rho:.3f} (n={n})")

    # ---------- Wilcoxon / BH-FDR baseline comparison ----------
    print("\n========== ReSeT vs. best non-ReSeT baseline (paired Wilcoxon, two-sided, BH-FDR) ==========")
    for location in LOCATIONS:
        abundance_sets = [(per_location_difficulty[location][d], per_location_difficulty_methods[location])
                           for d in DIFFICULTIES]
        analyse_dataset(f"SARS-CoV-2 {location}", abundance_sets, DIFFICULTIES, MAPPING_SC2, "SARS-CoV-2")

    # ---------- Dirichlet: rank baselines, rank summary, strip plots ----------
    print("\n" + "=" * 80)
    print("DIRICHLET: RANK ANALYSIS ACROSS ALL COMPOSITIONS")
    print("=" * 80)

    all_diffs = {"F1": [], "L1": []}
    loaded = []
    for location in LOCATIONS:
        label = f"SARS-CoV-2 {location} coverage={DIRICHLET_COVERAGE}"
        abundances = per_location_dirichlet[location]
        method_names = per_location_dirichlet_methods[location]

        per_comp = analyze_per_composition(abundances, method_names)
        best_baseline = rank_baselines(per_comp, method_names, name=label)
        baseline = {t: best_baseline for t in RESET_SC2}

        for tuning in RESET_SC2:
            for a in range(len(ALPHAS)):
                l1_d, f1_d = get_pair_diffs(per_comp, method_names, RESET_SC2[tuning], baseline[tuning], a)
                all_diffs["F1"].append(f1_d)
                all_diffs["L1"].append(l1_d)

        loaded.append(dict(label=label, filename=f"sars-cov-2_{location.lower()}_{DIRICHLET_COVERAGE}",
                            per_composition=per_comp, method_names=method_names,
                            reset_methods=RESET_SC2, baseline=baseline))

    global_limits = compute_axis_limits(all_diffs, padding=0.05)

    for d in loaded:
        for metric in ("F1", "L1"):
            plot_comparison(
                d["per_composition"], d["method_names"], d["reset_methods"], d["baseline"],
                metric=metric,
                output_path=f"{OUT_DIR}/dirichlet_{d['filename']}_{metric.lower()}_comparison.{FILE_FORMAT}",
                xlims=global_limits[metric],
            )

    for d in loaded:
        compute_rank_summary(d["per_composition"], d["method_names"], MAPPING_SC2, name=d["label"])

if __name__ == "__main__":
    main()