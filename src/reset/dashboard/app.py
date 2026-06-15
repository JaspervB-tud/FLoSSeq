import streamlit as st
from reset.dashboard import pipeline
from reset import Solution, Solution_shm
import numpy as np
import os
import subprocess
import pandas as pd
import time

# Default to the bundled toy example so the dashboard works out of the box.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TOY_EXAMPLE_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "examples", "toy_example")
)


def run_app():
    st.title("ReSeT Dashboard")

    # Sidebar — data folder and distance format
    def choose_folder_macos():
        try:
            script = 'POSIX path of (choose folder with prompt "Select the data folder (contains clusters.csv and distances file)")'
            out = subprocess.check_output(["osascript", "-e", script])
            return out.decode("utf-8").strip()
        except Exception:
            return None

    st.sidebar.header("Data Selection")
    if "data_folder" not in st.session_state:
        st.session_state["data_folder"] = _TOY_EXAMPLE_DIR
    if st.sidebar.button("Choose folder (macOS Finder)"):
        folder = choose_folder_macos()
        if folder and folder != st.session_state["data_folder"]:
            st.cache_data.clear()
        if folder:
            st.session_state["data_folder"] = folder
            st.sidebar.success(f"Selected: {folder}")
            st.rerun()
        else:
            st.sidebar.error("No folder selected.")
    data_folder = st.sidebar.text_input(
        "Or enter data folder path", st.session_state["data_folder"]
    )
    st.session_state["data_folder"] = data_folder

    st.sidebar.header("File names")
    clusters_filename  = st.sidebar.text_input("Clusters file", value="clusters.csv")
    distances_filename = st.sidebar.text_input("Distances file", value="distances.tsv")

    st.sidebar.header("Distance format")
    distance_format = st.sidebar.selectbox(
        "Format",
        pipeline.DISTANCE_FORMATS,
        index=0,
        help="mash: Mash lower-triangular output. "
             "sourmash_cosine / sourmash_jaccard: Sourmash CSV output. "
             "generic: any delimited (i, j, distance) file.",
    )

    if not data_folder:
        st.info("Select a data folder or paste a path to proceed.")
        st.stop()

    clusters_path  = os.path.join(data_folder, clusters_filename)
    distances_path = os.path.join(data_folder, distances_filename)

    if not os.path.isfile(clusters_path):
        st.error(f"{clusters_filename} not found in {data_folder}")
        st.stop()
    if not os.path.isfile(distances_path):
        st.error(f"{distances_filename} not found in {data_folder}")
        st.stop()

    # Data loading
    def select_genomes():
        genomes     = st.session_state["genomes"]
        max_genomes = st.session_state.get("max_genomes", 1)
        random_state = st.session_state.get("random_state", None)

        selected = pipeline.downsample(genomes, max_genomes=max_genomes, random_state=random_state)

        # Build local index (position within the selected subset)
        seq2index = {}
        index2seq = []
        for seq_id in selected:
            seq2index[seq_id] = len(index2seq)
            index2seq.append(seq_id)

        # Global index of each selected sequence (row in the distance file)
        global_seq2index = st.session_state["global_seq2index"]
        global_indices = [global_seq2index[seq_id] for seq_id in index2seq]

        # Cluster assignments (local index)
        cluster2index = st.session_state["cluster2index"]
        selected_clusters = [cluster2index[genomes[seq_id]["cluster"]] for seq_id in index2seq]

        st.session_state["seq2index"]       = seq2index
        st.session_state["index2seq"]       = index2seq
        st.session_state["selected_genomes"] = selected
        st.session_state["selected_clusters"] = selected_clusters
        st.session_state["global_indices"]   = global_indices

        # Invalidate distance submatrix when selection changes
        last_params = st.session_state.get("_last_distance_params")
        if last_params is not None and last_params != (
            st.session_state["seed"], st.session_state["max_genomes"]
        ):
            st.session_state["distance_matrix"] = None
            st.session_state["_needs_distance_reload"] = True
        else:
            st.session_state["_needs_distance_reload"] = False

    @st.cache_data(show_spinner=True)
    def load_data(clusters_path, distances_path, distance_format):
        genomes, ordered_seq_ids = pipeline.read_clusters_for_dashboard(clusters_path)

        clusters      = {}
        cluster2index = {}
        index2cluster = []
        global_seq2index = {}

        for global_idx, seq_id in enumerate(ordered_seq_ids):
            global_seq2index[seq_id] = global_idx
            cluster = genomes[seq_id]["cluster"]
            if cluster not in cluster2index:
                cluster2index[cluster] = len(index2cluster)
                index2cluster.append(cluster)
            clusters.setdefault(cluster, []).append(seq_id)

        n = len(ordered_seq_ids)
        D_full = pipeline.load_distance_matrix(distances_path, n, distance_format=distance_format)

        st.session_state["genomes"]          = genomes
        st.session_state["clusters"]         = clusters
        st.session_state["cluster2index"]    = cluster2index
        st.session_state["index2cluster"]    = index2cluster
        st.session_state["global_seq2index"] = global_seq2index
        st.session_state["max_clustersize"]  = max(len(v) for v in clusters.values())
        st.session_state["D_full"]           = D_full
        select_genomes()

    need_reload = (
        "genomes" not in st.session_state
        or st.session_state.get("_loaded_folder",  "") != data_folder
        or st.session_state.get("_loaded_format",  "") != distance_format
        or st.session_state.get("_loaded_clusters", "") != clusters_filename
        or st.session_state.get("_loaded_distances", "") != distances_filename
    )
    if need_reload:
        load_data(clusters_path, distances_path, distance_format)
        st.session_state["_loaded_folder"]    = data_folder
        st.session_state["_loaded_format"]    = distance_format
        st.session_state["_loaded_clusters"]  = clusters_filename
        st.session_state["_loaded_distances"] = distances_filename

    st.write(f"Total sequences: **{len(st.session_state.get('genomes', {}))}**")
    st.write(f"Total clusters: **{len(st.session_state.get('clusters', {}))}**")

    # Downsampling controls
    def on_seed_change():
        st.session_state["random_state"] = np.random.RandomState(st.session_state["seed"])
        select_genomes()

    st.session_state.setdefault(
        "random_state",
        np.random.RandomState(st.session_state.get("seed", 0)),
    )
    st.number_input(
        "Random seed",
        min_value=0, max_value=2**32 - 1, value=42, step=1,
        key="seed", on_change=on_seed_change,
    )

    st.slider(
        "Max sequences per cluster to include",
        min_value=1,
        max_value=st.session_state["max_clustersize"],
        value=st.session_state["max_clustersize"],
        step=1,
        help="Clusters larger than this will be downsampled randomly",
        key="max_genomes",
        on_change=on_seed_change,
    )

    # Distance (sub)matrix
    st.session_state.setdefault("distance_matrix", None)
    st.session_state.setdefault("_last_distance_params", None)

    if st.button("Load distance matrix"):
        global_indices = st.session_state["global_indices"]
        D_full = st.session_state["D_full"]
        D = pipeline.extract_submatrix(D_full, global_indices)
        st.session_state["distance_matrix"] = D
        st.session_state["_last_distance_params"] = (
            st.session_state["seed"], st.session_state["max_genomes"]
        )
        st.session_state["_needs_distance_reload"] = False
        n_sel = len(global_indices)
        st.success(f"Distance submatrix loaded ({n_sel}×{n_sel}).")

    if st.session_state.get("_needs_distance_reload"):
        st.warning("Selection changed — click 'Load distance matrix' to refresh.")

    if st.session_state.get("distance_matrix") is not None:
        D = st.session_state["distance_matrix"]
        index2seq = st.session_state["index2seq"]
        genomes   = st.session_state["genomes"]
        labels = [
            f"{seq_id} [{genomes[seq_id]['cluster']}]" for seq_id in index2seq
        ]
        st.dataframe(pd.DataFrame(D, index=labels, columns=labels))

    # Local search optimisation
    st.header("Local Search Optimisation")

    if st.session_state.get("distance_matrix") is None:
        st.info("Load the distance matrix first to enable optimisation.")
    else:
        st.subheader("Initialise solution")

        init_method = st.radio(
            "Initialisation method", ["Random", "Medoid"], horizontal=True
        )
        selection_cost = st.number_input(
            "Cost for selecting a sequence",
            min_value=0.0, max_value=1.0, value=0.1, step=0.0001, format="%.5f",
        )
        scale_enabled = st.checkbox("Apply inter-cluster similarity penalty (scale)", value=True)
        scale = st.number_input(
            "Scale (inter-cluster penalty weight)",
            min_value=0.0, value=1e-5, step=1e-6, format="%.2e",
            disabled=not scale_enabled,
            help="Scaling factor applied to the inter-cluster similarity term. "
                 "Set to 0 to disable inter-cluster penalty entirely.",
        ) if scale_enabled else None
        if init_method == "Random":
            fraction = st.slider(
                "Fraction of sequences to select (at least one per cluster)",
                min_value=0.01, max_value=1.0, value=0.5, step=0.01,
            )

        avail_cores = os.cpu_count() or 1
        num_processes = st.slider(
            "Number of parallel processes for local search",
            min_value=1, max_value=avail_cores, value=1, step=1,
        )

        if st.button("Initialise solution"):
            D = st.session_state["distance_matrix"]
            cluster_assignments = np.array(
                st.session_state["selected_clusters"], dtype=np.int32
            )
            n = len(cluster_assignments)
            rng = st.session_state["random_state"]

            if num_processes > 1:
                SolClass = Solution_shm
            else:
                SolClass = Solution

            if init_method == "Random":
                sol = SolClass.generate_random_solution(
                    D, cluster_assignments,
                    selection_cost=selection_cost,
                    scale=scale,
                    max_fraction=fraction,
                    seed=rng,
                )
                st.success(f"Random solution: {int(np.sum(sol.selection))}/{n} selected.")
            else:
                sol = SolClass.generate_medoid_solution(
                    D, cluster_assignments,
                    selection_cost=selection_cost,
                    scale=scale,
                )
                st.success(f"Medoid solution: {int(np.sum(sol.selection))}/{n} selected.")

            objective = sol.objective[0] if isinstance(sol, Solution_shm) else sol.objective
            st.success(f"Initial objective: {objective:.6f}")
            st.session_state["solution"]              = sol
            st.session_state["num_processes"]         = num_processes
            st.session_state["_solution_initialized"] = True

        if st.session_state.get("_solution_initialized", False):
            st.divider()
            sol = st.session_state["solution"]

            col1, col2 = st.columns(2)
            with col1:
                max_iterations = st.number_input(
                    "Max iterations", min_value=1, value=1_000_000, step=1
                )
            with col2:
                max_runtime = st.number_input(
                    "Max runtime (seconds)", min_value=1, value=60, step=1
                )

            st.subheader("Move types")
            c1, c2, c3, c4 = st.columns(4)
            move_order = []
            if c1.checkbox("Add",        value=True): move_order.append("add")
            if c2.checkbox("Swap",       value=True): move_order.append("swap")
            if c3.checkbox("Doubleswap", value=True): move_order.append("doubleswap")
            if c4.checkbox("Remove",     value=True): move_order.append("remove")

            if not move_order:
                st.error("Select at least one move type.")
            else:
                st.write(f"Active moves: {', '.join(move_order)}")

            start_obj = sol.objective[0] if isinstance(sol, Solution_shm) else sol.objective
            st.session_state["starting_objective"] = start_obj

            if st.button("Run local search", disabled=not move_order):
                with st.spinner("Running local search…"):
                    t0 = time.time()
                    n_proc = st.session_state.get("num_processes", 1)
                    if n_proc > 1 and isinstance(sol, Solution_shm):
                        sol.local_search(
                            num_processes=n_proc,
                            max_iterations=max_iterations,
                            max_runtime=max_runtime,
                            move_order=move_order,
                            logging=True,
                            logging_frequency=100,
                            doubleswap_time_threshold=5.0,
                        )
                    else:
                        sol.local_search(
                            max_iterations=max_iterations,
                            max_runtime=max_runtime,
                            move_order=move_order,
                            logging=True,
                            logging_frequency=100,
                            doubleswap_time_threshold=5.0,
                        )
                    elapsed = time.time() - t0

                end_obj = sol.objective[0] if isinstance(sol, Solution_shm) else sol.objective
                st.session_state["solution"]           = sol
                st.session_state["ending_objective"]   = end_obj
                st.session_state["_solution_optimized"] = True
                st.success(f"Completed in {elapsed:.2f}s.")
                st.success(f"Objective: {st.session_state['starting_objective']:.6f} → {end_obj:.6f}")
                st.success(f"Selected: {int(np.sum(sol.selection))}/{len(sol.selection)}")

            # Cluster view
            if st.session_state.get("_solution_optimized", False):
                st.divider()
                st.subheader("Cluster view")

                sol         = st.session_state["solution"]
                all_genomes = st.session_state["genomes"]
                all_clusters = st.session_state["clusters"]
                index2seq   = st.session_state["index2seq"]
                included_ids = set(index2seq)
                selected_ids = set(
                    index2seq[idx] for idx in np.flatnonzero(sol.selection)
                )

                cluster_names = []
                cluster_mapping = {}
                for cluster, seq_ids in all_clusters.items():
                    sel = sum(1 for s in seq_ids if s in selected_ids)
                    tot = sum(1 for s in seq_ids if s in included_ids)
                    label = f"{cluster} (selected: {sel}/{tot})"
                    cluster_names.append(label)
                    cluster_mapping[label] = cluster
                cluster_names.sort(
                    key=lambda x: int(x.split("selected: ")[1].split("/")[0]),
                    reverse=True,
                )

                chosen_label   = st.selectbox("Select cluster", cluster_names)
                chosen_cluster = cluster_mapping[chosen_label]
                cluster_seqs   = all_clusters[chosen_cluster]

                st.write(f"**Cluster:** {chosen_cluster}")
                st.write(f"**Total sequences:** {len(cluster_seqs)}")
                st.write(f"**In selection pool:** {sum(1 for s in cluster_seqs if s in included_ids)}")
                st.write(f"**Selected by optimiser:** {sum(1 for s in cluster_seqs if s in selected_ids)}")

                data = []
                for seq_id in cluster_seqs:
                    if seq_id in selected_ids:
                        status, style, pri = "✅ Selected", "selected", 0
                    elif seq_id in included_ids:
                        status, style, pri = "⚪ Included (not selected)", "included", 1
                    else:
                        status, style, pri = "○ Not included", "excluded", 2
                    data.append({"Sequence ID": seq_id, "Status": status,
                                 "_style": style, "_priority": pri})
                data.sort(key=lambda x: x["_priority"])
                df = pd.DataFrame(data)

                def style_row(row):
                    s = df.loc[row.name, "_style"]
                    if s == "selected":
                        return ["font-weight: bold; color: #00AA00;"] * 2
                    elif s == "excluded":
                        return ["opacity: 0.4; color: #888888;"] * 2
                    return [""] * 2

                st.dataframe(
                    df[["Sequence ID", "Status"]].style.apply(style_row, axis=1),
                    use_container_width=True, height=400,
                )


def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False


if _in_streamlit():
    run_app()
