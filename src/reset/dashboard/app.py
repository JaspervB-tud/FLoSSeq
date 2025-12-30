import streamlit as st
from reset.dashboard import pipeline
import numpy as np
import os
import subprocess
import pandas as pd
import time

def run_app():
    st.title("Dashboard")

    def choose_folder_macos():
        """
        Opens a dialog to choose a folder on MacOS.
        """
        try:
            script = 'POSIX path of (choose folder with prompt "Select the data folder (contains sequences.fasta and clusters.tsv)")'
            out = subprocess.check_output(["osascript", "-e", script])
            return out.decode("utf-8").strip()
        except Exception:
            return None

    st.sidebar.header("Data Selection")
    if "data_folder" not in st.session_state:
        st.session_state["data_folder"] = ""
    pick = st.sidebar.button("Choose folder (macOS Finder)")
    if pick:
        folder = choose_folder_macos()
        if folder != st.session_state["data_folder"]:
            st.cache_data.clear()
        if folder:
            st.session_state["data_folder"] = folder
            st.sidebar.success(f"Selected folder: {folder}")
            st.rerun()
        else:
            st.sidebar.error("No folder selected.")
    data_folder = st.sidebar.text_input("Or enter data folder path", st.session_state["data_folder"])
    st.session_state["data_folder"] = data_folder

    if not data_folder:
        st.info("Select a folder, or paste a path containing sequences.fasta and clusters.tsv to proceed.")
        st.stop()
    seqpath = os.path.join(data_folder, "sequences.fasta")
    clustpath = os.path.join(data_folder, "clusters.tsv")
    if not os.path.isfile(seqpath):
        st.error(f"sequences.fasta not found in {data_folder}")
        st.stop()
    if not os.path.isfile(clustpath):
        st.error(f"clusters.tsv not found in {data_folder}")
        st.stop()

    # Functionality for downsampling genomes
    def select_genomes():
        """
        Select genomes according to current settings (max_genomes, random_state).
        Stores the selected genomes in session state.
        """
        print("Selecting genomes...")
        genomes = st.session_state["genomes"]
        max_genomes = st.session_state.get("max_genomes", 1)
        random_state = st.session_state.get("random_state", None)
        # Select genomes
        selected = pipeline.downsample(genomes, max_genomes=max_genomes, random_state=random_state)

        # Index selected genomes
        seq2index = {}
        index2seq = []
        for seq_id in selected:
            seq2index[seq_id] = len(index2seq)
            index2seq.append(seq_id)

        # Find selected clusters (as index)
        selected_clusters = []
        for seq in index2seq:
            cluster = genomes[seq]["cluster"]
            cluster_idx = st.session_state["cluster2index"][cluster]
            selected_clusters.append(cluster_idx)

        st.session_state["seq2index"] = seq2index
        st.session_state["index2seq"] = index2seq
        st.session_state["selected_genomes"] = selected
        st.session_state["selected_clusters"] = selected_clusters
        print(f"Finished selecting genomes (selected {len(index2seq)} in total).")
        # Check if distance matrix needs to be recomputed
        last_params = st.session_state.get("_last_distance_params", None)
        if last_params is not None:
            if last_params != (st.session_state["seed"], st.session_state["max_genomes"]):
                st.session_state["distance_matrix"] = None #invalidate distance matrix
                st.session_state["_needs_distance_recompute"] = True
            else:
                st.session_state["_needs_distance_recompute"] = False

    @st.cache_data(show_spinner=True)
    def load_data(sequences_path, clusters_path):
        """
        Load sequences and clusters from files.
        Also indexes the clusters for easy access.
        Caches the result to avoid reloading on every interaction.
        NOTE: Sequences will be indexed later (during downsampling).
        """
        # Fetch genomes and clusters
        genomes = pipeline.read_fasta(sequences_path)
        genomes = pipeline.determine_clusters(clusters_path, genomes)

        # Build index
        clusters = {} #cluster_name -> list of sequence IDs
        cluster2index = {}
        index2cluster = []
        for seq_id, g in genomes.items():
            cluster = g["cluster"]
            if cluster not in clusters:
                cluster2index[cluster] = len(index2cluster)
                index2cluster.append(cluster)
                clusters[cluster] = []
            clusters[cluster].append(seq_id)

        st.session_state["genomes"] = genomes
        st.session_state["clusters"] = clusters
        st.session_state["max_clustersize"] = max(len(v) for v in clusters.values())
        st.session_state["cluster2index"] = cluster2index
        st.session_state["index2cluster"] = index2cluster
        select_genomes() # initial selection

    need_reload = (
        "genomes" not in st.session_state or
        st.session_state.get("_loaded_folder", "") != data_folder
    )
    if need_reload:
        load_data(seqpath, clustpath)
        st.session_state["_loaded_folder"] = data_folder
    st.write(f"Total sequences loaded: {len(st.session_state.get('genomes', {}))}")
    st.write(f"Total clusters loaded: {len(st.session_state.get('clusters', {}))}") # Show the clusters

    # Seed for controlling randomness
    def on_seed_change():
        """
        Update the random state when the seed changes.
        NOTE: This will also re-select genomes according to the new seed if a selection
        was already made previously.
        """
        print(f"Setting seed to {st.session_state['seed']}...")
        st.session_state["random_state"] = np.random.RandomState(st.session_state["seed"])
        print("Finished setting seed.")
        select_genomes()

    st.session_state.setdefault(
        "random_state",
        np.random.RandomState(st.session_state.get("seed", 0)),
    )
    st.number_input(
        "Random seed",
        min_value=0,
        max_value=2**32 - 1,
        value=42,
        step=1,
        key="seed",
        on_change=on_seed_change,
    )

    # Maximum number of genomes per cluster
    def on_maxclustersize_change():
        on_seed_change() #re-initialize random state

    st.slider(
        "Max genomes per cluster to include",
        min_value=1,
        max_value=st.session_state["max_clustersize"],
        value=1, #default value
        step=1,
        help="Clusters larger than this will be downsampled",
        key="max_genomes",
        on_change=on_maxclustersize_change,
    )

    # Number of cores
    def on_cores_change():
        print(f"Setting cores to {st.session_state['cores']}...")
        print("Finished setting cores.")

    avail_cores = os.cpu_count() or 1
    st.slider(
        "Number of CPU cores to use",
        min_value=1,
        max_value=avail_cores,
        value=min(1, avail_cores),
        key="cores", #default value
        on_change=on_cores_change,
        step=1,
    )

    # Distance matrix calculation
    st.session_state.setdefault("distance_matrix", None)
    st.session_state.setdefault("_last_distance_params", None)
    if st.button("Calculate distances"):
        print("Computing distance matrix...")
        start_time = time.time()
        D = pipeline.compute_distances(
            st.session_state["selected_genomes"],
            st.session_state["index2seq"],
            st.session_state["seq2index"],
            cores=st.session_state["cores"],
        )
        end_time = time.time()
        print(f"Finished computing distance matrix. Took {end_time-start_time:.2f}s.")
        st.session_state["distance_matrix"] = D
        st.success(f"Distance matrix computed ({end_time-start_time:.2f}s) for {len(st.session_state['index2seq'])} genomes.")
        # Set up a state value that is used to check if the seed and number of genomes changed
        st.session_state["_last_distance_params"] = (
            st.session_state["seed"],
            st.session_state["max_genomes"],
        )
        st.session_state["_needs_distance_recompute"] = False

    # Display distance matrix if it exists
    if st.session_state.get("distance_matrix") is not None:
        D = st.session_state["distance_matrix"]
        labels = [
            f"{seq_id} [{st.session_state['selected_genomes'][seq_id]['cluster']}]" for seq_id in st.session_state["index2seq"]
        ]
        df = pd.DataFrame(D, index=labels, columns=labels)
        st.dataframe(df)

    # After the button, show warning if parameters changed
    if st.session_state.get("_needs_distance_recompute"):
        st.error("Parameters changed: distance matrix invalidated. Click 'Calculate distances' to recompute.")

    # --- Local Search Optimization ---
    st.header("Local Search Optimization (LoSeR)")

    if st.session_state.get("distance_matrix") is None:
        st.info("Compute the distance matrix first to enable Local Search Optimization.")
    else:
        st.subheader("Initialize Solution")

        init_method = st.radio(
            "Select initialization method",
            ["Random", "Centroid"],
            horizontal=True
        )
        selection_cost = st.number_input(
            "Cost for selecting a sequence",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.0001,
            format="%.5f",  #show 5 decimal places
        )
        # Set parameters if initialization method is Random
        if init_method == "Random":
            fraction = st.slider(
                "Fraction of sequences to select (will select at least one per cluster)",
                min_value=0.01,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )
        
        if st.button("Initialize Solution"):
            from loser.solution import Solution

            D = st.session_state["distance_matrix"]
            cluster_assignments = np.array(st.session_state["selected_clusters"], dtype=np.int32)
            n = len(cluster_assignments)

            rng = st.session_state["random_state"]
            if init_method == "Random": #Random
                sol = Solution.generate_random_solution(
                    D,
                    cluster_assignments,
                    selection_cost=selection_cost,
                    max_fraction=fraction,
                    seed=rng,
                )
                st.success(f"Random solution initialized with {np.sum(sol.selection)}/{n} sequences selected.")
                st.session_state["solution"] = sol
                st.session_state["_solution_initialized"] = True
            else: #Centroid
                sol = Solution.generate_centroid_solution(
                    D,
                    cluster_assignments,
                    selection_cost=selection_cost,
                )
                st.success(f"Centroid solution initialized with {np.sum(sol.selection)}/{n} sequences selected.")
                st.session_state["solution"] = sol
                st.session_state["_solution_initialized"] = True
            st.success(f"**Initial objective value:** {sol.objective:.4f}")

        if st.session_state.get("_solution_initialized", False):
            st.divider()

            sol = st.session_state["solution"] 

            col1, col2 = st.columns(2)
            with col1:
                max_iterations = st.number_input(
                    "Max iterations for local search",
                    min_value=1,
                    #max_value=10_000,
                    value=1000,
                    step=1,
                )
            with col2:
                max_runtime = st.number_input(
                    "Max runtime (seconds) for local search",
                    min_value=1,
                    #max_value=10_000,
                    value=60,
                    step=1,
                )
            st.subheader("Move Types")
            col1, col2, col3, col4 = st.columns(4)
            move_add = col1.checkbox("Add", value=True)
            move_remove = col2.checkbox("Remove", value=True)
            move_swap = col3.checkbox("Swap", value=True)
            move_doubleswap = col4.checkbox("Double swap", value=True)

            # Build move order list
            move_order = []
            if move_add:
                move_order.append("add")
            if move_swap:
                move_order.append("swap")
            if move_doubleswap:
                move_order.append("doubleswap")
            if move_remove:
                move_order.append("remove")

            # Validate at least one move type is selected
            if len(move_order) == 0:
                st.error("Select at least one move type to perform local search!!!")
            else:
                st.write(f"Selected move types: {', '.join(move_order)}")

            st.session_state["starting_objective"] = sol.objective
            if st.button("Run Local Search!", disabled=(len(move_order) == 0)):
                with st.spinner("Running local search..."):
                    start_time = time.time()
                    if st.session_state.get("cores", 1) == 1:
                        sol.local_search_sp(
                            max_iterations = max_iterations,
                            max_runtime = max_runtime,
                            move_order = move_order,
                            logging = True,
                            logging_frequency = 100,
                            doubleswap_time_threshold = 5.0
                        )
                    else:
                        sol.local_search_mp(
                            max_iterations = max_iterations,
                            max_runtime = max_runtime,
                            num_cores = st.session_state["cores"],
                            move_order = move_order,
                            logging = True,
                            logging_frequency = 100,
                            doubleswap_time_threshold = 5.0
                        )
                    end_time = time.time()

                st.session_state["solution"] = sol
                st.session_state["ending_objective"] = sol.objective
                st.session_state["_solution_optimized"] = True
                st.success(f"Local search completed in {end_time-start_time:.2f}s.")
                st.success(f"**Starting objective value:** {st.session_state['starting_objective']:.4f}")
                st.success(f"**Final objective value:** {st.session_state['ending_objective']:.4f}")
                st.success(f"**Sequences selected:** {np.sum(sol.selection)}/{len(sol.selection)}")

            if st.session_state.get("_solution_optimized", False):
                st.divider()
                st.subheader("Cluster View")

                sol = st.session_state["solution"]

                # Get all genomes and cluster info
                all_genomes = st.session_state["genomes"]
                all_clusters = st.session_state["clusters"]
                selected_genomes = st.session_state["selected_genomes"]
                index2seq = st.session_state["index2seq"]
                seq2index = st.session_state["seq2index"]

                # Get selected sequence IDs
                selected_indices = np.where(sol.selection)[0]
                selected_seq_ids = set(index2seq[idx] for idx in selected_indices)
                included_seq_ids = set(index2seq)

                # Cluster selector
                cluster_names = []
                cluster_mapping = {}
                for cluster in all_clusters.keys():
                    sel = 0
                    tot = 0
                    for seq_id in all_clusters[cluster]:
                        if seq_id in included_seq_ids:
                            tot += 1
                            if seq_id in selected_seq_ids:
                                sel += 1
                    display_name = f"{cluster} (selected: {sel}/{tot})"
                    cluster_names.append(display_name)
                    cluster_mapping[display_name] = cluster
                cluster_names = sorted(cluster_names, key = lambda x: int(x.split(" (selected: ")[1].split("/")[0]), reverse=True) #sort by selected count

                selected_cluster = st.selectbox(
                    "Select cluster to view",
                    cluster_names,
                    key="cluster_view_selector"
                )
                selected_cluster = cluster_mapping[selected_cluster] #map back to actual cluster name

                # Get sequences in this cluster
                cluster_seqs = all_clusters[selected_cluster]

                st.write(f"**Cluster:** {selected_cluster}")
                st.write(f"**Total sequences in cluster:** {len(cluster_seqs)}")
                st.write(f"**Included in selection pool:** {sum(1 for seq in cluster_seqs if seq in included_seq_ids)}")
                st.write(f"**Selected by optimizer:** {sum(1 for seq in cluster_seqs if seq in selected_seq_ids)}")
                
                # Build styled dataframe
                data = []
                for seq_id in cluster_seqs:
                    is_included = seq_id in included_seq_ids
                    is_selected = seq_id in selected_seq_ids
                    
                    if is_selected:
                        status = "✅ Selected"
                        style = "included_selected"
                        priority = 0 # Highest priority for sorting
                    elif is_included:
                        status = "⚪ Included (not selected)"
                        style = "included_not_selected"
                        priority = 1 # Second highest priority
                    else:
                        status = "○ Not included"
                        style = "not_included"
                        priority = 2 # Lowest priority
                    
                    data.append({
                        "Sequence ID": seq_id,
                        "Status": status,
                        "_style": style,
                        "_priority": priority,
                    })
                
                data.sort(key = lambda x: x["_priority"])
                df = pd.DataFrame(data)
                
                # Display with custom styling using HTML
                def style_row(row):
                    style_val = df.loc[row.name, "_style"]
                    if style_val == "included_selected":
                        return ['font-weight: bold; color: #00AA00;'] * 2
                    elif style_val == "included_not_selected":
                        return [''] * 2
                    else:  # not_included
                        return ['opacity: 0.4; color: #888888;'] * 2
                
                styled_df = df[["Sequence ID", "Status"]].style.apply(style_row, axis=1)
                st.dataframe(styled_df, width="stretch", height=400)     

def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

if _in_streamlit():
    run_app()