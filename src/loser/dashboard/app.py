import streamlit as st
from loser.dashboard import pipeline
import numpy as np
import os
import subprocess
import pandas as pd
import time

def run_app():
    st.title("LoSeR Dashboard")

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

        st.session_state["seq2index"] = seq2index
        st.session_state["index2seq"] = index2seq
        st.session_state["selected_genomes"] = selected
        print(f"Finished selecting genomes (selected {len(index2seq)} in total).")

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
        None,
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
    cores = st.slider(
        "Number of CPU cores to use",
        min_value=1,
        max_value=avail_cores,
        value=min(1, avail_cores),
        key="cores",
        on_change=on_cores_change,
        step=1,
    )
    """
    # Downsample and calculate distances
    if st.button("Downsample and calculate distances"):
        start_time = time.time()
        D, clusters, id2index, index2id = pipeline.downsample_and_compute_distances(genomes, max_genomes=max_genomes, cores=cores)
        end_time = time.time()
        st.success(f"Distance matrix computed ({end_time-start_time:.2f}s) for {len(index2id)} genomes across {len(set(clusters))} clusters.")
        labels = [
            f"{index2id[i]} [{genomes[index2id[i]]['cluster']}]" for i in range(len(index2id))
        ]
        df = pd.DataFrame(D, index=labels, columns=labels)
        st.dataframe(df)
    """

def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

if _in_streamlit():
    run_app()

"""
Look at this later

selected_cluster = st.selectbox("Select cluster", sorted(cluster_counts.keys()))
info = cluster_counts[selected_cluster]
st.subheader(f"Cluster: {selected_cluster}")
st.write(f"Sequences in cluster: {info['count']}")

# Show example IDs in a scrollable box
ids_text = "\n".join(info["example_ids"])
st.text_area("Example IDs", ids_text, height=220, key=f"examples_{selected_cluster}", disabled=True)
st.code("\n".join(info["example_ids"]), language="text")

# Show detailed sequence selection
seq_id = st.selectbox("Inspect sequence ID", info["example_ids"])
seq_record = genomes[seq_id]
st.write(f"Length: {len(seq_record['sequence'])}")
if st.checkbox("Show sequence"):
    st.code(seq_record["sequence"][:500] + ("..." if len(seq_record["sequence"]) > 500 else ""), language="text")

# Optional: pairwise similarity (simple Jaccard via sourmash)
if st.button("Compute intra-cluster minhash similarity (subset)"):
    subset_ids = info["example_ids"]
    mhs = [genomes[sid]["minhash"] for sid in subset_ids]
    sims = []
    for i in range(len(mhs)):
        row = []
        for j in range(len(mhs)):
            if i == j:
                row.append(1.0)
            else:
                row.append(mhs[i].similarity(mhs[j]))
        sims.append(row)
    st.write("Similarity matrix (example IDs order):")
    st.dataframe(sims)

# Placeholder for integrating Solution
if st.button("Initialize Solution (demo)"):
    from ..solution import Solution
    sol = Solution()  # adapt to required constructor
    st.success("Solution object created")
"""