# Preface
This README documents, end-to-end, the processing steps used to obtain the results reported in "ReSeT: a taxonomy-aware reference genome selection tool", which can be found on [bioRxiv](https://doi.org/10.64898/2026.06.17.732946). In all steps outlined below, we differentiate between the processing for SARS-CoV-2 (both USA and China) and IAV.

# Fetching/downloading data
All data used in this manuscript was downloaded from [GISAID](https://gisaid.org).

## SARS-CoV-2
We downloaded the complete SARS-CoV-2 genome database including metadata from GISAID on 16/11/2025 using the bulk download functionality. This downloads both a `metadata.tsv` and `sequences.fasta` (in compressed format). We assume that these have been uncompressed, and will refer to them as `data/SARS-CoV-2/metadata.tsv` and `data/SARS-CoV-2/sequences.fasta` in what follows.

## IAV
We downloaded IAV genomes (HA + NA segments from H1N1pdm09 and H3N2 with human host) including metadata from GISAID on 20/11/2025 from the web-interface, downloading up to 3000 genomes per download. The genomes and metadata (in .xls format) were stored as `{H1N1pdm09,H3N2}_DD-MM-YYYY_DD-MM-YYYY_down20-11-2025.{fasta,xls}` in `data/IAV`. The dates were either of the following:

- 01-01-2023_30-06-2023
- 01-07-2023_31-12-2023
- 01-01-2024_30-06-2024
- 01-07-2024_31-12-2024
- 01-01-2025_31-03-2025

# Pre-processing
## SARS-CoV-2
In what follows we will describe the overall workflow for end-to-end processing. However, since the processing for both the USA and China datasets (see below) is more or less identical, we will keep it general with the understanding that this should be run for both USA and China. For simplicity, we set the bash variable `COUNTRY` to either `China` or `USA`.

### Filtering
The first step is pre-processing the genomes by selecting high-quality genomes that have an associated Pango lineage, restricting to a single collection-date window, and filtering on geographic origin. For this, we used the `manuscript/sarscov2/filter.py` script by running it as follows:

```bash
NCORES=8

# Create output directory
mkdir -p data/SARS-CoV-2/${COUNTRY}

python manuscript/sarscov2/filter.py \
    --input_fasta data/SARS-CoV-2/sequences.fasta \
    --input_metadata data/SARS-CoV-2/metadata.tsv \
    --output_folder data/SARS-CoV-2/${COUNTRY} \
    --ref_start 2022-01-01 \
    --ref_end 2023-12-31 \
    --sim_start 2024-01-01 \
    --sim_end 2024-06-30 \
    --location ${COUNTRY} \
    --min_length 29000 \
    --max_n_content 0.001 \
    --max_non_acgt_content 0.01 \
    --num_workers $NCORES
```

This filters out low quality genomes (length < 29,000bp, N-content > 0.1%, non-ACGT-content > 1%, not marked as complete by GISAID), and retains only sequences whose host is listed as human. Resulting genomes and metadata are stored in `data/SARS-CoV-2/{China,USA}`, which hosts two subfolders: `Reference` (candidate reference genomes collected between 2022-01-01 and 2023-12-31) and `Simulation` (candidate simulation genomes collected between 2024-01-01 and 2024-06-30).

### Downsampling (USA)
As the USA reference genome collection exceeds 200,000 genomes, we downsampled it to a feasible scale with the `manuscript/sarscov2/downsample.py` script:

```bash
python manuscript/sarscov2/downsample.py \
    --input_fasta data/SARS-CoV-2/USA/Reference/sequences.fasta \
    --input_metadata data/SARS-CoV-2/USA/Reference/metadata.tsv \
    --output_folder data/SARS-CoV-2/USA/Reference_downsampled \
    --ref_start 2022-01-01 \
    --ref_end 2023-12-31 \
    --monthly_cap 3500 \
    --seed 1234
```

This randomly selects up to 3,500 sequences per month, allocated across lineages proportionally to their monthly representation guaranteeing at least one sequence per lineage. The resulting `sequences.fasta` and `metadata.tsv` are stored in `data/SARS-CoV-2/USA/Reference_downsampled`, along with a `sequence_mapping.txt` file mapping every sequence ID to a unique integer counting up from 0. This step is only applied to the USA dataset (and not China), resulting in a reference set containing 85,760 genomes.

**NOTE**: although the downsampled data is stored in a new folder (`Reference_downsampled`), we simply assume in what follows that they live in `data/SARS-CoV-2/USA/Reference` to keep the description applicable to both USA and China. To reproduce our results, however, `Reference` should be replaced with `Reference_downsampled` for the USA dataset.

### Splitting into Train/Test
As every selection method (including ReSeT) relies on tunable input parameters, we ran a grid search to find good parameters, using a temporal train/test split of the reference genomes (distinct from the reference/simulation split above). This was done with `manuscript/sarscov2/split.py`:

```bash
python manuscript/sarscov2/split.py \
    --input_fasta data/SARS-CoV-2/${COUNTRY}/Reference/sequences.fasta \
    --input_metadata data/SARS-CoV-2/${COUNTRY}/Reference/metadata.tsv \
    --output_folder data/SARS-CoV-2/${COUNTRY}/Reference \
    --train_start 2022-01-01 \
    --train_end 2022-12-31 \
    --test_start 2023-01-01 \
    --test_end 2023-06-30 \
    --max_genomes_train 200 \
    --max_genomes_test 100 \
    --seed 12345
```

This creates a Train/Test split with up to 200 and 100 randomly selected genomes per lineage, respectively (only lineages present in the training window are retained). Resulting fasta/metadata files are written both as one multi-FASTA per split (`data/SARS-CoV-2/${COUNTRY}/Reference/{Train,Test}`), and for the Train split only as one fasta/metadata pair per lineage (`data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}`). Training genomes are collected from 2022 and tested against genomes from the first half of 2023, so there is no leakage of information from the train set to the test set, nor from the simulation genomes (which start at 2024-01-01).

## IAV
Unlike the SARS-CoV-2 datasets, we only have a single dataset for IAV meaning that we do not have a `${COUNTRY}` subfolder.

### Filtering, Downsampling and Splitting
For IAV, we combined all scripts used to pre-process SARS-CoV-2 genomes into a single script (`preprocess.py`) that performs those functions simultaneously. Additionally, this script accounts for the fact that the data is initially stored in separate fasta and .xls files. We run the script as follows:

```bash
python -u manuscript/iav/preprocess.py \
    --input data/IAV \
    --output data/IAV \
    --reference_start_date "2024-01-01" \
    --simulation_start_date "2025-01-01" \
    --test_start_date "2024-10-01" \
    --max_monthly_cap 2000 \
    --seed 1234
```

This creates the subfolders `data/IAV/{Reference,Simulation}` as well as `data/IAV/Reference/{Train,Test}` with subfolders for every clade found in the metadata. The genome files are stored in three distinct ways (both aggregated, and per lineage):

1. `sequences_HA.fasta` and `sequences_NA.fasta`: separate files for the segments
2. `sequences_concatenated.fasta`: a single file with NA concatenated after HA segment
3. `sequences_concatenated_ns.fasta`: same as 2. but with 100 'N' characters inbetween the segments

These different versions are used by different selection methods, depending on how they process the genomes. Similar to the SARS-CoV-2 script, this also downsamples to at most 2,000 genomes per month, and imposes a limit of 1000 genomes per clade in the train set, and 500 genomes per clade in the test set (for parameter tuning), selecting genomes from the first nine months of 2024 as training data and the final three months as test data.

**NOTE**: Unlike SARS-CoV-2, we did not impose a minimum sequence length, and we hardcoded the ambiguous character limits in the code.

# Parameter tuning
## SARS-CoV-2
Since the distance estimation tools we use in this work (mash, sourmash) store distances in plain text files, we first run a small python script on the aggregated (i.e. not per lineage) sequence collections (for USA and China individually) to create a re-labeling of the sequences in order to save storage space when saving distance estimates:

```python
from Bio import SeqIO

if __name__ == "__main__":
    for location in ["USA", "China"]:
        sequence_path = f"data/SARS-CoV-2/{location}/Reference/Train/sequences.fasta"
        mapping = {}

        records = list(SeqIO.parse(sequence_path, "fasta"))
        for idx, record in enumerate(records):
            mapping[record.id] = idx
            record.id = str(idx)
            record.description = "" #clear description

        SeqIO.write(records, f"data/SARS-CoV-2/{location}/Reference/Train/sequences_remapped.fasta", "fasta")
        with open(f"data/SARS-CoV-2/{location}/Reference/Train/sequence_mapping.txt", "w") as f_out:
            for record_id, new_id in mapping.items():
                f_out.write(f"{new_id}\t{record_id}\n")
```

This stores the full sequences in a new fasta file along with a mapping file for finding the correspondence between old and new sequence identifiers.

### (Dis-)similarity estimations
Because the choice of distance estimation method and its resolution is itself a tunable parameter, we tune over mash with sketch sizes of 5,000 and 10,000, and sourmash (branchwater plugin) with scaling factors of 6 and 3 (these correspond, respectively, to roughly the same effective resolution as the two mash sketch sizes), always using `k=31`. We run both mash and sourmash on a per-lineage basis as well as overall.

#### mash
Per-lineage processing:

```bash
NCORES=16

for SKETCHSIZE in 5000 10000; do
    # Sketch
    mash sketch -p ${NCORES} -s ${SKETCHSIZE} -S 12345 -k 31 -i \
        -o data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/mash_sketch_s${SKETCHSIZE} \
        data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/sequences.fasta

    # Distance estimations
    mash triangle -p ${NCORES} \
        data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/mash_sketch_s${SKETCHSIZE}.msh > \
        data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/mash_triangle_s${SKETCHSIZE}.dist
done
```

Overall processing (using the remapped multi-fasta):

```bash
NCORES=16

for SKETCHSIZE in 5000 10000; do
    # Sketch
    mash sketch -p ${NCORES} -s ${SKETCHSIZE} -S 12345 -k 31 -i \
        -o data/SARS-CoV-2/${COUNTRY}/Reference/Train/mash_sketch_s${SKETCHSIZE} \
        data/SARS-CoV-2/${COUNTRY}/Reference/Train/sequences_remapped.fasta

    # Distance estimations
    mash triangle -p ${NCORES} \
        data/SARS-CoV-2/${COUNTRY}/Reference/Train/mash_sketch_s${SKETCHSIZE}.msh > \
        data/SARS-CoV-2/${COUNTRY}/Reference/Train/mash_triangle_s${SKETCHSIZE}.dist
done
```

This produces a sketch file and distance file for both sketch sizes, on both a per-lineage and overall basis.

#### sourmash
We run sourmash (branchwater plug-in) similarly, first per lineage (this simultaneously produces Jaccard and cosine similarities):

```bash
NCORES=16

for SCALE in 6 3; do
    # Create manifest file required for branchwater
    printf "name,genome_filename,protein_filename\nx,%s,\n" data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/sequences.fasta > data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/manifest.csv

    sourmash scripts manysketch -o data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/sourmash_sketch_s${SCALE}.zip \
        -p "k=31,scaled=${SCALE},abund,dna,seed=12345" \
        -c ${NCORES} --singleton -q \
        data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/manifest.csv

    sourmash scripts pairwise -o data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/sourmash_distance_s${SCALE}.csv \
        -k 31 -s ${SCALE} -m DNA -c ${NCORES} -t 0 --calc-abund-stats -q \
        data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/sourmash_sketch_s${SCALE}.zip
done
```

and afterwards overall (same substitution of `sequences_remapped.fasta` and omittance of `${LINEAGE}` as for mash). This produces `sourmash_distance_s{6,3}.csv` files containing both Jaccard and cosine similarities for every sequence pair.

**NOTE**: sourmash's `scaled` parameter is a down-sampling factor, meaning that smaller values give higher resolution (in contrast to sketch sizes).

### Selection
#### Medoid
Medoid selections are obtained with `manuscript/sarscov2/medoid.py`:

```bash
# mash-based
python manuscript/sarscov2/medoid.py \
    --input_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/mash_triangle_s${SKETCHSIZE}.dist \
    --output_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/medoid_mash_s${SKETCHSIZE}.txt

# sourmash-based (jaccard)
python manuscript/sarscov2/medoid.py \
    --input_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/sourmash_distance_s${SCALE}.csv \
    --sourmash_jaccard \
    --output_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/medoid_sourmash-jaccard_s${SCALE}.txt

# sourmash-based (cosine)
python manuscript/sarscov2/medoid.py \
    --input_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/sourmash_distance_s${SCALE}.csv \
    --sourmash_cosine \
    --output_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/medoid_sourmash-cosine_s${SCALE}.txt
```

This generates the medoid selection for every lineage individually, using the provided distance estimation metric and parameters.

#### Hierarchical clustering
[Prior work](https://doi.org/10.1186/s12864-026-12874-w) found complete-linkage clustering to work best for reference genome selection. Because we want to consider a range of parameter settings, we include both fixed similarity thresholds (99%, 99.9%, 99.99%, 99.999%) and percentile-based (dynamically determined per lineage) similarity thresholds (1st, 25th, 50th, 75th, 90th percentile of the observed non-zero pairwise distances). These are obtained with `manuscript/sarscov2/hierarchical.py`:

```bash
# Fixed thresholds
for THRESHOLD in 0.99 0.999 0.9999 0.99999; do
    python manuscript/sarscov2/hierarchical.py \
        --input_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/mash_triangle_s${SKETCHSIZE}.dist \
        --output_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/hierarchical_mash_s${SKETCHSIZE}_t${THRESHOLD}.txt \
        --threshold ${THRESHOLD}

done

# Dynamic thresholds
for PERCENTILE in 1 25 50 75 90; do
    python manuscript/sarscov2/hierarchical.py \
        --input_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/mash_triangle_s${SKETCHSIZE}.dist \
        --output_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/hierarchical_mash_s${SKETCHSIZE}_p${PERCENTILE}.txt \
        --dynamic --threshold ${PERCENTILE}

done
# (repeat both with --sourmash_jaccard / --sourmash_cosine against sourmash_distance_s${SCALE}.csv)   
```

This clusters genomes with complete linkage at the provided distance thresholds (converting similarity to dissimilarity), and, per cluster, selects the cluster's medoid as its representative.

#### VSEARCH
VSEARCH internally computes exact edit distances to perform clustering, so its dynamic percentile thresholds must first be translated into concrete distance cut-offs with `manuscript/sarscov2/thresholds.py`:

```bash
for PERCENTILE in 1 25 50 75 90; do
    python manuscript/sarscov2/thresholds.py \
        --input_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/mash_triangle_s${SKETCHSIZE}.dist \
        --output_path data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/${PERCENTILE}_mash_s${SKETCHSIZE}.txt \
        --threshold ${PERCENTILE}
    # (repeat with --sourmash_jaccard / --sourmash_cosine against sourmash_distance_s${SCALE}.csv)  
done
```

VSEARCH is then run with fixed thresholds directly:

```bash
NCORES=16

for THRESHOLD in 0.99 0.999 0.9999 0.99999; do
    vsearch --cluster_fast data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/sequences.fasta \
        --centroids data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/vsearch_t${THRESHOLD}.fasta \
        --id ${THRESHOLD} --iddef 0 \
        --maxaccepts 1 --maxrejects 8 \
        --wordlength 10 \
        --threads ${NCORES}
done
```

and with dynamic thresholds:

```bash
NCORES=16

for PERCENTILE in 1 25 50 75 90; do
    THRESHOLD=$(cat "data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/${PERCENTILE}_mash_s${SKETCHSIZE}.txt")

    vsearch --cluster_fast data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/sequences.fasta \
        --centroids data/SARS-CoV-2/${COUNTRY}/Reference/Train/${LINEAGE}/vsearch_mash_s${SKETCHSIZE}_p${PERCENTILE}.fasta \
        --id ${THRESHOLD} --iddef 0 \
        --maxaccepts 1 --maxrejects 8 \
        --wordlength 10 \
        --threads ${NCORES}
    # (repeat for sourmash-jaccard and sourmash-cosine)
done
```

#### ReSeT
We ran ReSeT using a range of selection costs and scaling values in order to determine a reasonable default, as well as produce results for our manuscript. The first step is to convert the metadata file provided by GISAID to a `clusters.csv` file that can be used by ReSeT:

```python
ID_COL = 0
LIN_COL = 13

location = "USA" #set to China or USA

# Read seqid -> lineage
seq2lin = {}
with open(f"data/SARS-CoV-2/{location}/Reference/Train/metadata.tsv", "r") as f_in:
    next(f_in) #skip header

    for line in f_in:
        parts = line.strip().split("\t")
        seq2lin[parts[ID_COL]] = parts[LIN_COL]

# Read index -> seqid
mapping = {}
with open(f"data/SARS-CoV-2/{location}/Reference/Train/sequence_mapping.txt", "r") as f_in:
    for line in f_in:
        parts = line.strip().split("\t")
        mapping[int(parts[0])] = parts[1]

# Write clusters.csv
with open(f"data/SARS-CoV-2/{location}/Reference/Train/clusters.csv", "w") as f_out:
    for idx in sorted(mapping):
        seq_id = mapping[idx]
        lineage = seq2lin[seq_id]
        f_out.write(f"{seq_id},{lineage}\n")
```
With this file (and the `sequence_mapping.txt` produced before), ReSeT was ran:

```bash
NCORES=8

for COST in 0.000000 0.000001 0.000010 0.000100 0.001000 0.010000 0.100000 1.000000; do
    for SCALE in 0.00000 0.00001 0.00010 0.00100 0.01000 0.10000 1.00000; do
        # mash-based
        python -m reset.solution \
            --clusters "data/SARS-CoV-2/${COUNTRY}/Reference/Train/clusters.csv" \
            --distances "data/SARS-CoV-2/${COUNTRY}/Reference/Train/mash_triangle_s${SKETCHSIZE}.dist" \
            --distance_format mash \
            --scale ${SCALE} \
            --selection_cost ${COST} \
            --seed 12345 \
            --max_fraction 0.5 \
            --max_iterations 1000000 \
            --max_runtime 432000 \
            --doubleswap_time_threshold 300 \
            --num_processes 8 \
            --output "data/SARS-CoV-2/${COUNTRY}/Reference/Train/reset_mash_s${SKETCHSIZE}_cost-${COST}_scale-${SCALE}.txt"

        # sourmash-based (jaccard)
        python -m reset.solution \
            --clusters "data/SARS-CoV-2/${COUNTRY}/Reference/Train/clusters.csv" \
            --distances "data/SARS-CoV-2/${COUNTRY}/Reference/Train/sourmash_distance_s${SOURMASH_SCALE}.csv" \
            --distance_format sourmash_jaccard \
            --scale ${SCALE} \
            --selection_cost ${COST} \
            --seed 12345 \
            --max_fraction 0.5 \
            --max_iterations 1000000 \
            --max_runtime 432000 \
            --doubleswap_time_threshold 300 \
            --num_processes 8 \
            --output "data/SARS-CoV-2/${COUNTRY}/Reference/Train/reset_sourmash-jaccard_s${SOURMASH_SCALE}_cost-${COST}_scale-${SCALE}.txt"

        # sourmash-based (cosine)
        python -m reset.solution \
            --clusters "data/SARS-CoV-2/${COUNTRY}/Reference/Train/clusters.csv" \
            --distances "data/SARS-CoV-2/${COUNTRY}/Reference/Train/sourmash_distance_s${SOURMASH_SCALE}.csv" \
            --distance_format sourmash_cosine \
            --scale ${SCALE} \
            --selection_cost ${COST} \
            --seed 12345 \
            --max_fraction 0.5 \
            --max_iterations 1000000 \
            --max_runtime 432000 \
            --doubleswap_time_threshold 300 \
            --num_processes 8 \
            --output "data/SARS-CoV-2/${COUNTRY}/Reference/Train/reset_sourmash-cosine_s${SOURMASH_SCALE}_cost-${COST}_scale-${SCALE}.txt"
    done
done
```

Here `--scale` is ReSeT's inter-taxon scale parameter, referred to as $\lamba$ in the manuscript (not to be confused with sourmash's `scaled` factor). This produces a plain textfile with the selected genomes for every combination of distance estimator (including parameters), cost and $\lambda$.

### Creating indices
For all datasets we followed the VLQ pipeline in combination with kallisto v0.44.0 to estimate taxon abundances. Kallisto requires inputting a (multi-)fasta file containing all of the reference genomes which we create for every selection using the `gather_tuning_genomes.py` script:

```bash
python manuscript/sarscov2/gather_tuning_genomes.py \
    --metadata data/SARS-CoV-2/${COUNTRY}/Reference/Train/metadata.tsv \
    --fasta data/SARS-CoV-2/${COUNTRY}/Reference/Train/sequences.fasta \
    --input_dir data/SARS-CoV-2/${COUNTRY}/Reference/Train \
    --output_dir data/SARS-CoV-2/${COUNTRY}/Reference/Train/selections
```

This creates a folder called `selections` and makes a fasta file per selection (for every parameterization) containing the selected genomes, and saves it in `data/SARS-CoV-2/${COUNTRY}/Reference/Train/selections`.

We then build the kallisto indices:
```bash
mkdir -p data/SARS-CoV-2/${COUNTRY}/Reference/Train/indices
kallisto index -i data/SARS-CoV-2/${COUNTRY}/Reference/Train/indices/${METHOD_AND_PARAMETERS}.idx \
    data/SARS-CoV-2/${COUNTRY}/Reference/Train/selections/${METHOD_AND_PARAMETERS}.fasta
```

### Simulating samples
We build difficulty-stratified samples, where difficulty relates to the taxonomic ambiguity (i.e. the degree to which lineages are entangled). To this end, we first calculate pairwise distances between all genomes in the `Test` folder based on exact alignment distances to not bias towards any of the distance estimation methods used. The distances are computed by first creating a Multiple Sequence Alignment (MSA) using `mafft`:

```bash
mafft --auto --retree 1 --thread 16 \
    data/SARS-CoV-2/${COUNTRY}/Reference/Test/sequences.fasta \
    > data/SARS-CoV-2/${COUNTRY}/Reference/Test/sequences_aligned.fasta
```

based on the aligned sequences we run `compute_distances.py`:

```bash
python manuscript/sarscov2/compute_distances.py \
    --msa data/SARS-CoV-2/${COUNTRY}/Reference/Test/sequences_aligned.fasta \
    --output data/SARS-CoV-2/${COUNTRY}/Reference/Test/distances.tsv \
    --num_workers 8
```

which will create a tab-separated file called `distances.tsv` with all pairwise distances.

For our purposes we create "Easy", "Medium", and "Hard" samples based on taxonomic ambiguity (see manuscript):

```bash
PREFIX="data/SARS-CoV-2/${COUNTRY}/Reference/Test"

python manuscript/sarscov2/pick_simulation_genomes.py \
    --fasta ${PREFIX}/sequences.fasta \
    --distance_matrix ${PREFIX}/distances.tsv \
    --metadata ${PREFIX}/metadata.tsv \
    --output ${PREFIX} \
    --min_cluster_size 5 \
    --max_sequences 5 \
    --max_lineages 10
```

This selects up to 10 lineages, and up to 5 sequences from each selected lineage to simulate reads from for profiling performance evaluation. The full details can be found in the manuscript, but in brief, we find difficult and non-difficult lineages based on clustering and proximity to medoid sequences. From these lineages we select sequences based on how many sequences are closer to the medoid of other lineages than their own (within clusters). The resulting selected sequences are individually stored in `data/SARS-CoV-2/${COUNTRY}/Reference/Test/{Easy,Medium,Hard}/${LINEAGE}`, using the GISAID EPI_ISL identifiers as filenames. We use these sequences to simulate 20 difficulty-stratified samples per difficulty:

```bash
PREFIX="data/SARS-CoV-2/${COUNTRY}/Reference/Test"

for DIFF in Easy Medium Hard; do
    ROOT_FOLDER="${PREFIX}/${DIFF}"
    for SAMPLE in {1..20}; do
        python manuscript/sarscov2/generate_test_samples.py \
            --root_folder "${ROOT_FOLDER}" \
            --output_folder "${ROOT_FOLDER}" \
            --sample "${SAMPLE}" \
            --coverage 100 \
            --read_len 150 --frag_mean 250 --frag_sd 10 --profile HS25 \
            --num_seqs 3
    done
done
```

selecting up to 3 sequences per lineage, and simulates lineages at 100x coverage uniformly distributed over their constituent genomes, storing results in `data/SARS-CoV-2/${COUNTRY}/Reference/Test/{Easy,Medium,Hard}/sample_XX_{R1,R2}.fastq`.

**NOTE**: while we make use of stable hashes for creating random seeds, we were asked not to publish the exact root folder from our compute cluster. This makes exactly replicating our experiments impossible. To resolve this, we provide all generated reads on [Zenodo](https://doi.org/10.5281/zenodo.20553987)

### Profiling samples
We profile the samples using the VLQ pipeline, by running kallisto:

```bash
INDEX_LOC=data/SARS-CoV-2/${COUNTRY}/Reference/Train/indices/${METHOD_AND_PARAMETERS}.idx
OUTPUT_DIR=data/SARS-CoV-2/${COUNTRY}/Reference/Test/estimations/${METHOD_AND_PARAMETERS}
mkdir -p ${OUTPUT_DIR}

for DIFF in Easy Medium Hard; do    
    READS_PREFIX=data/SARS-CoV-2/${COUNTRY}/Reference/Test/${DIFF}
    for SAMPLE in {1..20}; do
        SAMPLE_NUM=$(printf "%02d" ${SAMPLE})
        kallisto quant -b 0 -t 2 \
            -i "${INDEX_LOC}" \
            -o "${OUTPUT_DIR}/${DIFF}_${SAMPLE_NUM}" \
            "${READS_PREFIX}/sample_${SAMPLE_NUM}_R1.fastq" \
            "${READS_PREFIX}/sample_${SAMPLE_NUM}_R2.fastq"
    done
done
```

This stores all of the results for every method-parameter combination in `data/SARS-CoV-2/${COUNTRY}/Reference/Test/estimations/${METHOD_AND_PARAMETERS}`. By default, kallisto creates a folder (in this case called `${DIFF}_${SAMPLE_NUM}`) which contains `abundance.h5`, `abundance.tsv`, `run_info.json`.

### Parameter selection
The final step in determining the parameter selections, is to evaluate the profiling performance (L1 error and F1-score) with `analyze_tuning.py`:

```bash
python manuscript/sarscov2/analyze_tuning.py \
    --output_dir data/SARS-CoV-2/tuning \
    --min_abundance 0.001
```

This compares estimated abundances (filtering out abundance estimations below 0.1%) against groundtruth abundances (re-calculated from read files) to compute L1 errors and F1-scores. Afterwards both metrics are weighted and combined into a single score per method-parameters combination and these are sorted based on L1 error, F1-score and minimax rank of both metrics, storing ordered results (best to worst) in `data/SARS-CoV-2/tuning` for every method category. In the experiments we ran, this resulted in the following selections:

**Medoid**
- L1: mash_s10000
- F1: mash_s10000
- Minimax: mash_s10000

(thus, only a single parameterization was selected)

**Hierarchical clustering**
- L1: mash_s10000, threshold=0.99
- F1: mash_s10000, threshold=0.99
- Minimax: mash_s10000, threshold=0.99

(thus, only a single parameterization was selected)

**VSEARCH**
- L1: mash_s5000, threshold=90th percentile
- F1: sourmash_cosine_s3, threshold=1st percentile
- Minimax: sourmash_jaccard_s6, threshold=25th percentile

**ReSeT**
- L1: sourmash_cosine_s3, cost=$10^{-1}$, scale=$10^{-5}$
- F1: mash_s10000, cost=$10^{-1}$, scale=$10^{-5}$
- Minimax: mash_s5000, cost=1, scale=0

## IAV
We run a similar script as for SARS-CoV-2 to create a re-labeling of sequences in order to save space:

```python
from Bio import SeqIO
import os

def main():
    path = "data/IAV/Reference/Train"
    filename = "sequences_concatenated_ns.fasta"

    mapping = {}
    records = list(SeqIO.parse(f"{path}/{filename}", "fasta"))
    clades = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

    # First create re-mapping for aggregated file
    for i, record in enumerate(records):
        mapping[record.id] = i
        record.id = str(i)
        record.description = ""
    SeqIO.write(records, f"{path}/sequences_concatenated_ns_remapped.fasta", "fasta")
    with open(f"{path}/sequences_concatenated_ns_mapping.txt", "w") as f_out:
        for record_id, new_id in mapping.items():
            f_out.write(f"{new_id}\t{record_id}\n")

    # Then create re-mapping per clade
    for clade in clades:
        mapping = {}
        records = list(SeqIO.parse(f"{path}/{clade}/{filename}", "fasta"))
        for i, record in enumerate(records):
            mapping[record.id] = i
            record.id = str(i)
            record.description = ""
        SeqIO.write(records, f"{path}/{clade}/sequences_concatenated_ns_remapped.fasta", "fasta")
        with open(f"{path}/{clade}/sequences_concatenated_ns_mapping.txt", "w") as f_out:
            for record_id, new_id in mapping.items():
                f_out.write(f"{new_id}\t{record_id}\n")

if __name__ == "__main__":
    main()
```

This stores the concatenated sequences in a new fasta file, along with a mapping file for finding the correspondence between old and new sequence identifiers in the Train folder (both at the aggregated level and per clade).

### (Dis-)similarity estimations
As for SARS-CoV-2, we tune over mash, now with sketch sizes of 500 and 1,000, and sourmash with scaling factors of 6 and 3. We run both tools on the `sequences_concatenated_ns_remapped.fasta` files as both mash and sourmash skip contexts that contain ambiguous nucleotides, meaning that per sequence, only the segments themselves are considered and no k-mers are included from the join site of the two segments.

#### mash
Per-clade processing:

```bash
NCORES=16

for SKETCHSIZE in 500 1000; do
    # Sketch
    mash sketch -p ${NCORES} -s ${SKETCHSIZE} -S 12345 -k 31 -i \
        -o data/IAV/Reference/Train/${CLADE}/mash_sketch_s${SKETCHSIZE} \
        data/IAV/Reference/Train/${CLADE}/sequences_concatenated_ns_remapped.fasta

    # Distance estimations
    mash triangle -p ${NCORES} \
        data/IAV/Reference/Train/${CLADE}/mash_sketch_s${SKETCHSIZE} > \
        data/IAV/Reference/Train/${CLADE}/mash_triangle_s${SKETCHSIZE}.dist
done
```

Overall processing:

```bash
NCORES=16

for SKETCHSIZE in 500 1000; do
    # Sketch
    mash sketch -p ${NCORES} -s ${SKETCHSIZE} -S 12345 -k 31 -i \
        -o data/IAV/Reference/Train/mash_sketch_s${SKETCHSIZE} \
        data/IAV/Reference/Train/sequences_concatenated_ns_remapped.fasta

    # Distance estimations
    mash triangle -p ${NCORES} \
        data/IAV/Reference/Train/mash_sketch_s${SKETCHSIZE} > \
        data/IAV/Reference/Train/mash_triangle_s${SKETCHSIZE}.dist
done
```

#### sourmash
Per clade processing:

```bash
NCORES=16

for SCALE in 6 3; do
    # Create manifest file required for branchwater
    printf "name,genome_filename,protein_filename\nx,%s,\n" data/IAV/Reference/Train/${CLADE}/sequences_concatenated_ns_remapped.fasta > data/IAV/Reference/Train/${CLADE}/manifest.csv

    sourmash scripts manysketch -o data/IAV/Reference/Train/${CLADE}/sourmash_sketch_s${SCALE}.zip \
        -p "k=31,scaled=${SCALE},abund,dna,seed=12345" \
        -c ${NCORES} --singleton -q \
        data/IAV/Reference/Train/${CLADE}/manifest.csv

    sourmash scripts pairwise -o data/IAV/Reference/Train/${CLADE}/sourmash_distance_s${SCALE}.csv \
        -k 31 -s ${SCALE} -m DNA -c ${NCORES} -t 0 --calc-abund-stats -q \
        data/IAV/Reference/Train/${CLADE}/sourmash_sketch_s${SCALE}.zip
done
```

Overall processing is done similarly, leaving out the `${CLADE}` subfolders to operate on the aggregated set of sequences.

### Selection
#### Medoid
Medoid selections are obtained with `manuscript/iav/medoid.py`:

```bash
PREFIX="data/IAV/Reference/Train/${CLADE}"

# mash-based
python manuscript/iav/medoid.py \
    --input_path "${PREFIX}/mash_triangle_s${SKETCHSIZE}.dist" \
    --output_path "${PREFIX}/medoid_mash_s${SKETCHSIZE}.txt" \
    --mapping_path "${PREFIX}/sequences_concatenated_ns_mapping.txt"

# sourmash-based (jaccard)
python manuscript/iav/medoid.py \
    --input_path "${PREFIX}/sourmash_distance_s${SCALE}.csv" \
    --sourmash_jaccard \
    --output_path "${PREFIX}/medoid_sourmash-jaccard_s${SCALE}.txt" \
    --mapping_path "${PREFIX}/sequences_concatenated_ns_mapping.txt"

# sourmash-based (cosine)
python manuscript/iav/medoid.py \
    --input_path "${PREFIX}/sourmash_distance_s${SCALE}.csv" \
    --sourmash_cosine \
    --output_path "${PREFIX}/medoid_sourmash-cosine_s${SCALE}.txt" \
    --mapping_path "${PREFIX}/sequences_concatenated_ns_mapping.txt"
```

#### Hierarchical clustering
In accordance with the SARS-CoV-2-based experiements we consider a range of parameter settings. However, as IAV genomes are generally less similar than SARS-CoV-2, we instead consider fixed similarity thresholds of 95%, 97%, 99% and 99.9%, while also including the percentile (per-clade) similarity thresholds used before. The selections were obtained with `manuscript/iav/hierarchical.py`:

```bash
PREFIX="data/IAV/Reference/Train/${CLADE}"

# Fixed thresholds
for THRESHOLD in 0.95 0.97 0.99 0.999; do
    python manuscript/iav/hierarchical.py \
        --input_path ${PREFIX}/mash_triangle_s${SKETCHSIZE}.dist \
        --output_path ${PREFIX}/hierarchical_mash_s${SKETCHSIZE}_t${THRESHOLD}.txt \
        --mapping_path ${PREFIX}/sequences_concatenated_ns_mapping.txt \
        --threshold "${THRESHOLD}"
done

# Dynamic thresholds
for PERCENTILE in 1 25 50 75 90; do
    python manuscript/iav/hierarchical.py \
        --input_path ${PREFIX}/mash_triangle_s${SKETCHSIZE}.dist \
        --output_path ${PREFIX}/hierarchical_mash_s${SKETCHSIZE}_p${PERCENTILE}.txt \
        --mapping_path ${PREFIX}/sequences_concatenated_ns_mapping.txt \
        --dynamic --threshold "${PERCENTILE}"
done
# (repeat both with --sourmash_jaccard / --sourmash_cosine against sourmash_distance_s${SCALE}.csv)
```

#### VSEARCH
As before, we need to first compute the dynamic distance cut-offs, this time with `manuscript/iav/thresholds.py` (which is identical to that for SARS-CoV-2):

```bash
for PERCENTILE in 1 25 50 75 90; do
    python manuscript/iav/thresholds.py \
        --input_path data/IAV/Reference/Train/${CLADE}/mash_triangle_s${SKETCHSIZE}.dist \
        --output_path data/IAV/Reference/Train/${CLADE}/${PERCENTILE}_mash_s${SKETCHSIZE}.txt \
        --threshold ${PERCENTILE}
    # (repeat with --sourmash_jaccard / --sourmash_cosine against sourmash_distance_s${SCALE}.csv)
done
```

fixed threshold selections are then obtained with:

```bash
NCORES=16

for THRESHOLD in 0.95 0.97 0.99 0.999; do
    # We use the directly concatenated sequences with VSEARCH
    vsearch --cluster_fast data/IAV/Reference/Train/${CLADE}/sequences_concatenated.fasta \
        --centroids data/IAV/Reference/Train/${CLADE}/vsearch_t${THRESHOLD}.fasta \
        --id ${THRESHOLD} --iddef 0 \
        --maxaccepts 1 --maxrejects 8 \
        --threads ${NCORES}
        # --wordlength is omitted as these sequences are shorter
done
```

and with dynamic thresholds:

```bash
NCORES=16

for PERCENTILE in 1 25 50 75 90; do
    THRESHOLD=$(cat "data/IAV/Reference/Train/${CLADE}/${PERCENTILE}_mash_s${SKETCHSIZE}.txt")

    vsearch --cluster_fast data/IAV/Reference/Train/${CLADE}/sequences_concatenated.fasta \
        --centroids data/IAV/Reference/Train/${CLADE}/vsearch_mash_s${SKETCHSIZE}_p${PERCENTILE}.fasta \
        --id ${THRESHOLD} --iddef 0 \
        --maxaccepts 1 --maxrejects 8 \
        --threads ${NCORES}

    # (repeat for sourmash-jaccard and sourmash-cosine)
done
```

#### ReSeT
The first step again consists of creating the required `clusters.csv` by running the following python script:

```python
ID_COL = 0
CLADE_COL = 15

# Read seqid -> clade
seq2clade = {}
with open("data/IAV/Reference/Train/metadata.tsv", "r") as f_in:
    next(f_in) # skip header

    for line in f_in:
        parts = line.strip().split("\t")
        seq2clade[parts[ID_COL]] = parts[CLADE_COL]

# Read index -> seq_id
mapping = {}
with open("data/IAV/Reference/Train/sequences_concatenated_ns_mapping.txt", "r") as f_in:
    for line in f_in:
        parts = line.strip().split("\t")
        mapping[int(parts[0])] = parts[1]

# Write clusters.csv
with open("data/IAV/Reference/Train/clusters.csv", "w") as f_out:
    for idx in sorted(mapping):
        seq_id = mapping[idx]
        clade = seq2clade[seq_id]
        f_out.write(f"{seq_id},{clade}\n")
```

ReSeT was ran as:

```bash
NCORES=16

for COST in 0.000000 0.000001 0.000010 0.000100 0.001000 0.010000 0.100000 1.000000; do
    for SCALE in 0.00000 0.00001 0.00010 0.00100 0.01000 0.10000 1.00000; do
        # mash-based
        python -m reset.solution \
            --clusters "data/IAV/Reference/Train/clusters.csv" \
            --distances "data/IAV/Reference/Train/mash_triangle_s${SKETCHSIZE}.dist" \
            --distance_format mash \
            --scale ${SCALE} \
            --selection_cost ${COST} \
            --seed 12345 \
            --max_fraction 0.5 \
            --max_iterations 1000000 \
            --max_runtime 432000 \
            --doubleswap_time_threshold 300 \
            --num_processes 16 \
            --sequences_mapping "data/IAV/Reference/Train/sequences_concatenated_ns_mapping.txt" \
            --output "data/IAV/Reference/Train/reset_mash_s${SKETCHSIZE}_cost-${COST}_scale-${SCALE}.txt"

        # sourmash-based (jaccard)
        python -m reset.solution \
            --clusters "data/IAV/Reference/Train/clusters.csv" \
            --distances "data/IAV/Reference/Train/sourmash_distance_s${SOURMASH_SCALE}.csv" \
            --distance_format sourmash_jaccard \
            --scale ${SCALE} \
            --selection_cost ${COST} \
            --seed 12345 \
            --max_fraction 0.5 \
            --max_iterations 1000000 \
            --max_runtime 432000 \
            --doubleswap_time_threshold 300 \
            --num_processes 16 \
            --sequences_mapping "data/IAV/Reference/Train/sequences_concatenated_ns_mapping.txt" \
            --output "data/IAV/Reference/Train/reset_sourmash-jaccard_s${SOURMASH_SCALE}_cost-${COST}_scale-${SCALE}.txt"

        # sourmash-based (cosine)
        python -m reset.solution \
            --clusters "data/IAV/Reference/Train/clusters.csv" \
            --distances "data/IAV/Reference/Train/sourmash_distance_s${SOURMASH_SCALE}.csv" \
            --distance_format sourmash_cosine \
            --scale ${SCALE} \
            --selection_cost ${COST} \
            --seed 12345 \
            --max_fraction 0.5 \
            --max_iterations 1000000 \
            --max_runtime 432000 \
            --doubleswap_time_threshold 300 \
            --num_processes 16 \
            --sequences_mapping "data/IAV/Reference/Train/sequences_concatenated_ns_mapping.txt" \
            --output "data/IAV/Reference/Train/reset_sourmash-cosine_s${SOURMASH_SCALE}_cost-${COST}_scale-${SCALE}.txt"
    done
done
```

### Creating indices
We again follow the same steps, starting with running `manuscript/iav/gather_tuning_genomes.py`:

```bash
python manuscript/iav/gather_tuning_genomes.py \
    --metadata data/IAV/Reference/Train/metadata.tsv \
    --fasta_HA data/IAV/Reference/Train/sequences_HA.fasta \
    --fasta_NA data/IAV/Reference/Train/sequences_NA.fasta \
    --mapping_prefix data/IAV/Reference/Train \
    --input_dir data/IAV/Reference/Train \
    --output_dir data/IAV/Reference/Train/selections
```

This creates a folder `selections` and makes a fasta file per selection (for every parameterization) containing the selected genomic segments (as individual entries), and saves it in `data/IAV/Reference/Train/selections`.

We then build kallisto indices:
```bash
mkdir -p data/IAV/Reference/Train/indices
kallisto index -i data/IAV/Reference/Train/indices/${METHOD_AND_PARAMETERS}.idx \
    data/IAV/Reference/Train/selections/${METHOD_AND_PARAMETERS}.fasta
```

### Simulating samples
As for the SARS-CoV-2 experiments, we only build difficulty-stratified samples for which we compute pairwise distances between all genomes in the `Test` folder based on `mafft`:

```bash
NCORES=16

mafft --auto --retree 1 --thread ${NCORES} \
    data/IAV/Reference/Test/sequences_HA.fasta > \
    data/IAV/Reference/Test/sequences_HA_aligned.fasta

mafft --auto --retree 1 --thread ${NCORES} \
    data/IAV/Reference/Test/sequences_NA.fasta > \
    data/IAV/Reference/Test/sequences_NA_aligned.fasta
```

and based on the aligned sequences we run `manuscript/iav/compute_distances.py`:

```bash
python manuscript/iav/compute_distances.py \
    --msa_HA data/IAV/Reference/Test/sequences_HA_aligned.fasta \
    --msa_NA data/IAV/Reference/Test/sequences_NA_aligned.fasta \
    --output data/IAV/Reference/Test/distances.tsv \
    --num_workers 8
```

which will create a tab-separated file called `distances.tsv` with all pairwise distances.

**NOTE**: Unlike SARS-CoV-2, which is a single segmented virus, IAV has two segments in our experiments (HA + NA). For the distance calculations we thus aligned per segment, and computed distance by aggregating over the two segments (summing).

In contrast to SARS-CoV-2, the clustering strategy did not work for IAV (resulting degenerate clusterings), so instead we score sequences by their proximity to their own clade's medoid versus the closest medoid of another cluster (globally) and use this to select "Easy" and "Hard" sequences (for more info, see manuscript):

```bash
PREFIX=data/IAV/Reference/Test

python manuscript/iav/pick_simulation_genomes.py \
    --fasta_HA ${PREFIX}/sequences_HA.fasta \
    --fasta_NA ${PREFIX}/sequences_NA.fasta \
    --distance_matrix ${PREFIX}/distances.tsv \
    --metadata ${PREFIX}/metadata.tsv \
    --output ${PREFIX} \
    --max_sequences 5
```

Due to the limited number of clades, we include all, but still limit ourselves to up to 5 sequences from each clade. Selected sequences (both segments) are individually stored in `data/IAV/Reference/Test/{Easy,Hard}/${CLADE}`, with filenames corresponding to the accession id of the assemblies (derived from metadata). We again use these to simulate 20 difficulty-stratified samples per difficulty:

```bash
PREFIX=data/IAV/Reference/Test

for DIFF in Easy Hard; do
    ROOT_FOLDER="${PREFIX}/${DIFF}"
    for SAMPLE in {1..20}; do
        python manuscript/iav/generate_test_samples.py \
            --root_folder "${ROOT_FOLDER}" \
            --output_folder "${ROOT_FOLDER}" \
            --sample "${SAMPLE}" \
            --coverage 100
            --read_len 150 --frag_mean 250 --frag_sd 10 --profile HS25 \
            --num_seqs 3
    done
done
```

selecting up to 3 sequences per clade, and simulating clades at 100x coverage uniformly distributed over their constituent genomes (represented by HA + NA), storing results in `data/IAV/Reference/Test/{Easy,Hard}/sample_XX_{R1,R2}.fastq`.

**NOTE**: while we make use of stable hashes for creating random seeds, we were asked not to publish the exact root folder from our compute cluster. This makes exactly replicating our experiments impossible. To resolve this, we provide all generated reads on [Zenodo](https://doi.org/10.5281/zenodo.20553987)

### Profiling samples
We profile the samples using the VLQ pipeline, by running kallisto:

```bash
INDEX_LOC=data/IAV/Reference/Train/indices/${METHOD_AND_PARAMETERS}.idx
OUTPUT_DIR=data/IAV/Reference/Test/estimations/${METHOD_AND_PARAMETERS}
mkdir -p ${OUTPUT_DIR}

for DIFF in Easy Hard; do
    READS_PREFIX=data/IAV/Reference/Test/${DIFF}
    for SAMPLE in {1..20}; do
        SAMPLE_NUM=$(printf "%02d" ${SAMPLE})
        kallisto quant -b 0 -t 2 \
            -i ${INDEX_LOC} \
            -o ${OUTPUT_DIR}/${DIFF}_${SAMPLE_NUM} \
            ${READS_PREFIX}/sample_${SAMPLE_NUM}_R1.fastq \
            ${READS_PREFIX}/sample_${SAMPLE_NUM}_R2.fastq
    done
done
```

This stores all of the results (for every method-parameters combination) in `data/IAV/Reference/Test/estimations/${METHOD_AND_PARAMETERS}`.

### Parameter selection
The final step in determining the parameter selections, is to evaluate the profiling performance, as we did for SARS-CoV-2, with `manuscript/iav/analyze_tuning.py`:

```bash
python manuscript/iav/analyze_tuning.py \
    --output_dir data/IAV/tuning \
    --min_abundance 0.001
```

This does the same comparison as for [SARS-CoV-2](#parameter-selection) (except for having to average between China and USA). In the experiments we ran, this resulted in the following selections:

**Medoid**
- L1: mash_s500
- F1: mash_s500
- Minimax: mash_s500

(thus, only a single parameterization was selected)

**Hierarchical clustering**
- L1: mash_s1000, threshold=90th percentile
- F1: mash_s1000, threshold=90th percentile
- Minimax: mash_s1000, threshold=90th percentile

(thus, only a single parameterization was selected)

**VSEARCH**
- L1: mash_s1000, threshold=75th percentile
- F1: default, threshold=0.99
- Minimax: default, threshold=0.99

(thus, only two parameterizations were selected)

**ReSeT**
- L1: mash_s500, cost=1, scale=0
- F1: mash_s500, cost=$10^{-2}$, scale=0
- Minimax: mash_s500, cost=$10^{-2}$, scale=0

(thus, only two parameterizations were selected)

# Benchmarking experiments
In our benchmarking experiments, we timed all commands using `/usr/bin/time -v` to monitor resource usage.

## SARS-CoV-2
### Pre-processing
We again run a small python script on the aggregated sequence collections (USA and China individually) to create a re-labeling of sequences if it doesn't exist, and to distribute genomes over their corresponding lineages:

```bash
python manuscript/sarscov2/split_reference.py \
    --fasta data/SARS-CoV-2/${COUNTRY}/Reference/sequences.fasta \
    --metadata data/SARS-CoV-2/${COUNTRY}/Reference/metadata.tsv \
    --output_dir data/SARS-CoV-2/${COUNTRY}/Reference
```

This script performs two functions: first it creates a lineage directory in `data/SARS-CoV-2/${COUNTRY}/Reference` for every lineage, containing its constituent genomes (relabeled to save storage), and second it creates a remapped fasta file if it doesn't already exist.

### (Dis-)similarity estimations
Based on the parameter tuning we need all per-lineage distance estimates (both mash and sourmash with both sketch sizes), and mash (s=5000,10000) and sourmash (s=3) for the global collection.

#### mash
Per-lineage processing:

```bash
NCORES=32

for SKETCHSIZE in 5000 10000; do
    # Sketch
    mash sketch -p ${NCORES} -s ${SKETCHSIZE} -S 12345 -k 31 -i \
        -o data/SARS-CoV-2/${COUNTRY}/Reference/${LINEAGE}/mash_sketch_s${SKETCHSIZE} \
        data/SARS-CoV-2/${COUNTRY}/Reference/${LINEAGE}/sequences_remapped.fasta

    # Distance estimations
    mash triangle -p ${NCORES} \
        data/SARS-CoV-2/${COUNTRY}/Reference/${LINEAGE}/mash_sketch_s${SKETCHSIZE}.msh > \
        data/SARS-CoV-2/${COUNTRY}/Reference/${LINEAGE}/mash_triangle_s${SKETCHSIZE}.dist
done
```

Overall processing (using the remapped multi-fasta):

```bash
NCORES=32

for SKETCHSIZE in 5000 10000; do
    # Sketch
    mash sketch -p ${NCORES} -s ${SKETCHSIZE} -S 12345 -k 31 -i \
        -o data/SARS-CoV-2/${COUNTRY}/Reference/mash_sketch_s${SKETCHSIZE} \
        data/SARS-CoV-2/${COUNTRY}/Reference/sequences_remapped.fasta

    # Distance estimations
    mash triangle -p ${NCORES} \
        data/SARS-CoV-2/${COUNTRY}/Reference/mash_sketch_s${SKETCHSIZE}.msh > \
        data/SARS-CoV-2/${COUNTRY}/Reference/mash_triangle_s${SKETCHSIZE}.dist
done
```

#### sourmash
Per-lineage processing:

```bash
NCORES=32

for SCALE in 6 3; do
    printf "name,genome_filename,protein_filename\nx,%s,\n" data/SARS-CoV-2/${COUNTRY}/Reference/${LINEAGE}/sequences_remapped.fasta > data/SARS-CoV-2/${COUNTRY}/Reference/${LINEAGE}/manifest.csv

    sourmash scripts manysketch -o data/SARS-CoV-2/${COUNTRY}/Reference/${LINEAGE}/sourmash_sketch_s${SCALE}.zip \
        -p "k=31,scaled=${SCALE},abund,dna,seed=12345" \
        -c ${NCORES} --singleton -q \
        data/SARS-CoV-2/${COUNTRY}/Reference/${LINEAGE}/manifest.csv

    sourmash scripts pairwise -o data/SARS-CoV-2/${COUNTRY}/Reference/${LINEAGE}/sourmash_distance_s${SCALE}.csv \
        -k 31 -s ${SCALE} -m DNA -c ${NCORES} -t 0 --calc-abund-stats -q \
        data/SARS-CoV-2/${COUNTRY}/Reference/${LINEAGE}/sourmash_sketch_s${SCALE}.zip
done
```

Due to the large number of comparisons required for the USA dataset (see this [github issue](https://github.com/sourmash-bio/sourmash_plugin_branchwater/issues/737)), we first partition the full genome set using the `partition.py` script:

```bash
python manuscript/sarscov2/partition.py \
    --input_fasta data/SARS-CoV-2/${COUNTRY}/Reference/sequences_remapped.fasta \
    --output_folder data/SARS-CoV-2/${COUNTRY}/Reference \
    --partition_size 20000
```

Afterwards, we ran sourmash (only for s=3):

```bash
NCORES=32

for PARTITION_FILE in data/SARS-CoV-2/${COUNTRY}/Reference/partition_*.fasta; do
    PARTITION_BASENAME=$(basename "${PARTITION_FILE}" .fasta)
    printf "name,genome_filename,protein_filename\n%s,%s,\n" "${PARTITION_BASENAME}" "${PARTITION_FILE}" > data/SARS-CoV-2/${COUNTRY}/Reference/ms_${PARTITION_BASENAME}.csv

    # Sketch
    sourmash scripts manysketch -o data/SARS-CoV-2/${COUNTRY}/Reference/${PARTITION_BASENAME}.zip -p k=31,scaled=3,abund,dna,seed=12345 -c ${NCORES} --singleton data/SARS-CoV-2/${COUNTRY}/Reference/ms_${PARTITION_BASENAME}.csv
done

# Gather sketches
sourmash sig cat data/SARS-CoV-2/${COUNTRY}/Reference/partition_*.zip -o data/SARS-CoV-2/${COUNTRY}/Reference/all_partitions_sketches.zip

# Calculate pairwise distances
sourmash scripts pairwise -o data/SARS-CoV-2/${COUNTRY}/Reference/sourmash_distance_s3.csv -k 31 -s 3 -m DNA -c ${NCORES} -t 0 --calc-abund-stats data/SARS-CoV-2/${COUNTRY}/Reference/all_partitions_sketches.zip
```

### Selection
Selections and indices can be obtained in the same way as [before](#selection). Where applicable, we ran tools using 32 threads/cores for the final benchmarking experiments storing selections in `data/SARS-CoV-2/${COUNTRY}/Reference/selections` and indices in `data/SARS-CoV-2/${COUNTRY}/Reference/indices`. In addition to the selected method parameterizations, we also create an index containing all available reference genomes (index stored as `all.idx`), to act as an additional baseline.

**NOTE**: Unlike before, we re-labeled sequences for the final benchmarking experiments, so this needs to be accounted (which isn't in the scripts supplied but can trivially be added).

### Simulating samples
In contrast to the parameter-tuning experiments, we ran experiments on both difficulty-stratified samples, as well as Dirichlet-distributed samples for the benchmarking experiments. We again start by running mafft (using the same parameters as [before](#simulating-samples), including threadcount) and then by running `compute_distances.py` now using `data/SARS-CoV-2/${COUNTRY}/Simulation` instead of `data/SARS-CoV-2/${COUNTRY}/Reference/Test`.

#### Difficulty-stratified samples
Difficulty-stratified samples were obtained as [before](#simulating-samples), using the exact same parameters, except storing the reads in `data/SARS-CoV-2/${COUNTRY}/Simulation`.

#### Dirichlet-distributed samples
In addition to the difficulty-distributed samples we created Dirichlet-distributed samples as described in the manuscript using `create_dirichlet_samples.py`:

```bash
PREFIX=data/SARS-CoV-2/${COUNTRY}/Simulation

python manuscript/sarscov2/create_dirichlet_samples.py \
    --fasta ${PREFIX}/sequences.fasta \
    --distance_matrix ${PREFIX}/distances.tsv \
    --metadata ${PREFIX}/metadata.tsv \
    --output ${PREFIX}/Dirichlet \
    --max_sequences 10 \
    --max_lineages 20 \
    --num_replicates 10 \
    --num_configs 10 \
    --min_abundance 0.001 \
    --seed 12345 \
    --coverage 2000 --read_len 150 --frag_mean 250 --frag_sd 10 --profile HS25
```

This stores the resulting samples in `data/SARS-CoV-2/${COUNTRY}/Simulation/Dirichlet/config_${CONFIG}/alpha_${ALPHA}/replicate_${REPLICATE}/sample_{1,2}.fq`. In our experiments we created 10 configurations (i.e. selections of different lineages and their assigned abundance given a Dirichlet distribution concentration parameter) with 3 different concentration parameters (0.1, 1.0, 10.0) with 10 replicates per combination.

**NOTE**: Unlike in the difficulty-stratified samples, we now made the stable hashes independent of the compute cluster file system. The corresponding seeds used can also be found on [Zenodo](https://doi.org/10.5281/zenodo.20553987), which hosts the created manifest files.

### Profiling samples
We again profile the samples in the same way (now profiling both difficulty-stratified samples and dirichlet-distributed samples) as [before](#profiling-samples), using a single thread for index construction, and using 2 threads for profiling storing results in `data/SARS-CoV-2/${COUNTRY}/Simulation/estimations/${METHOD_AND_PARAMETERS}`.

### Evaluation
Analysis of the output was done using the `analyze_benchmarking_output.py` script which performs the statistical comparisons and generates the individual figure panels in the manuscript. Note that in the script, we aggregate results for Medoid and Hierarchical clustering, as these selections were identical under the chosen parameters.

## IAV
### Pre-processing
Unlike SARS-CoV-2, I learned from my experiences and did most of the pre-processing for IAV at the start of the parameter tuning experiments. The only thing we need to do is the re-labeling of sequences with the following python script:

```python
from Bio import SeqIO
import os

def main():
    path = "data/IAV/Reference"
    filename = "sequences_concatenated_ns.fasta"

    mapping = {}
    records = list(SeqIO.parse(f"{path}/{filename}", "fasta"))
    clades = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

    # First create re-mapping for aggregated file
    for i, record in enumerate(records):
        mapping[record.id] = i
        record.id = str(i)
        record.description = ""
    SeqIO.write(records, f"{path}/sequences_concatenated_ns_remapped.fasta", "fasta")
    with open(f"{path}/sequences_concatenated_ns_mapping.txt", "w") as f_out:
        for record_id, new_id in mapping.items():
            f_out.write(f"{new_id}\t{record_id}\n")

    # Then create re-mapping per clade
    for clade in clades:
        if clade == "Test" or clade == "Train": # pass over Test and Train folders
            continue
        mapping = {}
        records = list(SeqIO.parse(f"{path}/{clade}/{filename}", "fasta"))
        for i, record in enumerate(records):
            mapping[record.id] = i
            record.id = str(i)
            record.description = ""
        SeqIO.write(records, f"{path}/{clade}/sequences_concatenated_ns_remapped.fasta", "fasta")
        with open(f"{path}/{clade}/sequences_concatenated_ns_mapping.txt", "w") as f_out:
            for record_id, new_id in mapping.items():
                f_out.write(f"{new_id}\t{record_id}\n")
```

### (Dis-)similarity estimations
Distance estimations were only computed for mash (both sketch sizes on per-lineage basis, sketch size = 500 for overall), which was done as [before](#dis-similarity-estimations-1), except using 32 threads rather than 16.

### Selection
Selections and indices were obtained in the same way as [before](#selection-1). Where applicable, we ran tools using 32 threads/cores for the final benchmarking experiments, storing selections in `data/IAV/Reference/selections` and indices in `data/IAV/Reference/indices`. In addition to the selected method parameterizations, we also create an index containing all available reference genomes (index stored as `all.idx`), to act as an additional baseline.

### Simulating samples
In contrast to the parameter-tuning experiments, we ran experiments on both difficulty-stratified samples, as well as Dirichlet-distributed samples for the benchmarking experiments. We again start by running mafft (using the same parameters as [before](#simulating-samples-1), including threadcount) and then by running `manuscript/iav/compute_distances.py` now using `data/IAV/Simulation` instead of `data/IAV/Reference/Test`.

#### Difficulty-stratified samples
Difficulty-stratified samples were obtained as [before](#simulating-samples-1), using the exact same parameters, except storing the reads in `data/IAV/Simulation`.

#### Dirichlet-distributed samples
Dirichlet-distributed samples were created with `manuscript/iav/create_dirichlet_samples.py`:

```bash
PREFIX=data/IAV/Simulation

python manuscript/iav/create_dirichlet_samples.py \
    --fasta_HA ${PREFIX}/sequences_HA.fasta \
    --fasta_NA ${PREFIX}/sequences_NA.fasta \
    --distance_matrix ${PREFIX}/distances.tsv \
    --metadata ${PREFIX}/metadata.tsv \
    --output ${PREFIX}/Dirichlet \
    --max_sequences 10 \
    --max_clades 5 \
    --num_replicates 10 \
    --num_configs 10 \
    --min_abundance 0.001 \
    --seed 12345 \
    --coverage 2000 --read_len 150 --frag_mean 250 --frag_sd 10 --profile HS25
```

### Profiling samples
We again profile the samples in the same way (now profiling both difficulty-stratified samples and Dirichlet-distributed samples) as [before](#profiling-samples-1), using a single thread for index construction, and using 2 threads for profiling, storing results in `data/IAV/Simulation/estimations/${METHOD_AND_PARAMETERS}`.

### Evaluation
Analysis of the output was done using the `manuscript/iav/analyze_benchmarking_output.py` script which performs the statistical comparisons and generates the individual figure panels in the manuscript.

# Additional figure creating scripts
In addition to the benchmarking results, we also created heatmaps for the SARS-CoV-2 parameter tuning experiments (`manuscript/sarscov2/heatmaps.py`) to highlight the (lack of) impact of the scale parameter for ReSeT. Finally, we created figures that show the distribution of inter- versus intra-taxon distances (`manuscript/data_view/data.py`), for which we first compute the overall mash distances with s=1000 for IAV, and we use the mash distances with s=10000 for SARS-CoV-2.