The objective of this document, is to detail how I proceeded to do things.

* DOWNLOADING DATA
Using NCBI datasets (https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/command-line/):
- downloading genomes: 
```bash
datasets download genome taxon TAXID --assembly-level complete --assembly-source GenBank --exclude-atypical --filename FILENAME
```
This downloads all "Complete Genome" quality genomes from GenBank (this excludes duplicates that may also appear in RefSeq) while excluding genomes listed as "atypical" (which are usually hidden when browsing NCBI Genomes).
- downloading metadata:
```bash
datasets summary genome taxon TAXID --assembly-level complete --assembly-source GenBank --exclude-atypical --as-json-lines | dataformat tsv genome --fields accession,organism-tax-id,source_database,assminfo-atypicalis-atypical > metadata.tsv
```
This downloads the metadata (primarily the taxid) for the genomes downloaded through the previous command and stores it in a tsv file called `metadata.tsv`.

* LOCAL SEARCH
For the local search, the general strategy for now is to start from a centroid solution (i.e. select a centroid for every taxon), and to perform local search until an iteration limit or local optimal point has been reached. With this starting point (and in general) it is important to note that in order to remain feasible, we should also consider swaps (i.e. removing one point and adding another) which increases the neighborhoods.

* TO-DO
- Implement parallel local search
    + generate neighborhood (i.e. local moves)
    + batch local moves to pass on to processes
    + evaluate local moves in parallel and stop when improvement is found
        - possibly allow for process to finish processing batch (overhead is not too significant if batch is small)
- Implement Simulated Annealing
    + mostly similar to local search, but needs to deal with problems of parallelism
- Low-priority
    + visualization of the local search (how does the solution change over time)


* IDEAS
- For the local search we currently only move if we find a better solution. However, there might be merit in considering multiple points in the neighborhood, and moving towards one based on softmax probabilities. Doing so enables searching larger neighborhoods at the cost of potentially obtaining a worse solution. This can be solved by storing the best found solution so far
- Currently implementation requires access to full distance matrix, which is of size O(N^2) with N=#genomes. Ideally, this can be circumvented, for example by only storing "local" distance matrices in memory (within cluster), and re-calculating distances between points of different clusters on a "per-need" basis. Alternatively, this might be able with memory mapping?
- Assigning a constant cost of 1 for picking a genome might be arbitrary. It would be better to find a way of setting this cost to a value that makes sense (e.g. setting it to 0.5 because then it adds a genome if the distance between a pair of genomes is more than 0.5)


## FLP-based model
### Parameters
$$
\begin{array}{ll}
G &\text{set of all genomes} \\
t: G \rightarrow \mathbb{N} &\text{taxonomic mapping} \\
\sim_t &\text{equivalence relation for all genomes based on taxonomic mapping} \\
\{ {T_1, T_2, \ldots}, T_m \} =: G \setminus \sim_t &\text{set of all taxonomic classes} \\
s: G \times G \rightarrow [0,1] \cap \mathbb{Q} &\text{similarity mapping between genomes} \\
f: g \rightarrow \mathbb{R}_{\geq 0} &\text{cost for selecting a genome (generally constant)}
\end{array}
$$

### Decision Variables
$$
\begin{array}{llr}
x_g &= 
\begin{cases}
    1, & \text{if genome } g \text{ is selected} \\
    0, & \text{otherwise}
\end{cases} &\qquad \forall g \in G\\
y_{g,g'} &= 
\begin{cases}
    1, & \text{if genome } g' \text{ is selected as representative for } g \\
    0, & \text{otherwise}
\end{cases} &\qquad \forall g \sim_{t} g' \\
z_{g,g'} &=
\begin{cases}
    1, & \text{if genome } g \text{ and } g' \text{ are both selected} \\
    0, & \text{otherwise}
\end{cases} &\qquad \forall g \not\sim_{t} g' \\
q_{i,j} &\geq 0 \text{ penalty for distance between closest selected genomes for } T_i \text{ and } T_j \text{} &\qquad \forall T_i \neq T_j
\end{array}
$$

### Objective function
$$
\begin{array}{llr}
\text{Minimize} & \sum_{g \in G}f(g)x_g + \sum_{g \sim_t g'} \left(1 - s(g,g')\right)y_{g,g'} + \sum_{i \neq j}q_{T_i, T_j} &(1)
\end{array}
$$

### Constraints
$$
\begin{array}{lllc}
\sum_{g' \sim_t g}y_{g,g'} & \geq 1 & \forall g \in G &(2) \\
y_{g,g'} & \leq x_{g'} & \forall g \sim_t g' &(3) \\
z_{g,g'} & \geq -1 + \left(x_g + x_{g'}\right) & \forall g \not\sim_t g' &(4a) \\
z_{g,g'} & \leq 0.5\left(x_g + x_{g'}\right) & \forall g \not\sim_t g' &(4b) \\
q_{g,g'} & \geq s(g,g')z_{g,g'} & \forall g \not\sim_t g' &(5) \\
x_g \in \{0,1\} & & \forall g \in G &(6) \\
y_{g,g'} \in \{0,1\} & & \forall g \sim_t g' &(7) \\
z_{g,g'} \in \{0,1\} & & \forall g \not\sim_t g' &(8) \\
q_{T_i, T_j} \geq 0 & & \forall i \neq j &(9)
\end{array}
$$
