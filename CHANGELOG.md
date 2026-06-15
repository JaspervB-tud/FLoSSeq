# Changelog
All relevant changes are (or will be) documented here.
This project attempts to follow Keep a Changelog and Semantic Versioning.
Changes are tracked by GenAI.

## [1.3.0] - 15/06/2026
Big set of changes to make the tool actually functional for users.

### Added
- Toy example with fictional 34-genome dataset across 4 taxa (`src/reset/examples/toy_example/`), including a MASH triangle-format distance file, cluster labels, generation script, run script, and README. The example is bundled inside the package so it is available after a plain `pip install`.
- `generate_distances_generic`: streams pairwise distances from any delimited `(index1, index2, distance)` file, enabling custom distance formats beyond MASH and Sourmash.
- CLI `--distance_format` argument (`mash`, `sourmash_cosine`, `sourmash_jaccard`, `generic`) replacing the previous binary `--mash` / `--jaccard` flags; generic format exposes `--dist_idx1_col`, `--dist_idx2_col`, `--dist_dist_col`, `--dist_delimiter`, `--dist_header`.
- CLI `--output` argument to write selected item IDs (one per line) to a flat file.
- CLI `--sequences_mapping` is now optional; when omitted the row order of the clusters file is used as the integer index.

### Fixed
- macOS POSIX shared memory names are limited to 30 characters; added `_shm_safe_name()` helper that replaces names exceeding this limit with a deterministic SHA-1 digest, fixing `OSError: [Errno 63] File name too long` on macOS.

### Changed
- Renamed `read_metadata` → `read_clusters` with configurable `id_col`, `cluster_col`, `delimiter`, and `header` arguments, replacing the GISAID-specific hardcoded column layout.
- Renamed `generate_centroid_solution` → `generate_medoid_solution` throughout (Solution, Solution_shm, dashboard).
- `read_sequence_mapping` and `generate_distances_mash` / `generate_distances_sourmash` now accept explicit column, delimiter, and header arguments and include full docstrings.
- `generate_distances_sourmash` signature changed: `jaccard: bool` replaced by `dist_col: int` (pass `6` for Jaccard, `12` for cosine).
- `main()` output simplified to a flat list of selected IDs; per-lineage folder structure removed.
- Fixed bug where `local_search` return value (3-tuple) was unpacked into 2 variables in `main()`.
- Dashboard (`app.py`, `pipeline.py`) rewritten to operate on pre-computed distance files (clusters CSV + distance file) instead of the previous FASTA/MinHash pipeline; defaults to the bundled toy example on startup (resolved relative to the installed package, so it works after `pip install`). Dashboard now exposes `selection_cost` and `scale` controls when initialising a solution.
- `pyproject.toml`: removed `biopython` and `sourmash` from dashboard dependencies (no longer used); added `pandas`; cleaned up stale placeholder URLs.
- Updated `README.md` with full CLI reference, Python API documentation, and a link to the toy example.

## [1.2.1] - 30/01/2026
### Changed
- Moved the epoch update to happen directly when entering the acceptance phase in Solution_shm to prevent race conditions in workers to accidentally try moves.

## [1.2.0] - 09/01/2026
### Added
- Added objective function normalization (using scale factor) which re-weighs inter-cluster costs in order to balance.

## [1.1.2] - 04/01/2026
### Changed
- Finalized main function

## [1.1.0] - 03/01/2026
### Changed
- Added single core processing to Solution_shm which enters multiprocessing after certain amount of time has passed.

## [1.0.1] - 03/01/2026
Code should be (more or less) completely safe now!
NOTE: version 1.0.0 was pushed but was missing some changes...

### Changed
- Fixed some safety problems in multiprocessing
- Changed requirements for distance in initialization (now accepts full distance matrix, OR generator object)
- Homogenized initialization between multiprocessing and single processing
    - Made it possible to pass generator for single processing as well
- Added main so that code can be run

## [0.2.6] - 30/12/2025
Tool is renamed to **Re**ference genome **Se**lection **T**ool now!

### Added
- Solution_shm: this is the multiprocessing implementation that fully runs on shared memory

### Changed
- Removed multiprocessing implementation for base Solution class
- Re-implemented move generation for removal and swap. Previously, clusters were exhausted, now a random cluster is selected every move call
- Removed SolutionAverage

### Not ready yet
Have not yet updated the dashboarding to account for changes.

## [0.1.4.4] - 27/11/2025
Again testing if versioning is correct.

## [0.1.4] - 20/11/2025
Testing if new PYPI package is correctly configured.

## [0.1.3.1] - 20/11/2025
### Changed
- Changed name of package.

## [0.1.3] - 19/11/2025
### Added
- Included basic dashboarding

### Changed
- Changed doubleswap behaviour
    - No longer remove after X non-occurrences in a window of Y moves
    - Removes doubleswap (if enabled) after an iteration has spent X seconds, re-adding next iteration.

## [0.1.1] - 11/11/2025
### Added
- GitHub Actions workflow to build, test, and publish on 'v*' tags.
- Version from git tags via hatch-vcs.

### Changed
- Packaging configuration cleanup.

## [0.1.0] - 11/11/2025
### Added
- Initial release