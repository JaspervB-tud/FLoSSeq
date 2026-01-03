# Changelog
All relevant changes are (or will be) documented here.
This project attempts to follow Keep a Changelog and Semantic Versioning.

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