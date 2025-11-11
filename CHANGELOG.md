# Changelog
All relevant changes are (or will be) documented here.
This project attempts to follow Keep a Changelog and Semantic Versioning.

## [0.1.2] - 17/11/2025
### Added
- Included dashboarding
    - Functional, but very minimal and should not be used on large instances right now.

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