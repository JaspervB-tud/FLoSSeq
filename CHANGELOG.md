# Changelog
All relevant changes are (or will be) documented here.
This project attempts to follow Keep a Changelog and Semantic Versioning.

## [0.1.3] - 19/11/2025
### Added
- Included basic dashboarding

### Changed
- Changed doubleswap behaviour
    - Instead of removing after non-occurrences, now drop doubleswaps after an iteration has spent X seconds, and re-adds them after iteration finishes (assuming that search does not terminate)

## [0.1.1] - 11/11/2025
### Added
- GitHub Actions workflow to build, test, and publish on 'v*' tags.
- Version from git tags via hatch-vcs.

### Changed
- Packaging configuration cleanup.

## [0.1.0] - 11/11/2025
### Added
- Initial release