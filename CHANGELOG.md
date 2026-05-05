# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Top-level `atomic_femdvr` package now re-exports the public API
  (`FullAtomicInput`, `PseudoAtomicInput`, `PseudoAtomicResult`,
  `solve_atomic`, `solve_pseudo_atomic`).
- `atomic_femdvr` CLI gains a ``-v / --verbose`` flag.
- Sphinx + autodoc-pydantic API documentation.
- Bundled S pseudopotential at `examples/data/S.upf`.

### Changed

- `solve_pseudo_atomic` now returns a `PseudoAtomicResult`
  (eigenvalues + per-`l` energy shifts) instead of a bare eigenvalues
  dict.
- Library output is routed through `logging` rather than `print`; the
  CLI installs a default handler so user-visible output is unchanged.

### Removed

- Dead modules and the old test-only `AtomicInput` / `solve_atomic(inp)`
  pair (the canonical all-electron entry point lives in `full_atomic`).
