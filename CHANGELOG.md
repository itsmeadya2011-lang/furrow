# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `Makefile` with `install`, `test`, `lint`, `typecheck`, and `clean` targets.
- `.env.example` documenting all `FURROW_*` environment variables.
- `CHANGELOG.md` tracking notable changes.

### Changed
- `LLMClient.list_files` now skips VCS directories (`.git`, `.hg`, `.svn`), cache
  directories (`__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`,
  `.tox`, `.cache`), dependency directories (`node_modules`, `.venv`, `venv`,
  `env`), editor directories (`.idea`, `.vscode`), and build output
  directories (`dist`, `build`, `target`, `.next`, `.nuxt`). This keeps the
  worker's directory listings focused on source files and prevents context
  bloat from auto-generated or vendored trees.