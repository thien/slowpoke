VENV = .venv
PYTHON = $(VENV)/bin/python
WHEEL = target/wheels/slowpoke-0.1.0-cp310-cp310-macosx_11_0_arm64.whl

# Slowpoke build system — Rust extension + Python package
.PHONY: build install test bench smoke dev

# Build the Rust extension wheel
build:
	maturin build --release

# Build + install into the active venv
install: build
	$(PYTHON) -m pip install --no-deps $(WHEEL) --force-reinstall

# Run fast tests (default, excludes slow)
test:
	$(PYTHON) -m pytest

# Run all tests including slow
test-all:
	$(PYTHON) -m pytest -m "slow or not slow"

# Run benchmarks
bench:
	$(PYTHON) -m pytest slowpoke/tests/bench_perf.py -v --no-header

# Quick smoke test that the Rust extension loads
smoke:
	$(PYTHON) -c "from slowpoke.core.checkers import CheckerBoard; b = CheckerBoard(); print(f'Rust: {b._has_core}'); print(f'Moves: {len(b.get_moves())}')"

# Dev shortcut: build, install, test
dev: install test
