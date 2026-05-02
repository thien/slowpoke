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

# Run all tests (must run install first)
test:
	PYTHONPATH=library $(PYTHON) -m pytest

# Run benchmarks
bench:
	PYTHONPATH=library $(PYTHON) -m pytest tests/bench_perf.py -v --no-header || cd library && ../$(PYTHON) tests/bench_perf.py

# Quick smoke test that the Rust extension loads
smoke:
	PYTHONPATH=library $(PYTHON) -c "from core.checkers import CheckerBoard; b = CheckerBoard(); print(f'Rust: {b._has_core}'); print(f'Moves: {len(b.get_moves())}')"

# Dev shortcut: build, install, test
dev: install test
