.PHONY: test test-core test-direct test-nca test-rule-based \
        inspect-direct inspect-nca inspect-rule-based \
        benchmark docs clean

PYTHON := python3

# SURROGATE=1 runs `make benchmark` with the fast, physics-free surrogate
# evaluator instead of real PPO (see evodevo/evaluation/ppo_runner.py).
SURROGATE ?= 0
BENCHMARK_FLAGS :=
ifeq ($(SURROGATE),1)
BENCHMARK_FLAGS := --surrogate
endif

# Runs the ENTIRE test suite -- every encoder, every shared-core module,
# real-PPO smoke tests included (skipped automatically if the evogym
# simulator isn't built). Run this before merging.
test:
	@echo ">> make test: running the full test suite (all encoders + shared core)"
	$(PYTHON) -m pytest

# Shared-core tests only: core/, morphology/, evaluation/, evolution/,
# surrogate evaluation only (no real PPO). These are the modules every
# encoding arm depends on -- see docs/module-ownership.md.
test-core:
	@echo ">> make test-core: running shared-core tests only (core/morphology/evaluation/evolution, surrogate-only)"
	$(PYTHON) -m pytest -m "not slow" tests/core tests/morphology tests/evaluation tests/evolution

# Per-method test runs: ONLY that encoder's contract-suite parametrization,
# its own method-specific tests, and the shared-core tests it depends on.
# Use these while iterating on your own arm -- see scripts/<method>/test.sh.
test-direct:
	@echo ">> make test-direct: running Direct-encoder-only tests (contract suite + method-specific + shared core)"
	bash scripts/direct/test.sh

test-nca:
	@echo ">> make test-nca: running NCA-encoder-only tests (contract suite + method-specific + shared core)"
	bash scripts/nca/test.sh

test-rule-based:
	@echo ">> make test-rule-based: running Rule-Based-encoder-only tests (contract suite + method-specific + shared core)"
	bash scripts/rule_based/test.sh

# PPO-free characterization of one encoder's prior over morphologies:
# validity rate, failure-reason breakdown, unique-body count, degeneracy
# rate, and (for developmental encoders) settling behaviour. Writes to
# results/inspection/<method>/. No PPO/EA is run.
inspect-direct:
	@echo ">> make inspect-direct: characterizing the Direct encoder (PPO-free, no EA)"
	bash scripts/direct/inspect.sh

inspect-nca:
	@echo ">> make inspect-nca: characterizing the NCA encoder (PPO-free, no EA)"
	bash scripts/nca/inspect.sh

inspect-rule-based:
	@echo ">> make inspect-rule-based: characterizing the Rule-Based encoder (PPO-free, no EA)"
	bash scripts/rule_based/inspect.sh

# The unified three-way benchmark: direct vs. nca vs. rule_based, run under
# identical conditions (evodevo/experiments/benchmark.py), followed by
# report generation (evodevo/analysis/report.py) into
# results/BENCHMARK_REPORT.md. SURROGATE=1 uses the fast physics-free
# evaluator instead of real PPO.
benchmark:
	@echo ">> make benchmark: running the three-way benchmark (SURROGATE=$(SURROGATE))"
	$(PYTHON) -m evodevo.experiments.benchmark --config evodevo/experiments/configs/benchmark.yaml $(BENCHMARK_FLAGS)
	@echo ">> make benchmark: generating results/BENCHMARK_REPORT.md"
	$(PYTHON) -m evodevo.analysis.report --root results/runs --output results/BENCHMARK_REPORT.md --figures-dir results/figures

# Sanity-checks that the expected documentation set exists and lists it.
# There is no generated-docs build step in this project.
docs:
	@echo ">> make docs: checking the documentation set"
	@for f in README.md CONTRIBUTING.md docs/adding-an-encoder.md docs/module-ownership.md docs/decisions.md; do \
		if [ -f "$$f" ]; then echo "  [ok] $$f"; else echo "  [MISSING] $$f"; fi; \
	done

# Removes caches and build artifacts. Never touches external/ (nested,
# independently-versioned checkouts), .venv/, or results/.
clean:
	@echo ">> make clean: removing __pycache__, .pytest_cache, and egg-info directories"
	find . -type d -name "__pycache__" -not -path "./external/*" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	find . -maxdepth 1 -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo ">> done"
