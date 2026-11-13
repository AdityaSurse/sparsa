.PHONY: help test run experiments lorenz validation noise systems clean

PYTHON ?= python3
MPLCONFIGDIR ?= /tmp/matplotlib

help:
	@echo "SPARSA: SINDy Nonlinear Dynamics Discovery"
	@echo "Available commands:"
	@echo "  make test         Run all unit test suites"
	@echo "  make run          Launch the Streamlit interactive dashboard"
	@echo "  make lorenz       Run Lorenz equation discovery and Pareto sweep"
	@echo "  make validation   Run out-of-sample dynamical validation benchmark"
	@echo "  make noise        Run full noise and differentiation ablation grid"
	@echo "  make systems      Run multi-system benchmark (Lotka-V, Van der Pol, etc.)"
	@echo "  make clean        Clean build artifacts and cache directories"

test:
	PYTHONPATH=src $(PYTHON) tests/test_simulation.py
	PYTHONPATH=src $(PYTHON) tests/test_differentiation.py
	PYTHONPATH=src $(PYTHON) tests/test_library.py
	PYTHONPATH=src $(PYTHON) tests/test_stlsq.py
	PYTHONPATH=src $(PYTHON) tests/test_metrics.py
	PYTHONPATH=src $(PYTHON) tests/test_weak_form.py

run:
	PYTHONPATH=src MPLCONFIGDIR=$(MPLCONFIGDIR) streamlit run app/dashboard.py

lorenz:
	PYTHONPATH=src MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) experiments/exp_lorenz_discovery.py

validation:
	PYTHONPATH=src MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) experiments/exp_validation.py

noise:
	PYTHONPATH=src MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) experiments/exp_noise.py

systems:
	PYTHONPATH=src MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) experiments/exp_systems.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build dist *.egg-info

