# Matrix Completion Research

Research code for synthetic matrix completion experiments with ALS, Soft-Impute,
vanilla Riemannian gradient descent, regularized RGD, and compact RGD.

The current project focuses on controlled experiments: rank sweeps, noise sweeps,
missingness patterns, runtime, iteration counts, and reconstruction quality.

## Quick Start

```bash
python -m pip install -e ".[dev,notebook]"
python -m pytest
```

## Main Files

- `matrix_completion_methods.py` contains reusable numerical solvers.
- `synthetic_api.py` builds scenarios, runs methods, computes metrics, and plots results.
- `synthetic_research_api.py` provides one-factor experiments and experiment suites.
- `synthetic_research_test_panel.ipynb` is the current notebook entry point.

## GitHub Hygiene

Generated outputs, PDFs, notebook caches, virtual environments, and legacy result
tables are ignored by default. Keep source code, config JSON, and small docs in
git; regenerate heavy artifacts locally when needed.
