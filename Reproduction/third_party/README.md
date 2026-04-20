# Third-party dependencies

This directory used to vendor a verbatim copy of the **`qqa`** package
(QQA4CO / Parallel Quasi-Quantum Annealing) so that the reproduction
code was self-contained.

The vendored copy has been **removed** in favour of the published
PyPI release. `qqa` is now a regular project dependency (declared in
the top-level `pyproject.toml`), so a third party only needs to run

```bash
uv sync                           # or: pip install qqa>=0.5
```

to install it. The runtime import path is identical (`import qqa`),
so no other code change is needed.

## Upstream

- PyPI: https://pypi.org/project/qqa/  (BSD-3-Clause)
- Source: https://github.com/Yuma-Ichikawa/QQA4CO
- Concept DOI: https://doi.org/10.5281/zenodo.19648231
- Companion paper: Ichikawa, Y. & Arai, Y.,
  *"Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based
  Sampling"*, ICLR 2025
  ([OpenReview](https://openreview.net/forum?id=9EfBeXaXf0),
  [arXiv:2409.02135](https://arxiv.org/abs/2409.02135)).

If you use this reproduction pipeline in academic work, please cite
QQA4CO alongside Del Bono, Ricci-Tersenghi & Zamponi (2025); the
recommended BibTeX entries are listed in the top-level `README.md`
("Citation" section) and in `Reproduction/README.md` §4.5.
