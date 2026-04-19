# Vendored QQA (Quasi-Quantum Annealing) Library

This directory contains a **verbatim copy** of the `qqa/` Python package from the
public QQA4CO reference implementation by Fujitsu Limited. It is vendored here so
that the `mlmc_optimization` reproduction code is fully self-contained (no external
checkout of QQA4CO is required).

## Origin

- Upstream repository : QQA4CO (Quasi-Quantum Annealing for Combinatorial Optimization)
- Upstream path      : `QQA4CO/src/qqa/`
- Upstream reference : Ichikawa, Y. et al., *"Continuous Tensor Relaxation for Finding
  Diverse Solutions in Combinatorial Optimization Problems"*, 2024.
- Copyright (c) 2025 Fujitsu Limited (see `LICENCE.txt` in this directory).

## License

The vendored code is redistributed under the BSD-3-Clause-style licence from
Fujitsu Limited reproduced verbatim in `LICENCE.txt`. That notice must be
preserved together with this code.

## Modifications

None. The files in this directory are byte-identical to the upstream `qqa/`
package at the time of vendoring. If local modifications are ever required,
they MUST be recorded here (file, line range, rationale) to satisfy the BSD
redistribution clause.

## Usage from `mlmc_optimization`

```python
# Add the third_party directory to sys.path so that `import qqa` works.
import sys, pathlib
VENDOR = pathlib.Path(__file__).resolve().parents[1] / "third_party"
sys.path.insert(0, str(VENDOR))
import qqa
```

The benchmark runners under `Reproduction/code/benchmark_3d_ea.py` and
`Reproduction/code/hyper_search_3d_ea.py` already perform this path injection,
so downstream SLURM jobs do not need to set `PYTHONPATH`.
