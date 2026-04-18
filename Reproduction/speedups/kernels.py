"""Drop-in optimised variants of the paper's hot GPU kernels.

Principle: every optimisation here must be **observably equivalent** to the
upstream code when driven with the same PyTorch RNG state. Concretely, this
means the *order and shape* of ``torch.rand`` / ``torch.bernoulli`` calls is
preserved bit-for-bit, and the arithmetic is mathematically identical up to
floating-point associativity (multiplication by -1 is exact in IEEE 754).

Activate by calling ``install()`` before importing any of the annealing
modules. Call ``uninstall()`` to restore the originals.

Summary of changes
------------------
1. ``monte_carlo_update_fast`` — *avoid an O(pop*N) clone per half-sweep.*
   The original builds ``proposed_population = population.clone()`` and
   flips the indexed columns. The optimised version computes the local
   field once (same einsum contraction as upstream so cuBLAS picks the
   identical GEMM kernel), then flips spins via
   ``torch.where(accept, -current_sigma, current_sigma)``. The accept /
   reject decision is bit-identical because ΔE is built from the same
   numbers and the RNG call ``torch.rand(pop_size, |indices|)`` is
   unchanged.

2. ``MLMC_fast`` — *drop `torch.cuda.empty_cache()` from the hot loop and
   drop the initial ``data.clone()``.* The ``empty_cache`` call is a
   device-sync / allocator reset with zero effect on tensor values; the
   ``data.clone()`` allocated a pop×N tensor that was read-only in the
   loop and immediately overwritten on the first iteration.

3. ``generate_config_fast`` — *drop `torch.cuda.empty_cache()` from the
   per-spin autoregressive loop.* At L=10 the original called it 1 000
   times per generated batch.

4. ``made.forward`` — *drop `torch.cuda.empty_cache()` from every forward
   pass.* This method runs during MADE pre-training (40 epochs × many
   batches), during every MLMC acceptance step (two forwards per move),
   and during every single-spin autoregressive evaluation via
   ``model.forward_n``'s sibling call sites. It was by far the most
   frequent empty-cache call in the pipeline.

None of these changes consume RNG, alter tensor values, or change the
order of floating-point reductions. See ``verify.py`` for an automated
bit-equivalence check and ``bench.py`` for the wall-clock comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Code" / "Legacy" / "packages"))
sys.path.insert(0, str(REPO_ROOT / "Code" / "Modern" / "optimization"))

import monte_carlo as _mc          # noqa: E402
import global_steps as _gs         # noqa: E402
import global_annealing as _ga     # noqa: E402
import made as _made               # noqa: E402


# ---------------------------------------------------------------------------
# 1. Optimised checkerboard Metropolis update.
# ---------------------------------------------------------------------------
def monte_carlo_update_fast_opt(pop, J, beta, even_indices, odd_indices):
    """Same semantics as ``monte_carlo.monte_carlo_update_fast`` but allocates
    only a small ``[pop_size, |indices|]`` tensor per half-sweep instead of a
    full ``[pop_size, N]`` clone of ``population``.
    """
    population = pop.clone()
    for indices in (even_indices, odd_indices):
        current_sigma = population[:, indices]
        # Use the same einsum contraction as upstream so cuBLAS picks the
        # identical GEMM kernel and every ΔE is bit-identical. The outer
        # einsum("ki, ki->ki", ...) in upstream is pure element-wise multiply;
        # we fold it into Python-level arithmetic, which maps to torch.mul and
        # therefore produces the same tensor.
        h = torch.einsum("kj, ji->ki", population, J[indices, :].T)
        # Upstream computes  -2 * proposed_sigma * h with proposed_sigma = -current_sigma,
        # i.e. +2 * current_sigma * h. Negation of a finite float is exact in IEEE 754,
        # so the two forms are bit-identical.
        delta_E = -2.0 * (-current_sigma) * h
        random_vals = torch.rand(population.shape[0], len(indices),
                                 device=population.device)
        acceptance_prob = torch.exp(-beta * delta_E)
        accept = (delta_E < 0) | (random_vals < acceptance_prob)
        population[:, indices] = torch.where(accept, -current_sigma,
                                             current_sigma)
    return population


# ---------------------------------------------------------------------------
# 2. Optimised autoregressive sampler.
# ---------------------------------------------------------------------------
def generate_config_fast_opt(model, N_spins, N_config, J):
    """Drop ``torch.cuda.empty_cache()`` from the per-spin loop."""
    with torch.no_grad():
        config = torch.zeros((N_config, N_spins), device="cuda")
        for n in range(N_spins):
            probs = model.forward_n(config, n)
            config[:, n] = torch.bernoulli(probs) * 2 - 1
    return config


# ---------------------------------------------------------------------------
# 3. Optimised MLMC Metropolis-Hastings.
# ---------------------------------------------------------------------------
def MLMC_fast_opt(model, data, beta, N, J, num_steps=10, return_correlations=False):
    """Same as ``global_annealing.MLMC_fast`` but without
    ``torch.cuda.empty_cache()`` inside the loop."""
    acc_rates = []
    if return_correlations:
        correlations = [1]
    with torch.no_grad():
        bce = nn.BCELoss(reduction="none")
        # The upstream code calls ``data.clone()`` here but never writes to
        # ``current_config`` in-place — every loop iteration re-binds it to a
        # fresh tensor built by einsum — so the clone is pure waste (a
        # pop_size × N allocation per MLMC invocation, of which there are
        # num_steps_MC × num_temps per GA run).
        current_config = data
        for _ in range(num_steps):
            new_config = _gs.generate_config_fast(model, N, len(data), J)

            current_energy = _mc.compute_energy(current_config, J)
            current_probability = torch.sum(
                bce(model(current_config), (current_config + 1) / 2), axis=1)

            new_energy = _mc.compute_energy(new_config, J)
            new_probability = torch.sum(
                bce(model(new_config), (new_config + 1) / 2), axis=1)

            arg_new = -beta * new_energy + new_probability
            arg_current = -beta * current_energy + current_probability

            acceptances = (torch.log(torch.rand(size=(len(data),), device="cuda"))
                           < (arg_new - arg_current)).int()
            current_config = (torch.einsum("i, ij->ij", (1 - acceptances), current_config)
                              + torch.einsum("i, ij->ij", acceptances, new_config))
            acc_rates.append(torch.sum(acceptances) / len(data))
            if return_correlations:
                correlations.append(float(
                    torch.mean(data * current_config)
                    - torch.mean(data) * torch.mean(current_config)))
    if return_correlations:
        return current_config, acc_rates, correlations
    else:
        return current_config, acc_rates


# ---------------------------------------------------------------------------
# 4. Optimised MADE forward.
# ---------------------------------------------------------------------------
def _made_forward_opt(self, x):
    """``made.made.forward`` without the per-call ``torch.cuda.empty_cache()``.

    The upstream implementation calls ``empty_cache()`` on every forward pass.
    That fires during MADE pre-training (≈ 40 epochs × many batches) and
    twice per MLMC acceptance step — by far the most frequent empty-cache
    site in the GA pipeline. The call has zero effect on tensor values or
    RNG state.
    """
    x = self.layer(x)
    x = self.activation(2 * x)
    return x


# ---------------------------------------------------------------------------
# Install / uninstall helpers.
# ---------------------------------------------------------------------------
_originals: Dict[str, Callable] = {}


def install() -> None:
    """Monkey-patch the optimised kernels into the upstream modules."""
    if _originals:
        return  # already installed
    _originals["mc_update"] = _mc.monte_carlo_update_fast
    _originals["gen_cfg_fast"] = _gs.generate_config_fast
    _originals["MLMC_fast"] = _ga.MLMC_fast
    _originals["made_forward"] = _made.made.forward
    _mc.monte_carlo_update_fast = monte_carlo_update_fast_opt
    _gs.generate_config_fast = generate_config_fast_opt
    _ga.MLMC_fast = MLMC_fast_opt
    _made.made.forward = _made_forward_opt

    # The annealing modules imported the originals by name via ``from X import *``.
    import simulated_annealing as _sa, population_annealing as _pa
    for mod in (_sa, _pa, _ga):
        if hasattr(mod, "monte_carlo_update_fast"):
            mod.monte_carlo_update_fast = monte_carlo_update_fast_opt
    if hasattr(_ga, "generate_config_fast"):
        _ga.generate_config_fast = generate_config_fast_opt


def uninstall() -> None:
    if not _originals:
        return
    _mc.monte_carlo_update_fast = _originals["mc_update"]
    _gs.generate_config_fast = _originals["gen_cfg_fast"]
    _ga.MLMC_fast = _originals["MLMC_fast"]
    _made.made.forward = _originals["made_forward"]
    import simulated_annealing as _sa, population_annealing as _pa
    for mod in (_sa, _pa, _ga):
        if hasattr(mod, "monte_carlo_update_fast"):
            mod.monte_carlo_update_fast = _originals["mc_update"]
    if hasattr(_ga, "generate_config_fast"):
        _ga.generate_config_fast = _originals["gen_cfg_fast"]
    _originals.clear()
