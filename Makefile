# One-liner entry points for the reproduction pipeline.
# All targets are safe to invoke from the repository root.

PY := .venv/bin/python
SUBMIT := sbatch --exclude=kagura-gpu07
EASY_SEED := 1736329224
HARD_SEED := 310411727
EASY_CSV := Reproduction/fresh_runs/sweep_L10_seed$(EASY_SEED).csv
HARD_CSV := Reproduction/fresh_runs/sweep_L10_seed$(HARD_SEED).csv
PQQA_CSV := Reproduction/fresh_runs/winning/qqa_winner_G1.csv

.PHONY: help install smoke sweep-l6 sweep-l10-easy sweep-l10-hard \
        sweep-l10-all verify-bench plot-easy plot-hard plots clean-figures \
        pqqa-winner plot-pqqa-vs-ga test-mc-polish

help:
	@echo "Entry points (see Reproduction/README.md for the full story):"
	@echo
	@echo "  make install          uv sync (one-time dependency install)"
	@echo "  make smoke            submit scripts/smoketest.sbatch (~2 min)"
	@echo "  make sweep-l6         submit Reproduction/scripts/sweep_L6.sbatch (~5 min)"
	@echo "  make sweep-l10-easy   submit L=10 sweep on seed $(EASY_SEED) (~30 min)"
	@echo "  make sweep-l10-hard   submit L=10 sweep on seed $(HARD_SEED) (~30 min)"
	@echo "  make sweep-l10-all    submit both L=10 sweeps"
	@echo "  make verify-bench     submit equivalence + speedup benchmark"
	@echo "  make plot-easy        render figures/success_vs_time_L10_easy.png"
	@echo "  make plot-hard        render figures/success_vs_time_L10_hard.png"
	@echo "  make plots            render both figures"
	@echo
	@echo "  --- PQQA (Parallel Quasi-Quantum Annealing) winner ----"
	@echo "  make pqqa-winner      submit the winning PQQA config (n=50, ~30 min)"
	@echo "                        100% success @ 31.88 s vs GA's 47.73 s (33.2% faster)"
	@echo "  make plot-pqqa-vs-ga  render figures/pqqa_vs_ga_pareto_L10_hard.png"
	@echo "  make test-mc-polish   CPU bit-equivalence test for the fp32 path"
	@echo
	@echo "  make clean-figures    rm Reproduction/figures/*.{png,pdf,stats.csv}"

install:
	uv sync

smoke:
	$(SUBMIT) scripts/smoketest.sbatch

sweep-l6:
	$(SUBMIT) Reproduction/scripts/sweep_L6.sbatch

sweep-l10-easy:
	$(SUBMIT) Reproduction/scripts/sweep_L10.sbatch

sweep-l10-hard:
	$(SUBMIT) Reproduction/scripts/sweep_L10_hard.sbatch

sweep-l10-all: sweep-l10-easy sweep-l10-hard

verify-bench:
	$(SUBMIT) Reproduction/scripts/verify_and_bench.sbatch

plot-easy: $(EASY_CSV)
	$(PY) Reproduction/code/plot_success_vs_time.py \
	    --csv $(EASY_CSV) \
	    --out Reproduction/figures/success_vs_time_L10_easy.png \
	    --title "L=10 easy instance (seed $(EASY_SEED)) – $$M=2^{13}$$ population"

plot-hard: $(HARD_CSV)
	$(PY) Reproduction/code/plot_success_vs_time.py \
	    --csv $(HARD_CSV) \
	    --out Reproduction/figures/success_vs_time_L10_hard.png \
	    --title "L=10 hard instance (seed $(HARD_SEED)) – $$M=2^{13}$$ population"

plots: plot-easy plot-hard

# --- PQQA winner over GA -----------------------------------------------------

pqqa-winner:
	$(SUBMIT) Reproduction/scripts/qqa_winner_run.sbatch

plot-pqqa-vs-ga: $(HARD_CSV) $(PQQA_CSV)
	$(PY) Reproduction/code/plot_pqqa_winner.py \
	    --baseline-csv $(HARD_CSV) \
	    --winner-csv   $(PQQA_CSV) \
	    --out          Reproduction/figures/pqqa_vs_ga_pareto_L10_hard.png

test-mc-polish:
	$(PY) Reproduction/code/test_mc_polish_correctness.py

clean-figures:
	rm -f Reproduction/figures/*.png \
	      Reproduction/figures/*.pdf \
	      Reproduction/figures/*.stats.csv
