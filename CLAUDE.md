# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

> **All commands must be run inside the virtual environment.** The project depends on compiled C extensions in `pyTsetlinMachineParallel` and system fonts (CMU Serif) that are only available in the container environment.

```bash
source venv/bin/activate
```

All scripts must be run from the **repo root** — the code uses relative paths (e.g., `data/`, `results/`) everywhere.

CUDA errors on startup are expected and can be ignored if running CPU-only.

### Run experiments

```bash
python src/main.py
# or
bash main.sh
```

### Postprocessing (tables + charts only, no training)

```bash
python src/postprocessing.py
```

### Grid search for hyperparameters

```bash
python src/grid_search.py
```

## Architecture

This project implements **knowledge distillation (KD) for Tsetlin Machines** — training a small student TM using soft labels from a larger teacher TM. There are two KD methods:

### 1. Distribution-based KD (DKD / paper 2)
- Teacher generates soft probability distributions over classes (via temperature scaling)
- Student is initialized with the teacher's most important clauses (`z` parameter controls what fraction)
- Student trains with a weighted mix of hard labels and soft teacher labels (`alpha` parameter)
- Key hyperparams: `temperature` (τ), `alpha` (α), `z` (clause initialization fraction)

### 2. Clause-based KD (CKD / paper 1)
- Teacher's clauses are directly transplanted into the student
- `downsample` parameter controls what fraction of teacher clauses are copied
- Student trained normally after clause initialization

### Experiment flow (distillation.py)

Each experiment trains three models for comparison:
1. **Teacher baseline** — large TM trained normally
2. **Student baseline** — small TM trained normally (no KD)
3. **Distilled model** — small TM trained with KD from teacher

For clause-based KD, a 4th model (**distilled_ds**) is also trained with downsampled clauses.

### Naming in outputs and tables

In JSON/output and code, the three comparison models use internal names that differ from paper labels:

| Internal (JSON, code) | Display (tables, figures) |
|-----------------------|----------------------------|
| `teacher`             | Teacher                    |
| `student`             | **Baseline** (small TM, no KD) |
| `distilled`           | **Student** (small TM with KD)  |

So "student" in `output.json` / `aggregated_output.json` (e.g. `avg_acc_test_student`) is the **baseline** small model; "distilled" is the **KD student** we compare against it. Postprocessing uses a `display_names` mapping (e.g. in `make_formatted_tables`, `make_combined_graphs_aggregate`) so tables and charts show "Baseline" and "Student" and avoid confusion.

Results are saved to a directory named by experiment parameters (e.g., `MNIST_tC1000_sC100_tT10_sT10_ts4.0_ss4.0_te120_se240_temp3.0_a0.5_z0.3`).

### Aggregate experiments

`aggregate_distribution_distillation_experiment(n, ...)` runs `n` independent trials and saves each run to a numbered subdirectory (`_n1`, `_n2`, ...). An `aggregated_output.json` is computed with mean/std across all runs.

### Key files

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — defines experiment configs and runs them |
| `src/distillation.py` | Core experiment logic: `distribution_distillation_experiment`, `clause_distillation_experiment`, `aggregate_distribution_distillation_experiment`, `plot_results` |
| `src/datasets.py` | Dataset classes (all inherit from `Dataset` ABC); data is booleanized for TM compatibility |
| `src/postprocessing.py` | Post-hoc chart and LaTeX table generation: `make_paper_2_tables_aggregate`, `make_formatted_tables` (writes `combined_test_table.tex`, `combined_train_table.tex` — datasets as columns, metrics as rows, with accuracy/time break), `make_combined_graphs_aggregate`. The old `ttest_table.tex` is no longer generated. |
| `src/grid_search.py` | Hyperparameter search utility |
| `src/__init__.py` | Shared constants: file path constants, column name constants, plot settings |
| `src/util.py` | I/O helpers: `load_or_create` (pkl cache), `save_json`/`load_json`, `save_pkl`/`load_pkl` |
| `src/activation_maps.py` | Visualization of TM clause activation patterns (image datasets only) |

### Results directory structure

```
results/
  distribution/          # single-run DKD experiments
  clause/                # single-run CKD experiments
  aggregate_distribution/ # multi-run DKD experiments
    MNIST/
      aggregated_output.json
      MNIST_..._n1/      # per-run subdirectory
      MNIST_..._n2/
      ...
combined_results/        # archived/combined results used for paper figures
assets/
  paper_1/               # CKD figures and .tex tables
  paper_2/               # DKD figures and .tex tables
```

Each experiment directory contains:
- `output.json` — all results, params, and analysis
- `results.csv` — per-epoch metrics
- `teacher_baseline.pkl`, `student_baseline.pkl`, `distilled.pkl` — serialized TM models
- PNG plots for accuracy and timing

### Data caching

Datasets are expensive to prepare (especially IMDB N-gram encoding). `load_or_create` in `util.py` caches them as pkl files under `data/`. Delete the pkl to force a rebuild.

### TM parameters

Tsetlin Machine hyperparameters follow this naming convention throughout the codebase:
- `C` — number of clauses
- `T` — threshold
- `s` — specificity
- `epochs` — training epochs
- Teacher params are prefixed `t` in folder names (e.g., `tC1000`), student params prefixed `s` (e.g., `sC100`)

### `overwrite` flag

Experiments check if the output directory already exists. Set `"overwrite": False` to skip completed runs (default), or `True` to rerun and overwrite.
