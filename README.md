# Robust ICA Made Easy — Reproducibility Package

This archive accompanies **“Robust ICA Made Easy: IRLS with $\ell_{1,2}$ Data Fidelity.”** It contains the exact CPU implementation, benchmark scripts, protocol, and numerical results used in the article.

## Scope

The archive contains only the publication-grade experiments. The earlier internal smoke-test results are deliberately excluded.

The reported experiment suite comprises:

- Main benchmark: 5 real-source collections, 5 contamination fractions, 10 seeds, 11 methods (2,750 runs).
- Contamination geometry: dense isotropic, single-channel, and coherent corruption.
- Outlier magnitude: 5, 10, and 20 times the median clean sample norm.
- Ablation: full method, uniform weights, no source penalty, no robust initialization, and no final ICA rotation.
- Paired Wilcoxon signed-rank tests with Holm correction.

## Repository layout

```
code/
  run_main_benchmark.py          complete main benchmark
  run_main_batch.py              CPU-safe batched execution
  aggregate_main_results.py      merge batches and produce tables
  run_supplementary_studies.py   geometry, magnitude, ablation, significance
results/
  main/                          raw and aggregate main benchmark results
  geometry/                      raw and aggregate geometry results
  magnitude/                     reported magnitude summary
  ablation/                      reported ablation summary
  statistics/                    Holm-corrected tests
metadata/
  EXPERIMENT_PROTOCOL.md
  ica_real_benchmark_machine.json
```

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate robust-ica-mir
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `picard` import is supplied by the PyPI package `python-picard`.

## Reproducing the main benchmark

A direct run is:

```bash
python code/run_main_benchmark.py
```

The script writes its CSVs to `/mnt/data` because that was the execution directory used for the reported run. For a portable CPU-safe run, use the batched script from a working directory and edit its output prefix, or run the ten batches in the original environment:

```bash
for b in $(seq 0 9); do
  python code/run_main_batch.py "$b"
done
```

Then aggregate:

```bash
python code/aggregate_main_results.py \
  --input-dir /mnt/data \
  --output-dir reproduced/main
```

## Reproducing supplementary studies

```bash
python code/run_supplementary_studies.py \
  --core code/run_main_benchmark.py \
  --output reproduced/supplementary \
  --seeds 10 \
  --main-results results/main/ica_real_benchmark_full.csv
```

This command writes the raw geometry, magnitude, and ablation runs, their aggregate tables, and the corrected significance tests.

## Data access

No external manual download is required. The tabular datasets are loaded through `scikit-learn`; the natural images are distributed with `scikit-image`.

## Reproducibility notes

- Python random seeds are fixed per configuration.
- Sources: 5; sensors: 8; sample length: 150.
- All runs are CPU-only.
- Runtime values depend on processor, BLAS backend, and thread configuration; accuracy metrics should be reproducible up to ordinary numerical tolerance.
- The raw main and geometry runs are included. The archived magnitude and ablation files are the exact aggregate tables reported in the article; the script regenerates their complete raw rows.
- The manuscript distinguishes Amari error from matched source correlation. The proposed method is significantly better in Amari error after Holm correction, but not universally superior in matched source correlation.

## Suggested citation

Please cite the accompanying article and the public arXiv record associated with this archive.
