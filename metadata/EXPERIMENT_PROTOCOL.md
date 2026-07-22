# Robust ICA experiment protocol

## Main benchmark
- Real-source collections: Digits, Wine, Breast Cancer Wisconsin, Diabetes, and five natural images distributed with scikit-image.
- For tabular collections, five nonconstant real feature marginals are selected and independently permuted to form independent non-Gaussian sources while preserving each real empirical marginal.
- Sources: 5; sensors: 8; samples per run: 150.
- Mixing matrices: random well-conditioned overdetermined matrices with unit-norm columns.
- Contamination: dense samplewise additive outliers at fractions 0, 0.05, 0.10, 0.20, 0.30; magnitude 10 times the median clean sample norm.
- Repetitions: 10 seeds.
- Methods: proposed IRLS-l1,2 plus ten named ICA/BSS baselines.
- Metrics: mean absolute source correlation after optimal assignment, Amari error, runtime, and success rate.
- Main run count: 2,750.

## Contamination-geometry study
- Geometries: dense isotropic, single-channel, coherent common-direction.
- Contamination fraction: 0.10.
- Five datasets, ten seeds.
- Methods: proposed, FastICA-deflation-cube, Picard-nonorthogonal, JADE, SOBI.
- Run count: 750.

## Outlier-magnitude study
- Magnitudes: 5, 10, 20 times median clean sample norm.
- Dense contamination fraction: 0.10.
- Five datasets, five seeds.
- Methods: proposed, FastICA-deflation-cube, Picard-nonorthogonal, JADE.
- Run count: 300.

## Ablation study
- Variants: full model, uniform weights, no source penalty, no robust initialization, no final ICA rotation.
- Dense contamination fraction: 0.10; magnitude 10.
- Five datasets, five seeds.
- Run count: 125.

## Statistical testing
- Paired Wilcoxon signed-rank tests across all 200 contaminated main-benchmark pairs per competitor.
- One-sided alternatives: lower Amari error and higher source correlation for the proposed method.
- Holm family-wise correction across ten competitors, separately for each metric.

## Important interpretation
The source-correlation and Amari metrics measure different aspects. The proposed method is significantly better in Amari error against every baseline after Holm correction, but it is not significantly superior in source correlation. Claims must preserve this distinction.
