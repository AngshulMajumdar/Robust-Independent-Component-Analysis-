"""Reproduce contamination-geometry, magnitude, ablation, and significance studies.

This script imports the exact model, datasets, metrics, and baseline implementations
from run_main_benchmark.py. It writes raw and aggregated CSV files.
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def load_core(path: Path):
    source = path.read_text()
    # Load definitions only; avoid executing the main benchmark body.
    marker = "dsets={'Digits'"
    if marker not in source:
        raise RuntimeError("Could not locate benchmark execution block.")
    namespace = {"__file__": str(path), "__name__": "robust_ica_core"}
    exec(source[: source.index(marker)], namespace)
    return namespace


def contam_geometry(X, eps, seed, magnitude=10.0, geometry="dense"):
    rng = np.random.default_rng(seed + 17000)
    Y = X.copy()
    k = int(round(eps * X.shape[1]))
    if k == 0:
        return Y
    idx = rng.choice(X.shape[1], size=k, replace=False)
    scale = np.median(np.linalg.norm(X, axis=0)) + 1e-12
    amp = magnitude * scale * rng.uniform(0.8, 1.2, size=k)
    if geometry == "dense":
        D = rng.normal(size=(X.shape[0], k))
        D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    elif geometry == "single-channel":
        D = np.zeros((X.shape[0], k))
        rows = rng.integers(0, X.shape[0], size=k)
        D[rows, np.arange(k)] = rng.choice([-1.0, 1.0], size=k)
    elif geometry == "coherent":
        v = rng.normal(size=(X.shape[0], 1))
        v /= np.linalg.norm(v) + 1e-12
        D = v @ rng.choice([-1.0, 1.0], size=(1, k))
    else:
        raise ValueError(geometry)
    Y[:, idx] += D * amp
    return Y


def run_method(fn, Y, seed, S, A, core):
    start = time.time()
    try:
        Sh, W = fn(Y, seed)
        corr = float(core["align_corr"](S, Sh))
        ae = float(core["amari"](W @ A))
        if not np.isfinite(corr + ae):
            raise ValueError("nonfinite metric")
        return corr, ae, time.time() - start, True, ""
    except Exception as exc:
        return np.nan, np.nan, time.time() - start, False, f"{type(exc).__name__}:{exc}"


def proposed_variant(X, seed, variant, core):
    R = core["R"]
    X0 = X - X.mean(1, keepdims=True)
    norms = np.linalg.norm(X0, axis=0)
    keep = norms <= np.quantile(norms, 0.85)
    if variant == "No robust initialization":
        U, _, _ = np.linalg.svd(X0, full_matrices=False)
    else:
        U, _, _ = np.linalg.svd(X0[:, keep], full_matrices=False)
    A = U[:, :R]
    S = A.T @ X0
    lam = 0.0 if variant == "No source penalty" else 0.03
    delta = max(1e-4, 1e-3 * np.median(np.linalg.norm(X0 - A @ S, axis=0)))
    previous = np.inf
    for _ in range(40):
        residual = X0 - A @ S
        if variant == "Uniform weights":
            w = np.full(X0.shape[1], 0.5)
        else:
            w = 1 / (2 * np.sqrt(np.sum(residual * residual, 0) + delta * delta))
        L = 2 * np.linalg.norm(A, 2) ** 2 * np.max(w) + max(lam, 1e-8)
        eta = 0.7 / (L + 1e-12)
        for _ in range(8):
            G = 2 * A.T @ ((A @ S - X0) * w[None, :])
            if lam > 0:
                G += lam * np.tanh(S)
            G -= G.mean(1, keepdims=True)
            S -= eta * G
            S -= S.mean(1, keepdims=True)
        SD = S * w[None, :]
        A = (X0 * w[None, :]) @ S.T @ np.linalg.inv(SD @ S.T + 1e-6 * np.eye(R))
        scale = np.linalg.norm(A, axis=0) + 1e-12
        A /= scale
        S *= scale[:, None]
        objective = np.sum(np.sqrt(np.sum((X0 - A @ S) ** 2, 0) + delta**2) - delta)
        if lam > 0:
            objective += lam * np.log(np.cosh(np.clip(S, -20, 20))).sum()
        if abs(previous - objective) / max(1, abs(previous)) < 1e-7:
            break
        previous = objective
    if variant == "No ICA rotation":
        W = np.linalg.pinv(A)
        return W @ X0, W
    residual = X0 - A @ S
    w = 1 / (2 * np.sqrt(np.sum(residual * residual, 0) + delta * delta))
    good = w >= np.quantile(w, 0.25)
    try:
        K, B, _ = core["picard"](
            S[:, good], n_components=R, ortho=False, extended=True,
            whiten=True, max_iter=200, tol=1e-5, random_state=seed
        )
        W = (B @ K) @ np.linalg.pinv(A)
    except Exception:
        W = np.linalg.pinv(A)
    return W @ X0, W


def aggregate(df, groups):
    return df.groupby(groups, as_index=False).agg(
        mean_corr=("mean_abs_corr", "mean"), std_corr=("mean_abs_corr", "std"),
        mean_amari=("amari_error", "mean"), std_amari=("amari_error", "std"),
        runtime=("runtime_sec", "mean"), success_rate=("success", "mean")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, default=Path(__file__).with_name("run_main_benchmark.py"))
    parser.add_argument("--output", type=Path, default=Path("results_reproduced"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--main-results", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    core = load_core(args.core)
    datasets = {
        "Digits": core["load_digits"]().data,
        "Wine": core["load_wine"]().data,
        "BreastCancer": core["load_breast_cancer"]().data,
        "Diabetes": core["load_diabetes"]().data,
        "NaturalImages": None,
    }
    selected = {
        "Proposed IRLS-l12": core["methods"]["Proposed IRLS-l12"],
        "FastICA-exp": core["methods"]["FastICA-exp"],
        "FastICA-deflation-cube": core["methods"]["FastICA-deflation-cube"],
        "Picard-nonorthogonal": core["methods"]["Picard-nonorthogonal"],
        "JADE": core["methods"]["JADE"],
        "SOBI": core["methods"]["SOBI"],
    }
    def sources(D, seed):
        return core["natural_image_sources"](150, seed) if D is None else core["real_tabular_sources"](D, 150, seed)

    geometry_rows = []
    for name, D in datasets.items():
        for seed in range(args.seeds):
            S = sources(D, seed); A = core["mixing"](seed); X = A @ S
            for geometry in ["dense", "single-channel", "coherent"]:
                for eps in [0.10, 0.20]:
                    Y = contam_geometry(X, eps, seed, 10.0, geometry)
                    for method, fn in selected.items():
                        corr, ae, runtime, success, error = run_method(fn, Y, seed, S, A, core)
                        geometry_rows.append(locals() | {"dataset": name, "epsilon": eps, "mean_abs_corr": corr, "amari_error": ae, "runtime_sec": runtime})
    geometry = pd.DataFrame([{k: r[k] for k in ["dataset","seed","epsilon","geometry","method","mean_abs_corr","amari_error","runtime_sec","success","error"]} for r in geometry_rows])
    geometry.to_csv(args.output / "ica_geometry_full.csv", index=False)
    aggregate(geometry, ["geometry", "epsilon", "method"]).to_csv(args.output / "ica_geometry_summary.csv", index=False)

    magnitude_rows = []
    for name, D in datasets.items():
        for seed in range(args.seeds):
            S = sources(D, seed); A = core["mixing"](seed); X = A @ S
            for magnitude in [5.0, 10.0, 20.0]:
                Y = contam_geometry(X, 0.10, seed, magnitude, "dense")
                for method in ["Proposed IRLS-l12", "FastICA-deflation-cube", "Picard-nonorthogonal", "JADE"]:
                    corr, ae, runtime, success, error = run_method(selected[method], Y, seed, S, A, core)
                    magnitude_rows.append({"dataset":name,"seed":seed,"magnitude":magnitude,"method":method,"mean_abs_corr":corr,"amari_error":ae,"runtime_sec":runtime,"success":success,"error":error})
    magnitude = pd.DataFrame(magnitude_rows)
    magnitude.to_csv(args.output / "ica_magnitude_full.csv", index=False)
    aggregate(magnitude, ["magnitude", "method"]).to_csv(args.output / "ica_magnitude_summary.csv", index=False)

    ablation_rows = []
    variants = ["Full", "Uniform weights", "No source penalty", "No robust initialization", "No ICA rotation"]
    for name, D in datasets.items():
        for seed in range(args.seeds):
            S = sources(D, seed); A = core["mixing"](seed); X = A @ S
            Y = contam_geometry(X, 0.10, seed, 10.0, "dense")
            for variant in variants:
                fn = core["proposed"] if variant == "Full" else (lambda Z, sd, v=variant: proposed_variant(Z, sd, v, core))
                corr, ae, runtime, success, error = run_method(fn, Y, seed, S, A, core)
                ablation_rows.append({"dataset":name,"seed":seed,"variant":variant,"mean_abs_corr":corr,"amari_error":ae,"runtime_sec":runtime,"success":success,"error":error})
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(args.output / "ica_ablation_full.csv", index=False)
    aggregate(ablation, ["variant"]).to_csv(args.output / "ica_ablation_summary.csv", index=False)

    if args.main_results is not None:
        main_df = pd.read_csv(args.main_results)
        contaminated = main_df[main_df["epsilon"] > 0]
        proposed = contaminated[contaminated["method"] == "Proposed IRLS-l12"][["dataset","seed","epsilon","amari_error","mean_abs_corr"]]
        rows = []
        for metric, alternative in [("amari_error", "less"), ("mean_abs_corr", "greater")]:
            tests = []
            for competitor in sorted(set(contaminated["method"]) - {"Proposed IRLS-l12"}):
                comp = contaminated[contaminated["method"] == competitor][["dataset","seed","epsilon",metric]]
                pair = proposed.merge(comp, on=["dataset","seed","epsilon"], suffixes=("_proposed", "_competitor")).dropna()
                x = pair[f"{metric}_proposed"].to_numpy(); y = pair[f"{metric}_competitor"].to_numpy()
                stat, p = wilcoxon(x, y, alternative=alternative)
                tests.append((competitor, len(pair), float(np.mean(x-y)), stat, p))
            reject, corrected, _, _ = multipletests([t[4] for t in tests], method="holm")
            for test, p_holm, significant in zip(tests, corrected, reject):
                rows.append({"metric":metric,"competitor":test[0],"n_pairs":test[1],"mean_proposed_minus_competitor":test[2],"wilcoxon_statistic":test[3],"p_raw":test[4],"p_holm":p_holm,"significant_0_05":bool(significant),"alternative":alternative})
        pd.DataFrame(rows).to_csv(args.output / "ica_significance_holm.csv", index=False)


if __name__ == "__main__":
    main()
