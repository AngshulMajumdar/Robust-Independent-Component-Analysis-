from pathlib import Path
import argparse
import json
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("ica_batch_*.csv"))
    if not files:
        raise FileNotFoundError(f"No batch files in {args.input_dir}")
    result = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    result.to_csv(args.output_dir / "ica_real_benchmark_full.csv", index=False)

    summary = result.groupby(["dataset", "epsilon", "method"], as_index=False).agg(
        mean_corr=("mean_abs_corr", "mean"),
        std_corr=("mean_abs_corr", "std"),
        mean_amari=("amari_error", "mean"),
        std_amari=("amari_error", "std"),
        runtime=("runtime_sec", "mean"),
        success_rate=("success", "mean"),
    )
    summary.to_csv(args.output_dir / "ica_real_benchmark_summary.csv", index=False)

    overall = result.groupby("method", as_index=False).agg(
        mean_corr=("mean_abs_corr", "mean"),
        mean_amari=("amari_error", "mean"),
        runtime=("runtime_sec", "mean"),
        success_rate=("success", "mean"),
    )
    overall["rank_corr"] = overall["mean_corr"].rank(ascending=False)
    overall["rank_amari"] = overall["mean_amari"].rank(ascending=True)
    overall["avg_rank"] = (overall["rank_corr"] + overall["rank_amari"]) / 2
    overall.sort_values("avg_rank").to_csv(
        args.output_dir / "ica_real_benchmark_overall.csv", index=False
    )

    by_epsilon = result.groupby(["epsilon", "method"], as_index=False).agg(
        mean_corr=("mean_abs_corr", "mean"),
        mean_amari=("amari_error", "mean"),
        runtime=("runtime_sec", "mean"),
        success_rate=("success", "mean"),
    )
    by_epsilon.to_csv(args.output_dir / "ica_real_benchmark_by_epsilon.csv", index=False)

    print(json.dumps({"batch_files": len(files), "rows": len(result)}, indent=2))


if __name__ == "__main__":
    main()
