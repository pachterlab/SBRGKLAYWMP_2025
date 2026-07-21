#!/usr/bin/env python3
"""Run the full simulation benchmark workflow and combine method outputs."""

import argparse
import importlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse


FINAL_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

TASK_ORDER = [
    "Category 1: CT1",
    "Category 2: CT1 + CT2",
    "Category 3: Housekeeping",
    "Category 4: StrainA:CT1",
    "Category 5: StrainA:CT1 + StrainB:CT2",
]
TASK_LABELS = {
    "Category 1: CT1": "1 CT",
    "Category 2: CT1 + CT2": "2 CTs",
    "Category 3: Housekeeping": "Housekeeping",
    "Category 4: StrainA:CT1": "1 CT, 1 strain",
    "Category 5: StrainA:CT1 + StrainB:CT2": "Strain switch",
}
METHOD_ORDER = [
    "ember",
    "DESeq2",
    "variancePartition",
    "Tau",
    "PEM",
    "Gini",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the simulation dataset, run ember-only simulation "
            "benchmarking, run each evaluator, and combine results."
        )
    )
    parser.add_argument(
        "--run-name",
        default=datetime.now().strftime("simulation_%Y%m%d_%H%M%S"),
        help="Name of the subdirectory created under --results-root.",
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_DIR,
        help="Root directory for final benchmark runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing run directory.",
    )

    parser.add_argument("--cells-per-strain-celltype", type=int, default=500)
    parser.add_argument("--mice-per-strain", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fold-change", type=float, default=8.0)
    parser.add_argument("--dispersion", type=float, default=8.0)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--min-detected-cells", type=int, default=100)

    parser.add_argument("--partition-col", default="cell_type")
    parser.add_argument("--category-col", default="strain")
    parser.add_argument("--interaction-col", default="strain_celltype")
    parser.add_argument("--sample-id-col", default="mouse_id")
    parser.add_argument("--n-cpus", type=int, default=4)
    parser.add_argument("--num-draws", type=int, default=16)
    parser.add_argument("--n-pval-iterations", type=int, default=1000)

    parser.add_argument("--r-env", default="ember_r_env")
    parser.add_argument("--variance-partition-cores", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=200)
    return parser.parse_args()


def run_command(command, description):
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)
    print(" ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def call_module_main(module_name, argv, description):
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)
    print(f"{module_name} {' '.join(str(part) for part in argv)}", flush=True)

    if str(FINAL_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(FINAL_EVAL_DIR))

    module = importlib.import_module(module_name)
    old_argv = sys.argv[:]
    try:
        sys.argv = [f"{module_name}.py", *[str(part) for part in argv]]
        module.main()
    finally:
        sys.argv = old_argv


def make_r_compatible_h5ad(h5ad_path, output_dir):
    output_path = output_dir / f"{h5ad_path.stem}_r_compatible.h5ad"
    if output_path.exists():
        return output_path

    adata = ad.read_h5ad(h5ad_path)
    adata.X = (
        adata.X.astype(np.float64)
        if sparse.issparse(adata.X)
        else np.asarray(adata.X, dtype=np.float64)
    )

    for layer_name in list(adata.layers.keys()):
        layer = adata.layers[layer_name]
        adata.layers[layer_name] = (
            layer.astype(np.float64)
            if sparse.issparse(layer)
            else np.asarray(layer, dtype=np.float64)
        )

    adata.write_h5ad(output_path)
    return output_path


def run_ember_only_stage(args, h5ad_path, sim_output_dir):
    if str(FINAL_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(FINAL_EVAL_DIR))
    from run_simulation_benchmark import ensure_interaction_h5ad

    ember_output_dir = sim_output_dir / "ember"
    ember_output_dir.mkdir(parents=True, exist_ok=True)

    analysis_h5ad_path = ensure_interaction_h5ad(
        h5ad_path=h5ad_path,
        output_dir=sim_output_dir,
        partition_col=args.partition_col,
        category_col=args.category_col,
        interaction_col=args.interaction_col,
    )

    for partition_col in dict.fromkeys([
        args.partition_col,
        args.interaction_col,
    ]):
        run_command(
            [
                "ember",
                "light_ember",
                analysis_h5ad_path,
                partition_col,
                ember_output_dir,
                "--sample_id_col",
                args.sample_id_col,
                "--category_col",
                args.category_col,
                "--n_cpus",
                args.n_cpus,
                "--num_draws",
                args.num_draws,
                "--n_pval_iterations",
                args.n_pval_iterations,
            ],
            f"Running ember for {partition_col}",
        )

    return analysis_h5ad_path


def load_method_results(path, method):
    frame = pd.read_csv(path)
    frame["method"] = method
    return frame


def combine_results(run_dir):
    sources = [
        (
            run_dir / "ember_only_evaluation" / "ember_recovery_results.csv",
            "ember",
        ),
        (
            run_dir / "deseq2_evaluation" / "deseq2_recovery_results.csv",
            "DESeq2",
        ),
        (
            run_dir
            / "variance_partition_evaluation"
            / "variance_partition_recovery_results.csv",
            "variancePartition",
        ),
    ]

    frames = [load_method_results(path, method) for path, method in sources]

    tau_pem_gini = pd.read_csv(
        run_dir
        / "tau_pem_gini_evaluation"
        / "tau_pem_gini_recovery_results.csv"
    )
    frames.append(tau_pem_gini)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["method"] = combined["method"].replace({"EMBER": "ember"})
    combined["method"] = pd.Categorical(
        combined["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    combined["category"] = pd.Categorical(
        combined["category"],
        categories=TASK_ORDER,
        ordered=True,
    )
    combined = combined.sort_values(["method", "category"]).reset_index(drop=True)

    combined_path = run_dir / "combined_recovery_results.csv"
    combined.to_csv(combined_path, index=False)
    return combined, combined_path


def plot_heatmap(combined, run_dir):
    combined = combined.copy()
    combined["method"] = combined["method"].astype(str).replace({"EMBER": "ember"})
    values = (
        combined
        .pivot_table(
            index="method",
            columns="category",
            values="recall_at_200",
            observed=False,
        )
        .reindex(index=METHOD_ORDER, columns=TASK_ORDER)
    )
    recovered = (
        combined
        .pivot_table(
            index="method",
            columns="category",
            values="recovered",
            observed=False,
        )
        .reindex(index=METHOD_ORDER, columns=TASK_ORDER)
    )
    labels = recovered.map(
        lambda value: "" if pd.isna(value) else f"{int(value)}/200"
    )

    fig, axis = plt.subplots(figsize=(11, 5.8))
    sns.heatmap(
        values,
        annot=labels,
        fmt="",
        vmin=0,
        vmax=1,
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Recall at 200"},
        ax=axis,
    )
    axis.set_xlabel("Simulated specificity pattern")
    axis.set_ylabel("Method")
    axis.set_title("Simulation Benchmark Recovery")
    axis.set_xticklabels(
        [TASK_LABELS[task] for task in TASK_ORDER],
        rotation=25,
        ha="right",
    )
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
    fig.tight_layout()

    png_path = run_dir / "combined_recovery_heatmap.png"
    pdf_path = run_dir / "combined_recovery_heatmap.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    run_dir = results_root / args.run_name
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Pass --overwrite or choose a different --run-name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = run_dir / "simulation_data.h5ad"
    call_module_main(
        "generate_sim",
        [
            "--output",
            h5ad_path,
            "--cells-per-strain-celltype",
            args.cells_per_strain_celltype,
            "--mice-per-strain",
            args.mice_per_strain,
            "--seed",
            args.seed,
            "--fold-change",
            args.fold_change,
            "--dispersion",
            args.dispersion,
            "--target-sum",
            args.target_sum,
            "--min-detected-cells",
            args.min_detected_cells,
        ],
        "Generating simulation dataset",
    )

    analysis_h5ad_path = run_ember_only_stage(args, h5ad_path, run_dir)

    call_module_main(
        "ember_only_eval",
        [
            "--sim-output",
            run_dir,
            "--output-dir",
            run_dir / "ember_only_evaluation",
            "--h5ad",
            analysis_h5ad_path,
        ],
        "Evaluating ember recovery",
    )

    r_h5ad_dir = run_dir / "r_input"
    r_h5ad_dir.mkdir(parents=True, exist_ok=True)
    r_h5ad_path = make_r_compatible_h5ad(analysis_h5ad_path, r_h5ad_dir)

    run_command(
        [
            "conda",
            "run",
            "-n",
            args.r_env,
            "--no-capture-output",
            "Rscript",
            FINAL_EVAL_DIR / "deseq_eval.R",
            r_h5ad_path,
            run_dir / "deseq2_evaluation",
        ],
        "Running DESeq2 simulation evaluation",
    )
    run_command(
        [
            "conda",
            "run",
            "-n",
            args.r_env,
            "--no-capture-output",
            "Rscript",
            FINAL_EVAL_DIR / "variance_partition_eval.R",
            r_h5ad_path,
            run_dir / "variance_partition_evaluation",
            args.variance_partition_cores,
        ],
        "Running variancePartition evaluation",
    )
    call_module_main(
        "evaluate_tau_pem_gini",
        [
            "--h5ad",
            analysis_h5ad_path,
            "--output-dir",
            run_dir / "tau_pem_gini_evaluation",
            "--top-n",
            args.top_n,
        ],
        "Evaluating Tau, PEM, and Gini",
    )

    combined, combined_path = combine_results(run_dir)
    heatmap_png, heatmap_pdf = plot_heatmap(combined, run_dir)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Run directory:     {run_dir}")
    print(f"Combined results:  {combined_path}")
    print(f"Heatmap PNG:       {heatmap_png}")
    print(f"Heatmap PDF:       {heatmap_pdf}")
    print("\nCombined recovery:")
    print(
        combined[
            ["method", "category", "recovered", "out_of", "recall_at_200"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(
            f"\nERROR: command exited with status {error.returncode}",
            file=sys.stderr,
        )
        sys.exit(error.returncode)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
