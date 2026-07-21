#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run EMBER, calculate Tau/PEM/Gini, and run "
            "DESeq2/variancePartition."
        )
    )

    parser.add_argument(
        "--h5ad",
        default=str(SCRIPT_DIR / "results" / "simulation_data.h5ad"),
        help="Input h5ad file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results"),
        help="Main output directory.",
    )
    parser.add_argument(
        "--partition-col",
        default="cell_type",
        help="EMBER partition and Tau/PEM/Gini category.",
    )
    parser.add_argument(
        "--category-col",
        default="strain",
        help="EMBER category column.",
    )
    parser.add_argument(
        "--interaction-col",
        default="strain_celltype",
        help=(
            "Combined strain/cell-type partition column. If absent, it is "
            "created from --category-col and --partition-col."
        ),
    )
    parser.add_argument(
        "--sample-id-col",
        default="mouse_id",
        help="Biological replicate column.",
    )
    parser.add_argument(
        "--r-script",
        default=str(SCRIPT_DIR / "run_deseq2_variance_partition.R"),
        help="R script for DESeq2 and variancePartition.",
    )
    parser.add_argument("--n-cpus", type=int, default=4)
    parser.add_argument("--num-draws", type=int, default=9)
    parser.add_argument("--n-pval-iterations", type=int, default=1000)

    return parser.parse_args()


def run_command(command, description):
    print("\n" + "=" * 72)
    print(description)
    print("=" * 72)
    print(" ".join(map(str, command)), flush=True)

    subprocess.run(
        [str(x) for x in command],
        check=True,
    )


def get_matrix(adata, layer=None):
    if layer is None:
        return adata.X

    if layer not in adata.layers:
        raise KeyError(
            f"Requested layer {layer!r} is not in adata.layers. "
            f"Available layers: {list(adata.layers.keys())}"
        )

    return adata.layers[layer]


def ensure_interaction_h5ad(
    h5ad_path,
    output_dir,
    partition_col,
    category_col,
    interaction_col,
):
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs_columns = set(adata.obs.columns)

    if interaction_col in obs_columns:
        adata.file.close()
        return h5ad_path

    if partition_col not in obs_columns or category_col not in obs_columns:
        adata.file.close()
        missing = [
            col
            for col in [partition_col, category_col]
            if col not in obs_columns
        ]
        raise KeyError(
            "Cannot create interaction column because these obs columns "
            f"are missing: {missing}"
        )

    interaction_values = (
        adata.obs[category_col].astype(str)
        + ":"
        + adata.obs[partition_col].astype(str)
    ).to_numpy()
    adata.file.close()

    adata = ad.read_h5ad(h5ad_path)
    adata.obs[interaction_col] = interaction_values

    interaction_h5ad = output_dir / "simulation_data_with_interaction.h5ad"
    adata.write_h5ad(interaction_h5ad)

    return interaction_h5ad


def make_r_compatible_h5ad(h5ad_path, output_dir):
    """
    zellkonverter/Matrix can reject sparse H5AD matrices whose x slot is
    stored as an integer or single-precision type. Keep count values intact,
    but store sparse payloads as doubles for the R import path.
    """
    r_h5ad_path = output_dir / (
        f"{h5ad_path.stem}_r_compatible.h5ad"
    )

    adata = ad.read_h5ad(h5ad_path)

    if sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float64)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float64)

    for layer_name in list(adata.layers.keys()):
        layer = adata.layers[layer_name]

        if sparse.issparse(layer):
            adata.layers[layer_name] = layer.astype(np.float64)
        else:
            adata.layers[layer_name] = np.asarray(
                layer,
                dtype=np.float64,
            )

    adata.write_h5ad(r_h5ad_path)

    return r_h5ad_path


def calculate_tau(expression):
    """
    Tau:
        0 = uniform expression across categories
        1 = expression specific to one category
    """
    n_categories = expression.shape[1]

    if n_categories < 2:
        raise ValueError("Tau requires at least two categories.")

    maximum = expression.max(axis=1)

    normalized = np.divide(
        expression,
        maximum[:, None],
        out=np.zeros_like(expression, dtype=float),
        where=maximum[:, None] > 0,
    )

    tau = np.sum(1.0 - normalized, axis=1) / (n_categories - 1)
    tau[maximum == 0] = np.nan

    return tau


def calculate_gini(expression):
    """
    Gini:
        0 = uniform expression
        larger value = more unequal/category-specific expression
    """
    expression = np.maximum(expression, 0)
    sorted_expression = np.sort(expression, axis=1)

    n_categories = sorted_expression.shape[1]
    totals = sorted_expression.sum(axis=1)
    ranks = np.arange(1, n_categories + 1)

    weighted_sum = (
        sorted_expression * ranks[None, :]
    ).sum(axis=1)

    gini = np.divide(
        2.0 * weighted_sum,
        n_categories * totals,
        out=np.full(sorted_expression.shape[0], np.nan),
        where=totals > 0,
    )

    gini -= (n_categories + 1.0) / n_categories

    return gini


def calculate_pem(summed_expression, pseudocount=1e-12):
    """
    PEM(g,c) = log2(observed(g,c) / expected(g,c))

    Expected expression is calculated under independence between
    gene identity and category identity.
    """
    observed = np.maximum(summed_expression, 0)

    gene_totals = observed.sum(axis=1, keepdims=True)
    category_totals = observed.sum(axis=0, keepdims=True)
    grand_total = observed.sum()

    if grand_total <= 0:
        return np.full_like(observed, np.nan, dtype=float)

    expected = gene_totals @ category_totals / grand_total

    return np.log2(
        (observed + pseudocount) /
        (expected + pseudocount)
    )


def aggregate_by_category(matrix, category_codes, n_categories):
    """
    Return genes × categories matrices containing sums and means.
    """
    n_cells = matrix.shape[0]

    membership = sparse.csr_matrix(
        (
            np.ones(n_cells, dtype=np.float64),
            (np.arange(n_cells), category_codes),
        ),
        shape=(n_cells, n_categories),
    )

    category_sizes = np.bincount(
        category_codes,
        minlength=n_categories,
    )

    # matrix: cells × genes
    # membership.T @ matrix: categories × genes
    summed = membership.T @ matrix

    if sparse.issparse(summed):
        summed = summed.toarray()
    else:
        summed = np.asarray(summed)

    summed = summed.T  # genes × categories

    means = np.divide(
        summed,
        category_sizes[None, :],
        out=np.zeros_like(summed, dtype=np.float64),
        where=category_sizes[None, :] > 0,
    )

    return summed, means, category_sizes


def run_specificity_metrics(
    h5ad_path,
    category_col,
    output_dir,
    layer=None,
):
    print("\n" + "=" * 72)
    print(f"Calculating Tau, Gini, and PEM for {category_col}")
    print("=" * 72)

    adata = ad.read_h5ad(h5ad_path)

    if category_col not in adata.obs.columns:
        raise KeyError(
            f"{category_col!r} is not in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    matrix = get_matrix(adata, layer=layer)

    categories = pd.Categorical(adata.obs[category_col])
    category_names = categories.categories.astype(str).to_numpy()
    category_codes = categories.codes

    if np.any(category_codes < 0):
        raise ValueError(
            f"{category_col!r} contains missing values."
        )

    summed, means, category_sizes = aggregate_by_category(
        matrix=matrix,
        category_codes=category_codes,
        n_categories=len(category_names),
    )

    tau = calculate_tau(means)
    gini = calculate_gini(means)
    pem = calculate_pem(summed)

    max_expression_index = np.argmax(means, axis=1)
    max_pem_index = np.argmax(pem, axis=1)

    gene_summary = pd.DataFrame({
        "gene": adata.var_names.astype(str),
        "Tau": tau,
        "Gini": gini,
        "max_expression_category": category_names[
            max_expression_index
        ],
        "max_mean_expression": means[
            np.arange(adata.n_vars),
            max_expression_index,
        ],
        "max_PEM_category": category_names[max_pem_index],
        "max_PEM": pem[
            np.arange(adata.n_vars),
            max_pem_index,
        ],
    })

    category_results = []

    for category_index, category in enumerate(category_names):
        category_results.append(
            pd.DataFrame({
                "gene": adata.var_names.astype(str),
                "category": category,
                "n_cells": category_sizes[category_index],
                "mean_expression": means[:, category_index],
                "summed_expression": summed[:, category_index],
                "PEM": pem[:, category_index],
            })
        )

    category_results = pd.concat(
        category_results,
        ignore_index=True,
    )

    metric_dir = output_dir / "tau_pem_gini"
    metric_dir.mkdir(parents=True, exist_ok=True)

    summary_path = metric_dir / (
        f"{category_col}_gene_summary.csv"
    )
    category_path = metric_dir / (
        f"{category_col}_category_scores.csv"
    )

    gene_summary.to_csv(summary_path, index=False)
    category_results.to_csv(category_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {category_path}")


def main():
    args = parse_args()

    h5ad_path = Path(args.h5ad).resolve()
    output_dir = Path(args.output_dir).resolve()
    r_script = Path(args.r_script).resolve()

    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"Input h5ad does not exist: {h5ad_path}"
        )

    if not r_script.exists():
        raise FileNotFoundError(
            f"R script does not exist: {r_script}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    ember_output_dir = output_dir / "ember"
    r_output_dir = output_dir / "r_results"

    ember_output_dir.mkdir(parents=True, exist_ok=True)
    r_output_dir.mkdir(parents=True, exist_ok=True)

    analysis_h5ad_path = ensure_interaction_h5ad(
        h5ad_path=h5ad_path,
        output_dir=output_dir,
        partition_col=args.partition_col,
        category_col=args.category_col,
        interaction_col=args.interaction_col,
    )

    # --------------------------------------------------------
    # 1. Run EMBER
    # --------------------------------------------------------
    ember_partition_cols = list(dict.fromkeys([
        args.partition_col,
        args.interaction_col,
    ]))

    for ember_partition_col in ember_partition_cols:
        ember_command = [
            "ember",
            "light_ember",
            str(analysis_h5ad_path),
            ember_partition_col,
            str(ember_output_dir),
            "--sample_id_col",
            args.sample_id_col,
            "--category_col",
            args.category_col,
            "--n_cpus",
            str(args.n_cpus),
            "--num_draws",
            str(args.num_draws),
            "--n_pval_iterations",
            str(args.n_pval_iterations),
        ]

        run_command(
            ember_command,
            f"Step 1: Running EMBER for {ember_partition_col}",
        )

    # --------------------------------------------------------
    # 2. Calculate Tau, PEM, and Gini
    # --------------------------------------------------------
    run_specificity_metrics(
        h5ad_path=analysis_h5ad_path,
        category_col=args.partition_col,
        output_dir=output_dir,
        layer=None,
    )

    run_specificity_metrics(
        h5ad_path=analysis_h5ad_path,
        category_col=args.interaction_col,
        output_dir=output_dir,
        layer=None,
    )

    # --------------------------------------------------------
    # 3. Run DESeq2 and variancePartition in R
    # --------------------------------------------------------
    r_h5ad_path = make_r_compatible_h5ad(
        h5ad_path=analysis_h5ad_path,
        output_dir=output_dir,
    )
    print(f"Prepared R-compatible h5ad: {r_h5ad_path}")

    r_command = [
        "conda",
        "run",
        "-n",
        "ember_r_env",
        "--no-capture-output",
        "Rscript",
        str(r_script),
        str(r_h5ad_path),
        str(r_output_dir),
        ]

    run_command(
        r_command,
        "Step 3: Running DESeq2 and variancePartition",
    )

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print(f"EMBER results:       {ember_output_dir}")
    print(f"Tau/PEM/Gini:        {output_dir / 'tau_pem_gini'}")
    print(f"DESeq2/variancePart: {r_output_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(
            f"\nERROR: command exited with status "
            f"{error.returncode}",
            file=sys.stderr,
        )
        sys.exit(error.returncode)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
