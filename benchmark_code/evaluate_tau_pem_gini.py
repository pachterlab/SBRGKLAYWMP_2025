#!/usr/bin/env python3
"""Evaluate Tau, PEM, and Gini on the five simulated specificity patterns."""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom


METHODS = ["Tau", "PEM", "Gini"]
TASKS = [
    "Category 1: CT1",
    "Category 2: CT1 + CT2",
    "Category 3: Housekeeping",
    "Category 4: StrainA:CT1",
    "Category 5: StrainA:CT1 + StrainB:CT2",
]
TRUTH = {
    "Category 1: CT1": (0, 199),
    "Category 2: CT1 + CT2": (200, 399),
    "Category 3: Housekeeping": (400, 599),
    "Category 4: StrainA:CT1": (600, 799),
    "Category 5: StrainA:CT1 + StrainB:CT2": (800, 999),
}
SHORT_LABELS = {
    "Category 1: CT1": "1 CT",
    "Category 2: CT1 + CT2": "2 CTs",
    "Category 3: Housekeeping": "Housekeeping",
    "Category 4: StrainA:CT1": "1 CT, 1 strain",
    "Category 5: StrainA:CT1 + StrainB:CT2": "Strain switch",
}
COLORS = ["#3B82F6", "#6366F1", "#10B981", "#F59E0B", "#EF4444"]
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5ad",
        default=str(SCRIPT_DIR / "results" / "simulation_data.h5ad"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results" / "tau_pem_gini_evaluation"),
    )
    parser.add_argument("--celltype-col", default="cell_type")
    parser.add_argument("--strain-col", default="strain")
    parser.add_argument("--joint-col", default="strain_celltype")
    parser.add_argument("--raw-counts-layer", default="raw_counts")
    parser.add_argument(
        "--housekeeping-expression-cutoff",
        type=float,
        default=1.0,
        help="Minimum mean raw count for housekeeping eligibility.",
    )
    parser.add_argument("--top-n", type=int, default=200)
    return parser.parse_args()


def aggregate(matrix, labels):
    categories = pd.Categorical(labels)
    if np.any(categories.codes < 0):
        raise ValueError("Partition contains missing category labels.")

    names = categories.categories.astype(str).tolist()
    codes = categories.codes
    membership = sparse.csr_matrix(
        (
            np.ones(len(codes), dtype=float),
            (np.arange(len(codes)), codes),
        ),
        shape=(len(codes), len(names)),
    )
    sums = membership.T @ matrix
    sums = sums.toarray() if sparse.issparse(sums) else np.asarray(sums)
    sums = sums.T
    sizes = np.bincount(codes, minlength=len(names))
    means = sums / sizes[None, :]
    return names, sums, means


def calculate_tau(means):
    maximum = means.max(axis=1)
    scaled = np.divide(
        means,
        maximum[:, None],
        out=np.zeros_like(means, dtype=float),
        where=maximum[:, None] > 0,
    )
    tau = np.sum(1.0 - scaled, axis=1) / (means.shape[1] - 1)
    tau[maximum == 0] = np.nan
    return tau


def calculate_gini(means):
    values = np.sort(np.maximum(means, 0), axis=1)
    n = values.shape[1]
    totals = values.sum(axis=1)
    weighted = (values * np.arange(1, n + 1)[None, :]).sum(axis=1)
    gini = np.divide(
        2.0 * weighted,
        n * totals,
        out=np.full(values.shape[0], np.nan),
        where=totals > 0,
    )
    return gini - (n + 1.0) / n


def calculate_pem(sums, pseudocount=1e-12):
    observed = np.maximum(sums, 0)
    gene_totals = observed.sum(axis=1, keepdims=True)
    category_totals = observed.sum(axis=0, keepdims=True)
    expected = gene_totals @ category_totals / observed.sum()
    return np.log2((observed + pseudocount) / (expected + pseudocount))


def build_metrics(adata, args):
    if args.joint_col not in adata.obs:
        adata.obs[args.joint_col] = (
            adata.obs[args.strain_col].astype(str)
            + ":"
            + adata.obs[args.celltype_col].astype(str)
        )

    expression = adata.X
    genes = adata.var_names.astype(str)
    metrics = {}

    for partition, column in [
        ("cell_type", args.celltype_col),
        ("strain_celltype", args.joint_col),
    ]:
        names, sums, means = aggregate(expression, adata.obs[column])
        metrics[partition] = {
            "Tau": pd.Series(calculate_tau(means), index=genes),
            "Gini": pd.Series(calculate_gini(means), index=genes),
            "PEM": pd.DataFrame(
                calculate_pem(sums),
                index=genes,
                columns=names,
            ),
        }

    if args.raw_counts_layer not in adata.layers:
        raise KeyError(
            f"Missing layer {args.raw_counts_layer!r}. "
            f"Available layers: {list(adata.layers.keys())}"
        )
    raw = adata.layers[args.raw_counts_layer]
    raw_mean = (
        np.asarray(raw.mean(axis=0)).ravel()
        if sparse.issparse(raw)
        else np.asarray(raw).mean(axis=0)
    )
    metrics["mean_raw_count"] = pd.Series(raw_mean, index=genes)
    return metrics


def require_columns(matrix, columns, partition):
    missing = [column for column in columns if column not in matrix.columns]
    if missing:
        raise KeyError(
            f"Missing {partition} categories {missing}. "
            f"Available: {list(matrix.columns)}"
        )


def make_ranking(method, task, metrics, expression_cutoff):
    mean_expression = metrics["mean_raw_count"]
    housekeeping = task == "Category 3: Housekeeping"
    partition = (
        "cell_type"
        if task in TASKS[:3]
        else "strain_celltype"
    )

    if method in ["Tau", "Gini"]:
        values = metrics[partition][method]
        frame = pd.DataFrame({
            "gene": values.index,
            "ranking_value": values.to_numpy(),
            method: values.to_numpy(),
            "mean_raw_count": mean_expression.reindex(values.index).to_numpy(),
        })
        if housekeeping:
            frame = frame[frame["mean_raw_count"] >= expression_cutoff]
            ascending = True
        else:
            ascending = False

    elif method == "PEM":
        pem = metrics[partition]["PEM"]

        if task == "Category 1: CT1":
            require_columns(pem, ["CT1"], partition)
            frame = pd.DataFrame({
                "gene": pem.index,
                "PEM_CT1": pem["CT1"],
                "ranking_value": pem["CT1"],
            })
            ascending = False

        elif task == "Category 2: CT1 + CT2":
            require_columns(pem, ["CT1", "CT2"], partition)
            frame = pd.DataFrame({
                "gene": pem.index,
                "PEM_CT1": pem["CT1"],
                "PEM_CT2": pem["CT2"],
                # Both categories must be enriched; the weaker one sets rank.
                "ranking_value": pem[["CT1", "CT2"]].min(axis=1),
            })
            ascending = False

        elif housekeeping:
            # Uniform genes have PEM near zero in every cell type.
            maximum_absolute_pem = pem.abs().max(axis=1)
            frame = pd.DataFrame({
                "gene": pem.index,
                "max_absolute_PEM": maximum_absolute_pem,
                "ranking_value": maximum_absolute_pem,
                "mean_raw_count": mean_expression.reindex(pem.index),
            })
            frame = frame[frame["mean_raw_count"] >= expression_cutoff]
            ascending = True

        elif task == "Category 4: StrainA:CT1":
            require_columns(pem, ["StrainA:CT1"], partition)
            frame = pd.DataFrame({
                "gene": pem.index,
                "PEM_StrainA_CT1": pem["StrainA:CT1"],
                "ranking_value": pem["StrainA:CT1"],
            })
            ascending = False

        else:
            targets = ["StrainA:CT1", "StrainB:CT2"]
            require_columns(pem, targets, partition)
            frame = pd.DataFrame({
                "gene": pem.index,
                "PEM_StrainA_CT1": pem[targets[0]],
                "PEM_StrainB_CT2": pem[targets[1]],
                "ranking_value": pem[targets].min(axis=1),
            })
            ascending = False
    else:
        raise ValueError(method)

    frame = frame.reset_index(drop=True)
    frame["ranking_value"] = pd.to_numeric(
        frame["ranking_value"], errors="coerce"
    )
    frame = frame[np.isfinite(frame["ranking_value"])].copy()
    frame = frame.sort_values(
        ["ranking_value", "gene"],
        ascending=[ascending, True],
    ).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["method"] = method
    frame["category"] = task
    return frame


def evaluate(ranking, task, top_n):
    top = ranking.head(top_n).copy()
    gene_index = pd.to_numeric(
        top["gene"].str.replace("gene_", "", regex=False),
        errors="coerce",
    )
    start, end = TRUTH[task]
    recovered = int(gene_index.between(start, end).sum())
    selected = len(top)
    random_expected = selected * 200 / 3000
    return top, {
        "category": task,
        "n_selected": selected,
        "recovered": recovered,
        "out_of": 200,
        "precision_at_200": recovered / selected if selected else np.nan,
        "recall_at_200": recovered / 200,
        "false_positives": selected - recovered,
        "fold_enrichment_over_random": (
            recovered / random_expected if random_expected else np.nan
        ),
        "hypergeometric_pvalue": hypergeom.sf(
            recovered - 1, 3000, 200, selected
        ),
    }


def plot_results(results, method, output_dir):
    subset = results[results["method"] == method].set_index("category").loc[TASKS]
    labels = [SHORT_LABELS[task] for task in TASKS]
    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(
        labels,
        subset["recovered"],
        color=COLORS,
        edgecolor="black",
        linewidth=0.8,
    )
    random_expected = 200 * 200 / 3000
    axis.axhline(
        random_expected,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Random expectation ({random_expected:.1f}/200)",
    )
    axis.bar_label(
        bars,
        labels=[f"{value}/200" for value in subset["recovered"]],
        padding=4,
        fontsize=10,
    )
    axis.set_ylim(0, 210)
    axis.set_ylabel(f"True genes recovered in {method} top 200")
    axis.set_xlabel("Simulated specificity pattern")
    axis.set_title(f"{method} recovery of simulated specificity patterns")
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    stem = method.lower()
    fig.savefig(output_dir / f"{stem}_top200_recovery.png", dpi=300)
    fig.savefig(output_dir / f"{stem}_top200_recovery.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)
    if adata.n_vars != 3000:
        raise ValueError(f"Expected 3,000 genes; found {adata.n_vars}.")
    metrics = build_metrics(adata, args)

    result_rows = []
    top_frames = []
    for method in METHODS:
        for task in TASKS:
            ranking = make_ranking(
                method,
                task,
                metrics,
                args.housekeeping_expression_cutoff,
            )
            top, result = evaluate(ranking, task, args.top_n)
            result["method"] = method
            result_rows.append(result)
            top_frames.append(top)

    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "tau_pem_gini_recovery_results.csv", index=False)
    all_top = pd.concat(top_frames, ignore_index=True, sort=False)
    all_top.to_csv(output_dir / "tau_pem_gini_top_200.csv", index=False)

    for method in METHODS:
        results[results["method"] == method].to_csv(
            output_dir / f"{method.lower()}_recovery_results.csv",
            index=False,
        )
        all_top[all_top["method"] == method].to_csv(
            output_dir / f"{method.lower()}_top_200.csv",
            index=False,
        )
        plot_results(results, method, output_dir)

    print(results.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
