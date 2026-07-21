#!/usr/bin/env python3
"""Evaluate recovery of five simulated gene patterns using EMBER only."""

import argparse
from pathlib import Path
import anndata as ad
from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom


TRUTH = {
    "Category 1: CT1": (0, 199),
    "Category 2: CT1 + CT2": (200, 399),
    "Category 3: Housekeeping": (400, 599),
    "Category 4: StrainA:CT1": (600, 799),
    "Category 5: StrainA:CT1 + StrainB:CT2": (800, 999),
}
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim-output",
        default=str(SCRIPT_DIR / "results"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--celltype-psi-cutoff", type=float, default=0.5)
    parser.add_argument("--strain-celltype-psi-cutoff", type=float, default=0.5)

    # These must exactly match columns in the corresponding Psi-block files.
    parser.add_argument("--ct1-block", default="CT1")
    parser.add_argument("--ct2-block", default="CT2")
    parser.add_argument("--strain-a-ct1-block", default="StrainA:CT1")
    parser.add_argument("--strain-a-ct2-block", default="StrainA:CT2")
    parser.add_argument("--strain-b-ct2-block", default="StrainB:CT2")
    parser.add_argument(
        "--housekeeping-expression-cutoff",
        type=float,
        default=1.0,
        help="Minimum mean raw count for housekeeping-gene eligibility.",
    )
    parser.add_argument(
        "--h5ad",
        default=str(SCRIPT_DIR / "results" / "simulation_data.h5ad"),
    )
    return parser.parse_args()


def standardize_gene_column(frame):
    for column in ["gene", "gene_name", "Gene"]:
        if column in frame.columns:
            return frame.rename(columns={column: "gene"})

    unnamed = [
        column for column in frame.columns
        if str(column).startswith("Unnamed")
    ]
    if unnamed:
        return frame.rename(columns={unnamed[0]: "gene"})

    first = frame.columns[0]
    if frame[first].astype(str).str.startswith("gene_").mean() > 0.5:
        return frame.rename(columns={first: "gene"})

    raise KeyError(f"Could not identify gene column in {list(frame.columns)}")


def load_block_matrix(path):
    raw = pd.read_csv(path)
    gene_columns = [
        column for column in raw.columns
        if str(column).startswith("gene_")
    ]

    # Transposed format: blocks are rows and genes are columns.
    if len(gene_columns) > 10:
        block_column = next(
            column for column in raw.columns
            if column not in gene_columns
        )
        matrix = (
            raw.set_index(block_column)[gene_columns]
            .apply(pd.to_numeric, errors="coerce")
            .T
        )
        matrix.index.name = "gene"
        return matrix

    # Standard format: genes are rows and blocks are columns.
    frame = standardize_gene_column(raw)
    return frame.set_index("gene").apply(pd.to_numeric, errors="coerce")


def load_partition(sim_output, partition):
    ember_dir = sim_output / "ember"
    metrics_path = ember_dir / f"pvals_entropy_metrics_{partition}.csv"
    blocks_path = (
        ember_dir
        / "Psi_block_df"
        / f"mean_Psi_block_df_{partition}.csv"
    )

    metrics = standardize_gene_column(pd.read_csv(metrics_path))

    def find_column(kind):
        normalized = {
            column: "".join(ch for ch in str(column).lower() if ch.isalnum())
            for column in metrics.columns
        }
        if kind == "pvalue":
            matches = [
                column for column, name in normalized.items()
                if "psi" in name
                and ("pval" in name or "pvalue" in name)
                and "qval" not in name
                and "qvalue" not in name
            ]
        else:
            matches = [
                column for column, name in normalized.items()
                if "psi" in name
                and ("qval" in name or "qvalue" in name)
            ]
        if len(matches) != 1:
            raise KeyError(
                f"Could not uniquely identify Psi {kind} column in "
                f"{metrics_path}. Candidates: {matches}. "
                f"All columns: {list(metrics.columns)}"
            )
        return matches[0]

    psi_pvalue = find_column("pvalue")
    psi_qvalue = find_column("qvalue")
    print(
        f"{partition}: using {psi_pvalue!r} and {psi_qvalue!r} "
        "for Psi significance"
    )

    metrics = metrics.rename(columns={
        psi_pvalue: "Psi_pvalue",
        psi_qvalue: "Psi_qvalue",
    })
    metrics = metrics.set_index("gene")[[
        "Psi", "Zeta", "Psi_pvalue", "Psi_qvalue"
    ]]
    metrics = metrics.apply(pd.to_numeric, errors="coerce")

    blocks = load_block_matrix(blocks_path)
    metrics.index = metrics.index.astype(str)
    blocks.index = blocks.index.astype(str)
    return metrics.join(blocks, how="inner")


def require_blocks(frame, block_names, partition):
    missing = [block for block in block_names if block not in frame.columns]
    if missing:
        available = [c for c in frame.columns if c not in ["Psi", "Zeta"]]
        raise KeyError(
            f"Missing {partition} block(s): {missing}\n"
            f"Available blocks: {available}"
        )


def build_scores(celltype, joint, mean_expression, args):
    require_blocks(
        celltype,
        [args.ct1_block, args.ct2_block],
        "cell_type",
    )
    require_blocks(
        joint,
        [
            args.strain_a_ct1_block,
            args.strain_b_ct2_block,
        ],
        "strain_celltype",
    )

    scores = {}

    celltype_eligible = (
        (celltype["Psi"] >= args.celltype_psi_cutoff)
        & (celltype["Psi_pvalue"] < args.alpha)
        & (celltype["Psi_qvalue"] < args.alpha)
    )
    joint_eligible = (
        (joint["Psi"] >= args.strain_celltype_psi_cutoff)
        & (joint["Psi_pvalue"] < args.alpha)
        & (joint["Psi_qvalue"] < args.alpha)
    )

    # No composite score is created. Each table retains the EMBER columns that
    # will be used directly by pandas.sort_values().
    scores["Category 1: CT1"] = pd.DataFrame({
        "gene": celltype.index,
        "Psi": celltype["Psi"],
        "Psi_pvalue": celltype["Psi_pvalue"],
        "Psi_qvalue": celltype["Psi_qvalue"],
        "Psi_block_1": celltype[args.ct1_block],
        "ranking_value": celltype[args.ct1_block],
        "eligible": celltype_eligible,
    })

    # Category 2: high Psi and high Psi_block for both CT1 and CT2.
    scores["Category 2: CT1 + CT2"] = pd.DataFrame({
        "gene": celltype.index,
        "Psi": celltype["Psi"],
        "Psi_pvalue": celltype["Psi_pvalue"],
        "Psi_qvalue": celltype["Psi_qvalue"],
        "Psi_block_1": celltype[args.ct1_block],
        "Psi_block_2": celltype[args.ct2_block],
        # Both blocks must be high; the weaker block determines the rank.
        "ranking_value": celltype[
            [args.ct1_block, args.ct2_block]
        ].min(axis=1),
        "eligible": celltype_eligible,
    })

    # Category 3: high Psi, low Zeta, and sufficiently high expression.
    # Do not require Psi p/q significance here because a uniformly expressed
    # housekeeping gene may not differ from the partition-permutation null.
    housekeeping_eligible = (
        (celltype["Psi"] >= args.celltype_psi_cutoff)
        & (
            mean_expression.reindex(celltype.index)
            >= args.housekeeping_expression_cutoff
        )
    )

    scores["Category 3: Housekeeping"] = pd.DataFrame({
        "gene": celltype.index,
        "Psi": celltype["Psi"],
        "Psi_pvalue": celltype["Psi_pvalue"],
        "Psi_qvalue": celltype["Psi_qvalue"],
        "Zeta": celltype["Zeta"],
        "mean_expression": mean_expression.reindex(celltype.index),
        # Eligible genes are ranked from lowest to highest Zeta.
        "ranking_value": celltype["Zeta"],
        "eligible": housekeeping_eligible,
    })

    # Eligible strain × cell-type genes:
    # high Psi and significant Psi p/q-values in the joint partition.
    joint_eligible = (
        (joint["Psi"] >= args.strain_celltype_psi_cutoff)
        & (joint["Psi_pvalue"] < args.alpha)
        & (joint["Psi_qvalue"] < args.alpha)
    )

    # Category 4:
    # High Psi and high Psi_block for StrainA:CT1.
    scores["Category 4: StrainA:CT1"] = pd.DataFrame({
        "gene": joint.index,
        "Psi": joint["Psi"],
        "Psi_pvalue": joint["Psi_pvalue"],
        "Psi_qvalue": joint["Psi_qvalue"],
        "StrainA_CT1_Psi_block": joint[args.strain_a_ct1_block],
        "ranking_value": joint[args.strain_a_ct1_block],
        "eligible": joint_eligible,
    })

    # Category 5:
    # High in both StrainA:CT1 and StrainB:CT2.
    #
    # The minimum is high only when both requested blocks are high.
    scores["Category 5: StrainA:CT1 + StrainB:CT2"] = pd.DataFrame({
        "gene": joint.index,
        "Psi": joint["Psi"],
        "Psi_pvalue": joint["Psi_pvalue"],
        "Psi_qvalue": joint["Psi_qvalue"],
        "StrainA_CT1_Psi_block": joint[args.strain_a_ct1_block],
        "StrainB_CT2_Psi_block": joint[args.strain_b_ct2_block],
        "ranking_value": joint[
            [
                args.strain_a_ct1_block,
                args.strain_b_ct2_block,
            ]
        ].min(axis=1),
        "eligible": joint_eligible,
    })

    return scores


def evaluate(score_frame, category):
    start, end = TRUTH[category]
    # Score frames inherit a gene-named index from the EMBER tables while also
    # containing a gene column. Drop that redundant index so pandas does not
    # treat ``gene`` as an ambiguous sort key.
    frame = score_frame.copy().reset_index(drop=True)
    frame.index.name = None
    frame["gene"] = frame["gene"].astype(str)
    frame["gene_index"] = pd.to_numeric(
        frame["gene"].str.replace("gene_", "", regex=False),
        errors="coerce",
    )
    frame = frame[frame["gene_index"].between(0, 2999)].copy()

    if frame["gene_index"].nunique() != 3000:
        raise ValueError(
            f"{category}: expected 3,000 genes, found "
            f"{frame['gene_index'].nunique()}"
        )

    frame["is_target"] = frame["gene_index"].between(start, end)
    # Psi significance is an eligibility filter, not part of the ranking
    # score. Psi-block values have no p-values and are therefore not filtered.
    eligible = frame[frame["eligible"]].copy()

    # Psi is only an eligibility gate. Rank by the pattern-localizing readout.
    low_is_better = category == "Category 3: Housekeeping"
    eligible = eligible.sort_values(
        ["ranking_value", "gene"],
        ascending=[low_is_better, True],
    ).reset_index(drop=True)
    eligible["eligible_rank"] = np.arange(1, len(eligible) + 1)
    selected = eligible.head(200).copy()
    recovered = int(selected["is_target"].sum())
    return recovered, len(eligible), eligible, selected


def plot_recovery(results, output_dir):
    short_labels = [
        "1 CT",
        "2 CTs",
        "Housekeeping",
        "1 CT, 1 strain",
        "Strain switch",
    ]
    colors = ["#3B82F6", "#6366F1", "#10B981", "#F59E0B", "#EF4444"]

    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(
        short_labels,
        results["recovered"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    random_expected = 200 * (200 / 3000)
    axis.axhline(
        random_expected,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Random expectation ({random_expected:.1f}/200)",
    )

    axis.bar_label(
        bars,
        labels=[f"{value}/200" for value in results["recovered"]],
        padding=4,
        fontsize=10,
    )
    axis.set_ylim(0, 210)
    axis.set_ylabel("True genes recovered in EMBER top 200")
    axis.set_xlabel("Simulated specificity pattern")
    axis.set_title("EMBER recovery of simulated specificity patterns")
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    fig.savefig(output_dir / "ember_top200_recovery.png", dpi=300)
    fig.savefig(output_dir / "ember_top200_recovery.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    sim_output = Path(args.sim_output)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else sim_output / "ember_only_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    celltype = load_partition(sim_output, "cell_type")
    joint = load_partition(sim_output, "strain_celltype")
    adata = ad.read_h5ad(args.h5ad)
    raw_counts = adata.layers["raw_counts"]

    if sparse.issparse(raw_counts):
        gene_means = np.asarray(raw_counts.mean(axis=0)).ravel()
    else:
        gene_means = np.asarray(raw_counts).mean(axis=0)

    mean_expression = pd.Series(
        gene_means,
        index=adata.var_names.astype(str),
        name="mean_expression",
    )

    scores = build_scores(
        celltype,
        joint,
        mean_expression,
        args,
    )

    results = []
    all_rankings = []
    top_200 = []

    for category in TRUTH:
        recovered, n_eligible, ranked, selected = evaluate(
            scores[category], category
        )
        results.append({
            "category": category,
            "psi_significance_threshold": args.alpha,
            "psi_cutoff": (
                args.celltype_psi_cutoff
                if category in [
                    "Category 1: CT1",
                    "Category 2: CT1 + CT2",
                    "Category 3: Housekeeping",
                ]
                else args.strain_celltype_psi_cutoff
            ),
            "n_psi_significant_genes": n_eligible,
            "n_selected": len(selected),
            "recovered": recovered,
            "out_of": 200,
            "precision_at_200": (
                recovered / len(selected) if len(selected) else np.nan
            ),
            "recall_at_200": recovered / 200.0,
            "false_positives": len(selected) - recovered,
            "fold_enrichment_over_random": (
                recovered / (len(selected) * 200 / 3000)
                if len(selected) else np.nan
            ),
            "hypergeometric_pvalue": hypergeom.sf(
                recovered - 1,
                3000,  # total genes
                200,   # true genes in this category
                len(selected),
            ),
        })
        ranked["category"] = category
        selected["category"] = category
        all_rankings.append(ranked)
        top_200.append(selected)

    results = pd.DataFrame(results)
    results.to_csv(output_dir / "ember_recovery_results.csv", index=False)
    pd.concat(all_rankings, ignore_index=True).to_csv(
        output_dir / "ember_significant_gene_rankings.csv",
        index=False,
    )
    pd.concat(top_200, ignore_index=True).to_csv(
        output_dir / "ember_top_200.csv",
        index=False,
    )
    plot_recovery(results, output_dir)

    print(results.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")
    print("  ember_top200_recovery.png")
    print("  ember_top200_recovery.pdf")


if __name__ == "__main__":
    main()
