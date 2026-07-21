#!/usr/bin/env python3
"""Compare ember and PEM using pseudobulk-identical single-cell decoys."""

import argparse
import subprocess
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


N_PAIRS = 200
TRUE_START = 800
DECOY_START = 1000
TARGET_BLOCKS = ["StrainA:CT1", "StrainB:CT2"]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
METHOD_ORDER = ["ember", "DESeq2", "variancePartition", "Tau", "PEM", "Gini"]
CATEGORY_5 = "Category 5: StrainA:CT1 + StrainB:CT2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5ad",
        default=None,
        help=(
            "Clean simulation h5ad. Defaults to the newest "
            "final_eval/results/*/simulation_data.h5ad."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to singlecell_vs_pseudobulk inside "
            "the selected final_eval run directory."
        ),
    )
    parser.add_argument("--active-cell-fraction", type=float, default=0.10)
    parser.add_argument("--psi-cutoff", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-cpus", type=int, default=4)
    parser.add_argument("--num-draws", type=int, default=16)
    parser.add_argument("--n-pval-iterations", type=int, default=1000)
    parser.add_argument("--r-env", default="ember_r_env")
    parser.add_argument("--variance-partition-cores", type=int, default=4)
    parser.add_argument(
        "--skip-ember",
        action="store_true",
        help="Reuse ember results already present in OUTPUT_DIR/ember.",
    )
    parser.add_argument(
        "--skip-deseq2",
        action="store_true",
        help="Reuse DESeq2 results already present in OUTPUT_DIR/deseq2.",
    )
    parser.add_argument(
        "--skip-variance-partition",
        action="store_true",
        help=(
            "Reuse variancePartition results already present in "
            "OUTPUT_DIR/variance_partition."
        ),
    )
    return parser.parse_args()


def find_latest_final_eval_h5ad():
    candidates = sorted(
        RESULTS_DIR.glob("*/simulation_data.h5ad"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        direct_default = RESULTS_DIR / "simulation_data.h5ad"
        if direct_default.exists():
            return direct_default
        raise FileNotFoundError(
            "Could not find a final_eval simulation h5ad. Expected one of:\n"
            f"  {RESULTS_DIR}/*/simulation_data.h5ad\n"
            f"  {direct_default}\n"
            "Run run_final_benchmark.py first or pass --h5ad explicitly."
        )
    return candidates[0]


def resolve_input_h5ad(h5ad_arg):
    if h5ad_arg:
        h5ad_path = Path(h5ad_arg).expanduser().resolve()
    else:
        h5ad_path = find_latest_final_eval_h5ad().resolve()

    if not h5ad_path.exists():
        raise FileNotFoundError(f"Input h5ad does not exist: {h5ad_path}")
    return h5ad_path


def resolve_output_dir(output_dir_arg, h5ad_path):
    if output_dir_arg:
        return Path(output_dir_arg).expanduser().resolve()

    try:
        h5ad_path.relative_to(RESULTS_DIR)
    except ValueError:
        return RESULTS_DIR / "singlecell_vs_pseudobulk"

    return h5ad_path.parent / "singlecell_vs_pseudobulk"


def dense_integer_counts(adata):
    if "raw_counts" not in adata.layers:
        raise KeyError("The input h5ad must contain layers['raw_counts'].")
    raw = adata.layers["raw_counts"]
    values = raw.toarray() if sparse.issparse(raw) else np.asarray(raw)
    if not np.allclose(values, np.round(values)):
        raise ValueError("raw_counts contains non-integer-valued entries.")
    return np.round(values).astype(np.int32)


def create_matched_decoys(adata, cell_fraction, rng):
    """Overwrite genes 1000-1199 with paired rare-cell-driven decoys.

    For pair j, gene_(1000+j) has exactly the same total raw count as
    gene_(800+j) in every mouse x strain-cell-type pseudobulk. Within the two
    target blocks, however, the decoy total is redistributed into only a small
    fraction of individual cells.
    """
    if not 0 < cell_fraction < 1:
        raise ValueError("--active-cell-fraction must be between 0 and 1.")
    if adata.n_vars < DECOY_START + N_PAIRS:
        raise ValueError("The h5ad does not contain genes 1000-1199.")

    obs = adata.obs.copy()
    if "strain_celltype" not in obs:
        obs["strain_celltype"] = (
            obs["strain"].astype(str)
            + ":"
            + obs["cell_type"].astype(str)
        )
        adata.obs["strain_celltype"] = obs["strain_celltype"]

    counts = dense_integer_counts(adata)

    # Copy the complete coherent-gene single-cell vector first. Counts are
    # subsequently redistributed only in the two target joint blocks.
    counts[:, DECOY_START:DECOY_START + N_PAIRS] = counts[
        :, TRUE_START:TRUE_START + N_PAIRS
    ]

    for pair_index in range(N_PAIRS):
        source = TRUE_START + pair_index
        decoy = DECOY_START + pair_index

        for target_block in TARGET_BLOCKS:
            block_mice = obs.loc[
                obs["strain_celltype"].astype(str).eq(target_block),
                "mouse_id",
            ].unique()

            for mouse_id in block_mice:
                cells = np.flatnonzero(
                    obs["strain_celltype"].astype(str).eq(target_block).to_numpy()
                    & obs["mouse_id"].astype(str).eq(str(mouse_id)).to_numpy()
                )
                n_active = max(1, int(round(cell_fraction * len(cells))))
                active = rng.choice(cells, size=n_active, replace=False)
                total = int(counts[cells, source].sum())

                counts[cells, decoy] = 0
                if total:
                    counts[active, decoy] = rng.multinomial(
                        total,
                        np.full(n_active, 1.0 / n_active),
                    )

    verify_pseudobulk_equality(counts, obs)

    counts_sparse = sparse.csr_matrix(counts, dtype=np.int32)
    adata.layers["raw_counts"] = counts_sparse.astype(np.float64)
    adata.X = counts_sparse.astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.uns.pop("log1p", None)

    # AnnData commonly restores string metadata as pandas Categoricals. Convert
    # this column before introducing a truth label that was not present in the
    # original category levels.
    if "truth_category" not in adata.var.columns:
        adata.var["truth_category"] = "Null/background"
    else:
        adata.var["truth_category"] = (
            adata.var["truth_category"].astype("string").astype(object)
        )

    adata.var.loc[
        adata.var_names[DECOY_START:DECOY_START + N_PAIRS],
        "truth_category",
    ] = "Aggregation-confounded rare-cell decoy"
    adata.var["paired_gene"] = ""
    adata.var.loc[
        adata.var_names[DECOY_START:DECOY_START + N_PAIRS],
        "paired_gene",
    ] = [f"gene_{i}" for i in range(TRUE_START, TRUE_START + N_PAIRS)]
    adata.var["active_cell_fraction"] = np.nan
    adata.var.loc[
        adata.var_names[DECOY_START:DECOY_START + N_PAIRS],
        "active_cell_fraction",
    ] = cell_fraction
    return adata


def verify_pseudobulk_equality(counts, obs):
    sample_group = (
        obs["mouse_id"].astype(str)
        + "__"
        + obs["strain_celltype"].astype(str)
    )
    categories = pd.Categorical(sample_group)
    membership = sparse.csr_matrix(
        (
            np.ones(len(obs)),
            (np.arange(len(obs)), categories.codes),
        ),
        shape=(len(obs), len(categories.categories)),
    )
    pseudobulk = membership.T @ sparse.csr_matrix(counts)
    true_totals = pseudobulk[:, TRUE_START:TRUE_START + N_PAIRS].toarray()
    decoy_totals = pseudobulk[:, DECOY_START:DECOY_START + N_PAIRS].toarray()
    if not np.array_equal(true_totals, decoy_totals):
        difference = np.abs(true_totals - decoy_totals).max()
        raise AssertionError(
            f"True/decoy pseudobulks are not identical; max difference={difference}."
        )
    print(
        "Verified exact raw-count equality for all 200 true/decoy pairs "
        "in every mouse x strain-cell-type pseudobulk."
    )


def run_ember(h5ad_path, ember_dir, args):
    command = [
        "ember",
        "light_ember",
        str(h5ad_path),
        "strain_celltype",
        str(ember_dir),
        "--sample_id_col",
        "mouse_id",
        "--category_col",
        "strain",
        "--n_cpus",
        str(args.n_cpus),
        "--num_draws",
        str(args.num_draws),
        "--n_pval_iterations",
        str(args.n_pval_iterations),
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def make_r_compatible_h5ad(h5ad_path, output_dir):
    output_path = output_dir / f"{h5ad_path.stem}_r_compatible.h5ad"

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


def run_rscript(script_path, h5ad_path, output_dir, args, extra_args=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "conda",
        "run",
        "-n",
        args.r_env,
        "--no-capture-output",
        "Rscript",
        str(script_path),
        str(h5ad_path),
        str(output_dir),
    ]
    if extra_args:
        command.extend(str(value) for value in extra_args)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def standardize_gene_column(frame):
    for column in ["gene", "gene_name", "Gene"]:
        if column in frame:
            return frame.rename(columns={column: "gene"})
    unnamed = [column for column in frame if str(column).startswith("Unnamed")]
    if unnamed:
        return frame.rename(columns={unnamed[0]: "gene"})
    first = frame.columns[0]
    if frame[first].astype(str).str.startswith("gene_").mean() > 0.5:
        return frame.rename(columns={first: "gene"})
    raise KeyError(f"Cannot identify gene column in {list(frame.columns)}")


def find_psi_significance_columns(frame):
    normalized = {
        column: "".join(ch for ch in str(column).lower() if ch.isalnum())
        for column in frame.columns
    }
    p_columns = [
        column for column, name in normalized.items()
        if "psi" in name
        and ("pval" in name or "pvalue" in name)
        and "qval" not in name
        and "qvalue" not in name
    ]
    q_columns = [
        column for column, name in normalized.items()
        if "psi" in name and ("qval" in name or "qvalue" in name)
    ]
    if len(p_columns) != 1 or len(q_columns) != 1:
        raise KeyError(
            "Could not uniquely identify Psi p/q columns. "
            f"p={p_columns}, q={q_columns}, columns={list(frame.columns)}"
        )
    return p_columns[0], q_columns[0]


def load_block_matrix(path):
    raw = pd.read_csv(path)
    gene_columns = [column for column in raw if str(column).startswith("gene_")]
    if len(gene_columns) > 10:
        block_column = next(column for column in raw if column not in gene_columns)
        matrix = raw.set_index(block_column)[gene_columns].apply(
            pd.to_numeric, errors="coerce"
        ).T
        matrix.index.name = "gene"
        return matrix
    raw = standardize_gene_column(raw)
    return raw.set_index("gene").apply(pd.to_numeric, errors="coerce")


def load_ember_scores(ember_dir, psi_cutoff, alpha):
    metrics_path = ember_dir / "pvals_entropy_metrics_strain_celltype.csv"
    blocks_path = (
        ember_dir
        / "Psi_block_df"
        / "mean_Psi_block_df_strain_celltype.csv"
    )
    metrics = standardize_gene_column(pd.read_csv(metrics_path))
    p_column, q_column = find_psi_significance_columns(metrics)
    metrics = metrics.set_index("gene")
    blocks = load_block_matrix(blocks_path)
    frame = metrics.join(blocks[TARGET_BLOCKS], how="inner")

    frame["eligible"] = (
        (pd.to_numeric(frame["Psi"], errors="coerce") >= psi_cutoff)
        & (pd.to_numeric(frame[p_column], errors="coerce") < alpha)
        & (pd.to_numeric(frame[q_column], errors="coerce") < alpha)
    )
    frame["score"] = frame[TARGET_BLOCKS].min(axis=1)
    return frame


def calculate_pem_scores(adata):
    raw = adata.layers["raw_counts"]
    labels = pd.Categorical(adata.obs["strain_celltype"].astype(str))
    membership = sparse.csr_matrix(
        (
            np.ones(adata.n_obs),
            (np.arange(adata.n_obs), labels.codes),
        ),
        shape=(adata.n_obs, len(labels.categories)),
    )
    sums = membership.T @ raw
    sums = sums.toarray() if sparse.issparse(sums) else np.asarray(sums)
    observed = sums.T
    gene_totals = observed.sum(axis=1, keepdims=True)
    category_totals = observed.sum(axis=0, keepdims=True)
    expected = gene_totals @ category_totals / observed.sum()
    pem = np.log2((observed + 1e-12) / (expected + 1e-12))
    pem = pd.DataFrame(
        pem,
        index=adata.var_names.astype(str),
        columns=labels.categories.astype(str),
    )
    frame = pd.DataFrame(index=pem.index)
    frame["eligible"] = True
    frame["score"] = pem[TARGET_BLOCKS].min(axis=1)
    return frame


def aggregate_by_strain_celltype(adata):
    raw = adata.layers["raw_counts"]
    labels = pd.Categorical(adata.obs["strain_celltype"].astype(str))
    membership = sparse.csr_matrix(
        (
            np.ones(adata.n_obs),
            (np.arange(adata.n_obs), labels.codes),
        ),
        shape=(adata.n_obs, len(labels.categories)),
    )
    sums = membership.T @ raw
    sums = sums.toarray() if sparse.issparse(sums) else np.asarray(sums)
    sums = sums.T
    sizes = np.bincount(labels.codes, minlength=len(labels.categories))
    means = sums / sizes[None, :]
    return labels.categories.astype(str), sums, means


def calculate_tau_scores(adata):
    genes = adata.var_names.astype(str)
    _, _, means = aggregate_by_strain_celltype(adata)
    maximum = means.max(axis=1)
    scaled = np.divide(
        means,
        maximum[:, None],
        out=np.zeros_like(means, dtype=float),
        where=maximum[:, None] > 0,
    )
    tau = np.sum(1.0 - scaled, axis=1) / (means.shape[1] - 1)
    tau[maximum == 0] = np.nan
    frame = pd.DataFrame(index=genes)
    frame["eligible"] = True
    frame["score"] = tau
    return frame


def calculate_gini_scores(adata):
    genes = adata.var_names.astype(str)
    _, _, means = aggregate_by_strain_celltype(adata)
    values = np.sort(np.maximum(means, 0), axis=1)
    n_categories = values.shape[1]
    totals = values.sum(axis=1)
    weighted = (values * np.arange(1, n_categories + 1)[None, :]).sum(axis=1)
    gini = np.divide(
        2.0 * weighted,
        n_categories * totals,
        out=np.full(values.shape[0], np.nan),
        where=totals > 0,
    )
    gini -= (n_categories + 1.0) / n_categories
    frame = pd.DataFrame(index=genes)
    frame["eligible"] = True
    frame["score"] = gini
    return frame


def load_deseq2_scores(deseq2_dir):
    path = deseq2_dir / "Category_5_StrainA_CT1_StrainB_CT2_all_genes.csv"
    frame = standardize_gene_column(pd.read_csv(path)).set_index("gene")
    score_column = "stat" if "stat" in frame.columns else "ranking_value"
    if "eligible" in frame.columns:
        frame["eligible"] = (
            frame["eligible"].astype(str).str.lower().isin(["true", "1"])
        )
    else:
        frame["eligible"] = True
    frame["score"] = pd.to_numeric(frame[score_column], errors="coerce")
    return frame


def load_variance_partition_scores(variance_dir):
    path = variance_dir / "variance_partition_all_genes.csv"
    frame = standardize_gene_column(pd.read_csv(path))
    frame = frame.set_index("gene")
    frame["eligible"] = True
    frame["score"] = pd.to_numeric(frame["strain_celltype"], errors="coerce")
    return frame


def deterministic_tiebreak(genes):
    return pd.util.hash_pandas_object(
        pd.Series(genes, dtype=str), index=False
    ).to_numpy(dtype=np.uint64)


def rank_and_summarize(frame, method):
    ranking = frame.copy()
    ranking.index = ranking.index.astype(str)
    ranking["gene"] = ranking.index
    ranking["tie_break"] = deterministic_tiebreak(ranking["gene"])
    ranking = ranking[ranking["eligible"] & np.isfinite(ranking["score"])].copy()
    ranking = ranking.sort_values(
        ["score", "tie_break"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    top = ranking.head(200).copy()

    indices = pd.to_numeric(
        top["gene"].str.replace("gene_", "", regex=False), errors="coerce"
    )
    coherent = int(indices.between(TRUE_START, TRUE_START + N_PAIRS - 1).sum())
    decoy = int(indices.between(DECOY_START, DECOY_START + N_PAIRS - 1).sum())
    other = len(top) - coherent - decoy
    n_selected = len(top)
    return ranking, top, {
        "method": method,
        "n_selected": n_selected,
        "coherent_in_top200": coherent,
        "decoy_in_top200": decoy,
        "other_in_top200": other,
        "coherent_proportion": coherent / n_selected if n_selected else np.nan,
        "decoy_wrong_proportion": decoy / n_selected if n_selected else np.nan,
        "other_wrong_proportion": other / n_selected if n_selected else np.nan,
        "wrong_gene_proportion": (
            (decoy + other) / n_selected if n_selected else np.nan
        ),
    }


def paired_preference(frame, method):
    rows = []
    wins = losses = ties = 0
    for pair_index in range(N_PAIRS):
        true_gene = f"gene_{TRUE_START + pair_index}"
        decoy_gene = f"gene_{DECOY_START + pair_index}"
        if true_gene not in frame.index or decoy_gene not in frame.index:
            rows.append({
                "method": method,
                "pair": pair_index,
                "true_gene": true_gene,
                "decoy_gene": decoy_gene,
                "true_eligible": False,
                "decoy_eligible": False,
                "true_score": np.nan,
                "decoy_score": np.nan,
                "comparison": np.nan,
            })
            continue
        true = frame.loc[true_gene]
        decoy = frame.loc[decoy_gene]

        if bool(true["eligible"]) != bool(decoy["eligible"]):
            comparison = 1 if bool(true["eligible"]) else -1
        elif not bool(true["eligible"]):
            comparison = 0
        elif np.isclose(true["score"], decoy["score"], rtol=1e-10, atol=1e-12):
            comparison = 0
        else:
            comparison = 1 if true["score"] > decoy["score"] else -1

        wins += comparison > 0
        losses += comparison < 0
        ties += comparison == 0
        rows.append({
            "method": method,
            "pair": pair_index,
            "true_gene": true_gene,
            "decoy_gene": decoy_gene,
            "true_eligible": bool(true["eligible"]),
            "decoy_eligible": bool(decoy["eligible"]),
            "true_score": true["score"],
            "decoy_score": decoy["score"],
            "comparison": comparison,
        })

    return pd.DataFrame(rows), {
        "method": method,
        "paired_wins": wins,
        "paired_ties": ties,
        "paired_losses": losses,
        "paired_coherent_preference": (wins + 0.5 * ties) / N_PAIRS,
    }


def plot_challenge(summary, output_dir):
    summary = summary.copy()
    summary["method"] = summary["method"].astype(str).replace({"EMBER": "ember"})
    methods = [method for method in METHOD_ORDER if method in set(summary["method"])]
    summary = summary.set_index("method").loc[methods]
    fig, axis = plt.subplots(figsize=(10, 5.8))

    bottom = np.zeros(len(methods))
    parts = [
        ("coherent_proportion", "Coherent switch genes", "#2563EB"),
        ("decoy_wrong_proportion", "Rare-cell decoys", "#EF4444"),
        ("other_wrong_proportion", "Other wrong genes", "#9CA3AF"),
    ]
    for column, label, color in parts:
        values = summary[column].to_numpy()
        axis.bar(methods, values, bottom=bottom, label=label, color=color)
        bottom += values

    wrong = summary["wrong_gene_proportion"].to_numpy()
    for index, value in enumerate(wrong):
        axis.text(
            index,
            1.03,
            f"wrong {value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Proportion of switch-pattern top 200")
    axis.set_title("Wrong-gene proportion in each method's top-200 list")
    axis.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "singlecell_vs_pseudobulk.png", dpi=300)
    fig.savefig(output_dir / "singlecell_vs_pseudobulk.pdf")
    fig.savefig(output_dir / "wrong_gene_proportion_stacked.png", dpi=300)
    fig.savefig(output_dir / "wrong_gene_proportion_stacked.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    h5ad_path = resolve_input_h5ad(args.h5ad)
    output_dir = resolve_output_dir(args.output_dir, h5ad_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Using input h5ad: {h5ad_path}")
    print(f"Saving outputs to: {output_dir}")

    adata = ad.read_h5ad(h5ad_path)
    challenge = create_matched_decoys(
        adata,
        args.active_cell_fraction,
        rng,
    )
    challenge_path = output_dir / "singlecell_pseudobulk_challenge.h5ad"
    challenge.write_h5ad(challenge_path, compression="gzip")

    pairs = pd.DataFrame({
        "pair": np.arange(N_PAIRS),
        "coherent_gene": [f"gene_{i}" for i in range(800, 1000)],
        "decoy_gene": [f"gene_{i}" for i in range(1000, 1200)],
        "target_blocks": ";".join(TARGET_BLOCKS),
        "active_cell_fraction": args.active_cell_fraction,
    })
    pairs.to_csv(output_dir / "matched_gene_pairs.csv", index=False)

    ember_dir = output_dir / "ember"
    ember_dir.mkdir(exist_ok=True)
    if not args.skip_ember:
        run_ember(challenge_path, ember_dir, args)

    r_input_dir = output_dir / "r_input"
    r_input_dir.mkdir(exist_ok=True)
    r_h5ad_path = make_r_compatible_h5ad(challenge_path, r_input_dir)

    deseq2_dir = output_dir / "deseq2"
    if not args.skip_deseq2:
        run_rscript(
            SCRIPT_DIR / "deseq_eval.R",
            r_h5ad_path,
            deseq2_dir,
            args,
        )

    variance_dir = output_dir / "variance_partition"
    if not args.skip_variance_partition:
        run_rscript(
            SCRIPT_DIR / "variance_partition_eval.R",
            r_h5ad_path,
            variance_dir,
            args,
            extra_args=[args.variance_partition_cores],
        )

    method_frames = {
        "ember": load_ember_scores(
            ember_dir, args.psi_cutoff, args.alpha
        ),
        "DESeq2": load_deseq2_scores(deseq2_dir),
        "variancePartition": load_variance_partition_scores(variance_dir),
        "Tau": calculate_tau_scores(challenge),
        "PEM": calculate_pem_scores(challenge),
        "Gini": calculate_gini_scores(challenge),
    }

    summaries = []
    paired_tables = []
    for method, frame in method_frames.items():
        ranking, top, top_summary = rank_and_summarize(frame, method)
        paired, pair_summary = paired_preference(frame, method)
        summaries.append({**top_summary, **pair_summary})
        paired_tables.append(paired)
        ranking.to_csv(
            output_dir / f"{method.lower()}_challenge_ranking.csv",
            index=False,
        )
        top.to_csv(
            output_dir / f"{method.lower()}_challenge_top200.csv",
            index=False,
        )

    summary = pd.DataFrame(summaries)
    summary["method"] = pd.Categorical(
        summary["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("method").reset_index(drop=True)
    summary.to_csv(output_dir / "singlecell_vs_pseudobulk_results.csv", index=False)
    pd.concat(paired_tables, ignore_index=True).to_csv(
        output_dir / "paired_gene_results.csv", index=False
    )
    plot_challenge(summary, output_dir)

    print(summary.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
