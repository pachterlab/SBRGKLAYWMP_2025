#!/usr/bin/env python3
"""Generate the clean five-pattern, 3,000-gene benchmark with 8x effects."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


N_GENES = 3000
STRAINS = ["StrainA", "StrainB"]
CELL_TYPES = ["CT1", "CT2", "CT3", "CT4"]
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "results" / "simulation_data.h5ad"),
    )
    parser.add_argument("--cells-per-strain-celltype", type=int, default=500)
    parser.add_argument("--mice-per-strain", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fold-change", type=float, default=8.0)
    parser.add_argument("--dispersion", type=float, default=8.0)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--min-detected-cells", type=int, default=100)
    return parser.parse_args()


def make_obs(cells_per_group, mice_per_strain, rng):
    if cells_per_group % mice_per_strain != 0:
        raise ValueError(
            "--cells-per-strain-celltype must be divisible by --mice-per-strain"
        )

    cells_per_mouse_celltype = cells_per_group // mice_per_strain
    rows = []
    for strain in STRAINS:
        for mouse_number in range(1, mice_per_strain + 1):
            mouse_id = f"{strain}_Mouse{mouse_number}"
            for cell_type in CELL_TYPES:
                for _ in range(cells_per_mouse_celltype):
                    rows.append((mouse_id, strain, cell_type))

    obs = pd.DataFrame(rows, columns=["mouse_id", "strain", "cell_type"])
    obs = obs.iloc[rng.permutation(len(obs))].reset_index(drop=True)
    obs["strain_celltype"] = obs["strain"] + ":" + obs["cell_type"]
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    return obs


def gamma_poisson(mu, dispersion, rng):
    """Negative-binomial counts with Var(X) = mu + mu^2 / dispersion."""
    gamma_scale = mu / dispersion
    latent_rate = rng.gamma(shape=dispersion, scale=gamma_scale)
    return rng.poisson(latent_rate).astype(np.int32)


def add_effect(mu, gene_slice, mask, fold_change):
    rows = np.flatnonzero(np.asarray(mask))
    columns = np.arange(gene_slice.start, gene_slice.stop)
    mu[np.ix_(rows, columns)] *= fold_change


def make_truth_table():
    truth = pd.DataFrame({
        "gene": [f"gene_{i}" for i in range(N_GENES)],
        "gene_index": np.arange(N_GENES),
        "truth_category": "Null/background",
        "target_blocks": "",
    })

    definitions = [
        (0, 200, "1 CT", "CT1"),
        (200, 400, "2 CT", "CT1;CT2"),
        (400, 600, "Housekeeping", "all"),
        (600, 800, "1 CT (1 strain)", "StrainA:CT1"),
        (800, 1000, "Strain switch", "StrainA:CT1;StrainB:CT2"),
    ]
    for start, stop, category, blocks in definitions:
        truth.loc[start:stop - 1, "truth_category"] = category
        truth.loc[start:stop - 1, "target_blocks"] = blocks
    return truth


def main():
    args = parse_args()
    if args.fold_change <= 1:
        raise ValueError("--fold-change must be greater than 1.")

    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    obs = make_obs(
        args.cells_per_strain_celltype,
        args.mice_per_strain,
        rng,
    )
    n_cells = len(obs)

    # All genes have nonzero baseline expression, preventing benchmark genes
    # from disappearing during minimum-detection filtering.
    gene_baseline = rng.lognormal(
        mean=np.log(0.20),
        sigma=0.45,
        size=N_GENES,
    )
    gene_baseline = np.clip(gene_baseline, 0.10, 0.80)

    # The four specificity categories begin with comparable nonspecific
    # expression; only their target blocks differ.
    gene_baseline[:1000] = rng.lognormal(
        mean=np.log(0.25),
        sigma=0.20,
        size=1000,
    )

    # Housekeeping genes are highly expressed in all cells and receive no
    # partition-specific effect.
    gene_baseline[400:600] = rng.lognormal(
        mean=np.log(5.0),
        sigma=0.15,
        size=200,
    )

    cell_depth = rng.lognormal(mean=0.0, sigma=0.30, size=n_cells)
    mouse_effect_map = {
        mouse: rng.lognormal(mean=0.0, sigma=0.12)
        for mouse in obs["mouse_id"].unique()
    }
    mouse_effect = obs["mouse_id"].map(mouse_effect_map).to_numpy(float)

    mu = (
        cell_depth[:, None]
        * mouse_effect[:, None]
        * gene_baseline[None, :]
    ).astype(np.float32)

    # gene_0-gene_199: CT1 in both strains.
    add_effect(
        mu,
        slice(0, 200),
        obs["cell_type"].eq("CT1"),
        args.fold_change,
    )

    # gene_200-gene_399: CT1 and CT2 in both strains.
    add_effect(
        mu,
        slice(200, 400),
        obs["cell_type"].isin(["CT1", "CT2"]),
        args.fold_change,
    )

    # gene_400-gene_599: housekeeping; high baseline in every cell.

    # gene_600-gene_799: CT1 only in StrainA.
    add_effect(
        mu,
        slice(600, 800),
        obs["strain"].eq("StrainA") & obs["cell_type"].eq("CT1"),
        args.fold_change,
    )

    # gene_800-gene_999: StrainA:CT1 and StrainB:CT2.
    switch_mask = (
        (obs["strain"].eq("StrainA") & obs["cell_type"].eq("CT1"))
        | (obs["strain"].eq("StrainB") & obs["cell_type"].eq("CT2"))
    )
    add_effect(
        mu,
        slice(800, 1000),
        switch_mask,
        args.fold_change,
    )

    counts = gamma_poisson(mu, args.dispersion, rng)
    counts = sparse.csr_matrix(counts, dtype=np.int32)

    detected_cells = np.asarray((counts > 0).sum(axis=0)).ravel()
    if detected_cells.min() < args.min_detected_cells:
        failing = np.flatnonzero(detected_cells < args.min_detected_cells)
        raise RuntimeError(
            f"{len(failing)} genes were detected in fewer than "
            f"{args.min_detected_cells} cells. First failures: {failing[:10]}. "
            "Increase cells per group or the baseline-expression floor."
        )

    truth = make_truth_table()
    adata = ad.AnnData(
        X=counts.copy(),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_GENES)]),
    )
    adata.var["truth_category"] = truth["truth_category"].to_numpy()
    adata.var["target_blocks"] = truth["target_blocks"].to_numpy()
    adata.var["detected_cells"] = detected_cells
    adata.var["simulation_baseline_mean"] = gene_baseline
    adata.layers["raw_counts"] = counts.copy()

    # EMBER input: depth-normalized, log1p-transformed single-cell expression.
    adata.X = adata.X.astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)

    adata.uns.pop("log1p", None)
    adata.uns["simulation"] = {
        "seed": args.seed,
        "fold_change": args.fold_change,
        "dispersion": args.dispersion,
        "cells_per_strain_celltype": args.cells_per_strain_celltype,
        "mice_per_strain": args.mice_per_strain,
    }

    adata.write_h5ad(output, compression="gzip")
    truth_path = output.with_name(output.stem + "_truth.csv")
    truth.to_csv(truth_path, index=False)

    summary = (
        truth
        .assign(detected_cells=detected_cells)
        .groupby("truth_category", sort=False)["detected_cells"]
        .agg(["count", "min", "median", "max"])
    )
    print(summary)
    print(f"\nFold change: {args.fold_change}x")
    print(f"Saved: {output}")
    print(f"Saved: {truth_path}")


if __name__ == "__main__":
    main()
