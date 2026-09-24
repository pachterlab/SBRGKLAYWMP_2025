#!/usr/bin/env python3
"""Simulation-only EMBER input sensitivity: permutation significance and gene recovery."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib
import json
import subprocess
import sys

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir', type=Path, default=HERE/'results/input_sensitivity_run',
                    help='Output directory; use a new directory for changed code/settings.')
parser.add_argument('--ember-repo', type=Path, default=None,
                    help='EMBER source checkout (containing src/ember_py); otherwise use installed ember_py.')
parser.add_argument('--report-only', action='store_true',
                    help='Refresh figures and summaries from saved gene rankings; no simulation or permutations.')
ARGS = parser.parse_args()
EMBER_REPO = ARGS.ember_repo.resolve() if ARGS.ember_repo else PROJECT.parent/'ember_package'
if ARGS.ember_repo and not (EMBER_REPO/'src/ember_py').is_dir():
    parser.error('--ember-repo must contain src/ember_py')
sys.path[:0] = [str(EMBER_REPO / 'src'), str(PROJECT / 'benchmark_code')]
from ember_py.light_ember import light_ember
from ember_only_eval import TRUTH, build_scores, evaluate, load_partition
PVALS = importlib.import_module('ember_py.generate_pvals')
RESULTS = ARGS.results_dir.resolve()
SEED, N_PERMUTATIONS, N_DRAWS = 123, 1000, 16
PARTITIONS = ('cell_type', 'strain_celltype')
SHORT = ['1 CT', '2 CTs', 'Housekeeping', '1 CT, 1 strain', 'Strain switch']
RULES = SimpleNamespace(alpha=.05, celltype_psi_cutoff=.5, strain_celltype_psi_cutoff=.5,
                       housekeeping_expression_cutoff=1., ct1_block='CT1', ct2_block='CT2',
                       strain_a_ct1_block='StrainA:CT1', strain_b_ct2_block='StrainB:CT2')


def seeded_permutation(task):
    # Only set RNG state: call the unmodified canonical permutation calculation.
    seed = np.random.SeedSequence([SEED, task[0]]).generate_state(1)[0]
    np.random.seed(seed)
    return PVALS._run_permutation_task(*task)


def reweight(base, factors):
    repeats = base.obs.cell_type.astype(str).map(dict(zip(['CT1', 'CT2', 'CT3', 'CT4'], factors)))
    rows = np.repeat(np.arange(base.n_obs), repeats.to_numpy(int))
    obs = base.obs.iloc[rows].copy()
    obs['source_cell'] = obs.index
    obs.index = [f'cell_{i}' for i in range(len(rows))]
    data = ad.AnnData(base.X[rows].copy(), obs=obs, var=base.var.copy())
    data.layers['raw_counts'] = base.layers['raw_counts'][rows].astype(float)
    assert data.n_obs == 4000
    return data


def run_partition(job):
    folder, partition = job
    destination = folder / 'ember'
    destination.mkdir(exist_ok=True)
    # Completion markers enable interrupted runs to resume, guarded by the manifest.
    marker = destination / f'{partition}.complete'
    if marker.exists():
        return
    light_ember(str(folder / 'input.h5ad'), partition, str(destination),
                sample_id_col='mouse_id', category_col='strain', seed=SEED,
                num_draws=N_DRAWS, partition_pvals=False, n_cpus=2)
    previous = PVALS._worker_wrapper
    PVALS._worker_wrapper = seeded_permutation
    try:
        PVALS.generate_pvals(str(folder / 'input.h5ad'), partition, str(destination),
                             str(destination), 'mouse_id', 'strain', seed=SEED,
                             n_iterations=N_PERMUTATIONS, n_cpus=4)
    finally:
        PVALS._worker_wrapper = previous
    frame = load_partition(folder, partition)
    assert len(frame) == 3000 and frame.notna().all().all()
    marker.write_text('complete\n')


def recover(folder, data, resolution=None):
    coarse = load_partition(folder, 'cell_type')
    fine = load_partition(folder, 'strain_celltype')
    if resolution == 'coarse':
        # A strain-specific target is only identifiable at its parent cell type.
        joint = coarse.copy()
        joint['StrainA:CT1'], joint['StrainB:CT2'] = coarse.CT1, coarse.CT2
        celltype = coarse
    elif resolution == 'fine':
        # Parent targets comprise both strain children; sum their block shares.
        celltype = fine.copy()
        for ct in ('CT1', 'CT2'):
            celltype[ct] = fine[f'StrainA:{ct}'] + fine[f'StrainB:{ct}']
        joint = fine
    else:
        celltype, joint = coarse, fine
    means = pd.Series(np.asarray(data.layers['raw_counts'].mean(axis=0)).ravel(), index=data.var_names)
    scores = build_scores(celltype, joint, means, RULES)
    rows, rankings, selections = [], [], []
    for category, original in scores.items():
        frame = original.reset_index(drop=True).copy()
        partition = ('cell_type' if category in list(TRUTH)[:3] else 'strain_celltype') if resolution is None else ('cell_type' if resolution == 'coarse' else 'strain_celltype')
        metrics = pd.read_csv(folder / 'ember' / f'pvals_entropy_metrics_{partition}.csv', index_col=0)
        for name in ('Zeta', 'Zeta p-value', 'Zeta q-value'):
            frame[name] = frame.gene.map(metrics[name])
        first, last = TRUTH[category]
        frame['is_target'] = frame.gene.str.removeprefix('gene_').astype(int).between(first, last)
        low = category == 'Category 3: Housekeeping'
        frame = frame.sort_values(['ranking_value', 'gene'], ascending=[low, True]).reset_index(drop=True)
        frame['unfiltered_rank'] = np.arange(1, len(frame) + 1)
        recovered, n_eligible, ranked, selected = evaluate(frame, category)
        frame['eligible_rank'] = frame.gene.map(ranked.set_index('gene').eligible_rank)
        frame['selected'] = frame.gene.isin(selected.gene)
        frame['category'] = category
        selected['category'] = category
        significant = (frame.Psi_pvalue < .05) & (frame.Psi_qvalue < .05)
        target = frame.is_target
        rows.append(dict(category=category, recovered=recovered, out_of=200,
                         n_selected=len(selected), n_eligible=n_eligible,
                         true_genes_passing_psi_pq=int((target & significant).sum()),
                         true_genes_passing_psi_cutoff=int((target & frame.Psi.ge(.5)).sum()),
                         true_genes_eligible=int((target & frame.eligible).sum()),
                         true_genes_top200_without_filter=int(frame.head(200).is_target.sum()),
                         recall_at_200=recovered / 200))
        rankings.append(frame)
        selections.append(selected)
    return pd.DataFrame(rows), pd.concat(rankings, ignore_index=True), pd.concat(selections, ignore_index=True)


def recover_without_pvalues(rankings):
    """Remove only Psi p/q eligibility; retain the original effect/expression gates."""
    rows, ranked_frames, selected_frames = [], [], []
    for category in TRUTH:
        frame = rankings.loc[rankings.category.eq(category)].copy()
        frame = frame.drop(columns=['eligible_rank', 'selected'], errors='ignore')
        cutoff = RULES.celltype_psi_cutoff if category in list(TRUTH)[:3] else RULES.strain_celltype_psi_cutoff
        frame['eligible'] = frame.Psi.ge(cutoff)
        if category == 'Category 3: Housekeeping':
            frame['eligible'] &= frame.mean_expression.ge(RULES.housekeeping_expression_cutoff)
        recovered, n_eligible, ranked, selected = evaluate(frame, category)
        ranked['selected'] = ranked.gene.isin(selected.gene)
        selected['selected'] = True
        rows.append(dict(category=category, recovered_without_pvalue=recovered,
                         n_eligible_without_pvalue=n_eligible,
                         n_selected_without_pvalue=len(selected)))
        ranked_frames.append(ranked)
        selected_frames.append(selected)
    return pd.DataFrame(rows), pd.concat(ranked_frames, ignore_index=True), pd.concat(selected_frames, ignore_index=True)


def no_pvalue_summary(result, conditions):
    counts = result.pivot(index='category', columns='condition', values='recovered_without_pvalue').reindex(TRUTH)[list(conditions)]
    return ('\nRecovered out of 200 without the p/q-value cutoff (same ranking, Psi and housekeeping expression gates):\n'
            + counts.to_string() + '\nOnly significance filtering is removed; this is descriptive ranking recovery.\n')


def figure(results, output, name, titles):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    conditions = list(titles)
    width = .8 / len(conditions)
    for i, condition in enumerate(conditions):
        values = results.loc[results.condition.eq(condition)].set_index('category').reindex(TRUTH).recovered
        bars = ax.bar(np.arange(5) + (i - (len(conditions)-1)/2)*width, values, width,
                      label=titles[condition], color=['#4477AA', '#EE6677', '#228833'][i])
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.axhline(200 * 200 / 3000, color='grey', linestyle='--', linewidth=.8,
               label='Random top 200 (13.3/200)')
    ax.set(xticks=np.arange(5), xticklabels=SHORT, ylim=(0, 220),
           ylabel='True genes recovered in eligible top 200', title=name.replace('_', ' ').capitalize())
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(.5, 1.17), ncol=len(conditions)+1)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{suffix}', dpi=300)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not ARGS.report_only:
        base_path = RESULTS / 'base_simulation.h5ad'
        generator = PROJECT / 'benchmark_code/generate_sim.py'
        if not base_path.exists():
            subprocess.run([sys.executable, str(generator), '--output', str(base_path), '--seed', str(SEED),
                            '--cells-per-strain-celltype', '100', '--min-detected-cells', '0'], check=True)
        base = ad.read_h5ad(base_path)
        assert base.uns['simulation']['seed'] == SEED and base.shape == (800, 3000)
        base.X = base.layers['raw_counts'].astype(float)
        sc.pp.normalize_total(base, target_sum=10000)
        sc.pp.log1p(base)
        balanced = reweight(base, [5, 5, 5, 5])
        raw = balanced.copy()
        raw.X = raw.layers['raw_counts'].copy()
        normalized = raw.copy()
        sc.pp.normalize_total(normalized, target_sum=10000)
        datasets = dict(balanced=balanced, mild=reweight(base, [3, 4, 6, 7]),
                        strong=reweight(base, [1, 2, 5, 12]), raw=raw, normalized=normalized)
        config = dict(seed=SEED, permutations=N_PERMUTATIONS, draws=N_DRAWS,
                      permutation_rng='SeedSequence([seed, permutation_index]); numpy.random.seed per task',
                      parameters=vars(RULES), versions={m.__name__:m.__version__ for m in (ad,np,pd,sc)},
                      source_hashes={('ember_py/'+p.name if p.parent == Path(PVALS.__file__).parent else p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                     [Path(__file__), generator, PROJECT/'benchmark_code/ember_only_eval.py',
                                      *sorted(Path(PVALS.__file__).parent.glob('*.py'))]},
                      base_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest())
        manifest = RESULTS / 'recovery_run_config.json'
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            raise RuntimeError('Settings or source changed; use --results-dir with a fresh directory to rerun.')
        manifest.write_text(json.dumps(config, indent=2)+'\n')
        jobs = []
        for name, data in datasets.items():
            folder = RESULTS / '_ember' / name
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder/'input.h5ad').exists():
                data.write_h5ad(folder/'input.h5ad')
            jobs.extend((folder, partition) for partition in PARTITIONS)
        with ProcessPoolExecutor(max_workers=2) as executor:
            list(executor.map(run_partition, jobs))
    common = (
        'Simulation ONLY: seed 123, 3000 genes, 4000 cells; unchanged existing generate_sim.py.\n'
        'Canonical repository ember_py.light_ember.light_ember(partition_pvals=False, seed=123, num_draws=16), '
        'then ember_py.generate_pvals.generate_pvals(seed=123, n_iterations=1000). Both use mouse_id/strain.\n'
        'Only the RNG-setting wrapper is added: each permutation is seeded by its index; canonical calculations are unchanged. '
        'Source hashes and parameters: ../recovery_run_config.json. Native statistical outputs: ../_ember/. Binary simulation inputs are regenerated by a full run.\n'
        'Observed scores are replicate-sampled means, matched to the canonical permutation null; all-cell scores are replaced.\n'
        'All Psi and Zeta p/q-values are saved. BH correction is EMBER\'s combined Psi/Zeta correction within each partition. '
        'Block shares are saved without block-specific p-values, as in the original benchmark.\n'
        'Recovery reuses ember_only_eval.build_scores/evaluate: Psi>=0.5 and Psi p<0.05, q<0.05; '
        'rank by target block share (minimum for two targets). Housekeeping: raw mean>=1 and Psi>=0.5, '
        'rank ascending Zeta, with no significance requirement. Zeta p/q are reported, not extra gates.\n'
        'Readout: recovered true genes out of 200, at most 200 eligible selections; deterministic gene-name tie breaks. '
        'All-gene rankings include unfiltered rank, eligible rank, p/q-values, eligibility and selection.\n'
        'Recovery CSV additionally counts true genes passing significance and eligibility, and true genes in ungated top 200. '
        'Failure of a significance filter is not proof of no specificity or a calibrated measurement of statistical power.\n'
        'Figure: grouped bars of recovered genes, exactly one PNG/PDF per benchmark. Dashed line assumes 200 random selections.\n'
        'Replicated profiles preserve empirical expression but do not add independent biological evidence. One simulation seed only.\n'
    )
    experiments = [
        ('annotation_resolution', [('coarse','balanced','coarse'),('fine','balanced','fine')],
         {'coarse':'Coarse (4 groups)', 'fine':'Fine (8 groups)'},
         'Same cells/matrix; fine=strain_celltype, coarse=cell_type. Same five truth sets in both. '
         'Fine parent targets sum both strain-child block shares; coarse strain-specific targets map to their parent cell types. '
         'Coarse labels cannot resolve strain specificity. Fine labels are simulated strain subdivisions, not real subtypes.\n'),
        ('unequal_cell_numbers', [(n,n,None) for n in ('balanced','mild','strong')],
         {'balanced':'Balanced','mild':'Mild','strong':'Strong'},
         'Per-strain CT1/2/3/4 counts: 500/500/500/500, 300/400/600/700, 100/200/500/1200. '
         'Total fixed at 4000. Within-mouse/block expression distributions identical via integer repetition. '
         'Categories 1-3 use cell_type; 4-5 use strain_celltype, exactly as in the original recovery benchmark.\n'),
        ('normalization', [('raw','raw',None),('normalized','normalized',None),('log_normalized','balanced',None)],
         {'raw':'Raw counts','normalized':'Library normalized','log_normalized':'Log1p normalized'},
         'Same raw counts, cells, genes and partitions. Normalize to 10000 per cell, then log1p. '
         'Core scoring does neither preprocessing step internally. Categories use the same partitions as the original benchmark.\n'),
    ]
    summary = []
    for name, conditions, titles, note in experiments:
        output = RESULTS / name
        output.mkdir(exist_ok=True)
        records = []
        for condition, dataset, resolution in conditions:
            if ARGS.report_only:
                recovery = pd.read_csv(output/'recovery.csv')
                recovery = recovery.loc[recovery.condition.eq(condition)].drop(columns=[
                    'condition', 'recovered_without_pvalue', 'n_eligible_without_pvalue',
                    'n_selected_without_pvalue'], errors='ignore')
                ranks = pd.read_csv(output/f'{condition}_gene_rankings.csv')
                top = pd.read_csv(output/f'{condition}_top200.csv')
            else:
                folder, data = RESULTS/'_ember'/dataset, datasets[dataset]
                recovery, ranks, top = recover(folder, data, resolution)
            if name == 'unequal_cell_numbers':
                no_pvalue, ranked_no_pvalue, top_no_pvalue = recover_without_pvalues(ranks)
                recovery = recovery.merge(no_pvalue, on='category', validate='one_to_one')
                ranked_no_pvalue.to_csv(output/f'{condition}_without_pvalue_gene_rankings.csv', index=False)
                top_no_pvalue.to_csv(output/f'{condition}_without_pvalue_top200.csv', index=False)
            recovery['condition'] = condition
            records.append(recovery)
            ranks.to_csv(output/f'{condition}_gene_rankings.csv', index=False)
            top.to_csv(output/f'{condition}_top200.csv', index=False)
            if not ARGS.report_only:
                partition = 'strain_celltype' if resolution == 'fine' else 'cell_type'
                metrics = pd.read_csv(folder/'ember'/f'pvals_entropy_metrics_{partition}.csv')
                metrics.to_csv(output/f'{condition}_entropy_metrics.csv', index=False)
                blocks = pd.read_csv(folder/'ember/Psi_block_df'/f'mean_Psi_block_df_{partition}.csv', index_col=0)
                blocks.to_csv(output/f'{condition}_Psi_block.csv', index_label='gene')
                data.obs.groupby(partition, observed=True).size().rename('n_cells').to_csv(output/f'{condition}_cell_counts.csv')
        result = pd.concat(records, ignore_index=True)
        result.to_csv(output/'recovery.csv', index=False)
        figure(result, output, name, titles)
        (output/'README.txt').write_text(common+note+(no_pvalue_summary(result, titles) if name == 'unequal_cell_numbers' else ''))
        counts = result.pivot(index='category',columns='condition',values='recovered').reindex(TRUTH)[list(titles)]
        eligible = result.pivot(index='category',columns='condition',values='true_genes_eligible').reindex(TRUTH)[list(titles)]
        summary.append(name+'\nTested: '+note+'Recovered out of 200:\n'+counts.to_string()+
                       '\nTrue genes eligible after the existing gates:\n'+eligible.to_string()+
                       (no_pvalue_summary(result, titles) if name == 'unequal_cell_numbers' else '')+
                       '\nRecovery stability: '+('identical counts across conditions.' if counts.nunique(axis=1).eq(1).all()
                                                  else 'recovery changes across conditions; interpret with the eligibility counts above.')+
                       '\nCaveat: one seed, repeated profiles; non-significance alone cannot distinguish low power from other null-model effects.\n')
        print(name+'\n'+counts.to_string(), flush=True)
    (RESULTS/'summary.txt').write_text('\n'.join(summary))


if __name__ == '__main__':
    main()
