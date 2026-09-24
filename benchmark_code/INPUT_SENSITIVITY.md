# Input-sensitivity benchmark

Simulation-only EMBER recovery benchmark, using the existing `generate_sim.py`
and recovery rules from `ember_only_eval.py`. The three comparisons are:

- Annotation resolution: four cell types versus eight strain/cell-type groups.
- Unequal cell numbers: balanced, mild and strong imbalance with identical
  within-mouse/group empirical expression distributions and 4,000 total cells.
- Normalization: raw counts, library normalization to 10,000, and log1p normalization.

## Included results

[Summary](results/input_sensitivity/summary.txt) contains recovery counts for
all five gene patterns, eligibility counts, and unequal-size recovery with and
without the p/q-value cutoff. Each pattern has 200 true genes among 3,000 genes.

Each comparison directory contains a `recovery.csv`, selected `top200` tables,
expression metrics, cell counts, and one figure in PNG/PDF. Full gene-ranking
tables are retained locally but ignored by Git; a full run regenerates them.
`_ember/` retains native scores, both partitions' full Psi/Zeta p-values and
BH-adjusted q-values, and block-share means/standard deviations. Binary inputs
and completion flags are omitted from the included results; a full run generates
them. No real data is required.

The without-p-value comparison removes only the significance gate: Psi >= 0.5
and the housekeeping expression gate remain. `unfiltered_rank` and
`true_genes_top200_without_filter` additionally describe rankings without any
eligibility gates. Housekeeping recovery never requires significance in the
original benchmark rules. The unequal-size figure shows the original filtered
recovery; both recovery variants are included in the summary and CSV.

## Run

From the repository root, activate an environment with EMBER, anndata, NumPy,
pandas, Scanpy and matplotlib (see the repository's `environment.yaml`).

```bash
python benchmark_code/run_input_sensitivity_benchmark.py
```

This writes a new run to `benchmark_code/results/input_sensitivity_run/`, which
is ignored by Git. It leaves the included results unchanged. The script uses
`../ember_package/src` if that sibling checkout exists, otherwise installed
`ember_py`. To select the canonical source checkout explicitly:

```bash
python benchmark_code/run_input_sensitivity_benchmark.py \
  --ember-repo /path/to/ember_package \
  --results-dir benchmark_code/results/input_sensitivity_rerun
```

The benchmark uses EMBER's permutation worker API; use the same source version
as the included results for exact reproduction. The recorded source hashes and
Python package versions are in `recovery_run_config.json`. Results were generated
with Ember checkout commit `97f8a86`; the hashes identify the precise source files.
Source/settings changes require a fresh results directory; interrupted runs with
unchanged source/settings resume from completion markers.

To refresh the included figures and summaries from saved ranking/recovery tables,
without repeating simulation or permutation tests (requires the local full
gene-ranking tables; on a fresh clone, perform a full run first and use its
results directory):

```bash
python benchmark_code/run_input_sensitivity_benchmark.py \
  --results-dir benchmark_code/results/input_sensitivity --report-only
```

## Design and interpretation

Seed 123, 1,000 label permutations, 16 replicate draws, four mice per strain,
two strains, and the original five simulated expression patterns. Each
permutation receives a deterministic seed. Biological replicate sampling uses
`mouse_id` and `strain`. No additional significance test compares recovery counts.

Per-strain cell counts for CT1/CT2/CT3/CT4 are 500/500/500/500 (balanced),
300/400/600/700 (mild), and 100/200/500/1200 (strong). Integer repetition preserves
within-mouse/group expression distributions. Repeated profiles are not additional
biological replicates. These results cover one simulation seed; failure to pass
a significance cutoff alone does not establish lack of specificity or its cause.

The included inference results are preserved from the previous run. Integration
only changed paths, CLI and reporting; `integration_provenance.json` records
that migration. `no_pvalue_summary_provenance.json` records the earlier addition
of descriptive recovery without rerunning permutations.
