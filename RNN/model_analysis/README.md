# model_analysis

Analysis of trained RNN models, on generated test data and on the recorded
experimental trial sequences.

## Layout

Five library modules, one taxonomy module, and six entry points. Nothing in the
library modules writes output on import; everything runnable is a `run_*.py`.

### Library (import these, don't run them)

| module | holds |
| --- | --- |
| `analysis_config.py` | the three filesystem roots, module/column naming conventions, the named model lists, figure defaults |
| `analysis_core.py` | model loading (`ModelInfo`, `load_model`), synthetic test data, sequence I/O, the forward pass, likelihood helpers, module-activity statistics |
| `metrics.py` | MSE / log-likelihood / calibration / ctx-dpos-rule scores, the Kalman-filter benchmark, and the evaluation drivers |
| `plots.py` | every figure, plus `save_figure` — the one place that writes a PNG |
| `exp_sequence_analysis.py` | running a model over the experimental sequences, and aggregating the CSVs that produces |
| `alignment_cases.py` | the ctx/dpos detection case taxonomy and its per-trial / per-model tables |

Dependencies run one way: `analysis_config` → `analysis_core` → {`metrics`,
`exp_sequence_analysis`, `alignment_cases`} → `plots` → the `run_*` scripts.

### Entry points (run these)

| script | what it does |
| --- | --- |
| `run_exp_trials_pipeline.py` | model → experimental sequences: the activation / deviant-activation / probability CSVs and the three activity figure sets. **The only script with a command line** (`--help`); it is the heavy batch driver. |
| `run_exp_trials_summaries.py` | reads those CSVs back: likelihood averages per sequence, the dpos probability at the true deviant, and the per-trial activity-vs-likelihood join |
| `run_module_correlations.py` | module-pair correlation scores, then the distribution and association figures |
| `run_alignment_assessment.py` | ctx/dpos detection cases per trial, collapsed into one row per model |
| `run_model_evaluation.py` | compares models on generated test data; violin and calibration figures |
| `run_sample_figures.py` | per-sample prediction figures (synthetic and experimental) and hidden-activity trajectories |

Every entry point except the pipeline is configured by editing the
`SETTINGS` block at the top of its `__main__`.

## Order of operations

`run_exp_trials_pipeline.py` produces the CSVs the other experimental-sequence
scripts consume, so it runs first:

```
run_exp_trials_pipeline.py          # writes activations/ and probabilities*/
  ├── run_exp_trials_summaries.py   # needs probabilities_deviant/ (+ activations_deviant/)
  └── run_module_correlations.py    # needs activations/
run_alignment_assessment.py         # independent: runs the model itself
run_model_evaluation.py             # independent: generated test data only
run_sample_figures.py               # independent
```

## Figure provenance

Every figure written through `plots.save_figure` gets a line in a
`figure_metadoc.txt` next to it, recording which script produced it. Entry points
register themselves once:

```python
import plots
plots.set_script_path(__file__)
```

Figures that `pipeline_core_v2.plot_samples` writes itself (the per-sample
figures) are the exception — it names and saves its own files, so they carry no
metadoc entry.

## Paths

`analysis_config.py` derives every root from its own location, so there are no
absolute paths to edit:

- `TRAINING_RESULTS_DIR` — `RNN/training_results/N_ctx_2/HierarchicalGM`
- `TRIALS_PATH` — `Workspace/Jasmin/trialsequences2clem`
- `EXP_SEQ_OUTPUT_ROOT` — `RNN/exp_seq_act_output`
- `EVALUATION_RESULTS_DIR` — `RNN/evaluation_results`
