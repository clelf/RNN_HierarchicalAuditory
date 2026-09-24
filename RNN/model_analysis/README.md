# model_analysis

Analysis of trained RNN models, on generated test data and on the recorded
experimental trial sequences.

Core modules providing function and classes to be used in specific run_* modules.


### Core

- `analysis_config.py`: paths to models and data, module/column naming conventions, figure defaults
- `analysis_core.py`: model loading (`ModelInfo`, `load_model`), generation of artificial test data, experimental sequences data processing, running models' forward pass, likelihood computation, module-activity statistics
- `metrics.py`: MSE / log-likelihood / calibration / ctx-dpos-rule scores, Kalman-filter benchmark, evaluation drivers
- `plots.py`: figures generation functions
- `exp_sequence_analysis.py`: run a model over the experimental sequences and analyzes ground-truth detection likelihoods
- `dpos_ctx_detection_alignment_cases.py`: ctx/dpos detection case analysis


### Analysis / evaluation

- `run_exp_trials_pipeline.py`: applies model on experimental sequences and produces analysis of both activation (general, and at deviant positions) and probabilities/likelihoods-


- `run_exp_trials_summaries.py`: read CSVs of likelihoods and dpos probability, the dpos probability at the true deviant, and the per-trial activity-vs-likelihood join,  averages per sequence
- `run_module_correlations.py`: compute module-pair correlation scores, produce corr. scores distribution and association figures
- `run_sample_figures.py`: per-sample prediction figures (artificial test dataset, and experimental from the probability CSVs) and hidden-activity trajectories on synthetic data

- `run_model_evaluation.py`: compare models on generated test data; log-likelihood evaluation for every module (optional KS calibration)



- `run_dpos_ctx_detection_alignment_cases_assessment.py`: ctx/dpos detection alignment cases per trial. Read from the probability CSVs, summarize into one row per model




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

Paths used:

- `TRAINING_RESULTS_DIR` — `RNN/training_results/N_ctx_2/HierarchicalGM`
- `TRIALS_PATH` — `Workspace/Jasmin/trialsequences2clem`
- `EXP_SEQ_OUTPUT_ROOT` — `RNN/exp_seq_act_output`
- `EVALUATION_RESULTS_DIR` — `RNN/evaluation_results`
