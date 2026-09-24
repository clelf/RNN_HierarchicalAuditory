"""Shared configuration for the model_analysis scripts.

Single source of truth for the three filesystem roots, the module/column naming
conventions used by every CSV written here, and the named lists of models an
analysis is run over. Before this module the same three absolute paths were
repeated in twelve scripts and the model lists in four, which is how they drifted
apart (see MODEL_LIST_NOTES below).

Paths are derived from this file's location rather than hard-coded, so moving or
cloning the workspace does not require editing them.
"""

from pathlib import Path


# =============================================================================
# Filesystem roots
# =============================================================================

# .../Workspace/RNN_paradigm/RNN/model_analysis/analysis_config.py
#     parents[1] = .../RNN          parents[3] = .../Workspace
_HERE = Path(__file__).resolve()
RNN_DIR = _HERE.parents[1]
WORKSPACE_ROOT = _HERE.parents[3]

# Trained model checkpoints: one sub-folder per model, each holding a .pth and a
# config.json.
TRAINING_RESULTS_DIR = RNN_DIR / 'training_results' / 'N_ctx_2' / 'HierarchicalGM'

# Recorded experimental trial sequences (one CSV per sequence).
TRIALS_PATH = WORKSPACE_ROOT / 'Jasmin' / 'trialsequences2clem'

# Everything the experimental-sequence analyses write, under <root>/<model name>/.
EXP_SEQ_OUTPUT_ROOT = RNN_DIR / 'exp_seq_act_output'

# Figures from the synthetic test-set evaluation.
EVALUATION_RESULTS_DIR = RNN_DIR / 'evaluation_results'


# =============================================================================
# Sequence and module conventions
# =============================================================================

# Timesteps per trial. The within-trial position of timestep t is t % PERIOD and
# its trial index is t // PERIOD; the deviant of trial t sits at t * PERIOD + dpos.
PERIOD = 8

# Minimum deviant position labelled in the experimental sequence files (generated
# with rules_dpos_set=[[2,3,4],[4,5,6]], so dpos in {2..6}). A property of the
# files, not of any model -- see analysis_core.dpos_conventions for how a model's
# own dpos_min is reconciled with this one.
EXPERIMENTAL_DPOS_MIN = 2

# Order matters: the two cue columns of a sequence file are read back in this
# order (see analysis_core.load_trial_sequence).
CUE_LABELS = ['cue_2', 'cue_1']

# Module names as stored in every activation / probability CSV written here.
MODULE_NAMES = ['obs', 'ctx', 'dpos', 'rule']

MODULE_TITLES = {
    'obs':  'Observation module',
    'ctx':  'Type module',
    'dpos': 'Deviant position module',
    'rule': 'Rule module',
}

# Cue re-encoding seed. Must be identical across stages so the activity norms and
# the likelihoods refer to the same cue encoding -- they are merged per trial
# downstream by the activity/likelihood join.
CUE_SEED = 11

# Sequences per batched forward pass (bounds hidden-state memory).
CHUNK_SIZE = 128


# =============================================================================
# CSV column groups
# =============================================================================

# Per-sequence CSVs written by run_exp_trials_pipeline.py, one per experimental
# sequence and kind, at <EXP_SEQ_OUTPUT_ROOT>/<model>/<kind>/<sequence><suffix>.
SEQUENCE_CSV_SUFFIXES = {
    'activations':           '_activations.csv',
    'activations_deviant':   '_deviant_trial.csv',
    'probabilities':         '_probabilities.csv',
    'probabilities_deviant': '_probabilities_deviant.csv',
}

# Likelihood columns summarised from the *_probabilities_deviant.csv files.
# cdf_lik_obs and log_lik_obs are derived on read, not stored by the extraction.
LIKELIHOOD_COLS = ['lik_obs', 'cdf_lik_obs', 'log_lik_obs', 'lik_ctx', 'lik_dpos', 'lik_rule']

# Subset actually drawn: cdf_lik_obs stays in the CSVs but is not plotted.
PLOT_LIKELIHOOD_COLS = ['lik_obs', 'log_lik_obs', 'lik_ctx', 'lik_dpos', 'lik_rule']

# (module, kind of likelihood) shown in each subplot title.
LIKELIHOOD_LABELS = {
    'lik_obs':     ('obs',  'Gaussian likelihood'),
    'cdf_lik_obs': ('obs',  'Gaussian CDF'),
    'log_lik_obs': ('obs',  'log-likelihood'),
    'lik_ctx':     ('ctx',  'P(true class)'),
    'lik_dpos':    ('dpos', 'P(true class)'),
    'lik_rule':    ('rule', 'P(true class)'),
}

# Per-file scalar parameters carried in the deviant-probability CSVs themselves.
DEVIANT_PARAM_COLS = ['lim_std', 'd', 'tau_std']

# Wider set of generative parameters, constant within a sequence and read back
# from the original trial file. 'lim_std' is the mu shown in the file name.
SEQUENCE_PARAM_COLS = [
    'lim_std', 'lim_dev', 'tau_std', 'tau_dev', 'd',
    'sigma_q_std', 'sigma_q_dev', 'sigma_r',
    'duration_tones', 'ISI', 'run_n', 'session_n',
]


# =============================================================================
# Model lists
# =============================================================================

# Fixed sigma_r: the models the experimental-sequence analyses compare.
FIXED_SIGMA_R_MODELS = [
    'population_network_all_bn8_trainh0_fixedsir0.02_epochs300_lr0.002',
    'population_network_all_bn8_trainh0_fixedsir0.05_epochs300_lr0.002',
    'population_network_all_bn8_trainh0_fixedsir0.1_epochs300_lr0.002',
]

EVALUATION_MODELS = [
    'population_network_all_bn8_trainh0_fixedsir0.02_epochs300_lr0.002',
    'population_network_all_bn8_trainh0_fixedsir0.05_epochs300_lr0.002',
    'population_network_all_bn8_trainh0_fixedsir0.1_epochs300_lr0.002',
    'population_network_all_bn8_trainh0_epochs300_lr0.002'
]



# =============================================================================
# Figure defaults
# =============================================================================

DPI = 150


if __name__ == '__main__':
    pass
