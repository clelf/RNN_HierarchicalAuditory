"""Per-trial alignment between the ctx (context) and dpos (deviant-position) modules.

For each trained model and each trial of each experimental sequence, this script
asks two questions:

  1. *dpos module* — when, within the 8-tone trial, does the module commit to the
     trial's true deviant position, and does it hold that commitment?
  2. *ctx module* — the position at which the ctx module first calls a deviant:
     is it endorsed by the dpos module, and when?

Alignment convention
--------------------
The model is run on ``y[:, :-1]``, so output row ``k`` is the module's report
*about* timestep ``k + 1``: within-trial position ``(k + 1) % period``, trial
``(k + 1) // period``. This is the same alignment the dpos response window uses at
training time (``within_trial_pos = arange(1, T) % N_tones`` in
``pipeline_core_v2.compute_loss``), where ``rel == 0`` (the deviant timestep) and
``rel == 1`` (the step after) are the two up-weighted commitment steps — exactly the
positions cases 2/3 and 4/5 below are about. Timestep 0 of a sequence therefore has
no report, so the first trial is dropped by default (``--skip-trials``).

A dpos output *class* ``c`` encodes deviant position ``c + dpos_min`` in the model's
convention; ``dpos_conventions()`` converts that back to the raw within-trial
position used by the sequence files (2..6), so predictions, ctx report positions and
ground truth are all compared on one 0-indexed within-trial scale.

dpos assessment (one case per trial)
------------------------------------
Let ``correct[p]`` be "the dpos argmax at position p equals the trial's true deviant
position", and ``first`` the earliest such p.

  0  correct from position 0 and never changes (correct at every position)
  1  correct at position 0 but changes later
  2  first correct AT the deviant, and maintained to the end of the trial
  3  first correct AT the deviant, not maintained
  4  first correct the step AFTER the deviant, and maintained to the end
  5  first correct the step AFTER the deviant, not maintained
  6  never correct within the trial                    [not in the original list]
  7  first correct at some other position              [not in the original list]

Cases 6 and 7 are additions: the six requested categories are not exhaustive (a
trial can be wrong throughout, or first become correct at, say, position 1 with a
deviant at position 4), and silently dropping those trials would bias every
proportion computed from this table. "Maintained" is read literally throughout —
correct at *every* position from the first correct one to the end — so cases 2/3 and
4/5 partition their branch the same way C/D and E/F do below.

ctx assessment (one case per trial)
-----------------------------------
Let ``p_ctx`` be the first within-trial position at which the ctx module reports a
deviant (see ``--ctx-rule``), ``match[p]`` be "the dpos argmax at position p points
at ``p_ctx``", and ``first`` the earliest such p.

  A  no position in the trial matches p_ctx
  B  first matched by a prediction realised BEFORE p_ctx
  C  first matched AT p_ctx, and maintained afterwards
  D  first matched AT p_ctx, not maintained
  E  first matched at the IMMEDIATE NEXT position, and maintained afterwards
  F  first matched at the immediate next position, not maintained
  G  first matched later than the immediate next position   [not in the list]
  N  the ctx module never reports a deviant in this trial   [not in the list]

Case N is not a corner case: with the plain argmax rule the ctx module leaves most
trials without any label-1 timestep (P(deviant) rarely clears 0.5, since deviants are
1 tone in 8). ``--ctx-rule max-prob`` or ``--ctx-rule threshold`` give a report on
every trial if that is the question of interest. Ordering is by *first* match, so a
prediction realised before p_ctx yields B even if p_ctx is also matched and held.
Note a dpos prediction can only ever match p_ctx in 2..6, so a ctx report at
position 0, 1 or 7 is case A by construction.

Output
------
``<output-root>/<model name>/alignment/<model name>_trial_alignment.csv``
    one row per trial, with the two case labels, one 0/1 indicator column per case
    (case_0..case_7, case_A..case_G, case_N), the quantities the cases are derived
    from, and one string per module giving its per-position call across the trial
    (``dpos_pred_seq`` e.g. '44222222', ``ctx_report_seq`` e.g. '....D...').
``<output-root>/<model name>/alignment/<model name>_alignment_summary.csv``
    counts and proportions per case (suppress with --no-summary).

Both carry the model's observation-noise setting: ``si_r_fixed`` when the run pinned
sigma_r, otherwise ``si_r_bounds_low``/``si_r_bounds_high`` for the range it was
sampled from. Exactly one of the two is filled; the other is NaN.

Examples
--------
    # every model in DEFAULT_MODEL_NAMES, every sequence file
    python assess_module_alignment.py

    # two models, 200 sequences, ctx report = peak P(deviant) rather than argmax
    python assess_module_alignment.py --n-sequences 200 --ctx-rule max-prob \
        --model-names population_network_all_bn8_trainh0_fixedsir_lr0.002_epochs300 \
                      population_network_all_bn8_trainh0_fixedsir0.05_epochs300_lr0.002
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_models as eval
from model_activations import (
    load_trial_sequence,
    get_module_probabilities,
    dpos_conventions,
)


# =============================================================================
# Defaults (override on the command line; see --help)
# =============================================================================

DEFAULT_MODEL_DIR = Path(
    "/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/training_results/N_ctx_2/HierarchicalGM")

# Same list as run_exp_trials_pipeline.DEFAULT_MODEL_NAMES.
DEFAULT_MODEL_NAMES = [
    "population_network_all_bn8_trainh0_fixedsir_lr0.002_epochs200_lrsched",
    "population_network_all_bn8_trainh0_fixedsir_lr0.002_epochs300",
    "population_network_all_bn8_trainh0_fixedsir0.05_epochs300_lr0.002",
    "population_network_all_bn8_trainh0_fixedsir0.005_epochs300_lr0.002",
    "population_network_all_bn8_trainh0_fixedsir0.1_epochs300_lr0.002",
]

DEFAULT_TRIALS_PATH = Path("/home/clevyfidel/Documents/Workspace/Jasmin/trialsequences2clem")
DEFAULT_OUTPUT_ROOT = Path("/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/exp_seq_act_output")

DEFAULT_PERIOD = 8          # timesteps per trial
DEFAULT_CUE_SEED = 11       # cue re-encoding seed; same value as the other stages
DEFAULT_CHUNK_SIZE = 128    # sequences per batched forward pass
DEFAULT_SKIP_TRIALS = 1     # leading trials to drop (trial 0 has no report at position 0)
DEFAULT_CTX_THRESHOLD = 0.5

# Sentinel written into the position-0 slot that has no model report. Chosen far
# outside the dpos range so it can never be mistaken for a prediction or a match.
NO_REPORT = -99

DPOS_CASES = {
    0: 'correct from position 0, never changes',
    1: 'correct from position 0, changes later',
    2: 'first correct at the deviant, maintained to the end',
    3: 'first correct at the deviant, not maintained',
    4: 'first correct one step after the deviant, maintained to the end',
    5: 'first correct one step after the deviant, not maintained',
    6: 'never correct within the trial',
    7: 'first correct at some other position',
}

CTX_CASES = {
    'A': 'ctx report matched by no dpos prediction in the trial',
    'B': 'matched by a dpos prediction realised earlier',
    'C': 'matched at the same position, maintained afterwards',
    'D': 'matched at the same position, not maintained',
    'E': 'matched at the immediate next position, maintained afterwards',
    'F': 'matched at the immediate next position, not maintained',
    'G': 'first matched later than the immediate next position',
    'N': 'ctx module never reports a deviant in the trial',
}

CTX_RULES = ('argmax', 'max-prob', 'threshold')

# Observation-noise provenance carried into both output CSVs; see si_r_columns().
SI_R_COLUMNS = ('si_r_fixed', 'si_r_bounds_low', 'si_r_bounds_high')


# =============================================================================
# Sequence loading
# =============================================================================

def si_r_columns(info):
    """The model's observation-noise setting, as columns for the output tables.

    A training run either pins sigma_r (``si_r_fixed``) or samples it per sequence
    from ``si_r_bounds``; the two are mutually exclusive, so exactly one of
    ``si_r_fixed`` and (``si_r_bounds_low``, ``si_r_bounds_high``) is filled and the
    other is NaN.

    Read from the saved run config rather than ``info.data_config_dict``, because
    ``DataConfig.to_gm_dict`` encodes a pinned sigma_r as *degenerate* bounds
    (low == high == si_r_fixed) so the generative code needs no special case — which
    leaves the two situations indistinguishable in the gm dict. A ModelInfo inferred
    from the directory structure has no run config at all; there the degenerate-bounds
    encoding is the only signal available, so it is read back as a pinned value.
    """
    nan = float('nan')
    data = getattr(info.run_config, 'data', None)
    if data is not None:
        fixed = data.si_r_fixed
        bounds = data.si_r_bounds or {}
    else:
        bounds = (info.data_config_dict or {}).get('si_r_bounds') or {}
        low, high = bounds.get('low'), bounds.get('high')
        fixed = low if (low is not None and low == high) else None

    if fixed is not None:
        return {'si_r_fixed': float(fixed),
                'si_r_bounds_low': nan, 'si_r_bounds_high': nan}
    return {'si_r_fixed': nan,
            'si_r_bounds_low': float(bounds['low']) if bounds.get('low') is not None else nan,
            'si_r_bounds_high': float(bounds['high']) if bounds.get('high') is not None else nan}


def find_trial_files(trials_path):
    """Sequence files in `trials_path`, preferring .csv and falling back to .txt."""
    files = sorted(trials_path.glob("*.csv"))
    if not files:
        files = sorted(trials_path.glob("*.txt"))
    return files


def select_files(all_files, n_sequences, seed=0):
    """All files, or `n_sequences` of them sampled without replacement."""
    if n_sequences is None or n_sequences >= len(all_files):
        return list(all_files)
    rng = np.random.default_rng(seed=seed)
    return sorted(rng.choice(all_files, size=n_sequences, replace=False).tolist())


def load_sequences(files, n_cue_classes, cue_seed, period):
    """Read every sequence file once, keeping what the assessment needs per trial.

    The per-trial ground truth (true deviant position, rule) is constant within a
    trial, so it is sampled at the trial's first timestep.
    """
    seqs = []
    for f in files:
        obs, cue, ctx, dpos, rule, lim_std, d, tau_std, trial_n = load_trial_sequence(
            f, return_hierarch=True, n_cue_classes=n_cue_classes, cue_seed=cue_seed)
        n_trials = obs.shape[0] // period
        ctx_grid = ctx[:n_trials * period].reshape(n_trials, period)
        seqs.append(dict(
            name=f.stem,
            obs=obs,
            cue=cue,
            n_trials=n_trials,
            true_dpos=dpos[::period].astype(np.int16),
            rule=rule[::period].astype(np.int8),
            trial_id=np.asarray(trial_n)[::period].astype(np.int32),
            lim_std=lim_std, d=d, tau_std=tau_std,
            # Where the deviant actually sits, read off the trial_type column; used
            # only to check that it agrees with the dpos column.
            true_dev_count=(ctx_grid == 1).sum(axis=1).astype(np.int8),
            true_dev_pos=np.argmax(ctx_grid == 1, axis=1).astype(np.int16),
        ))
    return seqs


def check_ground_truth(seqs):
    """Warn if trial_type and dpos disagree about where the deviant is."""
    bad_count = sum(int((s['true_dev_count'] != 1).sum()) for s in seqs)
    bad_pos = sum(int((s['true_dev_pos'] != s['true_dpos']).sum()) for s in seqs)
    if bad_count:
        print(f"  Warning: {bad_count} trial(s) do not have exactly one trial_type==1 timestep")
    if bad_pos:
        print(f"  Warning: {bad_pos} trial(s) where trial_type==1 is not at the dpos position")


# =============================================================================
# Forward pass -> per-trial position grids
# =============================================================================

def to_trial_grid(arr, period, fill):
    """(N, T-1) model outputs -> (N, n_trials, period), indexed by within-trial position.

    Output row ``k`` reports about timestep ``k + 1``, so a sentinel column is
    prepended to put every value at the timestep it refers to. Timestep 0 has no
    report and receives `fill`.
    """
    n, length = arr.shape
    padded = np.concatenate(
        [np.full((n, 1), fill, dtype=arr.dtype), arr], axis=1)          # (N, T)
    n_trials, remainder = divmod(padded.shape[1], period)
    if remainder:
        raise ValueError(f"sequence length {padded.shape[1]} is not a multiple of period {period}")
    return padded.reshape(n, n_trials, period)


def run_model_on_sequences(model, seqs, period, chunk_size, dpos_min, dpos_shift, skip_trials):
    """Batch `seqs` through `model` and return per-trial grids of module reports.

    Returns
    -------
    dpos_pred : (M, period) int16
        Predicted deviant position (raw within-trial scale, same as the files),
        per within-trial position.
    ctx_lab : (M, period) int8
        ctx module argmax label per within-trial position.
    ctx_p1 : (M, period) float32
        ctx module P(deviant) per within-trial position.
    M is the number of retained trials, sequences concatenated in order.
    """
    lengths = {s['obs'].shape[0] for s in seqs}
    if len(lengths) != 1:
        raise ValueError(
            f"sequences have differing lengths {sorted(lengths)}; batching them would "
            "require truncation, which would silently change the trial grid")

    # class c -> model-convention position c + dpos_min -> raw file position minus the shift
    class_to_position = dpos_min - dpos_shift

    dpos_chunks, lab_chunks, p1_chunks = [], [], []
    for start in range(0, len(seqs), chunk_size):
        chunk = seqs[start:start + chunk_size]
        y = torch.tensor(np.stack([s['obs'] for s in chunk]),
                         dtype=torch.float32).unsqueeze(-1)             # (N, T, 1)
        q = torch.tensor(np.stack([s['cue'] for s in chunk]),
                         dtype=torch.float32)                           # (N, T, n_cue)

        probs = get_module_probabilities(model, y, q)                   # (T-1, N, dim)
        dpos_cls = np.argmax(probs['dpos'], axis=-1).T.astype(np.int16)  # (N, T-1)
        ctx_lab = np.argmax(probs['ctx'], axis=-1).T.astype(np.int8)     # (N, T-1)
        ctx_p1 = probs['ctx'][:, :, 1].T.astype(np.float32)              # (N, T-1)

        # Sentinels for the unreported timestep 0: a dpos position no trial can have,
        # "no deviant" for the ctx label, and a probability below any threshold.
        dpos_pos = (dpos_cls + class_to_position).astype(np.int16)       # (N, T-1)
        dpos_grid = to_trial_grid(dpos_pos, period, NO_REPORT)
        lab_grid = to_trial_grid(ctx_lab, period, 0)
        p1_grid = to_trial_grid(ctx_p1, period, -1.0)

        dpos_chunks.append(dpos_grid[:, skip_trials:, :].reshape(-1, period))
        lab_chunks.append(lab_grid[:, skip_trials:, :].reshape(-1, period))
        p1_chunks.append(p1_grid[:, skip_trials:, :].reshape(-1, period))
        print(f"  processed {min(start + chunk_size, len(seqs))}/{len(seqs)} sequences")

    return (np.concatenate(dpos_chunks, axis=0),
            np.concatenate(lab_chunks, axis=0),
            np.concatenate(p1_chunks, axis=0))


# =============================================================================
# Case assignment
# =============================================================================

def _suffix_all(mask):
    """suffix[m, p] = mask[m, p:].all() — "holds from p to the end of the trial"."""
    return np.logical_and.accumulate(mask[:, ::-1], axis=1)[:, ::-1]


def _first_true(mask):
    """(first index where mask is True, or -1 if never true) per row, plus the hit flag."""
    hit = mask.any(axis=1)
    return np.where(hit, mask.argmax(axis=1), -1), hit


def classify_dpos(dpos_pred, true_dpos):
    """Assign dpos case 0-7 per trial (see module docstring).

    Parameters
    ----------
    dpos_pred : (M, period) int
        Predicted deviant position at each within-trial position.
    true_dpos : (M,) int
        The trial's true deviant position.

    Returns
    -------
    case : (M,) int8
    first_correct : (M,) int8    first position where the prediction is correct, else -1
    maintained : (M,) bool       correct at every position from first_correct onwards
    """
    correct = dpos_pred == true_dpos[:, None]                            # (M, period)
    first_correct, has_correct = _first_true(correct)
    suffix = _suffix_all(correct)
    rows = np.arange(len(dpos_pred))
    maintained = has_correct & suffix[rows, np.maximum(first_correct, 0)]

    # Branch on where the first correct prediction falls; the checks are ordered so
    # "from the beginning" wins if a trial's deviant were ever at position 0.
    at_start = has_correct & (first_correct == 0)
    at_deviant = has_correct & ~at_start & (first_correct == true_dpos)
    after_deviant = has_correct & ~at_start & (first_correct == true_dpos + 1)

    case = np.full(len(dpos_pred), 7, dtype=np.int8)   # first correct elsewhere
    case[~has_correct] = 6                             # never correct
    case[at_start] = np.where(maintained[at_start], 0, 1)
    case[at_deviant] = np.where(maintained[at_deviant], 2, 3)
    case[after_deviant] = np.where(maintained[after_deviant], 4, 5)
    return case, first_correct.astype(np.int8), maintained


def ctx_report_mask(ctx_lab, ctx_p1, rule, threshold):
    """Boolean (M, period) mask of "the ctx module calls a deviant here".

    argmax     the module's own label is 1 (what the module actually outputs)
    threshold  P(deviant) reaches `threshold`
    max-prob   the single position with the highest P(deviant) in the trial
    """
    if rule == 'argmax':
        return ctx_lab == 1
    if rule == 'threshold':
        return ctx_p1 >= threshold
    if rule == 'max-prob':
        mask = np.zeros(ctx_p1.shape, dtype=bool)
        mask[np.arange(len(ctx_p1)), ctx_p1.argmax(axis=1)] = True
        return mask
    raise ValueError(f"unknown ctx rule: {rule!r}")


def classify_ctx(dpos_pred, report):
    """Assign ctx case 'A'-'G'/'N' per trial (see module docstring).

    Parameters
    ----------
    dpos_pred : (M, period) int
        Predicted deviant position at each within-trial position.
    report : (M, period) bool
        Where the ctx module calls a deviant.

    Returns
    -------
    case : (M,) '<U1'
    ctx_pos : (M,) int8      first ctx report position, else -1
    n_reports : (M,) int8    how many positions the ctx module calls a deviant at
    first_match : (M,) int8  first position whose dpos prediction points at ctx_pos, else -1
    maintained : (M,) bool   points at ctx_pos at every position from first_match onwards
    """
    ctx_pos, has_ctx = _first_true(report)

    # The dpos module "endorses" the ctx report at position p when the position it
    # predicts is the position the ctx module flagged.
    match = (dpos_pred == ctx_pos[:, None]) & has_ctx[:, None]
    first_match, has_match = _first_true(match)
    suffix = _suffix_all(match)
    rows = np.arange(len(dpos_pred))
    maintained = has_match & suffix[rows, np.maximum(first_match, 0)]

    same = has_match & (first_match == ctx_pos)
    nxt = has_match & (first_match == ctx_pos + 1)

    case = np.full(len(dpos_pred), 'G', dtype='<U1')   # matched later than ctx_pos + 1
    case[same] = np.where(maintained[same], 'C', 'D')
    case[nxt] = np.where(maintained[nxt], 'E', 'F')
    case[has_match & (first_match < ctx_pos)] = 'B'
    case[has_ctx & ~has_match] = 'A'
    case[~has_ctx] = 'N'
    return (case, ctx_pos.astype(np.int8), report.sum(axis=1).astype(np.int8),
            first_match.astype(np.int8), maintained)


# =============================================================================
# Assembly and output
# =============================================================================

def _dpos_strings(grid, no_report_char='.'):
    """Per-position predicted deviant position, one string per trial, e.g. '44444444'.

    Positions are single digits 2..6, so the string is fixed-width and survives being
    read back as an integer; `no_report_char` marks the position the model never
    reports on (only present when --skip-trials is 0).
    """
    text = np.where(grid == NO_REPORT, no_report_char, grid.astype(str))
    return [''.join(row) for row in text]


def _report_strings(report, hit='D', miss='.'):
    """Per-position ctx deviant calls, one string per trial, e.g. '....D...'.

    Deliberately non-numeric: a 0/1 string would lose its leading zeros the moment
    the CSV is read back with default dtypes.
    """
    text = np.where(report, hit, miss)
    return [''.join(row) for row in text]


def build_trial_frame(model_name, si_r, seqs, skip_trials, dpos_pred, ctx_lab, ctx_p1, cfg):
    """One row per retained trial: identifiers, both case labels, indicators, diagnostics.

    ``si_r`` (from ``si_r_columns``) is constant for the model, so it is broadcast
    down the frame to keep every row self-describing about the checkpoint it came from.
    """
    true_dpos = np.concatenate([s['true_dpos'][skip_trials:] for s in seqs]).astype(np.int16)

    dpos_case, dpos_first, dpos_maintained = classify_dpos(dpos_pred, true_dpos)
    report = ctx_report_mask(ctx_lab, ctx_p1, cfg.ctx_rule, cfg.ctx_threshold)
    ctx_case, ctx_pos, ctx_n, ctx_first, ctx_maintained = classify_ctx(dpos_pred, report)

    df = pd.DataFrame({
        'model':        model_name,
        **si_r,
        'sequence':     np.repeat([s['name'] for s in seqs],
                                  [s['n_trials'] - skip_trials for s in seqs]),
        'trial_n':      np.concatenate([s['trial_id'][skip_trials:] for s in seqs]),
        'true_dpos':    true_dpos,
        'rule':         np.concatenate([s['rule'][skip_trials:] for s in seqs]),
        'tau_std':      np.repeat([s['tau_std'] for s in seqs],
                                  [s['n_trials'] - skip_trials for s in seqs]),
        'd':            np.repeat([s['d'] for s in seqs],
                                  [s['n_trials'] - skip_trials for s in seqs]),
        'lim_std':      np.repeat([s['lim_std'] for s in seqs],
                                  [s['n_trials'] - skip_trials for s in seqs]),
        'dpos_case':    dpos_case,
        'ctx_case':     ctx_case,
    })

    # One 0/1 column per assessment category, so a trial's row reads directly as the
    # set of categories it falls in (exactly one per family).
    for code in DPOS_CASES:
        df[f'case_{code}'] = (dpos_case == code).astype(np.int8)
    for code in CTX_CASES:
        df[f'case_{code}'] = (ctx_case == code).astype(np.int8)

    df['dpos_first_correct'] = dpos_first
    df['dpos_maintained'] = dpos_maintained.astype(np.int8)
    df['ctx_report_pos'] = ctx_pos
    df['ctx_n_reports'] = ctx_n
    df['ctx_first_match'] = ctx_first
    df['ctx_match_maintained'] = ctx_maintained.astype(np.int8)
    df['dpos_pred_seq'] = _dpos_strings(dpos_pred)
    df['ctx_report_seq'] = _report_strings(report)
    return df


def build_summary_frame(df, model_name):
    """Counts and proportions per case, long format, both families stacked.

    The si_r columns are lifted off the trial frame rather than recomputed, so the
    two files can never disagree about the checkpoint they describe.
    """
    si_r = {name: df[name].iloc[0] for name in SI_R_COLUMNS if name in df.columns}
    rows = []
    for family, cases, column in (('dpos', DPOS_CASES, 'dpos_case'),
                                  ('ctx', CTX_CASES, 'ctx_case')):
        counts = df[column].value_counts()
        for code, description in cases.items():
            n = int(counts.get(code, 0))
            rows.append({'model': model_name, **si_r, 'family': family, 'case': code,
                         'description': description, 'n_trials': n,
                         'proportion': n / len(df) if len(df) else np.nan})
    return pd.DataFrame(rows)


def run_one_model(model_name, files, seq_cache, cfg):
    """Assess one model over every sequence file; returns (trial frame, summary frame)."""
    model_path = cfg.model_dir / model_name
    info = eval.ModelInfo.from_path(model_path)
    model = eval.load_model(info)
    model.eval()

    n_cue_classes = len(info.data_config_dict['cues_set'])
    dpos_min, dpos_shift = dpos_conventions(info)
    si_r = si_r_columns(info)
    print(f"Loaded model: {model_name}")
    print(f"  cue classes: {n_cue_classes}; dpos class-index offset: {dpos_min}; "
          f"experimental dpos shift: +{dpos_shift}")
    print("  sigma_r: " + (f"pinned at {si_r['si_r_fixed']}"
                           if si_r['si_r_fixed'] == si_r['si_r_fixed']
                           else f"sampled from [{si_r['si_r_bounds_low']}, "
                                f"{si_r['si_r_bounds_high']}]"))

    # Sequence loading depends only on the cue encoding, so models that share it
    # (all of DEFAULT_MODEL_NAMES do) read the 1600+ CSVs once between them.
    key = (n_cue_classes, cfg.cue_seed)
    if key not in seq_cache:
        print(f"  reading {len(files)} sequence file(s) (cue encoding {key})")
        seq_cache[key] = load_sequences(files, n_cue_classes, cfg.cue_seed, cfg.period)
        check_ground_truth(seq_cache[key])
    seqs = seq_cache[key]

    dpos_pred, ctx_lab, ctx_p1 = run_model_on_sequences(
        model, seqs, cfg.period, cfg.chunk_size, dpos_min, dpos_shift, cfg.skip_trials)

    df = build_trial_frame(model_name, si_r, seqs, cfg.skip_trials,
                           dpos_pred, ctx_lab, ctx_p1, cfg)
    summary = build_summary_frame(df, model_name)

    out_dir = cfg.output_root / model_name / 'alignment'
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_file = out_dir / f"{model_name}_trial_alignment.csv"
    df.to_csv(trial_file, index=False)
    print(f"  Saved: {trial_file}  ({len(df)} trials)")
    if not cfg.no_summary:
        summary_file = out_dir / f"{model_name}_alignment_summary.csv"
        summary.to_csv(summary_file, index=False)
        print(f"  Saved: {summary_file}")

    for family in ('dpos', 'ctx'):
        part = summary[summary['family'] == family]
        shares = '  '.join(f"{r.case}:{r.proportion:.1%}" for r in part.itertuples()
                           if r.n_trials)
        print(f"  {family:>4}: {shares}")
    return df, summary


# =============================================================================
# Entry point
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-names', nargs='+', default=list(DEFAULT_MODEL_NAMES),
                   metavar='NAME',
                   help='trained model directory names, run in order (default: the '
                        f'{len(DEFAULT_MODEL_NAMES)} models listed in DEFAULT_MODEL_NAMES)')
    p.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
                   help='directory containing the trained models (default: %(default)s)')
    p.add_argument('--trials-path', type=Path, default=DEFAULT_TRIALS_PATH,
                   help='directory of experimental sequence files (default: %(default)s)')
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help='base output directory; each model writes to '
                        '<output-root>/<model-name>/alignment/ (default: %(default)s)')
    p.add_argument('--combined-csv', type=Path, default=None,
                   help='also write every model\'s trials to this single CSV')
    p.add_argument('--n-sequences', type=int, default=None,
                   help='sequence files to use (default: all of them)')
    p.add_argument('--seed', type=int, default=0,
                   help='seed for random file sampling (default: %(default)s)')
    p.add_argument('--period', type=int, default=DEFAULT_PERIOD,
                   help='timesteps per trial (default: %(default)s)')
    p.add_argument('--cue-seed', type=int, default=DEFAULT_CUE_SEED,
                   help='seed for cue re-encoding; keep it fixed across stages '
                        '(default: %(default)s)')
    p.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                   help='sequences per batched forward pass (default: %(default)s)')
    p.add_argument('--skip-trials', type=int, default=DEFAULT_SKIP_TRIALS,
                   help='leading trials to drop per sequence; the default of 1 drops the '
                        'trial whose position 0 the model never reports on '
                        '(default: %(default)s)')
    p.add_argument('--ctx-rule', choices=CTX_RULES, default='argmax',
                   help="how the ctx module 'reports a deviant': argmax = its output "
                        'label is 1; threshold = P(deviant) >= --ctx-threshold; '
                        'max-prob = the trial\'s peak P(deviant) (default: %(default)s)')
    p.add_argument('--ctx-threshold', type=float, default=DEFAULT_CTX_THRESHOLD,
                   help='probability threshold for --ctx-rule threshold (default: %(default)s)')
    p.add_argument('--no-summary', action='store_true',
                   help='skip the per-case counts/proportions CSV')
    return p.parse_args(argv)


def main(argv=None):
    cfg = parse_args(argv)

    all_files = find_trial_files(cfg.trials_path)
    if not all_files:
        raise FileNotFoundError(f"No .csv or .txt sequence files in {cfg.trials_path}")
    files = select_files(all_files, cfg.n_sequences, seed=cfg.seed)
    print(f"Found {len(all_files)} trial sequence files in {cfg.trials_path}; using {len(files)}")
    print(f"Models ({len(cfg.model_names)}): {', '.join(cfg.model_names)}")
    print(f"ctx report rule: {cfg.ctx_rule}"
          + (f" (threshold {cfg.ctx_threshold})" if cfg.ctx_rule == 'threshold' else ''))

    # A failure on one model (a missing checkpoint, say) should not throw away the
    # models already done or block the ones still queued; report at the end instead.
    seq_cache = {}
    frames, summaries, failures = [], [], []
    for i, model_name in enumerate(cfg.model_names, start=1):
        print(f"\n{'=' * 79}\n[{i}/{len(cfg.model_names)}] {model_name}\n{'=' * 79}")
        try:
            df, summary = run_one_model(model_name, files, seq_cache, cfg)
            frames.append(df)
            summaries.append(summary)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures.append((model_name, exc))

    if cfg.combined_csv and frames:
        cfg.combined_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(cfg.combined_csv, index=False)
        print(f"\nSaved combined table: {cfg.combined_csv}")

    n_ok = len(cfg.model_names) - len(failures)
    print(f"\nDone: {n_ok}/{len(cfg.model_names)} model(s) completed.")
    for model_name, exc in failures:
        print(f"  FAILED {model_name}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
