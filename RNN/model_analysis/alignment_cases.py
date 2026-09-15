"""Per-trial alignment between the ctx (context) and dpos (deviant-position) modules.

For each trained model and each trial of each experimental sequence, two
questions are asked:

  1. *dpos module* -- when, within the 8-tone trial, does the module commit to the
     trial's true deviant position, and does it hold that commitment?
  2. *ctx module* -- the position at which the ctx module first calls a deviant:
     is it endorsed by the dpos module, and when?

Alignment convention
--------------------
The model is run on ``y[:, :-1]``, so output row ``k`` is the module's report
*about* timestep ``k + 1``: within-trial position ``(k + 1) % period``, trial
``(k + 1) // period``. This is the same alignment the dpos response window uses at
training time (``within_trial_pos = arange(1, T) % N_tones`` in
``pipeline_core_v2.compute_loss``), where ``rel == 0`` (the deviant timestep) and
``rel == 1`` (the step after) are the two up-weighted commitment steps. Timestep 0
of a sequence therefore has no report, so the first trial is dropped by default.

A dpos output *class* ``c`` encodes deviant position ``c + dpos_min`` in the model's
convention; ``dpos_conventions()`` converts that back to the raw within-trial
position used by the sequence files (2..6), so predictions, ctx report positions and
ground truth are all compared on one 0-indexed within-trial scale.

The two case families are exhaustive and mutually exclusive: every trial has
exactly one dpos case and exactly one ctx case, so ``case_0``..``case_7`` sum to 1
across a model's row of the summary table and ``case_A``..``case_N`` do too.

dpos cases 6 and 7 are additions to the six originally requested categories,
which are not exhaustive (a trial can be wrong throughout, or first become
correct at, say, position 1 with a deviant at position 4); silently dropping
those trials would bias every proportion computed from the table. "Maintained"
is read literally throughout -- correct at *every* position from the first
correct one to the end of the trial.

Split out of assess_dpos_and_ctx_detection.py and its _summary companion, which
between them held this taxonomy, the per-trial table, and the per-model
proportions table. The summary half previously failed to import at all: it
imported a module name that no longer existed after the script was renamed.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis_core import (
    ModelInfo,
    dpos_conventions,
    get_module_probabilities,
    load_model,
    load_trial_sequence,
)


# =============================================================================
# The two case taxonomies
# =============================================================================

# Sentinel written into the position-0 slot that has no model report. Chosen far
# outside the dpos range so it can never be mistaken for a prediction or a match.
NO_REPORT = -99

# Leading trials to drop: trial 0 has no report at position 0.
DEFAULT_SKIP_TRIALS = 1

# Probability above which the ctx module is taken to be calling a deviant, under
# the 'threshold' rule.
DEFAULT_CTX_THRESHOLD = 0.5

CTX_RULES = ('argmax', 'max-prob', 'threshold')

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

# Observation-noise provenance carried into both output CSVs; see si_r_columns().
SI_R_COLUMNS = ('si_r_fixed', 'si_r_bounds_low', 'si_r_bounds_high')

# Column order: dpos cases first (numeric), then ctx cases (letters), matching the
# order the per-trial CSV lays them out in.
DPOS_COLUMNS = [f'case_{code}' for code in DPOS_CASES]
CTX_COLUMNS = [f'case_{code}' for code in CTX_CASES]

DEFAULT_OUTPUT_NAME = 'alignment_case_proportions.csv'


# =============================================================================
# Run settings
# =============================================================================

@dataclass
class AlignmentConfig:
    """Everything one assessment run needs beyond the model list and the files.

    No defaults: the entry point sets every field explicitly, so a run's settings
    are readable in one place rather than spread between a parser and a call.
    `analysis_config` holds the values the project normally uses.
    """
    model_dir: Path          # directory holding the trained model folders
    output_root: Path        # <output_root>/<model>/alignment/ per model
    period: int              # timesteps per trial
    cue_seed: int            # cue re-encoding seed; fixed across stages
    chunk_size: int          # sequences per batched forward pass
    skip_trials: int         # leading trials dropped per sequence
    ctx_rule: str            # one of CTX_RULES
    ctx_threshold: float     # used only by the 'threshold' rule
    no_summary: bool         # skip the per-case proportions CSV


# =============================================================================
# Sequence loading and the forward pass
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
# Classifying one trial
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
# Per-trial and per-model tables
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
    info = ModelInfo.from_path(model_path)
    model = load_model(info)
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
# Collapsing per-trial tables into per-model case proportions
# =============================================================================

def find_trial_tables(output_root, model_names=None):
    """Per-trial alignment CSVs under `output_root`, optionally restricted and ordered."""
    if model_names:
        paths = [output_root / name / 'alignment' / f'{name}_trial_alignment.csv'
                 for name in model_names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "no per-trial alignment table for: "
                + ', '.join(p.parent.parent.name for p in missing)
                + " — run the per-trial stage for those models first")
        return paths
    return sorted(output_root.glob('*/alignment/*_trial_alignment.csv'))


def load_cases(paths):
    """Read the model, the two case columns, and any si_r provenance columns present.

    The si_r columns are optional: tables written before they existed are read
    unchanged, and the resulting per-model rows simply carry no si_r values.
    """
    frames = []
    for path in paths:
        # Only a handful of columns are needed, so the 90k-row tables load in a
        # fraction of the time (and memory) a full read would take.
        available = pd.read_csv(path, nrows=0).columns
        columns = (['model', 'dpos_case', 'ctx_case']
                   + [name for name in SI_R_COLUMNS if name in available])
        frames.append(pd.read_csv(path, usecols=columns))
        print(f"  read {path.name}: {len(frames[-1])} trials")
    return pd.concat(frames, ignore_index=True)


def build_case_table(df, model_names=None):
    """One row per model, one column per case, cells = proportion of that model's trials."""
    # si_r is a property of the checkpoint, so it is constant within a model group and
    # one value per group carries it into the wide table. Absent from older inputs.
    si_r_present = [name for name in SI_R_COLUMNS if name in df.columns]

    rows = []
    for model, group in df.groupby('model', sort=False):
        n_trials = len(group)
        row = {'model': model}
        row.update({name: group[name].iloc[0] for name in si_r_present})
        row['n_trials'] = n_trials
        for cases, column in ((DPOS_CASES, 'dpos_case'), (CTX_CASES, 'ctx_case')):
            counts = group[column].value_counts()
            for code in cases:
                row[f'case_{code}'] = counts.get(code, 0) / n_trials
        rows.append(row)

    table = pd.DataFrame(
        rows,
        columns=['model'] + si_r_present + ['n_trials'] + DPOS_COLUMNS + CTX_COLUMNS)
    if model_names:
        # Preserve the order the models were asked for, not the order they were read in.
        order = {name: i for i, name in enumerate(model_names)}
        table = (table.sort_values('model', key=lambda s: s.map(order))
                      .reset_index(drop=True))
    return table


def check_family_sums(table, tol=1e-9):
    """Warn if either family's proportions fail to sum to 1 for some model."""
    for family, columns in (('dpos', DPOS_COLUMNS), ('ctx', CTX_COLUMNS)):
        sums = table[columns].sum(axis=1)
        bad = table.loc[(sums - 1).abs() > tol, 'model']
        for model in bad:
            print(f"  Warning: {family} proportions do not sum to 1 for {model}")


def print_legend():
    print("\nCase legend")
    for cases, family in ((DPOS_CASES, 'dpos'), (CTX_CASES, 'ctx')):
        for code, description in cases.items():
            print(f"  case_{code}  [{family}]  {description}")


if __name__ == '__main__':
    pass
