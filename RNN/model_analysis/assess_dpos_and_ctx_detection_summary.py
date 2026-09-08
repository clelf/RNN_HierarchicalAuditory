"""Wide per-model table of ctx/dpos alignment case proportions.

Collapses the per-trial tables written by ``assess_module_alignment.py`` into one
row per model, with one column per assessment case (``case_0``..``case_7`` for the
dpos module, ``case_A``..``case_G`` and ``case_N`` for the ctx module — the same
column names the per-trial CSV uses) and, in each cell, the proportion of that
model's trials falling in that case.

The model's observation-noise setting (``si_r_fixed``, or ``si_r_bounds_low`` /
``si_r_bounds_high`` when sigma_r was sampled rather than pinned) is carried over
from the per-trial tables when they have it, so the table is self-describing.
Inputs written before those columns existed are read unchanged.

The two families are scored over the same trials: every trial has exactly one dpos
case and exactly one ctx case, so ``case_0``..``case_7`` sum to 1 across a row and
``case_A``..``case_N`` sum to 1 as well (a whole row therefore sums to 2). The
denominator is the ``n_trials`` column.

Input is any set of per-trial alignment CSVs — the per-model files under
``<output-root>/<model>/alignment/`` (the default), or the combined table written by
``assess_module_alignment.py --combined-csv``. Both carry a ``model`` column, so
they are read the same way and a mixture is fine.

Examples
--------
    # every model found under the default output root
    python alignment_case_table.py

    # from the combined table instead, as percentages rounded to 1 decimal
    python alignment_case_table.py --percent --decimals 1 \
        --inputs /path/to/exp_seq_act_output/all_models_trial_alignment.csv

    # two named models, in that order
    python alignment_case_table.py \
        --model-names population_network_all_bn8_trainh0_fixedsir0.02_epochs300_lr0.002 \
                      population_network_all_bn8_trainh0_fixedsir0.05_epochs300_lr0.002
"""

import argparse
from pathlib import Path

import pandas as pd

from assess_module_alignment import (
    DEFAULT_OUTPUT_ROOT,
    DPOS_CASES,
    CTX_CASES,
    SI_R_COLUMNS,
)

DEFAULT_OUTPUT_NAME = 'alignment_case_proportions.csv'

# Column order: dpos cases first (numeric), then ctx cases (letters), matching the
# order the per-trial CSV lays them out in.
DPOS_COLUMNS = [f'case_{code}' for code in DPOS_CASES]
CTX_COLUMNS = [f'case_{code}' for code in CTX_CASES]


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
                + " — run assess_module_alignment.py for those models first")
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help='base directory holding <model>/alignment/ trees '
                        '(default: %(default)s)')
    p.add_argument('--inputs', nargs='+', type=Path, default=None, metavar='CSV',
                   help='per-trial alignment CSVs to read instead of searching '
                        '--output-root; a combined table works here too')
    p.add_argument('--model-names', nargs='+', default=None, metavar='NAME',
                   help='restrict to these models, and order the rows this way '
                        '(default: every model found)')
    p.add_argument('--output', type=Path, default=None,
                   help=f'where to write the table (default: <output-root>/{DEFAULT_OUTPUT_NAME})')
    p.add_argument('--percent', action='store_true',
                   help='write proportions as percentages instead of fractions')
    p.add_argument('--decimals', type=int, default=None,
                   help='round the proportions to this many decimals (default: full precision)')
    p.add_argument('--no-legend', action='store_true',
                   help='skip printing the case descriptions')
    return p.parse_args(argv)


def main(argv=None):
    cfg = parse_args(argv)

    paths = cfg.inputs if cfg.inputs else find_trial_tables(cfg.output_root, cfg.model_names)
    if not paths:
        raise FileNotFoundError(
            f"no *_trial_alignment.csv under {cfg.output_root}/*/alignment/ — "
            "run assess_module_alignment.py first")
    print(f"Reading {len(paths)} per-trial table(s)")
    df = load_cases(paths)

    table = build_case_table(df, cfg.model_names)
    check_family_sums(table)

    case_columns = DPOS_COLUMNS + CTX_COLUMNS
    if cfg.percent:
        table[case_columns] *= 100
    if cfg.decimals is not None:
        table[case_columns] = table[case_columns].round(cfg.decimals)

    out_file = cfg.output or (cfg.output_root / DEFAULT_OUTPUT_NAME)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_file, index=False)

    unit = '%' if cfg.percent else 'fraction'
    with pd.option_context('display.width', 200, 'display.max_columns', None,
                           'display.float_format', lambda v: f'{v:.3f}'):
        print(f"\nProportion of trials per case ({unit}), one row per model:")
        print(table.to_string(index=False))
    if not cfg.no_legend:
        print_legend()
    print(f"\nSaved: {out_file}")


if __name__ == '__main__':
    main()
