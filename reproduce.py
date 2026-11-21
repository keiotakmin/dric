"""Recompute the paper's summary tables from the condition-level records and check them against records/summary.csv.

  python reproduce.py

For each horizon (24, 48, 96): the main view is the 126 conditions of the seven datasets with the filled Weather file
(table == 'filled'); the standard Weather file (table == 'standard', dataset == 'weather') is reported separately.
Retained state is the largest value over the three backbones. mean_panels weights the 12 dataset-backbone panels
equally (the four ETT series form one dataset); median_126 is the median over the 126 conditions.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RECORDS = Path(__file__).resolve().parent / 'records'
ETT = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')
KEYS = ['table', 'dataset', 'backbone', 'seed', 'phase']


def main():
    rows = pd.read_csv(RECORDS / 'conditions_main.csv')
    rows['main'] = rows.table.eq('filled') | (rows.table.eq('standard') & rows.dataset.ne('weather'))
    rows['scope'] = np.where(rows.dataset.isin(ETT), 'ETT', rows.dataset)
    main = rows[rows.main]
    group = ['horizon', 'method']
    assert main.groupby(group).size().eq(126).all()
    panels = main.groupby(group + ['scope', 'backbone']).mse_reduction_pct.mean()
    out = pd.DataFrame({
        'mean_126': main.groupby(group).mse_reduction_pct.mean(),
        'mean_panels': panels.groupby(group).mean(),
        'median_126': main.groupby(group).mse_reduction_pct.median(),
        'mae_mean_126': main.groupby(group).mae_reduction_pct.mean(),
        **{s: main[main.scope.eq(s)].groupby(group).mse_reduction_pct.mean()
           for s in ('ETT', 'weather', 'electricity', 'traffic')},
        'weather_standard': rows[rows.table.eq('standard') & rows.dataset.eq('weather')]
        .groupby(group).mse_reduction_pct.mean(),
        'bytes_ETT': main[main.scope.eq('ETT')].groupby(group).state_bytes.max(),
        'bytes_traffic': main[main.scope.eq('traffic')].groupby(group).state_bytes.max(),
    })
    expected = pd.read_csv(RECORDS / 'summary.csv').set_index(group)
    out = out.loc[expected.index, expected.columns]
    np.testing.assert_allclose(out.to_numpy(float), expected.to_numpy(float), rtol=1e-8)
    for h, part in out.groupby(level='horizon'):
        print(f'\nH = {h}')
        print(part.droplevel('horizon').round(2).to_string())
        wide = main[main.horizon.eq(h)].pivot_table(index=KEYS, columns='method', values='mse')
        for ref, label in (('elf', 'ELF (shared blend)'), ('elf_own', 'ELF (own weighting)')):
            for m in ('local8', 'local4', 'global'):
                print(f'{m}: lower MSE than {label} in {int((wide[m] < wide[ref]).sum())} of 126 conditions')
    print('\nsummary.csv reproduced')


if __name__ == '__main__':
    main()
