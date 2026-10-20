"""Дополнение к С6: неучтённая общая задержка d̄ = 1 мс на общем сигнале.

В series.py для прибора с неучтённой d̄ считались только режимы «разные сигналы»
и «перед включением» (working_dbar, sync_dbar). Здесь те же 2000 испытаний С6
(те же зёрна и тот же порядок расхода генератора) повторены для двух вариантов
с неучтённой задержкой, и после них добавлен режим общего сигнала common_dbar.
Совпадение working_dbar и sync_dbar с results/c6_summary.csv подтверждает, что
испытания те же. Результат: results/c6_dbar_common.csv.
"""
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import config as C
from core import run_record
from device import make_device, calibrate, Device
from estimators import AdaptiveEstimator, DFTEstimator
from series import rng_for, uphase, Pool_, write_csv, RES
from signals import Source, random_harmonics, random_unbalance


def _group(idx):
    pools = {}
    for i in idx:
        rng = rng_for(6, i)
        ests = [DFTEstimator(C.DFT_PERIODS_SHORT, 'mid', 'P'), AdaptiveEstimator('P', 'mid')]
        dev = calibrate(make_device(rng), ests, rng)       # расход генератора от набора оценивателей не зависит
        f1 = C.F0 + float(rng.uniform(-C.DF_GRID_95, C.DF_GRID_95))
        f2 = C.F0 + float(rng.uniform(-C.DF_ISO_95, C.DF_ISO_95))
        s1 = Source(f1, uphase(rng), harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT),
                    **random_unbalance(rng, C.K2U_LIMIT, C.K0U_LIMIT))
        s2 = Source(f2, uphase(rng), harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT),
                    **random_unbalance(rng, C.K2U_LIMIT, C.K0U_LIMIT))
        f2s = f1 + float(rng.uniform(-C.C6_SYNC_DF, C.C6_SYNC_DF))
        s2s = replace(s2, f=f2s)
        dev_d = Device(paths=tuple(replace(p, delay=p.delay + C.C6_DBAR_EXTRA) for p in dev.paths),
                       dbar=C.C6_DBAR_EXTRA, dbar_model=0.0, cal=dict(dev.cal))
        cases = (('common', s1, s2, True, dev), ('working', s1, s2, False, dev),
                 ('sync', s1, s2s, False, dev), ('working_dbar', s1, s2, False, dev_d),
                 ('sync_dbar', s1, s2s, False, dev_d), ('common_dbar', s1, s2, True, dev_d))
        for mode, a, b, common, d in cases:
            out = run_record(d, ests, a, b, rng, common=common)
            for name, (t, eps, age) in out.items():
                pools.setdefault((mode, name), Pool_()).add(eps)
    return pools


def main(n_trials=C.C6_TRIALS):
    items = list(range(n_trials))
    nproc = max(1, min(os.cpu_count() or 1, 20))
    with Pool(nproc) as p:
        res = p.map(_group, [items[k::nproc] for k in range(nproc)])
    pools = {}
    for pl in res:
        for k, v in pl.items():
            pools.setdefault(k, Pool_()).merge(v)
    rows = []
    for (mode, name), pl in sorted(pools.items()):
        s = pl.summary()
        rows.append(dict(mode=mode, est=name, n_trials=s['n_trials'], q95=s['p95_abs'],
                         m95=s['p95_trial_max'], max_abs=s['max_abs']))
    write_csv(os.path.join(RES, 'c6_dbar_common.csv'), rows)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else C.C6_TRIALS)
