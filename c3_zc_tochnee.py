"""Уточнение 95-го процентиля погрешности перехода через ноль в серии С3 (10 000 испытаний).

Повод: в основной серии С3 (2000 испытаний) P95 = 5,68° с бутстреп-интервалом
[5,53; 5,93], а независимая перепроверка (1000 испытаний) дала 5,95 [5,74; 6,20].
Генераторы случайных чисел те же, что в series.py (SEED, серия 3, номер испытания),
поэтому первые 2000 испытаний повторяют основную серию; считаются только
оцениватели по переходу через ноль. Результат: results/c3_zc_10000.csv.
"""
import csv
import os
from multiprocessing import Pool

import numpy as np

import config as C
from core import run_record
from device import make_device, calibrate
from estimators import ZCEstimator
from series import rng_for, uphase
from signals import Source, random_harmonics

HERE = os.path.dirname(os.path.abspath(__file__))
N_TRIALS = 10000


def one(i):
    rng = rng_for(3, i)
    ests = [ZCEstimator(), ZCEstimator(lpf=True)]
    # те же вызовы генератора, что в series._c3_group: калибровка, затем источники
    dev = calibrate(make_device(rng), ests, rng)
    s1 = Source(C.C3_FREQ, uphase(rng), harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT))
    s2 = Source(C.C3_FREQ, uphase(rng), harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT))
    out = run_record(dev, ests, s1, s2, rng, common=False)
    return i, {k: float(np.max(np.abs(v[1]))) for k, v in out.items()}


def main():
    with Pool(max(1, min(os.cpu_count() or 1, 4))) as pool:
        res = pool.map(one, range(N_TRIALS), chunksize=50)
    res.sort()
    res_dir = os.path.join(HERE, 'results' + C.OUT_SUFFIX)
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, 'c3_zc_10000.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['trial', 'zc_max_abs', 'zc_lpf_max_abs'])
        for i, d in res:
            w.writerow([i, f"{d['zc']:.6g}", f"{d['zc_lpf']:.6g}"])
    rng = np.random.default_rng(1)
    for key in ('zc', 'zc_lpf'):
        v = np.array([d[key] for _, d in res])
        bs = [np.percentile(rng.choice(v, len(v)), 95) for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        first = np.percentile(v[:2000], 95)
        print(f'{key}: P95 по {len(v)} = {np.percentile(v, 95):.3f} [{lo:.3f}; {hi:.3f}], '
              f'по первым 2000 = {first:.3f}, наибольшая = {v.max():.2f}')


if __name__ == '__main__':
    main()
