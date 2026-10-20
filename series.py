"""Серии моделирования С1–С6.

Каждый прогон идёт в три шага:
  1) калибровка межтрактовой задержки на общем чистом сигнале 50 Гц (поправка хранится как время);
  2) проверка общим сигналом в условиях серии (на оба модуля подаётся сигнал резерва);
  3) рабочий режим: основной источник → модуль 1, резерв → модуль 2.
Зёрна случайных чисел детерминированы: генератор каждого испытания строится из
(SEED, номер серии, номер испытания), поэтому результат не зависит от числа
процессов. Результаты пишутся в results/*.csv.
"""
import csv
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import config as C
from core import run_record, stats
from device import make_device, calibrate, Device
from estimators import all_estimators, AdaptiveEstimator, DFTEstimator, LABELS
from signals import Source, random_harmonics, random_unbalance, TWO_PI

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results' + C.OUT_SUFFIX)
HIST_EDGES = np.logspace(-6.0, np.log10(360.0), 2001)      # бины |ε|, градусы
NPROC = max(1, min(os.cpu_count() or 1, 4))


def rng_for(series, *idx):
    return np.random.default_rng([C.SEED, series, *[int(i) for i in idx]])


def uphase(rng):
    return float(rng.uniform(0.0, TWO_PI))


# ============================================================================ накопление статистики
class Pool_:
    """Объединённая статистика |ε| по испытаниям: гистограмма, суммы, максимум."""

    def __init__(self):
        self.hist = np.zeros(len(HIST_EDGES) - 1, dtype=np.int64)
        self.n = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.max_abs = 0.0
        self.trial_max = []

    def add(self, eps):
        e = np.asarray(eps, dtype=float)
        if e.size == 0:
            return
        a = np.clip(np.abs(e), HIST_EDGES[0] * 1.0001, HIST_EDGES[-1] * 0.9999)
        self.hist += np.histogram(a, HIST_EDGES)[0]
        self.n += e.size
        self.s1 += float(np.sum(e))
        self.s2 += float(np.sum(e ** 2))
        m = float(np.max(np.abs(e)))
        self.max_abs = max(self.max_abs, m)
        self.trial_max.append(m)

    def merge(self, other):
        self.hist += other.hist
        self.n += other.n
        self.s1 += other.s1
        self.s2 += other.s2
        self.max_abs = max(self.max_abs, other.max_abs)
        self.trial_max += other.trial_max

    def percentile(self, q):
        if self.n == 0:
            return float('nan')
        cum = np.cumsum(self.hist)
        target = q / 100.0 * self.n
        i = int(np.searchsorted(cum, target))
        lo, hi = HIST_EDGES[i], HIST_EDGES[i + 1]
        prev = cum[i - 1] if i > 0 else 0
        frac = (target - prev) / max(self.hist[i], 1)
        return float(lo * (hi / lo) ** frac)

    def summary(self):
        if self.n == 0:
            return dict(n=0, mean=np.nan, std=np.nan, max_abs=np.nan, p95_abs=np.nan,
                        p95_trial_max=np.nan, n_trials=0)
        mean = self.s1 / self.n
        std = float(np.sqrt(max(self.s2 / self.n - mean ** 2, 0.0)))
        return dict(n=self.n, mean=mean, std=std, max_abs=self.max_abs,
                    p95_abs=self.percentile(95.0),
                    p95_trial_max=float(np.percentile(self.trial_max, 95.0)),
                    n_trials=len(self.trial_max))


def merge_pools(dst, src):
    for key, p in src.items():
        if key not in dst:
            dst[key] = Pool_()
        dst[key].merge(p)


def add_out(pools, mode, out):
    for name, (_, eps, _) in out.items():
        key = (mode, name)
        if key not in pools:
            pools[key] = Pool_()
        pools[key].add(eps)


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def summary_rows(series, pools):
    rows = []
    for (mode, name), p in sorted(pools.items()):
        s = p.summary()
        rows.append(dict(series=series, mode=mode, est=name, label=LABELS.get(name, name), **s))
    return rows


def run_parallel(func, items, chunks=None):
    chunks = chunks or max(1, len(items) // (NPROC * 4))
    groups = [items[i:i + chunks] for i in range(0, len(items), chunks)]
    if NPROC == 1:
        return [func(g) for g in groups]
    with Pool(NPROC) as pool:
        return pool.map(func, groups)


# ============================================================================ С2. Скольжение частот
def _c2_group(items):
    rows = []
    for df, rep in items:
        rng = rng_for(2, int(round(df * 1000)) + 100000, rep)
        ests = all_estimators()
        dev = calibrate(make_device(rng), ests, rng)
        s1 = Source(C.F0, uphase(rng))
        s2 = Source(C.F0 + df, uphase(rng))
        for mode in ('working', 'common'):
            out = run_record(dev, ests, s1, s2, rng, common=(mode == 'common'))
            for name, (t, eps, age) in out.items():
                rows.append(dict(series='C2', df=df, rep=rep, mode=mode, est=name,
                                 **stats(eps), age_ms=float(np.mean(age) * 1e3) if len(age) else np.nan))
    return rows


def run_c2():
    dfs = sorted({s * a for a in C.C2_ABS_DF for s in (-1.0, 1.0)})
    items = [(df, rep) for df in dfs for rep in range(C.C2_REPEATS)]
    rows = [r for g in run_parallel(_c2_group, items, chunks=4) for r in g]
    write_csv(os.path.join(RES, 'c2_points.csv'), rows)
    summ = []
    keys = sorted({(r['df'], r['mode'], r['est']) for r in rows})
    for df, mode, est in keys:
        rr = [r for r in rows if r['df'] == df and r['mode'] == mode and r['est'] == est]
        summ.append(dict(series='C2', df=df, mode=mode, est=est, label=LABELS[est],
                         mean=float(np.mean([r['mean'] for r in rr])),
                         std=float(np.mean([r['std'] for r in rr])),
                         max_abs=float(np.max([r['max_abs'] for r in rr])),
                         p95_abs=float(np.max([r['p95_abs'] for r in rr])),
                         age_ms=float(np.mean([r['age_ms'] for r in rr]))))
    write_csv(os.path.join(RES, 'c2_summary.csv'), summ)
    return summ


# ============================================================================ С3. Гармоники
def _c3_group(idx):
    rows, pools = [], {}
    for i in idx:
        rng = rng_for(3, i)
        ests = all_estimators()
        dev = calibrate(make_device(rng), ests, rng)
        s1 = Source(C.C3_FREQ, uphase(rng), harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT))
        s2 = Source(C.C3_FREQ, uphase(rng), harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT))
        for mode in ('working', 'common'):
            out = run_record(dev, ests, s1, s2, rng, common=(mode == 'common'))
            add_out(pools, mode, out)
            for name, (t, eps, age) in out.items():
                rows.append(dict(series='C3', trial=i, mode=mode, est=name, thd1=s1.thd(), thd2=s2.thd(),
                                 **stats(eps)))
    return rows, pools


def run_c3():
    res = run_parallel(_c3_group, list(range(C.C3_TRIALS)))
    rows = [r for g in res for r in g[0]]
    pools = {}
    for _, p in res:
        merge_pools(pools, p)
    write_csv(os.path.join(RES, 'c3_trials.csv'), rows)
    summ = summary_rows('C3', pools)
    write_csv(os.path.join(RES, 'c3_summary.csv'), summ)
    return summ


# ============================================================================ С4. Несимметрия
def unbalance_deviation(n, k2_max, k0_max, rng):
    """Распределение отклонения угла между фазами (и линейными напряжениями) двух
    источников от угла между их прямыми последовательностями при случайной
    несимметрии каждого источника: K2U ~ U(0, k2_max), K0U ~ U(0, k0_max)."""
    def ratios():
        k2 = rng.uniform(0, k2_max, n); p2 = rng.uniform(0, TWO_PI, n)
        k0 = rng.uniform(0, k0_max, n); p0 = rng.uniform(0, TWO_PI, n)
        r = []
        for p in range(3):          # фаза p относительно своей прямой последовательности
            r.append(1 + k2 * np.exp(1j * (p2 + p * 2 * TWO_PI / 3)) + k0 * np.exp(1j * (p0 + p * TWO_PI / 3)))
        rab = 1 + k2 * np.exp(1j * (p2 - np.pi / 3))   # линейное U_AB относительно прямой посл.
        return r, rab
    r1, ab1 = ratios()
    r2, ab2 = ratios()
    dA = np.degrees(np.angle(r1[0] / r2[0]))
    dmax = np.max(np.abs([np.degrees(np.angle(r1[p] / r2[p])) for p in range(3)]), axis=0)
    dAB = np.degrees(np.angle(ab1 / ab2))
    return np.abs(dA), dmax, np.abs(dAB)


def _c4_group(idx):
    rows, pools = [], {}
    for i in idx:
        rng = rng_for(4, i)
        ests = all_estimators()
        dev = calibrate(make_device(rng), ests, rng)
        s1 = Source(C.F0, uphase(rng), **random_unbalance(rng, C.K2U_LIMIT, C.K0U_LIMIT))
        df = float(rng.uniform(-C.C4_DF_MAX, C.C4_DF_MAX))
        s2 = Source(C.F0 + df, uphase(rng), **random_unbalance(rng, C.K2U_LIMIT, C.K0U_LIMIT))
        # отклонение истинного угла фаз A от угла прямых последовательностей
        devA = float(np.degrees(np.angle(s1.phase_phasors()[0] / np.exp(1j * s1.phi))
                                - np.angle(s2.phase_phasors()[0] / np.exp(1j * s2.phi))))
        for mode in ('working', 'common'):
            out = run_record(dev, ests, s1, s2, rng, common=(mode == 'common'))
            add_out(pools, mode, out)
            for name, (t, eps, age) in out.items():
                rows.append(dict(series='C4', trial=i, mode=mode, est=name, df=df,
                                 k2_1=s1.k2, k0_1=s1.k0, k2_2=s2.k2, k0_2=s2.k0, devA=devA, **stats(eps)))
    return rows, pools


def run_c4():
    rng = rng_for(4, 999999)
    q = np.linspace(0.0, 1.0, 1001)
    dev_rows = []
    curves = {}
    for tag, k2m, k0m in (('95', C.K2U_LIMIT, C.K0U_LIMIT), ('100', 2 * C.K2U_LIMIT, 2 * C.K0U_LIMIT)):
        dA, dmax, dAB = unbalance_deviation(200000, k2m, k0m, rng)
        for name, arr in (('phaseA', dA), ('worst_phase', dmax), ('line_AB', dAB)):
            dev_rows.append(dict(limits=tag, quantity=name, mean=float(np.mean(arr)),
                                 p50=float(np.percentile(arr, 50)), p95=float(np.percentile(arr, 95)),
                                 p99=float(np.percentile(arr, 99)), max=float(np.max(arr))))
            curves[(tag, name)] = np.quantile(arr, q)
    write_csv(os.path.join(RES, 'c4_unbalance_deviation.csv'), dev_rows)
    with open(os.path.join(RES, 'c4_unbalance_quantiles.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        keys = sorted(curves)
        w.writerow(['quantile'] + [f"{k[1]}_{k[0]}" for k in keys])
        for j, qq in enumerate(q):
            w.writerow([f"{qq:.3f}"] + [f"{curves[k][j]:.6g}" for k in keys])
    res = run_parallel(_c4_group, list(range(C.C4_TRIALS)))
    rows = [r for g in res for r in g[0]]
    pools = {}
    for _, p in res:
        merge_pools(pools, p)
    write_csv(os.path.join(RES, 'c4_trials.csv'), rows)
    summ = summary_rows('C4', pools)
    write_csv(os.path.join(RES, 'c4_summary.csv'), summ)
    return summ, dev_rows


# ============================================================================ С5. Общая задержка и разность частот
def _c5_group(items):
    rows = []
    for dbar_ms, df in items:
        rng = rng_for(5, int(round(dbar_ms * 100)), int(round(df * 10)))
        ests = [AdaptiveEstimator('P', 'mid'), DFTEstimator(1, 'mid', 'P')]
        dev = calibrate(make_device(rng, dbar=dbar_ms * 1e-3, dbar_model=0.0), ests, rng)
        s1 = Source(C.F0, uphase(rng))
        s2 = Source(C.F0 + df, uphase(rng))
        for mode in ('working', 'common'):
            out = run_record(dev, ests, s1, s2, rng, common=(mode == 'common'))
            for name, (t, eps, age) in out.items():
                rows.append(dict(series='C5', dbar_ms=dbar_ms, df=df, mode=mode, est=name,
                                 analytic=360.0 * df * dbar_ms * 1e-3 if mode == 'working' else 0.0,
                                 **stats(eps)))
    return rows


def run_c5():
    items = [(d, f) for d in C.C5_DBAR_MS for f in C.C5_DF]
    rows = [r for g in run_parallel(_c5_group, items) for r in g]
    write_csv(os.path.join(RES, 'c5_map.csv'), rows)
    # допустимая общая задержка для заданной погрешности
    allow = []
    for eps_max in (0.1, 0.2, 0.5, 1.0):
        for df in (0.2, 0.5, 1.0, 2.0):
            allow.append(dict(eps_max_deg=eps_max, df_hz=df, dbar_max_ms=eps_max / (360.0 * df) * 1e3))
    write_csv(os.path.join(RES, 'c5_allowed_delay.csv'), allow)
    return rows


# ============================================================================ С6. Итоговая
def _c6_group(idx):
    rows, pools = [], {}
    for i in idx:
        rng = rng_for(6, i)
        ests = all_estimators()
        dev = calibrate(make_device(rng), ests, rng)
        f1 = C.F0 + float(rng.uniform(-C.DF_GRID_95, C.DF_GRID_95))
        f2 = C.F0 + float(rng.uniform(-C.DF_ISO_95, C.DF_ISO_95))
        s1 = Source(f1, uphase(rng), harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT),
                    **random_unbalance(rng, C.K2U_LIMIT, C.K0U_LIMIT))
        s2 = Source(f2, uphase(rng), harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT),
                    **random_unbalance(rng, C.K2U_LIMIT, C.K0U_LIMIT))
        f2s = f1 + float(rng.uniform(-C.C6_SYNC_DF, C.C6_SYNC_DF))
        s2s = replace(s2, f=f2s)
        devA = float(np.degrees(np.angle(s1.phase_phasors()[0] / np.exp(1j * s1.phi))
                                - np.angle(s2.phase_phasors()[0] / np.exp(1j * s2.phi))))
        cases = (('common', s1, s2, True, dev, ests),
                 ('working', s1, s2, False, dev, ests),
                 ('sync', s1, s2s, False, dev, ests))
        # прибор с неучтённой общей задержкой d̄ (поправка калибровки та же: d̄ её не меняет)
        dev_d = Device(paths=tuple(replace(p, delay=p.delay + C.C6_DBAR_EXTRA) for p in dev.paths),
                       dbar=C.C6_DBAR_EXTRA, dbar_model=0.0, cal=dict(dev.cal))
        good = [e for e in ests if e.name in ('adapt_mid_P', 'dft1_mid_P')]
        cases += (('working_dbar', s1, s2, False, dev_d, good),
                  ('sync_dbar', s1, s2s, False, dev_d, good))
        for mode, a, b, common, d, es in cases:
            out = run_record(d, es, a, b, rng, common=common)
            add_out(pools, mode, out)
            for name, (t, eps, age) in out.items():
                rows.append(dict(series='C6', trial=i, mode=mode, est=name, f1=f1,
                                 f2=(f2s if mode.startswith('sync') else f2),
                                 thd1=s1.thd(), thd2=s2.thd(), k2_1=s1.k2, k0_1=s1.k0,
                                 k2_2=s2.k2, k0_2=s2.k0, devA=devA, **stats(eps),
                                 age_ms=float(np.mean(age) * 1e3) if len(age) else np.nan))
    return rows, pools


def run_c6():
    res = run_parallel(_c6_group, list(range(C.C6_TRIALS)))
    rows = [r for g in res for r in g[0]]
    pools = {}
    for _, p in res:
        merge_pools(pools, p)
    write_csv(os.path.join(RES, 'c6_trials.csv'), rows)
    summ = summary_rows('C6', pools)
    write_csv(os.path.join(RES, 'c6_summary.csv'), summ)
    return summ


# ============================================================================ С1. Общий сигнал
def run_c1():
    """Сводка по проверке общим сигналом во всех условиях С2–С6 (из записанных CSV)."""
    rows = []
    def read(path):
        with open(path, encoding='utf-8') as fh:
            return list(csv.DictReader(fh))
    src = {'C2': read(os.path.join(RES, 'c2_points.csv')),
           'C3': read(os.path.join(RES, 'c3_trials.csv')),
           'C4': read(os.path.join(RES, 'c4_trials.csv')),
           'C5': read(os.path.join(RES, 'c5_map.csv')),
           'C6': read(os.path.join(RES, 'c6_trials.csv'))}
    names = [e.name for e in all_estimators()]
    for name in names:
        row = dict(est=name, label=LABELS[name])
        overall = 0.0
        for ser, data in src.items():
            vals = [float(r['max_abs']) for r in data if r['mode'] == 'common' and r['est'] == name]
            vals_p95 = [float(r['p95_abs']) for r in data if r['mode'] == 'common' and r['est'] == name]
            row[f'{ser}_max'] = max(vals) if vals else np.nan
            row[f'{ser}_p95max'] = max(vals_p95) if vals_p95 else np.nan
            if vals:
                overall = max(overall, max(vals))
        row['all_max'] = overall
        rows.append(row)
    write_csv(os.path.join(RES, 'c1_common_signal.csv'), rows)
    return rows


def main(which=('C2', 'C3', 'C4', 'C5', 'C6', 'C1')):
    os.makedirs(RES, exist_ok=True)
    tpath = os.path.join(RES, 'timing.json')
    timing = {}
    if os.path.exists(tpath):  # при запуске части серий сохраняем время остальных
        with open(tpath, encoding='utf-8') as fh:
            timing = json.load(fh)
    for ser in which:
        t0 = time.time()
        {'C1': run_c1, 'C2': run_c2, 'C3': run_c3, 'C4': run_c4, 'C5': run_c5, 'C6': run_c6}[ser]()
        timing[ser] = round(time.time() - t0, 1)
        print(f'{ser}: {timing[ser]} с', flush=True)
        with open(tpath, 'w', encoding='utf-8') as fh:
            json.dump(timing, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    import sys
    main(tuple(sys.argv[1:]) or ('C2', 'C3', 'C4', 'C5', 'C6', 'C1'))
