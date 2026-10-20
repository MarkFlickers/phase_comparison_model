"""Проверка причинности оценивателя с подстройкой частоты.

В AdaptiveEstimator частота оценивается один раз на запись около середины
интервала [EVAL_T0, EVAL_T1], то есть по данным до ≈ 0,72 с, и применяется ко
всем результатам начиная с 0,30 с. Здесь реализован причинный вариант:
частота для каждого результата оценивается только по отсчётам, полученным
к моменту его выдачи (окна МНК для оценки частоты центрированы в t0 − ADAPT_TF
и t0). Момент выдачи t_out — поступление самого позднего фактически
использованного отсчёта по обоим каналам и всем итерациям. В первой версии
проверки было t_out = t0 + T_ок/2 по частоте первого
канала, что не охватывало окно второго канала и промежуточные итерации;
её результаты сохранены в results/causal_check_v1.*.

1) Тесты причинности: (а) сигнал модуля 2 заменяется после 0,35 с;
   (б) граничный: меняются все отсчёты, поступившие строго после объявленного
   t_out, поочерёдно в каждом канале (results/causal_boundary.csv).
2) Повтор С6 для адаптивных вариантов на тех же испытаниях (те же зёрна,
   тот же расход генератора), исходный и причинный варианты рядом.
Результат: results/causal_check.csv и results/causal_check.json.
"""
import csv
import json
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import config as C
from core import run_record
from device import make_device, calibrate, Device, acquire, apply_time_model
from estimators import AdaptiveEstimator, select_quantity
from series import rng_for, uphase, Pool_
from signals import Source, random_harmonics, random_unbalance, wrap_rad, TWO_PI

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results' + C.OUT_SUFFIX)


class CausalAdaptiveEstimator(AdaptiveEstimator):
    """То же МНК, но частота для каждого результата — только по уже полученным отсчётам.

    Момент выдачи t_out — момент поступления самого позднего отсчёта, фактически
    использованного для этого результата: по обоим каналам и по всем окнам,
    включая промежуточные итерации оценки частоты. Берутся исходные метки
    поступления (Record.T_avail), а не метки, исправленные моделью времени.
    Время вычислений и передачи результата не учитывается.
    """

    def __init__(self, quantity='P', attribution='mid'):
        super().__init__(quantity, attribution)
        self.name = f"adaptc_{attribution}_{quantity}"
        self._used = -np.inf

    def ls_phasors(self, rec, t0, f):
        i1 = np.searchsorted(rec.T, t0 + self.periods / (2.0 * f))    # как в AdaptiveEstimator
        avail = rec.T if rec.T_avail is None else rec.T_avail
        self._used = max(self._used, float(avail[i1 - 1]))
        return super().ls_phasors(rec, t0, f)

    def estimate(self, r1, r2, t_lo, t_hi):
        t0 = np.linspace(t_lo, t_hi, self.n_points)
        dphi = np.empty_like(t0)
        t_out = np.empty_like(t0)
        for k, t in enumerate(t0):
            self._used = -np.inf
            tm = t - C.ADAPT_TF / 2.0            # окна частоты в t − ADAPT_TF и t
            f1, f2 = self.frequency(r1, tm), self.frequency(r2, tm)
            z1 = select_quantity(self.ls_phasors(r1, t, f1), self.quantity)
            z2 = select_quantity(self.ls_phasors(r2, t, f2), self.quantity)
            d = np.angle(z1 * np.conj(z2))
            t_out[k] = self._used
            if self.attribution == 'now':
                d = wrap_rad(d + TWO_PI * (f1 - f2) * (t_out[k] - t))
            dphi[k] = d
        if self.attribution == 'now':
            return t_out, dphi, np.zeros_like(t0)
        return t0, dphi, t_out - t0


def _perturbed(rec, t_cut, delta=0.1):
    """Копия записи, в которой ко всем отсчётам, поступившим строго после t_cut, добавлено delta."""
    avail = rec.T if rec.T_avail is None else rec.T_avail
    x = rec.x.copy()
    x[:, avail > t_cut] += delta
    return replace(rec, x=x)


# частоты (f1, f2): перестановка, ниже и выше 50 Гц, равные неноминальные
CAUSAL_FREQS = ((50.0, 49.0), (49.0, 50.0), (50.2, 49.0), (49.8, 51.0), (51.0, 49.0),
                (51.0, 51.0), (49.0, 49.0), (50.0, 50.0))


def causality_test():
    """1) Исходный тест: сигнал модуля 2 заменён после 0,35 с (выдача первого результата ≈ 0,32 с).
    2) Граничный тест: для результата меняются все отсчёты, поступившие строго после
       объявленного t_out, поочерёдно в каждом канале. Случайные экземпляры прибора
       (смещения сеток, задержки, поправка модуля 2 в модели времени), частоты CAUSAL_FREQS,
       первый и средний результаты записи, оба варианта отнесения.
    """
    rng = np.random.default_rng(1)
    dev = make_device(rng)
    s1 = Source(50.0, 0.3)
    s2 = Source(50.6, 1.1)
    s2b = Source(49.4, 2.0)          # «другое продолжение» после 0,35 с
    r1 = acquire(s1, dev.paths[0], C.REC_DURATION, rng)
    r2 = acquire(s2, dev.paths[1], C.REC_DURATION, np.random.default_rng(2))
    r2b = acquire(s2b, dev.paths[1], C.REC_DURATION, np.random.default_rng(2))
    cut = 0.35
    x = r2.x.copy()
    m = r2.T > cut
    x[:, m] = r2b.x[:, m]
    r2alt = replace(r2, x=x)
    out = {'replace_after_0.35s': {}}
    for est in (AdaptiveEstimator('P', 'now'), CausalAdaptiveEstimator('P', 'now'),
                AdaptiveEstimator('P', 'mid'), CausalAdaptiveEstimator('P', 'mid')):
        ta, da, _ = est.estimate(r1, r2, C.EVAL_T0, C.EVAL_T1)
        tb, db, _ = est.estimate(r1, r2alt, C.EVAL_T0, C.EVAL_T1)
        out['replace_after_0.35s'][est.name] = dict(
            t_first=float(ta[0]), change_first_deg=float(np.degrees(wrap_rad(db[0] - da[0]))))

    cases = []
    rng = np.random.default_rng(7)
    for f1, f2 in CAUSAL_FREQS:
        for rep in range(3):
            dev = make_device(rng)
            dev.cal = {'c': float(rng.uniform(-40e-6, 40e-6))}   # поправка модуля 2 в модели времени
            a = Source(f1, float(rng.uniform(0, TWO_PI)),
                       harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT))
            b = Source(f2, float(rng.uniform(0, TWO_PI)),
                       harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT))
            r1 = acquire(a, dev.paths[0], C.REC_DURATION, rng)
            r2 = acquire(b, dev.paths[1], C.REC_DURATION, rng)
            q1, q2 = apply_time_model(dev, 'c', r1, r2)
            for t_lo in (C.EVAL_T0, 0.5 * (C.EVAL_T0 + C.EVAL_T1)):
                for att in ('now', 'mid'):
                    est = CausalAdaptiveEstimator('P', att)
                    est.n_points = 1
                    t_attr, d0, age = est.estimate(q1, q2, t_lo, t_lo)
                    t_out = float(t_attr[0] + age[0])
                    row = dict(f1=f1, f2=f2, rep=rep, t_lo=t_lo, att=att, t_out=t_out,
                               nominal_t_out=t_lo + C.ADAPT_PERIODS / (2.0 * f1))
                    for ch in (0, 1):
                        p1 = _perturbed(q1, t_out) if ch == 0 else q1
                        p2 = _perturbed(q2, t_out) if ch == 1 else q2
                        n_changed = int(np.sum((r1.T_avail if ch == 0 else r2.T_avail) > t_out))
                        t_b, d_b, age_b = est.estimate(p1, p2, t_lo, t_lo)
                        row[f'n_changed_ch{ch + 1}'] = n_changed
                        row[f'change_ch{ch + 1}_deg'] = float(np.degrees(wrap_rad(d_b[0] - d0[0])))
                        row[f'tout_change_ch{ch + 1}_s'] = float(t_b[0] + age_b[0] - t_out)
                    cases.append(row)
    lat = [r['t_out'] - r['t_lo'] for r in cases]
    out['boundary'] = dict(
        n_cases=len(cases),
        max_abs_change_deg=max(max(abs(r['change_ch1_deg']), abs(r['change_ch2_deg'])) for r in cases),
        max_abs_tout_change_s=max(max(abs(r['tout_change_ch1_s']), abs(r['tout_change_ch2_s']))
                                  for r in cases),
        latency_min_s=min(lat), latency_max_s=max(lat),
        extra_vs_f1_window_max_s=max(r['t_out'] - r['nominal_t_out'] for r in cases))
    with open(os.path.join(RES, 'causal_boundary.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(cases[0]))
        w.writeheader()
        for r in cases:
            w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})
    return out


def _group(idx):
    pools = {}
    rows = []
    for i in idx:
        rng = rng_for(6, i)
        ests = [AdaptiveEstimator('A', 'mid'), AdaptiveEstimator('P', 'mid'), AdaptiveEstimator('P', 'now'),
                CausalAdaptiveEstimator('A', 'mid'), CausalAdaptiveEstimator('P', 'mid'),
                CausalAdaptiveEstimator('P', 'now')]
        dev = calibrate(make_device(rng), ests, rng)
        f1 = C.F0 + float(rng.uniform(-C.DF_GRID_95, C.DF_GRID_95))
        f2 = C.F0 + float(rng.uniform(-C.DF_ISO_95, C.DF_ISO_95))
        s1 = Source(f1, uphase(rng), harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT),
                    **random_unbalance(rng, C.K2U_LIMIT, C.K0U_LIMIT))
        s2 = Source(f2, uphase(rng), harmonics=random_harmonics(rng, C.HARM_LIMITS, C.THD_LIMIT),
                    **random_unbalance(rng, C.K2U_LIMIT, C.K0U_LIMIT))
        f2s = f1 + float(rng.uniform(-C.C6_SYNC_DF, C.C6_SYNC_DF))
        s2s = replace(s2, f=f2s)
        for mode, a, b, common in (('common', s1, s2, True), ('working', s1, s2, False),
                                   ('sync', s1, s2s, False)):
            out = run_record(dev, ests, a, b, rng, common=common)
            for name, (t, eps, age) in out.items():
                key = (mode, name)
                pools.setdefault(key, Pool_()).add(eps)
                rows.append(dict(trial=i, mode=mode, est=name, max_abs=float(np.max(np.abs(eps)))))
    return pools, rows


def main(n_trials):
    os.makedirs(RES, exist_ok=True)
    caus = causality_test()
    items = list(range(n_trials))
    nproc = max(1, min(os.cpu_count() or 1, 20))
    chunks = [items[k::nproc] for k in range(nproc)]
    with Pool(nproc) as p:
        res = p.map(_group, chunks)
    pools = {}
    for pl, _ in res:
        for k, v in pl.items():
            pools.setdefault(k, Pool_()).merge(v)
    rows = []
    for (mode, name), pl in sorted(pools.items()):
        s = pl.summary()
        rows.append(dict(mode=mode, est=name, n_trials=s['n_trials'], q95=s['p95_abs'],
                         m95=s['p95_trial_max'], max_abs=s['max_abs']))
    with open(os.path.join(RES, 'causal_check.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(RES, 'causal_check.json'), 'w', encoding='utf-8') as fh:
        json.dump(dict(causality_test=caus, n_trials=n_trials), fh, ensure_ascii=False, indent=1)
    print(json.dumps(caus, ensure_ascii=False, indent=1))
    for r in rows:
        print(r)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
