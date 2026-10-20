"""Прогон одной записи: истинная разность фаз, погрешность, статистика."""
import numpy as np

import config as C
from device import acquire, apply_time_model
from signals import wrap_deg, wrap_rad


def true_dphi(src1, src2, t, quantity):
    """Истинная разность фаз основных гармоник Δφ = θ_1 − θ_2 в моменты t, рад."""
    if quantity == 'A':
        return wrap_rad(src1.theta_phase(t, 0) - src2.theta_phase(t, 0))
    return wrap_rad(src1.theta_pos(t) - src2.theta_pos(t))


def run_record(dev, estimators, src1, src2, rng, common=False,
               duration=C.REC_DURATION, t_lo=C.EVAL_T0, t_hi=C.EVAL_T1):
    """Одна запись и все оцениватели.

    common=False — рабочий режим: src1 → модуль 1, src2 → модуль 2.
    common=True  — проверка общим сигналом: src2 подаётся на оба модуля,
                   истинная разность фаз равна нулю.
    Возвращает словарь: имя оценивателя → (t_attr, ε в градусах, возраст).
    """
    a_src, b_src = (src2, src2) if common else (src1, src2)
    r1 = acquire(a_src, dev.paths[0], duration, rng)
    r2 = acquire(b_src, dev.paths[1], duration, rng)
    out = {}
    for est in estimators:
        q1, q2 = apply_time_model(dev, est.name, r1, r2)
        t, dphi, age = est.estimate(q1, q2, t_lo, t_hi)
        eps = wrap_deg(np.degrees(dphi - true_dphi(a_src, b_src, t, est.quantity)))
        out[est.name] = (t, eps, age)
    return out


def stats(eps):
    """Среднее, СКО, наибольшая |ε|, 95-й процентиль |ε| (градусы)."""
    e = np.asarray(eps, dtype=float)
    if e.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, max_abs=np.nan, p95_abs=np.nan)
    a = np.abs(e)
    return dict(n=int(e.size), mean=float(np.mean(e)), std=float(np.std(e)),
                max_abs=float(np.max(a)), p95_abs=float(np.percentile(a, 95)))
