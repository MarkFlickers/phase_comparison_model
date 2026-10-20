"""Проверки корректности модели против аналитики.

1. Отнесение ко времени: среднее ε = 360°·Δf·τ, τ = (N − 1)/(2·FS).
2. Утечка ДПФ при отнесении к середине окна: амплитуда = arcsin|D(−(2f0+Δf))/D(Δf)|.
3. Переход через ноль: сдвиг от гармоники с относительной амплитудой k не больше
   arcsin(k) ≈ k рад; у двух источников — до 2·arcsin(k).
4. Переход через ноль после ФНЧ: ε = φ_Ф(f1) − φ_Ф(f2) по фазовой характеристике.
5. Неучтённая общая задержка: ε = 360°·Δf·d̄.
6. Прямая последовательность симметричной системы: утечка ДПФ отсутствует.
7. Несимметрия: наибольшее отклонение фазы A от прямой последовательности arcsin(K2 + K0).
8. Общий сигнал: погрешность всех оценивателей в пределах шума (подробно — серия С1).
Результат: results/checks.csv; код возврата 1, если хотя бы одна проверка не пройдена.
"""
import csv
import os
import sys

import numpy as np
from scipy import optimize

import config as C
from check_reference import ideal_device
from core import true_dphi, run_record
from device import acquire, make_device, calibrate
from estimators import DFTEstimator, ZCEstimator, AdaptiveEstimator, all_estimators
from signals import Source, wrap_deg, TWO_PI

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results' + C.OUT_SUFFIX)
ROWS = []


def record(name, model, analytic, tol_rel=None, tol_abs=None, not_above=False):
    """Запись результата проверки.

    not_above=True — проверяется неравенство model ≤ analytic (граница);
    иначе |model − analytic| ≤ tol_abs или ≤ tol_rel·|analytic|.
    """
    err = abs(model - analytic)
    if not_above:
        ok = model <= analytic
    elif tol_abs is not None:
        ok = err <= tol_abs
    else:
        ok = err <= tol_rel * abs(analytic)
    ROWS.append((name, model, analytic, err, 'да' if ok else 'НЕТ'))
    return ok


def errors(est, s1, s2, dev=None, duration=2.0, t_lo=0.3, t_hi=1.6):
    dev = dev or ideal_device()
    rng = np.random.default_rng(1)
    r1 = acquire(s1, dev.paths[0], duration, rng)
    r2 = acquire(s2, dev.paths[1], duration, rng)
    if dev.dbar_model:
        r1, r2 = r1.shifted(dev.dbar_model), r2.shifted(dev.dbar_model)
    t, dphi, _ = est.estimate(r1, r2, t_lo, t_hi)
    return wrap_deg(np.degrees(dphi - true_dphi(s1, s2, t, est.quantity)))


def dirichlet(nu, N):
    return np.sin(np.pi * nu * N / C.FS) / np.sin(np.pi * nu / C.FS) / N


def exact_zc_shift(src, h_guess=0.0):
    """Точный сдвиг перехода фазы A через ноль относительно чистой основной гармоники, рад."""
    w = TWO_PI * src.f
    t_clean = (-np.pi / 2 - src.phi) / w % (1.0 / src.f)
    g = lambda t: src.waveform(np.array([t]))[0, 0]
    lo, hi = t_clean - 0.2 / src.f, t_clean + 0.2 / src.f
    tz = optimize.brentq(g, lo, hi, xtol=1e-15)
    return w * (tz - t_clean)


def main():
    os.makedirs(RES, exist_ok=True)
    ok = True
    # 1. отнесение ко времени
    for per in (1, 10):
        for df in (-1.0, 0.5, 1.0, 2.0):
            est = DFTEstimator(per, 'end', 'A', step=1)
            e = errors(est, Source(50.0, 0.3), Source(50.0 + df, -0.4))
            tau = (est.N - 1) / (2 * C.FS)
            ok &= record(f'1. Среднее ε, ДПФ {per} пер., к концу окна, Δf = {df} Гц',
                         float(np.mean(e)), 360.0 * df * tau, tol_rel=0.01)
    # 2. утечка при отнесении к середине окна
    for per in (1, 10):
        for df in (0.2, 0.5, 1.0, 2.0):
            est = DFTEstimator(per, 'mid', 'A', step=1)
            e = errors(est, Source(50.0, 0.3), Source(50.0 + df, -0.4))
            N = est.N
            amp = np.degrees(np.arcsin(abs(dirichlet(-(2 * C.F0 + df), N) / dirichlet(df, N))))
            ok &= record(f'2. Наибольшая |ε| утечки, ДПФ {per} пер., к середине, Δf = {df} Гц',
                         float(np.max(np.abs(e))), amp, tol_rel=0.01)
    # 3. переход через ноль и гармоники
    zc = ZCEstimator()
    for h, k in ((5, 0.05), (7, 0.05), (11, 0.035), (13, 0.03)):
        shifts, model_err = [], []
        for psi in np.linspace(0.0, TWO_PI, 361):
            s2 = Source(50.0, 0.0, harmonics=((h, k, float(psi)),))
            shifts.append(exact_zc_shift(s2))
            e = errors(zc, Source(50.0, 0.0), s2, duration=0.6, t_lo=0.2, t_hi=0.5)
            model_err.append(float(np.mean(e)))
        shifts = np.degrees(np.array(shifts))
        model_err = np.array(model_err)
        # модельная ошибка = сдвиг перехода второго источника (знак: Δφ = θ1 − θ2)
        interp = float(np.max(np.abs(model_err - shifts)))
        bound = np.degrees(h ** 2 * k * (TWO_PI * 50.0 / C.FS) ** 2 / 8.0)
        ok &= record(f'3а. Наибольший сдвиг перехода, {h}-я гармоника {k*100:.1f} %, точно vs arcsin(k)',
                     float(np.max(np.abs(shifts))), float(np.degrees(np.arcsin(k))), tol_rel=0.002)
        ok &= record(f'3б. Ошибка линейной интерполяции перехода, {h}-я гармоника (граница h²k(ωΔt)²/8)',
                     interp, bound, not_above=True)
    # худшее сочетание двух источников, 5-я гармоника 5 %
    worst = 0.0
    for p1 in np.linspace(0.0, TWO_PI, 73):
        for p2 in np.linspace(0.0, TWO_PI, 73):
            e = errors(zc, Source(50.0, 0.0, harmonics=((5, 0.05, float(p1)),)),
                       Source(50.0, 0.0, harmonics=((5, 0.05, float(p2)),)),
                       duration=0.3, t_lo=0.15, t_hi=0.25)
            worst = max(worst, float(np.max(np.abs(e))))
    ok &= record('3в. Худшее сочетание двух источников, 5-я гармоника 5 % (2·arcsin k)',
                 worst, float(np.degrees(2 * np.arcsin(0.05))), tol_rel=0.01)
    # 4. переход через ноль после ФНЧ
    zl = ZCEstimator(lpf=True)
    for f1, f2 in ((50.0, 51.0), (49.0, 50.0), (50.0, 49.0), (49.8, 50.2), (50.0, 52.0)):
        e = errors(zl, Source(f1, 0.3), Source(f2, -0.4))
        ph = np.degrees(zl.filter_phase([f1, f2]))
        ok &= record(f'4. ПЧН после ФНЧ, f1 = {f1}, f2 = {f2} Гц: ε vs φ_Ф(f1) − φ_Ф(f2)',
                     float(np.mean(e)), float(ph[0] - ph[1]), tol_rel=0.005)
    zlc = ZCEstimator(lpf=True, compensate=True)
    e = errors(zlc, Source(50.0, 0.3), Source(51.0, -0.4))
    ok &= record('4б. ПЧН после ФНЧ с учётом фазы фильтра, 50 и 51 Гц: |ε| ≈ 0',
                 float(np.max(np.abs(e))), 0.0, tol_abs=0.01)
    # 5. неучтённая общая задержка
    ad = AdaptiveEstimator('P', 'mid')
    for dbar in (0.5e-3, 2e-3, 5e-3):
        for df in (0.2, 1.0, 2.0):
            dev = ideal_device()
            for p in dev.paths:
                p.delay = dbar
            e = errors(ad, Source(50.0, 0.3), Source(50.0 + df, -0.4), dev=dev)
            ok &= record(f'5. Неучтённая задержка d̄ = {dbar*1e3:.1f} мс, Δf = {df} Гц: ε vs 360·Δf·d̄',
                         float(np.mean(e)), 360.0 * df * dbar, tol_rel=0.01)
    # 6. прямая последовательность симметричной системы
    for per in (1, 10):
        est = DFTEstimator(per, 'mid', 'P', step=1)
        e = errors(est, Source(50.0, 0.3), Source(51.0, -0.4))
        ok &= record(f'6. ДПФ {per} пер., прямая посл., симметричная система, Δf = 1 Гц: |ε| ≈ 0',
                     float(np.max(np.abs(e))), 0.0, tol_abs=1e-6)
    # 7. несимметрия
    for k2, k0 in ((0.02, 0.0), (0.02, 0.02), (0.04, 0.04)):
        dev_max = 0.0
        for a in np.linspace(0.0, TWO_PI, 181):
            for b in np.linspace(0.0, TWO_PI, 181):
                s = Source(50.0, 0.0, k2=k2, psi2=float(a), k0=k0, psi0=float(b))
                d = abs(np.degrees(np.angle(s.phase_phasors()[0])))
                dev_max = max(dev_max, d)
        ok &= record(f'7. Наибольшее отклонение фазы A от прямой посл., K2U = {k2}, K0U = {k0}',
                     dev_max, float(np.degrees(np.arcsin(k2 + k0))), tol_rel=0.002)
    # 8. общий сигнал: шумовой уровень
    rng = np.random.default_rng(C.SEED)
    ests = all_estimators()
    dev = calibrate(make_device(rng), ests, rng)
    src = Source(51.0, 0.7, harmonics=((5, 0.05, 1.0), (7, 0.04, 2.0)), k2=0.02, psi2=0.5, k0=0.02, psi0=2.0)
    out = run_record(dev, ests, src, src, rng, common=True)
    worst_name, worst_val = max(((n, float(np.max(np.abs(v[1])))) for n, v in out.items()), key=lambda z: z[1])
    ok &= record(f'8. Общий сигнал 51 Гц с гармониками и несимметрией: max|ε| по всем оценивателям ({worst_name})',
                 worst_val, 0.0, tol_abs=0.1)

    with open(os.path.join(RES, 'checks.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['проверка', 'модель', 'аналитика', 'расхождение', 'пройдена'])
        for r in ROWS:
            w.writerow([r[0], f"{r[1]:.6g}", f"{r[2]:.6g}", f"{r[3]:.3g}", r[4]])
    for r in ROWS:
        print(f"{r[4]:>3s}  {r[0]:<95s} {r[1]:>11.5g} {r[2]:>11.5g}")
    print('ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if ok else 'ЕСТЬ НЕПРОЙДЕННЫЕ ПРОВЕРКИ')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
