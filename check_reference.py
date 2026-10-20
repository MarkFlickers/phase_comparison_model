"""Воспроизведение опорных чисел предварительного однофазного расчёта средствами модели.

Условия опорного расчёта: 8 кГц, 50 Гц, без шума, без задержек и смещений сеток.
Модель трёхфазная; сравнение ведётся по фазе A, что для симметричной системы
совпадает с однофазным опорным расчётом. Опорный расчёт ищет максимум
по началам окон с шагом 37 отсчётов в первые 0,5 с; модель — по всем окнам
с шагом 1 отсчёт, поэтому её максимум может быть чуть больше.
Результат: results/check_reference.csv.
"""
import csv
import os

import numpy as np

import config as C
from device import Device, ModulePath, acquire
from estimators import DFTEstimator, ZCEstimator, AdaptiveEstimator
from signals import Source, wrap_deg
from core import true_dphi

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results' + C.OUT_SUFFIX)

REFERENCE = [
    # (обозначение, опорное значение, °)
    ('ДПФ 1 период, к концу окна, Δf = 0,2 Гц', 0.83),
    ('ДПФ 1 период, к концу окна, Δf = 0,5 Гц', 2.07),
    ('ДПФ 1 период, к концу окна, Δf = 1,0 Гц', 4.14),
    ('ДПФ 10 периодов, к концу окна, Δf = 0,2 Гц', 7.31),
    ('ДПФ 10 периодов, к концу окна, Δf = 0,5 Гц', 18.27),
    ('ДПФ 10 периодов, к концу окна, Δf = 1,0 Гц', 36.54),
    ('ДПФ 1 период, к середине окна, Δf = 0,2 Гц', 0.114),
    ('ДПФ 1 период, к середине окна, Δf = 0,5 Гц', 0.285),
    ('ДПФ 1 период, к середине окна, Δf = 1,0 Гц', 0.567),
    ('ДПФ 10 периодов, к середине окна, Δf = 0,2 Гц', 0.114),
    ('ДПФ 10 периодов, к середине окна, Δf = 0,5 Гц', 0.285),
    ('ДПФ 10 периодов, к середине окна, Δf = 1,0 Гц', 0.567),
    ('Общий сигнал 49 Гц, ДПФ 1 период', 0.0),
    ('Переход через ноль, 5-я гармоника 5 %, θ5 второго источника перебирается', 2.86),
    ('ДПФ 1 период, те же сигналы, 50 Гц', 0.0),
    ('ДПФ 1 период, обе частоты 49,5 Гц, фазы 5-й гармоники разные', 0.073),
    ('ФНЧ: фаза на 49 Гц', -88.36),
    ('ФНЧ: фаза на 50 Гц', -90.0),
    ('ФНЧ: фаза на 51 Гц', -91.6),
    ('ФНЧ: ошибка разности фаз, 50 и 51 Гц', 1.64),
    ('ФНЧ: ошибка разности фаз, 49 и 50 Гц', 1.64),
]


def ideal_device():
    p = ModulePath(grid_offset=0.0, delay=0.0, ts_err=0.0, noise_sigma=0.0)
    return Device(paths=(p, p), dbar=0.0, dbar_model=0.0)


def max_err(est, s1, s2, t_lo=0.3, t_hi=1.6, duration=2.0):
    dev = ideal_device()
    rng = np.random.default_rng(0)
    r1 = acquire(s1, dev.paths[0], duration, rng)
    r2 = acquire(s2, dev.paths[1], duration, rng)
    t, dphi, _ = est.estimate(r1, r2, t_lo, t_hi)
    eps = wrap_deg(np.degrees(dphi - true_dphi(s1, s2, t, est.quantity)))
    return float(np.max(np.abs(eps)))


def main():
    os.makedirs(RES, exist_ok=True)
    model = []
    # 1. Скольжение частот (фазы как в опорном расчёте: 0,3 и −0,4 рад)
    for att in ('end', 'mid'):
        for per in (1, 10):
            for df in (0.2, 0.5, 1.0):
                est = DFTEstimator(per, att, 'A', step=1)
                model.append(max_err(est, Source(50.0, 0.3), Source(50.0 + df, -0.4)))
    # переставить в порядок таблицы REFERENCE: end-1, end-10, mid-1, mid-10
    # 2. Общий сигнал 49 Гц
    s = Source(49.0, 0.3)
    model.append(max_err(DFTEstimator(1, 'mid', 'A', step=1), s, s))
    # 3. Переход через ноль, 5-я гармоника 5 %
    zc = ZCEstimator()
    errs, errs_dft = [], []
    for th in np.linspace(0.0, 2.0 * np.pi, 73):
        s1 = Source(50.0, 0.0, harmonics=((5, 0.05, 0.0),))
        s2 = Source(50.0, 0.0, harmonics=((5, 0.05, float(th)),))
        errs.append(max_err(zc, s1, s2))
        errs_dft.append(max_err(DFTEstimator(1, 'mid', 'A', step=1), s1, s2))
    model.append(max(errs))
    model.append(max(errs_dft))
    # 4. ДПФ, обе частоты 49,5 Гц, фазы 5-й гармоники разные
    errs_495 = []
    for th in np.linspace(0.0, 2.0 * np.pi, 73):
        s1 = Source(49.5, 0.0, harmonics=((5, 0.05, 0.0),))
        s2 = Source(49.5, 0.0, harmonics=((5, 0.05, float(th)),))
        errs_495.append(max_err(DFTEstimator(1, 'mid', 'A', step=1), s1, s2))
    model.append(max(errs_495))
    # 5. ФНЧ: фазовая характеристика и ошибка детектора после ФНЧ
    zl = ZCEstimator(lpf=True)
    ph = np.degrees(zl.filter_phase([49.0, 50.0, 51.0]))
    model += [float(x) for x in ph]
    e51 = max_err(zl, Source(50.0, 0.3), Source(51.0, -0.4))
    e49 = max_err(zl, Source(49.0, 0.3), Source(50.0, -0.4))
    model += [e51, e49]

    # порядок вычисленных значений: end1(3), end10(3), mid1(3), mid10(3), общий, ПЧН, ДПФ50, ДПФ49,5, ФНЧ(3), ФНЧ-ошибки(2)
    rows = []
    for (name, ref), val in zip(REFERENCE, model):
        if abs(ref) > 1e-9:
            rel = (val - ref) / abs(ref) * 100.0
            rel_s = f"{rel:+.2f}"
        else:
            rel_s = "—"
        rows.append((name, ref, val, rel_s))
    with open(os.path.join(RES, 'check_reference.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['случай', 'опорное_значение_град', 'модель_град', 'расхождение_проц'])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", r[3]])
    for r in rows:
        print(f"{r[0]:<75s} {r[1]:>9.4f} {r[2]:>9.4f} {r[3]:>7s}")
    # справочно: точная фазовая ошибка детектора после ФНЧ из характеристики
    print('φ_Ф(50) − φ_Ф(51) =', ph[1] - ph[2], '; φ_Ф(49) − φ_Ф(50) =', ph[0] - ph[1])
    return rows


if __name__ == '__main__':
    main()
