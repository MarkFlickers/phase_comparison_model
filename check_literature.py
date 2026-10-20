"""Сверка формул модели с опубликованными числами по фильтру Фурье при отклонении частоты.

1. Романюк Ф. А., Румянцев Ю. В., Румянцев В. Ю. // Энергетика. Изв. вузов и энерг. объединений СНГ.
   2025. Т. 68, № 1. С. 5–16. DOI 10.21122/1029-7448-2025-68-1-5-16.
   Нерекурсивный фильтр Фурье, частота дискретизации 1200 Гц; разность фаз выходного и входного
   сигналов: «в пределах ±5 Гц … изменяется в диапазоне ±18°» (с. 7); при 48 Гц в установившемся
   режиме фазовая погрешность Δφx = 5,7–8,2° (с. 13). Результат фильтра принимается за текущий,
   то есть относится к концу окна.
2. Федотов Д. А., Бычков А. В., Дони Н. А., Дмитриев О. В. // Вестник Чувашского университета.
   2024. № 2. С. 160–170. DOI 10.47026/1810-1909-2024-2-160-170.
   Классический фильтр Фурье в УСВИ класса P, диапазон 46–52 Гц, после компенсации постоянной
   составляющей погрешности: «максимальная погрешность по амплитуде составляет 4,2 %, а по фазе −2,4°».
   Частота дискретизации в статье не используется в сверке: размах колеблющейся составляющей
   при прямоугольном окне из целого числа периодов от неё почти не зависит (проверяется ниже).

Расчёт — прямое ДПФ однофазного синусоидального сигнала без шума, окно в один номинальный период,
экспонента по абсолютному времени, все положения окна в течение 1 с.
Результат: results/check_literature.csv.
"""
import csv
import os

import numpy as np

import config as C

F0 = 50.0
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results' + C.OUT_SUFFIX)


def sweep(f, fs, duration=1.0, phi0=0.3):
    """Ошибки фазы (к концу и к середине окна) и модуль фазора для всех положений окна, °."""
    n_per = int(round(fs / F0))
    t = np.arange(int(duration * fs) + n_per) / fs
    x = np.cos(2 * np.pi * f * t + phi0)
    e = np.exp(-2j * np.pi * F0 * t)
    c = np.concatenate(([0], np.cumsum(x * e)))
    k = np.arange(len(t) - n_per + 1)
    X = (2.0 / n_per) * (c[k + n_per] - c[k])
    t_end = t[k + n_per - 1]
    t_mid = t[k] + (n_per - 1) / (2 * fs)
    ph = np.angle(X)
    true = lambda tt: 2 * np.pi * f * tt + phi0  # noqa: E731
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    eps_end = np.degrees(wrap(ph + 2 * np.pi * F0 * t_end - true(t_end)))
    eps_mid = np.degrees(wrap(ph + 2 * np.pi * F0 * t_mid - true(t_mid)))
    return eps_end, eps_mid, np.abs(X), n_per


def leak(df):
    """Амплитуда колеблющейся составляющей по формуле: arcsin[|Δf|/(2f₀ + Δf)], ° и отн. амплитуда."""
    r = abs(df) / (2 * F0 + df)
    return np.degrees(np.arcsin(r)), r


def main():
    os.makedirs(RES, exist_ok=True)
    rows = []

    # --- Романюк и др. 2025: fs = 1200 Гц, результат принимается за текущий (к концу окна)
    fs = 1200.0
    for f, published in ((48.0, '5,7–8,2'), (45.0, '±18 (в пределах ±5 Гц)'), (55.0, '±18 (в пределах ±5 Гц)')):
        eps_end, _, _, n = sweep(f, fs)
        df = f - F0
        tau = (n - 1) / (2 * fs)
        sys_part = 360.0 * df * tau
        lk, _ = leak(df)
        rows.append({
            'источник': 'Романюк и др., 2025',
            'случай': f'фильтр Фурье, 1 период, fs = {fs:.0f} Гц, f = {f:.0f} Гц, результат как текущий',
            'опубликовано': published,
            'формула': f'{abs(sys_part) - lk:.2f}…{abs(sys_part) + lk:.2f} (360°·|Δf|·τ = {abs(sys_part):.2f} ± {lk:.2f})',
            'модель_мин': f'{np.min(np.abs(eps_end)):.2f}',
            'модель_макс': f'{np.max(np.abs(eps_end)):.2f}',
            'величина': '|ε|, °',
        })

    # --- Федотов и др. 2024: 46–52 Гц, колеблющаяся составляющая после компенсации постоянной
    for fs in (1200.0, 4000.0, 8000.0):
        worst_ph, worst_am, worst_f = 0.0, 0.0, None
        for f in np.arange(46.0, 52.0001, 0.5):
            if abs(f - F0) < 1e-9:
                continue
            _, eps_mid, amp, _ = sweep(f, fs)
            ph_osc = (np.max(eps_mid) - np.min(eps_mid)) / 2
            am_osc = (np.max(amp) - np.min(amp)) / 2 / np.mean(amp)
            if ph_osc > worst_ph:
                worst_ph, worst_am, worst_f = ph_osc, am_osc, f
        lk, r = leak(worst_f - F0)
        rows.append({
            'источник': 'Федотов и др., 2024',
            'случай': f'фильтр Фурье, 1 период, fs = {fs:.0f} Гц, 46–52 Гц, колеблющаяся составляющая (наихудшая f = {worst_f:.1f} Гц)',
            'опубликовано': 'фаза 2,4°; амплитуда 4,2 %',
            'формула': f'фаза {lk:.2f}°; амплитуда {100 * r:.2f} %',
            'модель_мин': '—',
            'модель_макс': f'фаза {worst_ph:.2f}°; амплитуда {100 * worst_am:.2f} %',
            'величина': 'полуразмах',
        })

    path = os.path.join(RES, 'check_literature.csv')
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['источник']}: {r['случай']}\n   опубликовано {r['опубликовано']}; формула {r['формула']}; "
              f"модель {r['модель_мин']}…{r['модель_макс']}")
    print('results/check_literature.csv записан')
    return rows


if __name__ == '__main__':
    main()
