"""Сводка результатов в Markdown (results/summary.md) — источник чисел для 03_rezultaty.md.

Числа печатаются с десятичной запятой. Таблицы строятся только из CSV серий.
"""
import csv
import os

import numpy as np

import config as C
from estimators import LABELS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results' + C.OUT_SUFFIX)

MAIN_ORDER = ['zc', 'zc_lpf', 'zc_lpf_comp', 'dft1_end_A', 'dft1_mid_A', 'dft1_mid_P',
              'dft10_end_A', 'dft10_mid_A', 'dft10_mid_P', 'adapt_mid_A', 'adapt_mid_P', 'adapt_now_P']


def read(name):
    with open(os.path.join(RES, name), encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def f(x, sig=2):
    """Число с двумя значащими цифрами (не меньше одного знака после запятой) и десятичной запятой."""
    x = float(x)
    if not np.isfinite(x):
        return '—'
    if x == 0:
        return '0'
    mag = int(np.floor(np.log10(abs(x))))
    nd = max(0, sig - 1 - mag)
    s = f"{x:.{nd}f}"
    return s.replace('.', ',').replace('-', '−')


def table(header, rows):
    def cells(r):  # знак модуля |x| внутри ячейки экранируется, иначе таблица распадается
        return '| ' + ' | '.join(str(c).replace('|', '\\|') for c in r) + ' |'
    out = [cells(header), '|' + '|'.join(['---'] * len(header)) + '|']
    out += [cells(r) for r in rows]
    return '\n'.join(out)


def summ_index(rows):
    return {(r['mode'], r['est']): r for r in rows}


def main():
    md = ['# Сводка результатов моделирования', '',
          'Сформировано summary.py из results/*.csv. Все числа — результаты моделирования.', '']

    # ---------- проверки
    chk = read('checks.csv')
    n_ok = sum(1 for r in chk if r['пройдена'] == 'да')
    md += ['## Проверки корректности', '', f'Пройдено {n_ok} из {len(chk)} (results/checks.csv).', '']
    ref = read('check_reference.csv')
    md += ['## Опорные числа (предварительный однофазный расчёт)', '',
           table(['Случай', 'Опорное, °', 'Модель, °', 'Расхождение, %'],
                 [[r['случай'], f(r['опорное_значение_град'], 3), f(r['модель_град'], 3),
                   r['расхождение_проц'].replace('.', ',')] for r in ref]), '']

    lit_path = os.path.join(RES, 'check_literature.csv')
    if os.path.exists(lit_path):
        lit = read('check_literature.csv')
        md += ['## Сверка с опубликованными данными (фильтр Фурье при отклонении частоты)', '',
               table(['Источник', 'Случай', 'Опубликовано', 'Формула', 'Модель'],
                     [[r['источник'], r['случай'].replace('.', ','), r['опубликовано'], r['формула'].replace('.', ','),
                       (r['модель_макс'] if r['модель_мин'] == '—' else f"{r['модель_мин']}…{r['модель_макс']}").replace('.', ',')]
                      for r in lit]), '']

    # ---------- С1
    c1 = read('c1_common_signal.csv')
    md += ['## С1. Проверка общим сигналом (наибольшая |ε| по всем испытаниям серии), °', '',
           table(['Оцениватель', 'С2', 'С3', 'С4', 'С5', 'С6', 'Все'],
                 [[r['label']] + [f(r[f'{s}_max']) for s in ('C2', 'C3', 'C4', 'C5', 'C6')] + [f(r['all_max'])]
                  for r in c1]), '']

    # ---------- С2
    c2 = read('c2_summary.csv')
    idx = {(float(r['df']), r['mode'], r['est']): r for r in c2}
    dfs = (0.2, 0.5, 1.0, 2.0)
    rows = []
    for est in MAIN_ORDER:
        rows.append([LABELS[est]] + [f(idx[(df, 'working', est)]['max_abs']) for df in dfs]
                    + [f(max(float(idx[(df, 'common', est)]['max_abs']) for df in dfs))])
    md += ['## С2. Скольжение частот: наибольшая |ε|, ° (f₁ = 50 Гц, f₂ = 50 + Δf)', '',
           table(['Оцениватель'] + [f'Δf = {f(df)} Гц' for df in dfs] + ['общий сигнал, наиб.'], rows), '']
    rows = []
    for est in ('dft1_end_A', 'dft10_end_A', 'zc_lpf'):
        rows.append([LABELS[est]] + [f(idx[(df, 'working', est)]['mean'], 3) for df in dfs])
    md += ['Среднее ε (систематическая часть), °:', '',
           table(['Оцениватель'] + [f'Δf = {f(df)} Гц' for df in dfs], rows), '']

    # ---------- С3
    c3 = summ_index(read('c3_summary.csv'))
    rows = []
    for est in MAIN_ORDER:
        w, c = c3[('working', est)], c3[('common', est)]
        rows.append([LABELS[est], f(c['p95_trial_max']), f(w['p95_abs']), f(w['p95_trial_max']), f(w['max_abs'])])
    md += [f'## С3. Гармоники, f₁ = f₂ = {f(C.C3_FREQ)} Гц, {C.C3_TRIALS} испытаний, °', '',
           table(['Оцениватель', 'Общий сигнал: P95 наиб. в испытании', 'Рабочий: P95 |ε|',
                  'Рабочий: P95 наиб. в испытании', 'Рабочий: наибольшая'], rows), '']
    zpath = os.path.join(RES, 'c3_zc_10000.csv')
    if os.path.exists(zpath):
        z = read('c3_zc_10000.csv')
        rng = np.random.default_rng(1)
        lines = []
        for key, lab in (('zc_max_abs', LABELS['zc']), ('zc_lpf_max_abs', LABELS['zc_lpf'])):
            v = np.array([float(r[key]) for r in z])
            bs = [np.percentile(rng.choice(v, len(v)), 95) for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            lines.append([lab, str(len(v)), f(np.percentile(v, 95), 3), f'[{f(lo, 3)}; {f(hi, 3)}]', f(np.max(v), 3)])
        md += ['Уточнение для перехода через ноль (c3_zc_tochnee.py; первые 2000 испытаний совпадают с основной серией), °:', '',
               table(['Оцениватель', 'Испытаний', 'P95 наиб. в испытании', '95 % интервал (бутстреп)', 'Наибольшая'], lines), '']

    # ---------- С4
    dev = read('c4_unbalance_deviation.csv')
    names = {'phaseA': 'Фазные напряжения A', 'worst_phase': 'Наибольшее по фазам A, B, C',
             'line_AB': 'Линейные напряжения AB'}
    rows = [[names[r['quantity']], '95 %' if r['limits'] == '95' else '100 %', f(r['p50']), f(r['p95']),
             f(r['p99']), f(r['max'])] for r in dev]
    md += ['## С4. Несимметрия: отклонение угла от угла между прямыми последовательностями |δ|, °', '',
           table(['Сравниваемые напряжения', 'Нормы K2U, K0U', 'Медиана', 'P95', 'P99', 'Наибольшее'], rows), '']
    c4 = summ_index(read('c4_summary.csv'))
    rows = []
    for est in MAIN_ORDER:
        w, c = c4[('working', est)], c4[('common', est)]
        rows.append([LABELS[est], f(c['p95_trial_max']), f(w['p95_abs']), f(w['p95_trial_max']), f(w['max_abs'])])
    md += [f'Оцениватели при несимметрии, f₂ − 50 Гц ~ U(−{f(C.C4_DF_MAX)}; {f(C.C4_DF_MAX)}) Гц, °:', '',
           table(['Оцениватель', 'Общий сигнал: P95 наиб.', 'Рабочий: P95 |ε|', 'Рабочий: P95 наиб.',
                  'Рабочий: наибольшая'], rows), '']

    # ---------- С5
    c5 = [r for r in read('c5_map.csv') if r['est'] == 'adapt_mid_P']
    w = [r for r in c5 if r['mode'] == 'working' and float(r['df']) > 0 and float(r['dbar_ms']) > 0]
    rel = [abs(abs(float(r['mean'])) - float(r['analytic'])) / float(r['analytic']) for r in w]
    absdev = [abs(abs(float(r['mean'])) - float(r['analytic'])) for r in w]
    cm = max(float(r['max_abs']) for r in c5 if r['mode'] == 'common')
    md += ['## С5. Общая задержка и разность частот', '',
           f'Моделирование совпало с 360°·Δf·d̄: наибольшее абсолютное расхождение {f(max(absdev))}°, '
           f'наибольшее относительное {f(100 * max(rel))} %. На общем сигнале |ε| ≤ {f(cm)}° при всех d̄ до 5 мс.', '']
    allow = read('c5_allowed_delay.csv')
    eps_set = sorted({float(r['eps_max_deg']) for r in allow})
    df_set = sorted({float(r['df_hz']) for r in allow})
    rows = []
    for e in eps_set:
        rows.append([f(e)] + [f(next(float(r['dbar_max_ms']) for r in allow
                                     if float(r['eps_max_deg']) == e and float(r['df_hz']) == d))
                              for d in df_set])
    md += ['Допустимая неучтённая общая задержка d̄, мс:', '',
           table(['Допуск на |ε|, °'] + [f'Δf = {f(d)} Гц' for d in df_set], rows), '']

    # ---------- С6
    c6 = summ_index(read('c6_summary.csv'))
    rows = []
    for est in MAIN_ORDER:
        c, wk, sy = c6[('common', est)], c6[('working', est)], c6[('sync', est)]
        rows.append([LABELS[est], f(c['p95_abs']), f(wk['p95_abs']), f(wk['max_abs']), f(sy['p95_abs'])])
    for est in ('adapt_mid_P', 'dft1_mid_P'):
        wk, sy = c6[('working_dbar', est)], c6[('sync_dbar', est)]
        rows.append([LABELS[est] + f'; неучтённая d̄ = {f(C.C6_DBAR_EXTRA * 1e3)} мс', '— (сокращается, С5)',
                     f(wk['p95_abs']), f(wk['max_abs']), f(sy['p95_abs'])])
    tr = read('c6_trials.csv')
    devA = np.array([abs(float(r['devA'])) for r in tr if r['est'] == 'adapt_mid_P' and r['mode'] == 'working'])
    rows.append(['Различие угла фаз A и угла прямых посл. (несимметрия)', '0',
                 f(np.percentile(devA, 95)), f(np.max(devA)), f(np.percentile(devA, 95))])
    md += [f'## С6. Итоговая: 95-й процентиль |ε|, °, {C.C6_TRIALS} испытаний', '',
           'Общий сигнал — сигнал резерва на оба модуля. Рабочий режим — сеть: f₁ = 50 ± 0,2 Гц, резерв: '
           'f₂ = 50 ± 1 Гц, гармоники и несимметрия в пределах норм. «Перед включением» — |f₂ − f₁| ≤ 0,2 Гц.', '',
           table(['Оцениватель', 'Общий сигнал', 'Рабочий режим', 'Рабочий: наибольшая', 'Перед включением'], rows), '']
    rows = []
    for est in MAIN_ORDER:
        c, wk, sy = c6[('common', est)], c6[('working', est)], c6[('sync', est)]
        rows.append([LABELS[est], f(c['p95_trial_max']), f(wk['p95_trial_max']), f(sy['p95_trial_max']),
                     f(wk['mean']), f(wk['std'])])
    md += ['Дополнительно: 95-й процентиль наибольшей в испытании |ε|, среднее и СКО ε (рабочий режим), °:', '',
           table(['Оцениватель', 'Общий: P95 наиб.', 'Рабочий: P95 наиб.', 'Перед вкл.: P95 наиб.',
                  'Рабочий: среднее', 'Рабочий: СКО'], rows), '']

    with open(os.path.join(RES, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md) + '\n')
    print('results/summary.md записан')


if __name__ == '__main__':
    main()
