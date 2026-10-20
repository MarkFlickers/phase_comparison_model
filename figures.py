"""Рисунки статьи. Строятся только из CSV в results/ (кроме схем рис. 1 и 2).

Требования: подписи на русском, десятичная запятая, единицы; рисунок читается
в оттенках серого (стили линий и маркеры); PNG 300 dpi и PDF; один рисунок —
одна мысль. Подписи «Рис. n. …» — в тексте статьи, на самих рисунках заголовков нет.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import FancyBboxPatch, Rectangle

import config as C
from estimators import ZCEstimator

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results' + C.OUT_SUFFIX)
FIG = os.path.join(HERE, 'figures' + C.OUT_SUFFIX)
CM = 1 / 2.54

INK = '#000000'
INK2 = '#4d4d4d'
INK3 = '#808080'
GRID = '#d9d9d9'

STYLES = [   # (цвет, линия, маркер) — различимы в оттенках серого
    dict(color=INK, linestyle='-', marker='o'),
    dict(color=INK2, linestyle='--', marker='s'),
    dict(color=INK, linestyle='-.', marker='^'),
    dict(color=INK3, linestyle=':', marker='D'),
]


def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': C.FONT_SIZE,
        'axes.linewidth': 0.6, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'lines.linewidth': 1.3, 'lines.markersize': 5.5,
        'legend.frameon': False, 'legend.fontsize': C.FONT_SIZE - 1,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-',
        'axes.axisbelow': True, 'savefig.dpi': C.FIG_DPI, 'pdf.fonttype': 42,
    })


def comma(x, nd=None):
    """Число с десятичной запятой; nd — число знаков после запятой (None — по величине)."""
    if nd is None:
        s = np.format_float_positional(float(x), precision=6, unique=True, trim='-')
    else:
        s = f"{x:.{nd}f}"
    return s.replace('.', ',').replace('-', '−')


def log_formatter():
    return ticker.FuncFormatter(lambda v, _: comma(v))


def lin_formatter(nd=None):
    return ticker.FuncFormatter(lambda v, _: comma(v, nd))


def save(fig, name):
    """PNG 300 dpi (RGB, белый непрозрачный фон) и векторный PDF."""
    os.makedirs(FIG, exist_ok=True)
    png = os.path.join(FIG, name + '.png')
    fig.savefig(png, dpi=C.FIG_DPI, facecolor='white')
    fig.savefig(os.path.join(FIG, name + '.pdf'), facecolor='white')
    plt.close(fig)
    try:
        from PIL import Image
        im = Image.open(png)
        if im.mode != 'RGB':
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1] if im.mode == 'RGBA' else None)
            bg.save(png, dpi=(C.FIG_DPI, C.FIG_DPI))
    except ImportError:
        pass


def read(path):
    with open(os.path.join(RES, path), encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def figsize(h=C.FIG_HEIGHT_CM):
    return (C.FIG_WIDTH_CM * CM, h * CM)


# ============================================================================ рис. 1
def fig_structure():
    """Обобщённая структурная схема класса устройств (не схема конкретного изделия).

    Оси построены в масштабе 1:1 к размеру рисунка: единица — 1 см.
    """
    W, H = C.FIG_WIDTH_CM, 10.0
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    fs = 7.5

    def box(x, y, w, h, text, lw=1.0, size=fs, weight='normal'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.12',
                                    linewidth=lw, edgecolor=INK, facecolor='white'))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size,
                    color=INK, linespacing=1.35, fontweight=weight)

    def arrow(pts, dashed=False, lw=1.0):
        xs, ys = zip(*pts)
        ls = (0, (4, 2.5)) if dashed else '-'
        if len(pts) > 2:
            ax.plot(xs[:-1], ys[:-1], color=INK, lw=lw, linestyle=ls, solid_capstyle='butt')
        ax.annotate('', xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle='-|>', lw=lw, color=INK, linestyle=ls,
                                    shrinkA=0, shrinkB=0, mutation_scale=8))

    mods = ((6.0, 'Основной\nисточник\n(сеть)', 1), (1.7, 'Резервный\nисточник\n(ДГУ)', 2))
    for ym, src, k in mods:
        # электрический домен модуля
        ax.add_patch(Rectangle((2.95, ym - 0.45), 4.55, 3.1, fill=False, linestyle=(0, (4, 3)),
                               linewidth=0.7, edgecolor=INK2))
        if k == 1:
            ax.text(3.12, ym + 2.5, 'Электрический домен 1', fontsize=fs - 0.5, color=INK2, va='center')
        else:
            ax.text(3.12, ym - 0.22, 'Электрический домен 2', fontsize=fs - 0.5, color=INK2, va='center')
        box(0.2, ym + 0.35, 1.75, 1.5, src)
        arrow([(1.95, ym + 1.1), (3.2, ym + 1.1)])
        ax.text(2.57, ym + 1.22, 'u$_A$, u$_B$, u$_C$', fontsize=fs - 0.5, ha='center', va='bottom', color=INK)
        box(3.2, ym, 4.0, 2.2, '')
        ax.text(5.2, ym + 1.65, f'Измерительный модуль {k}', ha='center', va='center', fontsize=fs,
                fontweight='bold', color=INK)
        ax.text(5.2, ym + 0.8, 'АЦП, цифровые фильтры,\nвыдача кадров отсчётов', ha='center',
                va='center', fontsize=fs, color=INK, linespacing=1.35)
    # гальванический барьер
    for x in (7.8, 8.05):
        ax.add_patch(Rectangle((x, 0.5), 0.13, 8.75, facecolor='#bfbfbf', edgecolor='none'))
    ax.text(7.99, 9.45, 'Гальванический барьер', ha='center', va='center', fontsize=fs, color=INK)
    # центральный контроллер
    box(8.6, 0.8, 5.0, 8.2, '', lw=1.3)
    ax.text(11.1, 8.6, 'Центральный контроллер', ha='center', va='center', fontsize=fs + 0.5,
            fontweight='bold', color=INK)
    box(8.9, 4.2, 1.7, 1.4, 'Генератор\nобщего\nтакта')
    box(10.9, 1.1, 2.45, 7.0, 'Приём кадров\n\nАппаратный\nтаймер:\nметки начала\nкадров\n\n'
        'Оценка фаз\nи частот\n\nПриведение\nк общему\nмоменту,\nпоправка\nкалибровки\n\nСравнение')
    # общий такт к модулям (пунктир)
    arrow([(8.9, 5.2), (5.2, 5.2), (5.2, 6.0)], dashed=True)
    arrow([(8.9, 4.6), (5.2, 4.6), (5.2, 3.9)], dashed=True)
    ax.text(6.45, 4.9, 'общий такт', fontsize=fs - 0.5, ha='center', va='center', color=INK)
    # кадры отсчётов к контроллеру
    for y in (6.8, 2.5):
        arrow([(7.2, y), (10.9, y)])
        ax.text(9.75, y + 0.12, 'кадры отсчётов', fontsize=fs - 0.5, ha='center', va='bottom', color=INK)
    # выход
    arrow([(13.35, 4.6), (14.0, 4.6)])
    box(14.0, 3.6, 1.85, 2.0, 'Δφ, Δf, ΔU\n→ АВР,\nсинхрони-\nзатор')
    save(fig, 'ris1_struktura')


# ============================================================================ рис. 2
def fig_time_attribution():
    """Механизм отнесения ко времени: оценка по окну относится к середине окна в аналоговом времени.

    Знак согласован с моделью (core.true_dphi): Δφ = φ₁ − φ₂, Δf = f₂ − f₁, поэтому при Δf > 0
    истинная разность фаз убывает: Δφ(t) = Δφ(t₀) − 360°·Δf·(t − t₀). Запаздывающая оценка
    больше текущего значения, ε = Δφ(t − τ) − Δφ(t) = +360°·Δf·τ.
    Общая задержка цепочки d̄ входит в τ наравне с половиной окна.
    """
    fig, ax = plt.subplots(figsize=figsize(9.5))
    df = 1.0                          # Гц
    slope = 360.0 * df / 1000.0       # °/мс
    phi0 = 28.0
    t = np.linspace(0, 50, 400)
    ax.plot(t, phi0 - slope * t, color=INK, lw=1.4)
    ax.annotate('истинная разность фаз Δφ(t)\nпри Δf = f₂ − f₁ = 1 Гц', xy=(6.0, phi0 - slope * 6.0),
                xytext=(1.0, 20.0), fontsize=C.FONT_SIZE - 1, ha='left', va='top',
                arrowprops=dict(arrowstyle='-', lw=0.6, color=INK2, shrinkA=2, shrinkB=2))
    te, Tw, dbar = 38.0, 20.0, 4.0    # момент выдачи, длина окна, общая задержка, мс
    ax.axvspan(te - Tw - dbar, te - dbar, color='#e6e6e6', zorder=0)
    ax.text(te - Tw - dbar + 0.4, 27.6, 'отсчёты окна\nв аналоговом времени', ha='left', va='top',
            fontsize=C.FONT_SIZE - 2, color=INK2)
    tc = te - Tw / 2 - dbar
    yc = phi0 - slope * tc
    ye = phi0 - slope * te
    ax.plot([tc, te], [yc, yc], color=INK, lw=1.1, linestyle='--')
    ax.text((tc + te) / 2 + 1.0, yc + 0.15, 'оценка по окну Δφ̂', ha='center', va='bottom', fontsize=C.FONT_SIZE - 1)
    ax.plot([tc], [yc], marker='o', color=INK, ms=5, linestyle='none')
    ax.plot([te], [ye], marker='s', color=INK, ms=5, linestyle='none')
    ax.text(tc - 0.4, yc - 0.45, 'середина\nокна', ha='right', va='top', fontsize=C.FONT_SIZE - 2, color=INK2)
    ax.axvline(te, color=INK3, lw=0.8)
    ax.text(te + 0.4, 27.6, 'момент выдачи', ha='left', va='top', fontsize=C.FONT_SIZE - 2, color=INK2)
    ax.annotate('', xy=(te + 0.7, ye), xytext=(te + 0.7, yc),
                arrowprops=dict(arrowstyle='<->', lw=0.9, color=INK, shrinkA=0, shrinkB=0))
    ax.text(te + 1.2, (yc + ye) / 2 + 0.8, 'ε = 360°·Δf·τ', va='center', fontsize=C.FONT_SIZE - 1)
    ax.annotate('', xy=(tc, 13.2), xytext=(te, 13.2),
                arrowprops=dict(arrowstyle='<->', lw=0.9, color=INK, shrinkA=0, shrinkB=0))
    ax.text((tc + te) / 2 - 2.0, 13.45, 'τ = T$_{ок}$/2 + d̄', ha='center', va='bottom', fontsize=C.FONT_SIZE - 1)
    # окно по меткам прибора и сдвиг на общую задержку
    ax.annotate('', xy=(te - Tw, 10.6), xytext=(te, 10.6),
                arrowprops=dict(arrowstyle='|-|', lw=0.8, color=INK, shrinkA=0, shrinkB=0, mutation_scale=3))
    ax.text(te - Tw / 2, 10.8, 'окно по меткам прибора, T$_{ок}$', ha='center', va='bottom',
            fontsize=C.FONT_SIZE - 2, color=INK)
    ax.annotate('', xy=(te - dbar, 12.0), xytext=(te, 12.0),
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color=INK, shrinkA=0, shrinkB=0, mutation_scale=7))
    ax.text(te - dbar / 2, 12.1, 'd̄', ha='center', va='bottom', fontsize=C.FONT_SIZE - 1)
    ax.set_xlim(0, 50)
    ax.set_ylim(10, 28)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_xlabel('Время t, мс')
    ax.set_ylabel('Разность фаз Δφ = φ₁ − φ₂, °')
    ax.xaxis.set_major_formatter(lin_formatter(0))
    ax.yaxis.set_major_formatter(lin_formatter(0))
    fig.tight_layout()
    save(fig, 'ris2_otnesenie_ko_vremeni')


# ============================================================================ рис. 3 (С2)
def c2_curve(rows, est, mode='working', stat='max_abs', positive=True):
    pts = sorted((float(r['df']), float(r[stat])) for r in rows
                 if r['est'] == est and r['mode'] == mode and (float(r['df']) > 0) == positive)
    x = np.array([abs(p[0]) for p in pts])
    y = np.array([abs(p[1]) for p in pts])
    o = np.argsort(x)
    return x[o], y[o]


def fig_c2_time():
    rows = read('c2_summary.csv')
    fig, ax = plt.subplots(figsize=figsize())
    zl = ZCEstimator(lpf=True)
    xs = np.logspace(np.log10(0.02), np.log10(2.0), 200)
    series = [
        ('dft10_end_A', 'ДПФ, 10 периодов, результат как текущий (τ ≈ 100 мс)', (10 * C.FS / C.F0 - 1) / (2 * C.FS)),
        ('dft1_end_A', 'ДПФ, 1 период, результат как текущий (τ ≈ 10 мс)', (C.FS / C.F0 - 1) / (2 * C.FS)),
        ('zc_lpf', 'Переход через ноль после ФНЧ (τ ≈ 4,5 мс)', None),
    ]
    for k, (est, lab, tau) in enumerate(series):
        x, y = c2_curve(rows, est)
        st = STYLES[k]
        ax.plot(x, y, linestyle='none', marker=st['marker'], color=st['color'], label=lab,
                markerfacecolor='white' if k == 1 else st['color'], markeredgewidth=1.0)
        if tau is not None:
            ax.plot(xs, 360.0 * xs * tau, color=st['color'], lw=0.9, linestyle=st['linestyle'])
        else:
            ph = np.degrees(zl.filter_phase(C.F0 + xs))
            ax.plot(xs, np.degrees(zl.filter_phase([C.F0]))[0] - ph, color=st['color'], lw=0.9,
                    linestyle=st['linestyle'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.018, 2.2)
    ax.set_ylim(0.005, 120)
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]))
    ax.xaxis.set_major_formatter(log_formatter())
    ax.yaxis.set_major_formatter(log_formatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel('Разность частот источников Δf = f₂ − f₁, Гц (f₁ = 50 Гц)')
    ax.set_ylabel('Наибольшая погрешность |ε|, °')
    ax.grid(True, which='major')
    ax.legend(loc='upper left')
    ax.text(0.98, 0.04, 'точки — моделирование; линии — 360°·Δf·τ\n(для ФНЧ — по фазовой характеристике)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=C.FONT_SIZE - 2, color=INK2)
    fig.tight_layout()
    save(fig, 'ris3_C2_otnesenie')


# ============================================================================ рис. 4 (С2)
def fig_c2_leakage():
    rows = read('c2_summary.csv')
    fig, ax = plt.subplots(figsize=figsize())
    series = [
        ('dft1_mid_A', 'ДПФ, 1 период, фаза A'),
        ('dft10_mid_A', 'ДПФ, 10 периодов, фаза A'),
        ('dft1_mid_P', 'ДПФ, 1 период, прямая последовательность'),
        ('adapt_mid_P', 'МНК с подстройкой частоты, прямая последовательность'),
    ]
    for k, (est, lab) in enumerate(series):
        x, y = c2_curve(rows, est)
        st = STYLES[k]
        ax.plot(x, y, label=lab, markerfacecolor='white' if k in (1, 3) else st['color'],
                markeredgewidth=1.0, **st)
    xs = np.linspace(0.02, 2.0, 200)
    ax.plot(xs, np.degrees(np.arcsin(xs / (2 * C.F0 + xs))), color=INK3, lw=0.8, linestyle='-')
    # прямые подписи кривых
    fs = C.FONT_SIZE - 1
    ax.text(0.62, 1.25, 'ДПФ, фаза A: окно 1 период (●) и 10 периодов (□);\n'
            'тонкая линия — arcsin[Δf/(2f₀ + Δf)]', ha='left', va='bottom', fontsize=fs - 1, color=INK)
    ax.text(0.62, 0.042, 'ДПФ, 1 период, прямая последовательность', ha='left', va='bottom',
            fontsize=fs - 1, color=INK)
    ax.text(0.62, 0.0042, 'МНК с подстройкой частоты, прямая последовательность', ha='left', va='top',
            fontsize=fs - 1, color=INK)
    ax.set_yscale('log')
    ax.set_xlim(0, 2.05)
    ax.set_ylim(1e-3, 5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(lin_formatter(1))
    ax.yaxis.set_major_formatter(log_formatter())
    ax.set_xlabel('Разность частот источников Δf = f₂ − f₁, Гц (f₁ = 50 Гц)')
    ax.set_ylabel('Наибольшая погрешность |ε|, °')
    fig.tight_layout()
    save(fig, 'ris4_C2_utechka')


# ============================================================================ рис. 5 (С3)
def ecdf(vals):
    v = np.sort(np.asarray(vals, dtype=float))
    p = np.arange(1, len(v) + 1) / len(v) * 100.0
    return v, p


def fig_c3():
    """С3: M95 — распределение наибольшей в испытании |ε|. Переходы через ноль — 10 000 испытаний
    (c3_zc_10000.csv), остальные оцениватели — 2000 испытаний основной серии (c3_trials.csv)."""
    rows = read('c3_trials.csv')
    fig, ax = plt.subplots(figsize=figsize())
    series = [
        ('zc', 'Переход\nчерез ноль', (3.0, 80), 'right'),
        ('zc_lpf', 'Переход через\nноль после ФНЧ', (0.085, 33), 'left'),
        ('dft1_mid_A', 'ДПФ, 1 период,\nфаза A', (0.0085, 66), 'left'),
        ('adapt_mid_A', 'МНК с подстройкой\nчастоты, фаза A', (0.0026, 62), 'right'),
    ]
    zc10k = read('c3_zc_10000.csv')      # уточнение для переходов через ноль: 10 000 испытаний
    for k, (est, lab, (lx, ly), ha) in enumerate(series):
        if est in ('zc', 'zc_lpf'):
            vals = [float(r[est + '_max_abs']) for r in zc10k]
        else:
            vals = [float(r['max_abs']) for r in rows if r['est'] == est and r['mode'] == 'working']
        x, p = ecdf(vals)
        st = STYLES[k]
        ax.plot(x, p, color=st['color'], linestyle=st['linestyle'], lw=1.4)
        i95 = int(np.searchsorted(p, 95.0))
        ax.plot([x[i95]], [95.0], marker=st['marker'], color=st['color'],
                markerfacecolor='white' if k in (1, 3) else st['color'], markeredgewidth=1.0, ms=6)
        ax.text(lx, ly, lab, ha=ha, va='center', fontsize=C.FONT_SIZE - 1, color=INK, linespacing=1.2)
    ax.axhline(95.0, color=INK3, lw=0.6)
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 20)
    ax.set_ylim(0, 101)
    ax.xaxis.set_major_formatter(log_formatter())
    ax.yaxis.set_major_formatter(lin_formatter(0))
    ax.set_xlabel('Наибольшая погрешность в испытании |ε|, °')
    ax.set_ylabel('Доля испытаний, %')
    fig.tight_layout()
    save(fig, 'ris5_C3_garmoniki')


# ============================================================================ рис. 6 (С4)
def fig_c4():
    rows = read('c4_unbalance_quantiles.csv')
    q = np.array([float(r['quantile']) for r in rows]) * 100.0
    fig, ax = plt.subplots(figsize=figsize())
    series = [
        ('phaseA_95', 'Фазные напряжения A'),
        ('worst_phase_95', 'Фазные напряжения, наибольшее по фазам A, B, C'),
        ('line_AB_95', 'Линейные напряжения AB'),
    ]
    for k, (col, lab) in enumerate(series):
        x = np.array([float(r[col]) for r in rows])
        st = STYLES[k]
        ax.plot(x, q, color=st['color'], linestyle=st['linestyle'], lw=1.4, label=lab)
        i95 = int(np.searchsorted(q, 95.0))
        ax.plot([x[i95]], [95.0], marker=st['marker'], color=st['color'],
                markerfacecolor='white' if k == 1 else st['color'], markeredgewidth=1.0, ms=6)
    ax.axhline(95.0, color=INK3, lw=0.6)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 101)
    ax.xaxis.set_major_formatter(lin_formatter(0))
    ax.yaxis.set_major_formatter(lin_formatter(0))
    ax.set_xlabel('Отклонение угла от угла между прямыми последовательностями |δ|, °')
    ax.set_ylabel('Доля сочетаний несимметрии, %')
    ax.legend(loc='lower right')
    ax.text(0.98, 0.30, 'K2U, K0U каждого источника ~ U(0; 2 %)', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=C.FONT_SIZE - 2, color=INK2)
    fig.tight_layout()
    save(fig, 'ris6_C4_nesimmetriya')


# ============================================================================ доп. рисунок (С5)
def fig_c5():
    """Допустимая неучтённая общая задержка. В набор рисунков статьи не входит (в тексте — таблица С5)."""
    rows = [r for r in read('c5_map.csv') if r['est'] == 'adapt_mid_P' and r['mode'] == 'working']
    D = sorted({float(r['dbar_ms']) for r in rows})
    F = sorted({float(r['df']) for r in rows})
    Z = np.zeros((len(D), len(F)))
    for r in rows:
        Z[D.index(float(r['dbar_ms'])), F.index(float(r['df']))] = abs(float(r['mean']))
    fig, ax = plt.subplots(figsize=figsize())
    ax.grid(False)
    cs = ax.contour(F, D, Z, levels=C.C5_LEVELS_DEG, colors=INK, linewidths=1.0)
    ax.clabel(cs, fmt=lambda v: comma(v) + '°', fontsize=C.FONT_SIZE - 2, inline=True, inline_spacing=3)
    tau_lpf = 1e3 * float(np.sqrt(2) / (2 * np.pi * C.LPF_FC))
    ax.axhline(tau_lpf, color=INK3, lw=0.8, linestyle='--')
    ax.text(1.98, tau_lpf - 0.08, 'групповая задержка ФНЧ 2-го порядка, 50 Гц (≈ 4,5 мс)', ha='right',
            va='top', fontsize=C.FONT_SIZE - 2, color=INK2,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 5.0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(lin_formatter(1))
    ax.yaxis.set_major_formatter(lin_formatter(0))
    ax.set_xlabel('Разность частот источников |Δf|, Гц')
    ax.set_ylabel('Неучтённая общая задержка d̄, мс')
    fig.tight_layout()
    save(fig, 'dop_C5_zaderzhka')


# ============================================================================ рис. 7 (С6)
C6_ROWS = [  # (оцениватель, подпись строки) — порядок задаётся по рабочему режиму
    ('dft10_end_A', 'ДПФ 10 периодов, результат\nокна принят за текущий'),
    ('zc', 'Переход через ноль'),
    ('dft1_end_A', 'ДПФ 1 период, результат\nокна принят за текущий'),
    ('zc_lpf', 'Переход через ноль после ФНЧ'),
    ('dft1_mid_A', 'ДПФ 1 период, к середине\nокна, фаза A'),
    ('zc_lpf_comp', 'Переход через ноль после ФНЧ\nс учётом фазы фильтра'),
    ('dft1_mid_P', 'ДПФ 1 период, к середине\nокна, прямая последовательность'),
    ('adapt_mid_P', 'МНК с подстройкой частоты,\nпрямая последовательность'),
]


def fig_c6():
    """Главный результат: один и тот же прибор на общем сигнале и в рабочем режиме (P95 |ε|).

    Форма — «гантель» по оценивателям на логарифмической оси: проверка общим сигналом (○),
    рабочий режим (●), перед включением (◆). Одна мысль: на общем сигнале оцениватели
    неразличимы, в рабочем режиме расходятся на три порядка.
    """
    s = {(r['mode'], r['est']): float(r['p95_abs']) for r in read('c6_summary.csv')}
    rows = sorted(C6_ROWS, key=lambda e: -s[('working', e[0])])
    fig = plt.figure(figsize=figsize(10.5))
    ax = fig.add_axes([0.37, 0.13, 0.60, 0.84])
    n = len(rows)
    mk_common = dict(marker='o', ms=6.5, markerfacecolor='none', markeredgecolor=INK, markeredgewidth=1.1,
                     linestyle='none')
    mk_sync = dict(marker='D', ms=5.0, markerfacecolor=INK3, markeredgecolor='white', markeredgewidth=0.8,
                   linestyle='none')
    mk_work = dict(marker='o', ms=6.5, markerfacecolor=INK, markeredgecolor='white', markeredgewidth=0.8,
                   linestyle='none')
    for i, (est, _) in enumerate(rows):
        y = n - 1 - i
        c, w, b = s[('common', est)], s[('working', est)], s[('sync', est)]
        ax.plot([c, w], [y, y], color=INK3, lw=0.9, zorder=1, solid_capstyle='butt')
        ax.plot(c, y, zorder=4, **mk_common)   # кольцо без заливки: видно и при наложении на ◆
        ax.plot(w, y, zorder=5, **mk_work)
        # ◆ поверх ●: при совпадении значений (переход через ноль) ромб виден внутри точки
        same = abs(np.log10(b / w)) < 0.08
        ax.plot(b, y, zorder=6, **(dict(mk_sync, ms=3.2, markeredgecolor=INK3, markeredgewidth=0.3)
                                   if same else mk_sync))
        # подписываем только рабочий режим — значение, ради которого строится рисунок
        ax.text(w * 1.4, y, comma(float(f'{w:.2g}')) + '°', va='center', ha='left',
                fontsize=C.FONT_SIZE - 2, color=INK, zorder=7,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in rows][::-1], fontsize=C.FONT_SIZE - 2, linespacing=1.1)
    ax.tick_params(axis='y', length=0)
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 200)
    ax.set_ylim(-0.6, n - 0.2)
    ax.xaxis.set_major_locator(ticker.FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax.xaxis.set_major_formatter(log_formatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis='x', labelsize=C.FONT_SIZE - 1)
    ax.grid(True, axis='x', which='major')
    ax.grid(False, axis='y')
    ax.set_xlabel('Q95 — 95-й процентиль объединённых значений |ε|, °', fontsize=C.FONT_SIZE - 1)
    # ориентир масштаба: 1° — порядок допуска на погрешность угла опережения синхронизатора [8];
    # это не норма погрешности прибора, сравнение с линией соответствия не означает
    ax.axvline(1.0, color=INK2, lw=0.8, zorder=0, linestyle=(0, (4, 2)))
    ax.text(0.92, n - 0.45, 'ориентир 1°', ha='right', va='center', fontsize=C.FONT_SIZE - 2, color=INK2)
    # легенда — в пустом правом нижнем углу поля
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], label='общий сигнал', **mk_common),
        Line2D([], [], label='разные сигналы, резерв 50 ± 1 Гц', **mk_work),
        Line2D([], [], label='перед включением, |Δf| ≤ 0,2 Гц', **mk_sync),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=C.FONT_SIZE - 2, handletextpad=0.3,
              borderaxespad=0.4, frameon=True, framealpha=1.0, edgecolor='none', facecolor='white')
    save(fig, 'ris7_C6_itog')


# ============================================================================ рис. 8 (схема опыта)
def fig_test_setup():
    """Обобщённая схема проверки прибора двумя разными сигналами с общей меткой времени."""
    W, H = C.FIG_WIDTH_CM, 8.0
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    fs = 7.5

    def box(x, y, w, h, text='', lw=1.0, size=fs, weight='normal'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.12',
                                    linewidth=lw, edgecolor=INK, facecolor='white'))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size,
                    color=INK, linespacing=1.35, fontweight=weight)

    def arrow(pts, dashed=False, lw=1.0, both=False):
        xs, ys = zip(*pts)
        ls = (0, (4, 2.5)) if dashed else '-'
        if len(pts) > 2:
            ax.plot(xs[:-1], ys[:-1], color=INK, lw=lw, linestyle=ls, solid_capstyle='butt')
        ax.annotate('', xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle='<|-|>' if both else '-|>', lw=lw, color=INK, linestyle=ls,
                                    shrinkA=0, shrinkB=0, mutation_scale=8))

    # генератор
    box(0.2, 1.0, 4.3, 6.6, '', lw=1.3)
    ax.text(2.35, 6.95, 'Имитатор двух\nтрёхфазных источников\n(сеть и ДГУ)', ha='center', va='center',
            fontsize=fs, fontweight='bold', color=INK, linespacing=1.3)
    box(0.5, 4.45, 3.7, 1.75, 'Система 1: u₁A, u₁B, u₁C\nf₁, гармоники,\nнесимметрия')
    box(0.5, 2.0, 3.7, 1.75, 'Система 2: u₂A, u₂B, u₂C\nf₂, свои гармоники\nи несимметрия')
    ax.text(2.35, 1.35, 'фазы выходов заданы\nотносительно метки t₀', ha='center', va='center', fontsize=fs - 0.5,
            color=INK2, linespacing=1.2)
    # прибор
    box(6.0, 1.0, 5.2, 6.6, '', lw=1.3)
    ax.text(8.6, 7.15, 'Проверяемый прибор', ha='center', va='center', fontsize=fs, fontweight='bold', color=INK)
    box(6.3, 4.6, 1.9, 1.45, 'Модуль 1')
    box(6.3, 2.15, 1.9, 1.45, 'Модуль 2')
    box(8.7, 1.4, 2.2, 5.2, 'Контроллер\n\nаппаратный\nтаймер:\nметки кадров\nи метка t₀\n\nΔφ̂(t)\nс меткой\nвремени')
    for y in (5.32, 2.87):
        arrow([(4.2, y), (6.3, y)])
        arrow([(8.2, y), (8.7, y)])
    ax.text(5.25, 5.45, 'u₁A, B, C', ha='center', va='bottom', fontsize=fs, color=INK)
    ax.text(5.25, 3.0, 'u₂A, B, C', ha='center', va='bottom', fontsize=fs, color=INK)
    # общая метка времени
    arrow([(2.35, 1.0), (2.35, 0.45), (9.8, 0.45), (9.8, 1.4)], dashed=True)
    ax.text(6.05, 0.58, 'общая метка времени t₀ (импульс)', ha='center', va='bottom', fontsize=fs - 0.5, color=INK)
    # сравнение с истинной разностью фаз
    arrow([(10.9, 4.0), (11.75, 4.0)])
    box(11.75, 2.35, 4.05, 3.3, 'Сравнение с истинной\nразностью фаз\n\nΔφ(t) = Δφ₀ −\n− 360°·(f₂ − f₁)·(t − t₀)\n\n'
        'ε(t) = Δφ̂(t) − Δφ(t)')
    save(fig, 'ris8_opyt_dva_signala')


def main():
    setup()
    fig_structure()
    fig_time_attribution()
    fig_c2_time()
    fig_c2_leakage()
    fig_c3()
    fig_c4()
    fig_c6()
    fig_test_setup()
    fig_c5()


if __name__ == '__main__':
    main()
