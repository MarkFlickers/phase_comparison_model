"""Оцениватели разности фаз двух модулей.

Каждый оцениватель реализует метод estimate(r1, r2, t_lo, t_hi), который
возвращает три массива:
    t_attr — момент (на шкале прибора), к которому прибор относит результат, с;
    dphi   — оценка разности фаз Δφ̂ = φ̂_1 − φ̂_2, рад;
    age    — «возраст» результата: время от t_attr до момента выдачи, с.
Отбираются результаты с t_attr в интервале [t_lo, t_hi].

Поле quantity: 'A' — сравнение по фазе A, 'P' — по прямой последовательности.
"""
import numpy as np
from scipy import signal as sps

import config as C
from signals import A_OP, TWO_PI, wrap_rad


def pos_seq(P):
    """Прямая последовательность из фазоров фаз A, B, C (по первой оси)."""
    return (P[0] + A_OP * P[1] + A_OP ** 2 * P[2]) / 3.0


def select_quantity(P, quantity):
    return P[0] if quantity == 'A' else pos_seq(P)


# ============================================================================ ДПФ
class DFTEstimator:
    """ДПФ на номинальной частоте с прямоугольным окном из целого числа номинальных периодов.

    Экспонента берётся по меткам времени отсчётов, поэтому фазы обоих модулей
    приведены к общей шкале (перенос по номинальной частоте). Оценка по окну
    относится либо к середине окна ('mid'), либо к его концу ('end') — так
    поступает прибор, выдающий результат окна как текущий.
    """

    def __init__(self, periods, attribution='mid', quantity='A', step=C.DFT_EVAL_STEP):
        self.periods = periods
        self.N = int(round(periods * C.FS / C.F0))
        self.attribution = attribution
        self.quantity = quantity
        self.step = step
        self.name = f"dft{periods}_{attribution}_{quantity}"

    def sliding(self, rec):
        y = rec.x * np.exp(-1j * TWO_PI * C.F0 * rec.T)[None, :]
        c = np.concatenate([np.zeros((3, 1), complex), np.cumsum(y, axis=1)], axis=1)
        return (c[:, self.N:] - c[:, :-self.N]) * (2.0 / self.N)     # (3, n − N + 1)

    def estimate(self, r1, r2, t_lo, t_hi):
        X1, X2 = self.sliding(r1), self.sliding(r2)
        nw = min(X1.shape[1], X2.shape[1])
        z1 = select_quantity(X1[:, :nw], self.quantity)
        z2 = select_quantity(X2[:, :nw], self.quantity)
        dphi = np.angle(z1 * np.conj(z2))
        k = np.arange(nw)
        t_mid = 0.5 * (r1.T[k] + r1.T[k + self.N - 1])
        t_end = r1.T[k + self.N - 1]
        t_attr = t_mid if self.attribution == 'mid' else t_end
        sel = (t_attr >= t_lo) & (t_attr <= t_hi)
        idx = np.nonzero(sel)[0][::self.step]
        return t_attr[idx], dphi[idx], (t_end - t_attr)[idx]


# ============================================================================ переход через ноль
class ZCEstimator:
    """Разность фаз по переходам фазы A через ноль (снизу вверх).

    Момент перехода — линейная интерполяция между отсчётами; повторные
    переходы ближе ZC_DEAD_TIME периода отбрасываются. Для каждого перехода
    модуля 1 берётся ближайший следующий переход модуля 2:
        Δφ̂ = 2π·f̂_1·(t_2 − t_1),
    где f̂_1 — частота модуля 1 по ZC_FREQ_PERIODS последним периодам.
    Такая оценка точна в момент t_2, к нему результат и относится; это же
    момент выдачи. Вариант lpf=True — переходы ищутся после ФНЧ Баттерворта
    в контроллере. Вариант compensate=True — моменты переходов исправляются на
    фазовую задержку фильтра на измеренной частоте: τ_Ф(f̂) = −φ_Ф(f̂)/(2π f̂).
    """

    def __init__(self, lpf=False, compensate=False):
        self.lpf = lpf
        self.compensate = compensate
        self.quantity = 'A'
        self.name = "zc" + ("_lpf" if lpf else "") + ("_comp" if compensate else "")
        if lpf:
            self.b, self.a = sps.butter(C.LPF_ORDER, C.LPF_FC, fs=C.FS)

    def filter_phase(self, f):
        """Фазовая характеристика ФНЧ на частотах f, рад (вблизи −π/2 на 50 Гц)."""
        _, h = sps.freqz(self.b, self.a, worN=np.atleast_1d(f), fs=C.FS)
        return np.angle(h)

    def crossings(self, rec):
        x = rec.x[0]
        if self.lpf:
            x = sps.lfilter(self.b, self.a, x)
        T = rec.T
        i = np.nonzero((x[:-1] < 0.0) & (x[1:] >= 0.0))[0]
        tz = T[i] + (-x[i]) / (x[i + 1] - x[i]) * (T[i + 1] - T[i])
        dead = C.ZC_DEAD_TIME / C.F0
        keep, last = [], -np.inf
        for t in tz:
            if t - last >= dead:
                keep.append(t)
                last = t
        tz = np.array(keep)
        if self.lpf:                                   # переходный процесс фильтра
            tz = tz[tz > T[0] + 0.1]
        Q = C.ZC_FREQ_PERIODS
        f_hat = np.full(tz.shape, np.nan)
        if len(tz) > Q:
            f_hat[Q:] = Q / (tz[Q:] - tz[:-Q])
        t_out = tz.copy()
        if self.compensate:
            ok = np.isfinite(f_hat)
            tau = np.zeros_like(tz)
            tau[ok] = -self.filter_phase(f_hat[ok]) / (TWO_PI * f_hat[ok])
            tz = tz - tau
        return tz, f_hat, t_out

    def estimate(self, r1, r2, t_lo, t_hi):
        t1, f1, _ = self.crossings(r1)
        t2, _, out2 = self.crossings(r2)
        ok = np.isfinite(f1)
        t1, f1 = t1[ok], f1[ok]
        j = np.searchsorted(t2, t1, side='left')
        valid = j < len(t2)
        t1, f1, j = t1[valid], f1[valid], j[valid]
        dphi = wrap_rad(TWO_PI * f1 * (t2[j] - t1))
        t_attr = t2[j]
        age = out2[j] - t2[j]
        sel = (t_attr >= t_lo) & (t_attr <= t_hi)
        return t_attr[sel], dphi[sel], age[sel]


# ============================================================================ адаптивный (МНК)
class AdaptiveEstimator:
    """Оцениватель с подстройкой под частоту: многогармоническая синусная аппроксимация.

    Частота каждого модуля оценивается итерационно по приращению фазы прямой
    последовательности за ADAPT_TF. Затем в окне из ADAPT_PERIODS периодов
    измеренной частоты, центрированном в t0, по методу наименьших квадратов
    оцениваются постоянная составляющая и гармоники 1…ADAPT_HMAX измеренной
    частоты; время в модели — метки отсчётов, поэтому оценка фазы относится
    точно к t0 (середине окна). Вариант attribution='now' переносит результат
    к моменту выдачи (концу окна) по измеренным частотам:
        Δφ̂(t_out) = Δφ̂(t0) + 2π·(f̂_1 − f̂_2)·(t_out − t0).
    """

    def __init__(self, quantity='P', attribution='mid', periods=C.ADAPT_PERIODS,
                 hmax=C.ADAPT_HMAX, n_points=C.ADAPT_EVAL_POINTS):
        self.quantity = quantity
        self.attribution = attribution
        self.periods = periods
        self.hmax = hmax
        self.n_points = n_points
        self.name = f"adapt_{attribution}_{quantity}"
        self.last_freqs = None

    def ls_phasors(self, rec, t0, f):
        half = self.periods / (2.0 * f)
        i0 = np.searchsorted(rec.T, t0 - half)
        i1 = np.searchsorted(rec.T, t0 + half)
        tt = rec.T[i0:i1] - t0
        h = np.arange(1, self.hmax + 1)
        arg = TWO_PI * f * np.outer(tt, h)
        A = np.hstack([np.ones((len(tt), 1)), np.cos(arg), np.sin(arg)])
        coef, *_ = np.linalg.lstsq(A, rec.x[:, i0:i1].T, rcond=None)
        return coef[1] - 1j * coef[1 + self.hmax]        # фазоры основной гармоники A, B, C

    def frequency(self, rec, tm):
        f = C.F0
        for _ in range(C.ADAPT_ITER):
            pa = pos_seq(self.ls_phasors(rec, tm - C.ADAPT_TF / 2, f))
            pb = pos_seq(self.ls_phasors(rec, tm + C.ADAPT_TF / 2, f))
            d = wrap_rad(np.angle(pb * np.conj(pa)) - TWO_PI * f * C.ADAPT_TF)
            f = f + d / (TWO_PI * C.ADAPT_TF)
        return f

    def estimate(self, r1, r2, t_lo, t_hi):
        tm = 0.5 * (t_lo + t_hi)
        f1, f2 = self.frequency(r1, tm), self.frequency(r2, tm)
        self.last_freqs = (f1, f2)
        t0 = np.linspace(t_lo, t_hi, self.n_points)
        dphi = np.empty_like(t0)
        for k, t in enumerate(t0):
            z1 = select_quantity(self.ls_phasors(r1, t, f1), self.quantity)
            z2 = select_quantity(self.ls_phasors(r2, t, f2), self.quantity)
            dphi[k] = np.angle(z1 * np.conj(z2))
        t_out = t0 + self.periods / (2.0 * f1)
        if self.attribution == 'now':
            dphi = wrap_rad(dphi + TWO_PI * (f1 - f2) * (t_out - t0))
            return t_out, dphi, np.zeros_like(t0)
        return t0, dphi, t_out - t0


# ============================================================================ набор вариантов
LABELS = {
    'zc': 'Переход через ноль',
    'zc_lpf': 'Переход через ноль после ФНЧ',
    'zc_lpf_comp': 'Переход через ноль после ФНЧ, с учётом фазы фильтра',
    'dft1_end_A': 'ДПФ, 1 период, к концу окна, фаза A',
    'dft1_mid_A': 'ДПФ, 1 период, к середине окна, фаза A',
    'dft1_end_P': 'ДПФ, 1 период, к концу окна, прямая посл.',
    'dft1_mid_P': 'ДПФ, 1 период, к середине окна, прямая посл.',
    'dft10_end_A': 'ДПФ, 10 периодов, к концу окна, фаза A',
    'dft10_mid_A': 'ДПФ, 10 периодов, к середине окна, фаза A',
    'dft10_end_P': 'ДПФ, 10 периодов, к концу окна, прямая посл.',
    'dft10_mid_P': 'ДПФ, 10 периодов, к середине окна, прямая посл.',
    'adapt_mid_A': 'С подстройкой частоты, к середине окна, фаза A',
    'adapt_mid_P': 'С подстройкой частоты, к середине окна, прямая посл.',
    'adapt_now_P': 'С подстройкой частоты, перенос к моменту выдачи, прямая посл.',
}


def all_estimators():
    """Полный набор вариантов оценивателей, используемых в сериях."""
    ests = [ZCEstimator(), ZCEstimator(lpf=True), ZCEstimator(lpf=True, compensate=True)]
    for p in (C.DFT_PERIODS_SHORT, C.DFT_PERIODS_LONG):
        for att in ('end', 'mid'):
            for q in ('A', 'P'):
                ests.append(DFTEstimator(p, att, q))
    ests += [AdaptiveEstimator('A', 'mid'), AdaptiveEstimator('P', 'mid'),
             AdaptiveEstimator('P', 'now')]
    return ests
