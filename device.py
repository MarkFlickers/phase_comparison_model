"""Двухмодульный прибор: сетки выборки, задержки трактов, метки, калибровка.

Модель времени. Контроллер ведёт общую шкалу t. Модуль i выдаёт отсчёты с
периодом 1/FS; его сетка смещена относительно шкалы контроллера на δ_i
(общий такт: смещение постоянно до перезапуска). Момент начала кадра
фиксируется аппаратным таймером контроллера с погрешностью e_i, поэтому
отсчёту n модуля i прибор приписывает метку

    T_i[n] = n/FS + δ_i + e_i.

Отсчёт при этом несёт значение аналогового сигнала в момент
n/FS + δ_i − d_i, где d_i = d̄ + Δd_i — задержка тракта (аналоговые цепи,
АЦП, цифровые фильтры модуля, передача кадра).

Калибровка на общем чистом сигнале 50 Гц даёт поправку c = d_2 − d_1 (плюс
разность погрешностей меток). Поправка хранится как время, а не как угол.
Модель времени прибора вычитает из меток модуля 2 поправку c, а из меток обоих
модулей — учтённую общую задержку d̂ (0, если прибор её не учитывает).
"""
from dataclasses import dataclass, field, replace
import numpy as np

import config as C
from signals import Source, TWO_PI


@dataclass
class ModulePath:
    grid_offset: float      # δ_i, с
    delay: float            # d_i = d̄ + Δd_i, с
    ts_err: float           # e_i, с
    noise_sigma: float      # СКО аддитивного шума, о. е.


@dataclass
class Record:
    """Запись модуля: метки отсчётов (известные прибору) и отсчёты трёх фаз."""
    T: np.ndarray           # (n,), с
    x: np.ndarray           # (3, n), о. е.
    T_avail: np.ndarray = None   # моменты поступления отсчётов в контроллер (исходные метки), с;
                                 # поправки модели времени их не сдвигают; None — совпадают с T

    def shifted(self, dt):
        """Та же запись с метками, сдвинутыми на −dt (применение поправки)."""
        return replace(self, T=self.T - dt)


@dataclass
class Device:
    paths: tuple                     # (ModulePath, ModulePath)
    dbar: float                      # фактическая общая задержка d̄, с
    dbar_model: float                # общая задержка, учтённая моделью времени, с
    cal: dict = field(default_factory=dict)   # поправки по оценивателям: имя → c, с


def noise_sigma(snr_db=C.SNR_DB, amp=C.U_AMP):
    """СКО шума по отношению сигнал/шум к действующему значению основной гармоники."""
    return amp / np.sqrt(2.0) * 10.0 ** (-snr_db / 20.0)


def make_device(rng, dbar=C.DBAR, dbar_model=None, dd_max=C.DELTA_D_MAX, snr_db=C.SNR_DB):
    """Случайный экземпляр прибора: смещения сеток, задержки трактов, погрешности меток."""
    sig = noise_sigma(snr_db)
    paths = tuple(
        ModulePath(grid_offset=float(rng.uniform(0.0, 1.0 / C.FS)),
                   delay=float(dbar + rng.uniform(-dd_max, dd_max)),
                   ts_err=float(rng.uniform(-C.TS_ERR_MAX, C.TS_ERR_MAX)),
                   noise_sigma=sig)
        for _ in range(2))
    return Device(paths=paths, dbar=dbar, dbar_model=dbar if dbar_model is None else dbar_model)


def acquire(source: Source, path: ModulePath, duration, rng, t_start=0.0):
    """Запись модуля: отсчёты сигнала source через тракт path."""
    n = int(round(duration * C.FS))
    t_true = t_start + np.arange(n) / C.FS + path.grid_offset
    x = source.waveform(t_true - path.delay)
    if path.noise_sigma > 0.0:
        x = x + rng.normal(0.0, path.noise_sigma, size=x.shape)
    T = t_true + path.ts_err
    return Record(T=T, x=x, T_avail=T)


def acquire_pair(dev: Device, src1: Source, src2: Source, duration, rng):
    """Записи обоих модулей: src1 → модуль 1, src2 → модуль 2."""
    return (acquire(src1, dev.paths[0], duration, rng),
            acquire(src2, dev.paths[1], duration, rng))


def calibration_source(rng):
    """Общий чистый симметричный сигнал 50 Гц для калибровки межтрактовой задержки."""
    return Source(f=C.F0, phi=float(rng.uniform(0.0, TWO_PI)), U=C.U_AMP)


def calibrate(dev: Device, estimators, rng):
    """Калибровка: каждый оцениватель определяет свою поправку c на общем сигнале 50 Гц.

    c = (круговое среднее Δφ̂)/(2π·50 Гц): угол переводится во время и хранится как время.
    """
    src = calibration_source(rng)
    r1, r2 = acquire_pair(dev, src, src, C.CAL_DURATION, rng)
    t_lo, t_hi = 0.25 * C.CAL_DURATION, 0.85 * C.CAL_DURATION
    for est in estimators:
        _, dphi, _ = est.estimate(r1, r2, t_lo, t_hi)
        mean_angle = np.angle(np.mean(np.exp(1j * dphi)))
        dev.cal[est.name] = float(mean_angle / (TWO_PI * C.F0))
    return dev


def apply_time_model(dev: Device, est_name, r1: Record, r2: Record):
    """Применение модели времени прибора: поправка калибровки и учтённая общая задержка."""
    c = dev.cal.get(est_name, 0.0)
    return r1.shifted(dev.dbar_model), r2.shifted(c + dev.dbar_model)
