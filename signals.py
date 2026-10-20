"""Сигналы: трёхфазная система напряжений источника.

Фазные напряжения источника (p = 0, 1, 2 — фазы A, B, C):

    u_p(t) = U·[cos(ωt + φ − p·2π/3)                      прямая последовательность
               + K2·cos(ωt + φ + ψ2 + p·2π/3)               обратная последовательность
               + K0·cos(ωt + φ + ψ0)]                       нулевая последовательность
             + Σ_h U·k_h·cos(h·(ωt + φ − p·2π/3) + θ_h)     высшие гармоники

Гармоники симметричны и имеют естественное чередование (5-я — обратная,
7-я — прямая последовательность и т. д.).

«Истинные» фазы основной гармоники:
    θ_+(t) = ωt + φ                                       прямая последовательность
    θ_p(t) = ωt + φ − p·2π/3 + arg(1 + K2·e^{j(ψ2 + p·4π/3)} + K0·e^{j(ψ0 + p·2π/3)})   фаза p
(второе выражение получено из фазоров, см. phase_phasors()).
"""
from dataclasses import dataclass, field
import numpy as np

TWO_PI = 2.0 * np.pi
A_OP = np.exp(1j * TWO_PI / 3.0)          # оператор поворота a = e^{j120°}


@dataclass
class Source:
    """Трёхфазный источник."""
    f: float                    # частота, Гц
    phi: float                  # начальная фаза прямой последовательности при t = 0, рад
    U: float = 1.0              # амплитуда прямой последовательности, о. е.
    k2: float = 0.0             # коэффициент несимметрии по обратной последовательности, о. е.
    psi2: float = 0.0           # угол обратной последовательности относительно прямой, рад
    k0: float = 0.0             # коэффициент несимметрии по нулевой последовательности, о. е.
    psi0: float = 0.0           # угол нулевой последовательности относительно прямой, рад
    harmonics: tuple = field(default_factory=tuple)   # ((h, k_h, θ_h), ...)

    # ---------------------------------------------------------------- фазоры
    def phase_phasors(self):
        """Комплексные амплитуды основной гармоники фаз A, B, C при t = 0."""
        p = np.arange(3)
        pos = np.exp(1j * (self.phi - p * TWO_PI / 3.0))
        neg = self.k2 * np.exp(1j * (self.phi + self.psi2 + p * TWO_PI / 3.0))
        zer = self.k0 * np.exp(1j * (self.phi + self.psi0)) * np.ones(3)
        return self.U * (pos + neg + zer)

    def theta_pos(self, t):
        """Истинная фаза прямой последовательности основной гармоники, рад."""
        return TWO_PI * self.f * np.asarray(t) + self.phi

    def theta_phase(self, t, p=0):
        """Истинная фаза основной гармоники фазы p (0 — A), рад."""
        ph = self.phase_phasors()[p]
        return TWO_PI * self.f * np.asarray(t) + np.angle(ph)

    def theta_line(self, t, p=0, q=1):
        """Истинная фаза основной гармоники линейного напряжения u_p − u_q, рад."""
        ph = self.phase_phasors()
        return TWO_PI * self.f * np.asarray(t) + np.angle(ph[p] - ph[q])

    # ---------------------------------------------------------------- мгновенные значения
    def waveform(self, t):
        """Мгновенные фазные напряжения, массив формы (3, len(t))."""
        t = np.asarray(t, dtype=float)
        wt = TWO_PI * self.f * t
        ph = self.phase_phasors()                       # основная гармоника всех трёх фаз
        x = np.real(ph[:, None] * np.exp(1j * wt)[None, :])
        for h, kh, th in self.harmonics:
            if kh == 0.0:
                continue
            for p in range(3):
                x[p] += self.U * kh * np.cos(h * (wt + self.phi - p * TWO_PI / 3.0) + th)
        return x

    def thd(self):
        return float(np.sqrt(sum(kh ** 2 for _, kh, _ in self.harmonics)))


def wrap_deg(a):
    """Приведение угла к интервалу [−180°, 180°)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def wrap_rad(a):
    return (np.asarray(a) + np.pi) % TWO_PI - np.pi


def random_harmonics(rng, limits, thd_limit):
    """Случайный гармонический состав в пределах норм.

    Амплитуды k_h ~ U(0, предел_h), фазы θ_h ~ U(0, 2π). Если суммарный
    коэффициент превышает thd_limit, все амплитуды пропорционально уменьшаются
    до thd_limit (так соблюдаются одновременно нормы на отдельные гармоники и на сумму).
    """
    orders = sorted(limits)
    k = np.array([rng.uniform(0.0, limits[h]) for h in orders])
    thd = np.sqrt(np.sum(k ** 2))
    if thd > thd_limit:
        k *= thd_limit / thd
    th = rng.uniform(0.0, TWO_PI, size=len(orders))
    return tuple((h, float(kh), float(t)) for h, kh, t in zip(orders, k, th))


def random_unbalance(rng, k2_max, k0_max):
    """Случайная несимметрия: K2U ~ U(0, k2_max), K0U ~ U(0, k0_max), углы ~ U(0, 2π)."""
    return dict(k2=float(rng.uniform(0.0, k2_max)), psi2=float(rng.uniform(0.0, TWO_PI)),
                k0=float(rng.uniform(0.0, k0_max)), psi0=float(rng.uniform(0.0, TWO_PI)))
