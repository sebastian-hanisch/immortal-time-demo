"""Szenario: die Welt der Demo und ihr Erzeuger (nur numpy und Standardbibliothek).

Jeder Container hat eine Lebenszeit T (Tage bis zum Ausfall) und einen Behandlungsbeginn S (Tage nach der Ankunft), den er auch dann „hat“, wenn er vorher ausfällt
(dann wird er nie behandelt). Ohne Behandlung ist die Grundrate lognormal (Median 4 Tage, σ = 0,9). Mit Behandlung gilt ab S die Rate mal e^β, vorher die Grundrate.
Der Erzeuger arbeitet auf der Skala der kumulierten Grundrate H0: Ein Container fällt aus, wenn sein Verbrauch an Grundrate den Wert E (Exponential mit Mittel 1)
erreicht; ab S verbraucht er ihn e^β-mal so schnell. Daraus folgt die Zeitumrechnung H0(T) = H0(S) + (E − H0(S))·e^(−β) für Container, die S erleben (E ≥ H0(S)),
sonst H0(T) = E. Die Umkehrung von H0 braucht die Umkehrfunktion der Normalverteilung; sie steht hier (`normal_ppf`), gegen scipy nur in tests/ geprüft.
"""

import math
from dataclasses import dataclass

import numpy as np

import imt_constants as C
from imt_core import SplitMix64

_ERFC = np.vectorize(math.erfc, otypes=[float])

# Koeffizienten der rationalen Näherung von P. J. Acklam (relativer Fehler unter 1,2e-9), danach Halley-Schritte bis zur Rechengenauigkeit.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_P_LOW = 0.02425


def normal_cdf_lower(x):
    """Φ(x) = ½ erfc(−x/√2), in beiden Rändern auf volle Genauigkeit (kein „1 + erf“ mit Auslöschung)."""
    return 0.5 * _ERFC(-np.asarray(x, dtype=float) / math.sqrt(2.0))


def _ppf_below_half(p):
    """Umkehrfunktion der Standardnormalverteilung für 0 < p ≤ 0,5 (Werte ≤ 0)."""
    p = np.asarray(p, dtype=float)
    x = np.empty_like(p)
    low = p < _P_LOW
    q = np.sqrt(-2.0 * np.log(p[low]))
    x[low] = ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))
    mid = ~low
    q = p[mid] - 0.5
    r = q * q
    x[mid] = ((((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q) / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    for _ in range(3):   # Halley: drei Schritte machen aus 1e-9 die Rechengenauigkeit
        e = normal_cdf_lower(x) - p
        u = e * math.sqrt(2.0 * math.pi) * np.exp(0.5 * x * x)
        x = x - u / (1.0 + 0.5 * x * u)
    return x


def normal_ppf(p):
    """Quantilfunktion der Standardnormalverteilung (Umkehrfunktion von Φ) für 0 < p < 1; ± unendlich an den Rändern 0 und 1."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    out = np.empty_like(p)
    out[p <= 0.0] = -np.inf
    out[p >= 1.0] = np.inf
    inner = (p > 0.0) & (p < 1.0)
    below = inner & (p <= 0.5)
    above = inner & (p > 0.5)
    if below.any():
        out[below] = _ppf_below_half(p[below])
    if above.any():
        out[above] = -_ppf_below_half(1.0 - p[above])
    return out


def baseline_cumhazard(t, median: float = C.MEDIAN, sigma: float = C.SIGMA):
    """Kumulierte Grundrate H0(t) = −ln S0(t) der Lognormalverteilung (Median, σ der Log-Skala), für t > 0."""
    t = np.asarray(t, dtype=float)
    z = (np.log(t) - math.log(median)) / sigma
    lower = 0.5 * _ERFC(np.abs(z) / math.sqrt(2.0))          # Φ(−|z|)
    return np.where(z >= 0.0, -np.log(lower), -np.log1p(-lower))


def baseline_time(h, median: float = C.MEDIAN, sigma: float = C.SIGMA):
    """Umkehrung von `baseline_cumhazard`: die Zeit t mit H0(t) = h (h ≥ 0)."""
    h = np.atleast_1d(np.asarray(h, dtype=float))
    surv = np.exp(-h)
    fail = -np.expm1(-h)                                      # 1 − e^(−h), ohne Auslöschung bei kleinem h
    late = surv <= 0.5                                        # h ≥ ln 2: Φ(−z) = e^(−h) liegt im unteren Rand und ist dort genau
    z = np.empty_like(h)
    z[late] = -normal_ppf(np.maximum(surv[late], 1e-300))
    z[~late] = normal_ppf(np.maximum(fail[~late], 1e-300))
    return median * np.exp(sigma * z)


@dataclass(frozen=True)
class World:
    beta: float = C.DEFAULT_BETA     # wahres log-Hazard-Verhältnis der Behandlung (0 = keine Wirkung)
    wait: float = C.DEFAULT_WAIT     # mittlere Wartezeit bis zum Behandlungsbeginn in Tagen
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Dataset:
    time: np.ndarray    # T: Lebenszeit bis zum Ausfall (alle Ausfälle beobachtet)
    start: np.ndarray   # S: Behandlungsbeginn; ein Container mit S ≥ T wird nie behandelt

    @property
    def n(self) -> int:
        return int(self.time.size)


def run_seed(seed: int, run: int) -> int:
    """Zufalls-Seed des Laufs `run` aus dem Zufalls-Seed der Welt (einmal durch den Mischer, damit benachbarte Läufe unabhängig wirken)."""
    return int(SplitMix64(int(seed) * 1_000_003 + int(run)).u64(1)[0])


def simulate(world: World, run: int = 0) -> Dataset:
    """Ein Datensatz der Welt. Dieselben Zufallszahlen (S als wait mal Exp(1), E) erzeugen jede Wartezeit und jede Wirkung: Läufe verschiedener Welten sind gepaart."""
    rng = SplitMix64(run_seed(world.seed, run))
    z_s = rng.exponential(world.n)
    e = np.maximum(rng.exponential(world.n), 1e-300)
    return build_dataset(z_s * world.wait, e, world.beta)


def build_dataset(start, e, beta: float) -> Dataset:
    """Aus Behandlungsbeginn S und Verbrauch E (Exponential, Mittel 1) die Lebenszeit T: H0(T) = E vor dem Beginn, danach läuft die Rate mal e^β."""
    start = np.maximum(np.asarray(start, dtype=float), 1e-9)
    e = np.asarray(e, dtype=float)
    h_start = baseline_cumhazard(start)
    h_total = np.where(e < h_start, e, h_start + (e - h_start) * math.exp(-beta))
    return Dataset(baseline_time(h_total), start)


def treated_ever(ds: Dataset) -> np.ndarray:
    """Naiver Status: „behandelt“, wenn der Behandlungsbeginn vor dem Ausfall liegt (die Zeit bis dahin zählt dann fälschlich als behandelt)."""
    return ds.start < ds.time
