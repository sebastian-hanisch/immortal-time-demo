"""Orakel-Tests der Demo (unabhängige Rechenwege, DEMO-PLAYBOOK §12). Scipy nur hier (nicht in requirements.txt).

1. Die drei Auswertungen gegen eine Schleifen-Cox aus den ROHDATEN (T, S): Der Status zur Ausfallzeit wird je Container direkt aus S und T bestimmt, es werden keine Zeilen
   aufgeteilt und keine Zählungen sortiert; die Nullstelle des Scores kommt von `scipy.optimize.brentq` (kein Newton-Verfahren), der Standardfehler aus einer numerischen
   Differenz des Scores. Das ist die „aufgeteilte Person-Zeit-Zählung“ in einer zweiten, unabhängigen Fassung.
2. Der Erzeuger: Umkehrfunktion und Grundrate gegen scipy; der Rückweg T -> E (Zeitumrechnung rückwärts) ergibt Exp(1) (Kolmogorov-Smirnov), mit und ohne Wirkung.
3. Simulation mit bekannter Wahrheit als Band: ohne Wirkung naiv rund −1,28, Landmarke und zeitabhängig bei 0; mit Wirkung zeitabhängig bei der Wahrheit, Landmarke
   mit richtigem Vorzeichen und zu schwach; die Vorab-Zuteilung (kein Warten) hat keine Scheinwirkung; die zensierte Landmarke findet die Wirkung.
"""

import math

import numpy as np
import pytest

import imt_constants as C
import imt_model as M
import imt_scenario as S
from measure_immortal_time import simulate_upfront

sp = pytest.importorskip("scipy.optimize")
stats = pytest.importorskip("scipy.stats")


# ------------------------------------------------------------------------------------------------ 1. Schleifen-Cox aus den Rohdaten


def _cox_from_raw(times, status_at, include):
    """Breslow-Cox mit Schleifen. `include[i]`: Container i zählt mit; `status_at(i, t)`: 0/1-Status des Containers i zur Ausfallzeit t.
    Risikomenge zur Zeit t: alle eingeschlossenen Container mit Ausfallzeit ≥ t. Nullstelle des Scores mit brentq, Standardfehler aus der Differenz des Scores."""
    idx = [i for i in range(len(times)) if include[i]]
    events = [(times[i], status_at(i, times[i])) for i in idx]

    def parts(beta):
        score = info = 0.0
        for t, x_event in events:
            risk = [status_at(j, t) for j in idx if times[j] >= t]
            w = [math.exp(beta * x) for x in risk]
            m = sum(wi * x for wi, x in zip(w, risk)) / sum(w)
            score += x_event - m
            info += sum(wi * x * x for wi, x in zip(w, risk)) / sum(w) - m * m
        return score, info

    return parts


def _solve(parts):
    try:
        beta = sp.brentq(lambda b: parts(b)[0], -12.0, 12.0, xtol=1e-12)
    except ValueError:
        return None
    info = parts(beta)[1]
    return None if info <= 1e-12 else (beta, 1.0 / math.sqrt(info))


def _oracle_naive(T, Sx):
    return _solve(_cox_from_raw(T, lambda i, t: 1 if Sx[i] < T[i] else 0, [True] * len(T)))


def _oracle_landmark(T, Sx, tau):
    return _solve(_cox_from_raw(T, lambda i, t: 1 if Sx[i] <= tau else 0, [T[i] > tau for i in range(len(T))]))


def _oracle_timedep(T, Sx):
    return _solve(_cox_from_raw(T, lambda i, t: 1 if Sx[i] < t else 0, [True] * len(T)))


def _random_dataset(rng, n, ties):
    T = np.round(rng.uniform(0.5, 9.0, n), 1 if ties else 4)
    Sx = np.where(rng.uniform(size=n) < 0.8, np.round(rng.uniform(0.1, 8.0, n), 1 if ties else 4), 1e9)
    return np.maximum(T, 0.1), np.maximum(Sx, 0.1)


@pytest.mark.parametrize("ties", [False, True])
def test_drei_auswertungen_gegen_schleifen_cox_aus_den_rohdaten(ties):
    rng = np.random.default_rng(17 + int(ties))
    compared = 0
    for _ in range(80):
        n = int(rng.integers(10, 30))
        T, Sx = _random_dataset(rng, n, ties)
        ds = S.Dataset(T, Sx)
        tau = float(rng.choice([1.0, 2.0, 3.5]))
        for ours, theirs in ((M.estimate_naive(ds), _oracle_naive(T, Sx)), (M.estimate_landmark(ds, tau), _oracle_landmark(T, Sx, tau)), (M.estimate_timedep(ds), _oracle_timedep(T, Sx))):
            if theirs is None or not ours["ok"]:
                assert (theirs is None) == (not ours["ok"]), (ours, theirs)   # beide ohne endlichen Schätzer (Trennung) oder beide mit
                continue
            assert ours["beta"] == pytest.approx(theirs[0], abs=1e-6) and ours["se"] == pytest.approx(theirs[1], abs=1e-6)
            compared += 1
    assert compared > 150


def test_zeitabhaengig_stimmt_auch_mit_der_kopierten_cox_fit_auf_aufgeteilten_zeilen():
    rng = np.random.default_rng(5)
    T, Sx = _random_dataset(rng, 25, True)
    ds = S.Dataset(T, Sx)
    a, b = M.estimate_timedep(ds), M.estimate_timedep(ds, fast=False)
    assert a["beta"] == pytest.approx(b["beta"], abs=1e-8) and a["se"] == pytest.approx(_oracle_timedep(T, Sx)[1], abs=1e-6)


# ------------------------------------------------------------------------------------------------ 2. Erzeuger gegen scipy


def test_umkehrfunktion_und_grundrate_gegen_scipy():
    p = np.concatenate([np.logspace(-300, -1, 40), np.linspace(0.1, 0.9, 20), 1 - np.logspace(-12, -1, 10)])
    assert S.normal_ppf(p) == pytest.approx(stats.norm.ppf(p), rel=1e-12, abs=1e-12)
    t = np.array([1e-3, 0.05, 1.0, 4.0, 9.0, 40.0, 300.0])
    assert S.baseline_cumhazard(t) == pytest.approx(-stats.lognorm.logsf(t, C.SIGMA, scale=C.MEDIAN), rel=1e-9, abs=1e-14)
    h = np.array([1e-6, 0.01, 0.5, 0.69, 0.7, 3.0, 12.0])
    assert S.baseline_time(h) == pytest.approx(stats.lognorm.isf(np.exp(-h), C.SIGMA, scale=C.MEDIAN), rel=1e-9)


@pytest.mark.parametrize("beta", [0.0, -0.5, 0.5])
def test_rueckweg_der_zeitumrechnung_ergibt_exponential(beta):
    """Der erzeugte Datensatz (T, S) wird rückwärts in den Verbrauch E' = Λ_i(T_i) umgerechnet (Λ_i: kumulierte Rate des Containers, ab S mit dem Faktor e^β); E' muss Exp(1) sein."""
    ds = S.simulate(S.World(beta=beta, wait=4.0, n=20000, seed=3), 0)
    h_t = S.baseline_cumhazard(ds.time)
    h_s = S.baseline_cumhazard(ds.start)
    e_back = np.where(ds.time <= ds.start, h_t, h_s + math.exp(beta) * (h_t - h_s))
    assert stats.kstest(e_back, "expon").pvalue > 1e-3
    assert abs(e_back.mean() - 1.0) < 0.03


def test_ohne_wirkung_ist_die_lebenszeit_lognormal_unabhaengig_vom_beginn():
    ds = S.simulate(S.World(beta=0.0, wait=4.0, n=20000, seed=3), 0)
    assert stats.kstest(ds.time, "lognorm", args=(C.SIGMA, 0, C.MEDIAN)).pvalue > 1e-3
    assert abs(stats.spearmanr(ds.time, ds.start)[0]) < 0.03                         # T und S sind unabhängig (β = 0)


# ------------------------------------------------------------------------------------------------ 3. Simulation mit bekannter Wahrheit


@pytest.fixture(scope="module")
def null_study():
    return M.run_study(S.World(beta=0.0), C.REPS)


@pytest.fixture(scope="module")
def effect_study():
    return M.run_study(S.World(beta=-0.5), C.REPS)


def test_ohne_wirkung_naiv_findet_scheinschutz_landmarke_und_zeitabhaengig_nicht(null_study):
    naive = M.summarize_naive(null_study)
    assert -1.31 < naive.mean < -1.25                                              # Eigenmessung rund −1,28 (Vormessung −1,274)
    for s in (M.summarize_landmark(null_study, 2.0), M.summarize_landmark(null_study, 4.0), M.summarize_timedep(null_study)):
        assert abs(s.mean) < 3 * s.se_mean and s.failed == 0


def test_mit_wirkung_zeitabhaengig_trifft_landmarke_ist_zu_schwach_naiv_uebertreibt(effect_study):
    td = M.summarize_timedep(effect_study)
    assert abs(td.mean + 0.5) < 3 * td.se_mean
    for tau in (2.0, 4.0):
        lm = M.summarize_landmark(effect_study, tau)
        assert lm.mean < -2 * lm.se_mean and lm.mean > -0.5 * 0.5                  # richtiges Vorzeichen, weniger als die Hälfte der Wirkung
    assert M.summarize_naive(effect_study).mean < -1.5


def test_vorab_zuteilung_am_tag_0_hat_keine_scheinwirkung():
    vals = [M.estimate_naive(simulate_upfront(C.DEFAULT_N, 35, r)) for r in range(60)]
    b = np.array([v["beta"] for v in vals])
    assert all(v["ok"] for v in vals) and abs(b.mean()) < 3 * b.std(ddof=1) / math.sqrt(len(b))


def test_landmarke_mit_zensur_bei_spaeterem_beginn_findet_die_wirkung():
    betas = []
    for r in range(60):
        ds = S.simulate(S.World(beta=-0.5), r)
        fit = M.fit_rows(M.rows_landmark_censored(ds, 2.0))
        assert fit["ok"]
        betas.append(fit["beta"])
    betas = np.array(betas)
    assert abs(betas.mean() + 0.5) < 3 * betas.std(ddof=1) / math.sqrt(len(betas))
