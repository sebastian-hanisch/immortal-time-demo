"""Eigenschaftstests (metamorph): Aussagen über das Verfahren, die ohne bekanntes richtiges Ergebnis gelten (ORACLE-PLAYBOOK §5.2).

Jede Eigenschaft nennt ihre Voraussetzung: Cox ist invariant gegen monotone Zeittransformationen und Umsortieren; die Landmarke bei τ = 0 mit Behandlungsbeginn 0 ist dieselbe
Rechnung wie die naive Einteilung; ohne Wirkung hängt die Lebenszeit nicht am Behandlungsbeginn."""

import numpy as np
import pytest

import imt_model as M
import imt_scenario as S
from measure_immortal_time import simulate_upfront


def _datasets(k=12, beta=-0.5, n=300, seed=21):
    return [S.simulate(S.World(beta=beta, n=n, seed=seed), r) for r in range(k)]


def test_zeitskala_aendern_aendert_beta_nicht():
    """Alle Zeiten (T und S) mit c > 0 multipliziert: Rangfolge der Ereignisse und Zeilenordnung bleiben, β̂ bleibt (naiv und zeitabhängig; die Landmarke mit mitskaliertem τ)."""
    for ds in _datasets():
        scaled = S.Dataset(ds.time * 3.7, ds.start * 3.7)
        assert M.estimate_naive(scaled)["beta"] == pytest.approx(M.estimate_naive(ds)["beta"], abs=1e-9)
        assert M.estimate_timedep(scaled)["beta"] == pytest.approx(M.estimate_timedep(ds)["beta"], abs=1e-9)
        assert M.estimate_landmark(scaled, 2.0 * 3.7)["beta"] == pytest.approx(M.estimate_landmark(ds, 2.0)["beta"], abs=1e-9)


def test_monotone_zeittransformation_aendert_beta_nicht():
    for ds in _datasets(8):
        f = lambda t: np.log1p(t) ** 1.5   # streng monoton  # noqa: E731
        t2 = S.Dataset(f(ds.time), f(ds.start))
        assert M.estimate_naive(t2)["beta"] == pytest.approx(M.estimate_naive(ds)["beta"], abs=1e-8)
        assert M.estimate_timedep(t2)["beta"] == pytest.approx(M.estimate_timedep(ds)["beta"], abs=1e-8)


def test_umsortieren_der_container_aendert_nichts():
    rng = np.random.default_rng(4)
    for ds in _datasets(8):
        p = rng.permutation(ds.n)
        shuffled = S.Dataset(ds.time[p], ds.start[p])
        assert M.estimate_naive(shuffled)["beta"] == pytest.approx(M.estimate_naive(ds)["beta"], abs=1e-9)
        assert M.estimate_timedep(shuffled)["beta"] == pytest.approx(M.estimate_timedep(ds)["beta"], abs=1e-9)
        assert M.estimate_landmark(shuffled, 3.0)["beta"] == pytest.approx(M.estimate_landmark(ds, 3.0)["beta"], abs=1e-9)


def test_vertauschte_status_kehren_das_vorzeichen_um():
    """Cox mit 0/1-Merkmal: Status vertauscht (behandelt <-> unbehandelt) kehrt β̂ um (Landmarke, Zeilen mit festem Status)."""
    for ds in _datasets(8):
        start, stop, ev, x = M.rows_landmark(ds, 2.0)
        a = M.cox_binary(start, stop, ev, x)["beta"]
        b = M.cox_binary(start, stop, ev, ~x)["beta"]
        assert a == pytest.approx(-b, abs=1e-8)


def test_landmarke_bei_tau_null_mit_beginn_null_ist_die_naive_einteilung():
    """Wo jeder Behandelte am Tag 0 beginnt (Vorab-Zuteilung), ist `Landmarke τ ≈ 0` dieselbe Rechnung wie die naive Einteilung (gleiche Zeilen, gleicher Status)."""
    for r in range(10):
        ds = simulate_upfront(300, 5, r)
        assert M.estimate_landmark(ds, 1e-6)["beta"] == pytest.approx(M.estimate_naive(ds)["beta"], abs=1e-12)


def test_ohne_wartezeit_verschwindet_die_scheinwirkung_aber_nicht_bei_kurzer_wartezeit():
    """Vorab-Zuteilung: naiv ≈ 0 (Mittel über 40 Läufe innerhalb von 3 Standardfehlern); kurze Wartezeit im Erzeuger (Mittel 0,5 Tage, nie Vorab): naiv bleibt stark negativ."""
    up = np.array([M.estimate_naive(simulate_upfront(500, 8, r))["beta"] for r in range(40)])
    assert abs(up.mean()) < 3 * up.std(ddof=1) / np.sqrt(up.size)
    short = np.array([M.estimate_naive(S.simulate(S.World(beta=0.0, wait=0.5, n=500, seed=8), r))["beta"] for r in range(40)])
    assert short.max() < -1.5


def test_mehr_container_verringern_die_streuung_der_schaetzer():
    sd = {}
    for n in (200, 800):
        vals = [M.estimate_timedep(S.simulate(S.World(beta=-0.5, n=n, seed=3), r))["beta"] for r in range(40)]
        sd[n] = float(np.std(vals, ddof=1))
    assert sd[800] < sd[200]                                                         # Zielverhältnis 1/2 bei vierfachem n; hier nur die Richtung


def test_staerkere_wirkung_gibt_kleineres_beta_bei_jeder_auswertung():
    """Gepaarte Welten (gleiche Zufallszahlen): ein kleineres wahres β gibt in der Summe über 30 Läufe ein kleineres β̂, für alle drei Auswertungen."""
    def mean(beta, fn):
        return float(np.mean([fn(S.simulate(S.World(beta=beta, n=300, seed=6), r)) for r in range(30)]))
    for fn in (lambda d: M.estimate_naive(d)["beta"], lambda d: M.estimate_landmark(d, 2.0)["beta"], lambda d: M.estimate_timedep(d)["beta"]):
        assert mean(-0.8, fn) < mean(0.0, fn) < mean(0.4, fn)


def test_gleicher_zufalls_seed_gleiche_studie():
    w = S.World(beta=-0.5, n=300, seed=99)
    a, b = M.run_study(w, 5), M.run_study(w, 5)
    assert np.array_equal(a.timedep, b.timedep) and np.array_equal(a.lm_remaining, b.lm_remaining)
