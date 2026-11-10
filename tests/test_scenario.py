"""Szenario: Umkehrfunktion der Normalverteilung, Grundrate, Erzeuger der Welt (Handwerte, Eigenschaften, Zeitumrechnung)."""

import math

import numpy as np
import pytest

import imt_scenario as S


def test_normal_ppf_bekannte_werte_und_raender():
    q = S.normal_ppf([0.5, 0.975, 0.025, 0.8413447460685429])
    assert q.tolist() == pytest.approx([0.0, 1.959963984540054, -1.959963984540054, 1.0], abs=1e-12)
    r = S.normal_ppf([0.0, 1.0])
    assert r[0] == -math.inf and r[1] == math.inf
    # Rundlauf über die Verteilungsfunktion bis in den fernen Rand
    p = np.array([1e-300, 1e-100, 1e-20, 1e-8, 1e-3, 0.02425, 0.3, 0.5])
    assert S.normal_cdf_lower(S.normal_ppf(p)) == pytest.approx(p, rel=1e-12)


def test_grundrate_handwerte():
    # Median 4: S0(4) = ½, also H0(4) = ln 2; H0(4·e^(0,9·1,959964)) = −ln 0,025
    assert float(S.baseline_cumhazard(4.0)[()]) == pytest.approx(math.log(2.0), abs=1e-14)
    t = 4.0 * math.exp(0.9 * 1.959963984540054)
    assert float(S.baseline_cumhazard(t)[()]) == pytest.approx(-math.log(0.025), abs=1e-10)


def test_grundrate_und_umkehrung_sind_inverse():
    t = np.array([1e-3, 0.1, 1.0, 4.0, 10.0, 50.0, 200.0])
    assert S.baseline_time(S.baseline_cumhazard(t)) == pytest.approx(t, rel=1e-9)
    h = np.array([1e-9, 1e-4, 0.1, 0.5, 0.7, 2.0, 12.0])
    assert S.baseline_cumhazard(S.baseline_time(h)) == pytest.approx(h, rel=1e-9)


def test_zeitumrechnung_von_hand():
    """Ein Container mit H0(S) = ln 2 (S = 4 Tage = Median) und E = ln 2 + 1: ohne Wirkung H0(T) = E, mit β = ln 2 wird der Rest durch 2 geteilt: H0(T) = ln 2 + 1/2."""
    h_start = math.log(2.0)
    e = h_start + 1.0
    plain = S.build_dataset([4.0], [e], 0.0)
    assert float(S.baseline_cumhazard(plain.time)[0]) == pytest.approx(e, abs=1e-9)
    treated = S.build_dataset([4.0], [e], math.log(2.0))
    assert float(S.baseline_cumhazard(treated.time)[0]) == pytest.approx(h_start + 0.5, abs=1e-9)
    # ein Container, der vor dem Beginn ausfällt (E < H0(S)), ist von der Wirkung unberührt
    early = S.build_dataset([4.0], [0.3], -1.0)
    assert float(S.baseline_cumhazard(early.time)[0]) == pytest.approx(0.3, abs=1e-12)
    assert early.time[0] < 4.0


def test_ohne_wirkung_haengt_die_lebenszeit_nicht_am_behandlungsbeginn():
    e = np.array([0.2, 1.0, 3.0])
    a = S.build_dataset([0.5, 2.0, 9.0], e, 0.0)
    b = S.build_dataset([50.0, 0.1, 1.0], e, 0.0)
    assert a.time == pytest.approx(b.time, rel=1e-12)


def test_gleicher_zufalls_seed_gleiche_daten_und_verschiedene_laeufe_verschieden():
    w = S.World(beta=-0.5, wait=4.0, n=200, seed=7)
    a, b, c = S.simulate(w, 3), S.simulate(w, 3), S.simulate(w, 4)
    assert np.array_equal(a.time, b.time) and np.array_equal(a.start, b.start)
    assert not np.array_equal(a.time, c.time)


def test_gepaarte_welten_gleiche_zufallszahlen_andere_wartezeit():
    a = S.simulate(S.World(wait=2.0, n=50, seed=1), 0)
    b = S.simulate(S.World(wait=8.0, n=50, seed=1), 0)
    assert b.start == pytest.approx(4.0 * a.start, rel=1e-12)


def test_werte_sind_positiv_und_endlich_in_allen_reglerecken():
    for beta in (-1.0, 0.0, 0.5):
        for wait in (0.5, 16.0):
            ds = S.simulate(S.World(beta=beta, wait=wait, n=2000, seed=11), 0)
            assert np.all(np.isfinite(ds.time)) and ds.time.min() > 0 and ds.start.min() > 0


def test_naiver_status():
    ds = S.Dataset(np.array([2.0, 3.0, 4.0]), np.array([1.0, 5.0, 4.0]))
    assert S.treated_ever(ds).tolist() == [True, False, False]   # S = T zählt nicht als behandelt
