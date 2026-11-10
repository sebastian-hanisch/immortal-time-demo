"""Randfälle der Auswertungen: Status statt stiller Zahlen (n = 1, n = 2, nur eine Gruppe, τ jenseits der Zeiten, Gleichstände, Behandlung nach dem Ausfall)."""

import numpy as np
import pytest

import imt_model as M
import imt_scenario as S


def _ds(times, starts):
    return S.Dataset(np.array(times, float), np.array(starts, float))


def test_ein_und_zwei_container_geben_eine_meldung():
    for ds in (_ds([2.0], [1.0]), _ds([2.0, 3.0], [1.0, 1e9])):
        for est in (M.estimate_naive(ds), M.estimate_timedep(ds), M.estimate_landmark(ds, 0.5)):
            assert est["ok"] is False and est["message"]


def test_behandlung_nach_dem_ausfall_zaehlt_als_nie_behandelt():
    """S ≥ T: nie behandelt; die aufgeteilte Person-Zeit hat dann genau eine Zeile je solchem Container."""
    ds = _ds([2.0, 4.0, 6.0], [2.0, 9.0, 1.0])     # Container 1: S = T (nie), Container 2: S > T (nie), Container 3: S < T
    assert S.treated_ever(ds).tolist() == [False, False, True]
    start, stop, ev, x = M.rows_split(ds)
    assert len(start) == 4 and x.tolist() == [False, False, False, True]


def test_alle_behandelt_gibt_keine_schaetzung():
    ds = _ds([3.0, 4.0, 5.0, 6.0], [0.5, 1.0, 1.5, 2.0])
    r = M.estimate_naive(ds)
    assert r["ok"] is False and "singulär" in r["message"]


def test_niemand_behandelt_gibt_keine_schaetzung():
    ds = _ds([3.0, 4.0, 5.0, 6.0], [9.0, 9.0, 9.0, 9.0])
    assert M.estimate_naive(ds)["ok"] is False and M.estimate_landmark(ds, 1.0)["ok"] is False


def test_landmarke_jenseits_aller_zeiten_und_genau_auf_einer_zeit():
    ds = _ds([2.0, 3.0, 4.0, 6.0, 8.0, 10.0], [1.0, 5.0, 2.5, 1e9, 7.0, 3.0])
    far = M.estimate_landmark(ds, 50.0)
    assert far["ok"] is False and far["remaining"] == 0
    assert M.estimate_landmark(ds, 10.0)["remaining"] == 0                           # T > τ: ein Container mit T = τ fällt weg (nicht inklusive)
    assert M.estimate_landmark(ds, 9.99)["remaining"] == 1


def test_gleichstaende_in_den_zeiten_werden_nach_breslow_behandelt():
    ds = _ds([2.0, 2.0, 3.0, 3.0, 5.0, 5.0, 7.0, 8.0], [1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 9.0, 1.0])
    a, b = M.estimate_naive(ds), M.estimate_naive(ds, fast=False)
    assert a["ok"] and b["ok"] and a["beta"] == pytest.approx(b["beta"], abs=1e-9)


def test_reglerecken_der_welt_laufen_ohne_fehler():
    for beta in (-1.0, 0.5):
        for wait in (0.5, 16.0):
            for n in (100, 2000):
                st = M.run_study(S.World(beta=beta, wait=wait, n=n, seed=1), 3)
                assert st.naive.shape == (3,) and (st.naive_ok.sum() + (~st.naive_ok).sum()) == 3


def test_sehr_kurze_wartezeit_meldet_trennung_statt_zahl():
    """Mittlere Wartezeit 0,1 Tage (unter dem Reglerminimum): fast alle Container zählen als behandelt, die Läufe ohne Ergebnis haben eine Meldung."""
    fails = 0
    for r in range(15):
        fit = M.estimate_naive(S.simulate(S.World(beta=0.0, wait=0.1, n=1000, seed=35), r))
        if not fit["ok"]:
            fails += 1
            assert fit["message"] and np.isnan(fit["se"])
    assert fails > 0
