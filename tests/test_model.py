"""Modell: Handbeispiel in Brüchen, aufgeteilte Person-Zeit, schnelle Cox-Anpassung gegen die kopierte `cox_fit`, Läufe und Zusammenfassungen."""

from fractions import Fraction as F

import numpy as np
import pytest

import imt_model as M
import imt_scenario as S


# ------------------------------------------------------------------------------------------------ Handbeispiel (von Hand gerechnet)


def test_handbeispiel_naiv(six):
    r = M.hand_naive(six)
    assert r["treated"] == [2, 4, 8, 10] and r["untreated"] == [3, 6]
    assert r["mean_treated"] == F(6) and r["mean_untreated"] == F(9, 2)          # (2+4+8+10)/4 = 6; (3+6)/2 = 4,5


def test_handbeispiel_landmarke_3_5(six):
    r = M.hand_landmark(F(7, 2), six)
    assert r["treated"] == [4, 10] and r["untreated"] == [6, 8]                   # Container 3 (S 2,5) und 6 (S 3) bis 3,5 behandelt; 4 und 5 nicht
    assert r["mean_treated"] == F(7) and r["mean_untreated"] == F(7)              # (4+10)/2 = 7; (6+8)/2 = 7


def test_handbeispiel_landmarke_raender(six):
    assert M.hand_landmark(0, six)["treated"] == [] and M.hand_landmark(0, six)["mean_treated"] is None   # bei τ = 0 hat noch niemand begonnen
    assert M.hand_landmark(0, six)["untreated"] == [2, 3, 4, 6, 8, 10]
    assert M.hand_landmark(10, six)["treated"] == [] and M.hand_landmark(10, six)["untreated"] == []      # T > τ fällt für alle weg
    r = M.hand_landmark(7, six)                                                    # S = 7 zählt als „bis τ begonnen“ (S ≤ τ)
    assert r["treated"] == [8, 10]                                                  # Container 5 (T 8, S 7) und 6 (T 10, S 3)


def test_handbeispiel_tabelle(six):
    rows = M.hand_rows(F(7, 2), six)
    assert [r["naive"] for r in rows] == ["behandelt", "unbehandelt", "behandelt", "unbehandelt", "behandelt", "behandelt"]
    assert [r["landmark"] for r in rows] == ["fällt weg", "fällt weg", "behandelt", "unbehandelt", "unbehandelt", "behandelt"]


def test_aufgeteilte_person_zeit_des_handbeispiels(six):
    rows = M.hand_split(six)
    assert len(rows) == 10                                                         # vier Container (1, 3, 5, 6) werden zu zwei Zeilen
    assert rows[:2] == [(1, F(0), F(1), False, False), (1, F(1), F(2), True, True)]
    assert (2, F(0), F(3), False, True) in rows and (4, F(0), F(6), False, True) in rows   # nie behandelt: eine Zeile mit Ausfall
    assert sum(1 for r in rows if r[4]) == 6                                       # jeder Container fällt genau einmal aus
    # Summe der Zeilenlängen je Container = Lebenszeit
    for i, (t, _) in enumerate(six, start=1):
        assert sum(b - a for c, a, b, _, _ in rows if c == i) == t


# ------------------------------------------------------------------------------------------------ Zeilen der Auswertungen


def _tiny():
    return S.Dataset(np.array([2.0, 3.0, 4.0, 6.0, 8.0, 10.0]), np.array([1.0, 5.0, 2.5, 1e9, 7.0, 3.0]))


def test_zeilen_naiv_landmarke_und_aufgeteilt():
    ds = _tiny()
    start, stop, ev, x = M.rows_naive(ds)
    assert start.tolist() == [0.0] * 6 and stop.tolist() == [2, 3, 4, 6, 8, 10] and ev.all()
    assert x.tolist() == [True, False, True, False, True, True]
    start, stop, ev, x = M.rows_landmark(ds, 3.5)
    assert start.tolist() == [3.5] * 4 and stop.tolist() == [4, 6, 8, 10] and x.tolist() == [True, False, False, True]
    start, stop, ev, x = M.rows_split(ds)
    assert len(start) == 10 and int(ev.sum()) == 6
    assert (start[:6].tolist(), stop[:6].tolist()) == ([0.0] * 6, [1.0, 3.0, 2.5, 6.0, 7.0, 3.0])   # erste Zeilen: bis S oder bis T
    assert ev[:6].tolist() == [False, True, False, True, False, False]


def test_zensierte_landmarke_zensiert_spaeter_behandelte_beim_beginn():
    ds = _tiny()
    start, stop, ev, x = M.rows_landmark_censored(ds, 3.5)
    # Container 4 (T 6, nie behandelt) bleibt bis 6, Container 5 (T 8, S 7) wird bei 7 zensiert (kein Ausfall in der Zeile)
    assert stop.tolist() == [4.0, 6.0, 7.0, 10.0] and ev.tolist() == [True, True, False, True] and x.tolist() == [True, False, False, True]


# ------------------------------------------------------------------------------------------------ schnelle Cox-Anpassung gegen die kopierte


def test_cox_binary_handbeispiel_vier_container():
    r = M.cox_binary(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4, bool), np.array([1, 0, 1, 0], bool))
    assert r["ok"] and r["beta"] == pytest.approx(0.9406, abs=5e-4)


def test_cox_binary_stimmt_mit_der_kopierten_cox_fit_ueberein():
    rng = np.random.default_rng(2)
    for k in range(60):
        n = int(rng.integers(6, 40))
        stop = np.round(rng.uniform(1, 9, n), int(rng.integers(0, 3)))   # Gleichstände
        start = np.where(rng.uniform(size=n) < 0.5, 0.0, np.round(rng.uniform(0, 0.9, n) * stop, 1))
        start = np.minimum(start, stop - 0.05)
        ev = rng.uniform(size=n) < 0.8
        x = rng.uniform(size=n) < 0.5
        a = M.fit_rows((start, stop, ev, x), True)
        b = M.fit_rows((start, stop, ev, x), False)
        assert a["ok"] == b["ok"], (k, a, b)
        if a["ok"]:
            assert a["beta"] == pytest.approx(b["beta"], abs=1e-7) and a["se"] == pytest.approx(b["se"], abs=1e-7)


def test_schaetzer_auf_simulierten_daten_schnell_gleich_kern():
    ds = S.simulate(S.World(beta=-0.5, n=300, seed=5), 1)
    for fast_rows in (M.rows_naive(ds), M.rows_landmark(ds, 2.0), M.rows_split(ds)):
        a, b = M.fit_rows(fast_rows, True), M.fit_rows(fast_rows, False)
        assert a["beta"] == pytest.approx(b["beta"], abs=1e-8) and a["se"] == pytest.approx(b["se"], abs=1e-8)


def test_status_statt_stiller_zahlen():
    # alle behandelt: Merkmal konstant
    r = M.cox_binary(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4, bool), np.ones(4, bool))
    assert r["ok"] is False and "singulär" in r["message"]
    # Trennung: alle Behandelten fallen zuerst aus
    r = M.cox_binary(np.zeros(6), np.arange(1.0, 7.0), np.ones(6, bool), np.array([1, 1, 1, 0, 0, 0], bool))
    assert r["ok"] is False and "Trennung" in r["message"] and np.isnan(r["se"])
    # ohne Ereignis
    r = M.cox_binary(np.zeros(3), np.array([1.0, 2.0, 3.0]), np.zeros(3, bool), np.array([1, 0, 1], bool))
    assert r["ok"] is False and "Ereignis" in r["message"]
    # ein einziges Ereignis bei verschiedenem Merkmal: ebenfalls Trennung (wie in der Kernfassung)
    r = M.cox_binary(np.zeros(3), np.array([1.0, 2.0, 3.0]), np.array([1, 0, 0], bool), np.array([1, 0, 0], bool))
    assert r["ok"] is False
    # leere Landmarke: kein Container überlebt τ
    ds = S.Dataset(np.array([1.0, 2.0]), np.array([0.5, 5.0]))
    assert M.estimate_landmark(ds, 5.0)["ok"] is False and M.estimate_landmark(ds, 5.0)["remaining"] == 0
    assert M.estimate_landmark(ds, 5.0, fast=False)["ok"] is False                   # auch über die Kernfassung: Meldung statt Fehler


# ------------------------------------------------------------------------------------------------ Läufe und Zusammenfassungen


def test_zusammenfassung_an_gesetzten_werten():
    s = M.summarize([1.0, 2.0, 3.0, 99.0], [0.5, 0.5, 0.5, 9.0], [True, True, True, False], truth=2.0)
    assert s.mean == pytest.approx(2.0) and s.sd == pytest.approx(1.0) and s.se_mean == pytest.approx(1 / 3 ** 0.5)
    assert s.runs == 3 and s.failed == 1 and s.mean_model_se == pytest.approx(0.5)
    assert s.coverage == pytest.approx(1 / 3)                                       # 1,96 · 0,5 = 0,98: nur β̂ = 2 liegt innerhalb, die Abstände 1 und 1 nicht
    e = M.summarize([1.0, 2.0], [0.1, 0.1], [False, False], truth=0.0)
    assert e.runs == 0 and e.failed == 2 and np.isnan(e.mean)


def test_studie_ist_deterministisch_und_hat_die_erwartete_form():
    w = S.World(beta=-0.5, n=300, seed=9)
    a, b = M.run_study(w, 8), M.run_study(w, 8)
    assert np.array_equal(a.naive, b.naive) and np.array_equal(a.timedep, b.timedep) and np.array_equal(a.lm, b.lm)
    assert a.lm.shape == (8, 2) and a.taus == (2.0, 4.0) and a.naive.shape == (8,)
    assert np.all(a.lm_remaining[:, 0] >= a.lm_remaining[:, 1])                      # τ = 4 lässt weniger Container übrig als τ = 2
    assert np.all(a.lm_treated <= a.lm_remaining)


def test_zusammenfassung_der_landmarke_nutzt_die_richtige_spalte():
    st = M.run_study(S.World(beta=-0.5, n=300, seed=9), 6, taus=(1.0, 3.0))
    assert M.summarize_landmark(st, 3.0).mean == pytest.approx(float(np.mean(st.lm[st.lm_ok[:, 1], 1])))
    assert M.mean_remaining(st, 1.0) == pytest.approx(float(st.lm_remaining[:, 0].mean()))
    rows = M.landmark_sweep(st)
    assert [r["tau"] for r in rows] == [1.0, 3.0] and rows[1]["remaining"] == pytest.approx(M.mean_remaining(st, 3.0))


def test_reihe_ueber_die_wartezeit_hat_je_wartezeit_alle_auswertungen():
    rows = M.wait_sweep(0.0, 200, 4, waits=(1.0, 4.0), reps=5)
    assert [r["wait"] for r in rows] == [1.0, 4.0]
    for r in rows:
        assert set(r["landmarks"]) == {2.0, 4.0} and r["naive"].runs + r["naive"].failed == 5 and 0 < r["treated_share"] < 1
    assert rows[0]["treated_share"] > rows[1]["treated_share"]                       # kürzere Wartezeit: mehr Container erleben den Beginn
