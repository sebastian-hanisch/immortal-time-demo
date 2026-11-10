"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

from fractions import Fraction as F

import numpy as np

import imt_evaluation as ev
import imt_model as M
from imt_scenario import World


def _study(beta, naive, td, lm, remaining=(800, 500), treated=(300, 320), share=0.6, n=1000, taus=(2.0, 4.0), failed_naive=0):
    """Eine Studie mit gesetzten Schätzwerten: je Schätzer zwei Läufe mit Mittel `x` (Werte x ± 0,1, Standardfehler 0,1)."""
    def pair(x):
        return np.array([x - 0.1, x + 0.1])
    ok = np.array([True, True])
    nv_ok = ok.copy()
    if failed_naive:
        nv_ok = np.array([True, False])
    lm_arr = np.stack([pair(v) for v in lm], axis=1)
    return M.Study(
        World(beta=beta, n=n), 2, taus, pair(naive), np.full(2, 0.1), nv_ok, pair(td), np.full(2, 0.1), ok, lm_arr, np.full((2, len(taus)), 0.1), np.ones((2, len(taus)), bool),
        np.tile(np.array(remaining), (2, 1)), np.tile(np.array(treated), (2, 1)), np.full(2, share),
    )


def test_hilfen_zahlen_und_brueche():
    assert ev.num(-1.2776, 2) == "−1,28" and ev.num(0.004, 2) == "0,00" and ev.num(0.5, 1, sign=True) == "+0,5" and ev.num(float("nan")) == "–"
    assert ev.frac_de(F(6)) == "6,0" and ev.frac_de(F(9, 2)) == "4,5" and ev.frac_de(F(1, 4)) == "0,25" and ev.frac_de(None) == "–"
    assert ev.frac_de(F(13, 3)) == "≈ 4,33 (13/3)"
    assert ev.times_de([F(2), F(5, 2)]) == "{2, 2,5}" and ev.times_de([]) == "keine"
    assert ev.hr(-1.2776) == "0,28" and ev.hr(0.0) == "1,00"


def test_verdikt_handbeispiel_standard_nennt_alle_zahlen():
    kind, text = ev.verdict_hand(F(7, 2))
    assert kind == "info"
    for needle in ("behandelte {2, 4, 8, 10} im Mittel 6,0 Tage", "unbehandelte {3, 6} im Mittel 4,5 Tage", "Unterschied 1,5 Tage", "Landmarke τ = 3,5 Tage",
                   "behandelte {4, 10} im Mittel 7,0 Tage", "unbehandelte {6, 8} im Mittel 7,0 Tage", "Unterschied 0,0 Tage", "bewertet beide Gruppen gleich"):
        assert needle in text, needle


def test_verdikt_handbeispiel_andere_landmarke_und_leere_gruppen():
    kind, text = ev.verdict_hand(F(7, 2) + F(1, 2))                               # τ = 4: Container 3 (T = 4) fällt weg; 4, 5, 6 bleiben
    assert kind == "info" and "behandelte {10} im Mittel 10,0 Tage" in text and "unbehandelte {6, 8} im Mittel 7,0 Tage" in text and "Unterschied 3,0 Tage" in text
    kind, text = ev.verdict_hand(0)
    assert kind == "warning" and "keine behandelten Container" in text and "keinen Vergleich" in text
    kind, text = ev.verdict_hand(9)
    assert kind == "warning" and "behandelte {10} im Mittel 10,0 Tage" in text and "keine unbehandelten Container" in text


def test_verdikt_naiv_ohne_wirkung_und_mit_wirkung_und_schaden():
    kind, text = ev.verdict_naive(_study(0.0, -1.28, 0.0, (0.0, 0.0)))
    assert kind == "warning" and "wirkt die Behandlung nicht (β = 0)" in text
    assert "β̂ = −1,28 ± 0,100" in text and "Hazard-Verhältnis 0,28" in text and "Der Schutz ist Schein, 60 % der Container zählen als behandelt" in text
    kind, text = ev.verdict_naive(_study(-0.5, -1.72, -0.5, (-0.2, -0.2)))
    assert kind == "warning" and "die Behandlung die Rate (β = −0,5, Hazard-Verhältnis 0,61)" in text and "mehr Schutz, als die Wahrheit enthält (Wahrheit β = −0,5)" in text
    kind, text = ev.verdict_naive(_study(0.5, -0.87, 0.5, (0.3, 0.3)))
    assert kind == "warning" and "die Behandlung die Rate (β = +0,5, Hazard-Verhältnis 1,65)" in text and "Der Schutz ist Schein" in text
    kind, text = ev.verdict_naive(_study(-0.5, -0.5, -0.5, (-0.2, -0.2)))
    assert kind == "info" and "nicht kleiner als die Wahrheit" in text
    assert "1 Läufen gab es kein Ergebnis" in ev.verdict_naive(_study(0.0, -1.28, 0.0, (0.0, 0.0), failed_naive=1))[1]


def test_verdikt_naiv_ohne_ergebnis():
    st = _study(0.0, -1.0, 0.0, (0.0, 0.0))
    st = M.Study(st.world, 2, st.taus, st.naive, st.naive_se, np.zeros(2, bool), st.timedep, st.timedep_se, st.timedep_ok, st.lm, st.lm_se, st.lm_ok, st.lm_remaining, st.lm_treated, st.treated_share)
    kind, text = ev.verdict_naive(st)
    assert kind == "warning" and "in keinem der 2 Läufe ein Ergebnis" in text


def _sweep_row(wait, naive, share=0.5):
    def s(x):
        return M.Summary(x, 0.1, 0.01, 100, 0, 0.9, 0.1)
    return {"wait": wait, "naive": s(naive), "timedep": s(0.0), "landmarks": {2.0: s(0.0), 4.0: s(0.0)}, "remaining": {2.0: 780, 4.0: 500}, "treated_share": share}


def test_verdikt_reihe_ueber_die_wartezeit():
    rows = [_sweep_row(0.5, -2.75), _sweep_row(1.0, -2.09), _sweep_row(4.0, -1.28), _sweep_row(16.0, -0.93)]
    kind, text = ev.verdict_wait_sweep(rows, 0.0)
    assert kind == "info" and "in 4 von 4 Wartezeiten unter der Wahrheit (β = 0,0)" in text
    assert "am weitesten bei 0,5 Tagen (β̂ = −2,75)" in text and "am wenigsten bei 16,0 Tagen (β̂ = −0,93)" in text and "wird mit längerer Wartezeit kleiner" in text
    rows[2] = _sweep_row(4.0, -2.5)                                               # nicht mehr gleichmäßig
    assert "verläuft nicht gleichmäßig" in ev.verdict_wait_sweep(rows, 0.0)[1]
    assert ev.verdict_wait_sweep(rows[:1], 0.0)[0] == "warning"


def test_verdikt_landmarke_ohne_und_mit_wirkung():
    st0 = _study(0.0, -1.28, 0.0, (0.04, 0.0))
    kind, text = ev.verdict_landmark(st0, 2.0)
    assert kind == "success" and "Es bleiben im Mittel 800 von 1000 Container (20 % fallen weg), davon 300 bis τ behandelt" in text
    assert "keine Scheinwirkung (Abstand zur Null 0,4 Standardfehler" in text
    kind, text = ev.verdict_landmark(_study(0.0, -1.28, 0.0, (0.5, 0.0)), 2.0)
    assert kind == "warning" and "ein Rest von Scheinwirkung" in text
    kind, text = ev.verdict_landmark(_study(-0.5, -1.7, -0.5, (-0.19, -0.2)), 2.0)
    assert kind == "warning" and "findet 38 % der wahren Wirkung (Wahrheit β = −0,5; Verhältnis der Mittel)" in text
    kind, text = ev.verdict_landmark(_study(-0.5, -1.7, -0.5, (-0.19, -0.2)), 4.0)
    assert "Es bleiben im Mittel 500 von 1000 Container (50 % fallen weg)" in text and "findet 40 % der wahren Wirkung" in text


def test_verdikt_zeitabhaengig():
    kind, text = ev.verdict_timedep(_study(-0.5, -1.7, -0.5, (-0.2, -0.2)))
    assert kind == "success" and "β̂ = −0,50 ± 0,100" in text and "(Wahrheit β = −0,5;" in text and "innerhalb von 0,0 Standardfehlern" in text
    kind, text = ev.verdict_timedep(_study(-0.5, -1.7, -0.9, (-0.2, -0.2)))
    assert kind == "warning" and "4,0 Standardfehler des Mittels von der Wahrheit entfernt" in text


def test_verdikt_reihe_ueber_tau():
    def row(tau, mean, sd, remaining):
        return {"tau": tau, "summary": M.Summary(mean, sd, 0.01, 100, 0, 0.5, 0.1), "remaining": remaining, "treated": 1.0}
    rows = [row(0.5, -0.176, 0.09, 990), row(2.0, -0.19, 0.07, 800), row(8.0, -0.166, 0.21, 324)]
    kind, text = ev.verdict_tau_sweep(rows, -0.5, 1000)
    assert kind == "info"
    for needle in ("Über die 3 Landmarken von τ = 0,5 bis 8,0 Tage", "zwischen −0,19 (τ = 2,0) und −0,17 (τ = 8,0)", "Wahrheit β = −0,5", "wächst von 0,09 auf 0,21",
                   "fallen von 990 auf 324 von 1000", "In 3 von 3 Landmarken liegt das Mittel um mehr als die Hälfte des Betrags der Wahrheit daneben"):
        assert needle in text, needle
    assert "daneben" not in ev.verdict_tau_sweep(rows, 0.0, 1000)[1]
    assert ev.verdict_tau_sweep(rows[:1], 0.0, 1000)[0] == "warning"
