"""Abbildungen: alle Achsen gesperrt, richtige Spuren, Wahrheit eingezeichnet, kein Absturz bei leeren Ergebnissen."""

import numpy as np

import imt_model as M
import imt_visualization as V
from imt_scenario import World


def _locked(fig):
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return axes and all(a.get("fixedrange") for a in axes)


def test_handbeispiel_diagramm(six):
    fig = V.build_hand(six, 3.5)
    assert _locked(fig)
    labels = {t.name for t in fig.data}
    assert {"behandelt", "unbehandelt", "Ausfall", "Zeit vor dem Beginn (naiv als behandelt gezählt)"} <= labels
    assert fig.layout.shapes[0].x0 == 3.5                                             # senkrechte Linie bei τ
    assert len([t for t in fig.data if t.name == "behandelt"]) == 4                  # vier Container wurden vor dem Ausfall behandelt


def test_schaetzer_diagramm_mit_wahrheit():
    st = M.run_study(World(beta=-0.5, n=200, seed=2), 6, taus=(2.0, 4.0))
    fig = V.build_estimates(st, 2.0)
    assert _locked(fig) and len(fig.data) == 3 and fig.layout.shapes[0].y0 == -0.5
    assert len(V.build_estimates(st, None, show=("naive",)).data) == 1


def test_reihen_diagramme():
    rows = M.wait_sweep(0.0, 200, 3, waits=(1.0, 4.0), reps=4)
    fig = V.build_wait_sweep(rows, 0.0, 4.0)
    assert _locked(fig) and len(fig.data) == 4 and fig.layout.xaxis.type == "log"
    st = M.run_study(World(beta=0.0, n=200, seed=2), 4, taus=(1.0, 2.0, 3.0))
    fig = V.build_tau_sweep(M.landmark_sweep(st), 0.0, 2.0)
    assert _locked(fig) and len(fig.data) == 3


def test_diagramm_ohne_ergebnis_der_landmarke_stuerzt_nicht_ab():
    rows = M.wait_sweep(0.0, 100, 3, waits=(0.5,), reps=2, taus=(2.0, 4.0))
    rows[0]["naive"] = M.Summary(np.nan, np.nan, np.nan, 0, 2, np.nan, np.nan)
    fig = V.build_wait_sweep(rows, 0.0, 4.0)
    assert fig.data[0].y[0] is None
