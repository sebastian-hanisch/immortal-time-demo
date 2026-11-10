"""Tests der kopierten Kernfunktionen (aus ueberlebenszeit-planung/kern/test_uz_core.py übernommen, soweit die Demo die Funktionen braucht)."""

import math

import numpy as np
import pytest

import imt_core as K


def test_cox_handbeispiel_vier_container():
    x = [1, 0, 1, 0]
    r = K.cox_fit([1, 2, 3, 4], [1, 1, 1, 1], x)
    assert r["ok"] and r["beta"][0] == pytest.approx(0.9406, abs=5e-4)

    # Score am Ergebnis ist 0 (unabhängige Schleife über Risikomengen)
    def score(beta):
        s = 0.0
        for i in range(4):
            w = [math.exp(beta * v) for v in x[i:]]
            m = sum(wi * v for wi, v in zip(w, x[i:])) / sum(w)
            s += x[i] - m
        return s

    assert abs(score(r["beta"][0])) < 1e-8
    assert score(0.0) == pytest.approx(2 / 3, abs=1e-12)


def test_cox_gegen_eintritt():
    rng = K.SplitMix64(3)
    n = 300
    x = rng.normal(n)
    t = 8 * (rng.exponential(n) / np.exp(0.7 * x)) ** (1 / 1.5)
    c = rng.uniform(n) * 12
    obs, ev = np.minimum(t, c), t <= c
    r = K.cox_fit(obs, ev, x)
    assert r["ok"] and 0.4 < r["beta"][0] < 1.0 and r["se"][0] > 0
    # Eintritt: Teilstichprobe jenseits 3 mit Eintritt 3 gleich der verschobenen Stichprobe ohne Eintritt
    m = obs > 3.0
    a = K.cox_fit(obs[m], ev[m], x[m], entry=np.full(m.sum(), 3.0))
    b = K.cox_fit(obs[m] - 3.0, ev[m], x[m])
    assert a["beta"][0] == pytest.approx(b["beta"][0], abs=1e-9)


def test_cox_trennung_singularitaet_und_ohne_ereignis_melden_status():
    sep = K.cox_fit([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0])  # alle x = 1 sterben zuerst
    assert sep["ok"] is False and "Trennung" in sep["message"]
    const = K.cox_fit([1, 2, 3, 4], [1, 1, 1, 1], [1, 1, 1, 1])
    assert const["ok"] is False and "singulär" in const["message"]
    none = K.cox_fit([1, 2], [0, 0], [0, 1])
    assert none["ok"] is False
    # ein einziges Ereignis bei verschiedenem Merkmal ist ebenfalls eine Trennung
    one = K.cox_fit([1, 2, 3], [1, 0, 0], [1, 0, 0])
    assert one["ok"] is False and "Trennung" in one["message"]
    assert K.cox_fit([1, 2, 3, 4], [1, 1, 1, 1], [1, 0, 1, 0])["ok"] is True


def test_cox_vorzeichen_und_skalierung():
    x = np.array([1, 0, 1, 0, 1, 0, 0, 1], float)
    t = np.array([1, 2, 3, 4, 5, 6, 7, 8], float)
    e = np.ones(8, bool)
    a = K.cox_fit(t, e, x)["beta"][0]
    b = K.cox_fit(t, e, 1 - x)["beta"][0]
    c = K.cox_fit(t, e, 2 * x)["beta"][0]
    assert a == pytest.approx(-b, abs=1e-8) and a == pytest.approx(2 * c, abs=1e-8)


def test_eingabepruefung():
    with pytest.raises(K.SurvivalError):
        K.cox_fit([3, 3, 5], [1, 1, 1], [1, 0, 1], [3, 3, 0])  # Eintritt = Zeit
    with pytest.raises(K.SurvivalError):
        K.cox_fit([0, 1], [1, 1], [1, 0])  # Zeit 0
    with pytest.raises(K.SurvivalError):
        K.cox_fit([1, 2], [1, 2], [1, 0])  # Ereignis 2
    with pytest.raises(K.SurvivalError):
        K.cox_fit([1, 2], [1], [1, 0])  # Länge
    with pytest.raises(K.SurvivalError):
        K.cox_fit([], [], [])


def test_splitmix_referenzwert_und_determinismus():
    # erster Ausgabewert von SplitMix64 bei Seed 0 (Referenz: Vigna, splitmix64.c)
    assert int(K.SplitMix64(0).u64(1)[0]) == 0xE220A8397B1DCDAF
    a, b = K.SplitMix64(5), K.SplitMix64(5)
    assert np.array_equal(a.uniform(10), b.uniform(10))
    u = K.SplitMix64(9).uniform(50000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01
    x = K.SplitMix64(9).exponential(50000)
    assert abs(x.mean() - 1.0) < 0.03
    z = K.SplitMix64(9).normal(50000)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02
