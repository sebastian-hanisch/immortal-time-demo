"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink."""

import pytest

import imt_constants as C
import imt_presets as P
from imt_scenario import World


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert C.BETA_MIN <= preset["beta"] <= C.BETA_MAX and C.WAIT_MIN <= preset["wait"] <= C.WAIT_MAX
        assert C.N_MIN <= preset["n"] <= C.N_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    first = C.PRESETS[next(iter(C.PRESETS))]
    assert World(**first) == World()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_preset_hilfen_stimmen_mit_den_werten_ueberein():
    """Die Namen und Hilfetexte nennen Zahlen (β, Wartezeit, n): sie stehen so in den Werten der Presets."""
    assert C.PRESETS["Echte Wirkung (β = −0,5)"]["beta"] == -0.5
    assert C.PRESETS["Kurze Wartezeit (1 Tag)"]["wait"] == 1.0 and C.PRESETS["Lange Wartezeit (12 Tage)"]["wait"] == 12.0
    assert C.PRESETS["Kleine Stichprobe (n = 200)"]["n"] == 200
    assert "0,61-fache" in C.PRESET_HELP["Echte Wirkung (β = −0,5)"]
    import math
    assert f"{math.exp(-0.5):.2f}".replace(".", ",") == "0,61"


def test_wertebereiche_der_regler_sind_gueltig():
    assert C.N_MIN % C.N_STEP == 0 and C.DEFAULT_N % C.N_STEP == 0 and C.N_MAX % C.N_STEP == 0
    assert C.TAU_MIN <= C.DEFAULT_TAU <= C.TAU_MAX and set(C.LANDMARKS) <= set(C.TAU_GRID)
    assert C.TAU_GRID[0] == C.TAU_MIN and C.TAU_GRID[-1] == C.TAU_MAX and C.DEFAULT_TAU in C.TAU_GRID


def test_permalink_wird_auf_die_grenzen_begrenzt():
    spec = P.SETTING_SPECS["beta_slider"]
    assert P._clean(spec, -5.0) == C.BETA_MIN and P._clean(spec, 9.0) == C.BETA_MAX and P._clean(spec, -0.3) == -0.3
    assert P._clean(P.SETTING_SPECS["n_slider"], 10**9) == C.N_MAX


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_aktives_preset_wird_erkannt(name, monkeypatch):
    monkeypatch.setattr(P.st, "session_state", {k: C.PRESETS[name][pk] for pk, k in P.PRESET_KEYS.items()})
    assert P.aktives_preset() == name
