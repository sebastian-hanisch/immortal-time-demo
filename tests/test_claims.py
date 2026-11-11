"""Jede Zahl der README steht in einer Messung (`tools/measure_immortal_time_result.json`, eingefroren, feste Zufalls-Seeds) oder in einem Test.

Dieser Test prüft viererlei:
  1. Alle Messzellen werden mit dem Code der Demo neu gerechnet und stimmen mit der eingefrorenen Datei überein (Band 1e-6, Zählungen exakt). Das Band ist so weit, weil die Cox-Anpassung iterativ ist (Abbruch bei Schrittweite 1e-7): im Linux-Check unterschied sich eine Standardabweichung von der Windows-Messung um 1e-9 relativ, schärfer sind iterativ geschätzte Zahlen über Plattformen nicht vergleichbar.
  2. Die Tabellen der README stehen wörtlich so in der README, wie `tools/measure_immortal_time.tables()` sie aus der Datei bildet; ebenso die Prosa-Zahlen (`values()`).
  3. Die Aussagen der README (Zählungen, Spannen, Monotonie, Bänder) folgen aus denselben Zahlen, jeweils mit ihren Voraussetzungen.
  4. Der Annahmen-Text steht gleich in App und README; die Hilfen der Presets und der App nennen nur gemessene Zahlen."""

import json
import math
from pathlib import Path

import pytest

import imt_constants as C
import imt_evaluation as ev
import measure_immortal_time as MI

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
DATA = json.loads((ROOT / "tools" / "measure_immortal_time_result.json").read_text(encoding="utf-8"))
Z, E = DATA["zero"], DATA["effect"]


def _same(a, b, path=""):
    """Rekursiver Vergleich: Zahlen mit Band, Ganzzahlen und Texte exakt."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            _same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert b == pytest.approx(a, rel=1e-6, abs=1e-9), (path, a, b)
    else:
        assert a == b, (path, a, b)


# ------------------------------------------------------------------------------------------------ 1. neu gerechnet


@pytest.fixture(scope="module")
def fresh():
    return MI.measure()


def test_alle_messzellen_neu_gerechnet_stimmen_mit_der_eingefrorenen_datei(fresh):
    _same(DATA, fresh)


# ------------------------------------------------------------------------------------------------ 2. Tabellen und Prosa-Zahlen


def test_alle_tabellen_der_readme_stehen_woertlich_in_der_readme():
    for name, table in MI.tables(DATA).items():
        assert table in README, name


def test_alle_prosa_zahlen_der_readme_stehen_in_der_readme():
    for name, text in MI.values(DATA).items():
        assert text in README, (name, text)


def test_annahmen_text_steht_gleich_in_app_und_readme():
    assert C.ASSUMPTIONS_MARKDOWN.rstrip() in README
    assert "C.ASSUMPTIONS_MARKDOWN" in APP


def test_readme_nennt_die_annahme_der_unabhaengigen_zensierung_und_die_linie():
    assert "Die Zensierung ist unabhängig vom Ereignis" in README and "**Überlebenszeit**" in README
    assert "Die Zensierung ist unabhängig vom Ereignis" in APP


# ------------------------------------------------------------------------------------------------ 3. Aussagen der README


def test_welt_und_laeufe_der_hauptzellen():
    for c in (Z, E):
        assert (c["wait"], c["n"], c["seed"], c["reps"]) == (4.0, 1000, 35, 200)
        assert c["naive"]["runs"] == c["timedep"]["runs"] == 200 and all(v["runs"] == 200 for v in c["landmarks"].values())
    assert (Z["beta"], E["beta"]) == (0.0, -0.5)


def test_scheinwirkung_ohne_wirkung_naiv_hr_028_landmarke_und_zeitabhaengig_nahe_null():
    assert ev.num(Z["naive"]["mean"], 3) == "−1,278" and ev.hr(Z["naive"]["mean"]) == "0,28" and ev.num(100 * Z["treated_share"], 0) == "62"
    assert Z["naive"]["coverage"] == 0.0 and E["naive"]["coverage"] == 0.0                         # „in keinem der 200 Läufe“
    for lm in (Z["landmarks"]["2"], Z["landmarks"]["4"], Z["timedep"]):
        assert abs(lm["mean"]) < 2 * lm["se_mean"]                                                 # „innerhalb von 2 Standardfehlern des Mittels von 0“
    assert ev.num(Z["landmarks"]["2"]["remaining"], 0) == "779" and ev.num(Z["landmarks"]["4"]["remaining"], 0) == "500"


def test_mit_wirkung_landmarke_zu_schwach_zeitabhaengig_trifft():
    assert ev.num(E["naive"]["mean"], 3) == "−1,720" and ev.hr(E["naive"]["mean"]) == "0,18"
    assert abs(E["timedep"]["mean"] + 0.5) < 2 * E["timedep"]["se_mean"] + 0.01 and E["timedep"]["coverage"] > 0.9
    share2 = E["landmarks"]["2"]["mean"] / -0.5
    share4 = E["landmarks"]["4"]["mean"] / -0.5
    assert 0.37 < share2 < 0.39 and 0.39 < share4 < 0.41                                          # „38 %“ und „40 %“
    assert E["landmarks"]["2"]["coverage"] == 0.0 and ev.num(100 * E["landmarks"]["4"]["coverage"], 0) == "10"
    assert ev.num(E["landmarks"]["2"]["remaining"], 0) == "801" and ev.num(E["landmarks"]["4"]["remaining"], 0) == "567"


def test_stabilitaet_ueber_sechs_zufalls_seeds():
    st = DATA["stability"]
    assert list(st) == ["35", "101", "202", "303", "404", "505"] and all(c["zero"]["reps"] == 200 and c["effect"]["reps"] == 200 for c in st.values())
    assert all(-1.30 < c["zero"]["naive"]["mean"] < -1.25 for c in st.values())
    assert all(-0.21 < c["effect"]["landmarks"]["2"]["mean"] < -0.18 for c in st.values())
    assert all(abs(c["effect"]["timedep"]["mean"] + 0.5) < 0.02 for c in st.values())


def test_naive_scheinwirkung_ist_bei_kurzer_wartezeit_am_groessten_und_verschwindet_nie():
    for key, truth in (("sweep_zero", 0.0), ("sweep_effect", -0.5)):
        rows = DATA[key]
        assert [r["wait"] for r in rows] == list(C.SWEEP_WAITS) and all(r["naive"]["runs"] == 100 for r in rows)
        means = [r["naive"]["mean"] for r in rows]
        assert all(b > a for a, b in zip(means, means[1:]))                                         # „fällt über 1, 2, 4, 8 auf …“: wird mit der Wartezeit kleiner
        assert all(r["naive"]["mean"] < truth - 5 * r["naive"]["se_mean"] for r in rows)            # „in keiner der sechs Wartezeiten“
    assert ev.num(DATA["sweep_zero"][0]["naive"]["mean"], 2) == "−2,75" and ev.num(DATA["sweep_zero"][-1]["naive"]["mean"], 2) == "−0,93"


def test_zeitabhaengig_trifft_in_jeder_wartezeit_der_reihe():
    for key, truth in (("sweep_zero", 0.0), ("sweep_effect", -0.5)):
        assert all(abs(r["timedep"]["mean"] - truth) < 0.04 for r in DATA[key]) and all(r["timedep"]["failed"] == 0 for r in DATA[key])


def test_landmarke_findet_bei_kurzer_wartezeit_weniger_und_bei_langer_mehr_der_wirkung():
    means = [r["landmarks"]["2"]["mean"] for r in DATA["sweep_effect"]]
    assert all(b < a for a, b in zip(means, means[1:]))                                             # −0,05 bis −0,33, nie den ganzen Betrag
    assert ev.num(means[0], 2) == "−0,05" and ev.num(means[-1], 2) == "−0,33" and means[-1] > -0.5 + 0.1


def test_kurze_wartezeit_laesst_der_landmarke_tag_4_keine_zwei_gruppen():
    r = DATA["sweep_zero"][0]["landmarks"]["4"]
    assert (r["failed"], r["runs"]) == (83, 17) and DATA["sweep_effect"][0]["landmarks"]["4"]["failed"] == 83
    assert all(c["landmarks"][t]["failed"] == 0 for key in ("sweep_zero", "sweep_effect") for c in DATA[key][1:] for t in ("2", "4"))


def test_tau_reihe_groesse_des_verlusts_haengt_kaum_am_tau():
    rows = DATA["tau_effect"]
    assert [r["tau"] for r in rows] == list(C.TAU_GRID) and all(r["failed"] == 0 for r in rows)
    means = [r["mean"] for r in rows]
    assert max(means) - min(means) < 0.04 and all(-0.5 < m < -0.15 for m in means)
    assert ev.num(rows[0]["sd"], 2) == "0,09" and ev.num(rows[-1]["sd"], 2) == "0,21" and ev.num(rows[0]["remaining"], 0) == "990" and ev.num(rows[-1]["remaining"], 0) == "324"
    assert all(b["remaining"] < a["remaining"] for a, b in zip(rows, rows[1:]))


def test_verduennung_zensierte_landmarke_findet_die_wirkung():
    for t in ("2", "4"):
        c = DATA["censored_effect"][t]
        assert abs(c["mean"] + 0.5) < 3 * c["se_mean"] and c["failed"] == 0


def test_vorab_zuteilung_hat_keine_scheinwirkung():
    u = DATA["upfront"]
    assert abs(u["mean"]) < 2 * u["se_mean"] and u["share"] == 0.5 and u["reps"] == 200 and u["failed"] == 0


def test_schaedliche_behandlung_sieht_naiv_wie_schutz_aus():
    h = DATA["harm"]
    assert h["beta"] == 0.5 and h["naive"]["mean"] < 0 < h["timedep"]["mean"] and abs(h["timedep"]["mean"] - 0.5) < 0.02
    assert 0 < h["landmarks"]["2"]["mean"] < 0.5 and 0 < h["landmarks"]["4"]["mean"] < 0.5


def test_stichprobengroesse_aendert_den_verlust_der_landmarke_nicht_aber_die_streuung():
    rows = DATA["small_n"]
    assert list(rows) == ["200", "500", "1000", "2000"]
    assert all(-0.21 < c["landmarks"]["2"]["mean"] < -0.18 for c in rows.values())
    assert rows["200"]["timedep"]["sd"] > rows["500"]["timedep"]["sd"] > rows["1000"]["timedep"]["sd"] > rows["2000"]["timedep"]["sd"]


def test_vergleich_mit_der_vormessung_steht_mit_der_quelle_in_der_readme():
    for needle in ("Vormessung = eine erste, schnelle Messung", "−1,276 ± 0,005", "−1,719", "−0,192 / −0,189 (802 / 570 Container)"):
        assert needle in README, needle
    # die Abstände zur Vormessung (numpy-Zufall) liegen im Bereich der Streuung zwischen den Zufalls-Seeds der Demo
    naive = [c["zero"]["naive"]["mean"] for c in DATA["stability"].values()]
    assert min(naive) - 0.01 < -1.276 < max(naive) + 0.01
    assert abs(E["naive"]["mean"] + 1.719) < 0.02 and abs(E["timedep"]["mean"] + 0.512) < 0.02


# ------------------------------------------------------------------------------------------------ 4. Hilfen und Texte der App


def test_preset_hilfen_und_annahmen_text_nennen_gemessene_zahlen():
    assert round(math.exp(-0.5), 2) == 0.61 and "0,61-fache" in C.PRESET_HELP["Echte Wirkung (β = −0,5)"]
    assert "(−0,19 statt −0,5)" in C.ASSUMPTIONS_MARKDOWN and ev.num(E["landmarks"]["2"]["mean"], 2) == "−0,19"
    assert (C.REPS, C.SWEEP_REPS) == (200, 100) and "über 100 statt 200 Läufe" in C.ASSUMPTIONS_MARKDOWN
    assert "für kurze Wartezeit" not in C.ASSUMPTIONS_MARKDOWN


def test_readme_nennt_die_quellen_mit_lesestufe():
    for needle in ("arXiv:2202.02369", "im Volltext gelesen", "nur der Abstract gelesen", "10.1097/EDE.0000000000001808", "nur Suchtreffer", "steht im gelesenen Abstract nicht"):
        assert needle in README, needle
    assert "Das stützt die Richtung der Eigenmessung" in README.replace("**", "") or "stützt die Richtung der Eigenmessung" in README.replace("**", "")
