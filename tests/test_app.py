"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# ------------------------------------------------------------------------------------------------ Ergänzungen dieser Demo


def _texte(at) -> str:
    parts = [m.value for m in at.markdown] + [e.value for e in at.success] + [e.value for e in at.info] + [e.value for e in at.warning] + [e.value for e in at.error] + [c.value for c in at.caption]
    return "\n".join(parts)


def test_kombination_aller_minima_und_aller_maxima_der_seitenleiste():
    for edge in ("min", "max"):
        at = _fresh_app()
        for s in at.sidebar.slider:
            value = s.min if edge == "min" else s.max
            s.set_value(int(value) if isinstance(s.value, int) else value)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])
        assert not at.error                                                           # Meldungen statt Absturz, aber kein Fehlerkasten in den Reglerecken


def test_experimentregler_an_den_raendern_geben_meldung_statt_absturz():
    at = _fresh_app()
    at.slider(key="hand_tau").set_value(0.0).run(timeout=TIMEOUT)
    assert not at.exception and any("Eine Gruppe ist leer" in w.value for w in at.warning)
    at.slider(key="hand_tau").set_value(9.0).run(timeout=TIMEOUT)
    assert not at.exception and any("Eine Gruppe ist leer" in w.value for w in at.warning)
    for tau in (0.5, 8.0):
        at.slider(key="tau_slider").set_value(tau).run(timeout=TIMEOUT)
        assert not at.exception


def test_voreinstellung_zeigt_die_scheinwirkung_und_die_annahme_zur_zensierung():
    at = _fresh_app()
    text = _texte(at)
    assert "Die Zensierung ist unabhängig vom Ereignis" in text and "keine Zensierung" in text
    assert any("Der Schutz ist Schein" in w.value for w in at.warning)
    assert any("keine Scheinwirkung" in s.value for s in at.success)                  # Landmarke bei β = 0
    assert any(c.label == "Naiv: β̂" for c in at.metric)


def test_preset_mit_wirkung_zeigt_zu_schwache_landmarke_und_treffende_zeitabhaengige_auswertung():
    at = _fresh_app()
    at.sidebar.button(key="preset_Echte Wirkung (β = −0,5)").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any("der wahren Wirkung" in w.value for w in at.warning)
    assert any(s.value.startswith("Zeitabhängig") for s in at.success)


def test_fachbloecke_stehen_im_expander_per_titel():
    at = _fresh_app()
    titel = [e.label for e in at.expander]
    assert "📐 Mathematische Formulierung" in titel and "⚠️ Annahmen, Grenzen und bekannte Probleme" in titel
