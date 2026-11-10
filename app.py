"""Immortal Time: behandelt, weil man lange genug lebte - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Eine Behandlung beginnt irgendwann nach der Ankunft eines Containers. Wer die Container einteilt in „je behandelt“ und „nie behandelt“ und beide ab Tag 0 vergleicht, rechnet die Zeit
bis zum Beginn der Behandlung als behandelt, obwohl ein Container sie überleben musste, um in diese Gruppe zu kommen (unsterbliche Zeit). Vier Experimente: sechs Container von Hand,
die naive Einteilung und die Wartezeit, die Wahl der Landmarke, die zeitabhängige Auswertung.

Lauffähig mit: streamlit run app.py
"""

from fractions import Fraction

import streamlit as st

import imt_constants as C
import imt_evaluation as ev
import imt_model as M
import imt_visualization as V
from imt_core import SurvivalError
from imt_layout import dunkles_design, kopf_css
from imt_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from imt_scenario import World

st.set_page_config(page_title="Immortal Time – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=f"Rechne {C.REPS} Läufe …")
def _study(world: World):
    """Alle Landmarken der Reihe in einem Durchgang: derselbe Datensatz je Lauf für Experiment 2 (naiv), 3 (Landmarke) und 4 (zeitabhängig)."""
    return M.run_study(world, C.REPS, C.TAU_GRID)


@st.cache_data(show_spinner=f"Rechne die Reihe über die Wartezeit ({len(C.SWEEP_WAITS)} Wartezeiten mal {C.SWEEP_REPS} Läufe) …")
def _sweep(beta: float, n: int, seed: int):
    """Die Reihe hängt nicht an der Wartezeit der Seitenleiste: der Cache trifft, wenn nur sie sich ändert."""
    return M.wait_sweep(beta, n, seed)


def md_table(headers, rows) -> str:
    """Markdown-Tabelle aus Texten (kein | in den Zellen)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in imt_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⏳ Immortal Time: behandelt, weil man lange genug lebte")
st.markdown(
    """
Eine **Behandlung** (hier: eine Wartung des Kühlaggregats) beginnt irgendwann nach der Ankunft eines Containers, sobald ein Techniker Zeit hat. Wer die Container einteilt in **„je behandelt“** und
**„nie behandelt“** und beide **ab Tag 0** vergleicht, rechnet die Wartezeit als behandelt: Ein Container musste den Beginn **erleben**, um in die erste Gruppe zu kommen. Diese **unsterbliche Zeit**
lässt die Behandlung schützend aussehen, auch wenn sie nichts bewirkt. Eine **Landmarke** (Status zu einem festen Tag τ) vermeidet das, verliert aber Container und findet eine echte Wirkung nur zum
Teil; die **zeitabhängige** Auswertung teilt die Person-Zeit auf und trifft sie (so zeigt es diese Welt, nicht jede Welt).

**Annahme in dieser Demo: Die Zensierung ist unabhängig vom Ereignis.** Hier gibt es keine Zensierung, jeder Container wird bis zum Ausfall beobachtet; die Annahme wird nicht gebraucht und nicht geprüft.
Die Welt ist ausgedacht (Lebenszeit lognormal, Median 4 Tage, σ = 0,9); die Zahlen stammen aus keinem echten Terminal.

Weiter unten: **🔬 Experiment 1** (sechs Container von Hand), **❓ Kernfrage mit Experiment 2** (die naive Einteilung und die Wartezeit), **🔬 Experiment 3** (Landmarke wählen), **🔬 Experiment 4**
(zeitabhängig), **🚧 Wo die Annahmen enden** sowie die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Auswertung mit Behandlungsbeginn", expanded=True):
    st.markdown(
        """
1. **Naiv:** Ein Container zählt als „behandelt“, wenn seine Behandlung vor dem Ausfall begann, und das **ab Tag 0**. Die Zeit vor dem Beginn war aber unbehandelt; sie wird der behandelten Gruppe gutgeschrieben.
   Ausfälle vor dem Beginn landen dagegen in der Gruppe „unbehandelt“.
2. **Landmarke τ:** Nur wer τ erlebt, bleibt; der Status gilt zu τ (behandelt = Beginn bis τ), die Risikomenge beginnt erst bei τ. Spätere Behandlungen bleiben unbeachtet: Wer erst nach τ behandelt wird, zählt als unbehandelt.
3. **Zeitabhängig:** Die Person-Zeit wird aufgeteilt. Bis zum Beginn ist ein Container unbehandelt, danach behandelt; jeder Abschnitt ist eine eigene Zeile mit Start und Ende, der Ausfall gehört zum letzten Abschnitt.
4. **Warum das hilft:** In jedem Zeitpunkt vergleicht die Cox-Rechnung die Container, die zu diesem Zeitpunkt noch da sind, nach ihrem Status **zu diesem Zeitpunkt**. Mehr braucht die Auswertung in dieser Welt nicht (Beginn unabhängig von Lebenszeit und Zustand).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt der Daten** (Experimente 2 bis 4)")
    beta = st.slider("Wahre Wirkung β der Behandlung", *bounds("beta_slider"), step=C.BETA_STEP, key="beta_slider",
                     help="Log-Hazard-Verhältnis: Ab dem Behandlungsbeginn gilt die Rate mal e hoch β. 0 heißt keine Wirkung, −0,5 senkt die Rate auf das 0,61-fache, positive Werte schaden.")
    wait = st.slider("Mittlere Wartezeit bis zur Behandlung (Tage)", *bounds("wait_slider"), step=C.WAIT_STEP, key="wait_slider",
                     help="Der Behandlungsbeginn ist exponentialverteilt mit diesem Mittel, unabhängig von der Lebenszeit. Ein Container, der vorher ausfällt, wird nie behandelt.")
    n = st.slider("Container n je Lauf", *bounds("n_slider"), step=C.N_STEP, key="n_slider")
    seed = st.number_input("Zufalls-Seed der Daten", *bounds("seed_input"), key="seed_input", step=1,
                           help="Jeder Lauf hat einen eigenen, festen Zufallsstrom, der aus diesem Zufalls-Seed abgeleitet ist.")
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"beta_slider": float(beta), "wait_slider": float(wait), "n_slider": int(n), "seed_input": int(seed)})
world = World(beta=round(float(beta), 1), wait=float(wait), n=int(n), seed=int(seed))

# --- Experiment 1: von Hand ------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🧮 Das Verfahren von Hand: sechs Container")
st.subheader("🔬 Experiment 1: Naiv und Landmarke in Brüchen")
st.markdown(
    """
Sechs ausgedachte Container mit Lebenszeit $T$ und Behandlungsbeginn $S$ (ein Strich heißt: nie behandelt). **Naiv** zählt ein Container als behandelt, wenn $S < T$. Die **Landmarke** $\\tau$ behält nur Container mit
$T > \\tau$ und zählt als behandelt, wer bis $\\tau$ begonnen hat ($S \\le \\tau$). Verglichen werden hier die Mittel der Lebenszeit; Schieberegler: die Landmarke.
"""
)
hand_tau = st.slider("Landmarke τ im Handbeispiel (Tage)", C.HAND_TAU_MIN, C.HAND_TAU_MAX, C.HAND_DEFAULT_TAU, step=0.5, key="hand_tau",
                     help="Der Standardwert 3,5 ist die Landmarke des Handbeispiels. Bei τ = 0 hat noch niemand begonnen, bei großem τ bleibt kaum ein Container übrig: dann gibt es keinen Vergleich.")
tau_f = Fraction(hand_tau)
hand = M.hand_rows(tau_f)
st.markdown(md_table(
    ["Container", "Lebenszeit T", "Behandlungsbeginn S", "naiv", f"Landmarke τ = {ev.frac_de(tau_f)}"],
    [[r["container"], ev.frac_de(r["t"]).replace(",0", ""), "–" if r["s"] is None else ev.frac_de(r["s"]).replace(",0", ""), r["naive"], r["landmark"]] for r in hand],
))
st.plotly_chart(V.build_hand(M.PEOPLE, hand_tau), width="stretch", key="hand_chart")
st.caption("Rot schraffiert: Zeit vor dem Beginn bei einem Container, der naiv als behandelt zählt (unsterbliche Zeit). Blau: tatsächlich behandelt. Kreuz: Ausfall.")
kind, text = ev.verdict_hand(tau_f)
getattr(st, kind)(text)

st.markdown("---")

# --- Kernfrage und Experiment 2 --------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viel Schutz findet man, wenn man Container danach einteilt, ob sie je behandelt wurden?")
st.subheader("🔬 Experiment 2: Die naive Einteilung und die Wartezeit")
st.markdown(
    f"""
Welt: Die Lebenszeit ohne Behandlung ist lognormal (Median {C.MEDIAN:g} Tage, σ = {C.SIGMA:g}). Der Behandlungsbeginn $S$ ist exponentialverteilt mit der mittleren Wartezeit der Seitenleiste, unabhängig von der Lebenszeit;
ab $S$ gilt die Rate mal $e^{{\\beta}}$. Es laufen **{C.REPS} Läufe** mit festen Zufalls-Seeds, jeder mit $n$ Containern; die Einteilung ist „je behandelt“ ($S < T$), verglichen ab Tag 0 (Cox-Regression).
"""
)
try:
    study = _study(world)
except SurvivalError as err:
    st.error(f"Die Rechnung ist nicht möglich: {err}")
    st.stop()
naive = M.summarize_naive(study)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wahrheit β", ev.num(world.beta, 1), help="Die in der Seitenleiste gewählte Wirkung.")
c2.metric("Naiv: β̂", ev.num(naive.mean, 2), help=f"Mittel über {naive.runs} Läufe mit Ergebnis.")
c3.metric("Naiv: Hazard-Verhältnis", ev.hr(naive.mean) if naive.runs else "–", help="e hoch (mittleres β̂). Die Wahrheit hat das Hazard-Verhältnis " + ev.hr(world.beta) + ".")
c4.metric("Zählen als behandelt", f"{100 * study.treated_share.mean():.0f} %", help="Anteil der Container, deren Behandlung vor dem Ausfall begann (Mittel über die Läufe).")
st.caption(f"Mittel über {naive.runs} Läufe; Standardfehler des Mittels {ev.num(naive.se_mean, 3)}, Streuung zwischen den Läufen (Standardabweichung) {ev.num(naive.sd, 2)}.")
if naive.runs:
    st.plotly_chart(V.build_estimates(study, None, show=("naive",)), width="stretch", key="naive_chart")
    st.caption("Je Lauf ein Punkt, Kasten: mittlere Hälfte der Läufe, gestrichelte Linie im Kasten: Mittel. Graue Linie: die Wahrheit.")
kind, text = ev.verdict_naive(study)
getattr(st, kind)(text)

st.markdown(
    f"**Wie ändert sich das mit der Wartezeit?** Dieselbe Welt mit mittleren Wartezeiten von {C.SWEEP_WAITS[0]:g} bis {C.SWEEP_WAITS[-1]:g} Tagen (gleiche Zufallszahlen, nur die Wartezeit ändert sich), "
    f"je Wartezeit {C.SWEEP_REPS} Läufe; dazu die beiden Landmarken Tag {C.LANDMARKS[0]:g} und Tag {C.LANDMARKS[1]:g} und die zeitabhängige Auswertung."
)
try:
    sweep = _sweep(world.beta, world.n, world.seed)
except SurvivalError as err:
    st.error(f"Die Reihe ist nicht möglich: {err}")
    st.stop()
st.plotly_chart(V.build_wait_sweep(sweep, world.beta, world.wait), width="stretch", key="wait_chart")
st.caption(f"Mittel über {C.SWEEP_REPS} Läufe je Wartezeit, Balken: ± 2 Standardfehler des Mittels. Fehlt ein Punkt, hatte keiner der Läufe ein Ergebnis (zum Beispiel perfekte Trennung, wenn fast alle Container behandelt sind); in der Tabelle steht dann, wie viele Läufe ohne Ergebnis in das Mittel nicht eingehen.")
st.markdown(md_table(
    ["Mittlere Wartezeit (Tage)", "behandelt (naiv)", "naiv β̂", f"Landmarke τ = {C.LANDMARKS[0]:g}: β̂ (bleiben)", f"Landmarke τ = {C.LANDMARKS[1]:g}: β̂ (bleiben)", "zeitabhängig β̂"],
    [[ev.num(r["wait"], 1), f"{100 * r['treated_share']:.0f} %", ev.num(r["naive"].mean, 2),
      *[ev.landmark_cell(r['landmarks'][t].mean, r['landmarks'][t].failed, r['landmarks'][t].runs, r['remaining'][t]) for t in C.LANDMARKS], ev.num(r["timedep"].mean, 2)] for r in sweep],
))
kind, text = ev.verdict_wait_sweep(sweep, world.beta)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3: Landmarke wählen -----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Landmarke wählen")
st.markdown(
    f"""
Die Landmarke behält nur die Container, die τ erleben, und zählt als behandelt, wer bis τ begonnen hat; **spätere Behandlungen bleiben unbeachtet**. Das verliert Container und Präzision, und es verdünnt die Wirkung:
Wer erst nach τ behandelt wird, steht als unbehandelt in der Vergleichsgruppe. Dieselben {C.REPS} Datensätze wie in Experiment 2; der Regler wählt die Landmarke.
"""
)
tau = st.slider("Landmarke τ (Tage)", C.TAU_MIN, C.TAU_MAX, C.DEFAULT_TAU, step=C.TAU_STEP, key="tau_slider",
                help="Nur Container, die τ erleben, bleiben; behandelt zählt, wer bis τ begonnen hat. Ein größeres τ lässt weniger Container übrig.")
tau = float(tau)
lm = M.summarize_landmark(study, tau)
d1, d2, d3, d4 = st.columns(4)
d1.metric("Verbleibende Container", f"{M.mean_remaining(study, tau):.0f} von {world.n}", help="Mittel über die Läufe: Container mit Lebenszeit über τ.")
d2.metric("Davon bis τ behandelt", f"{M.mean_treated_at_tau(study, tau):.0f}", help="Mittel über die Läufe.")
d3.metric("Landmarke: β̂", ev.num(lm.mean, 2), help=f"Mittel über {lm.runs} Läufe mit Ergebnis.")
d4.metric("Wahrheit β", ev.num(world.beta, 1))
st.caption(f"Standardfehler des Mittels {ev.num(lm.se_mean, 3)}, Streuung zwischen den Läufen (Standardabweichung) {ev.num(lm.sd, 2)}, Überdeckung des wahren Werts durch das 95-%-Intervall in {ev.num(100 * lm.coverage, 0)} % der Läufe.")
kind, text = ev.verdict_landmark(study, tau)
getattr(st, kind)(text)
tau_rows = M.landmark_sweep(study)
st.plotly_chart(V.build_tau_sweep(tau_rows, world.beta, tau), width="stretch", key="tau_chart")
st.caption("Oben das mittlere β̂ je Landmarke mit Band (± 1 Standardabweichung über die Läufe), unten die verbleibenden Container. Die senkrechte Linie ist das gewählte τ.")
kind, text = ev.verdict_tau_sweep(tau_rows, world.beta, world.n)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4: zeitabhängig -----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Richtig geführt: die Behandlung als zeitabhängiges Merkmal")
st.markdown(
    """
Die Person-Zeit wird **aufgeteilt**: Ein Container, dessen Behandlung vor dem Ausfall begann, wird zu zwei Zeilen, **unbehandelt von 0 bis $S$** (ohne Ausfall) und **behandelt von $S$ bis $T$** (mit Ausfall);
jeder andere bleibt eine Zeile. So sieht die Rechnung zu jedem Ausfall nur den Status, den der Container **zu diesem Zeitpunkt** hat. Die sechs Container von Experiment 1 ergeben zehn Zeilen:
"""
)
split = M.hand_split()
st.markdown(md_table(
    ["Container", "von", "bis", "Status", "Ausfall in dieser Zeile"],
    [[i, ev.frac_de(a).replace(",0", ""), ev.frac_de(b).replace(",0", ""), "behandelt" if x else "unbehandelt", "ja" if e else "nein"] for i, a, b, x, e in split],
))
st.markdown(f"Jetzt alle drei Auswertungen auf denselben {C.REPS} Datensätzen (Landmarke: τ aus Experiment 3).")
st.plotly_chart(V.build_estimates(study, tau), width="stretch", key="all_chart")
st.caption("Je Lauf ein Punkt, Kasten: mittlere Hälfte der Läufe, gestrichelte Linie im Kasten: Mittel. Graue Linie: die Wahrheit.")
td = M.summarize_timedep(study)
st.markdown(md_table(
    ["Auswertung", "β̂ (Mittel ± Standardfehler)", "Hazard-Verhältnis", "Überdeckung 95 %", "Läufe ohne Ergebnis"],
    [[name, f"{ev.num(s.mean, 3)} ± {ev.num(s.se_mean, 3)}", ev.hr(s.mean) if s.runs else "–", f"{ev.num(100 * s.coverage, 0)} %" if s.runs else "–", s.failed]
     for name, s in (("naiv", naive), (f"Landmarke τ = {tau:g}", lm), ("zeitabhängig", td))] + [["Wahrheit", ev.num(world.beta, 1), ev.hr(world.beta), "–", "–"]],
))
kind, text = ev.verdict_timedep(study)
getattr(st, kind)(text)
st.caption(
    "Die Zielstudien-Emulation (Klonen, Zensieren, Gewichten) wird als vollständigere Lösung genannt (nur Suchtreffer, nicht gelesen); diese Demo rechnet und misst sie nicht."
)

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was gilt, wenn sie verletzt ist |
|---|---|
| **Zensierung unabhängig vom Ereignis** | Nicht gemessen: Die Welt hat keine Zensierung. Mit Zensierung sagt keine Zahl hier etwas. |
| **Behandlungsbeginn unabhängig von Lebenszeit und Zustand** | Nicht gemessen: Wird nach dem Zustand des Containers behandelt, entsteht ein Störfaktor, den keine der drei Auswertungen beseitigt. Die Messungen gelten nur für diesen einen Mechanismus. |
| **Konstante Wirkung ab Beginn** (Rate mal e hoch β) | Die zeitabhängige Auswertung schätzt dann ein Mittel über die Zeit; nicht gemessen. |
| **Eine Behandlung, einmal begonnen, bleibt** | Abbruch, Wiederholung und mehrere Behandlungen sind nicht modelliert. |
| **Genug Container mit beiden Status** | Gemessen (Reihe über die Wartezeit): Bei sehr kurzer Wartezeit sind fast alle Container behandelt, die Gruppen trennen sich, und mehrere Auswertungen haben kein oder ein unzuverlässiges Ergebnis. |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(C.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* je Container die Lebenszeit $T_i$ (alle Ausfälle beobachtet) und der Behandlungsbeginn $S_i$ (die Behandlung wirkt nur, wenn $S_i < T_i$). *Gesucht:* das log-Hazard-Verhältnis $\beta$ der Behandlung.

| Symbol | Bedeutung |
|---|---|
| $T_i$, $S_i$ | Lebenszeit bis zum Ausfall und Behandlungsbeginn des Containers $i$ |
| $h_0(t)$, $H_0(t)$ | Grundrate (lognormal, Median 4 Tage, $\sigma = 0{,}9$) und ihr Integral |
| $\beta$ | wahres log-Hazard-Verhältnis: ab $S_i$ gilt die Rate $h_0(t)\,e^{\beta}$ |
| $E_i$ | Zufallszahl $\mathrm{Exp}(1)$, der „Verbrauch“ an Grundrate bis zum Ausfall |
| $x_i(t)$ | Behandlungsstatus des Containers $i$ zur Zeit $t$ (1 = behandelt) |
| $R(t)$ | Risikomenge: Container, die zur Zeit $t$ noch da sind (bei Eintritt $a_i$: $a_i < t \le$ Ende der Zeile) |
| $\tau$ | Landmarke |

**Die Welt.** Die Rate des Containers $i$ ist $h_i(t) = h_0(t)\,e^{\beta\,\mathbf 1[t \ge S_i]}$. Mit $H_0(t) = -\ln\!\big(1 - \Phi((\ln t - \ln 4)/0{,}9)\big)$ ($\Phi$: Verteilungsfunktion der Standardnormalverteilung) fällt ein Container aus, sobald
seine kumulierte Rate den Wert $E_i$ erreicht. Daraus folgt die Zeitumrechnung (die der Erzeuger benutzt):

$$H_0(T_i) = \begin{cases} E_i & E_i < H_0(S_i) \\ H_0(S_i) + \big(E_i - H_0(S_i)\big)\,e^{-\beta} & \text{sonst} \end{cases}$$

Ohne Wirkung ($\beta = 0$) ist $H_0(T_i) = E_i$ in beiden Fällen: Die Lebenszeit hängt dann nicht an $S_i$.

**Cox-Regression mit Status.** Die partielle Likelihood (Breslow) für ein 0/1-Merkmal über die Ausfallzeiten $t_{(k)}$ ist

$$\ell(\beta) = \sum_k \Big( \beta\, x_{(k)} - \ln \sum_{i \in R(t_{(k)})} e^{\beta\, x_i(t_{(k)})} \Big), \qquad U(\beta) = \sum_k \Big( x_{(k)} - \frac{\sum_{i \in R} x_i\, e^{\beta x_i}}{\sum_{i \in R} e^{\beta x_i}} \Big)$$

mit dem Status $x_{(k)}$ des ausfallenden Containers; $\hat\beta$ löst $U(\hat\beta) = 0$ (Newton-Verfahren), der Standardfehler ist $1/\sqrt{I(\hat\beta)}$ mit der beobachteten Information $I$.

**Die drei Auswertungen** unterscheiden sich nur in den Zeilen und im Status:

- *naiv:* eine Zeile $(0, T_i]$ je Container, $x_i = \mathbf 1[S_i < T_i]$ für alle Zeiten;
- *Landmarke:* nur Container mit $T_i > \tau$, Zeile $(\tau, T_i]$, $x_i = \mathbf 1[S_i \le \tau]$;
- *zeitabhängig:* Zeile $(0, S_i]$ mit $x = 0$ ohne Ausfall und Zeile $(S_i, T_i]$ mit $x = 1$ mit Ausfall, wenn $S_i < T_i$; sonst eine Zeile $(0, T_i]$ mit $x = 0$ und Ausfall.

**Handbeispiel.** Mit $(T, S) = (2,1), (3,5), (4,2{,}5), (6,-), (8,7), (10,3)$ sind naiv die Container 1, 3, 5, 6 behandelt (Mittel der Lebenszeit $6{,}0$ gegen $4{,}5$);
die Landmarke $\tau = 3{,}5$ behält die Container 3, 4, 5, 6 mit $7{,}0$ gegen $7{,}0$.

**Deutung (eigene Lesart, nicht aus einer Quelle).** In der naiven Einteilung enthält die Gruppe „unbehandelt“ nur Container, die vor ihrem Behandlungsbeginn ausfielen oder nie behandelt wurden; die Wartezeit der behandelten Container steht in der Gruppe „behandelt“.
Dass die Landmarke eine echte Wirkung unterschätzt, lesen wir als Verdünnung: Wer nach $\tau$ behandelt wird, bleibt in der Vergleichsgruppe. Die Prüfung steht in der README (Variante, die später Behandelte beim Beginn zensiert).

**Literatur.** J. Wang, P. N. Peduzzi, M. Wininger und S. Ma, Statistical Methods for Accommodating Immortal Time: A Selective Review and Comparison, arXiv:2202.02369: im Volltext gelesen. In ihrer Simulation
(wahres log-Hazard-Verhältnis +0,5) verzerren Verfahren mit neu gesetztem Nullpunkt stark; die Landmarke mindert die Verzerrung, verzerrt aber weiter (in ihrer Simulation etwa −0,05 bis −0,17 bei wahr +0,5), und die zeitabhängige Modellierung gilt dort als Goldstandard.
M. A. Hernán, J. A. C. Sterne, J. P. T. Higgins, I. Shrier und S. Hernández-Díaz, A structural description of biases that generate immortal time, Epidemiology (online 4. 11. 2024, DOI 10.1097/EDE.0000000000001808): nur der Abstract gelesen.
Die Größe des Verlusts der Landmarke in dieser Demo ist Eigenmessung, kein Literaturwert. Die Formeln oben sind gegen unabhängige Rechnungen geprüft (`tests/test_oracle_immortal_time.py`).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
