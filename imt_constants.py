"""Konstanten: Regler-Grenzen, feste Annahmen der Welt, Farben, Presets und der Annahmen-Text (App und README haben denselben Wortlaut)."""

# Welt: Lebenszeit eines Containers ohne Behandlung lognormal (Median 4 Tage, σ 0,9 der Log-Skala), wie in verweilzeit-demo
MEDIAN, SIGMA = 4.0, 0.9

BETA_MIN, BETA_MAX, DEFAULT_BETA, BETA_STEP = -1.0, 0.5, 0.0, 0.1   # wahre Wirkung der Behandlung auf die Rate (log-Hazard-Verhältnis)
WAIT_MIN, WAIT_MAX, DEFAULT_WAIT, WAIT_STEP = 0.5, 16.0, 4.0, 0.5    # mittlere Wartezeit bis zum Behandlungsbeginn (Exponential), in Tagen
N_MIN, N_MAX, DEFAULT_N, N_STEP = 100, 2000, 1000, 100                # Container je Lauf
SEED_MAX, DEFAULT_SEED = 999999, 35

TAU_MIN, TAU_MAX, DEFAULT_TAU, TAU_STEP = 0.5, 8.0, 2.0, 0.5          # Landmarke in Tagen (Experiment 3)
HAND_TAU_MIN, HAND_TAU_MAX, HAND_DEFAULT_TAU = 0.0, 9.0, 3.5          # Landmarke im Handbeispiel (Experiment 1), in Tagen
LANDMARKS = (2.0, 4.0)                                                 # die beiden Landmarken der Messungen und der Reihe in Experiment 2

# Läufe: Zufall nur über SplitMix64; jeder Lauf hat einen eigenen, aus dem Zufalls-Seed abgeleiteten Strom.
# Komplexität am Regler-Maximum (n = 2000): je Lauf Erzeugung O(n) und fünf Cox-Anpassungen mit sortierten Zählungen O(n log n); 200 Läufe.
REPS = 200            # Experimente 3 und 4 und die Kopfzahlen von Experiment 2
SWEEP_REPS = 100      # je Wartezeit der Reihe in Experiment 2
SWEEP_WAITS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
TAU_GRID = tuple(float(x) / 2 for x in range(1, 17))                  # 0,5 bis 8 in Schritten von 0,5 (Reihe in Experiment 3)
CI_Z = 1.959964                                                        # 95-%-Intervall der Standardfehler

COLOR_NAIVE = "#C44E52"
COLOR_LANDMARK = "#DD8452"
COLOR_TIMEDEP = "#2F6B65"
COLOR_TRUTH = "#555555"
COLOR_TREATED = "#4c78a8"
COLOR_UNTREATED = "rgba(150,150,150,0.75)"

PRESETS = {
    "Keine Wirkung (Voreinstellung)": {"beta": 0.0, "wait": 4.0, "n": 1000, "seed": DEFAULT_SEED},
    "Echte Wirkung (β = −0,5)": {"beta": -0.5, "wait": 4.0, "n": 1000, "seed": DEFAULT_SEED},
    "Kurze Wartezeit (1 Tag)": {"beta": 0.0, "wait": 1.0, "n": 1000, "seed": DEFAULT_SEED},
    "Lange Wartezeit (12 Tage)": {"beta": 0.0, "wait": 12.0, "n": 1000, "seed": DEFAULT_SEED},
    "Kleine Stichprobe (n = 200)": {"beta": 0.0, "wait": 4.0, "n": 200, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Keine Wirkung (Voreinstellung)": "Die Behandlung verändert die Ausfallrate nicht (β = 0); sie beginnt im Mittel nach 4 Tagen. Die naive Auswertung findet trotzdem einen starken Schutz.",
    "Echte Wirkung (β = −0,5)": "Die Behandlung senkt die Rate ab ihrem Beginn auf das 0,61-fache (e hoch −0,5). Die naive Auswertung übertreibt, die Landmarke findet nur einen Teil, die zeitabhängige Auswertung trifft.",
    "Kurze Wartezeit (1 Tag)": "Die Behandlung beginnt im Mittel nach 1 Tag, ohne Wirkung (β = 0). Zeigt, wie sich die Scheinwirkung bei kurzer Wartezeit verhält.",
    "Lange Wartezeit (12 Tage)": "Die Behandlung beginnt im Mittel nach 12 Tagen, ohne Wirkung (β = 0): Nur etwa ein Drittel der Container erlebt den Beginn vor dem Ausfall.",
    "Kleine Stichprobe (n = 200)": "200 Container je Lauf statt 1000 (ohne Wirkung, Wartezeit 4 Tage): die Läufe streuen stärker, und die Landmarke Tag 2 behält wie bei 1000 Containern rund 80 % (160 von 200).",
}

# Derselbe Text steht in der README unter „Annahmen, Grenzen und bekannte Probleme“ (ein Test sucht ihn dort).
ASSUMPTIONS_MARKDOWN = """\
**Annahmen dieser Demo**
- **Zensierung unabhängig vom Ereignis:** In dieser Welt gibt es keine Zensierung, jeder Container wird bis zu seinem Ausfall beobachtet. Die Annahme wird hier nicht gebraucht und nicht geprüft; mit Zensierung ist nichts gemessen.
- **Die Behandlung beginnt unabhängig von der Lebenszeit ohne Behandlung und vom Zustand des Containers** (Beginn nach einer Exponential-Wartezeit, ohne Blick auf den Container; bei β ≠ 0 hängt die beobachtete Lebenszeit über die Wirkung vom Beginn ab). Wird nach dem Zustand behandelt, kommt ein Störfaktor hinzu; das ist hier nicht gemessen.
- **Die Wirkung ist konstant:** Ab dem Beginn gilt die Rate mal e^β, vorher die Grundrate. Die Grundrate ist lognormal (Median 4 Tage, σ = 0,9). Andere Formen der Wirkung oder der Grundrate sind nicht gemessen.
- **Die Container sind unabhängig voneinander;** es gibt keine unbeobachtete Anfälligkeit und keine weiteren Merkmale.

**Grenzen**
- Eine Behandlung je Container; einmal begonnen, bleibt sie bis zum Ausfall. Abbruch, Wiederholung oder mehrere Behandlungen kommen nicht vor.
- Gemessen sind zwei Landmarken (Tag 2 und Tag 4) in den Messungen der README und eine frei wählbare in der App; wie man eine Landmarke wählt, zeigt die Demo nicht.
- Die Zielstudien-Emulation (Klonen, Zensieren, Gewichten) wird nicht gerechnet; sie ist nur als Suchtreffer genannt.
- Die Cox-Anpassung (Breslow) ist eigener Code, gegen unabhängige Rechenwege geprüft, aber nicht gegen ein Fremdpaket (lifelines, statsmodels).

**Bekannte Probleme und Unsicherheiten**
- Die Zahlen sind Eigenmessungen mit festen Zufalls-Seeds an einer ausgedachten Welt; sie sind keine Literaturwerte. Die Größe des Verlusts der Landmarke (−0,19 statt −0,5) ist Eigenmessung; die Literatur stützt nur die Richtung.
- Die Reihe in Experiment 2 mittelt je Wartezeit über 100 statt 200 Läufe, der Standardfehler des Mittels ist dort entsprechend größer. Läufe ohne Ergebnis (perfekte Trennung bei sehr kurzer Wartezeit) gehen nicht in das Mittel ein; die Tabelle nennt ihre Zahl.
- Nur am Rechner angesehen, nicht auf Mobilgeräten.
"""
