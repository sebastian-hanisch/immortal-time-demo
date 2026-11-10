"""Kennzahlen, Formatierung und Verdikte (reine Funktionen: `(Art, Text)`; die App ruft `getattr(st, Art)(Text)`, tests/test_verdicts.py prüft jede Zahl im Text)."""

import math
from fractions import Fraction

import imt_model as M


# ------------------------------------------------------------------------------------------------ Formatierung


def num(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma und echtem Minuszeichen; `sign` setzt bei positiven Werten ein Pluszeichen."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"{abs(x):.{nd}f}".replace(".", ",")
    if round(x, nd) < 0:
        return "−" + s
    return ("+" + s) if sign and round(x, nd) > 0 else s


def frac_de(f: Fraction | None) -> str:
    """Bruch als Dezimalzahl mit Komma (eine oder zwei Stellen, wenn das genau aufgeht, sonst „≈“ mit dem Bruch); None heißt leere Gruppe."""
    if f is None:
        return "–"
    f = Fraction(f)
    if (f * 10).denominator == 1:
        return f"{float(f):.1f}".replace(".", ",")
    if (f * 100).denominator == 1:
        return f"{float(f):.2f}".replace(".", ",")
    return f"≈ {float(f):.2f}".replace(".", ",") + f" ({f.numerator}/{f.denominator})"


def times_de(values) -> str:
    """Menge von Zeiten: ganze Tage ohne Nachkommastelle, sonst mit Komma; leere Menge als „keine“."""
    if not values:
        return "keine"
    return "{" + ", ".join(str(int(v)) if Fraction(v).denominator == 1 else frac_de(v) for v in values) + "}"


def hr(beta: float) -> str:
    """Hazard-Verhältnis e^β̂ mit zwei Stellen."""
    return num(math.exp(beta), 2)


# ------------------------------------------------------------------------------------------------ Verdikte


def _group_phrase(label: str, times, mean) -> str:
    if not times:
        return f"keine {label}en Container"
    return f"{label}e {times_de(times)} im Mittel {frac_de(mean)} Tage"


def verdict_hand(tau, people=M.PEOPLE) -> tuple[str, str]:
    """Experiment 1: naive Mittel gegen Landmarken-Mittel der Lebenszeit (Brüche)."""
    a = M.hand_naive(people)
    b = M.hand_landmark(tau, people)
    naive = (f"Naiv (behandelt = Behandlungsbeginn vor dem Ausfall): {_group_phrase('behandelt', a['treated'], a['mean_treated'])}, "
             f"{_group_phrase('unbehandelt', a['untreated'], a['mean_untreated'])}.")
    if a["mean_treated"] is not None and a["mean_untreated"] is not None:
        naive += f" Unterschied {frac_de(a['mean_treated'] - a['mean_untreated'])} Tage."
    lm = (f"Landmarke τ = {frac_de(Fraction(tau))} Tage (nur wer länger lebt, Status zu τ): {_group_phrase('behandelt', b['treated'], b['mean_treated'])}, "
          f"{_group_phrase('unbehandelt', b['untreated'], b['mean_untreated'])}.")
    if b["mean_treated"] is None or b["mean_untreated"] is None:
        return "warning", naive + " " + lm + " Eine Gruppe ist leer: Bei diesem τ gibt es keinen Vergleich."
    d = b["mean_treated"] - b["mean_untreated"]
    tail = f" Unterschied {frac_de(d)} Tage." + (" Die Landmarke bewertet beide Gruppen gleich." if d == 0 else "")
    return "info", naive + " " + lm + tail + " Sechs ausgedachte Container sagen nichts über eine Welt; es ist eine Rechenübung."


def _truth_phrase(beta: float) -> str:
    if beta == 0:
        return "In dieser Welt wirkt die Behandlung nicht (β = 0)"
    if beta < 0:
        return f"In dieser Welt senkt die Behandlung die Rate (β = {num(beta, 1)}, Hazard-Verhältnis {hr(beta)})"
    return f"In dieser Welt erhöht die Behandlung die Rate (β = {num(beta, 1, True)}, Hazard-Verhältnis {hr(beta)})"


def verdict_naive(study: M.Study) -> tuple[str, str]:
    """Experiment 2, Kopfzahl: was die naive Einteilung „je behandelt“ findet."""
    s = M.summarize_naive(study)
    w = study.world
    if s.runs == 0:
        return "warning", f"Die naive Auswertung hat in keinem der {study.reps} Läufe ein Ergebnis (meist perfekte Trennung: fast alle Container zählen als behandelt). Bei Wartezeit {num(w.wait, 1)} Tage ist hier nichts zu schätzen."
    head = f"{_truth_phrase(w.beta)}. Die naive Auswertung findet im Mittel über {s.runs} Läufe β̂ = {num(s.mean, 2)} ± {num(s.se_mean, 3)} (Hazard-Verhältnis {hr(s.mean)})"
    if s.failed:
        head += f"; in {s.failed} Läufen gab es kein Ergebnis"
    if s.mean < w.beta - 2 * s.se_mean and w.beta >= 0:
        return "warning", head + f": Der Schutz ist Schein, {int(round(100 * study.treated_share.mean()))} % der Container zählen als behandelt, weil sie den Beginn erlebt haben."
    if s.mean < w.beta - 2 * s.se_mean:
        return "warning", head + f": mehr Schutz, als die Wahrheit enthält (Wahrheit β = {num(w.beta, 1)})."
    return "info", head + ": nicht kleiner als die Wahrheit."


def verdict_wait_sweep(rows: list[dict], truth: float) -> tuple[str, str]:
    """Experiment 2, Reihe über die Wartezeit: wo die naive Verzerrung am größten ist und ob sie irgendwo verschwindet."""
    ok = [r for r in rows if r["naive"].runs > 0]
    if len(ok) < 2:
        return "warning", "Für die Reihe über die Wartezeit gibt es zu wenige Läufe mit Ergebnis."
    biggest = min(ok, key=lambda r: r["naive"].mean)
    smallest = max(ok, key=lambda r: r["naive"].mean)
    below = sum(1 for r in ok if r["naive"].mean < truth - 2 * r["naive"].se_mean)
    steps = [b["naive"].mean >= a["naive"].mean for a, b in zip(ok, ok[1:])]
    shape = "wird mit längerer Wartezeit kleiner" if all(steps) else "verläuft nicht gleichmäßig über die Wartezeit"
    return "info", (f"Die naive Schätzung liegt in {below} von {len(ok)} Wartezeiten unter der Wahrheit (β = {num(truth, 1)}), am weitesten bei {num(biggest['wait'], 1)} Tagen "
                    f"(β̂ = {num(biggest['naive'].mean, 2)}), am wenigsten bei {num(smallest['wait'], 1)} Tagen (β̂ = {num(smallest['naive'].mean, 2)}); der Abstand {shape}.")


def verdict_landmark(study: M.Study, tau: float) -> tuple[str, str]:
    """Experiment 3: Landmarke τ, verbleibende Container und wie viel der Wirkung sie findet."""
    s = M.summarize_landmark(study, tau)
    w = study.world
    rem = M.mean_remaining(study, tau)
    lost = 100.0 * (1.0 - rem / w.n)
    if s.runs == 0:
        return "warning", f"Bei τ = {num(tau, 1)} Tagen gibt es in keinem Lauf ein Ergebnis: Es bleiben im Mittel {rem:.0f} Container, fast alle sind schon behandelt oder keiner ist es (keine zwei Gruppen, perfekte Trennung)."
    head = f"Landmarke τ = {num(tau, 1)} Tage: Es bleiben im Mittel {rem:.0f} von {w.n} Container ({num(lost, 0)} % fallen weg), davon {M.mean_treated_at_tau(study, tau):.0f} bis τ behandelt. β̂ = {num(s.mean, 2)} ± {num(s.se_mean, 3)} im Mittel über {s.runs} Läufe"
    if s.failed:
        head += f" ({s.failed} Läufe ohne Ergebnis)"
    if w.beta == 0:
        z = abs(s.mean) / s.se_mean if s.se_mean and s.se_mean > 0 else float("inf")
        if z <= 2:
            return "success", head + f": keine Scheinwirkung (Abstand zur Null {num(z, 1)} Standardfehler des Mittels, Wahrheit β = 0)."
        return "warning", head + f": ein Rest von Scheinwirkung, {num(z, 1)} Standardfehler des Mittels von der Null (Wahrheit β = 0)."
    share = 100.0 * s.mean / w.beta
    return "warning", head + f": die Landmarke findet {num(share, 0)} % der wahren Wirkung (Wahrheit β = {num(w.beta, 1)}; Verhältnis der Mittel)."


def verdict_timedep(study: M.Study) -> tuple[str, str]:
    """Experiment 4: zeitabhängige Auswertung gegen die Wahrheit."""
    s = M.summarize_timedep(study)
    w = study.world
    if s.runs == 0:
        return "warning", "Die zeitabhängige Auswertung hat in keinem Lauf ein Ergebnis."
    z = abs(s.mean - w.beta) / s.se_mean if s.se_mean and s.se_mean > 0 else float("inf")
    head = f"Zeitabhängig: β̂ = {num(s.mean, 2)} ± {num(s.se_mean, 3)} im Mittel über {s.runs} Läufe (Wahrheit β = {num(w.beta, 1)}; Überdeckung des 95-%-Intervalls in {num(100 * s.coverage, 0)} % der Läufe)"
    if z <= 2:
        return "success", head + f": innerhalb von {num(max(z, 0.0), 1)} Standardfehlern des Mittels von der Wahrheit."
    return "warning", head + f": {num(z, 1)} Standardfehler des Mittels von der Wahrheit entfernt."


def verdict_tau_sweep(rows: list[dict], truth: float, n: int) -> tuple[str, str]:
    """Experiment 3, Reihe über τ: Bereich von β̂, Streuung über die Läufe und verbleibende Container, nur als Zählung über die Reihe."""
    ok = [r for r in rows if r["summary"].runs > 1]
    if len(ok) < 2:
        return "warning", "Für die Reihe über τ gibt es zu wenige Läufe mit Ergebnis."
    lo = min(ok, key=lambda r: r["summary"].mean)
    hi = max(ok, key=lambda r: r["summary"].mean)
    first, last = ok[0], ok[-1]
    farther = sum(1 for r in ok if abs(r["summary"].mean - truth) > 0.5 * abs(truth)) if truth != 0 else None
    text = (f"Über die {len(ok)} Landmarken von τ = {num(first['tau'], 1)} bis {num(last['tau'], 1)} Tage liegt das mittlere β̂ zwischen {num(lo['summary'].mean, 2)} (τ = {num(lo['tau'], 1)}) und "
            f"{num(hi['summary'].mean, 2)} (τ = {num(hi['tau'], 1)}); Wahrheit β = {num(truth, 1)}. Die Streuung über die Läufe (Standardabweichung) wächst von {num(first['summary'].sd, 2)} auf {num(last['summary'].sd, 2)}, "
            f"die verbleibenden Container fallen von {first['remaining']:.0f} auf {last['remaining']:.0f} von {n}.")
    if farther is not None:
        text += f" In {farther} von {len(ok)} Landmarken liegt das Mittel um mehr als die Hälfte des Betrags der Wahrheit daneben."
    return "info", text


def landmark_cell(mean: float, failed: int, runs: int, remaining: float) -> str:
    """Zelle der Reihe über die Wartezeit: β̂ der Landmarke (verbleibende Container); Läufe ohne Ergebnis stehen dabei, damit ein Mittel aus wenigen Läufen nicht wie ein volles aussieht."""
    if runs == 0:
        return f"kein Ergebnis ({remaining:.0f})"
    text = f"{num(mean, 2)} ({remaining:.0f})"
    return text + (f", {failed} Läufe ohne Ergebnis" if failed else "")
