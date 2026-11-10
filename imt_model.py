"""Modell: die drei Auswertungen (naiv, Landmarke, zeitabhängig), das Handbeispiel in Brüchen und die Läufe über viele Datensätze.

Alle drei Auswertungen sind dieselbe Cox-Anpassung mit einem 0/1-Merkmal („behandelt“) und verschiedenen Zeilen:
  naiv         eine Zeile je Container, Status „behandelt“ = der Behandlungsbeginn liegt vor dem Ausfall, ab Zeit 0 (die Wartezeit zählt als behandelt);
  Landmarke    nur Container, die den Zeitpunkt τ erleben; Status = Behandlung bis τ begonnen; Risikomenge erst ab τ (Eintritt τ), spätere Behandlungen bleiben unbeachtet;
  zeitabhängig die Person-Zeit wird aufgeteilt: Zeile (0, S) unbehandelt, Zeile (S, T) behandelt, die Ausfälle gehören zur letzten Zeile.
Für die vielen Läufe (Zahl der Läufe mal fünf Anpassungen) steht `cox_binary`: die Newton-Rechnung für ein 0/1-Merkmal mit sortierten Zählungen statt Schleifen über die
Ereigniszeiten. Sie ist gegen die kopierte `imt_core.cox_fit` geprüft (tests/test_model.py, tests/test_oracle_immortal_time.py); für einzelne Datensätze in der App läuft `imt_core.cox_fit`.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import imt_constants as C
import imt_core as K
from imt_scenario import Dataset, World, simulate, treated_ever

_NO_CONVERGENCE = ("Keine Konvergenz; bei perfekter Trennung der Gruppen (alle Ereignisse in einer Gruppe zuerst) gibt es keinen endlichen Schätzer.")
_SINGULAR = "Die Information ist singulär (konstantes oder kollineares Merkmal, oder zu wenige Ereignisse)."


# ------------------------------------------------------------------------------------------------ Cox mit einem 0/1-Merkmal, schnell


def cox_binary(start, stop, event, x, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Cox-Partial-Likelihood (Breslow) für ein 0/1-Merkmal und Zeilen (Start, Ende]; gleiche Rechnung wie `imt_core.cox_fit` mit einem Merkmal, aber die Summen über die
    Risikomengen kommen aus sortierten Zählungen (je Gruppe: Zeilen mit Start < t minus Zeilen mit Ende < t). Gibt beta, se, ok, message (wie cox_fit, mit Skalaren)."""
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    ev = np.asarray(event, dtype=bool)
    xb = np.asarray(x, dtype=bool)
    ts, inv = np.unique(stop[ev], return_inverse=True)
    if ts.size == 0:
        return {"beta": float("nan"), "se": float("nan"), "ok": False, "message": "Ohne Ereignis gibt es keinen Schätzer."}
    d = np.bincount(inv, minlength=ts.size).astype(float)
    d1 = np.bincount(inv, weights=xb[ev].astype(float), minlength=ts.size)
    n1 = _at_risk(start[xb], stop[xb], ts)
    n0 = _at_risk(start[~xb], stop[~xb], ts)
    sum_d1 = float(d1.sum())

    def evaluate(beta):
        if abs(beta) > 60.0:   # weit außerhalb jedes sinnvollen Bereichs: die Schrittweitenhalbierung kürzt den Schritt
            return None
        w = n1 * np.exp(beta)
        s0 = n0 + w
        if np.any(s0 <= 0):
            return None
        p = w / s0
        ll = sum_d1 * beta - float(np.sum(d * np.log(s0)))
        return ll, sum_d1 - float(np.sum(d * p)), float(np.sum(d * p * (1.0 - p)))

    beta = 0.0
    cur = evaluate(beta)
    if cur is None:
        return {"beta": float("nan"), "se": float("nan"), "ok": False, "message": "Die Risikomengen sind leer."}
    ll, grad, info = cur
    it = 0
    for it in range(1, max_iter + 1):
        if not info > 1e-12:
            return {"beta": float("nan"), "se": float("nan"), "ok": False, "message": _SINGULAR}
        step = grad / info
        scale = 1.0
        for _ in range(40):
            new = evaluate(beta + scale * step)
            if new is not None and new[0] >= ll - 1e-12:
                break
            scale *= 0.5
        else:
            break
        beta = beta + scale * step
        converged = abs(new[0] - ll) < tol * (1.0 + abs(ll)) and abs(scale * step) < 1e-7
        ll, grad, info = new
        if converged:
            se = float(np.sqrt(abs(1.0 / info))) if info > 0 else float("inf")
            if not np.isfinite(se) or se > 50.0 or abs(beta) > 20.0:
                break
            return {"beta": float(beta), "se": se, "ok": True, "message": ""}
        if abs(beta) > 25:
            break
    return {"beta": float(beta), "se": float("nan"), "ok": False, "message": _NO_CONVERGENCE}


def _at_risk(start, stop, ts):
    """Zahl der Zeilen mit Start < t ≤ Ende an den Zeiten ts."""
    return (np.searchsorted(np.sort(start), ts, side="left") - np.searchsorted(np.sort(stop), ts, side="left")).astype(float)


# ------------------------------------------------------------------------------------------------ die drei Auswertungen als Zeilen


def rows_naive(ds: Dataset):
    """(Start, Ende, Ereignis, Merkmal) der naiven Auswertung: eine Zeile je Container ab Zeit 0, alle Ausfälle sind Ereignisse."""
    n = ds.n
    return np.zeros(n), ds.time, np.ones(n, bool), treated_ever(ds)


def rows_landmark(ds: Dataset, tau: float):
    """Zeilen der Landmarke τ: nur Container mit T > τ, Eintritt τ, Merkmal = Behandlung bis τ begonnen (S ≤ τ)."""
    keep = ds.time > tau
    m = int(keep.sum())
    return np.full(m, float(tau)), ds.time[keep], np.ones(m, bool), ds.start[keep] <= tau


def rows_split(ds: Dataset):
    """Aufgeteilte Person-Zeit: ein Container mit S < T ergibt (0, S) unbehandelt ohne Ereignis und (S, T) behandelt mit Ereignis, jeder andere (0, T) unbehandelt mit Ereignis."""
    treated = treated_ever(ds)
    idx = np.where(treated)[0]
    start = np.concatenate([np.zeros(ds.n), ds.start[idx]])
    stop = np.concatenate([np.where(treated, ds.start, ds.time), ds.time[idx]])
    event = np.concatenate([~treated, np.ones(idx.size, bool)])
    x = np.concatenate([np.zeros(ds.n, bool), np.ones(idx.size, bool)])
    return start, stop, event, x


def fit_rows(rows, fast: bool = True) -> dict:
    """Cox-Anpassung auf Zeilen (Start, Ende, Ereignis, Merkmal): schnell (Läufe) oder mit der kopierten `imt_core.cox_fit` (einzelne Datensätze). Ein Fehler der Eingabe wird zur Meldung."""
    start, stop, event, x = rows
    if fast:
        return cox_binary(start, stop, event, x)
    try:
        r = K.cox_fit(stop, event, np.asarray(x, dtype=float), entry=start)
    except K.SurvivalError as err:
        return {"beta": float("nan"), "se": float("nan"), "ok": False, "message": str(err)}
    return {"beta": float(r["beta"][0]), "se": float(r["se"][0]), "ok": bool(r["ok"]), "message": r["message"]}


def estimate_naive(ds: Dataset, fast: bool = True) -> dict:
    return fit_rows(rows_naive(ds), fast)


def estimate_landmark(ds: Dataset, tau: float, fast: bool = True) -> dict:
    """Landmarke τ; mit Zahl der verbleibenden Container und der davon bis τ behandelten."""
    start, stop, event, x = rows_landmark(ds, tau)
    out = fit_rows((start, stop, event, x), fast)
    out["remaining"] = int(stop.size)
    out["treated_at_tau"] = int(np.sum(x))
    return out


def estimate_timedep(ds: Dataset, fast: bool = True) -> dict:
    return fit_rows(rows_split(ds), fast)


# ------------------------------------------------------------------------------------------------ Handbeispiel in Brüchen

PEOPLE = ((2, 1), (3, 5), (4, Fraction(5, 2)), (6, None), (8, 7), (10, 3))   # (Zeit T, Behandlungsbeginn S oder None = nie)


def hand_naive(people=PEOPLE) -> dict:
    """Naiv: „behandelt“, wenn S vor T liegt. Gibt je Gruppe die Zeiten und das Mittel (als Bruch; None bei leerer Gruppe)."""
    treated = [Fraction(t) for t, s in people if s is not None and s < t]
    untreated = [Fraction(t) for t, s in people if not (s is not None and s < t)]
    return _group_means(treated, untreated)


def hand_landmark(tau, people=PEOPLE) -> dict:
    """Landmarke τ: nur Container mit T > τ; behandelt, wenn S ≤ τ."""
    tau = Fraction(tau)
    alive = [(t, s) for t, s in people if t > tau]
    treated = [Fraction(t) for t, s in alive if s is not None and s <= tau]
    untreated = [Fraction(t) for t, s in alive if not (s is not None and s <= tau)]
    return _group_means(treated, untreated)


def _group_means(treated, untreated) -> dict:
    return {
        "treated": treated, "untreated": untreated,
        "mean_treated": sum(treated, Fraction(0)) / len(treated) if treated else None,
        "mean_untreated": sum(untreated, Fraction(0)) / len(untreated) if untreated else None,
    }


def hand_split(people=PEOPLE) -> list[tuple[int, Fraction, Fraction, bool, bool]]:
    """Aufgeteilte Person-Zeit des Handbeispiels: Zeilen (Container, Start, Ende, behandelt, Ausfall) mit Ausfall nur in der letzten Zeile je Container."""
    rows = []
    for i, (t, s) in enumerate(people, start=1):
        if s is not None and s < t:
            rows.append((i, Fraction(0), Fraction(s), False, False))
            rows.append((i, Fraction(s), Fraction(t), True, True))
        else:
            rows.append((i, Fraction(0), Fraction(t), False, True))
    return rows


# ------------------------------------------------------------------------------------------------ Läufe über viele Datensätze


@dataclass(frozen=True)
class Study:
    """Die Schätzwerte der Läufe (Zeile je Lauf). `lm_*` haben eine Spalte je Landmarke in `taus`."""
    world: World
    reps: int
    taus: tuple
    naive: np.ndarray
    naive_se: np.ndarray
    naive_ok: np.ndarray
    timedep: np.ndarray
    timedep_se: np.ndarray
    timedep_ok: np.ndarray
    lm: np.ndarray
    lm_se: np.ndarray
    lm_ok: np.ndarray
    lm_remaining: np.ndarray
    lm_treated: np.ndarray
    treated_share: np.ndarray   # Anteil der Container mit Behandlungsbeginn vor dem Ausfall (naiver Status), je Lauf


def run_study(world: World, reps: int = C.REPS, taus=C.LANDMARKS, with_timedep: bool = True) -> Study:
    """`reps` Datensätze der Welt (feste Zufalls-Seeds: Lauf r hat den Strom run_seed(world.seed, r)); je Datensatz die naive Auswertung, jede Landmarke in `taus` und die zeitabhängige."""
    taus = tuple(float(t) for t in taus)
    nv = np.full((2, reps), np.nan)
    td = np.full((2, reps), np.nan)
    nv_ok = np.zeros(reps, bool)
    td_ok = np.zeros(reps, bool)
    lm = np.full((len(taus), reps), np.nan)
    lm_se = np.full((len(taus), reps), np.nan)
    lm_ok = np.zeros((len(taus), reps), bool)
    rem = np.zeros((len(taus), reps), int)
    trt = np.zeros((len(taus), reps), int)
    share = np.zeros(reps)
    for r in range(reps):
        ds = simulate(world, r)
        share[r] = float(treated_ever(ds).mean())
        a = estimate_naive(ds)
        nv[:, r] = (a["beta"], a["se"])
        nv_ok[r] = a["ok"]
        for j, tau in enumerate(taus):
            b = estimate_landmark(ds, tau)
            lm[j, r], lm_se[j, r], lm_ok[j, r] = b["beta"], b["se"], b["ok"]
            rem[j, r], trt[j, r] = b["remaining"], b["treated_at_tau"]
        if with_timedep:
            c = estimate_timedep(ds)
            td[:, r] = (c["beta"], c["se"])
            td_ok[r] = c["ok"]
    return Study(world, reps, taus, nv[0], nv[1], nv_ok, td[0], td[1], td_ok, lm.T, lm_se.T, lm_ok.T, rem.T, trt.T, share)


@dataclass(frozen=True)
class Summary:
    """Zusammenfassung eines Schätzers über die Läufe, in denen er konvergierte."""
    mean: float
    sd: float
    se_mean: float          # Standardfehler des Mittels = sd / sqrt(k)
    runs: int               # Läufe mit Ergebnis (konvergiert)
    failed: int             # Läufe ohne Ergebnis (z. B. perfekte Trennung)
    coverage: float         # Anteil der Läufe, in denen β̂ ± 1,96·SE den wahren Wert der Welt enthält
    mean_model_se: float    # mittlerer Standardfehler aus der Information (Modell)


def summarize(values, ses, oks, truth: float) -> Summary:
    values = np.asarray(values, dtype=float)
    oks = np.asarray(oks, dtype=bool)
    v = values[oks]
    s = np.asarray(ses, dtype=float)[oks]
    k = int(v.size)
    if k == 0:
        return Summary(float("nan"), float("nan"), float("nan"), 0, int(oks.size), float("nan"), float("nan"))
    sd = float(v.std(ddof=1)) if k > 1 else float("nan")
    cover = float(np.mean(np.abs(v - truth) <= C.CI_Z * s))
    return Summary(float(v.mean()), sd, sd / np.sqrt(k) if k > 1 else float("nan"), k, int(oks.size) - k, cover, float(s.mean()))


def summarize_naive(study: Study) -> Summary:
    return summarize(study.naive, study.naive_se, study.naive_ok, study.world.beta)


def summarize_timedep(study: Study) -> Summary:
    return summarize(study.timedep, study.timedep_se, study.timedep_ok, study.world.beta)


def summarize_landmark(study: Study, tau: float) -> Summary:
    j = study.taus.index(float(tau))
    return summarize(study.lm[:, j], study.lm_se[:, j], study.lm_ok[:, j], study.world.beta)


def mean_remaining(study: Study, tau: float) -> float:
    return float(study.lm_remaining[:, study.taus.index(float(tau))].mean())


def mean_treated_at_tau(study: Study, tau: float) -> float:
    return float(study.lm_treated[:, study.taus.index(float(tau))].mean())


def wait_sweep(beta: float, n: int, seed: int, waits=C.SWEEP_WAITS, reps: int = C.SWEEP_REPS, taus=C.LANDMARKS) -> list[dict]:
    """Die Schätzer über die mittlere Wartezeit (gepaarte Welten: dieselben Zufallszahlen, nur die Wartezeit ändert sich). Je Wartezeit ein Eintrag mit den Zusammenfassungen."""
    rows = []
    for w in waits:
        study = run_study(World(beta=beta, wait=float(w), n=n, seed=seed), reps, taus)
        rows.append({
            "wait": float(w),
            "naive": summarize_naive(study),
            "timedep": summarize_timedep(study),
            "landmarks": {tau: summarize_landmark(study, tau) for tau in taus},
            "remaining": {tau: mean_remaining(study, tau) for tau in taus},
            "treated_share": float(study.treated_share.mean()),
        })
    return rows


def landmark_sweep(study: Study) -> list[dict]:
    """Die Landmarken-Reihe über alle τ des Studienobjekts: verbleibende Container und β̂ je τ."""
    return [{"tau": tau, "summary": summarize_landmark(study, tau), "remaining": mean_remaining(study, tau), "treated": mean_treated_at_tau(study, tau)} for tau in study.taus]


def hand_rows(tau, people=PEOPLE) -> list[dict]:
    """Zeilen der Tabelle des Handbeispiels: je Container naive Gruppe und Gruppe der Landmarke τ („fällt weg“, wenn T ≤ τ)."""
    tau = Fraction(tau)
    out = []
    for i, (t, s) in enumerate(people, start=1):
        naive = "behandelt" if (s is not None and s < t) else "unbehandelt"
        landmark = "fällt weg" if t <= tau else ("behandelt" if (s is not None and s <= tau) else "unbehandelt")
        out.append({"container": i, "t": Fraction(t), "s": None if s is None else Fraction(s), "naive": naive, "landmark": landmark})
    return out


def rows_landmark_censored(ds: Dataset, tau: float):
    """Prüfvariante der Landmarke (nicht in der App): wer bei τ noch unbehandelt ist und später behandelt wird, wird bei seinem Behandlungsbeginn zensiert, statt als unbehandelt
    weiterzuzählen. Nur Container mit T > τ; Eintritt τ."""
    keep = ds.time > tau
    t, s = ds.time[keep], ds.start[keep]
    treated = s <= tau
    stop = np.where(treated, t, np.minimum(t, s))
    event = np.where(treated, True, t < s)
    return np.full(t.size, float(tau)), stop, event, treated
