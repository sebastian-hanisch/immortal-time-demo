"""Messskript der Demo: jede Zahl der README steht in `tools/measure_immortal_time_result.json` (eingefroren, feste Zufalls-Seeds).

Aufruf (Repo-Wurzel):  _venvs/test/Scripts/python.exe -X utf8 tools/measure_immortal_time.py            schreibt die JSON-Datei
                       _venvs/test/Scripts/python.exe -X utf8 tools/measure_immortal_time.py --print    zeigt die Tabellen der README, schreibt nichts

Alle Messungen laufen mit dem Code der Demo (SplitMix64, imt_scenario, imt_model); Läufe je Zelle: C.REPS = 200 (Reihe über die Wartezeit: C.SWEEP_REPS = 100),
Lauf r hat den Zufalls-Seed `run_seed(Zufalls-Seed, r)`. Welt, wenn nichts anderes steht: Lebenszeit lognormal (Median 4 Tage, σ 0,9), Behandlungsbeginn exponentialverteilt
(Mittel 4 Tage), n = 1000 Container, Zufalls-Seed 35, keine Zensierung. Tests (tests/test_claims.py) rechnen die Zellen neu und prüfen, dass die Tabellen der README wörtlich
aus der Datei stammen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imt_constants as C  # noqa: E402
import imt_evaluation as ev  # noqa: E402
import imt_model as M  # noqa: E402
from imt_scenario import Dataset, World, baseline_time, run_seed, simulate  # noqa: E402
from imt_core import SplitMix64  # noqa: E402

RESULT = Path(__file__).with_name("measure_immortal_time_result.json")
SEEDS_STABILITY = (35, 101, 202, 303, 404, 505)   # sechs Zufalls-Seeds für die Stabilitätsprobe
N_SERIES = (200, 500, 1000, 2000)
UPFRONT_SHARE = 0.5                                # Anteil, der bei der Vorab-Zuteilung am Tag 0 behandelt wird


def simulate_upfront(n: int, seed: int, run: int, share: float = UPFRONT_SHARE) -> Dataset:
    """Vergleichswelt ohne Wartezeit: Ein zufällig gewählter Anteil wird am Tag 0 behandelt (unabhängig von der Lebenszeit), die anderen nie; keine Wirkung (β = 0).
    Hier steht der Status zu Beginn der Nachbeobachtung fest, es gibt keine unsterbliche Zeit."""
    rng = SplitMix64(run_seed(seed, run))
    treated = rng.uniform(n) < share
    e = np.maximum(rng.exponential(n), 1e-300)
    return Dataset(baseline_time(e), np.where(treated, 1e-9, 1e12))


def _cell(s: M.Summary) -> dict:
    return {"mean": s.mean, "sd": s.sd, "se_mean": s.se_mean, "runs": s.runs, "failed": s.failed, "coverage": s.coverage, "mean_model_se": s.mean_model_se}


def world_cell(world: World, reps: int = C.REPS, taus=C.LANDMARKS) -> dict:
    """Eine Welt: alle Auswertungen mit ihren Zusammenfassungen, verbleibende Container und Anteil „behandelt“."""
    study = M.run_study(world, reps, taus)
    return {
        "beta": world.beta, "wait": world.wait, "n": world.n, "seed": world.seed, "reps": reps,
        "treated_share": float(study.treated_share.mean()),
        "naive": _cell(M.summarize_naive(study)),
        "timedep": _cell(M.summarize_timedep(study)),
        "landmarks": {f"{t:g}": {**_cell(M.summarize_landmark(study, t)), "remaining": M.mean_remaining(study, t), "treated_at_tau": M.mean_treated_at_tau(study, t)} for t in taus},
    }


def sweep_cell(beta: float) -> list[dict]:
    rows = M.wait_sweep(beta, C.DEFAULT_N, C.DEFAULT_SEED)
    return [{"wait": r["wait"], "treated_share": r["treated_share"], "naive": _cell(r["naive"]), "timedep": _cell(r["timedep"]),
             "landmarks": {f"{t:g}": {**_cell(s), "remaining": r["remaining"][t]} for t, s in r["landmarks"].items()}} for r in rows]


def tau_cell(beta: float) -> list[dict]:
    study = M.run_study(World(beta=beta), C.REPS, C.TAU_GRID)
    return [{"tau": r["tau"], "remaining": r["remaining"], "treated": r["treated"], **_cell(r["summary"])} for r in M.landmark_sweep(study)]


def censored_cell(beta: float, taus=C.LANDMARKS, reps: int = C.REPS) -> dict:
    """Prüfung der Deutung „Verdünnung“: Landmarke, bei der später Behandelte beim Beginn zensiert werden, statt als unbehandelt weiterzuzählen."""
    out = {}
    for tau in taus:
        vals, ses, oks = [], [], []
        for r in range(reps):
            fit = M.fit_rows(M.rows_landmark_censored(simulate(World(beta=beta), r), tau))
            vals.append(fit["beta"])
            ses.append(fit["se"])
            oks.append(fit["ok"])
        out[f"{tau:g}"] = _cell(M.summarize(vals, ses, oks, beta))
    return out


def upfront_cell(reps: int = C.REPS) -> dict:
    """Vorab-Zuteilung am Tag 0 (keine Wartezeit), keine Wirkung: naive Auswertung und Landmarke bei τ = 0 sind dieselbe Rechnung."""
    vals, ses, oks = [], [], []
    for r in range(reps):
        fit = M.estimate_naive(simulate_upfront(C.DEFAULT_N, C.DEFAULT_SEED, r))
        vals.append(fit["beta"])
        ses.append(fit["se"])
        oks.append(fit["ok"])
    return {"share": UPFRONT_SHARE, "n": C.DEFAULT_N, "reps": reps, **_cell(M.summarize(vals, ses, oks, 0.0))}


def measure() -> dict:
    zero, effect = World(beta=0.0), World(beta=-0.5)
    return {
        "zero": world_cell(zero),
        "effect": world_cell(effect),
        "stability": {str(s): {"zero": world_cell(World(beta=0.0, seed=s)), "effect": world_cell(World(beta=-0.5, seed=s))} for s in SEEDS_STABILITY},
        "small_n": {str(n): world_cell(World(beta=-0.5, n=n)) for n in N_SERIES},
        "sweep_zero": sweep_cell(0.0),
        "sweep_effect": sweep_cell(-0.5),
        "tau_zero": tau_cell(0.0),
        "tau_effect": tau_cell(-0.5),
        "censored_effect": censored_cell(-0.5),
        "upfront": upfront_cell(),
        "harm": world_cell(World(beta=0.5)),
    }


# ------------------------------------------------------------------------------------------------ Tabellen der README (aus der Datei gebildet)


def _md(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _pm(c: dict, nd: int = 3) -> str:
    return f"{ev.num(c['mean'], nd)} ± {ev.num(c['se_mean'], 3)}"


def tables(data: dict) -> dict[str, str]:
    z, e = data["zero"], data["effect"]
    out = {}
    out["main"] = _md(
        ["Auswertung", "Wahrheit β = 0: β̂ (Mittel ± Standardfehler)", "Überdeckung", "Wahrheit β = −0,5: β̂", "Überdeckung"],
        [["naiv (je behandelt, ab Tag 0)", _pm(z["naive"]), f"{ev.num(100 * z['naive']['coverage'], 0)} %", _pm(e["naive"]), f"{ev.num(100 * e['naive']['coverage'], 0)} %"]]
        + [[f"Landmarke Tag {t} (bleiben: {ev.num(z['landmarks'][t]['remaining'], 0)} / {ev.num(e['landmarks'][t]['remaining'], 0)})", _pm(z["landmarks"][t]), f"{ev.num(100 * z['landmarks'][t]['coverage'], 0)} %",
            _pm(e["landmarks"][t]), f"{ev.num(100 * e['landmarks'][t]['coverage'], 0)} %"] for t in z["landmarks"]]
        + [["zeitabhängig (aufgeteilte Person-Zeit)", _pm(z["timedep"]), f"{ev.num(100 * z['timedep']['coverage'], 0)} %", _pm(e["timedep"]), f"{ev.num(100 * e['timedep']['coverage'], 0)} %"]],
    )
    for key, label in (("sweep_zero", "zero"), ("sweep_effect", "effect")):
        out[key] = _md(
            ["Mittlere Wartezeit (Tage)", "behandelt (naiv)", "naiv β̂", "Landmarke Tag 2: β̂ (bleiben)", "Landmarke Tag 4: β̂ (bleiben)", "zeitabhängig β̂"],
            [[ev.num(r["wait"], 1), f"{ev.num(100 * r['treated_share'], 0)} %", ev.num(r["naive"]["mean"], 2),
              *[ev.landmark_cell(r['landmarks'][t]['mean'], r['landmarks'][t]['failed'], r['landmarks'][t]['runs'], r['landmarks'][t]['remaining']) for t in ("2", "4")], ev.num(r["timedep"]["mean"], 2)] for r in data[key]],
        )
    out["tau_effect"] = _md(
        ["Landmarke τ (Tage)", "bleiben", "davon behandelt", "β̂ (Mittel ± Standardfehler)", "Streuung zwischen den Läufen", "Überdeckung"],
        [[ev.num(r["tau"], 1), ev.num(r["remaining"], 0), ev.num(r["treated"], 0), _pm(r), ev.num(r["sd"], 2), f"{ev.num(100 * r['coverage'], 0)} %"] for r in data["tau_effect"]],
    )
    out["small_n"] = _md(
        ["Container n", "naiv β̂", "Landmarke Tag 2: β̂ (bleiben)", "zeitabhängig β̂ (Überdeckung)"],
        [[n, _pm(c["naive"]), f"{_pm(c['landmarks']['2'])} ({ev.num(c['landmarks']['2']['remaining'], 0)})",
          f"{_pm(c['timedep'])} ({ev.num(100 * c['timedep']['coverage'], 0)} %)"] for n, c in data["small_n"].items()],
    )
    out["stability"] = _md(
        ["Zufalls-Seed", "naiv β̂ (β = 0)", "Landmarke Tag 2 (β = 0)", "zeitabhängig (β = 0)", "naiv β̂ (β = −0,5)", "Landmarke Tag 2 (β = −0,5)", "zeitabhängig (β = −0,5)"],
        [[s, ev.num(c["zero"]["naive"]["mean"], 3), ev.num(c["zero"]["landmarks"]["2"]["mean"], 3), ev.num(c["zero"]["timedep"]["mean"], 3),
          ev.num(c["effect"]["naive"]["mean"], 3), ev.num(c["effect"]["landmarks"]["2"]["mean"], 3), ev.num(c["effect"]["timedep"]["mean"], 3)] for s, c in data["stability"].items()],
    )
    return out


def values(data: dict) -> dict[str, str]:
    """Die Prosa-Zahlen der README (Platzhalter `{{name}}` im Text), aus der Datei gebildet."""
    z, e = data["zero"], data["effect"]
    sw0, sw1 = data["sweep_zero"], data["sweep_effect"]
    tau = data["tau_effect"]
    st = data["stability"].values()

    def rng(vals, nd=3):
        return f"{ev.num(min(vals), nd)} und {ev.num(max(vals), nd)}"

    return {
        "zero.naive": _pm(z["naive"]), "zero.hr": ev.hr(z["naive"]["mean"]), "zero.share": f"{ev.num(100 * z['treated_share'], 0)} %",
        "zero.lm2": _pm(z["landmarks"]["2"]), "zero.lm4": _pm(z["landmarks"]["4"]), "zero.td": _pm(z["timedep"]),
        "effect.naive": _pm(e["naive"]), "effect.hr": ev.hr(e["naive"]["mean"]), "effect.td": _pm(e["timedep"]),
        "effect.lm2share": f"{ev.num(100 * e['landmarks']['2']['mean'] / e['beta'], 0)} %", "effect.lm4share": f"{ev.num(100 * e['landmarks']['4']['mean'] / e['beta'], 0)} %",
        "sw0.naive05": ev.num(sw0[0]["naive"]["mean"], 2), "sw0.naive16": ev.num(sw0[-1]["naive"]["mean"], 2),
        "sw1.lm05": ev.num(sw1[0]["landmarks"]["2"]["mean"], 2), "sw1.lm16": ev.num(sw1[-1]["landmarks"]["2"]["mean"], 2),
        "tau.min": ev.num(min(r["mean"] for r in tau), 3), "tau.max": ev.num(max(r["mean"] for r in tau), 3),
        "tau.sd0": ev.num(tau[0]["sd"], 2), "tau.sd1": ev.num(tau[-1]["sd"], 2), "tau.rem0": ev.num(tau[0]["remaining"], 0), "tau.rem1": ev.num(tau[-1]["remaining"], 0),
        "cens2": _pm(data["censored_effect"]["2"]), "cens4": _pm(data["censored_effect"]["4"]),
        "cov.lm2": f"{ev.num(100 * e['landmarks']['2']['coverage'], 0)} %", "cov.lm4": f"{ev.num(100 * e['landmarks']['4']['coverage'], 0)} %", "cov.td": f"{ev.num(100 * e['timedep']['coverage'], 0)} %",
        "harm.naive": _pm(data["harm"]["naive"]), "harm.lm2": _pm(data["harm"]["landmarks"]["2"]), "harm.lm4": _pm(data["harm"]["landmarks"]["4"]), "harm.td": _pm(data["harm"]["timedep"]),
        "stab.nz": rng([c["zero"]["naive"]["mean"] for c in st]), "stab.le": rng([c["effect"]["landmarks"]["2"]["mean"] for c in st]), "stab.te": rng([c["effect"]["timedep"]["mean"] for c in st]),
        "upfront": _pm(data["upfront"]),
    }


if __name__ == "__main__":
    result = measure()
    if "--print" in sys.argv:
        for name, table in tables(result).items():
            print(f"\n## {name}\n{table}")
        for k in ("censored_effect", "upfront", "harm"):
            print(f"\n## {k}\n{json.dumps(result[k], indent=1, ensure_ascii=False)[:1500]}")
    else:
        RESULT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print("geschrieben:", RESULT)
