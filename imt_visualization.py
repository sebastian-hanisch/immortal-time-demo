"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import imt_constants as C
import imt_model as M


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h", yanchor="top", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_hand(people, tau):
    """Sechs Zeitbalken: grau = unbehandelt, rot schraffiert = Zeit vor dem Behandlungsbeginn bei einem naiv als behandelt gezählten Container (die „unsterbliche Zeit“),
    blau = behandelt. Senkrechte gestrichelte Linie: die Landmarke τ."""
    fig = go.Figure()
    names = [f"Container {i}" for i in range(1, len(people) + 1)]
    seen = set()

    def bar(name, x0, x1, color, label, pattern=False):
        show = label not in seen
        seen.add(label)
        fig.add_trace(go.Bar(
            y=[name], x=[float(x1 - x0)], base=[float(x0)], orientation="h", marker=dict(color=color, pattern_shape="/" if pattern else "", line=dict(width=0)),
            name=label, legendgroup=label, showlegend=show, hovertemplate=f"{label}: Tag {float(x0):g} bis {float(x1):g}<extra></extra>"))

    for name, (t, s) in zip(names, people):
        if s is not None and s < t:
            bar(name, 0, s, C.COLOR_NAIVE, "Zeit vor dem Beginn (naiv als behandelt gezählt)", pattern=True)
            bar(name, s, t, C.COLOR_TREATED, "behandelt")
        else:
            bar(name, 0, t, C.COLOR_UNTREATED, "unbehandelt")
    fig.add_trace(go.Scatter(y=names, x=[float(t) for t, _ in people], mode="markers", marker=dict(symbol="x", size=9, color="black"), name="Ausfall", hovertemplate="Ausfall: Tag %{x:g}<extra></extra>"))
    fig.add_vline(x=float(tau), line_dash="dash", line_color=C.COLOR_TRUTH, annotation_text=f"τ = {float(tau):g}", annotation_position="top")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Tage seit der Ankunft", range=[0, 10.6])
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 360)


def build_estimates(study: M.Study, tau: float | None, show=("naive", "landmark", "timedep")):
    """Die β̂ aller Läufe (je Lauf ein Punkt, Kasten = mittlere Hälfte) für die gewählten Auswertungen; graue Linie: die Wahrheit."""
    fig = go.Figure()
    entries = []
    if "naive" in show:
        entries.append(("naiv (je behandelt, ab Tag 0)", study.naive[study.naive_ok], C.COLOR_NAIVE))
    if "landmark" in show and tau is not None:
        j = study.taus.index(float(tau))
        entries.append((f"Landmarke τ = {tau:g}", study.lm[study.lm_ok[:, j], j], C.COLOR_LANDMARK))
    if "timedep" in show:
        entries.append(("zeitabhängig", study.timedep[study.timedep_ok], C.COLOR_TIMEDEP))
    for label, values, color in entries:
        fig.add_trace(go.Box(y=np.round(values, 4).tolist(), name=label, marker_color=color, boxpoints="all", jitter=0.5, pointpos=0, boxmean=True, hovertemplate="β̂ = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=study.world.beta, line_color=C.COLOR_TRUTH, line_width=2, annotation_text=f"Wahrheit β = {study.world.beta:g}", annotation_position="top left")
    fig.update_yaxes(title_text="geschätztes β̂ (log-Hazard-Verhältnis)", zeroline=False)
    fig.update_layout(showlegend=False)
    return _base(fig, 400)


def build_wait_sweep(rows, truth: float, current_wait: float):
    """β̂ (Mittel über die Läufe) gegen die mittlere Wartezeit (logarithmische Achse): naiv, Landmarken, zeitabhängig; graue Linie: die Wahrheit."""
    xs = [r["wait"] for r in rows]
    fig = go.Figure()

    def line(label, key, color, dash="solid", symbol="circle"):
        ys = [key(r).mean if key(r).runs else None for r in rows]
        err = [2 * key(r).se_mean if key(r).runs > 1 else None for r in rows]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=label, line=dict(color=color, width=3, dash=dash), marker=dict(symbol=symbol, size=8),
                                 error_y=dict(type="data", array=err, visible=True, thickness=1), hovertemplate="Wartezeit %{x:g} Tage: β̂ = %{y:.3f}<extra>" + label + "</extra>"))

    line("naiv", lambda r: r["naive"], C.COLOR_NAIVE)
    for tau, dash in zip(sorted(rows[0]["landmarks"]), ("dash", "dot")):
        line(f"Landmarke τ = {tau:g}", lambda r, t=tau: r["landmarks"][t], C.COLOR_LANDMARK, dash, "diamond")
    line("zeitabhängig", lambda r: r["timedep"], C.COLOR_TIMEDEP, "solid", "square")
    fig.add_hline(y=truth, line_color=C.COLOR_TRUTH, line_width=2, annotation_text=f"Wahrheit β = {truth:g}", annotation_position="bottom right")
    fig.add_vline(x=current_wait, line_dash="dot", line_color=C.COLOR_TREATED, annotation_text="Wartezeit der Seitenleiste", annotation_position="top")
    fig.update_xaxes(title_text="mittlere Wartezeit bis zur Behandlung (Tage, logarithmisch)", type="log", tickvals=xs, ticktext=[f"{x:g}" for x in xs])
    fig.update_yaxes(title_text="geschätztes β̂ (Mittel, Balken ± 2 Standardfehler)", zeroline=False)
    return _base(fig, 420)


def build_tau_sweep(summary_rows, truth: float, tau: float):
    """Oben β̂ gegen τ (Mittel, Band ± 1 Standardabweichung über die Läufe), unten die verbleibenden Container."""
    taus = [r["tau"] for r in summary_rows]
    means = [r["summary"].mean for r in summary_rows]
    sds = [r["summary"].sd if r["summary"].runs > 1 else 0.0 for r in summary_rows]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=taus + taus[::-1], y=[m + s for m, s in zip(means, sds)] + [m - s for m, s in zip(means, sds)][::-1], fill="toself", fillcolor="rgba(221,132,82,0.2)",
                             line=dict(width=0), hoverinfo="skip", name="± 1 Standardabweichung über die Läufe"), row=1, col=1)
    fig.add_trace(go.Scatter(x=taus, y=means, mode="lines+markers", line=dict(color=C.COLOR_LANDMARK, width=3), name="Landmarke, Mittel von β̂", hovertemplate="τ = %{x:g}: β̂ = %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=truth, line_color=C.COLOR_TRUTH, line_width=2, annotation_text=f"Wahrheit β = {truth:g}", annotation_position="bottom right", row=1, col=1)
    fig.add_trace(go.Bar(x=taus, y=[r["remaining"] for r in summary_rows], marker_color=C.COLOR_TREATED, name="verbleibende Container", hovertemplate="τ = %{x:g}: %{y:.0f} Container<extra></extra>"), row=2, col=1)
    fig.add_vline(x=tau, line_dash="dash", line_color=C.COLOR_TRUTH)
    fig.update_yaxes(title_text="β̂", row=1, col=1, zeroline=False)
    fig.update_yaxes(title_text="Container", row=2, col=1)
    fig.update_xaxes(title_text="Landmarke τ (Tage)", row=2, col=1)
    return _base(fig, 520)
