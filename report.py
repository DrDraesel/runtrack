"""Report generation for RunTrack: matplotlib charts + self-contained HTML.

The HTML report is print-ready: @media print rules produce a clean A4/Letter
handout and a "Print / Save as PDF" button is included in the page.
"""
from __future__ import annotations

import csv
import html
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import advisor
import config
import cycling as CY
import gait_events as GA
from session_store import Store

CLR_L = "#e4572e"
CLR_R = "#17a2b8"


def _load_cols(csv_path):
    cols: dict = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                for k, v in row.items():
                    if k is None:
                        continue
                    try:
                        val = float(v) if v not in ("", None) else np.nan
                    except (TypeError, ValueError):
                        val = np.nan
                    cols.setdefault(k, []).append(val)
    except FileNotFoundError:
        return {}
    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def _series(cols, key):
    return cols.get(key, np.array([]))


def make_charts(sid: int, cols: dict, out_dir) -> list[str]:
    made = []
    t = _series(cols, "t_rel")
    if len(t) == 0:
        return made

    def _ok(key):
        y = _series(cols, key)
        return len(y) == len(t) and np.sum(~np.isnan(y)) > 3

    def _plot(ax, key, label, color):
        if _ok(key):
            ax.plot(t, _series(cols, key), lw=1.2, color=color, label=label)

    def _style(ax, title, ylabel):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="upper right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def _empty(ax, msg):
        ax.text(0.5, 0.55, msg, ha="center", va="center", fontsize=9.5,
                color="#777", transform=ax.transAxes)

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p)
        plt.close(fig)
        made.append(name)

    def _mark_events(ax, events, colors=None):
        """Vertical markers for latched gait events (for the report charts)."""
        for e in events or []:
            c = (colors or {}).get((e.get("type"), e.get("side")), "#999")
            ax.axvline(float(e.get("t", 0.0)), color=c, lw=0.7, alpha=0.7)

    # 1. knee angles
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=110)
    _plot(ax, "left_knee", "knee L", CLR_L)
    _plot(ax, "right_knee", "knee R", CLR_R)
    _style(ax, "Knee angle", "deg")
    if not (_ok("left_knee") or _ok("right_knee")):
        _empty(ax, "No knee data in this session — legs were not visible in frame")
    fig.tight_layout()
    _save(fig, "chart_knee.png")

    # 2. hip + ankle
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=110)
    _plot(ax, "left_hip", "hip L", CLR_L)
    _plot(ax, "right_hip", "hip R", CLR_R)
    _plot(ax, "left_ankle", "ankle L", "#7a5195")
    _plot(ax, "right_ankle", "ankle R", "#ef5675")
    _style(ax, "Hip & ankle angles", "deg")
    if not any(_ok(k) for k in ("left_hip", "right_hip", "left_ankle", "right_ankle")):
        _empty(ax, "No hip/ankle data in this session — legs were not visible in frame")
    fig.tight_layout()
    _save(fig, "chart_hip_ankle.png")

    # 3. elbow + trunk
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.4), dpi=110, sharex=True)
    _plot(ax1, "left_elbow", "elbow L", CLR_L)
    _plot(ax1, "right_elbow", "elbow R", CLR_R)
    _style(ax1, "Elbow angle", "deg")
    if not (_ok("left_elbow") or _ok("right_elbow")):
        _empty(ax1, "No elbow data in this session")
    _plot(ax2, "trunk_lean", "trunk lean (raw)", "#003f5c")
    _plot(ax2, "trunk_lean_fwd", "trunk lean (forward +)", "#ffa600")
    _style(ax2, "Trunk lean vs vertical", "deg")
    if not (_ok("trunk_lean") or _ok("trunk_lean_fwd")):
        _empty(ax2, "No trunk data in this session")
    fig.tight_layout()
    _save(fig, "chart_arm_trunk.png")

    # 4. ankle traces (strike detection view)
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=110)
    for key, label, color in (("l_ankle_y", "ankle L y", CLR_L),
                              ("r_ankle_y", "ankle R y", CLR_R)):
        if _ok(key):
            ax.plot(t, _series(cols, key), lw=1.0, color=color, label=label)
    ax.invert_yaxis()
    _style(ax, "Ankle vertical position (foot strikes near the peaks)", "norm y (inverted)")
    if not (_ok("l_ankle_y") or _ok("r_ankle_y")):
        _empty(ax, "No ankle data in this session")
    fig.tight_layout()
    _save(fig, "chart_ankles.png")

    # 5. hip oscillation
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=110)
    if _ok("_midhip_y"):
        y = _series(cols, "_midhip_y")
        sm = y.copy()
        if len(sm) > 9:
            k = np.ones(9) / 9
            sm = np.convolve(sm, k, mode="same")
        ax.plot(t, y - sm, lw=1.2, color="#bc5090", label="hip oscillation")
    _style(ax, "Hip vertical oscillation (detrended)", "norm units")
    if not _ok("_midhip_y"):
        _empty(ax, "No hip data in this session")
    fig.tight_layout()
    _save(fig, "chart_osc.png")

    # 6. sagittal gait metrics (shin angle, overstride)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.4), dpi=110, sharex=True)
    _plot(ax1, "shin_angle_l", "shin L", CLR_L)
    _plot(ax1, "shin_angle_r", "shin R", CLR_R)
    ax1.axhspan(0, 5, color="#4ade80", alpha=0.12)
    ax1.axhline(8, color="#f87171", lw=0.8, ls="--")
    _style(ax1, "Touchdown shin angle vs vertical (green = 0-5 deg optimal)", "deg")
    if not (_ok("shin_angle_l") or _ok("shin_angle_r")):
        _empty(ax1, "No shin-angle data in this session")
    _plot(ax2, "overstride_l", "overstride L", CLR_L)
    _plot(ax2, "overstride_r", "overstride R", CLR_R)
    _style(ax2, "Overstride vector (ankle ahead of pelvis, norm units)", "norm")
    if not (_ok("overstride_l") or _ok("overstride_r")):
        _empty(ax2, "No overstride data in this session")
    fig.tight_layout()
    _save(fig, "chart_sagittal.png")

    # 7. frontal metrics (pelvic tilt, knee valgus)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.4), dpi=110, sharex=True)
    _plot(ax1, "pelvic_tilt_deg", "pelvic tilt (+ = right hip low)", "#b45309")
    ax1.axhline(0, color="#666", lw=0.7)
    _style(ax1, "Inter-hip line tilt (contralateral pelvic drop)", "deg")
    if not _ok("pelvic_tilt_deg"):
        _empty(ax1, "No pelvis data in this session")
    _plot(ax2, "knee_valgus_l", "valgus L", CLR_L)
    _plot(ax2, "knee_valgus_r", "valgus R", CLR_R)
    ax2.axhline(0, color="#666", lw=0.7)
    _style(ax2, "Frontal knee collapse vs hip-ankle axis (+ = valgus)", "deg")
    if not (_ok("knee_valgus_l") or _ok("knee_valgus_r")):
        _empty(ax2, "No frontal knee data in this session")
    fig.tight_layout()
    _save(fig, "chart_frontal.png")

    # 8. cycling: pedal-stroke knee angle with bottom-dead-centre markers
    if _ok("left_knee") or _ok("right_knee"):
        fig, ax = plt.subplots(figsize=(9, 2.8), dpi=110)
        _plot(ax, "left_knee", "knee L", CLR_L)
        _plot(ax, "right_knee", "knee R", CLR_R)
        for key, color in (("left_knee", CLR_L), ("right_knee", CLR_R)):
            if not _ok(key):
                continue
            y = _series(cols, key)
            for i in range(1, len(y) - 1):
                if (np.isfinite(y[i - 1]) and np.isfinite(y[i]) and np.isfinite(y[i + 1])
                        and y[i] >= y[i - 1] and y[i] > y[i + 1] and y[i] > 95):
                    ax.axvline(t[i], color=color, lw=0.6, alpha=0.45)
        ax.axhspan(140, 150, color="#4ade80", alpha=0.10)
        _style(ax, "Pedal stroke — knee angle (green band = BDC fit target "
                   "140-150 road; lines = bottom dead centre)", "deg")
        fig.tight_layout()
        _save(fig, "chart_bike_knee.png")

    # 9. rear camera: hip travel + knee lateral travel (second camera)
    if _ok("r__midhip_y") or _ok("r_l_knee_x") or _ok("r_pelvic_tilt_deg"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.4), dpi=110, sharex=True)
        if _ok("r__midhip_y"):
            y = _series(cols, "r__midhip_y")
            sm = y.copy()
            if len(sm) > 9:
                k = np.ones(9) / 9
                sm = np.convolve(sm, k, mode="same")
            ax1.plot(t, y - sm, lw=1.2, color="#bc5090", label="hip vertical travel")
        if _ok("r_pelvic_tilt_deg"):
            ax1.plot(t, _series(cols, "r_pelvic_tilt_deg"), lw=1.0, color="#b45309",
                     label="pelvic tilt (deg)")
        ax1.axhline(0, color="#666", lw=0.7)
        _style(ax1, "Rear view — hip rocking / pelvic obliquity", "norm / deg")
        if not (_ok("r__midhip_y") or _ok("r_pelvic_tilt_deg")):
            _empty(ax1, "No rear hip data")
        for key, label, color in (("r_l_knee_x", "knee L lateral", CLR_L),
                                  ("r_r_knee_x", "knee R lateral", CLR_R)):
            if _ok(key):
                y = _series(cols, key)
                ax2.plot(t, y - np.nanmean(y), lw=1.2, color=color, label=label)
        _style(ax2, "Rear view — knee lateral travel (centred)", "norm units")
        if not (_ok("r_l_knee_x") or _ok("r_r_knee_x")):
            _empty(ax2, "No rear knee data")
        fig.tight_layout()
        _save(fig, "chart_bike_rear.png")

    # 10. aero: frontal-area silhouette share over time
    for key, label in (("fa_frac", "frontal area (primary cam)"),
                       ("r_fa_frac", "frontal area (second cam)")):
        if _ok(key):
            fig, ax = plt.subplots(figsize=(9, 2.4), dpi=110)
            y = _series(cols, key)
            ax.plot(t, y, lw=1.2, color="#2563eb", label=label)
            ax.axhline(float(np.nanmedian(y)), color="#f59e0b", lw=1.0, ls="--",
                       label="median (position value)")
            _style(ax, "Frontal area — silhouette share of frame (aero)", "share")
            fig.tight_layout()
            _save(fig, f"chart_aero_{key}.png")
    return made


def _fmt(v, unit=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}{unit}"
    return f"{v}{unit}"


def _dval(v):
    """Value of a summary metric that may be a stats dict."""
    if isinstance(v, dict):
        for k in ("mean", "value"):
            if v.get(k) is not None:
                return v[k]
        return None
    return v


def _badge(verdict: str) -> str:
    cls = {"Improved": "ok", "Regressed": "bad", "Neutral": "warn"}.get(verdict, "muted")
    return f'<span class="badge {cls}">{html.escape(str(verdict))}</span>'


def build(sid: int, store: Store) -> str:
    """Build report.html + summary.json for a finished session. Returns path."""
    sess = store.get_session(sid)
    if not sess:
        raise KeyError(f"session {sid} not found")
    out_dir = config.SESSIONS_DIR / str(sid)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = json.loads(sess.get("summary_json") or "{}")
    except Exception:
        summary = {}
    if summary and "form_score" not in summary:
        advisor.enrich(summary)
    cols = _load_cols(out_dir / "metrics.csv")
    charts = make_charts(sid, cols, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")

    coach = store.read_json(sid, "coach.json")
    plane = summary.get("view_plane") or sess.get("view_plane") or config.DEFAULT_VIEW_PLANE
    not_assessed = summary.get("not_assessed") or {}

    def esc(x):
        return html.escape(str(x))

    # ---- joint table
    joint_rows = ""
    for name in ("left_knee", "right_knee", "left_hip", "right_hip", "left_ankle",
                 "right_ankle", "left_elbow", "right_elbow", "left_shoulder", "right_shoulder"):
        j = summary.get("joints", {}).get(name)
        if not j:
            continue
        joint_rows += (
            f"<tr><td>{esc(name.replace('_', ' '))}</td><td>{_fmt(j['mean'], '°')}</td>"
            f"<td>{_fmt(j['min'], '°')}</td><td>{_fmt(j['max'], '°')}</td>"
            f"<td>{_fmt(j['rom'], '°')}</td><td>{_fmt(j['ang_vel_mean'], '°/s')}</td>"
            f"<td>{_fmt(j['coverage_pct'], '%')}</td></tr>")

    sym_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{_fmt(v, '%')}</td></tr>"
        for k, v in (summary.get("symmetry") or {}).items())

    asym = summary.get("asymmetry") or {}
    asym_rows = "".join(
        f"<tr><td>{esc(k.replace('_', ' '))}</td><td>{_fmt(v, '%') if not isinstance(v, str) else esc(v)}</td></tr>"
        for k, v in asym.items() if k != "formula")

    # ---- form metrics vs bands
    form_rows = ""
    for k, v in (summary.get("form") or {}).items():
        if v.get("in_band") is None:
            status = '<span class="muted">reported only</span>'
        else:
            status = "✅ in range" if v.get("in_band") else "⚠️ outside"
        band = v.get("band")
        band_txt = f"{band[0]}–{band[1]}" if band else "—"
        label = v.get("label") or k.replace("_", " ")
        form_rows += (f"<tr><td>{esc(label)}</td><td>{_fmt(v.get('value'))}</td>"
                      f"<td>{esc(band_txt)}</td><td>{status}</td></tr>")
    for k in sorted(not_assessed):
        reason = not_assessed[k]
        form_rows += (f"<tr class=\"na\"><td>{esc(k.replace('_', ' '))}</td>"
                      f"<td colspan=\"3\">{esc(reason)}</td></tr>")

    # ---- form score breakdown
    fs = summary.get("form_score_detail") or advisor.form_score(summary)
    score_rows = ""
    for c in fs.get("components", []):
        w = c.get("weight")
        wtxt = f"{w * 100:.0f}%" if w else "—"
        if c.get("score") is None:
            val = esc(c.get("status", "—"))
            sc = "—"
        else:
            val = _fmt(c.get("value"), "")
            sc = _fmt(c.get("score"), "")
        score_rows += (f"<tr><td>{esc(c.get('label'))}</td><td>{val}</td>"
                       f"<td>{esc(str(c.get('band')))}</td><td>{sc}</td>"
                       f"<td>{esc(wtxt)}</td></tr>")

    # ---- risk stratification
    risk = summary.get("risk") or advisor.risk_stratification(summary)
    risk_flags = "".join(f"<li>{esc(f.get('finding'))}</li>" for f in risk.get("flags", []))
    drills = "".join(f"<li>{esc(d.get('drill'))}</li>" for d in risk.get("drills", []))
    risk_cls = {"Low": "ok", "Moderate": "warn", "Elevated": "warn", "High": "bad"}.get(
        risk.get("level"), "muted")

    # ---- keyframes
    kfs = summary.get("keyframes") or []
    if not kfs:
        kdir = out_dir / config.KEYFRAME_DIR
        if kdir.exists():
            for p in sorted(kdir.glob("*.jpg")):
                parts = p.stem.split("_")
                kfs.append({"file": f"{config.KEYFRAME_DIR}/{p.name}",
                            "type": parts[1] if len(parts) > 1 else "?",
                            "label": GA.EVENT_LABEL.get(parts[1] if len(parts) > 1 else "", "Event"),
                            "side": parts[2] if len(parts) > 2 else "?",
                            "t": None})
    kf_html = ""
    for k in kfs:
        cap = [f"{esc(k.get('label') or k.get('type'))} · {esc(k.get('side'))}"]
        tval = k.get("t")
        if tval is not None:
            try:
                cap.append("t = %.2f s" % float(tval))
            except (TypeError, ValueError):
                pass
        bits = []
        if k.get("shin_angle_deg") is not None:
            bits.append("shin %.1f°" % abs(float(k["shin_angle_deg"])))
        if k.get("knee_flexion_deg") is not None:
            bits.append("knee flex %.1f°" % float(k["knee_flexion_deg"]))
        if k.get("pelvic_tilt_deg") is not None:
            bits.append("pelvic %+.1f°" % float(k["pelvic_tilt_deg"]))
        if k.get("hip_extension_deg") is not None:
            bits.append("hip ext %+.1f°" % float(k["hip_extension_deg"]))
        if k.get("refined"):
            bits.append("HD refined")
        zoom = k.get("zoom_file")
        zoom_html = ("<img src=\"%s\" alt=\"%s HD zoom\">" % (esc(zoom), esc(cap[0]))
                     if zoom else "")
        kf_html += ('<figure class="kf"><img src="%s" alt="%s">%s'
                    '<figcaption>%s<br><span class="muted">%s</span></figcaption></figure>'
                    % (esc(k.get("file")), esc(cap[0]), zoom_html,
                       esc(" · ".join(cap)),
                       esc(", ".join(bits))))

    kf_note = (not_assessed.get("gait_events")
               or ("No keyframes were captured for this session (no gait events were "
                   "latched, or the source produced no annotated frames)."))

    labels = "".join(f"<li>{esc(x)}</li>" for x in summary.get("labels", []))
    chart_imgs = "".join(
        f'<img class="chart" src="{esc(c)}" alt="{esc(c)}">' for c in charts)

    coach_text = ""
    coach_meta = ""
    if coach:
        coach_text = esc(coach.get("text", "")).replace(chr(10), "<br>")
        coach_meta = f' <span class="muted">({esc(coach.get("model"))} · {esc(coach.get("ts"))})</span>'
    coach_html = f"""<h2 class="noprint">AI form review — local model{coach_meta}</h2>
<div id="coachBox" class="coach noprint">{coach_text or 'No review yet. Press the button below (needs the RunTrack app running; the local model can take 1–3 minutes).'}</div>
<button class="noprint" onclick="runCoach()" style="margin-top:8px">Run local AI form review</button>
<script>
async function runCoach() {{
  const b = document.getElementById('coachBox');
  b.textContent = 'Running local model — this can take 1-3 minutes...';
  try {{
    const r = await fetch('/api/coach/{sid}', {{method: 'POST'}});
    const d = await r.json();
    b.innerHTML = d.ok ? d.text.replace(/\\n/g, '<br>') : ('Review failed: ' + d.error);
  }} catch (e) {{
    b.textContent = 'Review failed (open this report from the RunTrack app): ' + e;
  }}
}}
</script>"""

    # ---- discipline-aware sections ---------------------------------------
    discipline = str(summary.get("discipline") or "running").lower()
    bike_type = str(summary.get("bike_type") or config.DEFAULT_BIKE_TYPE).lower()
    bike = summary.get("bike") or {}
    rear = summary.get("rear") or {}
    aero = summary.get("aero") or {}
    if discipline == "cycling":
        disc_label = ("cycling position analysis — "
                      + config.BIKE_LABELS.get(bike_type, bike_type))
        coach_title, clinical_title = ("Coach review — cycling position & fit",
                                       "Clinical review — chiropractic & physical therapy")
    else:
        disc_label = "running form analysis"
        coach_title = "Coach review — running technique"
        clinical_title = "Clinical review — chiropractic & physical therapy"

    # cycling position table + suggestions
    bike_rows = ""
    for key, e in (bike.get("form") or {}).items():
        band = e.get("band")
        band_txt = f"{band[0]:g}–{band[1]:g}" if band else "—"
        if e.get("in_band") is None:
            status = '<span class="muted">reported only</span>'
        else:
            status = "✅ in range" if e.get("in_band") else "⚠️ outside"
        unit = e.get("unit", "")
        val = f"{e.get('value'):g} {unit}".strip() if e.get("value") is not None else "—"
        bike_rows += (f"<tr><td>{esc(e.get('label') or key)}</td><td>{esc(val)}</td>"
                      f"<td>{esc(band_txt)}</td><td>{status}</td></tr>")
    ps = bike.get("per_side") or {}
    for side, d in ps.items():
        bits = []
        if d.get("knee_bdc_mean_deg") is not None:
            bits.append(f"knee BDC {d['knee_bdc_mean_deg']:g}°")
        if d.get("knee_tdc_min_deg") is not None:
            bits.append(f"knee TDC {d['knee_tdc_min_deg']:g}°")
        if d.get("hip_tdc_min_deg") is not None:
            bits.append(f"hip TDC {d['hip_tdc_min_deg']:g}°")
        if d.get("knee_rom_deg") is not None:
            bits.append(f"ROM {d['knee_rom_deg']:g}°")
        if d.get("revolutions") is not None:
            bits.append(f"{d['revolutions']} revs")
        if bits:
            bike_rows += (f"<tr><td>{esc(side)} leg (per-revolution)</td>"
                          f"<td colspan=\"3\">{esc(', '.join(bits))}</td></tr>")
    bike_sug_html = "".join(
        f"<li><b>{esc(s.get('finding'))}</b><br>{esc(s.get('action'))}</li>"
        for s in (bike.get("suggestions") or []))
    bike_section = ""
    if discipline == "cycling":
        bike_section = f"""
<h2>{esc(coach_title)}</h2>
<table><tr><th>Metric</th><th>Value</th><th>Target ({esc(bike_type)})</th><th>Status</th></tr>
{bike_rows or '<tr><td colspan="4">No cycling metrics were tracked — film a steady '
              'side view with both legs visible.</td></tr>'}</table>
{"<div class='drill'><b>Fit adjustments to try (change 3-5 mm at a time and re-film)</b><ul>" + bike_sug_html + "</ul></div>" if bike_sug_html else ""}
<p class="muted" style="font-size:12px">{esc(CY.BIKE_NOTES)}</p>"""

    # rear-view control table
    rear_order = ["pelvic_obliquity_deg", "hip_vertical_travel", "hip_horizontal_travel",
                  "knee_lateral_travel_left", "knee_lateral_travel_right",
                  "knee_travel_angle_left", "knee_travel_angle_right",
                  "shoulder_lateral_travel_left", "shoulder_lateral_travel_right",
                  "ankle_swivel_left", "ankle_swivel_right"]
    rear_rows = ""
    for name in rear_order:
        e = rear.get(name)
        if not isinstance(e, dict):
            continue
        if name == "pelvic_obliquity_deg":
            val_txt = f"{e:.1f} °" if isinstance(e, (int, float)) else "—"
            rear_rows += (f"<tr><td>Pelvic obliquity (hip rocking)</td><td>{val_txt}</td>"
                          f"<td>≤ 5 °</td><td>—</td></tr>")
            continue
        if "value_mm" in e and e.get("value_mm") is not None:
            val_txt = f"{e['value_mm']:g} mm"
            tgt = (f"avg ≤ {e.get('target_mm_avg'):g} / good ≤ {e.get('target_mm_good'):g}"
                   if e.get("target_mm_avg") else "—")
            if e.get("within_good") is True:
                status = "✅ good"
            elif e.get("within_avg") is True:
                status = "◎ average"
            elif e.get("within_avg") is False:
                status = "⚠️ above guide"
            else:
                status = '<span class="muted">reported</span>'
        else:
            val_txt = f"{e.get('value_norm', 0):.3f} × shoulder width"
            tgt = ("share of shoulder width (give height or shoulder width for mm)"
                   if e.get("target_norm_avg") is None else
                   f"avg ≤ {e.get('target_norm_avg'):.3f} / good ≤ {e.get('target_norm_good'):.3f}")
            status = "—"
        if "value_deg" in e:
            val_txt = f"{e['value_deg']:g} °"
            tgt = f"avg ≤ {e.get('target_deg_avg', 8):g} / good ≤ {e.get('target_deg_good', 2):g}"
            status = ("✅ good" if e.get("within_good") is True else
                      "◎ average" if e.get("within_avg") is True else
                      "⚠️ above guide" if e.get("within_avg") is False else "—")
        rear_rows += (f"<tr><td>{esc(e.get('label') or name.replace('_', ' '))}</td>"
                      f"<td>{esc(val_txt)}</td><td>{esc(tgt)}</td><td>{status}</td></tr>")
    rear_section = ""
    if rear_rows:
        rear_section = f"""
<h2>Rear-view control — second camera ({esc(rear.get('prefix', ''))}stream)</h2>
<table><tr><th>Metric</th><th>Value (peak-to-peak)</th><th>Fit-tool guide</th><th>Status</th></tr>
{rear_rows}</table>
<p class="muted" style="font-size:12px">{esc(rear.get('scale_note') or '')}</p>"""

    # aero section (+ delta against the patient's previous cycling session)
    aero_section = ""
    if aero and aero.get("fa_frac_median") is not None:
        fa_txt = (f"{aero['fa_m2_median']:g} m² (median)" if aero.get("fa_m2_median")
                  else "scale unavailable — give height or shoulder width")
        prev_note = ""
        try:
            pid = sess.get("patient_id")
            if pid and aero.get("fa_m2_median"):
                for pss in store.patient_sessions(int(pid), limit=30):
                    if int(pss["id"]) >= int(sid):
                        continue
                    prev = store.get_session(int(pss["id"]))
                    if not prev:
                        continue
                    try:
                        psm = json.loads(prev.get("summary_json") or "{}")
                    except Exception:
                        psm = {}
                    pa = (psm.get("aero") or {})
                    if pa.get("fa_m2_median"):
                        dd = CY.aero_delta(pa["fa_m2_median"], aero["fa_m2_median"],
                                           speed_kmh=(aero.get("speed_kmh") or 40.0),
                                           cd=aero.get("cd_assumed"))
                        if dd:
                            prev_note = (
                                f"<p><b>Change vs session {pss['id']} "
                                f"({esc(prev.get('started_ts'))}):</b> "
                                f"{pa['fa_m2_median']:g} → {aero['fa_m2_median']:g} m² "
                                f"({dd['delta_area_pct']:+.1f} %)".replace("{", "").replace("}", "")
                                + (f", ≈ {dd['watts_delta_est']:+.1f} W aero drag at "
                                   f"{dd.get('speed_kmh', 40):g} km/h (Cd "
                                   f"{dd.get('cd_assumed', config.AERO_CD_DEFAULT):g}, estimate)"
                                   if dd.get("watts_delta_est") is not None else "")
                                + ")</p>")
                        break
        except Exception:
            prev_note = ""
        aero_section = f"""
<h2>Aero analysis — projected frontal area</h2>
<div class="drill">
 <div><b>Frontal area (median over the session):</b> {esc(fa_txt)}</div>
 <div class="muted" style="margin-top:4px">Silhouette share of the frame: 
 {aero['fa_frac_median']:.4f} (mean {aero.get('fa_frac_mean', 0):.4f}, n = {aero.get('n', '—')})
 — measured from the pose model's segmentation mask on the aero camera.</div>
 {prev_note}
 <div class="muted" style="margin-top:6px">{esc(aero.get('assumptions') or '')}</div>
</div>"""

    # ---- gait events / phases (running only — cycling uses BDC/TDC keyframes)
    ev = summary.get("gait_events") or {}
    ev_rows = ""
    for typ in ("IC", "MS", "TO"):
        by = (ev.get("by_side") or {}).get(typ) or {}
        ev_rows += (f"<tr><td>{esc(GA.EVENT_LABEL.get(typ, typ))}</td>"
                    f"<td>{_fmt(ev.get(typ))}</td><td>{_fmt(by.get('left'))}</td>"
                    f"<td>{_fmt(by.get('right'))}</td></tr>")
    ev_note = ""
    if ev.get("note"):
        ev_note = f'<tr class="na"><td colspan="4">{esc(ev.get("note"))}</td></tr>'
    if discipline == "cycling":
        phases_section = ""
    else:
        phases_section = ("<h2>Gait phases latched</h2>\n"
                          "<table><tr><th>Event</th><th>Total</th><th>Left</th><th>Right</th></tr>\n"
                          f"{ev_rows}{ev_note}</table>")
    cycling_notes = CY.BIKE_NOTES

    cards = [
        ("Patient", f'{esc(sess.get("patient_name"))}<br><span class="muted">{esc(sess.get("patient_phone"))}</span>'),
        ("Date", esc(sess.get("started_ts"))),
        ("Camera view", esc(plane)),
        ("Duration", _fmt(round(summary.get("duration_s") or sess.get("duration_s") or 0, 1), " s")),
        ("Frames", _fmt(sess.get("frames"))),
        ("Cadence", _fmt(summary.get("cadence_spm"), " spm")),
        ("Steps (est.)", _fmt(summary.get("strike_count"))),
        ("Vertical osc.", _fmt(summary.get("vertical_osc_cm"), " cm")),
        ("Power est.", _fmt(summary.get("power_est_watts"), " W")),
        ("Power / kg", _fmt(summary.get("power_per_kg_watts"), " W/kg")),
        ("Form index", _fmt(summary.get("form_index"), " / 100")),
        ("Form score", _fmt(summary.get("form_score"), " / 100")),
        ("Risk level", f'<span class="{risk_cls}">{esc(risk.get("level"))}</span>'),
        ("Source", esc(sess.get("source_label"))),
    ]
    if summary.get("step_length_m"):
        cards.append(("Step length", _fmt(summary.get("step_length_m"), " m")))
    if summary.get("stride_length_m"):
        cards.append(("Stride length", _fmt(summary.get("stride_length_m"), " m")))
    if summary.get("touchdown_shin_deg"):
        cards.append(("Touchdown shin", _fmt(_dval(summary.get("touchdown_shin_deg")), " °")))
    if summary.get("overstride_leg_frac"):
        cards.append(("Overstride", _fmt(_dval(summary.get("overstride_leg_frac")), " leg len")))
    if summary.get("knee_flexion_ms_deg"):
        cards.append(("Mid-stance knee flex.", _fmt(_dval(summary.get("knee_flexion_ms_deg")), " °")))
    if summary.get("pelvic_drop_deg"):
        cards.append(("Pelvic drop", _fmt(_dval(summary.get("pelvic_drop_deg")), " °")))
    if summary.get("knee_valgus_deg"):
        cards.append(("Knee valgus", _fmt(_dval(summary.get("knee_valgus_deg")), " °")))
    if summary.get("hip_extension_toeoff_deg"):
        cards.append(("Hip ext. @ toe-off", _fmt(_dval(summary.get("hip_extension_toeoff_deg")), " °")))
    if summary.get("stance_time_s"):
        cards.append(("Stance time", _fmt(_dval(summary.get("stance_time_s")), " s")))
    if asym.get("stance_time_pct") is not None:
        cards.append(("Stance asymmetry", _fmt(asym.get("stance_time_pct"), " %")))
    if discipline == "cycling":
        bcad = bike.get("cadence_rpm")
        if bcad is not None:
            cards.append(("Cadence", _fmt(bcad, " rpm")))
        kb = bike.get("knee_bdc_deg")
        if kb is not None:
            cards.append(("Knee @ BDC", _fmt(kb, " °")))
        tp = bike.get("torso_deg")
        if tp is not None:
            cards.append(("Torso angle", _fmt(tp, " °")))
        if aero.get("fa_m2_median"):
            cards.append(("Frontal area", _fmt(aero.get("fa_m2_median"), " m²")))
    card_html = "".join(
        f'<div class="card"><div class="k">{esc(k)}</div><div class="v">{v}</div></div>'
        for k, v in cards)

    dq = summary.get("data_quality") or {}
    disc_meta = (" · discipline: <b>cycling</b> · bike: <b>"
                 + esc(config.BIKE_LABELS.get(bike_type, bike_type)) + "</b>"
                 if discipline == "cycling" else "")
    cam2_meta = (" · second camera: <b>signal present</b> (rear/front view metrics)"
                 if rear else "")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>RunTrack report — session {sid}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body {{ font-family: system-ui, 'Segoe UI', sans-serif; background:#101418; color:#e8edf2;
        margin:0; padding:24px; }}
 h1 {{ font-size:20px; margin:0 0 4px; }}
 h2 {{ font-size:15px; margin:26px 0 8px; color:#9fd3ff; }}
 h2.group {{ font-size:17px; margin:30px 0 10px; color:#7dd3fc;
             border-bottom:1px solid #2a3441; padding-bottom:5px; letter-spacing:.02em; }}
 .muted {{ color:#8a97a5; font-weight:400; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
 .card {{ background:#1a2129; border:1px solid #2a3441; border-radius:8px; padding:8px 12px; min-width:110px; }}
 .card .k {{ font-size:10px; color:#8a97a5; text-transform:uppercase; letter-spacing:.06em; }}
 .card .v {{ font-size:16px; margin-top:3px; }}
 table {{ border-collapse:collapse; width:100%; font-size:13px; margin-top:6px; }}
 th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid #232c36; }}
 th {{ color:#9fd3ff; font-weight:600; font-size:12px; }}
 tr.na td {{ color:#c9a227; }}
 .chart {{ width:100%; max-width:880px; border-radius:8px; border:1px solid #2a3441; margin:6px 0; }}
 .coach {{ background:#1a2129; border:1px solid #2a3441; border-radius:8px; padding:12px; line-height:1.5; max-width:880px; }}
 ul.labels {{ color:#c9a227; font-size:12.5px; }}
 .disc {{ margin-top:26px; color:#8a97a5; font-size:12px; max-width:880px; line-height:1.5; }}
 button {{ background:#2563eb; border:0; color:#fff; border-radius:6px; padding:8px 14px; cursor:pointer; font-size:13px; }}
 .ok {{ color:#4ade80; }} .bad {{ color:#f87171; }} .warn {{ color:#fbbf24; }}
 .scorebox {{ display:flex; gap:18px; align-items:center; background:#1a2129; border:1px solid #2a3441;
              border-radius:10px; padding:12px 16px; max-width:520px; }}
 .scoreval {{ font-size:40px; font-weight:700; line-height:1; }}
 .badge {{ padding:2px 8px; border-radius:12px; font-size:12px; border:1px solid currentColor; }}
 .kfgrid {{ display:flex; flex-wrap:wrap; gap:10px; }}
 .kf {{ margin:0; width:300px; }}
 .kf img {{ width:100%; border-radius:8px; border:1px solid #2a3441; }}
 .kf figcaption {{ font-size:11.5px; color:#cbd5e1; padding-top:4px; line-height:1.4; }}
 .drill {{ background:#132018; border:1px solid #24512f; border-radius:8px; padding:10px 12px; margin-top:8px; }}
 .drill ul {{ margin:6px 0 0 18px; }}
 .caveat {{ color:#fbbf24; font-size:12px; margin-top:8px; max-width:880px; }}
 @page {{ size: A4 portrait; margin: 12mm; }}
 @media print {{
   body {{ background:#fff; color:#111; padding:0; font-size:11.5px; }}
   h1 {{ font-size:17px; }} h2 {{ font-size:13px; color:#0b4a75; page-break-after:avoid; }}
   .muted, .disc, .caveat {{ color:#555; }}
   .noprint {{ display:none !important; }}
   .card, table, .scorebox, .drill, .kf, .coach {{ page-break-inside:avoid; }}
   .card {{ background:#f6f7f9; border:1px solid #ccc; color:#111; }}
   .card .k {{ color:#555; }}
   table th, table td {{ border-bottom:1px solid #ddd; color:#111; }}
   tr.na td {{ color:#7a5c00; }}
   .chart, .kf img {{ border:1px solid #ccc; }}
   .ok {{ color:#0a7d33; }} .bad {{ color:#b3261e; }} .warn {{ color:#8a5b00; }}
   .scorebox, .drill, .coach {{ background:#f6f7f9; border:1px solid #ccc; }}
   a {{ color:#111; text-decoration:none; }}
 }}
</style></head><body>
<h1>RunTrack — {esc(disc_label)}</h1>
<div class="muted">Session {sid} · {esc(sess.get('status'))} · source: {esc(sess.get('source_label'))}
 · camera view: <b>{esc(plane)}</b>{disc_meta}{cam2_meta}
{('· lower-body landmark coverage: <b>' + esc(dq.get('lower_body_coverage_pct')) + '%</b> (' + esc('ok' if dq.get('lower_body_ok') else 'too low for gait events') + ')') if dq.get('lower_body_coverage_pct') is not None else ''}</div>
<button class="noprint" style="margin-top:10px" onclick="window.print()">🖨 Print / Save as PDF</button>
<div class="cards">{card_html}</div>

<h2 class="group">{esc(clinical_title)}</h2>
<h2>Injury risk stratification — {esc(risk.get("level"))} ({esc(risk.get("points"))} pts)</h2>
{"<ul>" + risk_flags + "</ul>" if risk_flags else '<p class="muted">No risk flags were triggered by the measured metrics.</p>'}
<div class="drill"><b>Prescribed PT drills for the flagged findings</b>
{"<ul>" + drills + "</ul>" if drills else '<div class="muted">No drills — nothing flagged.</div>'}
</div>
<p class="caveat">{esc(risk.get("caveat") or advisor.EVIDENCE_CAVEAT)}</p>

<h2>Gait event keyframes ({len(kfs)})</h2>
{('<div class="kfgrid">' + kf_html + '</div>') if kf_html else '<p class="muted">' + esc(kf_note) + '</p>'}

{phases_section}

<h2>Joint angles</h2>
<table><tr><th>Joint</th><th>Mean</th><th>Min</th><th>Max</th><th>ROM</th><th>Angular vel.</th><th>Tracked</th></tr>
{joint_rows or '<tr><td colspan="7">No joint data was tracked.</td></tr>'}</table>

<h2>Left / right symmetry (mean angle agreement) and asymmetry index</h2>
<table><tr><th>Joint</th><th>Symmetry</th></tr>{sym_rows or '<tr><td colspan="2">—</td></tr>'}</table>
<table><tr><th>Asymmetry (spec formula)</th><th>Value</th></tr>{asym_rows or '<tr><td colspan="2">—</td></tr>'}</table>

<h2>Form metrics vs reference bands ({esc(plane)} view)</h2>
<table><tr><th>Metric</th><th>Value</th><th>Reference band</th><th>Status</th></tr>
{form_rows or '<tr><td colspan="4">—</td></tr>'}</table>

{bike_section}
{rear_section}
{aero_section}

<h2 class="group">{esc(coach_title)}</h2>
<h2>Overall form score (0-100)</h2>
<div class="scorebox">
  <div class="scoreval">{_fmt(summary.get("form_score"), "")}</div>
  <div><div><b>{"Cycling position score" if discipline == "cycling" else "Composite form score"}</b> — weighted across symmetry, cadence, overstride,
   joint angles and oscillation for the <b>{esc(plane)}</b> view plane.</div>
  <div class="muted">Method: {esc(fs.get("method") or "weighted composite")} · coverage
   {esc(fs.get("coverage"))}.</div></div>
</div>
<table><tr><th>Component</th><th>Value</th><th>Reference band</th><th>Score</th><th>Weight</th></tr>
{score_rows or '<tr><td colspan="5">—</td></tr>'}</table>

<h2>Charts</h2>
{chart_imgs or '<p class="muted">No charts (no data).</p>'}

{coach_html}

<h2>Notes &amp; limits</h2>
<ul class="labels">
 <li>Reference bands are heuristics from common running-gait literature for recreational runners — not clinical norms.</li>
 {("<li>Cycling: fit bands follow accepted cycling-fit practice/industry guides (Velogic Studio triathlon metrics, Holmes method knee angles, tri/TT torso conventions) and the front camera measures the projected frontal area from the pose-model silhouette — an index, not a drag-coefficient measurement. Bike-fit changes are 3-5 mm steps at most, re-film to compare.</li>" if discipline == "cycling" else "")}
 {("<li>Second camera: rear-view numbers are peak-to-peak excursions of landmarks relative to a rolling mean (the rider/athlete shifts on the machine over a set). Treat them as relative control indicators, not laboratory kinematics.</li>" if rear else "")}
 {("<li>Aero: frontal area in m² needs height or shoulder width; the watts estimate assumes the drag coefficient is unchanged between sessions.</li>" if aero else "")}
 <li>Angles are 2D, from a single camera view. Sagittal metrics need a side view, frontal metrics need a front/back view; the other plane is reported as “{esc(config.NOT_ASSESSED_TEXT)}” instead of a number.</li>
 <li>Overstride in centimetres is a height-based estimate (leg length ≈ {config.LEG_LENGTH_HEIGHT_FRACTION} × height), not a measured length.</li>
 <li>Left/right labels follow MediaPipe's convention; verify orientation against the camera view.</li>
 <li>Form score weights (plane aware): {esc(json.dumps(fs.get("weights") or dict()))}.</li>
 <li>{esc(advisor.EVIDENCE_CAVEAT)}</li>
 {labels}
</ul>
<p class="muted">Files: <a href="metrics.csv">metrics.csv</a> · <a href="summary.json">summary.json</a>
{(' · <a href="' + config.KEYFRAME_DIR + '/">keyframes/</a>') if kfs else ''}</p>
<div class="disc">RunTrack is a local research/coaching aid. It is not a medical device and does not
provide diagnosis. Keep this data on this PC. Files: metrics.csv, summary.json, charts, keyframes
in the session folder.</div>
</body></html>"""

    p = out_dir / "report.html"
    p.write_text(doc, encoding="utf-8")
    return str(p)
