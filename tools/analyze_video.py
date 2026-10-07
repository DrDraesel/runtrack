"""Offline video analysis CLI for RunTrack.

Analyze a recorded video (e.g. filmed on your phone at the treadmill / on the
turbo trainer) with the same pipeline the live app uses, and produce the full
report — gait events (running) or pedal-stroke keyframes (cycling), annotated
keyframes, the form score, per-professional sections and the risk stratification.

A second video can be analysed at the same time (rear/frontal view) — its
metrics are logged with an "r_" prefix and feed the rear-view control block.

Usage:
  python tools/analyze_video.py VIDEO [--name N --phone P --weight 70
        --height 175 --speed 10 --start 0 --duration 60 --proc-width 960
        --plane sagittal|frontal --discipline running|cycling
        --bike-type road|tt|tri --rear REAR_VIDEO --seg --seg2
        --cd 0.80 --saddle 73]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

import advisor  # noqa: E402
import config  # noqa: E402
import cycling as CY  # noqa: E402
import gait_events as GA  # noqa: E402
from filters import LandmarkFilter  # noqa: E402
from pose_engine import (PoseEngine, draw_skeleton, make_crop_refiner,  # noqa: E402
                         overlay_angles, person_crop_box)
from side_identity import SideIdentity  # noqa: E402
import metrics as M  # noqa: E402
from session_store import Store  # noqa: E402
import report as R  # noqa: E402


def _attach_hi(ev, ring):
    """Attach a native-resolution crop of the event frame (HD pass input)."""
    for j, fr in ring:
        if j == ev.get("i"):
            box = person_crop_box(ev.get("lms"), pad=config.KEYFRAME_CROP_PAD)
            if box is None:
                return
            h, w = fr.shape[:2]
            x0, y0 = int(max(0, round(box[0] * w))), int(max(0, round(box[1] * h)))
            x1, y1 = int(min(w, round(box[2] * w))), int(min(h, round(box[3] * h)))
            if x1 - x0 < 16 or y1 - y0 < 16:
                return
            ok, buf = cv2.imencode(".jpg", fr[y0:y1, x0:x1],
                                   [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                ev["hi"] = buf.tobytes()
                ev["hi_box"] = [float(b) for b in box]
            return


def _read_second(cap2, args2, t_ms):
    """Read the matching frame from the second video/camera (lockstep)."""
    if cap2 is None:
        return None
    ok2, frame2 = cap2.read()
    if not ok2:
        return None
    return frame2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--name", default="Video Test")
    ap.add_argument("--phone", default="0000000000")
    ap.add_argument("--weight", type=float, default=None)
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--proc-width", type=int, default=config.PROC_WIDTH)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--plane", default=config.DEFAULT_VIEW_PLANE,
                    choices=list(config.VIEW_PLANES),
                    help="camera view the clip was filmed in (default: sagittal)")
    ap.add_argument("--discipline", default=config.DEFAULT_DISCIPLINE,
                    choices=list(config.DISCIPLINES))
    ap.add_argument("--bike-type", default=config.DEFAULT_BIKE_TYPE,
                    choices=list(config.BIKE_TYPES))
    ap.add_argument("--rear", default=None,
                    help="second video, rear/frontal view — analysed in lockstep")
    ap.add_argument("--seg", action="store_true",
                    help="estimate frontal area (aero) from the primary silhouette")
    ap.add_argument("--seg2", action="store_true",
                    help="estimate frontal area from the second camera silhouette")
    ap.add_argument("--cd", type=float, default=None,
                    help="assumed drag coefficient for the aero watts estimate")
    ap.add_argument("--saddle", type=float, default=None,
                    help="saddle height in cm (optional, recorded)")
    args = ap.parse_args()

    discipline = str(args.discipline).lower()
    bike_type = str(args.bike_type).lower()
    if discipline not in config.DISCIPLINES:
        discipline = config.DEFAULT_DISCIPLINE
    if bike_type not in config.BIKE_TYPES:
        bike_type = config.DEFAULT_BIKE_TYPE

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        sys.exit(1)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    cap2 = None
    if args.rear:
        cap2 = cv2.VideoCapture(args.rear)
        if not cap2.isOpened():
            print(f"WARNING: cannot open second video {args.rear} — continuing "
                  "without the rear view")
            cap2 = None
        elif args.start > 0:
            cap2.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    engine = PoseEngine(config.MODEL_PATH, seg=bool(args.seg))
    engine2 = (PoseEngine(config.MODEL_PATH, seg=bool(args.seg2))
               if cap2 is not None else None)
    store = Store(config.DB_PATH)
    pid = store.upsert_patient(args.name, args.phone)
    src_label = f"File {Path(args.video).name}"
    if args.rear:
        src_label += f" + rear {Path(args.rear).name}"
    params = {"weight_kg": args.weight, "height_cm": args.height,
              "speed_kmh": args.speed, "cd_est": args.cd,
              "saddle_cm": args.saddle}
    sid = store.create_session(
        pid, params, src_label, view_plane=args.plane,
        discipline=discipline,
        bike_type=(bike_type if discipline == "cycling" else None))
    sess_dir = config.SESSIONS_DIR / str(sid)

    rows, n, det, det2, snaps = [], 0, 0, 0, 0
    last_snap = -1e9
    lf = LandmarkFilter()
    lf2 = LandmarkFilter()
    si = SideIdentity() if config.SIDE_IDENTITY_ENABLED else None
    si2 = SideIdentity() if config.SIDE_IDENTITY_ENABLED else None
    latch = CY.CycleLatch() if discipline == "cycling" else GA.GaitLatch()
    ring = collections.deque(maxlen=config.FRAME_RING)
    t_wall0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vt = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if vt > args.start + args.duration:
                break
            if args.max_frames and n >= args.max_frames:
                break
            n += 1
            h, w = frame.shape[:2]
            scale = args.proc_width / w if w > args.proc_width else 1.0
            proc = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
            lms_raw = engine.detect(proc, int(vt * 1000))
            if si is not None:
                lms_raw, _ = si.update(lms_raw, vt)
            lms = lf(lms_raw, vt) if lms_raw is not None else None
            fm = M.frame_metrics(lms)
            if lms is not None:
                det += 1
            if args.seg and engine.seg_frac is not None:
                fm["fa_frac"] = engine.seg_frac

            # ---- second camera (rear / frontal view), lockstep -------------
            fm2 = {}
            frame2 = _read_second(cap2, None, vt) if cap2 is not None else None
            if frame2 is not None and engine2 is not None:
                h2, w2 = frame2.shape[:2]
                scale2 = args.proc_width / w2 if w2 > args.proc_width else 1.0
                proc2 = (cv2.resize(frame2, None, fx=scale2, fy=scale2)
                         if scale2 != 1.0 else frame2)
                lms_raw2 = engine2.detect(proc2, int(vt * 1000))
                if si2 is not None:
                    lms_raw2, _ = si2.update(lms_raw2, vt)
                lms2 = lf2(lms_raw2, vt) if lms_raw2 is not None else None
                fm2 = M.frame_metrics(lms2)
                if lms2 is not None:
                    det2 += 1
                if args.seg2 and engine2.seg_frac is not None:
                    fm2["fa_frac"] = engine2.seg_frac

            rel = vt - args.start
            fm["t_rel"] = rel
            row = dict(fm)
            for k2, v2 in fm2.items():
                row["r_" + k2] = v2
            rows.append(row)

            annotated = draw_skeleton(proc.copy(), lms)
            overlay_angles(annotated, lms, fm)
            # clean frame for keyframes (skeleton is drawn at save time)
            ok_c, cbuf = cv2.imencode(".jpg", proc,
                                      [cv2.IMWRITE_JPEG_QUALITY,
                                       config.KEYFRAME_JPEG_QUALITY])
            idx = getattr(latch, "_i", None)
            if idx is not None:
                ring.append((idx, frame.copy()))
            new_evs = []
            if ok_c:
                new_evs = latch.push(rel, fm, cbuf.tobytes(), lms)
            for ev in new_evs:
                try:
                    if float(ev.get("quality") or 0.0) >= 1.0:
                        _attach_hi(ev, ring)
                except Exception:
                    pass
            if vt - last_snap >= max(1.0, args.duration / 3.0) and snaps < 4:
                cv2.imwrite(str(sess_dir / f"frame_{snaps}.jpg"), annotated)
                snaps += 1
                last_snap = vt
            if n % 200 == 0:
                print(f"  … {n} frames processed ({det} with pose)")
    finally:
        engine.close()
        if engine2 is not None:
            engine2.close()
        cap.release()
        if cap2 is not None:
            cap2.release()

    events = latch.events() if discipline == "running" else []
    summary = M.analyze_session(rows, weight_kg=args.weight,
                                height_cm=args.height, speed_kmh=args.speed,
                                view_plane=args.plane, events=events,
                                discipline=discipline)
    CY.enrich(summary, rows, params, bike_type=bike_type, discipline=discipline)
    if si is not None:
        summary["side_identity"] = si.info()
    try:
        ref = make_crop_refiner()
        summary["keyframes"] = latch.save_snapshots(
            sess_dir, refine=(ref[1] if ref else None))
        if ref is not None:
            ref[0].close()
    except Exception as e:
        summary.setdefault("labels", []).append(f"keyframe capture failed: {e}")
    advisor.enrich(summary)
    store.save_csv(sid, rows)
    store.finish_session(sid, summary.get("duration_s"), n, summary, status="complete")
    report_path = R.build(sid, store)

    bike = summary.get("bike") or {}
    rear = summary.get("rear") or {}
    aero = summary.get("aero") or {}
    print(json.dumps({
        "session_id": sid,
        "discipline": discipline,
        "bike_type": bike_type if discipline == "cycling" else None,
        "view_plane": summary.get("view_plane"),
        "frames_processed": n,
        "frames_with_pose": det,
        "frames_with_pose_rear": det2,
        "pipeline_seconds": round(time.time() - t_wall0, 1),
        "cadence_spm": summary.get("cadence_spm"),
        "cadence_rpm": bike.get("cadence_rpm"),
        "knee_bdc_deg": bike.get("knee_bdc_deg"),
        "knee_tdc_min_deg": bike.get("knee_tdc_min_deg"),
        "hip_tdc_min_deg": bike.get("hip_tdc_min_deg"),
        "torso_deg": bike.get("torso_deg"),
        "bike_suggestions": [s.get("finding") for s in (bike.get("suggestions") or [])],
        "rear_metrics": {k: v for k, v in rear.items()
                         if k not in ("labels", "scale_note", "prefix")},
        "aero_frontal_area_m2": aero.get("fa_m2_median"),
        "vertical_osc_cm": summary.get("vertical_osc_cm"),
        "power_est_watts": summary.get("power_est_watts"),
        "power_per_kg_watts": summary.get("power_per_kg_watts"),
        "form_score": summary.get("form_score"),
        "bike_form_score": (bike.get("form_score")
                            if discipline == "cycling" else None),
        "risk_level": (summary.get("risk") or {}).get("level"),
        "gait_events": summary.get("gait_events") if discipline == "running" else None,
        "bike_revolutions": (latch.revolutions()
                             if discipline == "cycling" else None),
        "keyframes": len(summary.get("keyframes") or []),
        "asymmetry": summary.get("asymmetry"),
        "report": report_path,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
