"""SQLite storage for RunTrack patients and sessions."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import sqlite3
import time
from pathlib import Path

import config


def _summary_brief(sm: dict) -> dict:
    """Flat key metrics pulled out of a stored summary (for list/history views)."""
    def s(v):
        return v.get("mean") if isinstance(v, dict) else v

    risk = sm.get("risk") or {}
    fs = sm.get("form_score_detail") or {}
    bike = sm.get("bike") or {}
    aero = sm.get("aero") or {}
    return {
        "cadence_spm": s(sm.get("cadence_spm")),
        "form_index": s(sm.get("form_index")),
        "form_score": s(sm.get("form_score")),
        "symmetry_mean_pct": s(sm.get("symmetry_mean_pct")),
        "vertical_osc_cm": s(sm.get("vertical_osc_cm")),
        "power_est_watts": s(sm.get("power_est_watts")),
        "power_per_kg_watts": s(sm.get("power_per_kg_watts")),
        "touchdown_shin_deg": s(sm.get("touchdown_shin_deg")),
        "knee_flexion_ms_deg": s(sm.get("knee_flexion_ms_deg")),
        "pelvic_drop_deg": s(sm.get("pelvic_drop_deg")),
        "knee_valgus_deg": s(sm.get("knee_valgus_deg")),
        "overstride_leg_frac": s(sm.get("overstride_leg_frac")),
        "discipline": sm.get("discipline"),
        "bike_type": sm.get("bike_type"),
        "cadence_rpm": s(bike.get("cadence_rpm")),
        "knee_bdc_deg": s(bike.get("knee_bdc_deg")),
        "torso_deg": s(bike.get("torso_deg")),
        "fa_m2": s(aero.get("fa_m2_median")),
        "view_plane": sm.get("view_plane"),
        "risk_level": risk.get("level"),
        "risk_points": risk.get("points"),
        "form_score_coverage": fs.get("coverage"),
    }


class Store:
    def __init__(self, db_path=None, sessions_dir=None):
        self.db_path = str(db_path or config.DB_PATH)
        self.sessions_dir = Path(sessions_dir) if sessions_dir else config.SESSIONS_DIR
        self._init()

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=15)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def _init(self):
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  phone TEXT NOT NULL UNIQUE,
                  created_ts TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS sessions(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id INTEGER NOT NULL REFERENCES patients(id),
                  started_ts TEXT NOT NULL,
                  ended_ts TEXT,
                  duration_s REAL,
                  source_label TEXT,
                  weight_kg REAL,
                  height_cm REAL,
                  speed_kmh REAL,
                  frames INTEGER,
                  status TEXT DEFAULT 'running',
                  summary_json TEXT,
                  dir TEXT,
                  view_plane TEXT DEFAULT 'sagittal'
                );
                """
            )
            # migration for databases created before the view-plane feature
            cols = {r["name"] for r in c.execute("PRAGMA table_info(sessions)").fetchall()}
            if "view_plane" not in cols:
                c.execute("ALTER TABLE sessions ADD COLUMN view_plane TEXT DEFAULT 'sagittal'")
            if "discipline" not in cols:
                c.execute("ALTER TABLE sessions ADD COLUMN discipline TEXT DEFAULT 'running'")
            if "bike_type" not in cols:
                c.execute("ALTER TABLE sessions ADD COLUMN bike_type TEXT")

    # ------------------------------------------------------------ patients --
    def upsert_patient(self, name: str, phone: str) -> int:
        name = name.strip()
        phone = phone.strip()
        with self._conn() as c:
            row = c.execute("SELECT id, name FROM patients WHERE phone=?", (phone,)).fetchone()
            if row:
                if row["name"] != name:
                    c.execute("UPDATE patients SET name=? WHERE id=?", (name, row["id"]))
                return int(row["id"])
            cur = c.execute("INSERT INTO patients(name, phone) VALUES(?,?)", (name, phone))
            return int(cur.lastrowid)

    def get_patient(self, pid: int):
        with self._conn() as c:
            row = c.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
        return dict(row) if row else None

    def list_patients(self):
        with self._conn() as c:
            rows = c.execute(
                "SELECT p.*, count(s.id) AS n_sessions FROM patients p "
                "LEFT JOIN sessions s ON s.patient_id=p.id GROUP BY p.id ORDER BY p.name"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_patient(self, pid, name=None, phone=None):
        """Edit a saved patient's name/phone. Returns the patient or None."""
        with self._conn() as c:
            row = c.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
            if not row:
                return None
            new_name = (name or row["name"]).strip()
            new_phone = (phone or row["phone"]).strip()
            if not new_name or not new_phone:
                raise ValueError("name and phone must not be empty")
            try:
                c.execute("UPDATE patients SET name=?, phone=? WHERE id=?",
                          (new_name, new_phone, pid))
            except sqlite3.IntegrityError:
                raise ValueError("another patient already uses this phone number")
        return self.get_patient(pid)

    # ------------------------------------------------------------ sessions --
    def create_session(self, patient_id: int, params: dict, source_label: str,
                       view_plane: str | None = None, discipline: str | None = None,
                       bike_type: str | None = None) -> int:
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        plane = str(view_plane or config.DEFAULT_VIEW_PLANE).lower()
        if plane not in config.VIEW_PLANES:
            plane = config.DEFAULT_VIEW_PLANE
        disc = str(discipline or config.DEFAULT_DISCIPLINE).lower()
        if disc not in config.DISCIPLINES:
            disc = config.DEFAULT_DISCIPLINE
        bike = None
        if disc == "cycling":
            bike = str(bike_type or config.DEFAULT_BIKE_TYPE).lower()
            if bike not in config.BIKE_TYPES:
                bike = config.DEFAULT_BIKE_TYPE
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO sessions(patient_id, started_ts, source_label, weight_kg, height_cm, speed_kmh, status, view_plane, discipline, bike_type) "
                "VALUES(?,?,?,?,?,?,'running',?,?,?)",
                (patient_id, started, source_label,
                 params.get("weight_kg"), params.get("height_cm"), params.get("speed_kmh"),
                 plane, disc, bike))
            sid = int(cur.lastrowid)
        (self.sessions_dir / str(sid)).mkdir(parents=True, exist_ok=True)
        return sid

    def update_session(self, sid: int, weight_kg=None, height_cm=None,
                       speed_kmh=None):
        """Edit a saved session's body / treadmill parameters. None = keep."""
        sets, args = [], []
        for col, val in (("weight_kg", weight_kg), ("height_cm", height_cm),
                         ("speed_kmh", speed_kmh)):
            if val is not None:
                sets.append(f"{col}=?")
                args.append(float(val))
        with self._conn() as c:
            row = c.execute("SELECT id FROM sessions WHERE id=?", (sid,)).fetchone()
            if not row:
                return None
            if sets:
                c.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id=?",
                          (*args, sid))
        return self.get_session(sid)

    def finish_session(self, sid: int, duration_s, frames: int, summary: dict,
                       status: str = "complete"):
        with self._conn() as c:
            c.execute(
                "UPDATE sessions SET ended_ts=datetime('now'), duration_s=?, frames=?, "
                "status=?, summary_json=?, dir=? WHERE id=?",
                (duration_s, frames, status, json.dumps(summary, default=str),
                 str(self.sessions_dir / str(sid)), sid))

    def save_csv(self, sid: int, rows: list) -> Path:
        p = self.sessions_dir / str(sid) / "metrics.csv"
        keys: list = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
        return p

    def save_json(self, sid: int, name: str, data: dict) -> Path:
        p = self.sessions_dir / str(sid) / name
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return p

    def read_json(self, sid: int, name: str):
        p = self.sessions_dir / str(sid) / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def get_session(self, sid: int):
        with self._conn() as c:
            row = c.execute(
                "SELECT s.*, p.name AS patient_name, p.phone AS patient_phone "
                "FROM sessions s LEFT JOIN patients p ON p.id=s.patient_id WHERE s.id=?",
                (sid,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 200, patient_id: int | None = None):
        q = ("SELECT s.id, s.started_ts, s.ended_ts, s.duration_s, s.status, "
             "s.view_plane, s.discipline, s.bike_type, s.source_label, s.summary_json, "
             "s.weight_kg, s.height_cm, s.speed_kmh, "
             "p.name AS patient_name, p.phone AS patient_phone "
             "FROM sessions s LEFT JOIN patients p ON p.id=s.patient_id ")
        args: tuple = ()
        if patient_id:
            q += "WHERE s.patient_id=? "
            args = (patient_id,)
        q += "ORDER BY s.id DESC LIMIT ?"
        args = args + (limit,)
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            s = d.pop("summary_json", None)
            try:
                sm = json.loads(s) if s else {}
            except Exception:
                sm = {}
            d.update(_summary_brief(sm))
            if not d.get("view_plane"):
                d["view_plane"] = sm.get("view_plane") or config.DEFAULT_VIEW_PLANE
            out.append(d)
        return out

    def patient_sessions(self, patient_id: int, limit: int = 30):
        """Recent sessions of one patient, newest first (default: last 30)."""
        return self.list_sessions(limit=limit, patient_id=patient_id)

    def history(self, per_patient_limit: int = 30):
        """Authoritative server-side patient history: patients + recent sessions."""
        out = []
        for p in self.list_patients():
            entry = dict(p)
            sess = self.patient_sessions(entry["id"], per_patient_limit)
            entry["sessions"] = sess
            entry["n_recent"] = len(sess)
            entry["last_session_ts"] = sess[0]["started_ts"] if sess else None
            out.append(entry)
        return out
