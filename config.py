"""RunTrack configuration and paths."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"
# Higher-accuracy variant used for the keyframe HD refinement pass (per-event,
# offline — its speed does not matter).  Falls back to MODEL_PATH when absent.
MODEL_PATH_HEAVY = MODELS_DIR / "pose_landmarker_heavy.task"
DB_PATH = DATA_DIR / "runtrack.db"

# Bind address / port.  RUNTRACK_HOST=0.0.0.0 opens the UI to other devices on
# the local network; the defaults keep it strictly local.
HOST = os.environ.get("RUNTRACK_HOST", "127.0.0.1")
PORT = int(os.environ.get("RUNTRACK_PORT", "8780"))

PROC_WIDTH = 960          # processing width for pose detection
STREAM_FPS = 15           # MJPEG stream pacing
JPEG_QUALITY = 80
TARGET_FPS = 60           # upper bound the pose model is asked to keep up with

DEFAULT_DURATION_S = 60   # default run session duration
MAX_DURATION_S = 900
MAX_UPLOAD_MB = 800

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.8:latest")  # default: the local vision model

VIS_MIN = 0.5             # minimum landmark visibility to use a landmark

# ---------------------------------------------------------------- view plane
# The camera view the session was filmed in.  Metric families that cannot be
# measured from a given plane are reported as NOT_ASSESSED_TEXT instead of a
# fabricated number (see metrics.analyze_session).
VIEW_PLANES = ("sagittal", "frontal")
DEFAULT_VIEW_PLANE = "sagittal"
NOT_ASSESSED_TEXT = "not assessed (wrong camera view)"
CYCLING_NOT_APPLICABLE_TEXT = "not applicable to a cycling session"

# Metrics that only exist in a sagittal (side) view / only in a frontal view.
SAGITTAL_ONLY = ("touchdown_shin_deg", "overstride_leg_frac", "knee_flexion_ic_deg",
                 "knee_flexion_ms_deg", "trunk_lean_deg", "hip_extension_toeoff_deg")
FRONTAL_ONLY = ("pelvic_drop_deg", "knee_valgus_deg", "arm_swing_sym_pct")

# ------------------------------------------------------------- 1-Euro filter
# 1-Euro landmark smoothing (Casiez, Roussel & Vogel 2012). Applied to the raw
# MediaPipe landmarks BEFORE metric computation; raw landmarks stay available
# for event detection / re-analysis. mincutoff removes high-frequency jitter,
# beta keeps latency low during fast athletic movement.
ONEEURO_ENABLED = True
ONEEURO_MIN_CUTOFF = 1.2       # Hz - lower = smoother, slightly more lag
ONEEURO_BETA = 0.35            # speed coefficient - higher = less lag on fast moves
ONEEURO_DCUTOFF = 1.0          # Hz - cutoff for the derivative estimate
ONEEURO_FREQ = 30.0            # fallback sampling rate (Hz) when no timestamp is given

# ------------------------------------------------- gait events + keyframes
KEYFRAME_DIR = "keyframes"
KEYFRAMES_MAX = 9              # best freeze-frames kept per session
KEYFRAMES_PER_TYPE = 3         # per event type (IC / MS / TO)
KEYFRAME_JPEG_QUALITY = 90     # clean (unannotated) frame stored per event
GAIT_BUFFER_FRAMES = 180       # ring buffer of recent frames (~6 s @30fps)
FRAME_RING = 16                # full-resolution frames kept for keyframe HD pass
KEYFRAME_CROP_PAD = 0.12       # bbox padding (share of span) for the HD crop
# HD keyframe refinement: re-run the (heavy) pose model on a native-resolution
# crop of each kept keyframe at session end.  Per keyframe, offline — the live
# loop is not affected.
REFINE_KEYFRAMES = True

# ------------------------------------------------------- side identity
# Undo MediaPipe left/right label flips per limb branch (legs, arms) and
# report a side whose key joints are hidden (see side_identity.py).
SIDE_IDENTITY_ENABLED = True
GAIT_STANCE_FRAC = 0.30        # ground band = top 30 % of the ankle-height range
GAIT_MIN_GAP_S = 0.18          # minimum time between two events of the same type
GAIT_LOOKAHEAD = 3             # frames used to confirm a toe-off
GAIT_LOOKBACK = 4              # frames used to refine an initial contact
# Initial contact = end of the landing descent: the descent counts as finished
# once its speed has decayed to this share of its own peak (the foot has
# stopped coming down).  See gait_events._advance_side for the shared machine.
GAIT_CONTACT_STOP_FRAC = 0.30
# A foot-lift (swing) episode must reach this many band-heights per second —
# but never less than GAIT_RISE_MIN — before it counts as a real toe-off.
GAIT_RISE_BAND_FRAC = 1.0
GAIT_RISE_MIN = 0.03
GAIT_EPISODE_MIN_FRAMES = 3    # shorter velocity bursts are landmark noise
# The session-opening check ("was the recording started with the foot already
# planted?") only runs once the foot-lift band is this big, so an immature
# rolling window cannot decide it wrongly (the check is one-shot).
GAIT_START_BAND_MIN = 0.05

# Gait-phase analysis needs the lower body in frame. When the foot/ankle
# landmarks are below this share of the session the events are reported as
# "not assessable" (never as a silent zero) — thresholds are NOT relaxed to
# make a clip fire.
LOWER_BODY_MIN_COVERAGE_PCT = 20.0
GAIT_NOT_ASSESSABLE_TEXT = ("gait events not assessable (lower-body landmarks "
                            "not visible in this footage)")

# ----------------------------------------------------------- media playback
PLAYBACK_SPEEDS = (0.25, 0.5, 0.75, 1.0)

# ------------------------------------------------------------ audio feedback
METRONOME_MIN_BPM = 120
METRONOME_MAX_BPM = 240
METRONOME_STEP_BPM = 5
METRONOME_PRESETS = (170, 180)
VOICE_COOLDOWN_S = 12          # min seconds between two spoken cues (no chatter)

# Reference bands for recreational runners at moderate pace (heuristic,
# from common running-gait literature; NOT clinical norms).
BANDS = {
    "cadence_spm": (160, 190),
    "trunk_lean_deg": (4, 12),           # anterior lean (forward-normalised)
    "vertical_osc_cm": (6.0, 9.5),       # target band for a moderate pace
    "knee_flexion_max_deg": (80, 125),   # peak knee flexion (min angle) midswing
    "symmetry_pct": (90, 100),
    # --- new biomechanical bands (label source in README) ---
    "touchdown_shin_deg": (0.0, 5.0),    # knee->ankle vs vertical at contact
    "knee_flexion_ms_deg": (35.0, 45.0),  # absorption at mid-stance
    "pelvic_drop_deg": (0.0, 4.0),       # contralateral drop during single stance
    "knee_valgus_deg": (0.0, 10.0),      # frontal knee collapse vs hip-ankle axis
    "overstride_leg_frac": (0.0, 0.15),  # overstride / estimated leg length
    "power_per_kg_w": (1.5, 4.5),        # vertical power per kg body mass
    "arm_swing_sym_pct": (90, 100),      # frontal: left/right arm-swing symmetry
}

# Warning thresholds (used for flags / risk stratification, not for bands).
FLAGS = {
    "touchdown_shin_high_deg": 8.0,   # >8 deg = high braking force
    "pelvic_drop_high_deg": 5.0,      # >4-5 deg = gluteus-medius weakness potential
    "knee_valgus_high_deg": 12.0,
    "cadence_low_spm": 160.0,
    "vertical_osc_high_cm": 10.0,
    "overstride_high_frac": 0.15,
}

# Height-based scale estimate used when no metric scale is available.
# leg length ~ 0.47 x body height is a common anthropometric approximation.
LEG_LENGTH_HEIGHT_FRACTION = 0.47

# ------------------------------------------------- cycle / bike-fit settings
# Session discipline: running (treadmill/overground gait) or cycling (road /
# TT / triathlon bike position).  Cycling uses its own metric set and its own
# reference targets (see cycling.py; sources documented in README).
DISCIPLINES = ("running", "cycling")
DEFAULT_DISCIPLINE = "running"
BIKE_TYPES = ("road", "tt", "tri")
DEFAULT_BIKE_TYPE = "road"
DISCIPLINE_LABELS = {"running": "Running", "cycling": "Cycling"}
BIKE_LABELS = {"road": "Road bike", "tt": "Time trial", "tri": "Triathlon"}

# Reference angle targets (degrees, included joint angle) by bike type.
# Sources: Velogic Studio triathlon/road fit metric ranges (industry-standard
# video fit tool), Holmes method 25-35 deg knee flexion at BDC (= 145-155
# included; slightly stricter band used here), Slowtwitch/TT fit convention
# (knee 137-143 included), common bike-fit guidance (torso 35-45 road).
CYCLING_BANDS = {
    "knee_bdc_deg": {"road": (140.0, 150.0), "tt": (140.0, 145.0),
                     "tri": (140.0, 145.0)},
    "knee_tdc_deg": {"road": (68.0, 95.0), "tt": (68.0, 95.0),
                     "tri": (68.0, 95.0)},          # Velogic: min > 68
    "hip_tdc_deg": {"road": (70.0, 90.0), "tt": (45.0, 75.0),
                    "tri": (45.0, 75.0)},            # closed hip angle at top
    "torso_deg": {"road": (35.0, 45.0), "tt": (10.0, 25.0),
                  "tri": (15.0, 30.0)},              # vs horizontal
    "ankle_deg": {"road": (80.0, 100.0), "tt": (100.0, 120.0),
                  "tri": (100.0, 120.0)},
}
# Hard warning thresholds (flag, not band) for cycling.
CYCLING_FLAGS = {
    "knee_bdc_high_deg": 152.0,     # knee too extended -> saddle too high
    "knee_bdc_low_deg": 135.0,      # too flexed -> saddle too low / PF load
    "hip_tdc_low_deg": 42.0,        # hip too closed (breathing/compression)
    "pelvic_obliquity_high_deg": 5.0,   # excess hip rocking
    "knee_lateral_travel_high_mm": 40.0,
}
# Rear-/frontal-view control targets (peak-to-peak travel).  Velogic "joint
# motion targets": average range / good range (mm where a scale is available,
# otherwise as a share of shoulder width).
REAR_TRAVEL_TARGETS = {
    "knee_lateral_travel_mm": (40.0, 20.0),     # avg / good
    "hip_vertical_travel_mm": (40.0, 15.0),
    "hip_horizontal_travel_mm": (40.0, 10.0),
    "shoulder_lateral_travel_mm": (40.0, 20.0),
    "ankle_swivel_mm": (10.0, 5.0),
}
# Cadence ranges (rpm) commonly used in coaching: road ~ 80-100, TT/tri
# slightly lower self-selected.  Reported, flagged only outside 60-110.
CYCLING_CADENCE_BAND = (70.0, 100.0)
CYCLING_CADENCE_HARD = (55.0, 115.0)

# --------------------------------------------------------- second camera
# Optional simultaneous second view (e.g. rear + side) with its own plane.
DEFAULT_PLANE2 = "frontal"
VIDEO_ASPECT_DEFAULT = 16.0 / 9.0   # assumed w/h when the source is unknown

# -------------------------------------------------- aero (frontal area)
# Video-based projected frontal area: MediaPipe segmentation mask per frame,
# scaled with the rider's shoulder width (or 0.245 x height anthropometry).
AERO_ENABLED_DEFAULT = False        # per-source opt-in checkbox in the UI
SHOULDER_WIDTH_HEIGHT_FRACTION = 0.245    # biacromial breadth / height
AIR_DENSITY_KGM3 = 1.225                   # sea level, 15 C
AERO_CD_DEFAULT = 0.80                     # assumed drag coefficient (rider+bike)

for d in (DATA_DIR, SESSIONS_DIR, UPLOADS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
