"""Synthetic cyclist: landmark model + stick-figure video renderer.

Used by the test-suite to generate ground-truth pedalling kinematics (known
crank rate, known saddle height) and to render a short stick-figure clip that
can be pushed through the real pose pipeline.

Geometry: 2-link leg (thigh L1, shank L2) with the ankle riding the pedal on a
crank circle (radius `crank`) below the hip.  Everything in metres, y down;
normalised image coordinates via `scale` (norm per metre).
"""
from __future__ import annotations

import math

import numpy as np

# default anatomy / setup (metres)
# hip_h = 0.6785 puts the knee at ~145 deg included angle at bottom dead
# centre with the 5 cm foot offset (the classic fit target).
GEOM = dict(L1=0.42, L2=0.42, crank=0.1725, foot_off=0.05,
            hip_h=0.6785, torso_deg=40.0, shoulder_len=0.55,
            scale=0.38, ox=0.45, oy=0.45, obliquity_deg=0.0,
            knee_lat_amp=0.0, hip_rock=0.0)


def _circle_intersect(p1, r1, p2, r2, prefer_forward=True):
    """Intersection of two circles; returns the knee position (metres)."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    d = math.hypot(dx, dy)
    d = min(d, (r1 + r2) * 0.999)
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h2 = max(0.0, r1 * r1 - a * a)
    h = math.sqrt(h2)
    xm, ym = p1[0] + a * dx / d, p1[1] + a * dy / d
    rx, ry = -dy / d, dx / d
    k1 = (xm + h * rx, ym + h * ry)
    k2 = (xm - h * rx, ym - h * ry)
    if prefer_forward:
        return k1 if k1[0] >= k2[0] else k2
    return k1 if k1[1] <= k2[1] else k2


def leg_points(phase, geom=None):
    """Ankle/knee/hip (metres) for one leg at crank `phase` (radians, 0 = up)."""
    g = dict(GEOM)
    g.update(geom or {})
    crank = (g["crank"] * math.sin(phase), g["crank"] * math.cos(phase))  # y down
    hip = (0.0, -g["hip_h"])
    ankle = (crank[0], crank[1] - g["foot_off"])
    knee = _circle_intersect(hip, g["L1"], ankle, g["L2"])
    return hip, knee, ankle


def knee_included_angle(phase, geom=None):
    hip, knee, ankle = leg_points(phase, geom)

    def ang(a, b, c):
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        c_ = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        return math.degrees(math.acos(max(-1.0, min(1.0, c_))))

    return ang(hip, knee, ankle)


def landmarks_sagittal(phase, geom=None):
    """(33,4) landmarks for a side-view cyclist at crank `phase`."""
    g = dict(GEOM)
    g.update(geom or {})
    s, ox, oy = g["scale"], g["ox"], g["oy"]

    def nrm(p):
        return (ox + p[0] * s, oy + p[1] * s)

    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 1.0
    hipL, kneeL, ankL = leg_points(phase, g)
    hipR, kneeR, ankR = leg_points(phase + math.pi, g)
    tor = math.radians(g["torso_deg"])
    sh = (g["shoulder_len"] * math.cos(tor), hipL[1] - g["shoulder_len"] * math.sin(tor))
    head = (sh[0] + 0.10, sh[1] - 0.20)
    bar = (0.38, sh[1] - 0.05)

    def put(i, p):
        x, y = nrm(p)
        lm[i, 0], lm[i, 1] = x, y

    put(11, (sh[0] - 0.004, sh[1]))
    put(12, (sh[0] + 0.004, sh[1]))
    put(23, (hipL[0] - 0.004, hipL[1]))
    put(24, (hipR[0] + 0.004, hipR[1]))
    put(25, kneeL)
    put(26, kneeR)
    put(27, ankL)
    put(28, ankR)
    put(29, (ankL[0] - 0.06, ankL[1] + 0.02))
    put(30, (ankR[0] - 0.06, ankR[1] + 0.02))
    put(31, (ankL[0] + 0.05, ankL[1] + 0.02))
    put(32, (ankR[0] + 0.05, ankR[1] + 0.02))
    # arms: shoulder -> elbow -> wrist (toward the bars)
    for sh_i, el_i, wr_i, off in ((11, 13, 15, -0.004), (12, 14, 16, 0.004)):
        put(el_i, ((sh[0] + bar[0]) / 2 + 0.03, (sh[1] + bar[1]) / 2 + 0.02))
        put(wr_i, (bar[0], bar[1]))
    put(7, (head[0] - 0.02, head[1]))
    put(8, (head[0] + 0.02, head[1]))
    put(0, (head[0] + 0.06, head[1] + 0.01))
    return lm


def landmarks_rear(phase, geom=None):
    """(33,4) landmarks for a rear-view cyclist at crank `phase`."""
    g = dict(GEOM)
    g.update(geom or {})
    s, ox, oy = g["scale"], g["ox"], g["oy"]

    def nrm(p):
        return (ox + p[0] * s, oy + p[1] * s)

    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 1.0
    obl = math.radians(g["obliquity_deg"])
    rock = g["hip_rock"] * math.sin(phase)
    hipL_y = -g["hip_h"] + rock
    hipR_y = -g["hip_h"] - rock
    # pelvic obliquity: oscillate the inter-hip tilt by +/- obliquity_deg
    if g["obliquity_deg"]:
        dy_amp = math.tan(obl) * 0.12
        hipL_y += 0.5 * dy_amp * math.sin(phase)
        hipR_y -= 0.5 * dy_amp * math.sin(phase)
    hipL = (0.06, hipL_y)
    hipR = (-0.06, hipR_y)
    _, kneeL_sag, ankL = leg_points(phase, g)
    _, kneeR_sag, ankR = leg_points(phase + math.pi, g)
    latL = g["knee_lat_amp"] * math.sin(phase)
    latR = g["knee_lat_amp"] * math.sin(phase + math.pi)
    kneeL = (0.06 + latL, kneeL_sag[1])
    kneeR = (-0.06 + latR, kneeR_sag[1])
    ankL2 = (0.045, ankL[1])
    ankR2 = (-0.045, ankR[1])
    shL = (0.155, hipL_y - 0.50)
    shR = (-0.155, hipR_y - 0.50)

    def put(i, p):
        x, y = nrm(p)
        lm[i, 0], lm[i, 1] = x, y

    put(11, shL)
    put(12, shR)
    put(23, hipL)
    put(24, hipR)
    put(25, kneeL)
    put(26, kneeR)
    put(27, ankL2)
    put(28, ankR2)
    put(29, (ankL2[0] - 0.03, ankL2[1] + 0.02))
    put(30, (ankR2[0] + 0.03, ankR2[1] + 0.02))
    put(31, (ankL2[0] + 0.02, ankL2[1] + 0.03))
    put(32, (ankR2[0] - 0.02, ankR2[1] + 0.03))
    put(13, (0.20, shL[1] + 0.18))
    put(14, (-0.20, shR[1] + 0.18))
    put(15, (0.22, shL[1] + 0.34))
    put(16, (-0.22, shR[1] + 0.34))
    put(7, (0.045, shL[1] - 0.10))
    put(8, (-0.045, shR[1] - 0.10))
    put(0, (0.0, shL[1] - 0.15))
    return lm


def render_frame(phase, view="sagittal", size=(640, 360), geom=None):
    """Draw a stick cyclist (BGR image).  Intended for detection probes."""
    import cv2
    w, h = size
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    lm = landmarks_rear(phase, geom) if view == "rear" else landmarks_sagittal(phase, geom)
    P = {i: (int(lm[i, 0] * w), int(lm[i, 1] * h)) for i in
         (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)}
    def seg(a, b, c=(30, 30, 30), t=7):
        cv2.line(img, P[a], P[b], c, t, cv2.LINE_AA)
    # torso + head
    seg(11, 23); seg(12, 24)
    cv2.circle(img, P[0], 13, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.line(img, P[0], ((P[7][0] + P[8][0]) // 2, (P[7][1] + P[8][1]) // 2), (40, 40, 40), 7)
    # arms
    seg(11, 13); seg(13, 15); seg(12, 14); seg(14, 16)
    # legs
    seg(23, 25); seg(25, 27); seg(27, 31)
    seg(24, 26); seg(26, 28); seg(28, 32)
    # crank circle
    g = dict(GEOM)
    g.update(geom or {})
    cx, cy = int((g["ox"]) * w), int((g["oy"]) * h)
    cv2.circle(img, (cx, cy), int(g["crank"] * g["scale"] * w), (120, 120, 200), 3, cv2.LINE_AA)
    return img


def render_video(path, seconds=20.0, fps=30, rpm=60.0, view="sagittal", geom=None,
                 size=(640, 360)):
    """Write a synthetic cycling clip (for pipeline probes / demos)."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, size)
    n = int(seconds * fps)
    omega = 2.0 * math.pi * (rpm / 60.0)
    for k in range(n):
        phase = omega * (k / fps)
        vw.write(render_frame(phase, view=view, size=size, geom=geom))
    vw.release()
    return path
