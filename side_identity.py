"""Left/right identity stabilisation for RunTrack landmarks.

MediaPipe labels every landmark independently on each frame.  During a run the
left/right labels of the limbs can flip — most often exactly when a leg or an
arm comes up and crosses in front of / behind its pair — so the "left" tiles
suddenly carry the right limb's values and back.  That is the left/right
confusion seen in the live view.

``SideIdentity`` keeps one identity per limb group (legs, arms) across frames:

* each side's key joints are tracked with a small velocity-aware predictor
  (knee/ankle/heel/foot for the legs, elbow/wrist for the arms);
* a per-group correction BIT decides whether the current frame's left/right
  labels are exchanged; the bit only toggles when the swapped reading fits the
  tracker much better than the current one for several consecutive frames
  (during those frames the mismatched readings are NOT adopted, so the
  evidence survives — see ``_update_group``);
* while the two sides are glued together (mid-crossing, occlusion) no
  decision is taken at all — identity simply holds the previous lock;
* a side whose key joints are not reliably visible is reported as ``hidden``
  so the UI can say "far side hidden behind the body" instead of presenting
  the near limb's numbers as if both sides were measured.

The correction permutes the whole kinematic branch (hip..foot / shoulder..
wrist) because a label flip is a per-branch property: splitting a branch
(a swapped knee with an unswapped hip) would make every joint angle that
spans the branch mix limbs.

The module never changes visibility values — metric gating stays the single
source of truth for "is this landmark trustworthy" (see ``config.VIS_MIN``).
It only keeps the left/right labels anatomically consistent over time.

Signal chain::

    detect ──► SideIdentity.update ──► 1-Euro ──► frame metrics

Reset it when the video time base jumps (new source, seek), together with the
smoothing filter.
"""
from __future__ import annotations

import numpy as np

import config

# ------------------------------------------------------------------- groups
# Every index of a branch is swapped together; the "key" joints are the ones
# that carry the identity information (they are the ones that cross / occlude)
# and therefore drive the decision.
GROUPS: dict = {
    "leg": {
        "left": (23, 25, 27, 29, 31),          # hip, knee, ankle, heel, foot
        "right": (24, 26, 28, 30, 32),
        "key": ((25, 27, 29, 31), (26, 28, 30, 32)),
    },
    "arm": {
        "left": (11, 13, 15, 17, 19, 21),      # shoulder, elbow, wrist, ..hand
        "right": (12, 14, 16, 18, 20, 22),
        "key": ((13, 15), (14, 16)),           # elbow + wrist decide
    },
}

_DT_MIN, _DT_MAX = 1.0 / 120.0, 0.5
_POS_DECAY = 0.55      # EMA weight of the new observation
_VEL_DECAY = 0.40      # EMA weight of the new velocity estimate


class _Limb:
    """Velocity-aware position tracker for one limb side (key joints only)."""

    __slots__ = ("pos", "vel", "t", "miss", "seen")

    def __init__(self):
        self.pos = None          # (k, 2) filtered positions
        self.vel = None          # (k, 2) velocity, units / s
        self.t = None
        self.miss = 0            # consecutive frames without a confident read
        self.seen = 0            # total confident frames

    def predicted(self, t) -> np.ndarray | None:
        if self.pos is None:
            return None
        if self.vel is None or self.t is None:
            return self.pos.copy()
        dt = min(max(float(t) - float(self.t), _DT_MIN), _DT_MAX)
        return self.pos + self.vel * dt

    def observe(self, obs: np.ndarray, mask: np.ndarray, t) -> None:
        """Update with the confident observations (mask True where valid)."""
        if self.pos is None or self.pos.shape != obs.shape:
            if not mask.any():
                return
            fallback = obs[mask].mean(axis=0)
            self.pos = np.where(mask[:, None], obs, fallback).astype(np.float32)
            self.vel = np.zeros_like(self.pos)
            self.t = float(t)
            self.seen = int(mask.sum())
            return
        dt = min(max(float(t) - float(self.t), _DT_MIN), _DT_MAX)
        pred = self.pos + self.vel * dt if self.vel is not None else self.pos
        m = mask[:, None]
        new_pos = np.where(m, (1.0 - _POS_DECAY) * pred + _POS_DECAY * obs,
                           self.pos).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            inst_vel = (new_pos - self.pos) / dt
        new_vel = np.where(m, (1.0 - _VEL_DECAY) * (self.vel if self.vel is not None else 0.0)
                           + _VEL_DECAY * inst_vel, self.vel if self.vel is not None else 0.0)
        self.pos = new_pos
        self.vel = new_vel.astype(np.float32)
        self.t = float(t)
        self.seen += int(mask.sum())

    def decay(self) -> None:
        self.miss += 1
        if self.vel is not None:
            self.vel = self.vel * 0.7      # coast, but drop slowly
        if self.miss > 60:                 # long loss: re-acquire from scratch
            self.pos = None
            self.vel = None
            self.t = None


class SideIdentity:
    """Keep left/right limb labels anatomically consistent across frames.

    Parameters
    ----------
    min_vis : float
        Visibility a key joint needs to count as "observed" (default
        ``config.VIS_MIN``).
    swap_ratio : float
        The swapped fit must be at least this much better than the kept fit
        (``cost_swap < swap_ratio * cost_keep``, default 0.6) before a
        correction can be confirmed.
    confirm_frames : int
        Consecutive frames the swapped reading must fit better before the
        correction bit is toggled (default 3).
    hold_frames : int
        Frames after a toggle during which the bit cannot toggle again
        (default 12).
    glue_dist : float
        When the two sides are closer than this (normalized units) their
        identity is ambiguous (mid-crossing) and no decision is taken.
    min_move : float
        Minimum mismatch (normalized units / frame-set) below which a swap can
        never trigger — guards the "everything standing still" case.
    hidden_share : float
        A side counts as hidden when fewer than this share of its key joints
        is confidently visible.
    """

    def __init__(self, min_vis: float | None = None, swap_ratio: float = 0.6,
                 confirm_frames: int = 3, hold_frames: int = 12,
                 glue_dist: float = 0.035, min_move: float = 0.02,
                 hidden_share: float = 0.5):
        self.min_vis = float(config.VIS_MIN if min_vis is None else min_vis)
        self.swap_ratio = float(swap_ratio)
        self.confirm_frames = int(confirm_frames)
        self.hold_frames = int(hold_frames)
        self.glue_dist = float(glue_dist)
        self.min_move = float(min_move)
        self.hidden_share = float(hidden_share)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._limb = {g: {s: _Limb() for s in ("left", "right")} for g in GROUPS}
        self._flip = {g: False for g in GROUPS}   # correction bit per group
        self._streak = {g: 0 for g in GROUPS}
        self._hold = {g: 0 for g in GROUPS}
        self._swaps = {g: 0 for g in GROUPS}
        self._hidden = {g: set() for g in GROUPS}

    def info(self) -> dict:
        """JSON-safe summary for the live state / session summary."""
        return {
            "swaps": {g: int(self._swaps[g]) for g in GROUPS},
            "total_swaps": int(sum(self._swaps.values())),
            "hidden": {g: sorted(self._hidden[g]) for g in GROUPS},
        }

    # -------------------------------------------------------------- internals
    def _key_obs(self, lms: np.ndarray, group: str, side: str):
        """Key-joint positions + visibility mask for one side of a group."""
        idx = GROUPS[group]["key"][0 if side == "left" else 1]
        obs = np.asarray(lms[list(idx), 0:2], dtype=np.float64)
        vis = np.asarray(lms[list(idx), 3], dtype=np.float64)
        return obs, (vis >= self.min_vis)

    def _update_group(self, lms: np.ndarray, group: str, t: float) -> bool:
        """Track one group; returns True when this frame's labels were permuted.

        Order matters: the identity decision runs on the RAW (input) labels
        against the tracker (which follows the corrected output); the flip is
        applied next, so the hidden bookkeeping and the tracker updates work
        on anatomically labelled observations.
        """
        spec = GROUPS[group]
        obs, mask = {}, {}
        for side in ("left", "right"):
            obs[side], mask[side] = self._key_obs(lms, group, side)

        # ---- identity decision -------------------------------------------
        # The correction is a persistent BIT: while it is on, this frame's
        # labels are exchanged.  A label flip in the input shows up as ONE
        # hard mismatch against the tracker (which follows the corrected
        # output); the evidence only survives while the tracker refuses to
        # adopt the mismatched reading, so during a streak the observations
        # are not fed to the tracker.  Transient noise falls back to "no
        # evidence" on the very next frame and nothing happens; a sustained
        # flip toggles the bit after confirm_frames.
        flip = False
        freeze = False
        both = mask["left"] & mask["right"]          # joints visible on both
        enough = int(both.sum()) >= 2
        cost_as = cost_sw = None
        if enough:
            o_l, o_r = obs["left"][both], obs["right"][both]
            p_l = self._limb[group]["left"].predicted(t)
            p_r = self._limb[group]["right"].predicted(t)
            dist_oo = float(np.mean(np.linalg.norm(o_l - o_r, axis=1)))
            if p_l is not None and p_r is not None and dist_oo > self.glue_dist:
                p_l, p_r = p_l[both], p_r[both]
                cost_as = (float(np.mean(np.linalg.norm(o_l - p_l, axis=1)))
                           + float(np.mean(np.linalg.norm(o_r - p_r, axis=1))))
                cost_sw = (float(np.mean(np.linalg.norm(o_l - p_r, axis=1)))
                           + float(np.mean(np.linalg.norm(o_r - p_l, axis=1))))
        if cost_as is not None and self._hold[group] <= 0:
            cur = cost_sw if self._flip[group] else cost_as
            alt = cost_as if self._flip[group] else cost_sw
            if alt < self.swap_ratio * cur and max(cur, alt) > self.min_move:
                self._streak[group] += 1
                freeze = True
            else:
                self._streak[group] = 0
            if self._streak[group] >= self.confirm_frames:
                self._flip[group] = not self._flip[group]
                self._swaps[group] += 1
                self._streak[group] = 0
                self._hold[group] = self.hold_frames
                freeze = False       # the (new) reading is adopted below
        elif self._hold[group] <= 0:
            self._streak[group] = 0

        if self._hold[group] > 0:
            self._hold[group] -= 1

        flip = self._flip[group]
        if flip:
            li = np.asarray(spec["left"], dtype=int)
            ri = np.asarray(spec["right"], dtype=int)
            tmp = lms[li, :].copy()
            lms[li, :] = lms[ri, :]
            lms[ri, :] = tmp
            # observations swap sides with the labels
            obs["left"], obs["right"] = obs["right"], obs["left"]
            mask["left"], mask["right"] = mask["right"], mask["left"]

        # ---- hidden bookkeeping (on the corrected, anatomical labels) -----
        for side in ("left", "right"):
            share = float(mask[side].sum()) / max(1, mask[side].size)
            if share < self.hidden_share:
                self._limb[group][side].decay()
                if self._limb[group][side].miss >= 2:
                    self._hidden[group].add(side)
            else:
                self._hidden[group].discard(side)

        # ---- tracker update ----------------------------------------------
        # freeze: a not-yet-confirmed mismatch is not adopted (see above)
        if not freeze:
            for side in ("left", "right"):
                if mask[side].any():
                    self._limb[group][side].observe(obs[side], mask[side], t)
        return flip

    # ---------------------------------------------------------------- public
    def update(self, lms, t: float):
        """One frame.

        ``lms`` is the raw (33, 4) landmark array (or None when no pose was
        found).  Returns ``(lms_out, info)``; ``lms_out`` is the same array
        with left/right labels corrected (a copy — the input is left alone)
        or None.  ``info`` is the JSON-safe summary (see :meth:`info`).
        """
        if lms is None:
            for g in GROUPS:
                for side in ("left", "right"):
                    self._limb[g][side].decay()
                self._hold[g] = max(0, self._hold[g] - 1)
            return None, self.info()
        arr = np.array(lms, dtype=np.float32, copy=True)
        if arr.ndim != 2 or arr.shape[0] < 33:
            return arr, self.info()
        for group in GROUPS:
            self._update_group(arr, group, t)
        return arr, self.info()
