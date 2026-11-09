"""Command sources (modes), the remote deadman and the final safety layer."""
import math

from .lane import LaneDetector, LaneResult, LaneSteeringController, coupled_camera_pan
from .sc_math import sc_clamp
from .types import Cmd


class RemoteDrive:
    """Latest drive command from MQTT (gamepad, Node-RED, ...). Stale commands lose their speed."""

    def __init__(self, timeout_s=0.3):
        self.timeout_s = timeout_s
        self._cmd = Cmd()
        self._ts = -1e9

    def set(self, data, now):
        """Validate and store a command dict. Returns False (and keeps the old one) if it is garbage."""
        try:
            vals = [float(data.get(key, 0.0)) for key in ("speed", "steer", "pan", "tilt")]
        except (AttributeError, TypeError, ValueError):
            return False
        if not all(math.isfinite(v) for v in vals):
            return False
        self._cmd = Cmd(sc_clamp(vals[0], -100, 100), sc_clamp(vals[1], -35, 35),
                        sc_clamp(vals[2], -35, 35), sc_clamp(vals[3], -35, 35))
        self._ts = now
        return True

    def get(self, now):
        cmd = self._cmd.copy()
        if now - self._ts > self.timeout_s:
            cmd.speed = 0.0
        return cmd

    def clear(self):
        self._cmd = Cmd()


class Safety:
    """Last stage, always applied, not removable by rules: estop, ultrasonic, caps, acceleration limit."""

    def __init__(self, cfg):
        self.max_speed = cfg.max_speed
        self.accel = cfg.accel_per_tick
        self.prev = 0.0

    def apply(self, cmd, world):
        out = cmd.copy()
        if not all(math.isfinite(v) for v in (out.speed, out.steer, out.pan, out.tilt)):
            out = Cmd()
        out.steer = sc_clamp(out.steer, -35, 35)
        out.pan = sc_clamp(out.pan, -35, 35)
        out.tilt = sc_clamp(out.tilt, -35, 35)
        if world.estop:
            world.notes.append("safety: estop")
            self.prev = 0.0
            out.speed = 0.0
            return out
        if out.speed > 0 and world.obstacle:
            world.notes.append("safety: obstacle %d cm" % world.distance)
            out.speed = 0.0
        cap = self.max_speed * sc_clamp(world.speed_limit, 0.0, 1.0)
        out.speed = sc_clamp(out.speed, -cap, cap)
        p = self.prev
        if out.speed > p and out.speed > 0:            # speeding up forward: ramp
            out.speed = min(out.speed, max(p, 0.0) + self.accel)
        elif out.speed < p and out.speed < 0:          # speeding up backward: ramp
            out.speed = max(out.speed, min(p, 0.0) - self.accel)
        self.prev = out.speed                          # braking / reversing direction is immediate
        return out


class LaneFollower:
    """Wraps the lane detector and steering controller; keeps their state between ticks."""

    def __init__(self):
        self.detector = LaneDetector()
        self.steering = LaneSteeringController()
        self.lane = LaneResult()
        self.steer = 0.0
        self.pan = 0.0
        self.good_at = None
        self._last_ts = None

    def reset(self, now):
        self.good_at = now                 # grace period before "lane lost"
        self.steer = self.pan = 0.0

    def update(self, frame, ts, moving, now):
        if frame is not None and ts != self._last_ts:
            self._last_ts = ts
            mask = self.detector.denoise(self.detector.color_mask(frame))
            self.lane = self.detector.detect(mask)
            if self.lane.lane_count >= 2:
                self.good_at = now
            if moving:                     # hold the steering angle while standing still
                self.steer = self.steering.update(self.lane, self.steer, frame.shape[1])
                self.pan = coupled_camera_pan(self.steer)
        return self.steer, self.pan

    def lost(self, now, limit_s):
        return self.good_at is None or now - self.good_at > limit_s


class Mode:
    name = ""

    def start(self, core, now):
        pass

    def step(self, core, now):
        return Cmd()


class Standby(Mode):
    name = "standby"


class Manual(Mode):
    """Gamepad / Node-RED drives, no vision."""
    name = "manual"

    def step(self, core, now):
        return core.remote.get(now)


class Assist(Mode):
    """Human controls speed (forward only), the lane follower steers."""
    name = "assist"

    def start(self, core, now):
        core.lane.reset(now)

    def step(self, core, now):
        r = core.remote.get(now)
        frame, ts = core.camera.latest()
        steer, pan = core.lane.update(frame, ts, r.speed > 0, now)
        return Cmd(max(0.0, r.speed), steer, pan, r.tilt)


class Auto(Mode):
    """Fully autonomous lane following at auto_speed; rules decide when to slow down or stop."""
    name = "auto"

    def start(self, core, now):
        core.lane.reset(now)

    def step(self, core, now):
        frame, ts = core.camera.latest()
        steer, pan = core.lane.update(frame, ts, True, now)
        if core.lane.lost(now, core.cfg.lane_lost_s):
            core.world.notes.append("auto: lane lost")
            return Cmd(0.0, steer, pan, 0.0)
        return Cmd(core.cfg.auto_speed, steer, pan, 0.0)


MODES = {m.name: m for m in (Standby(), Manual(), Assist(), Auto())}
