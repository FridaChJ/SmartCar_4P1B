"""Shared data types: drive command, detection and the world state."""
from dataclasses import dataclass


@dataclass
class Cmd:
    """What the car should do right now. speed -100..100 (%), angles in degrees, +steer = right."""
    speed: float = 0.0
    steer: float = 0.0
    pan: float = 0.0
    tilt: float = 0.0

    def copy(self):
        return Cmd(self.speed, self.steer, self.pan, self.tilt)


@dataclass
class Detection:
    """One detected object. Coordinates are normalized to 0..1, so rules do not depend on resolution."""
    label: str      # lower case, spaces replaced by underscores
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2.0

    @property
    def h(self):
        """Height as a fraction of the frame: a cheap proximity measure."""
        return self.y2 - self.y1


class World:
    """State shared between threads. Every attribute has exactly one writer."""

    def __init__(self, det_max_age=0.6):
        self.mode = "standby"            # name as requested (e.g. tlviacnn), for status
        self.estop = False
        self.obstacle = False            # ultrasonic: something close ahead
        self.distance = 255.0
        self.speed_limit = 1.0           # 0..1, set via MQTT (gamepad / voice)
        self.detections = []
        self.detections_ts = -1e9
        self.det_max_age = det_max_age
        self.lane = None
        self.tl_source = None            # "mqtt" | "detections" | None
        self.tl_phase = None             # last phase received over MQTT
        self.tl_phase_seen = "off"       # phase the traffic light rule currently uses
        self.notes = []                  # why the command was changed this tick
        self.fps = 0.0
        self.infer_ms = 0.0

    def fresh(self, now):
        """Detections that are recent enough to act on (empty list = unknown, not safe)."""
        return self.detections if now - self.detections_ts <= self.det_max_age else []
