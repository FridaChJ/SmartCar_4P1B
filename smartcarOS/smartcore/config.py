"""Settings with defaults; override any of them in config.json (next to the package)."""
import json
import os
from dataclasses import dataclass, field, fields


@dataclass
class Config:
    camera_index: int = 1
    frame_width: int = 640
    frame_height: int = 480
    calibration_file: str = "calibration.txt"
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_user: str = ""
    mqtt_password: str = ""
    tl_topic: str = "TL08"            # traffic light phase published by the lab system
    stream_port: int = 8080
    stream_fps: int = 15
    stream_quality: int = 70
    tick_hz: int = 20
    status_hz: int = 10
    remote_timeout_s: float = 0.3     # deadman: no drive command for this long -> speed 0
    max_speed: float = 60.0           # global speed cap in percent
    accel_per_tick: float = 5.0       # speed may rise by this much per tick, braking is instant
    auto_speed: float = 35.0
    lane_lost_s: float = 2.0          # lane pair not seen for this long -> stop (auto mode)
    start_mode: str = "standby"
    use_ultrasonic: bool = True
    detect_max_age_s: float = 0.6     # detections older than this are ignored by the rules
    detector: dict = field(default_factory=lambda: {"type": "null"})
    rules_file: str = "rules.json"


def load_config(path="config.json"):
    cfg = Config()
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        known = {fld.name for fld in fields(Config)}
        for key, value in data.items():
            if key in known:
                setattr(cfg, key, value)
            else:
                print("[CFG] unknown key ignored:", key)
    return cfg
