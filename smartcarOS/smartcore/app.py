"""Wiring and the main loop. tick() is the whole control pipeline:
   mode (source of a command) -> rules (may slow/stop/steer) -> safety (always last) -> vehicle."""
import collections
import signal
import threading
import time

import cv2

from .bus import Bus
from .config import load_config
from .control import MODES, LaneFollower, RemoteDrive, Safety
from .hal import Calibration, DistanceSensor, ThreadedCamera, Vehicle
from .perception import DetectorWorker, NullDetector, build_detector
from .rules import load_rules
from .stream import MJPEGServer
from .types import Cmd, World

# Legacy Node-RED mode names -> (mode, traffic light source)
ALIASES = {"lanedetection": ("auto", None), "tlviamqtt": ("auto", "mqtt"), "tlviacnn": ("auto", "detections")}


class SmartCore:
    def __init__(self, cfg, vehicle, camera, rules):
        self.cfg, self.vehicle, self.camera, self.rules = cfg, vehicle, camera, rules
        self.world = World(cfg.detect_max_age_s)
        self.remote = RemoteDrive(cfg.remote_timeout_s)
        self.safety = Safety(cfg)
        self.lane = LaneFollower()
        self.mode = MODES["standby"]
        self.running = True
        self.last_cmd = Cmd()
        self._mode_q = collections.deque()          # filled by the MQTT thread, drained in tick()
        self._rule_err = {}

    def request_mode(self, name):
        self._mode_q.append(name)

    def set_mode(self, name, now):
        if name == "quit":
            self.running = False
            return
        real, tl = ALIASES.get(name, (name, None))
        if real not in MODES:
            print("[CORE] unknown mode %r ignored" % name)
            return
        self.world.tl_source = tl
        self.world.mode = name
        self.mode = MODES[real]
        self.mode.start(self, now)

    def tick(self, now):
        w = self.world
        w.notes = []
        while self._mode_q:
            self.set_mode(self._mode_q.popleft(), now)
        if w.estop and self.mode.name != "standby":
            self.set_mode("standby", now)           # releasing estop must not resume driving
        cmd = self.mode.step(self, now)
        for r in self.rules:
            if self.mode.name not in r.modes:
                continue
            try:
                r.apply(cmd, w, now)
            except Exception as exc:                # fail safe: a broken rule stops the car
                cmd.speed = 0.0
                w.notes.append("%s: FAILED %s" % (r.name, exc))
                if now - self._rule_err.get(r.name, -1e9) > 5.0:
                    print("[RULE] %s failed: %r" % (r.name, exc))
                    self._rule_err[r.name] = now
        out = self.safety.apply(cmd, w)
        self.vehicle.apply(out)
        self.last_cmd = out
        return out

    def loop(self, bus=None):
        cfg = self.cfg
        dt = 1.0 / cfg.tick_hz
        nxt = last_status = last_t = time.monotonic()
        self.set_mode(cfg.start_mode, nxt)
        while self.running:
            now = time.monotonic()
            out = self.tick(now)
            self.world.fps = 0.9 * self.world.fps + 0.1 / max(now - last_t, 1e-3)
            last_t = now
            if bus is not None and now - last_status >= 1.0 / cfg.status_hz:
                bus.publish_status(out)
                last_status = now
            nxt += dt
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()              # fell behind: do not try to catch up


class Viewer(threading.Thread):
    """Draws overlays on a copy of the newest frame and feeds the stream. Idle without viewers."""

    def __init__(self, core, stream):
        super().__init__(daemon=True)
        self.core, self.stream = core, stream
        self._halt = threading.Event()

    def run(self):
        cfg, last = self.core.cfg, None
        while not self._halt.is_set():
            frame, ts = self.core.camera.latest()
            if self.stream.clients == 0 or frame is None or ts == last:
                self._halt.wait(0.1 if self.stream.clients == 0 else 0.02)
                continue
            last = ts
            img = frame.copy()
            self.draw(img)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, cfg.stream_quality])
            if ok:
                self.stream.publish(jpg.tobytes())
            self._halt.wait(1.0 / cfg.stream_fps)

    def draw(self, img):
        core, w = self.core, self.core.world
        h, wd = img.shape[:2]
        for d in w.fresh(time.monotonic()):
            p1, p2 = (int(d.x1 * wd), int(d.y1 * h)), (int(d.x2 * wd), int(d.y2 * h))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            cv2.putText(img, "%s %.2f" % (d.label, d.conf), (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        lane = core.lane.lane
        if core.mode.name in ("assist", "auto") and lane is not None and lane.target_x is not None:
            cv2.line(img, (int(lane.target_x), int(h * 0.5)), (int(lane.target_x), int(h * 0.85)), (255, 0, 0), 2)
        c = core.last_cmd
        lines = ["%s  v=%d steer=%d  d=%d cm%s" % (w.mode, c.speed, c.steer, w.distance,
                                                    "  ESTOP" if w.estop else "")] + w.notes[:3]
        for i, text in enumerate(lines):
            cv2.putText(img, text, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def stop(self):
        self._halt.set()
        self.join(1.0)


def _sensor_loop(core, sensor, halt):
    while not halt.is_set():
        try:
            sensor.measure()
            core.world.distance = sensor.distance
            core.world.obstacle = sensor.obstacle_ahead()
        except Exception as exc:
            print("[SENSOR]", exc)
        halt.wait(0.05)


def main():
    cfg = load_config()
    from picarx import Picarx                       # imported here so tests need no hardware
    px = Picarx()
    cal = Calibration(cfg.calibration_file)
    cal.load()
    vehicle = Vehicle(px, cal)
    camera = ThreadedCamera(cfg.camera_index, cfg.frame_width, cfg.frame_height)
    core = SmartCore(cfg, vehicle, camera, load_rules(cfg.rules_file))

    def on_signal(sig, frame):
        core.running = False
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    halt = threading.Event()
    workers = []
    stream = bus = None
    try:
        camera.start()
        detector = build_detector(cfg.detector)
        if not isinstance(detector, NullDetector):
            worker = DetectorWorker(detector, camera, core.world)
            worker.start()
            workers.append(worker)
        if cfg.use_ultrasonic:
            threading.Thread(target=_sensor_loop, args=(core, DistanceSensor(px.ultrasonic), halt),
                             daemon=True).start()
        stream = MJPEGServer(cfg.stream_port)
        stream.start()
        viewer = Viewer(core, stream)
        viewer.start()
        workers.append(viewer)
        bus = Bus(core, cfg)
        bus.start()
        print("[CORE] running; stream http://<car>:%d/stream.mjpg" % stream.port)
        core.loop(bus)
    finally:                                         # no sys.exit here: a crash must keep its exit code
        print("[CORE] shutting down")
        vehicle.shutdown()
        halt.set()
        for wk in workers:
            wk.stop()
        if bus:
            bus.stop()
        if stream:
            stream.stop()
        camera.stop()


if __name__ == "__main__":
    main()
