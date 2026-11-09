"""Hardware layer: Picarx wrapper that writes only on change, calibration, ultrasonic, threaded camera."""
import threading
import time

import cv2

from .sc_math import sc_clamp

SERVO_LIMIT = 35


class Calibration:
    """Servo offsets for mounting tolerances (same 3-line text file as before)."""

    def __init__(self, path="calibration.txt"):
        self.path = path
        self.steer = self.pan = self.tilt = 0

    def load(self):
        try:
            with open(self.path, "r") as fh:
                self.steer, self.pan, self.tilt = (int(fh.readline()) for _ in range(3))
        except (FileNotFoundError, ValueError):
            return False
        return True


class Vehicle:
    """Drive motor plus steering and camera servos. Hardware is touched only when a value changes."""

    def __init__(self, px, calibration):
        self.px = px
        self.cal = calibration
        self._last = None

    def apply(self, cmd):
        speed = int(round(sc_clamp(cmd.speed, -100, 100)))
        angles = [int(round(sc_clamp(v, -SERVO_LIMIT, SERVO_LIMIT))) for v in (cmd.steer, cmd.pan, cmd.tilt)]
        key = (speed,) + tuple(angles)
        if key == self._last:
            return
        old = self._last or (None, None, None, None)
        if speed != old[0]:
            if speed < 0:
                self.px.backward(-speed)
            else:
                self.px.forward(speed)
        setters = (self.px.set_dir_servo_angle, self.px.set_cam_pan_angle, self.px.set_cam_tilt_angle)
        offsets = (self.cal.steer, self.cal.pan, self.cal.tilt)
        for i, setter in enumerate(setters):
            if angles[i] != old[i + 1]:
                setter(sc_clamp(angles[i] + offsets[i], -SERVO_LIMIT, SERVO_LIMIT))
        self._last = key

    def stop(self):
        self.px.stop()
        self._last = None          # force a full rewrite on the next apply()

    def shutdown(self):
        """Stop the motor and leave the servos centered."""
        self.px.stop()
        for setter, off in ((self.px.set_dir_servo_angle, self.cal.steer),
                            (self.px.set_cam_pan_angle, self.cal.pan),
                            (self.px.set_cam_tilt_angle, self.cal.tilt)):
            setter(sc_clamp(off, -SERVO_LIMIT, SERVO_LIMIT))
        self._last = None


class DistanceSensor:
    """Ultrasonic distance in cm; keeps the previous reading to reject glitches."""

    MAX_DISTANCE_CM = 255
    EMERGENCY_STOP_CM = 20
    MAX_TRUSTED_JUMP_CM = 100

    def __init__(self, ultrasonic):
        self._ultrasonic = ultrasonic
        self.distance = self.MAX_DISTANCE_CM
        self.last_distance = self.MAX_DISTANCE_CM

    def measure(self):
        d = round(self._ultrasonic.read(), 2)
        if d <= -2:                  # -1 = signal error (ignored), -2 = timeout (nothing in range)
            d = self.MAX_DISTANCE_CM
        if d >= 0:
            self.last_distance = self.distance
            self.distance = d

    def obstacle_ahead(self):
        return (self.distance < self.EMERGENCY_STOP_CM
                and abs(self.last_distance - self.distance) < self.MAX_TRUSTED_JUMP_CM)


class ThreadedCamera:
    """Reads the camera in its own thread; consumers get the newest frame without waiting.

    Frames are never modified after publishing, so they can be shared by reference.
    Consumers that draw must copy first.
    """

    def __init__(self, index=1, width=640, height=480, fps=30):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            print("[CAM] unable to open camera", index)
        self._lock = threading.Lock()
        self._frame = None
        self._ts = 0.0
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._halt.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame, self._ts = frame, time.monotonic()
            else:
                self._halt.wait(0.05)

    def latest(self):
        """(frame, monotonic capture time); frame is None until the first image arrives."""
        with self._lock:
            return self._frame, self._ts

    def stop(self):
        self._halt.set()
        self._thread.join(1.0)
        self._cap.release()
