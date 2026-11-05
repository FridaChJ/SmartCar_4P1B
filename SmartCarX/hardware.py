"""Thin wrappers around the PiCar-X hardware: camera, motors/servos and ultrasonic sensor."""
import cv2

from sc_math import sc_clamp

CALIBRATION_FILE = "/home/mendel/SmartCarX/calibration.txt"


class Camera:
    """USB camera configured for low latency."""

    def __init__(self, index=1):
        # Resolutions tested with the original Pi cam: 640x480 (default), 1920x1088.
        # The ELP camera of the smart systems lab supports MJPEG up to 2592x1944 @ 15 fps,
        # and 30 fps up to 1280x720.
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        # A one-frame buffer keeps images up to date; MJPG gives the best frame rate
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if not self._cap.isOpened():
            print("Unable to read camera feed")
        self.width = int(self._cap.get(3))
        self.height = int(self._cap.get(4))

    def read(self):
        """Return (ok, frame) for a fresh frame."""
        self._cap.read()  # Discard the buffered frame, it may be stale
        return self._cap.read()

    def release(self):
        self._cap.release()


class Calibration:
    """Servo offsets that compensate for mounting tolerances, saved in a small text file."""

    def __init__(self, path=CALIBRATION_FILE):
        self.path = path
        self.steer = 0
        self.pan = 0
        self.tilt = 0
        self.active = False  # While active, servo keys adjust these offsets

    def load(self):
        """Read saved offsets. Returns False if none were saved yet."""
        try:
            with open(self.path, "r") as f:
                self.steer, self.pan, self.tilt = (int(f.readline()) for _ in range(3))
        except FileNotFoundError:
            return False
        return True

    def save(self):
        with open(self.path, "w+") as f:
            f.write(f"{self.steer}\n{self.pan}\n{self.tilt}\n")

    def adjust(self, axis, step):
        """Change one offset ("steer", "pan" or "tilt") and save immediately."""
        setattr(self, axis, getattr(self, axis) + step)
        self.save()


class Vehicle:
    """Drive motor plus steering and camera servos, with calibration offsets applied."""

    SERVO_LIMIT = 35
    CENTERING_STEPS = 100

    def __init__(self, px, calibration):
        self.px = px
        self.calibration = calibration
        self.speed = 0    # -100..100, negative drives backward
        self.steer = 0    # Servo angles in degrees
        self.campan = 0
        self.camtilt = 0

    def apply(self):
        """Send speed and servo angles to the hardware."""
        if self.speed < 0:
            self.px.backward(abs(self.speed))
        else:
            self.px.forward(self.speed)
        self.apply_servos()

    def apply_servos(self):
        limit = self.SERVO_LIMIT
        cal = self.calibration
        self.px.set_dir_servo_angle(sc_clamp(self.steer + cal.steer, -limit, limit))
        self.px.set_cam_pan_angle(sc_clamp(self.campan + cal.pan, -limit, limit))
        self.px.set_cam_tilt_angle(sc_clamp(self.camtilt + cal.tilt, -limit, limit))

    def center_servos(self):
        """Move all servos softly back to their center position."""
        start = (self.steer, self.campan, self.camtilt)
        steps = self.CENTERING_STEPS
        for i in range(steps):
            self.steer, self.campan, self.camtilt = (int(v - v * i / steps) for v in start)
            self.apply_servos()
        self._zero_servos()

    def shutdown(self):
        """Stop the motor and leave the servos centered."""
        self.px.stop()
        self._zero_servos()

    def _zero_servos(self):
        self.steer = self.campan = self.camtilt = 0
        self.apply_servos()


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
        # The sensor reports -1 for a signal error (ignored) and -2 for a timeout (max distance)
        if d <= -2:
            d = self.MAX_DISTANCE_CM
        if d >= 0:
            self.last_distance = self.distance
            self.distance = d

    def obstacle_ahead(self):
        """True for a close obstacle, unless the reading jumped suspiciously far."""
        return (self.distance < self.EMERGENCY_STOP_CM
                and abs(self.last_distance - self.distance) < self.MAX_TRUSTED_JUMP_CM)
