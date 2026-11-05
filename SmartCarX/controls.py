"""Keyboard controls: which ones are enabled, what each key does, and the matching help text."""
from dataclasses import dataclass

from hardware import Vehicle
from overlays import SC_DEBUG_MASK, SC_DEBUG_NONE, SC_DEBUG_OVERLAY
from sc_math import sc_clamp

DRIVE_KEYS = {"w": +1, "s": -1}

# Servo keys: key -> (Vehicle attribute, Calibration axis, direction)
STEER_KEYS = {
    "a": ("steer", "steer", -1),
    "d": ("steer", "steer", +1),
}
CAMERA_KEYS = {
    "f": ("campan", "pan", -1),
    "h": ("campan", "pan", +1),
    "g": ("camtilt", "tilt", -1),
    "t": ("camtilt", "tilt", +1),
}

# Threshold tuning keys: key -> (LaneDetector attribute, HSV channel, step)
THRESHOLD_KEYS = {
    "7": ("upper_threshold", 0, +1), "u": ("upper_threshold", 0, -1),
    "8": ("upper_threshold", 1, +1), "i": ("upper_threshold", 1, -1),
    "9": ("upper_threshold", 2, +1), "o": ("upper_threshold", 2, -1),
    "j": ("lower_threshold", 0, +1), "m": ("lower_threshold", 0, -1),
    "k": ("lower_threshold", 1, +1), ",": ("lower_threshold", 1, -1),
    "l": ("lower_threshold", 2, +1), ".": ("lower_threshold", 2, -1),
}

DEBUG_IMAGE_CYCLE = {
    SC_DEBUG_NONE: SC_DEBUG_MASK,
    SC_DEBUG_MASK: SC_DEBUG_OVERLAY,
    SC_DEBUG_OVERLAY: SC_DEBUG_NONE,
}

SPEED_STEP = 10
SPEED_LIMIT = 100
SERVO_STEP = 5


@dataclass
class Controls:
    """Which manual controls (and the ultrasonic sensor) are enabled."""
    drive: bool = False
    steer: bool = False
    camera: bool = False
    recording: bool = False
    ultrasonic: bool = False
    tune_thresholds: bool = False

    @property
    def any_actuator(self):
        return self.drive or self.steer or self.camera

    def handle_key(self, car, key):
        """Apply one key press to the car."""
        if key == "q":
            car.quit = True
        if self.drive and key in DRIVE_KEYS:
            car.speed = sc_clamp(car.speed + DRIVE_KEYS[key] * SPEED_STEP, -SPEED_LIMIT, SPEED_LIMIT)
            print("speed:", car.speed)
        if self.steer and key in STEER_KEYS:
            nudge_servo(car, *STEER_KEYS[key])
        if self.camera and key in CAMERA_KEYS:
            nudge_servo(car, *CAMERA_KEYS[key])
        if (self.steer or self.camera) and key == "c":
            car.calibration.active = not car.calibration.active
            if car.calibration.active:
                car.vehicle.center_servos()
        if self.any_actuator and key == "x":
            car.speed = 0
            car.vehicle.center_servos()
        if key == "?":
            car.showhelp = not car.showhelp
        if self.recording and key == "r":
            car.recorder.toggle()
        if self.tune_thresholds and key in THRESHOLD_KEYS:
            threshold, channel, step = THRESHOLD_KEYS[key]
            getattr(car.detector, threshold)[channel] += step
            print_thresholds(car.detector)
        if key == "y":  # Secret key to cycle through the debug images
            car.debug_image = DEBUG_IMAGE_CYCLE[car.debug_image]

    def help_lines(self, distance):
        """Help text listing only the controls that are currently enabled."""
        lines = ["Usage of the car control:", "show or hide this help: ?"]
        if self.any_actuator:
            lines.append("reset actuators: x")
        lines.append("quit program: q")
        if self.drive:
            lines += ["speed up: w", "speed down: s"]
        if self.steer:
            lines += ["left/right: a/d", "servo calibration: c"]
        if self.camera:
            lines += ["camera up: t", "camera left/right: f/h", "camera down: g"]
            if not self.steer:
                lines.append("servo calibration: c")
        if self.recording:
            lines.append("start or stop recording: r")
        if self.ultrasonic:
            lines.append(f"us distance: {distance:.1f}cm")
        return lines


def nudge_servo(car, servo, axis, direction):
    """Move a servo one step, or adjust its calibration offset while calibrating."""
    if car.calibration.active:
        car.calibration.adjust(axis, direction)
    else:
        position = getattr(car.vehicle, servo) + direction * SERVO_STEP
        setattr(car.vehicle, servo, sc_clamp(position, -Vehicle.SERVO_LIMIT, Vehicle.SERVO_LIMIT))
    print(f"{servo}:", getattr(car.vehicle, servo))


def print_threshold_help(detector):
    print("You can use these keys to tune the color thresholds:")
    print("  Upper HSV threshold +/-")
    print("  7    8    9")
    print("  u    i    o")
    print("  Lower HSV threshold +/-")
    print("  j    k    l")
    print("  m    ,    .")
    print("Current color tresholds")
    print_thresholds(detector)


def print_thresholds(detector):
    upper, lower = detector.upper_threshold, detector.lower_threshold
    print(f"{upper[0]} {upper[1]} {upper[2]}")
    print(f"{lower[0]} {lower[1]} {lower[2]}")
