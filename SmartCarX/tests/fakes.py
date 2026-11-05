"""
Test doubles for the hardware and OpenCV modules smartcar.py imports.

The real car runs on a PiCar-X with a camera, so none of picarx, robot_hat,
periphery or cv2 is available (or useful) on a development machine.
install() registers lightweight fakes in sys.modules before smartcar is
imported. They record what the SmartCar asked for so tests can assert on it.

The fake cv2 treats test frames as already being in HSV space (cvtColor
BGR2HSV is the identity), so tests can paint lane marks directly with HSV
values and exercise the real lane-detection logic.
"""
import sys
import types
from unittest import mock

import numpy as np


class FakeCapture:
    def __init__(self, index):
        self.index = index
        self.settings = []
        self.frames = []          # queue of (ret, frame) returned by read()
        self.read_count = 0
        self.opened = True
        self.released = False

    def set(self, prop, value):
        self.settings.append((prop, value))

    def get(self, prop):
        return {3: 640.0, 4: 480.0}[prop]

    def isOpened(self):
        return self.opened

    def read(self):
        self.read_count += 1
        if self.frames:
            return self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, filename, fourcc, fps, size):
        self.filename = filename
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.written = []
        self.released = False

    def write(self, frame):
        self.written.append(frame)

    def release(self):
        self.released = True


def _make_cv2():
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2HSV = "BGR2HSV"
    cv2.COLOR_GRAY2BGR = "GRAY2BGR"
    cv2.MORPH_OPEN = "OPEN"
    cv2.MORPH_CLOSE = "CLOSE"
    cv2.CAP_PROP_FPS = 5
    cv2.CAP_PROP_BUFFERSIZE = 38
    cv2.CAP_PROP_FOURCC = 6
    cv2.FONT_HERSHEY_SIMPLEX = 0

    cv2.draw_calls = []       # ("line"|"circle"|"putText", image, *args)
    cv2.shown = []            # (window name, frame) passed to imshow
    cv2.wait_key_result = -1
    cv2.captures = []
    cv2.writers = []
    cv2.destroyed = 0

    def VideoCapture(index):
        cap = FakeCapture(index)
        cv2.captures.append(cap)
        return cap

    def VideoWriter(*args):
        writer = FakeWriter(*args)
        cv2.writers.append(writer)
        return writer

    def VideoWriter_fourcc(*chars):
        return "".join(chars)

    def cvtColor(image, code):
        if code == cv2.COLOR_GRAY2BGR:
            return np.stack([image] * 3, axis=2)
        return image.copy()

    def inRange(image, lower, upper):
        inside = np.all((image >= lower) & (image <= upper), axis=2)
        return inside.astype(np.uint8) * 255

    def morphologyEx(image, op, kernel):
        return image.copy()

    def line(image, *args):
        cv2.draw_calls.append(("line", image) + args)

    def circle(image, *args):
        cv2.draw_calls.append(("circle", image) + args)

    def putText(image, *args):
        cv2.draw_calls.append(("putText", image) + args)

    def imshow(name, frame):
        cv2.shown.append((name, frame))

    def waitKey(delay):
        return cv2.wait_key_result

    def imencode(ext, frame):
        return True, np.array([1, 2, 3], dtype=np.uint8)

    def destroyAllWindows():
        cv2.destroyed += 1

    for fn in (VideoCapture, VideoWriter, VideoWriter_fourcc, cvtColor, inRange,
               morphologyEx, line, circle, putText, imshow, waitKey, imencode,
               destroyAllWindows):
        setattr(cv2, fn.__name__, fn)
    return cv2


class FakePicarx:
    instances = []

    def __init__(self):
        self.calls = []
        self.ultrasonic = mock.Mock()
        self.ultrasonic.read.return_value = 100.0
        FakePicarx.instances.append(self)

    def _record(name):
        def method(self, *args):
            self.calls.append((name,) + args)
        return method

    set_dir_servo_angle = _record("steer")
    set_cam_pan_angle = _record("pan")
    set_cam_tilt_angle = _record("tilt")
    forward = _record("forward")
    backward = _record("backward")
    stop = _record("stop")

    def servo_calls(self, name):
        return [c[1] for c in self.calls if c[0] == name]


def install():
    """Register fakes in sys.modules and return the fake cv2 module."""
    cv2 = _make_cv2()
    picarx = types.ModuleType("picarx")
    picarx.Picarx = FakePicarx
    robot_hat = types.ModuleType("robot_hat")
    robot_hat.ADC = object
    periphery = types.ModuleType("periphery")
    periphery.GPIO = object
    sys.modules.update(cv2=cv2, picarx=picarx, robot_hat=robot_hat, periphery=periphery)
    return cv2


def reset(cv2):
    cv2.draw_calls.clear()
    cv2.shown.clear()
    cv2.captures.clear()
    cv2.writers.clear()
    cv2.wait_key_result = -1
    cv2.destroyed = 0
    FakePicarx.instances.clear()
