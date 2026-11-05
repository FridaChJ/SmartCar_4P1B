"""
Characterization tests for SmartCar and its components.

These were written against the original single-class smartcar.py and pin
down its behaviour (including a few quirks). When the class was split into
components, only the attribute paths and patch targets changed; every
scenario and expected value is unchanged.

Run from the SmartCarX directory with either:

    python -m unittest discover tests
    python -m pytest tests
"""
import gc
import io
import os
import pickle
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests import fakes  # noqa: E402

cv2 = fakes.install()
import displays  # noqa: E402
import overlays  # noqa: E402
import perf  # noqa: E402
import recorder  # noqa: E402
import smartcar  # noqa: E402
from lane import LaneResult  # noqa: E402

CALIBRATION_PATH = "/home/mendel/SmartCarX/calibration.txt"
LANE_HSV = (100, 100, 150)    # inside the default HSV thresholds
HEIGHT, WIDTH = 480, 640
FIXED_LOCALTIME = (2026, 1, 2, 3, 4, 5, 0, 1, 0)


def blank_frame():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def stripes(*starts, width=20):
    """Frame with vertical lane stripes; a stripe at x has its mark at x + width/2."""
    frame = blank_frame()
    for x in starts:
        frame[:, x:x + width] = LANE_HSV
    return frame


def slanted_stripes(left, right, slope, width=20):
    """Stripes that drift right by `slope` px per row going up the image (a right curve)."""
    frame = blank_frame()
    for y in range(HEIGHT):
        shift = int((HEIGHT * 0.8 - y) * slope)
        for x in (left + shift, right + shift):
            frame[y, max(0, x):max(0, x + width)] = LANE_HSV
    return frame


def key(car, char):
    """Feed one keypress through a local window into user_command."""
    car.display = displays.LocalWindow()
    cv2.wait_key_result = ord(char)
    with redirect_stdout(io.StringIO()) as out:
        car.user_command()
    return out.getvalue()


def stream_window(conn, *more_conns):
    """StreamWindow on a mock server socket whose accept() hands out the given connections."""
    server = mock.Mock()
    server.accept.side_effect = [(c, ("10.0.0.%d" % (i + 2), 4242)) for i, c in enumerate((conn,) + more_conns)]
    with redirect_stdout(io.StringIO()):
        return displays.StreamWindow(server)


def set_path(obj, dotted, value):
    """set_path(car, "detector.last_left_x", 10) sets car.detector.last_left_x."""
    *parents, name = dotted.split(".")
    for parent in parents:
        obj = getattr(obj, parent)
    setattr(obj, name, value)


class SmartCarTestCase(unittest.TestCase):
    def setUp(self):
        fakes.reset(cv2)
        self.cars = []

    def tearDown(self):
        # Keep destructors (run at GC time) from writing CSV files into the repo
        # and from printing "Destructor called" all over the test output.
        with redirect_stdout(io.StringIO()):
            for car in self.cars:
                car.timer.enabled = False
            car = None
            self.cars.clear()
            gc.collect()

    def make_car(self, **kwargs):
        kwargs.setdefault("use_window", smartcar.SC_USE_NO_WINDOW)
        with redirect_stdout(io.StringIO()):
            car = smartcar.SmartCar(**kwargs)
        self.cars.append(car)
        return car

    @property
    def px(self):
        return fakes.FakePicarx.instances[-1]

    def texts(self):
        return [c[2] for c in cv2.draw_calls if c[0] == "putText"]


class ConstructionTests(SmartCarTestCase):
    def test_defaults(self):
        car = self.make_car()
        self.assertEqual((car.speed, car.steer, car.campan, car.camtilt), (0, 0, 0, 0))
        self.assertEqual((car.sensor.distance, car.sensor.last_distance), (255, 255))
        self.assertTrue(car.showhelp)
        self.assertFalse(car.recorder.recording or car.calibration.active or car.quit)
        self.assertEqual(list(car.detector.upper_threshold), [116, 219, 255])
        self.assertEqual(list(car.detector.lower_threshold), [96, 21, 81])
        self.assertEqual(car.debug_image, smartcar.SC_DEBUG_NONE)
        self.assertEqual(car.lane.path_mode, "search")
        self.assertIsNone(car.lane.target_x)
        self.assertIsNone(car.recorder.out)
        self.assertEqual((car.camera.width, car.camera.height), (640, 480))
        self.assertIsInstance(car.display, displays.NoWindow)

    def test_default_thresholds_are_not_shared_between_cars(self):
        first, second = self.make_car(), self.make_car()
        first.detector.upper_threshold[0] += 1
        self.assertEqual(second.detector.upper_threshold[0], 116)

    def test_camera_is_configured_for_fast_mjpg_capture(self):
        self.make_car()
        cap = cv2.captures[-1]
        self.assertEqual(cap.index, 1)
        self.assertEqual(cap.settings, [
            (cv2.CAP_PROP_FPS, 30),
            (cv2.CAP_PROP_BUFFERSIZE, 1.0),
            (cv2.CAP_PROP_FOURCC, "MJPG"),
        ])

    def test_warns_when_camera_cannot_be_opened(self):
        with mock.patch.object(fakes.FakeCapture, "isOpened", return_value=False):
            with redirect_stdout(io.StringIO()) as out:
                self.cars.append(smartcar.SmartCar(use_window=smartcar.SC_USE_NO_WINDOW))
        self.assertIn("Unable to read camera feed", out.getvalue())

    def test_reads_calibration_file_and_applies_it(self):
        with mock.patch("builtins.open", mock.mock_open(read_data="8\n10\n6\n")) as opened:
            car = self.make_car()
        opened.assert_called_once_with(CALIBRATION_PATH, "r")
        cal = car.calibration
        self.assertEqual((cal.steer, cal.pan, cal.tilt), (8, 10, 6))
        self.assertEqual(self.px.calls, [("steer", 8), ("pan", 10), ("tilt", 6)])

    def test_missing_calibration_file_is_ignored(self):
        with mock.patch("builtins.open", side_effect=FileNotFoundError):
            car = self.make_car()
        cal = car.calibration
        self.assertEqual((cal.steer, cal.pan, cal.tilt), (0, 0, 0))
        self.assertEqual(self.px.calls, [])

    def test_tune_thresholds_prints_key_help_and_current_values(self):
        with redirect_stdout(io.StringIO()) as out:
            self.cars.append(smartcar.SmartCar(tune_thresholds=True,
                                               use_window=smartcar.SC_USE_NO_WINDOW))
        text = out.getvalue()
        self.assertIn("You can use these keys to tune the color thresholds:", text)
        self.assertIn("  7    8    9", text)
        self.assertIn("116 219 255\n96 21 81\n", text)

    def test_stream_window_waits_for_a_client(self):
        server = mock.Mock()
        conn = mock.Mock()
        server.accept.return_value = (conn, ("10.0.0.2", 4242))
        with mock.patch.object(displays.socket, "socket", return_value=server):
            car = self.make_car(use_window=smartcar.SC_USE_STREAM_WINDOW)
        server.bind.assert_called_once_with(("0.0.0.0", 5001))
        server.listen.assert_called_once_with(5)
        self.assertIs(car.display.conn, conn)
        self.assertEqual(car.display.addr, ("10.0.0.2", 4242))

    def test_local_window_option(self):
        car = self.make_car(use_window=smartcar.SC_USE_LOCAL_WINDOW)
        self.assertIsInstance(car.display, displays.LocalWindow)


class KeyboardTests(SmartCarTestCase):
    def test_no_window_means_no_key_input(self):
        car = self.make_car(drive_control=True)
        cv2.wait_key_result = ord("w")
        car.user_command()
        self.assertEqual(car.speed, 0)

    def test_local_window_masks_key_code(self):
        car = self.make_car()
        car.display = displays.LocalWindow()
        cv2.wait_key_result = 0x100 | ord("q")
        car.user_command()
        self.assertTrue(car.quit)

    def test_stream_window_reads_key_from_socket(self):
        car = self.make_car(drive_control=True)
        conn = mock.Mock()
        conn.recv.return_value = b"w"
        car.display = stream_window(conn)
        with mock.patch.object(displays.select, "select", return_value=([conn], [], [])) as sel:
            with redirect_stdout(io.StringIO()) as out:
                car.user_command()
        sel.assert_called_once_with([conn], [], [], 0.1)
        conn.recv.assert_called_once_with(1)
        self.assertIn("Received key: w", out.getvalue())
        self.assertEqual(car.speed, 10)

    def test_stream_window_ignores_socket_errors_and_idle_socket(self):
        car = self.make_car(drive_control=True)
        conn = mock.Mock()
        conn.recv.side_effect = OSError
        car.display = stream_window(conn)
        with mock.patch.object(displays.select, "select", return_value=([conn], [], [])):
            car.user_command()
        with mock.patch.object(displays.select, "select", return_value=([], [], [])):
            car.user_command()
        self.assertEqual(car.speed, 0)

    def test_quit(self):
        car = self.make_car()
        key(car, "q")
        self.assertTrue(car.quit)

    def test_speed_steps_by_ten_and_clamps(self):
        car = self.make_car(drive_control=True)
        self.assertEqual(key(car, "w"), "speed: 10\n")
        for _ in range(15):
            key(car, "w")
        self.assertEqual(car.speed, 100)
        for _ in range(25):
            key(car, "s")
        self.assertEqual(car.speed, -100)

    def test_speed_keys_need_drive_control(self):
        car = self.make_car()
        key(car, "w")
        self.assertEqual(car.speed, 0)

    def test_steering_steps_by_five_and_clamps(self):
        car = self.make_car(steer_control=True)
        self.assertEqual(key(car, "d"), "steer: 5\n")
        for _ in range(10):
            key(car, "d")
        self.assertEqual(car.steer, 35)
        for _ in range(20):
            key(car, "a")
        self.assertEqual(car.steer, -35)

    def test_camera_keys_step_by_five_and_clamp(self):
        car = self.make_car(camera_control=True)
        self.assertEqual(key(car, "h"), "campan: 5\n")
        self.assertEqual(key(car, "f"), "campan: 0\n")
        self.assertEqual(key(car, "t"), "camtilt: 5\n")
        self.assertEqual(key(car, "g"), "camtilt: 0\n")
        for _ in range(10):
            key(car, "h")
            key(car, "g")
        self.assertEqual((car.campan, car.camtilt), (35, -35))
        for _ in range(20):
            key(car, "f")
            key(car, "t")
        self.assertEqual((car.campan, car.camtilt), (-35, 35))

    def test_servo_keys_need_their_control(self):
        car = self.make_car()
        for char in "adhftg":
            key(car, char)
        self.assertEqual((car.steer, car.campan, car.camtilt), (0, 0, 0))

    def test_c_toggles_calibration_once_and_centers(self):
        car = self.make_car(steer_control=True, camera_control=True)
        car.steer = 10
        key(car, "c")
        self.assertTrue(car.calibration.active)
        self.assertEqual(car.steer, 0)
        key(car, "c")
        self.assertFalse(car.calibration.active)

    def test_c_works_with_only_camera_control(self):
        car = self.make_car(camera_control=True)
        key(car, "c")
        self.assertTrue(car.calibration.active)

    def test_calibration_keys_adjust_offsets_and_save_them(self):
        car = self.make_car(steer_control=True, camera_control=True)
        car.calibration.active = True
        with mock.patch("builtins.open", mock.mock_open()) as opened:
            self.assertEqual(key(car, "a"), "steer: 0\n")
            key(car, "h")
            key(car, "h")
            key(car, "g")
        cal = car.calibration
        self.assertEqual((cal.steer, cal.pan, cal.tilt), (-1, 2, -1))
        self.assertEqual((car.steer, car.campan, car.camtilt), (0, 0, 0))
        opened.assert_called_with(CALIBRATION_PATH, "w+")
        self.assertEqual(opened.call_count, 4)
        written = "".join(c.args[0] for c in opened().write.call_args_list)
        self.assertTrue(written.endswith("0\n-1\n2\n-1\n"), written)

    def test_x_resets_speed_and_servos(self):
        car = self.make_car(drive_control=True)
        car.speed, car.steer, car.campan, car.camtilt = 50, 20, -10, 15
        key(car, "x")
        self.assertEqual((car.speed, car.steer, car.campan, car.camtilt), (0, 0, 0, 0))

    def test_x_needs_some_control(self):
        car = self.make_car()
        car.speed = 50
        key(car, "x")
        self.assertEqual(car.speed, 50)

    def test_question_mark_toggles_help(self):
        car = self.make_car()
        key(car, "?")
        self.assertFalse(car.showhelp)
        key(car, "?")
        self.assertTrue(car.showhelp)

    def test_r_starts_and_stops_recording(self):
        car = self.make_car(recording_control=True)
        with mock.patch.object(recorder.time, "time", return_value=0.0), \
             mock.patch.object(recorder.time, "localtime", return_value=FIXED_LOCALTIME):
            key(car, "r")
        self.assertTrue(car.recorder.recording)
        writer = cv2.writers[-1]
        self.assertIs(car.recorder.out, writer)
        self.assertEqual(
            (writer.filename, writer.fourcc, writer.fps, writer.size),
            ("2026-01-02-03-04-05.avi", "MJPG", 10, (640, 480)))
        key(car, "r")
        self.assertFalse(car.recorder.recording)
        self.assertTrue(writer.released)

    def test_r_needs_recording_control(self):
        car = self.make_car()
        key(car, "r")
        self.assertFalse(car.recorder.recording)

    def test_threshold_tuning_keys(self):
        cases = {
            "7": ("upper", 0, +1), "u": ("upper", 0, -1),
            "8": ("upper", 1, +1), "i": ("upper", 1, -1),
            "9": ("upper", 2, +1), "o": ("upper", 2, -1),
            "j": ("lower", 0, +1), "m": ("lower", 0, -1),
            "k": ("lower", 1, +1), ",": ("lower", 1, -1),
            "l": ("lower", 2, +1), ".": ("lower", 2, -1),
        }
        for char, (which, channel, delta) in cases.items():
            with self.subTest(key=char):
                car = self.make_car(tune_thresholds=True,
                                    upper_threshold=np.array([110, 200, 240]),
                                    lower_threshold=np.array([90, 30, 80]))
                before = getattr(car.detector, which + "_threshold").copy()
                out = key(car, char)
                expected = before.copy()
                expected[channel] += delta
                self.assertEqual(list(getattr(car.detector, which + "_threshold")), list(expected))
                up, lo = car.detector.upper_threshold, car.detector.lower_threshold
                self.assertEqual(out, f"{up[0]} {up[1]} {up[2]}\n{lo[0]} {lo[1]} {lo[2]}\n")

    def test_threshold_keys_need_tuning_mode(self):
        car = self.make_car()
        key(car, "7")
        self.assertEqual(list(car.detector.upper_threshold), [116, 219, 255])

    def test_y_cycles_debug_image(self):
        car = self.make_car()
        seen = []
        for _ in range(3):
            key(car, "y")
            seen.append(car.debug_image)
        self.assertEqual(seen, [smartcar.SC_DEBUG_MASK, smartcar.SC_DEBUG_OVERLAY,
                                smartcar.SC_DEBUG_NONE])


class ServoTests(SmartCarTestCase):
    def test_apply_servos_adds_calibration_and_clamps(self):
        car = self.make_car()
        car.steer, car.campan, car.camtilt = 30, -30, 5
        car.calibration.steer, car.calibration.pan, car.calibration.tilt = 10, -10, 3
        car.vehicle.apply_servos()
        self.assertEqual(self.px.calls, [("steer", 35), ("pan", -35), ("tilt", 8)])

    def test_center_servos_ramps_to_zero(self):
        car = self.make_car()
        car.steer, car.campan, car.camtilt = 20, -10, 7
        car.vehicle.center_servos()
        expected = [int(20 - 20 * i / 100) for i in range(100)] + [0]
        self.assertEqual(self.px.servo_calls("steer"), expected)
        self.assertEqual(self.px.servo_calls("pan")[-1], 0)
        self.assertEqual(self.px.servo_calls("tilt")[0], 7)
        self.assertEqual((car.steer, car.campan, car.camtilt), (0, 0, 0))


class SensorAndActuatorTests(SmartCarTestCase):
    def test_handle_sensors_flushes_one_frame_and_keeps_the_next(self):
        car = self.make_car()
        stale, fresh = blank_frame(), blank_frame()
        cv2.captures[-1].frames = [(True, stale), (True, fresh)]
        car.handle_sensors()
        self.assertTrue(car.ret)
        self.assertIs(car.frame, fresh)
        self.px.ultrasonic.read.assert_not_called()

    def test_ultrasonic_measurement(self):
        car = self.make_car(use_ultrasonic=True)
        self.px.ultrasonic.read.return_value = 42.4567
        car.handle_sensors()
        self.assertEqual((car.sensor.distance, car.sensor.last_distance), (42.46, 255))

    def test_ultrasonic_error_is_ignored_and_timeout_means_max(self):
        car = self.make_car()
        self.px.ultrasonic.read.return_value = -1
        car.sensor.measure()
        self.assertEqual((car.sensor.distance, car.sensor.last_distance), (255, 255))
        car.sensor.distance = 50
        self.px.ultrasonic.read.return_value = -2
        car.sensor.measure()
        self.assertEqual((car.sensor.distance, car.sensor.last_distance), (255, 50))

    def test_drives_forward_or_backward(self):
        car = self.make_car()
        car.speed = 30
        car.handle_actuators()
        car.speed = -20
        car.handle_actuators()
        drive = [c for c in self.px.calls if c[0] in ("forward", "backward")]
        self.assertEqual(drive, [("forward", 30), ("backward", 20)])
        self.assertEqual(self.px.servo_calls("steer"), [0, 0])

    def test_emergency_stop_only_for_trusted_close_readings(self):
        car = self.make_car(use_ultrasonic=True)
        car.speed = 30
        car.sensor.distance, car.sensor.last_distance = 10, 255   # jump > 100cm: not trusted
        car.handle_actuators()
        self.assertEqual(car.speed, 30)
        car.sensor.distance, car.sensor.last_distance = 10, 15
        with redirect_stdout(io.StringIO()) as out:
            car.handle_actuators()
        self.assertEqual(car.speed, 0)
        self.assertEqual(out.getvalue(), "stopped because d = 10\nLast distance was d = 15\n")
        self.assertEqual(self.px.calls[-4], ("forward", 0))

    def test_no_emergency_stop_when_reversing_or_sensor_off(self):
        car = self.make_car(use_ultrasonic=True)
        car.sensor.distance, car.sensor.last_distance = 10, 15
        car.speed = -30
        car.handle_actuators()
        self.assertEqual(car.speed, -30)
        car.speed = 0
        with redirect_stdout(io.StringIO()) as out:
            car.handle_actuators()
        self.assertEqual(out.getvalue(), "")
        car.controls.ultrasonic = False
        car.speed = 30
        car.handle_actuators()
        self.assertEqual(car.speed, 30)

    def test_recording_writes_the_clean_frame(self):
        car = self.make_car()
        car.recorder.recording = True
        car.recorder.out = fakes.FakeWriter("x.avi", "MJPG", 10, (640, 480))
        car.recordingframe = blank_frame()
        car.handle_actuators()
        self.assertEqual(len(car.recorder.out.written), 1)
        self.assertIs(car.recorder.out.written[0], car.recordingframe)


class WindowTests(SmartCarTestCase):
    def window_car(self, display=None, **kwargs):
        car = self.make_car(**kwargs)
        car.display = display or displays.LocalWindow()
        car.ret, car.frame = True, blank_frame()
        return car

    def test_nothing_happens_without_a_frame(self):
        car = self.window_car()
        car.ret = False
        car.handle_window("hello")
        self.assertEqual(cv2.draw_calls, [])
        self.assertEqual(cv2.shown, [])

    def test_local_window_shows_frame_with_help(self):
        car = self.window_car()
        car.handle_window()
        self.assertEqual(cv2.shown, [("SmartCar camera", car.frame)])
        self.assertEqual(self.texts(), ["Usage of the car control:",
                                        "show or hide this help: ?",
                                        "quit program: q"])
        first = cv2.draw_calls[0]
        self.assertIs(first[1], car.frame)
        self.assertEqual(first[3:], ((10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 0), 2))
        self.assertEqual(cv2.draw_calls[2][3], (10, 60))

    def test_recording_frame_is_a_copy_taken_before_overlays(self):
        car = self.window_car()
        car.handle_window("extra")
        self.assertIsNot(car.recordingframe, car.frame)
        self.assertTrue(np.array_equal(car.recordingframe, car.frame))

    def test_full_help_text(self):
        car = self.window_car(drive_control=True, steer_control=True, camera_control=True,
                              recording_control=True, use_ultrasonic=True)
        car.sensor.distance = 12.345
        car.handle_window()
        self.assertEqual(self.texts(), [
            "Usage of the car control:",
            "show or hide this help: ?",
            "reset actuators: x",
            "quit program: q",
            "speed up: w",
            "speed down: s",
            "left/right: a/d",
            "servo calibration: c",
            "camera up: t",
            "camera left/right: f/h",
            "camera down: g",
            "start or stop recording: r",
            "us distance: 12.3cm",
        ])

    def test_camera_only_help_mentions_calibration(self):
        car = self.window_car(camera_control=True)
        car.handle_window()
        self.assertEqual(self.texts()[-4:], ["camera up: t", "camera left/right: f/h",
                                             "camera down: g", "servo calibration: c"])

    def test_help_hidden_or_without_window(self):
        car = self.window_car()
        car.showhelp = False
        car.handle_window()
        car.showhelp = True
        car.display = displays.NoWindow()
        car.handle_window()
        self.assertEqual(self.texts(), [])
        self.assertEqual(len(cv2.shown), 1)

    def test_additional_text(self):
        car = self.window_car()
        car.showhelp = False
        car.handle_window("lane ok")
        self.assertEqual(cv2.draw_calls[0][2:], (
            "lane ok", (150, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2))

    def test_recording_text_blinks(self):
        car = self.window_car()
        car.showhelp = False
        car.recorder.recording = True
        visible = []
        for now in (100.0, 100.2, 100.7, 101.1, 101.3):
            cv2.draw_calls.clear()
            with mock.patch.object(overlays.time, "time", return_value=now):
                car.handle_window()
            visible.append(self.texts() == ["recording"])
        self.assertEqual(visible, [False, True, False, False, True])
        cv2.draw_calls.clear()
        with mock.patch.object(overlays.time, "time", return_value=101.4):
            car.handle_window()
        self.assertEqual(cv2.draw_calls[0][3:], ((10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2))

    def test_calibration_text_blinks(self):
        car = self.window_car()
        car.showhelp = False
        car.calibration.active = True
        with mock.patch.object(overlays.time, "time", return_value=100.0):
            car.handle_window()
        with mock.patch.object(overlays.time, "time", return_value=100.2):
            car.handle_window()
        self.assertEqual(self.texts(), ["calibrate servos"])
        self.assertEqual(cv2.draw_calls[0][3], (10, 420))

    def test_stream_window_sends_pickled_jpeg(self):
        conn = mock.Mock()
        car = self.window_car(display=stream_window(conn))
        car.handle_window()
        data = pickle.dumps(np.array([1, 2, 3], dtype=np.uint8))
        conn.sendall.assert_called_once_with(b"IMG" + struct.pack(">L", len(data)) + data)
        self.assertEqual(cv2.shown, [])

    def test_stream_window_reconnects_when_client_leaves(self):
        conn, new_conn = mock.Mock(), mock.Mock()
        conn.sendall.side_effect = BrokenPipeError
        car = self.window_car(display=stream_window(conn, new_conn))
        with redirect_stdout(io.StringIO()) as out:
            car.handle_window()
        self.assertIs(car.display.conn, new_conn)
        self.assertIn("Server listening on 0.0.0.0:5001", out.getvalue())
        self.assertIn("Connection from: ('10.0.0.3', 4242)", out.getvalue())


class LaneDetectionTests(SmartCarTestCase):
    def detect(self, frame, speed=0, **state):
        """Run lane_detection on `frame`; state keys are dotted paths, e.g. detector__last_left_x."""
        car = self.make_car()
        for name, value in state.items():
            set_path(car, name.replace("__", "."), value)
        car.speed = speed
        car.frame = frame
        car.lane_detection()
        return car

    def test_straight_lane_pair(self):
        car = self.detect(stripes(150, 490))
        self.assertEqual(car.lane.path_mode, "straight")
        self.assertEqual((car.lane.lane_seen, car.lane.lane_count), (True, 2))
        self.assertAlmostEqual(car.lane.target_x, 330.0)
        self.assertEqual(car.lane.curve_score_x, 0.0)
        d = car.detector
        self.assertEqual((d.last_left_x, d.last_right_x, d.last_lane_width), (160, 500, 340.0))

    def test_detection_draws_scan_rows_marks_and_target(self):
        self.detect(stripes(150, 490))
        lines = [c for c in cv2.draw_calls if c[0] == "line"]
        circles = [c for c in cv2.draw_calls if c[0] == "circle"]
        self.assertEqual([l[2] for l in lines[:5]],
                         [(0, 384), (0, 350), (0, 316), (0, 283), (0, 249)])
        self.assertEqual(len(circles), 11)          # 2 marks x 5 rows + center dot
        self.assertEqual(circles[-1][2:], ((320, 384), 5, (255, 0, 0), -1))
        self.assertEqual(lines[-2][2:], ((320, 480), (320, 220), (255, 0, 0), 1))
        self.assertEqual(lines[-1][2:], ((330, 480), (330, 220), (0, 255, 0), 2))

    def test_curve_is_detected_from_target_drift(self):
        car = self.detect(slanted_stripes(150, 490, slope=0.5))
        self.assertEqual(car.lane.path_mode, "curve")
        self.assertAlmostEqual(car.lane.curve_score_x, 67.0)
        self.assertAlmostEqual(car.lane.target_x, 355.02)

    def test_single_line_uses_default_lane_width(self):
        car = self.detect(stripes(150))
        self.assertEqual((car.lane.path_mode, car.lane.lane_count), ("search", 1))
        self.assertAlmostEqual(car.lane.target_x, 160 + 640 * 0.55 / 2)

    def test_single_line_is_classified_by_last_known_lines(self):
        car = self.detect(stripes(150), detector__last_left_x=10, detector__last_right_x=200,
                          detector__last_lane_width=300.0)
        self.assertAlmostEqual(car.lane.target_x, 10.0)     # right line: 160 - 150

    def test_single_line_target_is_clamped_to_frame(self):
        car = self.detect(stripes(600), detector__last_lane_width=300.0)
        self.assertAlmostEqual(car.lane.target_x, 460.0)
        car = self.detect(stripes(20), detector__last_left_x=500, detector__last_right_x=30,
                          detector__last_lane_width=300.0)
        self.assertEqual(car.lane.target_x, 0.0)

    def test_close_marks_count_as_two_without_a_pair(self):
        car = self.detect(stripes(150, 240))
        self.assertEqual((car.lane.path_mode, car.lane.lane_count), ("search", 2))
        self.assertAlmostEqual(car.lane.target_x, 336.0)

    def test_marks_too_narrow_or_wide_are_ignored(self):
        car = self.detect(stripes(150, 490, width=5))
        self.assertIsNone(car.lane.target_x)
        car = self.detect(stripes(150, width=120))
        self.assertIsNone(car.lane.target_x)

    def test_no_lane(self):
        car = self.detect(blank_frame(), lane=LaneResult(path_mode="curve", curve_score_x=50.0))
        lane = car.lane
        self.assertEqual((lane.lane_seen, lane.lane_count, lane.path_mode), (False, 0, "search"))
        self.assertIsNone(lane.target_x)
        self.assertEqual(lane.curve_score_x, 0.0)

    def test_widest_pair_wins(self):
        car = self.detect(stripes(50, 250, 590))
        self.assertAlmostEqual(car.lane.target_x, 330.0)
        self.assertEqual((car.detector.last_left_x, car.detector.last_right_x), (60, 600))

    def test_mask_debug_image(self):
        car = self.detect(stripes(150, 490), debug_image=smartcar.SC_DEBUG_MASK)
        self.assertEqual(car.frame.shape, (480, 640, 3))
        self.assertEqual(list(car.frame[10, 155]), [255, 255, 255])
        self.assertEqual(list(car.frame[10, 300]), [0, 0, 0])

    def test_overlay_debug_image(self):
        frame = stripes(150, 490)
        frame[10, 300] = (1, 2, 3)
        car = self.detect(frame, debug_image=smartcar.SC_DEBUG_OVERLAY)
        self.assertEqual(list(car.frame[10, 155]), [255, 255, 255])
        self.assertEqual(list(car.frame[10, 300]), [1, 2, 3])

    def test_no_steering_while_stopped(self):
        car = self.detect(stripes(230, 570), steer=7, campan=-3)
        self.assertEqual((car.steer, car.campan), (7, -3))
        self.assertEqual(car.steering.filtered_error_x, 0.0)


class LaneControllerTests(SmartCarTestCase):
    def drive(self, frame, filtered_error_x=0.0, **state):
        car = self.make_car()
        car.speed = 30
        car.steering.filtered_error_x = filtered_error_x
        for name, value in state.items():
            setattr(car, name, value)
        car.frame = frame
        car.lane_detection()
        return car

    def test_small_straight_error_is_ignored(self):
        car = self.drive(stripes(150, 490))
        self.assertAlmostEqual(car.steering.auto_error_x, 10.0)
        self.assertAlmostEqual(car.steering.filtered_error_x, 1.8)
        self.assertEqual((car.steering.auto_target_steer, car.steer, car.campan), (0, 0, 0))

    def test_deadband_returns_slowly_to_center(self):
        car = self.drive(stripes(150, 490), steer=5)
        self.assertAlmostEqual(car.steer, 4.1)
        self.assertAlmostEqual(car.campan, 4.1 * 0.95)

    def test_moderate_straight_error_steers_in_small_steps(self):
        car = self.drive(stripes(170, 510), filtered_error_x=40.0)
        self.assertAlmostEqual(car.steering.filtered_error_x, 38.2)
        self.assertAlmostEqual(car.steering.auto_target_steer, 38.2 * 0.22 * 0.38)
        self.assertAlmostEqual(car.steer, 3.0)
        self.assertAlmostEqual(car.campan, 2.85)

    def test_large_error_counts_as_curve(self):
        car = self.drive(stripes(220, 560))
        self.assertEqual(car.lane.path_mode, "straight")
        self.assertAlmostEqual(car.steering.filtered_error_x, 44.0)
        self.assertAlmostEqual(car.steering.auto_target_steer, 44.0 * 0.22 * 1.15)
        self.assertAlmostEqual(car.steer, 6.5)

    def test_curve_steering(self):
        car = self.drive(slanted_stripes(150, 490, slope=0.5), steer=-3)
        self.assertAlmostEqual(car.steering.auto_error_x, 35.02)
        self.assertAlmostEqual(car.steering.filtered_error_x, 0.55 * 35.02)
        self.assertAlmostEqual(car.steer, 3.5)
        self.assertAlmostEqual(car.campan, 3.5 * 0.95)

    def test_target_steer_is_clamped(self):
        car = self.drive(stripes(600, 600), filtered_error_x=300.0, steer=33)
        # one mark: lane_count 1 -> controller only re-couples the camera
        self.assertEqual(car.steer, 33)
        self.assertAlmostEqual(car.campan, 31.35)
        car = self.drive(stripes(400, 600), filtered_error_x=300.0, steer=33)
        self.assertEqual(car.steering.auto_target_steer, 35)
        self.assertEqual(car.steer, 35)

    def test_lost_lane_only_couples_camera(self):
        car = self.drive(blank_frame(), steer=-20, campan=0)
        self.assertEqual(car.steer, -20)
        self.assertEqual(car.campan, -19.0)
        self.assertEqual(car.lane.path_mode, "search")


class PerformanceTests(SmartCarTestCase):
    def test_timers_are_off_by_default(self):
        car = self.make_car()
        car.timer.start("x")
        car.timer.stop("x")
        self.assertEqual(car.timer.timings, {})

    def test_timers_record_elapsed_time(self):
        car = self.make_car(measure_performance=True)
        with mock.patch.object(perf.time, "time", side_effect=[10.0, 10.25, 11.0, 11.5]):
            car.timer.start("x")
            car.timer.stop("x")
            car.timer.start("x")
            car.timer.stop("x")
        self.assertEqual(car.timer.timings, {"x": [0.25, 0.5]})

    def test_stop_without_start_is_an_error_entry(self):
        car = self.make_car(measure_performance=True)
        car.timer.stop("y")
        self.assertEqual(car.timer.timings, {"y": ["error"]})

    def test_timer_names(self):
        car = self.make_car(measure_performance=True, use_ultrasonic=True)
        cv2.captures[-1].frames = [(True, blank_frame()), (True, stripes(150, 490))]
        car.handle_sensors()
        car.lane_detection()
        car.user_command()
        car.handle_actuators()
        car.ret = False
        car.handle_window()
        car.ret = True
        car.handle_window()
        self.assertEqual(list(car.timer.timings), [
            "handle_sensors.capture_image",
            "handle_sensors.measure_us_distance",
            "lane_detection.hsv",
            "lane_detection.mask",
            "lane_detection.debugimage",
            "lane_detection.detect_lanes",
            "lane_detection.controller",
            "user_command",
            "handle_actuators",
            "handle_window.add_text",
            "handle_window.copy_frame",
            "handle_window.showhelp",
            "handle_window.addrecording",
            "handle_window.imshow",
        ])
        self.assertEqual(len(car.timer.timings["handle_window.add_text"]), 1)


class ShutdownTests(SmartCarTestCase):
    def test_destructor_releases_everything_and_centers(self):
        car = self.make_car()
        car.calibration.steer = 4
        car.steer, car.campan, car.camtilt = 10, 10, 10
        car.recorder.out = fakes.FakeWriter("x.avi", "MJPG", 10, (640, 480))
        with redirect_stdout(io.StringIO()) as out:
            car.__del__()
        self.assertEqual(out.getvalue(), "Destructor called\n")
        self.assertTrue(cv2.captures[-1].released)
        self.assertTrue(car.recorder.out.released)
        self.assertEqual(cv2.destroyed, 1)
        self.assertEqual(self.px.calls, [("stop",), ("steer", 4), ("pan", 0), ("tilt", 0)])

    def test_destructor_closes_stream_connection(self):
        car = self.make_car()
        conn = mock.Mock()
        car.display = stream_window(conn)
        with redirect_stdout(io.StringIO()):
            car.__del__()
        conn.close.assert_called_once_with()

    def test_destructor_writes_time_consumption_csv(self):
        car = self.make_car(measure_performance=True)
        car.timer.timings = {"a": [0.5, 1.25], "b": [2.0]}
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with mock.patch.object(perf.time, "localtime", return_value=FIXED_LOCALTIME), \
                     redirect_stdout(io.StringIO()):
                    car.__del__()
                with open("time_consumption-2026-01-02-03-04-05.csv") as f:
                    content = f.read()
            finally:
                os.chdir(cwd)
        self.assertEqual(content, "a;b\n0,5;2,0\n1,25;end\nend;end\n")


class LoopTests(SmartCarTestCase):
    def run_loop(self, car, loop_name, iterations=2):
        calls = []

        def track(name):
            def step(*args):
                calls.append(name)
                if name == "user_command" and calls.count(name) == iterations:
                    car.quit = True
            return step
        for name in ("handle_sensors", "handle_window", "user_command",
                     "handle_actuators", "lane_detection"):
            setattr(car, name, track(name))
        getattr(car, loop_name)()
        return calls

    def controls(self, car):
        c = car.controls
        return (c.drive, c.steer, c.camera, c.recording, c.ultrasonic)

    def test_camera_window_loop(self):
        car = self.make_car(drive_control=True, steer_control=True, use_ultrasonic=True)
        calls = self.run_loop(car, "camera_window_loop")
        self.assertEqual(self.controls(car), (False, False, True, True, False))
        self.assertEqual(calls, ["handle_sensors", "handle_window", "user_command", "handle_actuators",
                                 "handle_sensors", "handle_window", "user_command"])

    def test_drive_and_record_loop(self):
        car = self.make_car()
        calls = self.run_loop(car, "drive_and_record_loop")
        self.assertEqual(self.controls(car), (True, True, True, True, True))
        self.assertEqual(calls, ["handle_sensors", "handle_window", "user_command", "handle_actuators",
                                 "handle_sensors", "handle_window", "user_command"])

    def test_lane_detection_loop(self):
        car = self.make_car(steer_control=True)
        calls = self.run_loop(car, "lane_detection_loop")
        self.assertEqual(self.controls(car), (True, False, True, True, True))
        self.assertEqual(calls, ["handle_sensors", "lane_detection", "user_command",
                                 "handle_actuators", "handle_window",
                                 "handle_sensors", "lane_detection", "user_command"])


class PublicInterfaceTests(SmartCarTestCase):
    """What the smartcar_0x scripts rely on."""

    def test_sc_clamp(self):
        self.assertEqual(smartcar.sc_clamp(5, 0, 3), 3)
        self.assertEqual(smartcar.sc_clamp(-5, 0, 3), 0)
        self.assertEqual(smartcar.sc_clamp(2, 0, 3), 2)

    def test_public_constants(self):
        self.assertEqual((smartcar.SC_DEBUG_NONE, smartcar.SC_DEBUG_MASK, smartcar.SC_DEBUG_OVERLAY),
                         (0, 1, 2))
        self.assertEqual((smartcar.SC_USE_NO_WINDOW, smartcar.SC_USE_LOCAL_WINDOW,
                          smartcar.SC_USE_STREAM_WINDOW), (0, 1, 2))

    def test_driving_state_is_shared_with_the_vehicle(self):
        car = self.make_car()
        car.speed, car.steer, car.campan, car.camtilt = 40, 5, -5, 3
        v = car.vehicle
        self.assertEqual((v.speed, v.steer, v.campan, v.camtilt), (40, 5, -5, 3))


if __name__ == "__main__":
    unittest.main()
