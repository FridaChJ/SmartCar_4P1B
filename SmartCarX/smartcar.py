"""
SmartCar class for WIP Smart Systems lecture
version 0.10:
Restructured the class for readability (no functional changes) and added unit tests
version 0.9:
Integrated new lane detection algorithm from students in summer semester 2026
version 0.8:
Adapted the class for the PiCarX platform.
Removed VCC measurement as this is done by LCD display
version 0.7:
Integrated option for streaming video feed to a client by socket connection.
version 0.6:
Video stream optimized with MJPG encoder (much faster now)
Lane detection optimized using gradient function of numpy (much faster now)
version 0.5:
Image size reduced to the rows to analyze to improve image conversion performance
version 0.4:
Time measurement functions were integrated to analyze performance of software parts
version 0.3:
Debugging functionality including modification of color thresholds was added
version 0.2:
Lane detection function was integrated from existing implementation 12.09.2022.
version 0.1:
Driving functions were integrated into the new class from existing implementation 12.09.2022.
"""
import pickle
import select
import socket
import struct
import time
from contextlib import contextmanager
from itertools import zip_longest

import cv2
import numpy as np
from picarx import Picarx

# Debug modes for the displayed video
SC_DEBUG_NONE = 0     # Plain camera image
SC_DEBUG_MASK = 1     # Black/white lane mask
SC_DEBUG_OVERLAY = 2  # Lane mask painted over the camera image

# How video and keyboard input are connected
SC_USE_NO_WINDOW = 0      # Headless, e.g. for dashboard control
SC_USE_LOCAL_WINDOW = 1   # Native OpenCV window, needs an X server on the remote computer
SC_USE_STREAM_WINDOW = 2  # Socket server that streams video to a client and receives its keys

STREAM_PORT = 5001
CALIBRATION_FILE = "/home/mendel/SmartCarX/calibration.txt"

# HSV thresholds for the blue lane tape, from students and Siavash in summer 2026.
# Earlier tunings (lower / upper):
#   original:       [99, 23, 65] / [123, 255, 255]
#   Markus' home:   [105, 43, 76] / [117, 255, 255]
#   Markus' office: [98, 79, 81] / [117, 219, 249]
DEFAULT_LOWER_THRESHOLD = (96, 21, 81)
DEFAULT_UPPER_THRESHOLD = (116, 219, 255)

# Manual servo keys: key -> (position attribute, calibration attribute, direction)
STEER_KEYS = {
    "a": ("steer", "cal_steer", -1),
    "d": ("steer", "cal_steer", +1),
}
CAMERA_KEYS = {
    "f": ("campan", "cal_pan", -1),
    "h": ("campan", "cal_pan", +1),
    "g": ("camtilt", "cal_tilt", -1),
    "t": ("camtilt", "cal_tilt", +1),
}
DRIVE_KEYS = {"w": +1, "s": -1}

# Threshold tuning keys: key -> (threshold attribute, HSV channel, step)
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

# Text overlay style (colors are BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX
HELP_COLOR = (180, 0, 0)
STATUS_COLOR = (0, 0, 255)
INFO_COLOR = (255, 100, 100)


def sc_clamp(value, low, high):
    """Clamp a numeric value into a closed range."""
    return max(low, min(high, value))


class SmartCar(object):
    # Manual control
    SPEED_STEP = 10
    SPEED_LIMIT = 100
    SERVO_STEP = 5
    SERVO_LIMIT = 35
    CENTERING_STEPS = 100
    EMERGENCY_STOP_CM = 20
    MAX_TRUSTED_DISTANCE_JUMP_CM = 100

    # Lane detection: scan rows as fractions of the image height, from the
    # bottom (close to the car) to the top (looking ahead)
    SC_SCAN_ROWS = (0.80, 0.73, 0.66, 0.59, 0.52)
    SC_MIN_PAIR_GAP_FRACTION = 0.28
    SC_DEFAULT_LANE_WIDTH_FRACTION = 0.55

    # Adaptive lane controller (Gruppe 3/8)
    SC_CURVE_SCORE_PX = 28.0
    SC_CURVE_ERROR_PX = 70.0
    SC_STRAIGHT_FILTER_ALPHA = 0.82
    SC_CURVE_FILTER_ALPHA = 0.45
    SC_STRAIGHT_STEP_LIMIT = 0.9
    SC_CURVE_STEP_LIMIT = 6.5
    SC_MID_STEP_LIMIT = 3.0
    SC_STRAIGHT_GAIN_FACTOR = 0.38
    SC_CURVE_GAIN_FACTOR = 1.15
    SC_AUTO_STEER_GAIN = 0.22
    SC_AUTO_MAX_STEER_DELTA = 40.0
    SC_AUTO_TARGET_DEADBAND_PX = 14.0
    SC_STEER_MIN = -35
    SC_STEER_CENTER = 0
    SC_STEER_MAX = 35
    SC_PAN_COUPLING_GAIN = 0.95
    SC_PAN_MIN = -35
    SC_PAN_CENTER = 0
    SC_PAN_MAX = 35

    def __init__(self, drive_control=False,
                 steer_control=False,
                 camera_control=False,
                 recording_control=False,
                 use_ultrasonic=False,
                 upper_threshold=None,
                 lower_threshold=None,
                 debug_image=SC_DEBUG_NONE,
                 tune_thresholds=False,
                 measure_performance=False,
                 use_window=SC_USE_STREAM_WINDOW):
        self.px = Picarx()

        # Actuator state
        self.speed = 0
        self.steer = 0
        self.campan = 0
        self.camtilt = 0
        self.cal_steer = 0  # Servo calibration offsets, persisted in CALIBRATION_FILE
        self.cal_pan = 0
        self.cal_tilt = 0

        # Which keyboard controls are available
        self.drive_control = drive_control
        self.steer_control = steer_control
        self.camera_control = camera_control
        self.recording_control = recording_control
        self.use_ultrasonic = use_ultrasonic

        # User interface state
        self.showhelp = True
        self.recording = False
        self.calibration = False
        self.quit = False
        self.recactivated = 0  # Time when the blinking "recording" text was last shown
        self.calactivated = 0  # Time when the blinking "calibrate servos" text was last shown
        self.use_window = use_window
        self.conn = None

        # Ultrasonic distance in cm; the previous value is kept to reject outliers
        self.distance = 255
        self.last_distance = 255

        # Lane mask color range (HSV) and debugging options
        self.upper_threshold = np.array(DEFAULT_UPPER_THRESHOLD) if upper_threshold is None else upper_threshold
        self.lower_threshold = np.array(DEFAULT_LOWER_THRESHOLD) if lower_threshold is None else lower_threshold
        self.debug_image = debug_image
        self.tune_thresholds = tune_thresholds
        self.measure_performance = measure_performance
        self.time_consumptions = {}  # Block name -> list of measured durations in s

        # Lane controller state (Gruppe 3/8 multi-row adaptive controller)
        self.lane_seen = False        # True while a lane target was found this frame
        self.lane_count = 0           # 2 = pair found, 1 = single mark, 0 = nothing
        self.lane_target_x = None     # Current steering target in pixels, or None
        self.path_mode = "search"     # "straight" | "curve" | "search"
        self.curve_score_x = 0.0      # Spread of the target between scan rows
        self.last_left_x = None       # Last known lane marks, to classify a single visible line
        self.last_right_x = None
        self.last_lane_width = None   # Last known lane width in pixels
        self.filtered_error_x = 0.0   # EMA-filtered pixel error used by the controller
        self.auto_error_x = 0.0       # Raw pixel error, for debugging/telemetry
        self.auto_target_steer = 0    # Steering angle the controller is moving toward

        if tune_thresholds:
            self._print_threshold_help()
        self._open_camera()
        self.out = None  # Video writer while recording
        self._read_calibfile()
        if use_window == SC_USE_STREAM_WINDOW:
            self._start_stream_server()

    def __del__(self):
        print("Destructor called")
        if self.measure_performance:
            self._write_time_consumptions()
        self.cap.release()
        if self.use_window == SC_USE_STREAM_WINDOW and self.conn:
            self.conn.close()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()
        # Leave the hardware stopped and centered
        self.px.stop()
        self.steer = self.campan = self.camtilt = 0
        self.handle_servos()

    # ------------------------------------------------------------------
    # Setup and persistence
    # ------------------------------------------------------------------

    def _open_camera(self):
        # Resolutions tested with the original Pi cam: 640x480 (default), 1920x1088.
        # The ELP camera of the smart systems lab supports MJPEG up to 2592x1944 @ 15 fps,
        # and 30 fps up to 1280x720.
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # A one-frame buffer keeps images up to date; MJPG gives the best frame rate
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if not self.cap.isOpened():
            print("Unable to read camera feed")
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

    def _read_calibfile(self):
        """Load servo calibration offsets, if saved, and apply them."""
        try:
            with open(CALIBRATION_FILE, "r") as f:
                self.cal_steer, self.cal_pan, self.cal_tilt = (int(f.readline()) for _ in range(3))
        except FileNotFoundError:
            return
        self.handle_servos()

    def write_calibfile(self):
        with open(CALIBRATION_FILE, "w+") as f:
            f.write(f"{self.cal_steer}\n{self.cal_pan}\n{self.cal_tilt}\n")

    def _print_threshold_help(self):
        print("You can use these keys to tune the color thresholds:")
        print("  Upper HSV threshold +/-")
        print("  7    8    9")
        print("  u    i    o")
        print("  Lower HSV threshold +/-")
        print("  j    k    l")
        print("  m    ,    .")
        print("Current color tresholds")
        self._print_thresholds()

    def _print_thresholds(self):
        upper, lower = self.upper_threshold, self.lower_threshold
        print(f"{upper[0]} {upper[1]} {upper[2]}")
        print(f"{lower[0]} {lower[1]} {lower[2]}")

    def _write_time_consumptions(self):
        """Write the measured durations to a CSV file, one column per code block.

        Uses ';' as separator and ',' as decimal mark (German spreadsheet format).
        Shorter columns are padded with "end", and an all-"end" row closes the file.
        """
        filename = time.strftime("time_consumption-%Y-%m-%d-%H-%M-%S.csv", time.localtime(time.time()))
        columns = list(self.time_consumptions.values())
        with open(filename, "w+") as f:
            f.write(";".join(self.time_consumptions) + "\n")
            for row in zip_longest(*columns, fillvalue="end"):
                f.write(";".join(str(value).replace(".", ",") for value in row) + "\n")
            f.write(";".join("end" for _ in columns) + "\n")

    # ------------------------------------------------------------------
    # Video stream server
    # ------------------------------------------------------------------

    def _start_stream_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("0.0.0.0", STREAM_PORT))
        self.server_socket.listen(5)
        self._wait_for_client()

    def _wait_for_client(self):
        """Block until a remote client connects."""
        print(f"Server listening on 0.0.0.0:{STREAM_PORT}")
        print("Please start your client program on remote computer and connect to the SmartCar IP Address.")
        self.conn, self.addr = self.server_socket.accept()
        print(f"Connection from: {self.addr}")

    def _send_frame(self):
        # Wire format: b"IMG" + 4-byte big-endian length + pickled JPEG buffer
        try:
            _, buffer = cv2.imencode(".jpg", self.frame)
            data = pickle.dumps(buffer)
            self.conn.sendall(b"IMG" + struct.pack(">L", len(data)) + data)
        except BrokenPipeError:
            self._wait_for_client()

    # ------------------------------------------------------------------
    # Window overlays
    # ------------------------------------------------------------------

    def _put_text(self, text, origin, color):
        cv2.putText(self.frame, text, origin, FONT, 0.5, color, 2)

    def _help_lines(self):
        """Help text listing only the controls that are currently available."""
        lines = ["Usage of the car control:", "show or hide this help: ?"]
        if self.drive_control or self.steer_control or self.camera_control:
            lines.append("reset actuators: x")
        lines.append("quit program: q")
        if self.drive_control:
            lines += ["speed up: w", "speed down: s"]
        if self.steer_control:
            lines += ["left/right: a/d", "servo calibration: c"]
        if self.camera_control:
            lines += ["camera up: t", "camera left/right: f/h", "camera down: g"]
            if not self.steer_control:
                lines.append("servo calibration: c")
        if self.recording_control:
            lines.append("start or stop recording: r")
        if self.use_ultrasonic:
            lines.append(f"us distance: {self.distance:.1f}cm")
        return lines

    def addhelp(self):
        for index, line in enumerate(self._help_lines()):
            self._put_text(line, (10, 20 + index * 20), HELP_COLOR)

    def _blink_text(self, text, offset_from_bottom, shown_since):
        """Show `text` for 0.5s out of every second. Must be called every frame.

        Returns the updated timestamp of when the current blink cycle started.
        """
        now = time.time()
        if shown_since > now - 0.5:
            self._put_text(text, (10, self.frame_height - offset_from_bottom), STATUS_COLOR)
        elif shown_since < now - 1:
            return now
        return shown_since

    def add_text(self, text):
        if text != "":
            self._put_text(text, (150, self.frame_height - 40), INFO_COLOR)

    # ------------------------------------------------------------------
    # Main loop steps: sensors -> (lane detection) -> window -> input -> actuators
    # ------------------------------------------------------------------

    def handle_sensors(self):
        """Grab a fresh camera frame and, if enabled, the ultrasonic distance."""
        with self._timed("handle_sensors.capture_image"):
            self.cap.read()  # Discard the buffered frame, it may be stale
            self.ret, self.frame = self.cap.read()
        with self._timed("handle_sensors.measure_us_distance"):
            if self.use_ultrasonic:
                self.measure_us_distance()

    def measure_us_distance(self):
        """Store the ultrasonic distance in cm, keeping the previous one in last_distance."""
        d = round(self.px.ultrasonic.read(), 2)
        # The sensor reports -1 for a signal error (ignored) and -2 for a timeout (max distance)
        if d <= -2:
            d = 255
        if d >= 0:
            self.last_distance = self.distance
            self.distance = d

    def handle_window(self, additional_text=""):
        """Draw the overlays and show the frame locally or send it to the stream client."""
        if not self.ret:
            return
        with self._timed("handle_window.add_text"):
            self.add_text(additional_text)
        with self._timed("handle_window.copy_frame"):
            # The remaining overlays are only for the viewer, not for the recording
            self.recordingframe = self.frame.copy()
        with self._timed("handle_window.showhelp"):
            if self.showhelp and self.use_window in (SC_USE_LOCAL_WINDOW, SC_USE_STREAM_WINDOW):
                self.addhelp()
        with self._timed("handle_window.addrecording"):
            if self.recording:
                self.recactivated = self._blink_text("recording", 40, self.recactivated)
            if self.calibration:
                self.calactivated = self._blink_text("calibrate servos", 60, self.calactivated)
        with self._timed("handle_window.imshow"):
            if self.use_window == SC_USE_STREAM_WINDOW:
                self._send_frame()
            elif self.use_window == SC_USE_LOCAL_WINDOW:
                cv2.imshow("SmartCar camera", self.frame)

    def user_command(self):
        """Read one key (if any) and apply it to the available controls."""
        with self._timed("user_command"):
            key = self._read_key()
            if key == "q":
                self.quit = True
            if self.drive_control and key in DRIVE_KEYS:
                self._change_speed(DRIVE_KEYS[key])
            if self.steer_control and key in STEER_KEYS:
                self._nudge_servo(*STEER_KEYS[key])
            if self.camera_control and key in CAMERA_KEYS:
                self._nudge_servo(*CAMERA_KEYS[key])
            if self.steer_control or self.camera_control:
                if key == "c":
                    self._toggle_calibration()
            if self.drive_control or self.steer_control or self.camera_control:
                if key == "x":
                    self.speed = 0
                    self.center_servos()
            if key == "?":
                self.showhelp = not self.showhelp
            if self.recording_control and key == "r":
                self._toggle_recording()
            if self.tune_thresholds and key in THRESHOLD_KEYS:
                self._tune_threshold(*THRESHOLD_KEYS[key])
            if key == "y":  # Secret key to cycle through the debug images
                self.debug_image = DEBUG_IMAGE_CYCLE[self.debug_image]

    def _read_key(self):
        """Return the pressed key as a one-character string, or None."""
        if self.use_window == SC_USE_STREAM_WINDOW:
            readable, _, _ = select.select([self.conn], [], [], 0.1)
            if readable:
                try:
                    key = self.conn.recv(1).decode("utf-8")
                except Exception:
                    return None  # Ignore glitches on the connection
                if key:
                    print(f"Received key: {key}")
                    return key
        elif self.use_window == SC_USE_LOCAL_WINDOW:
            return chr(cv2.waitKey(1) & 0xFF)
        return None

    def _change_speed(self, direction):
        self.speed = sc_clamp(self.speed + direction * self.SPEED_STEP, -self.SPEED_LIMIT, self.SPEED_LIMIT)
        print("speed:", self.speed)

    def _nudge_servo(self, position_attr, calibration_attr, direction):
        """Move a servo one step, or adjust and save its calibration offset in calibration mode."""
        if self.calibration:
            setattr(self, calibration_attr, getattr(self, calibration_attr) + direction)
            self.write_calibfile()
        else:
            position = getattr(self, position_attr) + direction * self.SERVO_STEP
            setattr(self, position_attr, sc_clamp(position, -self.SERVO_LIMIT, self.SERVO_LIMIT))
        print(f"{position_attr}:", getattr(self, position_attr))

    def _toggle_calibration(self):
        self.calibration = not self.calibration
        if self.calibration:
            self.center_servos()

    def _toggle_recording(self):
        if self.recording:
            if self.out is not None:
                self.out.release()
        else:
            filename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())) + ".avi"
            self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10,
                                       (self.frame_width, self.frame_height))
        self.recording = not self.recording

    def _tune_threshold(self, threshold_attr, channel, step):
        getattr(self, threshold_attr)[channel] += step
        self._print_thresholds()

    def handle_actuators(self):
        """Apply speed and servo angles to the car and record the frame if requested."""
        with self._timed("handle_actuators"):
            if self._obstacle_ahead():
                print(f"stopped because d = {self.distance}")
                print(f"Last distance was d = {self.last_distance}")
                self.speed = 0
            if self.speed < 0:
                self.px.backward(abs(self.speed))
            else:
                self.px.forward(self.speed)
            self.handle_servos()
            if self.recording:
                self.out.write(self.recordingframe)

    def _obstacle_ahead(self):
        """True if driving forward towards a close obstacle.

        Readings that jumped by more than MAX_TRUSTED_DISTANCE_JUMP_CM since
        the last measurement are treated as sensor glitches.
        """
        return (self.use_ultrasonic
                and self.speed > 0
                and self.distance < self.EMERGENCY_STOP_CM
                and abs(self.last_distance - self.distance) < self.MAX_TRUSTED_DISTANCE_JUMP_CM)

    def handle_servos(self):
        """Send the servo angles plus calibration offsets to the hardware."""
        limit = self.SERVO_LIMIT
        self.px.set_dir_servo_angle(sc_clamp(self.steer + self.cal_steer, -limit, limit))
        self.px.set_cam_pan_angle(sc_clamp(self.campan + self.cal_pan, -limit, limit))
        self.px.set_cam_tilt_angle(sc_clamp(self.camtilt + self.cal_tilt, -limit, limit))

    def center_servos(self):
        """Move all servos softly back to their center position."""
        start = (self.steer, self.campan, self.camtilt)
        steps = self.CENTERING_STEPS
        for i in range(steps):
            self.steer, self.campan, self.camtilt = (int(v - v * i / steps) for v in start)
            self.handle_servos()
        self.steer = self.campan = self.camtilt = 0
        self.handle_servos()

    # ------------------------------------------------------------------
    # Lane detection and automatic steering
    # ------------------------------------------------------------------

    def lane_detection(self):
        """Detect the blue lane markings and steer/pan the car automatically.

        This is the Gruppe 3/8 multi-row adaptive controller: several scan
        rows are evaluated per frame instead of stopping at the first hit,
        the path is classified as straight/curve from how much the target
        moves between rows, and the steering step size is limited to avoid
        jerky movement. Camera pan is coupled to the steering angle at the
        end of every call (Kamera an Spurhaltung verkuppelt).

        Vision always runs so the debug image / stream stays live, but the
        steering angle is only updated while the car is moving (self.speed
        != 0), mirroring the old PID controller's behaviour. When stopped,
        the last steering angle is held.
        """
        with self._timed("lane_detection.hsv"):
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_threshold, self.upper_threshold)

        with self._timed("lane_detection.mask"):
            # Remove speckles, then close small gaps in the tape
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        with self._timed("lane_detection.debugimage"):
            if self.debug_image == SC_DEBUG_MASK:
                self.frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            elif self.debug_image == SC_DEBUG_OVERLAY:
                self.frame = np.bitwise_or(self.frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

        with self._timed("lane_detection.detect_lanes"):
            height, width = self.frame.shape[:2]
            pairs, marks, marks_row = self._scan_rows(mask, height, width)
            self._update_lane_target(pairs, marks, width)
            self._draw_lane_target(height, width, marks_row)

        with self._timed("lane_detection.controller"):
            if self.speed != 0:
                self._lane_auto_steer(width)

    def _scan_rows(self, mask, height, width):
        """Look for lane marks along each scan row, nearest row first.

        Returns the lane pair (left, right) of every row that has one, plus
        the marks and y position of the nearest row with any marks at all.
        """
        min_pair_gap = int(width * self.SC_MIN_PAIR_GAP_FRACTION)
        rows = [int(height * fraction) for fraction in self.SC_SCAN_ROWS]
        pairs = []
        nearest_marks, nearest_row = [], rows[0]
        for row in rows:
            marks = self._extract_lane_marks(mask, row)
            cv2.line(self.frame, (0, row), (width - 1, row), (70, 70, 70), 1)
            for mark in marks:
                cv2.circle(self.frame, (mark, row), 4, (0, 255, 255), -1)

            pair = self._choose_lane_pair(marks, min_pair_gap)
            if pair:
                pairs.append(pair)
            if marks and not nearest_marks:
                nearest_marks, nearest_row = marks, row
        return pairs, nearest_marks, nearest_row

    def _update_lane_target(self, pairs, marks, width):
        """Choose the steering target (lane_target_x) and update the lane state."""
        self.lane_seen = bool(pairs or marks)
        if pairs:
            # Lower rows are close to the robot; upper rows look ahead.
            # Blend both: stable on straights, earlier response in curves.
            targets = [(float(left) + float(right)) / 2.0 for left, right in pairs]
            near, far = targets[0], targets[-1]
            average = sum(targets) / float(len(targets))
            self.curve_score_x = abs(far - near)
            if self.curve_score_x >= self.SC_CURVE_SCORE_PX:
                self.path_mode = "curve"
                target = 0.45 * near + 0.35 * average + 0.20 * far
            else:
                self.path_mode = "straight"
                target = 0.70 * near + 0.30 * average

            left, right = pairs[0]
            self.lane_count = 2
            self.last_left_x, self.last_right_x = int(left), int(right)
            self.last_lane_width = float(max(1, right - left))
        else:
            self.path_mode = "search"
            self.curve_score_x = 0.0
            self.lane_count = len(marks)  # Note: can be 2+ if the marks are too close for a pair
            target = self._target_from_single_line(marks[0], width) if marks else None
        self.lane_target_x = target

    def _target_from_single_line(self, x, width):
        """Aim half a lane width beside a single visible lane line."""
        lane_width = self.last_lane_width or float(width) * self.SC_DEFAULT_LANE_WIDTH_FRACTION
        if self._classify_single_line(x, width) == "left":
            target = x + lane_width / 2.0
        else:
            target = x - lane_width / 2.0
        return sc_clamp(target, 0.0, float(width - 1))

    def _draw_lane_target(self, height, width, marks_row):
        """Draw the image center (blue) and the chosen target (green) for the stream."""
        center_x = int(width / 2)
        top = int(height * 0.46)
        cv2.line(self.frame, (center_x, height), (center_x, top), (255, 0, 0), 1)
        if self.lane_target_x is not None:
            target_x = int(self.lane_target_x)
            cv2.line(self.frame, (target_x, height), (target_x, top), (0, 255, 0), 2)
        cv2.circle(self.frame, (center_x, marks_row), 5, (255, 0, 0), -1)

    def _extract_lane_marks(self, mask, row):
        """Find blue mark centers along one horizontal scan row (sorted, in pixels)."""
        height, width = mask.shape[:2]
        # Merge three pixel rows so a single noisy row does not break a mark
        band = mask[max(0, row - 1):min(height, row + 2)]
        row_mask = np.max(band, axis=0) > 0
        edges = np.diff(np.pad(row_mask.astype(np.int8), (1, 1), mode="constant"))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]

        # Accept only tape-sized segments; limits are tuned for a 640px wide image
        scale = width / 640.0
        min_width = max(3, int(7 * scale))
        max_width = max(min_width + 1, int(95 * scale))
        return sorted(int((start + end) / 2) for start, end in zip(starts, ends)
                      if min_width <= end - start <= max_width)

    def _choose_lane_pair(self, marks, min_pair_gap):
        """Choose the widest pair of lane marks that are at least min_pair_gap apart."""
        pairs = [(left, right) for i, left in enumerate(marks) for right in marks[i + 1:]
                 if right - left >= min_pair_gap]
        return max(pairs, key=lambda pair: pair[1] - pair[0], default=None)

    def _classify_single_line(self, x, width):
        """Estimate whether a single visible line is the left or right lane line."""
        if self.last_left_x is not None and self.last_right_x is not None:
            closer_to_left = abs(x - self.last_left_x) <= abs(x - self.last_right_x)
            return "left" if closer_to_left else "right"
        return "left" if x < width / 2 else "right"

    def _lane_auto_steer(self, frame_width):
        """Fast curve response, damped straight response.

        Classifies the path as straight/curve from the multi-row lane
        target. On straights it avoids chasing tiny pixel errors; in curves
        it uses less filtering and larger allowed steering steps. Updates
        self.steer and then couples the camera pan to it.
        """
        if self.lane_target_x is None or self.lane_count < 2:
            self.path_mode = "search"
            self._update_camera_pan()
            return

        error = float(self.lane_target_x) - frame_width / 2.0
        self.auto_error_x = error

        is_curve = (
            self.path_mode == "curve"
            or abs(error) >= self.SC_CURVE_ERROR_PX
            or self.curve_score_x >= self.SC_CURVE_SCORE_PX
        )
        alpha = self.SC_CURVE_FILTER_ALPHA if is_curve else self.SC_STRAIGHT_FILTER_ALPHA
        self.filtered_error_x = alpha * self.filtered_error_x + (1.0 - alpha) * error

        if not is_curve and abs(self.filtered_error_x) <= self.SC_AUTO_TARGET_DEADBAND_PX:
            # Straight and close enough: do not zig-zag around the center.
            target_steer = self.SC_STEER_CENTER
            step_limit = self.SC_STRAIGHT_STEP_LIMIT
        else:
            gain_factor = self.SC_CURVE_GAIN_FACTOR if is_curve else self.SC_STRAIGHT_GAIN_FACTOR
            steer_delta = self.filtered_error_x * self.SC_AUTO_STEER_GAIN * gain_factor
            steer_delta = sc_clamp(steer_delta, -self.SC_AUTO_MAX_STEER_DELTA, self.SC_AUTO_MAX_STEER_DELTA)
            target_steer = sc_clamp(self.SC_STEER_CENTER + steer_delta, self.SC_STEER_MIN, self.SC_STEER_MAX)
            step_limit = self.SC_CURVE_STEP_LIMIT if is_curve else self.SC_MID_STEP_LIMIT
        self.auto_target_steer = target_steer

        # Move toward the target, but never more than step_limit per frame
        change = sc_clamp(target_steer - self.steer, -step_limit, step_limit)
        self.steer = sc_clamp(self.steer + change, self.SC_STEER_MIN, self.SC_STEER_MAX)
        self._update_camera_pan()

    def _update_camera_pan(self):
        """Couple camera pan to the current steering angle (Kamera verkuppelt).

        Deliberately absolute, not incremental: for every steering angle
        there is exactly one matching camera-pan angle, matching the Gruppe
        3/8 implementation (gain 0.95).
        """
        steer_delta = float(self.steer) - self.SC_STEER_CENTER
        self.campan = sc_clamp(
            self.SC_PAN_CENTER + steer_delta * self.SC_PAN_COUPLING_GAIN,
            self.SC_PAN_MIN,
            self.SC_PAN_MAX,
        )

    # ------------------------------------------------------------------
    # Performance measurement
    # ------------------------------------------------------------------

    def start_timer(self, blockname):
        """Store the start time of a code block (only when measuring performance)."""
        if self.measure_performance:
            self.time_consumptions.setdefault(blockname, []).append(time.time())

    def stop_timer(self, blockname):
        """Replace the block's start time with its elapsed time."""
        if self.measure_performance:
            if blockname in self.time_consumptions:
                times = self.time_consumptions[blockname]
                times[-1] = time.time() - times[-1]
            else:
                self.time_consumptions[blockname] = ["error"]

    @contextmanager
    def _timed(self, blockname):
        self.start_timer(blockname)
        yield
        self.stop_timer(blockname)

    # ------------------------------------------------------------------
    # Ready-made main loops
    # ------------------------------------------------------------------

    def _set_controls(self, drive, steer, camera, recording, ultrasonic):
        self.drive_control = drive
        self.steer_control = steer
        self.camera_control = camera
        self.recording_control = recording
        self.use_ultrasonic = ultrasonic

    def _manual_loop(self):
        while True:
            self.handle_sensors()
            self.handle_window()
            self.user_command()
            if self.quit:
                break
            self.handle_actuators()

    def camera_window_loop(self):
        """Only show the camera window, with camera and recording controls."""
        self._set_controls(drive=False, steer=False, camera=True, recording=True, ultrasonic=False)
        self._manual_loop()

    def drive_and_record_loop(self):
        """Drive the car manually by keyboard while showing the camera window."""
        self._set_controls(drive=True, steer=True, camera=True, recording=True, ultrasonic=True)
        self._manual_loop()

    def lane_detection_loop(self):
        """Steer automatically along the lane; speed, camera and recording stay manual."""
        self._set_controls(drive=True, steer=False, camera=True, recording=True, ultrasonic=True)
        while True:
            self.handle_sensors()
            self.lane_detection()
            self.user_command()
            if self.quit:
                break
            self.handle_actuators()
            self.handle_window()
