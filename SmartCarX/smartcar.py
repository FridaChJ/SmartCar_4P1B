"""
SmartCar class for WIP Smart Systems lecture
version 0.10:
Split the class into small components (hardware, displays, overlays, recorder,
lane detection/steering, keyboard controls, performance timer); SmartCar now
only wires them together and runs the main loop steps. Added unit tests.
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
import cv2
from picarx import Picarx

from controls import Controls, print_threshold_help
from displays import SC_USE_LOCAL_WINDOW, SC_USE_NO_WINDOW, SC_USE_STREAM_WINDOW, create_display  # noqa: F401
from hardware import Calibration, Camera, DistanceSensor, Vehicle
from lane import LaneDetector, LaneResult, LaneSteeringController, coupled_camera_pan
from overlays import (SC_DEBUG_MASK, SC_DEBUG_NONE, SC_DEBUG_OVERLAY,  # noqa: F401
                      BlinkingText, debug_view, draw_help, draw_info, draw_lane)
from perf import PerformanceTimer
from recorder import VideoRecorder
from sc_math import sc_clamp  # noqa: F401  (kept importable as smartcar.sc_clamp)


def _vehicle_property(name):
    """Expose a Vehicle attribute directly on SmartCar (e.g. sc.speed)."""
    return property(lambda self: getattr(self.vehicle, name),
                    lambda self, value: setattr(self.vehicle, name, value))


class SmartCar(object):
    """Wires the car's components together and provides the main loop steps.

    A loop iteration is: handle_sensors -> (lane_detection) -> handle_window
    -> user_command -> handle_actuators. The ready-made loops below combine
    them for the lecture scripts.
    """

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
        self.controls = Controls(drive_control, steer_control, camera_control,
                                 recording_control, use_ultrasonic, tune_thresholds)
        self.calibration = Calibration()
        self.vehicle = Vehicle(self.px, self.calibration)
        self.sensor = DistanceSensor(self.px.ultrasonic)
        self.detector = LaneDetector(lower_threshold, upper_threshold)
        self.steering = LaneSteeringController()
        self.lane = LaneResult()
        self.timer = PerformanceTimer(measure_performance)

        self.debug_image = debug_image
        self.showhelp = True
        self.quit = False
        self.recording_text = BlinkingText("recording", 40)
        self.calibration_text = BlinkingText("calibrate servos", 60)

        if tune_thresholds:
            print_threshold_help(self.detector)
        self.camera = Camera()
        self.recorder = VideoRecorder(self.camera.width, self.camera.height)
        if self.calibration.load():
            self.vehicle.apply_servos()
        self.display = create_display(use_window)

    def __del__(self):
        print("Destructor called")
        if self.timer.enabled:
            self.timer.write_csv()
        self.camera.release()
        self.display.close()
        self.recorder.close()
        cv2.destroyAllWindows()
        self.vehicle.shutdown()

    # Driving state used by the scripts lives in the Vehicle
    speed = _vehicle_property("speed")
    steer = _vehicle_property("steer")
    campan = _vehicle_property("campan")
    camtilt = _vehicle_property("camtilt")

    # ------------------------------------------------------------------
    # Main loop steps
    # ------------------------------------------------------------------

    def handle_sensors(self):
        """Grab a fresh camera frame and, if enabled, the ultrasonic distance."""
        with self.timer.timed("handle_sensors.capture_image"):
            self.ret, self.frame = self.camera.read()
        with self.timer.timed("handle_sensors.measure_us_distance"):
            if self.controls.ultrasonic:
                self.sensor.measure()

    def lane_detection(self):
        """Find the lane and, while driving, steer along it with the camera following.

        Vision always runs so the debug image / stream stays live. When the
        car stands still the last steering angle is held.
        """
        with self.timer.timed("lane_detection.hsv"):
            mask = self.detector.color_mask(self.frame)
        with self.timer.timed("lane_detection.mask"):
            mask = self.detector.denoise(mask)
        with self.timer.timed("lane_detection.debugimage"):
            self.frame = debug_view(self.frame, mask, self.debug_image)
        with self.timer.timed("lane_detection.detect_lanes"):
            self.lane = self.detector.detect(mask)
            draw_lane(self.frame, self.lane)
        with self.timer.timed("lane_detection.controller"):
            if self.speed != 0:
                self.steer = self.steering.update(self.lane, self.steer, self.frame.shape[1])
                self.campan = coupled_camera_pan(self.steer)

    def handle_window(self, additional_text=""):
        """Draw the overlays and show the frame on the display."""
        if not self.ret:
            return
        with self.timer.timed("handle_window.add_text"):
            draw_info(self.frame, additional_text)
        with self.timer.timed("handle_window.copy_frame"):
            # The remaining overlays are only for the viewer, not for the recording
            self.recordingframe = self.frame.copy()
        with self.timer.timed("handle_window.showhelp"):
            if self.showhelp and self.display.shows_help:
                draw_help(self.frame, self.controls.help_lines(self.sensor.distance))
        with self.timer.timed("handle_window.addrecording"):
            if self.recorder.recording:
                self.recording_text.draw(self.frame)
            if self.calibration.active:
                self.calibration_text.draw(self.frame)
        with self.timer.timed("handle_window.imshow"):
            self.display.show(self.frame)

    def user_command(self):
        """Read one key (if any) from the display and apply it."""
        with self.timer.timed("user_command"):
            self.controls.handle_key(self, self.display.read_key())

    def handle_actuators(self):
        """Apply speed and servo angles, stopping for close obstacles, and record the frame."""
        with self.timer.timed("handle_actuators"):
            if self.controls.ultrasonic and self.speed > 0 and self.sensor.obstacle_ahead():
                print(f"stopped because d = {self.sensor.distance}")
                print(f"Last distance was d = {self.sensor.last_distance}")
                self.speed = 0
            self.vehicle.apply()
            if self.recorder.recording:
                self.recorder.write(self.recordingframe)

    # ------------------------------------------------------------------
    # Ready-made main loops
    # ------------------------------------------------------------------

    def _enable(self, drive, steer, camera, recording, ultrasonic):
        self.controls.drive = drive
        self.controls.steer = steer
        self.controls.camera = camera
        self.controls.recording = recording
        self.controls.ultrasonic = ultrasonic

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
        self._enable(drive=False, steer=False, camera=True, recording=True, ultrasonic=False)
        self._manual_loop()

    def drive_and_record_loop(self):
        """Drive the car manually by keyboard while showing the camera window."""
        self._enable(drive=True, steer=True, camera=True, recording=True, ultrasonic=True)
        self._manual_loop()

    def lane_detection_loop(self):
        """Steer automatically along the lane; speed, camera and recording stay manual."""
        self._enable(drive=True, steer=False, camera=True, recording=True, ultrasonic=True)
        while True:
            self.handle_sensors()
            self.lane_detection()
            self.user_command()
            if self.quit:
                break
            self.handle_actuators()
            self.handle_window()
