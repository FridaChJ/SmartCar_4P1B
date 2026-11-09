"""Lane detection and automatic steering (Gruppe 3/8 multi-row adaptive controller).

Several scan rows are evaluated per frame instead of stopping at the first
hit, the path is classified as straight/curve from how much the target
moves between rows, and the steering step size is limited to avoid jerky
movement. Camera pan is coupled to the steering angle (Kamera an
Spurhaltung verkuppelt).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .sc_math import sc_clamp

# HSV thresholds for the blue lane tape, from students and Siavash in summer 2026.
# Earlier tunings (lower / upper):
#   original:       [99, 23, 65] / [123, 255, 255]
#   Markus' home:   [105, 43, 76] / [117, 255, 255]
#   Markus' office: [98, 79, 81] / [117, 219, 249]
DEFAULT_LOWER_THRESHOLD = (96, 21, 81)
DEFAULT_UPPER_THRESHOLD = (116, 219, 255)

# Target drift (px) between the nearest and farthest scan row that marks a curve
CURVE_SCORE_PX = 28.0


@dataclass
class LaneResult:
    """What the detector found in one frame."""
    target_x: float = None       # Steering target in pixels, or None if nothing was found
    lane_count: int = 0          # 2 = pair found, otherwise the number of marks (see detect)
    path_mode: str = "search"    # "straight" | "curve" | "search"
    curve_score_x: float = 0.0   # Spread of the target between scan rows
    rows: list = field(default_factory=list)  # (y, marks) for every scan row, for drawing
    marks_row: int = 0           # y of the nearest row with marks

    @property
    def lane_seen(self):
        return self.target_x is not None


class LaneDetector:
    """Finds the blue lane tape in camera frames and picks a steering target."""

    # Scan rows as fractions of the image height, from the bottom (close to
    # the car) to the top (looking ahead)
    SCAN_ROWS = (0.80, 0.73, 0.66, 0.59, 0.52)
    MIN_PAIR_GAP_FRACTION = 0.28
    DEFAULT_LANE_WIDTH_FRACTION = 0.55

    def __init__(self, lower_threshold=None, upper_threshold=None):
        # HSV color range of the tape; tunable at runtime, so these are mutable arrays
        self.lower_threshold = np.array(DEFAULT_LOWER_THRESHOLD) if lower_threshold is None else lower_threshold
        self.upper_threshold = np.array(DEFAULT_UPPER_THRESHOLD) if upper_threshold is None else upper_threshold
        # Last known lane, used to classify a single visible line
        self.last_left_x = None
        self.last_right_x = None
        self.last_lane_width = None

    def color_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.lower_threshold, self.upper_threshold)

    @staticmethod
    def denoise(mask):
        """Remove speckles, then close small gaps in the tape."""
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def detect(self, mask):
        height, width = mask.shape[:2]
        min_pair_gap = int(width * self.MIN_PAIR_GAP_FRACTION)
        result = LaneResult()
        pairs = []
        nearest_marks = []
        result.marks_row = int(height * self.SCAN_ROWS[0])
        for fraction in self.SCAN_ROWS:
            row = int(height * fraction)
            marks = self._extract_lane_marks(mask, row)
            result.rows.append((row, marks))
            pair = self._choose_lane_pair(marks, min_pair_gap)
            if pair:
                pairs.append(pair)
            if marks and not nearest_marks:
                nearest_marks, result.marks_row = marks, row

        if pairs:
            self._target_from_pairs(result, pairs)
        elif nearest_marks:
            # Note: lane_count can be 2+ here if the marks are too close for a pair
            result.lane_count = len(nearest_marks)
            result.target_x = self._target_from_single_line(nearest_marks[0], width)
        return result

    def _target_from_pairs(self, result, pairs):
        # Lower rows are close to the robot; upper rows look ahead.
        # Blend both: stable on straights, earlier response in curves.
        targets = [(float(left) + float(right)) / 2.0 for left, right in pairs]
        near, far = targets[0], targets[-1]
        average = sum(targets) / float(len(targets))
        result.curve_score_x = abs(far - near)
        if result.curve_score_x >= CURVE_SCORE_PX:
            result.path_mode = "curve"
            result.target_x = 0.45 * near + 0.35 * average + 0.20 * far
        else:
            result.path_mode = "straight"
            result.target_x = 0.70 * near + 0.30 * average
        result.lane_count = 2

        left, right = pairs[0]
        self.last_left_x, self.last_right_x = int(left), int(right)
        self.last_lane_width = float(max(1, right - left))

    def _target_from_single_line(self, x, width):
        """Aim half a lane width beside a single visible lane line."""
        lane_width = self.last_lane_width or float(width) * self.DEFAULT_LANE_WIDTH_FRACTION
        if self._classify_single_line(x, width) == "left":
            target = x + lane_width / 2.0
        else:
            target = x - lane_width / 2.0
        return sc_clamp(target, 0.0, float(width - 1))

    def _classify_single_line(self, x, width):
        """Estimate whether a single visible line is the left or right lane line."""
        if self.last_left_x is not None and self.last_right_x is not None:
            closer_to_left = abs(x - self.last_left_x) <= abs(x - self.last_right_x)
            return "left" if closer_to_left else "right"
        return "left" if x < width / 2 else "right"

    @staticmethod
    def _extract_lane_marks(mask, row):
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

    @staticmethod
    def _choose_lane_pair(marks, min_pair_gap):
        """Choose the widest pair of lane marks that are at least min_pair_gap apart."""
        pairs = [(left, right) for i, left in enumerate(marks) for right in marks[i + 1:]
                 if right - left >= min_pair_gap]
        return max(pairs, key=lambda pair: pair[1] - pair[0], default=None)


class LaneSteeringController:
    """Fast curve response, damped straight response.

    On straights it avoids chasing tiny pixel errors; in curves it uses less
    filtering and larger allowed steering steps.
    """

    CURVE_ERROR_PX = 70.0
    STRAIGHT_FILTER_ALPHA = 0.82
    CURVE_FILTER_ALPHA = 0.45
    STRAIGHT_STEP_LIMIT = 0.9
    CURVE_STEP_LIMIT = 6.5
    MID_STEP_LIMIT = 3.0
    STRAIGHT_GAIN_FACTOR = 0.38
    CURVE_GAIN_FACTOR = 1.15
    STEER_GAIN = 0.22
    MAX_STEER_DELTA = 40.0
    TARGET_DEADBAND_PX = 14.0
    STEER_MIN = -35
    STEER_CENTER = 0
    STEER_MAX = 35

    def __init__(self):
        self.filtered_error_x = 0.0   # EMA-filtered pixel error used for steering
        self.auto_error_x = 0.0       # Raw pixel error, for debugging/telemetry
        self.auto_target_steer = 0    # Steering angle the controller is moving toward

    def update(self, lane, steer, frame_width):
        """Return the new steering angle; holds `steer` unless a lane pair is visible."""
        if lane.target_x is None or lane.lane_count < 2:
            return steer

        error = float(lane.target_x) - frame_width / 2.0
        self.auto_error_x = error

        is_curve = (
            lane.path_mode == "curve"
            or abs(error) >= self.CURVE_ERROR_PX
            or lane.curve_score_x >= CURVE_SCORE_PX
        )
        alpha = self.CURVE_FILTER_ALPHA if is_curve else self.STRAIGHT_FILTER_ALPHA
        self.filtered_error_x = alpha * self.filtered_error_x + (1.0 - alpha) * error

        if not is_curve and abs(self.filtered_error_x) <= self.TARGET_DEADBAND_PX:
            # Straight and close enough: do not zig-zag around the center.
            target_steer = self.STEER_CENTER
            step_limit = self.STRAIGHT_STEP_LIMIT
        else:
            gain_factor = self.CURVE_GAIN_FACTOR if is_curve else self.STRAIGHT_GAIN_FACTOR
            steer_delta = self.filtered_error_x * self.STEER_GAIN * gain_factor
            steer_delta = sc_clamp(steer_delta, -self.MAX_STEER_DELTA, self.MAX_STEER_DELTA)
            target_steer = sc_clamp(self.STEER_CENTER + steer_delta, self.STEER_MIN, self.STEER_MAX)
            step_limit = self.CURVE_STEP_LIMIT if is_curve else self.MID_STEP_LIMIT
        self.auto_target_steer = target_steer

        # Move toward the target, but never more than step_limit per frame
        change = sc_clamp(target_steer - steer, -step_limit, step_limit)
        return sc_clamp(steer + change, self.STEER_MIN, self.STEER_MAX)


PAN_COUPLING_GAIN = 0.95
PAN_MIN, PAN_CENTER, PAN_MAX = -35, 0, 35


def coupled_camera_pan(steer):
    """Camera pan angle that matches a steering angle (Kamera verkuppelt).

    Deliberately absolute, not incremental: for every steering angle there is
    exactly one matching camera-pan angle, as in the Gruppe 3/8 implementation.
    """
    steer_delta = float(steer) - LaneSteeringController.STEER_CENTER
    return sc_clamp(PAN_CENTER + steer_delta * PAN_COUPLING_GAIN, PAN_MIN, PAN_MAX)
