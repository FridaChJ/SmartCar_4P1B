"""Everything drawn on top of the camera image: texts, debug views and lane markings."""
import time

import cv2
import numpy as np

# Debug modes for the displayed video
SC_DEBUG_NONE = 0     # Plain camera image
SC_DEBUG_MASK = 1     # Black/white lane mask
SC_DEBUG_OVERLAY = 2  # Lane mask painted over the camera image

FONT = cv2.FONT_HERSHEY_SIMPLEX
# Colors are BGR
HELP_COLOR = (180, 0, 0)
STATUS_COLOR = (0, 0, 255)
INFO_COLOR = (255, 100, 100)


def put_text(frame, text, origin, color):
    cv2.putText(frame, text, origin, FONT, 0.5, color, 2)


def draw_help(frame, lines):
    for index, line in enumerate(lines):
        put_text(frame, line, (10, 20 + index * 20), HELP_COLOR)


def draw_info(frame, text):
    """Optional free text near the bottom of the image, e.g. from a calling script."""
    if text != "":
        put_text(frame, text, (150, frame.shape[0] - 40), INFO_COLOR)


class BlinkingText:
    """Status text that is visible for 0.5s out of every second. Draw it every frame."""

    def __init__(self, text, offset_from_bottom):
        self.text = text
        self.offset_from_bottom = offset_from_bottom
        self.cycle_start = 0

    def draw(self, frame):
        now = time.time()
        if self.cycle_start > now - 0.5:
            put_text(frame, self.text, (10, frame.shape[0] - self.offset_from_bottom), STATUS_COLOR)
        elif self.cycle_start < now - 1:
            self.cycle_start = now


def debug_view(frame, mask, mode):
    """Return the image to display for one of the SC_DEBUG_* modes."""
    if mode == SC_DEBUG_MASK:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if mode == SC_DEBUG_OVERLAY:
        return np.bitwise_or(frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    return frame


def draw_lane(frame, lane):
    """Draw scan rows, found marks, the image center (blue) and the steering target (green)."""
    height, width = frame.shape[:2]
    for row, marks in lane.rows:
        cv2.line(frame, (0, row), (width - 1, row), (70, 70, 70), 1)
        for mark in marks:
            cv2.circle(frame, (mark, row), 4, (0, 255, 255), -1)

    center_x = int(width / 2)
    top = int(height * 0.46)
    cv2.line(frame, (center_x, height), (center_x, top), (255, 0, 0), 1)
    if lane.target_x is not None:
        target_x = int(lane.target_x)
        cv2.line(frame, (target_x, height), (target_x, top), (0, 255, 0), 2)
    cv2.circle(frame, (center_x, lane.marks_row), 5, (255, 0, 0), -1)
