"""Recording of the camera image to timestamped .avi files."""
import time

import cv2


class VideoRecorder:
    FPS = 10

    def __init__(self, width, height):
        self.size = (width, height)
        self.recording = False
        self.out = None

    def toggle(self):
        """Start a new <timestamp>.avi file, or stop the current recording."""
        if self.recording:
            self.close()
        else:
            filename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())) + ".avi"
            self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                                       self.FPS, self.size)
        self.recording = not self.recording

    def write(self, frame):
        if self.recording:
            self.out.write(frame)

    def close(self):
        if self.out is not None:
            self.out.release()
