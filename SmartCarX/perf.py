"""Optional timing of code blocks, saved as a CSV file for later analysis."""
import time
from contextlib import contextmanager
from itertools import zip_longest


class PerformanceTimer:
    """Collects durations per named code block while enabled; does nothing otherwise."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.timings = {}  # Block name -> list of durations in s

    def start(self, blockname):
        if self.enabled:
            self.timings.setdefault(blockname, []).append(time.time())

    def stop(self, blockname):
        """Replace the block's start time with its elapsed time."""
        if not self.enabled:
            return
        if blockname in self.timings:
            times = self.timings[blockname]
            times[-1] = time.time() - times[-1]
        else:
            self.timings[blockname] = ["error"]

    @contextmanager
    def timed(self, blockname):
        self.start(blockname)
        yield
        self.stop(blockname)

    def write_csv(self):
        """Write one column per code block to time_consumption-<timestamp>.csv.

        Uses ';' as separator and ',' as decimal mark (German spreadsheet format).
        Shorter columns are padded with "end", and an all-"end" row closes the file.
        """
        filename = time.strftime("time_consumption-%Y-%m-%d-%H-%M-%S.csv", time.localtime(time.time()))
        columns = list(self.timings.values())
        with open(filename, "w+") as f:
            f.write(";".join(self.timings) + "\n")
            for row in zip_longest(*columns, fillvalue="end"):
                f.write(";".join(str(value).replace(".", ",") for value in row) + "\n")
            f.write(";".join("end" for _ in columns) + "\n")
