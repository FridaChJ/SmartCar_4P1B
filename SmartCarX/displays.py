"""Where the camera image is shown and where key presses come from.

Each display offers the same three methods, so SmartCar does not need to
know which one is in use:
    read_key() -> one-character string or None
    show(frame)
    close()
"""
import pickle
import select
import socket
import struct

import cv2

# How video and keyboard input are connected
SC_USE_NO_WINDOW = 0      # Headless, e.g. for dashboard control
SC_USE_LOCAL_WINDOW = 1   # Native OpenCV window, needs an X server on the remote computer
SC_USE_STREAM_WINDOW = 2  # Socket server that streams video to a client and receives its keys

STREAM_PORT = 5001


class NoWindow:
    """No video output and no keyboard input."""

    shows_help = False  # Whether on-screen help makes sense for this display

    def read_key(self):
        return None

    def show(self, frame):
        pass

    def close(self):
        pass


class LocalWindow(NoWindow):
    """Native OpenCV window."""

    shows_help = True

    def read_key(self):
        return chr(cv2.waitKey(1) & 0xFF)

    def show(self, frame):
        cv2.imshow("SmartCar camera", frame)


class StreamWindow(NoWindow):
    """Streams JPEG frames to a remote client over TCP and receives its key presses.

    Wire format per frame: b"IMG" + 4-byte big-endian length + pickled JPEG buffer.
    """

    shows_help = True

    def __init__(self, server_socket):
        self.server_socket = server_socket
        self.conn = None
        self.addr = None
        self._wait_for_client()

    @classmethod
    def listen(cls, port=STREAM_PORT):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("0.0.0.0", port))
        server_socket.listen(5)
        return cls(server_socket)

    def _wait_for_client(self):
        """Block until a remote client connects."""
        print(f"Server listening on 0.0.0.0:{STREAM_PORT}")
        print("Please start your client program on remote computer and connect to the SmartCar IP Address.")
        self.conn, self.addr = self.server_socket.accept()
        print(f"Connection from: {self.addr}")

    def read_key(self):
        readable, _, _ = select.select([self.conn], [], [], 0.1)
        if not readable:
            return None
        try:
            key = self.conn.recv(1).decode("utf-8")
        except Exception:
            return None  # Ignore glitches on the connection
        if not key:
            return None
        print(f"Received key: {key}")
        return key

    def show(self, frame):
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            data = pickle.dumps(buffer)
            self.conn.sendall(b"IMG" + struct.pack(">L", len(data)) + data)
        except BrokenPipeError:
            self._wait_for_client()

    def close(self):
        if self.conn:
            self.conn.close()


def create_display(use_window):
    """Build the display for one of the SC_USE_* options."""
    if use_window == SC_USE_STREAM_WINDOW:
        return StreamWindow.listen()
    if use_window == SC_USE_LOCAL_WINDOW:
        return LocalWindow()
    return NoWindow()
