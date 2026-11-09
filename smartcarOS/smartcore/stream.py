"""MJPEG HTTP server: /stream.mjpg (live) and /snapshot.jpg. Encoding only happens while someone watches."""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

BOUNDARY = "frame"


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True          # open viewers must never block process exit
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        owner = self.server.owner
        if self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            owner._clients_changed(+1)
            try:
                seq = 0
                while not owner.stopped:
                    jpg, seq = owner.wait_frame(seq, 1.0)
                    if jpg is None:
                        continue
                    head = "--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % (BOUNDARY, len(jpg))
                    self.wfile.write(head.encode() + jpg + b"\r\n")
            except OSError:
                pass
            finally:
                owner._clients_changed(-1)
        elif self.path.startswith("/snapshot.jpg"):
            jpg, _ = owner.wait_frame(-1, 0.0)
            if jpg is None:
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
        else:
            self.send_error(404)


class MJPEGServer:
    def __init__(self, port=8080, host="0.0.0.0"):
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self.clients = 0
        self.stopped = False
        self._httpd = _Server((host, port), _Handler)
        self._httpd.owner = self
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def publish(self, jpeg_bytes):
        with self._cond:
            self._jpeg = jpeg_bytes
            self._seq += 1
            self._cond.notify_all()

    def wait_frame(self, last_seq, timeout):
        """Block until a frame newer than last_seq exists. Returns (jpeg or None, seq)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or self.stopped, timeout)
            if self._seq != last_seq and self._jpeg is not None:
                return self._jpeg, self._seq
            return None, last_seq

    def _clients_changed(self, delta):
        with self._cond:
            self.clients += delta

    def stop(self):
        with self._cond:
            self.stopped = True
            self._cond.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()
