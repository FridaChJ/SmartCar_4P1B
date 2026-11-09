"""Detector plug-in point. A detector turns a BGR frame into a list of normalized Detections."""
import threading
import time

import cv2
import numpy as np

from .types import Detection


class Detector(object):
    def detect(self, frame_bgr):
        """Return list[Detection]. Do not draw, do not keep the frame."""
        raise NotImplementedError


class NullDetector(Detector):
    def detect(self, frame_bgr):
        return []


def _clamp01(v):
    return max(0.0, min(1.0, v))


class EdgeTpuSSD(Detector):
    """The existing Coral SSD model (pycoral). labels: text file, one label per line, index = line number."""

    def __init__(self, model, labels=None, threshold=0.5):
        from pycoral.adapters import common, detect
        from pycoral.utils.edgetpu import make_interpreter
        self._common, self._detect = common, detect
        self.interp = make_interpreter(model)
        self.interp.allocate_tensors()
        self.threshold = threshold
        self.labels = {}
        if labels:
            with open(labels) as fh:
                for i, line in enumerate(fh):
                    if line.strip():
                        self.labels[i] = line.strip().lower().replace(" ", "_")

    def detect(self, frame_bgr):
        import cv2
        from PIL import Image
        h, w = frame_bgr.shape[:2]
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        _, scale = self._common.set_resized_input(self.interp, img.size,
                                                  lambda size: img.resize(size, Image.LANCZOS))
        self.interp.invoke()
        out = []
        for o in self._detect.get_objects(self.interp, self.threshold, scale):
            b = o.bbox
            out.append(Detection(self.labels.get(o.id, str(o.id)), float(o.score),
                                 _clamp01(b.xmin / w), _clamp01(b.ymin / h),
                                 _clamp01(b.xmax / w), _clamp01(b.ymax / h)))
        return out


COCO_NAMES = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,"
    "parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,"
    "tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,"
    "tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,"
    "hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,"
    "cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush"
).split(",")


def letterbox(frame, size_hw, color=114):
    """Resize keeping the aspect ratio and pad to size_hw. Returns (image, scale, pad_x, pad_y)."""
    th, tw = size_hw
    h, w = frame.shape[:2]
    scale = min(tw / float(w), th / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    canvas = np.full((th, tw, 3), color, np.uint8)
    px, py = (tw - nw) // 2, (th - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale, px, py


def nms(boxes, scores, classes, iou_thr):
    """Greedy class-aware non-maximum suppression on xyxy boxes. Returns kept indices, best first."""
    b = boxes + classes[:, None].astype(np.float32) * 4096.0     # boxes of different classes never overlap
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if not rest.size:
            break
        iw = np.minimum(b[i, 2], b[rest, 2]) - np.maximum(b[i, 0], b[rest, 0])
        ih = np.minimum(b[i, 3], b[rest, 3]) - np.maximum(b[i, 1], b[rest, 1])
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        area_r = (b[rest, 2] - b[rest, 0]) * (b[rest, 3] - b[rest, 1])
        order = rest[inter / (area_i + area_r - inter + 1e-9) <= iou_thr]
    return keep


def traffic_light_color(crop_bgr, min_ratio=0.02):
    """Which lamp is lit in a traffic light crop: "red" / "yellow" / "green" / None (COCO has no lamp colors)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (sat > 100) & (val > 150)
    counts = {"red": int((lit & ((hue < 10) | (hue > 170))).sum()),
              "yellow": int((lit & (hue >= 15) & (hue <= 35)).sum()),
              "green": int((lit & (hue >= 40) & (hue <= 90)).sum())}
    best = max(counts, key=counts.get)
    return best if counts[best] >= min_ratio * hue.size else None


class YoloEdgeTpu(Detector):
    """Ultralytics YOLO (v8 / 11 / 26) exported with format=edgetpu, run through pycoral.

    Export on an x86_64 Linux PC (not on the car):  yolo export model=yolo11n.pt format=edgetpu imgsz=320
    Handles int8 input/output quantization, letterboxing, both output layouts (classic (1, 4+nc, N) and
    end2end (1, N, 6)), NMS. labels: "coco" or a text file (one label per line). With tl_color the COCO class
    "traffic_light" is relabeled to its lit lamp ("red"/"yellow"/"green"), which is what the traffic_light rule uses.
    """

    def __init__(self, model=None, labels="coco", threshold=0.4, iou=0.5, max_det=30, tl_color=True,
                 interpreter=None):
        if interpreter is None:
            from pycoral.utils.edgetpu import make_interpreter
            interpreter = make_interpreter(model)
        self.interp = interpreter
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.size_hw = (int(self.inp["shape"][1]), int(self.inp["shape"][2]))
        self.threshold, self.iou, self.max_det, self.tl_color = threshold, iou, max_det, tl_color
        if labels == "coco":
            names = COCO_NAMES
        else:
            with open(labels) as fh:
                names = [line.strip() for line in fh]
        self.names = [n.lower().replace(" ", "_") for n in names]

    def _quantize(self, x):
        dtype = self.inp["dtype"]
        if np.issubdtype(dtype, np.integer):
            scale, zero = self.inp["quantization"]
            info = np.iinfo(dtype)
            x = np.clip(x / scale + zero, info.min, info.max)
        return x.astype(dtype)

    def _decode(self, raw):
        """-> (xyxy boxes in input pixels, scores, class ids) before NMS."""
        th, tw = self.size_hw
        out = raw.astype(np.float32)
        if np.issubdtype(self.out["dtype"], np.integer):
            scale, zero = self.out["quantization"]
            out = (out - zero) * scale
        out = out[0]
        if out.shape[-1] == 6:                                     # end2end rows: x1 y1 x2 y2 conf cls
            keep = out[:, 4] > self.threshold
            out = out[keep]
            return out[:, :4] * np.array([tw, th, tw, th], np.float32), out[:, 4], out[:, 5].astype(int)
        pred = out.T if out.shape[0] < out.shape[1] else out       # -> (N, 4 + nc), xywh normalized
        cls = pred[:, 4:].argmax(1)
        score = pred[:, 4:].max(1)
        keep = score > self.threshold
        pred, cls, score = pred[keep], cls[keep], score[keep]
        xywh = pred[:, :4] * np.array([tw, th, tw, th], np.float32)
        xyxy = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                         xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
        return xyxy, score, cls

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img, scale, px, py = letterbox(frame_bgr, self.size_hw)
        x = np.ascontiguousarray(img[:, :, ::-1], dtype=np.float32) / 255.0        # BGR -> RGB, 0..1
        self.interp.set_tensor(self.inp["index"], self._quantize(x)[None])
        self.interp.invoke()
        boxes, scores, classes = self._decode(self.interp.get_tensor(self.out["index"]))
        if not len(scores):
            return []
        dets = []
        for i in nms(boxes, scores, classes, self.iou)[:self.max_det]:
            x1, y1, x2, y2 = (boxes[i] - np.array([px, py, px, py])) / scale     # back to original pixels
            x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
            y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            c = int(classes[i])
            label = self.names[c] if c < len(self.names) else str(c)
            if self.tl_color and label == "traffic_light":
                label = traffic_light_color(frame_bgr[int(y1):int(y2), int(x1):int(x2)]) or label
            dets.append(Detection(label, float(scores[i]), x1 / w, y1 / h, x2 / w, y2 / h))
        return dets


# Add new backends here (e.g. "remote_yolo") and select them in config.json, e.g.:
#   "detector": {"type": "yolo_edgetpu", "model": "yolo11n_edgetpu.tflite", "threshold": 0.4}
DETECTORS = {"null": NullDetector, "edgetpu_ssd": EdgeTpuSSD, "yolo_edgetpu": YoloEdgeTpu}


def build_detector(spec):
    spec = dict(spec)
    return DETECTORS[spec.pop("type", "null")](**spec)


class DetectorWorker(threading.Thread):
    """Runs the detector at its own pace on the newest frame and publishes the result in the world."""

    def __init__(self, detector, camera, world):
        super().__init__(daemon=True)
        self.detector, self.camera, self.world = detector, camera, world
        self._halt = threading.Event()

    def run(self):
        last_ts, last_err = None, 0.0
        while not self._halt.is_set():
            frame, ts = self.camera.latest()
            if frame is None or ts == last_ts:
                self._halt.wait(0.005)
                continue
            last_ts = ts
            t0 = time.monotonic()
            try:
                dets = self.detector.detect(frame)
            except Exception as exc:                      # a broken detector must not kill the car
                if time.monotonic() - last_err > 5.0:
                    print("[DET] detector failed:", exc)
                    last_err = time.monotonic()
                self._halt.wait(0.2)
                continue
            ms = (time.monotonic() - t0) * 1000.0
            self.world.infer_ms = 0.8 * self.world.infer_ms + 0.2 * ms
            self.world.detections = dets
            self.world.detections_ts = ts                 # age counts from capture time

    def stop(self):
        self._halt.set()
        self.join(1.0)
