import unittest

import numpy as np

from smartcore.perception import (COCO_NAMES, YoloEdgeTpu, build_detector, letterbox, nms,
                                  traffic_light_color)


class FakeInterp:
    """Minimal tflite Interpreter stand-in with quantization details."""

    def __init__(self, out, in_dtype=np.int8, in_q=(1 / 255.0, -128), out_q=(0.0, 0), shape=(1, 320, 320, 3)):
        self.out, self.in_dtype, self.in_q, self.out_q, self.shape = out, in_dtype, in_q, out_q, shape
        self.fed = None

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"index": 0, "shape": np.array(self.shape), "dtype": self.in_dtype, "quantization": self.in_q}]

    def get_output_details(self):
        return [{"index": 1, "dtype": self.out.dtype, "quantization": self.out_q}]

    def set_tensor(self, index, data):
        self.fed = data

    def invoke(self):
        pass

    def get_tensor(self, index):
        return self.out


def classic_output(rows, n=100, nc=80):
    """(1, 4+nc, N) with xywh normalized to the letterboxed input; rows = (cx, cy, w, h, class, score)."""
    out = np.zeros((1, 4 + nc, n), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(rows):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
    return out


FRAME = np.zeros((480, 640, 3), np.uint8)
# person at x 100..300, y 120..420 of the 640x480 frame -> letterbox 320x320 (scale .5, pad_y 40)
PERSON = (0.3125, 0.546875, 0.3125, 0.46875, 0, 0.9)


class YoloTests(unittest.TestCase):
    def test_letterbox_geometry(self):
        img, scale, px, py = letterbox(FRAME, (320, 320))
        self.assertEqual((img.shape, scale, px, py), ((320, 320, 3), 0.5, 0, 40))

    def test_person_mapped_back_to_original_frame(self):
        det = YoloEdgeTpu(interpreter=FakeInterp(classic_output([PERSON])))
        out = det.detect(FRAME)
        self.assertEqual(len(out), 1)
        d = out[0]
        self.assertEqual(d.label, "person")
        self.assertAlmostEqual(d.x1, 100 / 640, places=3)
        self.assertAlmostEqual(d.y1, 120 / 480, places=3)
        self.assertAlmostEqual(d.x2, 300 / 640, places=3)
        self.assertAlmostEqual(d.y2, 420 / 480, places=3)

    def test_input_is_quantized_int8_with_padding_value(self):
        fake = FakeInterp(classic_output([]))
        YoloEdgeTpu(interpreter=fake).detect(FRAME)
        self.assertEqual(fake.fed.dtype, np.int8)
        self.assertEqual(fake.fed.shape, (1, 320, 320, 3))
        self.assertEqual(int(fake.fed[0, 0, 0, 0]), 114 - 128)      # gray padding 114/255 -> -14

    def test_int8_output_is_dequantized(self):
        f = classic_output([PERSON])
        q = np.clip(np.round(f / (1 / 255.0)) - 128, -128, 127).astype(np.int8)
        d = YoloEdgeTpu(interpreter=FakeInterp(q, out_q=(1 / 255.0, -128))).detect(FRAME)
        self.assertEqual(d[0].label, "person")
        self.assertAlmostEqual(d[0].x1, 100 / 640, delta=0.01)

    def test_duplicates_removed_but_other_classes_kept(self):
        rows = [PERSON, (0.32, 0.55, 0.31, 0.47, 0, 0.7),              # duplicate person
                (0.32, 0.55, 0.31, 0.47, 46, 0.8)]                      # banana on top of the person
        labels = sorted(d.label for d in YoloEdgeTpu(interpreter=FakeInterp(classic_output(rows))).detect(FRAME))
        self.assertEqual(labels, ["banana", "person"])

    def test_end2end_layout(self):
        out = np.zeros((1, 5, 6), np.float32)
        out[0, 0] = (50 / 320, 100 / 320, 150 / 320, 250 / 320, 0.8, 46)   # xyxy normalized, conf, class
        d = YoloEdgeTpu(interpreter=FakeInterp(out)).detect(FRAME)
        self.assertEqual([x.label for x in d], ["banana"])
        self.assertAlmostEqual(d[0].y2, 420 / 480, places=3)

    def test_low_confidence_dropped(self):
        rows = [(0.3, 0.3, 0.2, 0.2, 0, 0.2)]
        self.assertEqual(YoloEdgeTpu(interpreter=FakeInterp(classic_output(rows))).detect(FRAME), [])

    def test_traffic_light_relabeled_by_lamp_color(self):
        frame = FRAME.copy()
        frame[120:420, 100:300] = (40, 40, 40)
        frame[130:200, 150:250] = (0, 0, 255)                         # red lamp (BGR)
        tl = (0.3125, 0.546875, 0.3125, 0.46875, 9, 0.9)              # class 9 = traffic light
        d = YoloEdgeTpu(interpreter=FakeInterp(classic_output([tl]))).detect(frame)
        self.assertEqual(d[0].label, "red")
        d = YoloEdgeTpu(interpreter=FakeInterp(classic_output([tl])), tl_color=False).detect(frame)
        self.assertEqual(d[0].label, "traffic_light")

    def test_lamp_colors(self):
        for bgr, want in (((0, 255, 0), "green"), ((0, 220, 255), "yellow"), ((0, 0, 255), "red")):
            crop = np.full((40, 20, 3), 30, np.uint8)
            crop[5:20, 3:17] = bgr
            self.assertEqual(traffic_light_color(crop), want)
        self.assertIsNone(traffic_light_color(np.full((40, 20, 3), 30, np.uint8)))

    def test_nms_unit(self):
        b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
        self.assertEqual(nms(b, np.array([.9, .8, .7]), np.array([0, 0, 0]), 0.5), [0, 2])

    def test_registered_and_names(self):
        self.assertEqual(len(COCO_NAMES), 80)
        self.assertEqual(COCO_NAMES[46], "banana")
        self.assertEqual(COCO_NAMES[11], "stop sign")
        self.assertIs(type(build_detector({"type": "null"})).__name__, "NullDetector")


if __name__ == "__main__":
    unittest.main()
