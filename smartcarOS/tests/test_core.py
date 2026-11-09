import http.client
import json
import time
import unittest
from types import SimpleNamespace

from smartcore.bus import Bus
from smartcore.config import Config
from smartcore.control import RemoteDrive, Safety
from smartcore.hal import Calibration, DistanceSensor, Vehicle
from smartcore.rules import load_rules, RULES
from smartcore.stream import MJPEGServer
from smartcore.types import Cmd, Detection, World
from smartcore.app import SmartCore


class FakePx:
    def __init__(self):
        self.calls = []
        self.ultrasonic = None

    def __getattr__(self, name):
        def method(*args):
            self.calls.append((name,) + args)
        return method


class FakeCam:
    def latest(self):
        return None, 0.0


def det(label, h=0.5, cx=0.5, t=0.0):
    return Detection(label, 0.9, cx - 0.1, 0.9 - h, cx + 0.1, 0.9)


def world_with(dets, now):
    w = World(0.6)
    w.detections, w.detections_ts = dets, now
    return w


class RemoteTests(unittest.TestCase):
    def test_timeout_drops_speed_keeps_steer(self):
        r = RemoteDrive(0.3)
        r.set({"speed": 50, "steer": 10}, now=10.0)
        self.assertEqual(r.get(10.1).speed, 50)
        c = r.get(10.5)
        self.assertEqual((c.speed, c.steer), (0.0, 10))

    def test_garbage_is_rejected_and_values_clamped(self):
        r = RemoteDrive()
        self.assertFalse(r.set([1, 2], 0))
        self.assertFalse(r.set({"speed": "fast"}, 0))
        self.assertFalse(r.set({"speed": float("nan")}, 0))
        self.assertTrue(r.set({"speed": 500, "steer": -99}, 0))
        c = r.get(0)
        self.assertEqual((c.speed, c.steer), (100, -35))


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.s = Safety(Config(max_speed=60, accel_per_tick=5))
        self.w = World()

    def test_estop_zero(self):
        self.w.estop = True
        self.assertEqual(self.s.apply(Cmd(50), self.w).speed, 0)

    def test_obstacle_blocks_forward_not_reverse(self):
        self.w.obstacle = True
        self.assertEqual(self.s.apply(Cmd(40), self.w).speed, 0)
        self.assertLess(self.s.apply(Cmd(-40), self.w).speed, 0)

    def test_ramp_up_and_instant_brake(self):
        speeds = [self.s.apply(Cmd(40), self.w).speed for _ in range(3)]
        self.assertEqual(speeds, [5, 10, 15])
        self.assertEqual(self.s.apply(Cmd(0), self.w).speed, 0)

    def test_cap_and_limit(self):
        self.s.prev = 100
        self.assertEqual(self.s.apply(Cmd(100), self.w).speed, 60)
        self.w.speed_limit = 0.5
        self.assertEqual(self.s.apply(Cmd(100), self.w).speed, 30)

    def test_nan_becomes_stop(self):
        self.assertEqual(self.s.apply(Cmd(float("nan")), self.w).speed, 0)


class RuleTests(unittest.TestCase):
    def rule(self, name, params=None):
        return RULES[name](params or {})

    def test_person_stop_holds(self):
        r, c = self.rule("person_stop"), Cmd(40)
        w = world_with([det("person", 0.6)], 1.0)
        r.apply(c, w, 1.0)
        self.assertEqual(c.speed, 0)
        c = Cmd(40)
        r.apply(c, world_with([], 1.0), 1.5)       # gone, but held 1 s
        self.assertEqual(c.speed, 0)
        c = Cmd(40)
        r.apply(c, world_with([], 2.5), 2.5)
        self.assertEqual(c.speed, 40)

    def test_person_far_away_ignored(self):
        c = Cmd(40)
        self.rule("person_stop").apply(c, world_with([det("person", 0.2)], 1.0), 1.0)
        self.assertEqual(c.speed, 40)

    def test_stale_detections_are_ignored(self):
        c = Cmd(40)
        self.rule("person_stop").apply(c, world_with([det("person", 0.8)], 1.0), 5.0)
        self.assertEqual(c.speed, 40)

    def test_speed_sign_latches_expires_and_end_sign(self):
        r = self.rule("speed_signs", {"limits": {"s30": 30, "end": None}, "hold_s": 10})
        c = Cmd(50)
        r.apply(c, world_with([det("s30", 0.2)], 0.0), 0.0)
        self.assertEqual(c.speed, 30)
        c = Cmd(50)
        r.apply(c, world_with([], 0.0), 5.0)       # sign left the view, limit sticks
        self.assertEqual(c.speed, 30)
        c = Cmd(50)
        r.apply(c, world_with([det("end", 0.2)], 6.0), 6.0)
        self.assertEqual(c.speed, 50)
        r.apply(Cmd(50), world_with([det("s30", 0.2)], 7.0), 7.0)
        c = Cmd(50)
        r.apply(c, world_with([], 7.0), 20.0)      # expired
        self.assertEqual(c.speed, 50)

    def test_stop_sign_waits_once_then_rearms(self):
        r = self.rule("sign_stops", {"signs": {"stop": {"mode": "always", "wait_s": 2}}})
        sign = [det("stop", 0.3)]
        c = Cmd(30)
        r.apply(c, world_with(sign, 0.0), 0.0)
        self.assertEqual(c.speed, 0)
        c = Cmd(30)
        r.apply(c, world_with(sign, 2.5), 2.5)     # waited, sign still visible: go
        self.assertEqual(c.speed, 30)
        r.apply(Cmd(30), world_with([], 3.0), 8.0)   # out of sight long enough: re-armed
        c = Cmd(30)
        r.apply(c, world_with(sign, 9.0), 9.0)
        self.assertEqual(c.speed, 0)

    def test_crosswalk_stops_only_with_person(self):
        r = self.rule("sign_stops", {"signs": {"crosswalk": {"mode": "if_person"}}})
        c = Cmd(30)
        r.apply(c, world_with([det("crosswalk", 0.3)], 0.0), 0.0)
        self.assertEqual(c.speed, 30)
        c = Cmd(30)
        r.apply(c, world_with([det("crosswalk", 0.3), det("person", 0.3, 0.8)], 0.5), 0.5)
        self.assertEqual(c.speed, 0)
        c = Cmd(30)
        r.apply(c, world_with([det("person", 0.3)], 1.0), 1.0)   # sign just left the view, person still there
        self.assertEqual(c.speed, 0)

    def test_traffic_light_mqtt_and_detection_hold(self):
        r, w = self.rule("traffic_light"), World()
        w.tl_source, w.tl_phase = "mqtt", "red"
        c = Cmd(40)
        r.apply(c, w, 0.0)
        self.assertEqual(c.speed, 0)
        w.tl_phase = "yellow"
        c = Cmd(40)
        r.apply(c, w, 1.0)
        self.assertEqual(c.speed, 30)
        r2, w2 = self.rule("traffic_light"), world_with([det("red", 0.3)], 0.0)
        w2.tl_source = "detections"
        r2.apply(Cmd(40), w2, 0.0)
        w2.detections, w2.detections_ts = [], 0.5   # one frame without red: still held
        c = Cmd(40)
        r2.apply(c, w2, 0.5)
        self.assertEqual(c.speed, 0)
        w2.detections, w2.detections_ts = [det("green", 0.3)], 0.6
        c = Cmd(40)
        r2.apply(c, w2, 0.6)
        self.assertEqual(c.speed, 40)

    def test_avoid_steers_away_from_banana(self):
        r = self.rule("avoid", {"slew_deg": 100})
        c = Cmd(35, 0)
        r.apply(c, world_with([det("banana", 0.4, cx=0.7)], 0.0), 0.0)
        self.assertLess(c.steer, 0)                  # banana on the right -> steer left
        self.assertLessEqual(c.speed, 20)
        r2 = self.rule("avoid", {"slew_deg": 100})
        c = Cmd(35, 0)
        r2.apply(c, world_with([det("banana", 0.4, cx=0.3)], 0.0), 0.0)
        self.assertGreater(c.steer, 0)

    def test_default_rules_json_loads(self):
        self.assertEqual(len(load_rules("rules.json")), 5)


class HalTests(unittest.TestCase):
    def test_writes_only_on_change(self):
        px = FakePx()
        v = Vehicle(px, Calibration("/nonexistent"))
        v.apply(Cmd(30, 5))
        n = len(px.calls)
        v.apply(Cmd(30, 5))
        self.assertEqual(len(px.calls), n)
        v.apply(Cmd(30, 6))
        self.assertEqual(px.calls[n:], [("set_dir_servo_angle", 6)])
        v.apply(Cmd(-20, 6))
        self.assertEqual(px.calls[-1], ("backward", 20))

    def test_distance_glitch_filter(self):
        m = SimpleNamespace(read=lambda: 10.0)
        s = DistanceSensor(m)
        s.measure()
        self.assertFalse(s.obstacle_ahead())        # 255 -> 10 is a suspicious jump
        s.measure()
        self.assertTrue(s.obstacle_ahead())


class BusTests(unittest.TestCase):
    def setUp(self):
        cfg = Config()
        self.core = SmartCore(cfg, Vehicle(FakePx(), Calibration("/x")), FakeCam(), [])
        self.bus = Bus(self.core, cfg)

    def test_messages(self):
        h = self.bus.handle
        h("SMARTCAR_control/drive", b'{"speed": 20, "steer": 3}', now=1.0)
        self.assertEqual(self.core.remote.get(1.0).speed, 20)
        h("SMARTCAR_control/estop", b"1")
        self.assertTrue(self.core.world.estop)
        h("SMARTCAR_control/estop", b"0")
        self.assertFalse(self.core.world.estop)
        self.assertEqual(self.core.remote.get(1.0).steer, 0)   # command cleared on release
        h("SMARTCAR_control/limit", b"50")
        self.assertEqual(self.core.world.speed_limit, 0.5)
        h("TL08", b"Red")
        self.assertEqual(self.core.world.tl_phase, "red")
        h("SMARTCAR_control/drive", b"not json")               # must not raise

    def test_bad_message_never_raises_in_callback(self):
        msg = SimpleNamespace(topic="SMARTCAR_control/limit", payload=b"abc")
        self.bus._on_message(None, None, msg)


class CoreTests(unittest.TestCase):
    def make(self, rules=()):
        self.px = FakePx()
        self.core = SmartCore(Config(), Vehicle(self.px, Calibration("/x")), FakeCam(), list(rules))
        return self.core

    def test_manual_drive_and_deadman(self):
        c = self.make()
        c.set_mode("manual", 0.0)
        c.remote.set({"speed": 100, "steer": 10}, 0.0)
        for i in range(20):                          # ramp up to the 60 % cap
            c.tick(0.01 * i)
        self.assertEqual(c.last_cmd.speed, 60)
        c.tick(1.0)                                  # no command for > 0.3 s
        self.assertEqual(c.last_cmd.speed, 0)

    def test_estop_forces_standby_and_stops(self):
        c = self.make()
        c.set_mode("manual", 0.0)
        c.remote.set({"speed": 30}, 0.0)
        c.tick(0.0)
        c.world.estop = True
        c.tick(0.05)
        self.assertEqual((c.mode.name, c.last_cmd.speed), ("standby", 0))
        c.world.estop = False
        c.remote.set({"speed": 30}, 0.1)
        c.tick(0.1)
        self.assertEqual(c.last_cmd.speed, 0)        # still standby: no resume

    def test_legacy_mode_names(self):
        c = self.make()
        c.set_mode("tlviamqtt", 0.0)
        self.assertEqual((c.mode.name, c.world.tl_source), ("auto", "mqtt"))
        c.set_mode("bogus", 0.0)
        self.assertEqual(c.mode.name, "auto")
        c.request_mode("quit")
        c.tick(0.0)
        self.assertFalse(c.running)

    def test_broken_rule_stops_the_car(self):
        class Boom:
            name, modes = "boom", {"manual"}

            def apply(self, cmd, world, now):
                raise RuntimeError("x")
        c = self.make([Boom()])
        c.set_mode("manual", 0.0)
        c.remote.set({"speed": 50}, 0.0)
        c.tick(0.0)
        self.assertEqual(c.last_cmd.speed, 0)

    def test_rules_apply_in_mode_only(self):
        c = self.make(load_rules("rules.json"))
        c.set_mode("manual", 0.0)
        c.world.detections, c.world.detections_ts = [det("person", 0.8)], 0.0
        c.remote.set({"speed": 50}, 0.0)
        c.tick(0.0)
        self.assertEqual(c.last_cmd.speed, 0)        # person_stop is active in manual mode too


class StreamTests(unittest.TestCase):
    def test_stream_serves_frames_and_stops_fast(self):
        s = MJPEGServer(0, "127.0.0.1")
        s.start()
        s.publish(b"\xff\xd8JPEGDATA\xff\xd9")
        conn = http.client.HTTPConnection("127.0.0.1", s.port, timeout=3)
        conn.request("GET", "/stream.mjpg")
        resp = conn.getresponse()
        self.assertIn("multipart/x-mixed-replace", resp.getheader("Content-Type"))
        data = resp.read1(4096)
        self.assertIn(b"JPEGDATA", data)
        self.assertEqual(s.clients, 1)
        t0 = time.time()
        s.stop()                                     # with a viewer still connected
        self.assertLess(time.time() - t0, 2.0)
        conn.close()


if __name__ == "__main__":
    unittest.main()
