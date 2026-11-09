"""Driving rules ("logic files"). A rule may cap speed, stop, or bias steering; parameters live in rules.json.

Add your own: subclass Rule, decorate with @rule("name"), list the name in rules.json "order"
(modules with extra rules go into rules.json "modules"). Rules run in order, then Safety runs last.
A rule that raises an exception stops the car (fail-safe), it never lets it drive on blind.
"""
import importlib
import json

from .sc_math import sc_clamp

RULES = {}
RED = ("red", "redyellow")


def rule(name):
    def deco(cls):
        cls.name = name
        RULES[name] = cls
        return cls
    return deco


class Rule(object):
    name = ""
    default_modes = ("assist", "auto")      # modes in which the rule is active (override with "modes")

    def __init__(self, params=None):
        self.p = params or {}
        self.modes = set(self.p.get("modes", self.default_modes))

    def apply(self, cmd, world, now):
        raise NotImplementedError

    def note(self, world, text):
        world.notes.append("%s: %s" % (self.name, text))

    @staticmethod
    def seen(world, now, labels, min_h=0.0, lo=0.0, hi=1.0):
        """Fresh detections with one of the labels, at least min_h of the frame tall, centered in lo..hi."""
        return [d for d in world.fresh(now) if d.label in labels and d.h >= min_h and lo <= d.cx <= hi]


@rule("person_stop")
class PersonStop(Rule):
    """A person close ahead (tall in the frame, near the center) -> no forward motion."""
    default_modes = ("manual", "assist", "auto")

    def __init__(self, params=None):
        super().__init__(params)
        self.until = -1e9

    def apply(self, cmd, world, now):
        p = self.p
        if self.seen(world, now, p.get("labels", ["person"]), p.get("min_height", 0.5),
                     p.get("corridor_min", 0.15), p.get("corridor_max", 0.85)):
            self.until = now + p.get("hold_s", 1.0)
        if now < self.until and cmd.speed > 0:
            cmd.speed = 0.0
            self.note(world, "person ahead")


@rule("speed_signs")
class SpeedSigns(Rule):
    """Signs that limit the speed. "limits": label -> max speed %, null = end of limit. Sticks for hold_s."""

    def __init__(self, params=None):
        super().__init__(params)
        self.cap, self.since = None, -1e9

    def apply(self, cmd, world, now):
        limits = self.p.get("limits", {})
        hits = self.seen(world, now, list(limits), self.p.get("min_height", 0.08))
        if hits:
            nearest = max(hits, key=lambda d: d.h)
            self.cap, self.since = limits[nearest.label], now
        if self.cap is not None and now - self.since > self.p.get("hold_s", 20.0):
            self.cap = None
        if self.cap is not None and cmd.speed > self.cap:
            cmd.speed = self.cap
            self.note(world, "limit %s" % self.cap)


@rule("sign_stops")
class SignStops(Rule):
    """Signs that require stopping. mode "always": stop wait_s once per sign. mode "if_person": stop while a person is near."""

    def __init__(self, params=None):
        super().__init__(params)
        self.state = {}

    def apply(self, cmd, world, now):
        p = self.p
        signs = p.get("signs", {})
        persons = self.seen(world, now, p.get("person_labels", ["person"]), p.get("person_min_height", 0.15))
        for d in self.seen(world, now, list(signs), p.get("min_height", 0.10)):
            st = self.state.setdefault(d.label, {"armed": True, "stop_until": -1e9, "last_seen": -1e9})
            if signs[d.label].get("mode", "always") == "always" and st["armed"]:
                st["stop_until"] = now + signs[d.label].get("wait_s", 2.0)
                st["armed"] = False
            st["last_seen"] = now
        reason = None
        for label, st in self.state.items():
            if now - st["last_seen"] > p.get("rearm_s", 3.0):
                st["armed"] = True
            if now < st["stop_until"]:
                reason = "%s (waiting)" % label
            elif (signs[label].get("mode", "always") == "if_person" and persons
                  and now - st["last_seen"] <= p.get("sign_hold_s", 3.0)):
                reason = "%s + person" % label
        if reason and cmd.speed > 0:
            cmd.speed = 0.0
            self.note(world, reason)


@rule("traffic_light")
class TrafficLight(Rule):
    """Red = stop, yellow = slow, green = go. Source: MQTT phase (TL08) or detections, set by the mode name
    (tlviamqtt / tlviacnn) or by "source" in rules.json. Detected red is held hold_s to ride out flicker."""
    default_modes = ("auto",)

    def __init__(self, params=None):
        super().__init__(params)
        self.red_until = -1e9

    def phase(self, world, now, src):
        if src == "mqtt":
            return world.tl_phase
        labels = self.p.get("labels", {"red": "red", "green": "green", "yellow": "yellow", "redyellow": "redyellow"})
        hits = self.seen(world, now, list(labels), self.p.get("min_height", 0.0))
        return labels[max(hits, key=lambda d: d.h).label] if hits else None

    def apply(self, cmd, world, now):
        src = world.tl_source or self.p.get("source")
        if not src:
            world.tl_phase_seen = "off"
            return
        ph = self.phase(world, now, src)
        world.tl_phase_seen = ph or "off"
        if src == "mqtt":
            stop = ph in RED
        else:
            if ph in RED:
                self.red_until = now + self.p.get("hold_s", 1.0)
            elif ph == "green":
                self.red_until = -1e9
            stop = now < self.red_until
        if stop:
            if cmd.speed > 0:
                cmd.speed = 0.0
                self.note(world, "red light")
        elif ph == "yellow":
            cap = self.p.get("yellow_speed", 30)
            if cmd.speed > cap:
                cmd.speed = cap
                self.note(world, "yellow light")


@rule("avoid")
class Avoid(Rule):
    """Swerve around objects (e.g. banana) close ahead: steer away from them and slow down.
    Open-loop bias on top of the lane steering; tune bias_deg / hold_s on the track."""

    def __init__(self, params=None):
        super().__init__(params)
        self.bias, self.side, self.release_at = 0.0, 1, -1e9

    def apply(self, cmd, world, now):
        p = self.p
        near = self.seen(world, now, p.get("labels", ["banana"]), p.get("min_height", 0.2),
                         p.get("corridor_min", 0.2), p.get("corridor_max", 0.8))
        if near:
            d = max(near, key=lambda x: x.h)
            if abs(d.cx - 0.5) > 0.05:                 # object dead ahead: keep the side chosen before
                self.side = -1 if d.cx > 0.5 else 1    # object right -> steer left (negative)
            self.release_at = now + p.get("hold_s", 1.2)
        target = self.side * p.get("bias_deg", 18.0) if now < self.release_at else 0.0
        slew = p.get("slew_deg", 4.0)
        self.bias += sc_clamp(target - self.bias, -slew, slew)
        if abs(self.bias) > 1.0:
            cmd.steer = sc_clamp(cmd.steer + self.bias, -35, 35)
            cap = p.get("slow_speed", 20)
            if cmd.speed > cap:
                cmd.speed = cap
            self.note(world, "avoiding")


def load_rules(path):
    with open(path) as fh:
        data = json.load(fh)
    for module in data.get("modules", []):
        importlib.import_module(module)
    rules = []
    for name in data.get("order", []):
        if name not in RULES:
            raise ValueError("unknown rule %r (known: %s)" % (name, ", ".join(sorted(RULES))))
        rules.append(RULES[name](data.get(name, {})))
    return rules
