#!/usr/bin/env python3
"""DualSense -> MQTT. Run on the laptop (python3 tools/teleop.py --host <car-ip>), pad via USB or Bluetooth.

Left stick X steer | R2 forward, L2 reverse | right stick camera pan/tilt | L1 = DEADMAN (hold to drive)
Cross standby | Square manual | Triangle assist | Create auto | Circle E-STOP | Options release E-STOP
D-pad up/down speed limit +-10 %.   First run:  tools/teleop.py --probe  to check the axis/button numbers.
Needs: pip install pygame paho-mqtt
"""
import argparse
import json
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame                                    # noqa: E402
import paho.mqtt.client as mqtt                  # noqa: E402

C = "SMARTCAR_control/"
AX = dict(lx=0, ly=1, rx=2, ry=3, l2=4, r2=5)    # SDL2 layout for DualSense, verify with --probe
BTN = dict(cross=0, circle=1, square=2, triangle=3, create=4, ps=5, options=6, l1=9, r1=10, up=11, down=12)
DEADZONE = 0.12


def dz(v):
    return 0.0 if abs(v) < DEADZONE else v


def make_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        return mqtt.Client()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--limit", type=float, default=0.5, help="start speed limit 0..1")
    ap.add_argument("--probe", action="store_true", help="print raw axes/buttons and exit with Ctrl+C")
    args = ap.parse_args()

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("no gamepad found")
    js = pygame.joystick.Joystick(0)
    js.init()
    print("pad:", js.get_name(), "axes", js.get_numaxes(), "buttons", js.get_numbuttons())
    if args.probe:
        while True:
            pygame.event.pump()
            print("axes", [round(js.get_axis(i), 2) for i in range(js.get_numaxes())],
                  "buttons", [i for i in range(js.get_numbuttons()) if js.get_button(i)])
            time.sleep(0.2)

    c = make_client()
    if args.user:
        c.username_pw_set(args.user, args.password)
    c.will_set(C + "drive", json.dumps({"speed": 0, "steer": 0, "pan": 0, "tilt": 0}))   # pad/PC died -> stop
    c.connect(args.host, args.port, 30)
    c.loop_start()

    limit, armed, prev = args.limit, set(), {}
    mode_buttons = {"cross": "standby", "square": "manual", "triangle": "assist", "create": "auto"}
    try:
        while True:
            pygame.event.pump()
            for name in ("l2", "r2"):                # triggers may report 0 until first use: ignore until seen at rest (-1)
                if js.get_axis(AX[name]) < -0.9:
                    armed.add(name)
            trig = {n: ((js.get_axis(AX[n]) + 1) / 2 if n in armed else 0.0) for n in ("l2", "r2")}
            deadman = js.get_button(BTN["l1"])
            speed = (trig["r2"] - trig["l2"]) * 100 * limit if deadman else 0.0
            c.publish(C + "drive", json.dumps({
                "speed": round(speed, 1),
                "steer": round(dz(js.get_axis(AX["lx"])) * 35, 1),
                "pan": round(dz(js.get_axis(AX["rx"])) * 35, 1),
                "tilt": round(-dz(js.get_axis(AX["ry"])) * 35, 1)}), qos=0)       # never retained
            now = {n: js.get_button(b) for n, b in BTN.items()}
            for n, pressed in now.items():           # act on button edges only
                if pressed and not prev.get(n):
                    if n in mode_buttons:
                        c.publish(C + "mode", mode_buttons[n])
                    elif n == "circle":
                        c.publish(C + "estop", "1")
                    elif n == "options":
                        c.publish(C + "estop", "0")
                    elif n in ("up", "down"):
                        limit = max(0.1, min(1.0, limit + (0.1 if n == "up" else -0.1)))
                        print("limit %d %%" % round(limit * 100))
            prev = now
            time.sleep(1 / 30)
    finally:
        c.publish(C + "drive", json.dumps({"speed": 0, "steer": 0, "pan": 0, "tilt": 0}))
        time.sleep(0.1)
        c.loop_stop()


if __name__ == "__main__":
    main()
