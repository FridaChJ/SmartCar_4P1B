"""MQTT bridge. Everything that steers the car (Node-RED, gamepad, voice) talks to it through these topics.

Control (-> car)                       Status (car ->), prefix SMARTCAR_status/
  SMARTCAR_control/mode    text          alive (retained, LWT), mode, speed, steer, campan, camtilt,
  SMARTCAR_control/drive   JSON          tlphase, distance, estop, fps, infer_ms, notes (JSON list),
  SMARTCAR_control/estop   1|0           detections (JSON list)
  SMARTCAR_control/limit   0..1 or %
  <tl_topic>               phase text
mode values: standby manual assist auto lanedetection tlviamqtt tlviacnn quit
"""
import json
import time

from .sc_math import sc_clamp

PREFIX_C, PREFIX_S = "SMARTCAR_control/", "SMARTCAR_status/"
ALIVE = PREFIX_S + "alive"
PHASES = ("red", "redyellow", "green", "yellow", "off")


def _make_client(client_id):
    import paho.mqtt.client as mqtt
    try:                                       # paho 2.x needs the callback API version
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
    except AttributeError:                     # paho 1.x
        return mqtt.Client(client_id=client_id)


class Bus:
    def __init__(self, core, cfg):
        self.core, self.cfg = core, cfg
        self.client = None

    # ---- incoming -------------------------------------------------------------------------------
    def handle(self, topic, payload, now=None):
        """Dispatch one message. Separate from the paho callback so it can be tested without a broker."""
        now = time.monotonic() if now is None else now
        core, w = self.core, self.core.world
        text = payload.decode("utf-8", "ignore").strip()
        if topic == PREFIX_C + "mode":
            core.request_mode(text.lower())
        elif topic == PREFIX_C + "drive":
            try:
                data = json.loads(text)
            except ValueError:
                return
            core.remote.set(data, now)
        elif topic == PREFIX_C + "estop":
            low = text.lower()
            if low in ("1", "true", "on", "estop"):
                w.estop = True
            elif low in ("0", "false", "off", "release"):
                w.estop = False
                core.remote.clear()                    # never resume with an old command
        elif topic == PREFIX_C + "limit":
            v = float(text)
            w.speed_limit = sc_clamp(v / 100.0 if v > 1.0 else v, 0.0, 1.0)
        elif topic == self.cfg.tl_topic:
            if text.lower() in PHASES:
                w.tl_phase = text.lower()

    def _on_message(self, client, userdata, msg):
        try:                                           # an exception here would kill paho's network thread
            self.handle(msg.topic, msg.payload)
        except Exception as exc:
            print("[MQTT] bad message on %s: %s" % (msg.topic, exc))

    def _on_connect(self, client, userdata, flags, rc):
        print("[MQTT] connected, rc=%s" % rc)
        client.publish(ALIVE, "true", qos=1, retain=True)
        for topic in ("mode", "drive", "estop", "limit"):    # subscribe here: survives reconnects
            client.subscribe(PREFIX_C + topic)
        client.subscribe(self.cfg.tl_topic)

    # ---- outgoing -------------------------------------------------------------------------------
    def publish_status(self, cmd):
        if self.client is None:
            return
        w, pub = self.core.world, self.client.publish
        pub(PREFIX_S + "mode", w.mode)
        pub(PREFIX_S + "speed", int(round(cmd.speed)))
        pub(PREFIX_S + "steer", int(round(cmd.steer)))
        pub(PREFIX_S + "campan", int(round(cmd.pan)))
        pub(PREFIX_S + "camtilt", int(round(cmd.tilt)))
        pub(PREFIX_S + "tlphase", w.tl_phase_seen)
        pub(PREFIX_S + "distance", int(round(w.distance)))
        pub(PREFIX_S + "estop", int(w.estop))
        pub(PREFIX_S + "fps", round(w.fps, 1))
        pub(PREFIX_S + "infer_ms", int(round(w.infer_ms)))
        pub(PREFIX_S + "notes", json.dumps(w.notes))
        now = time.monotonic()
        pub(PREFIX_S + "detections", json.dumps(
            [[d.label, round(d.conf, 2), round(d.x1, 3), round(d.y1, 3), round(d.x2, 3), round(d.y2, 3)]
             for d in w.fresh(now)]))

    # ---- lifecycle ------------------------------------------------------------------------------
    def start(self):
        c = _make_client("SmartCar_Core")
        if self.cfg.mqtt_user:
            c.username_pw_set(self.cfg.mqtt_user, self.cfg.mqtt_password)
        c.will_set(ALIVE, "false", qos=1, retain=True)       # also covers crashes and power loss
        c.on_connect, c.on_message = self._on_connect, self._on_message
        c.reconnect_delay_set(1, 10)
        c.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, 30)   # broker may start later
        c.loop_start()
        self.client = c

    def stop(self):
        if self.client is None:
            return
        self.client.publish(ALIVE, "false", qos=1, retain=True)
        time.sleep(0.2)                            # let the network thread flush it (no blocking wait)
        self.client.loop_stop()
        self.client.disconnect()
