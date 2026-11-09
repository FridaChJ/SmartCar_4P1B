#!/usr/bin/env python3
"""Offline Russian voice commands -> MQTT. Run on the laptop: python3 tools/voice.py --model <vosk-model-small-ru> --host <car-ip>

Design rule: voice can only make the car SAFER or change the mode. It never sets a speed, never releases the E-STOP
and never starts driving by itself (assist/auto still need the gamepad deadman / the operator's decision).
Needs: pip install vosk sounddevice paho-mqtt     Model: https://alphacephei.com/vosk/models
"""
import argparse
import json
import queue

import paho.mqtt.client as mqtt
import sounddevice as sd
from vosk import KaldiRecognizer, Model

C = "SMARTCAR_control/"
# phrase -> (topic suffix, payload). Edit freely; words must exist in the model vocabulary.
INTENTS = {
    "стоп": ("estop", "1"), "остановись": ("estop", "1"), "хватит": ("estop", "1"),
    "ожидание": ("mode", "standby"), "ручной режим": ("mode", "manual"),
    "помощник": ("mode", "assist"), "автопилот": ("mode", "auto"),
    "медленнее": ("limit", "-"), "быстрее": ("limit", "+"),
}


def make_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        return mqtt.Client()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    args = ap.parse_args()

    c = make_client()
    if args.user:
        c.username_pw_set(args.user, args.password)
    c.connect(args.host, args.port, 30)
    c.loop_start()

    rec = KaldiRecognizer(Model(args.model), 16000, json.dumps(list(INTENTS) + ["[unk]"], ensure_ascii=False))
    audio, limit = queue.Queue(), 0.5
    with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype="int16", channels=1,
                           callback=lambda data, frames, t, status: audio.put(bytes(data))):
        print("listening:", ", ".join(INTENTS))
        while True:
            if rec.AcceptWaveform(audio.get()):
                text = json.loads(rec.Result()).get("text", "").strip()
                if text in INTENTS:
                    topic, payload = INTENTS[text]
                    if topic == "limit":
                        limit = max(0.1, min(1.0, limit + (0.1 if payload == "+" else -0.1)))
                        payload = "%.1f" % limit
                    print(text, "->", topic, payload)
                    c.publish(C + topic, payload)


if __name__ == "__main__":
    main()
