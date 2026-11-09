# smartcore — SmartCar core

Deploy: copy `smartcore/`, `rules.json`, `config.json` into `/home/mendel/SmartCarX/` (next to calibration.txt),
then `python3 -m smartcore` (test) or install `smartcar.service`. Tests: `python3 -m unittest discover -s tests -t .`

## Pipeline (app.py: SmartCore.tick, 20 Hz)
mode.step (who drives) -> rules (slow / stop / steer, rules.json) -> Safety (estop, ultrasonic, caps, ramp) -> Vehicle (write on change)

Threads: camera, detector worker, ultrasonic, stream viewer, MQTT. They only write to `World`; the main loop reads it.

## Where new things plug in
| Feature | Where |
|---|---|
| YOLO | class `Detector` in perception.py returning normalized `Detection`s, register in `DETECTORS`, select with `"detector"` in config.json |
| Signs / banana / person logic | rules.json parameters, or a new `@rule("name")` class (rules.py or a module listed under "modules") |
| New mode (follow-me, ...) | subclass `Mode` in control.py, add to `MODES` |
| Gamepad / voice / Node-RED | MQTT only (see bus.py header); tools/teleop.py, tools/voice.py |

## Safety rules of the design
* Drive commands expire after `remote_timeout_s` (deadman). Drive messages are never retained.
* E-STOP latches, forces standby, release does not resume driving.
* Detections older than `detect_max_age_s` count as unknown, never as "all clear".
* A rule that raises stops the car. Voice can only stop / change mode / lower the limit.

## YOLO on the Edge TPU (`"type": "yolo_edgetpu"`)
1. On an x86_64 Linux PC (Ultralytics refuses Edge TPU export on aarch64 / the car):
   `yolo export model=yolo11n.pt format=edgetpu imgsz=320`  -> `*_edgetpu.tflite`
2. Copy the file to the car, then in config.json:
   `"detector": {"type": "yolo_edgetpu", "model": "yolo11n_edgetpu.tflite", "threshold": 0.4}`
3. COCO classes work out of the box (person, banana, traffic_light, stop_sign, ...). Own signs: fine-tune, export, pass
   `"labels": "labels.txt"` (one name per line, line number = class id) and use those names in rules.json.
4. Traffic lights: COCO "traffic_light" has no lamp color, so the backend relabels it to red / yellow / green from the crop (tl_color).
5. If invoke() complains about the runtime version, re-compile with an older `edgetpu_compiler -m <min_runtime_version>`.
   Check the compiler log: ops that did not map to the Edge TPU run on the car's CPU and cost FPS.
Pre/post-processing (letterbox, dequantize, decode, NMS) measured at ~1.6 ms (320 px) on a PC; expect several times more on the car.

## Node-RED (nodered/flows.json = all three old tabs, patched)
* Video: tcp/base64 chain removed, the Video widget now shows http://<car>:8080/stream.mjpg (host taken from the page URL).
* START/STOP use systemctl (no double start). STOP first sends `quit` (graceful), after 3 s `systemctl stop` as fallback.
* New: MANUAL / ASSIST / AUTO, E-STOP (+ release), mode and E-STOP indicators, "Service command" result.
* SHUTDOWN now needs a second click (CONFIRM SHUTDOWN) within 5 s.
* New "Vision" group: detections, rule/safety notes, distance, loop rate, inference time.
* Traffic light tab is unchanged. The campan/camtilt sliders are still display-only.
Import: Menu -> Import -> select file -> "Delete flow tabs"/replace, then Deploy. Back up the old flows first.

## Install on the car
`sh deploy.sh`, then `sudo systemctl start smartcar`. Logs: `journalctl -u smartcar -f`.
Gamepad/voice from a laptop: the broker must listen on the network (use a password): `tools/teleop.py --host <car> --port <port> --user .. --password ..`.
