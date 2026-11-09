#!/bin/sh
# Run on the car from the unzipped folder:  sh deploy.sh
# Code is always updated; config.json and rules.json are copied only if missing (your tuning is kept).
set -e
DEST=/home/mendel/SmartCarX
python3 -c "import paho.mqtt.client, cv2, numpy" || { echo "missing: pip3 install paho-mqtt numpy opencv-python"; exit 1; }
mkdir -p "$DEST"
rm -rf "$DEST/smartcore" && cp -r smartcore "$DEST/smartcore"
[ -f "$DEST/config.json" ] || cp config.json "$DEST/config.json"
[ -f "$DEST/rules.json" ]  || cp rules.json  "$DEST/rules.json"
sudo cp smartcar.service /etc/systemd/system/smartcar.service
sudo systemctl daemon-reload
sudo systemctl enable smartcar
echo "installed. start: sudo systemctl start smartcar   logs: journalctl -u smartcar -f"
echo "Node-RED: import nodered/flows.json (Import -> replace all flows), then Deploy."
