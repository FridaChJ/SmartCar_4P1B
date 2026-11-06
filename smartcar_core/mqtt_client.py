import json
import paho.mqtt.client as mqtt

class SmartCarMQTT:
    def __init__(self, broker="localhost", port=1883):
        self.client = mqtt.Client(client_id="SmartCar_Core")
        self.broker = broker
        self.port = port
        
        self.teleop_speed = 0.0
        self.teleop_steer = 0.0
        self.estop_active = False

        self.client.will_set("SMARTCAR_status/alive", payload="false", qos=2, retain=True)
        
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        print(f"[MQTT] Connected with result code {rc}")
        self.client.publish("SMARTCAR_status/alive", payload="true", qos=2, retain=True)
        self.client.subscribe("SMARTCAR_control/drive")
        self.client.subscribe("SMARTCAR_control/estop")

    def on_message(self, client, userdata, msg):
        if msg.topic == "SMARTCAR_control/drive":
            try:
                data = json.loads(msg.payload.decode())
                self.teleop_speed = data.get("speed", 0)
                self.teleop_steer = data.get("steer", 0)
            except json.JSONDecodeError:
                pass
        elif msg.topic == "SMARTCAR_control/estop":
            self.estop_active = msg.payload.decode().lower() == "true"
            if self.estop_active:
                print("[MQTT] E-STOP ACTIVATED!")

    def start(self):
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()

    def stop(self):
        self.client.publish("SMARTCAR_status/alive", payload="false", qos=2, retain=True)
        self.client.loop_stop()
        self.client.disconnect()
