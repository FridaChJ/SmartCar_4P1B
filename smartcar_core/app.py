import sys
import time
import signal
import threading
import cv2
from picarx import Picarx # Используем вашу библиотеку, которая уже есть в системе

from mjpeg_server import CameraStream, MJPEGHandler, ThreadedHTTPServer
from mqtt_client import SmartCarMQTT

class HAL:
    def __init__(self):
        self.px = Picarx()
        self.last_speed = None
        self.last_steer = None

    def apply_control(self, speed, steer):
        if speed != self.last_speed:
            if speed >= 0:
                self.px.forward(speed)
            else:
                self.px.backward(abs(speed))
            self.last_speed = speed
            
        if steer != self.last_steer:
            self.px.set_dir_servo_angle(steer)
            self.last_steer = steer

    def stop(self):
        self.apply_control(0, 0)
        self.px.stop()
        

running = True

def signal_handler(sig, frame):
    global running
    print("\n[CORE] SIGTERM/SIGINT received. Initiating graceful shutdown...")
    running = False

def camera_worker(cam_stream):
    """Выделенный поток для непрерывного чтения камеры."""
    global running
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while running:
        ret, frame = cap.read()
        if ret:
            # Быстрое кодирование JPEG для стрима
            ret_jpg, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret_jpg:
                cam_stream.set_frame(jpeg.tobytes())
            
            # TODO Stage 2: Передача frame (массив numpy) в очередь для YOLO/Lanedetect
            
        time.sleep(0.01) # Уступаем процессорное время
    cap.release()

def main():
    global running
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    hal = HAL()
    cam_stream = CameraStream()
    mqtt = SmartCarMQTT()

    # Запуск потока камеры
    cam_t = threading.Thread(target=camera_worker, args=(cam_stream,), daemon=True)
    cam_t.start()

    # Запуск MJPEG сервера
    MJPEGHandler.cam_stream = cam_stream
    httpd = ThreadedHTTPServer(('0.0.0.0', 8080), MJPEGHandler)
    server_t = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_t.start()

    # Запуск MQTT
    mqtt.start()

    print("[CORE] SmartCar system running.")
    try:
        while running:
            if mqtt.estop_active:
                hal.stop()
            else:
                hal.apply_control(mqtt.teleop_speed, mqtt.teleop_steer)
                
            time.sleep(0.05) 
    finally:
        print("[CORE] Cleaning up resources...")
        hal.stop()
        httpd.shutdown()
        mqtt.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()

