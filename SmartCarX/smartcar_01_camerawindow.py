# Import for the smartcar class
import smartcar

sc = smartcar.SmartCar()
sc.camera_window_loop()
# It is important for cleaning up to set the SmartCar object to None
sc = None
