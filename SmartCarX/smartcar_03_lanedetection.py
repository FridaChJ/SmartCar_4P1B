import smartcar

sc = smartcar.SmartCar()
sc.lane_detection_loop()
# It is important for cleaning up to set the SmartCar object to None
sc = None
