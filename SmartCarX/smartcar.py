"""
SmartCar class for WIP Smart Systems lecture
version 0.9:
Integrated new lane detection algorithm from students in summer semester 2026
version 0.8:
Adapted the class for the PiCarX platform.
Removed VCC measurement as this is done by LCD display
version 0.7:
Integrated option for streaming video feed to a client by socket connection.
version 0.6:
Video stream optimized with MJPG encoder (much faster now)
Lane detection optimized using gradient function of numpy (much faster now)
version 0.5:
Image size reduced to the rows to analyze to improve image conversion performance
version 0.4:
Time measurement functions were integrated to analyze performance of software parts
version 0.3:
Debugging functionality including modification of color thresholds was added
version 0.2:
Lane detection function was integrated from existing implementation 12.09.2022.
version 0.1:
Driving functions were integrated into the new class from existing implementation 12.09.2022.
"""
from picarx import Picarx
from robot_hat import ADC
import cv2
import numpy as np
import time
import copy
from periphery import GPIO
import socket
import struct
import pickle
import select

# Debug constants for displayed video
SC_DEBUG_NONE = 0 # No debug information in image
SC_DEBUG_MASK = 1 # Show the b/w mask image
SC_DEBUG_OVERLAY = 2 # Show the masked areas inside original image

# Options for using a window
SC_USE_NO_WINDOW = 0 # This does not open a window or stream the video, to be used for dashboardcontrol
SC_USE_LOCAL_WINDOW = 1 # This will open a native window, only to be used if X server is running on remote computer
SC_USE_STREAM_WINDOW = 2 # This will open a socket server on port 50001 to stream the video to a client

def sc_clamp(value, low, high):
    """Clamp a numeric value into a closed range."""
    return max(low, min(high, value))
class SmartCar(object):
    SC_CURVE_SCORE_PX = 28.0
    SC_CURVE_ERROR_PX = 70.0
    SC_STRAIGHT_FILTER_ALPHA = 0.82
    SC_CURVE_FILTER_ALPHA = 0.45
    SC_STRAIGHT_STEP_LIMIT = 0.9
    SC_CURVE_STEP_LIMIT = 6.5
    SC_MID_STEP_LIMIT = 3.0
    SC_STRAIGHT_GAIN_FACTOR = 0.38
    SC_CURVE_GAIN_FACTOR = 1.15
    SC_AUTO_STEER_GAIN = 0.22
    SC_AUTO_MAX_STEER_DELTA = 40.0
    SC_AUTO_TARGET_DEADBAND_PX = 14.0
    SC_DEFAULT_LANE_WIDTH_FRACTION = 0.55
    SC_STEER_MIN = -35
    SC_STEER_CENTER = 0
    SC_STEER_MAX = 35
    SC_PAN_COUPLING_GAIN = 0.95
    SC_PAN_MIN = -35
    SC_PAN_CENTER = 0
    SC_PAN_MAX = 35
    # Constructor
    def __init__(self, drive_control = False,
                        steer_control = False,
                        camera_control = False,
                        recording_control = False,
                        use_ultrasonic = False,
                        # original values
                        #upper_threshold = np.array([123, 255, 255]),
                        #lower_threshold = np.array([99, 23, 65]),
                        # home of Markus
                        #upper_threshold = np.array([117, 255, 255]),
                        #lower_threshold = np.array([105, 43, 76]),
                        # office Markus - both directions
                        #upper_threshold = np.array([117, 219, 249]),
                        #lower_threshold = np.array([98, 79, 81]),
						# from students and Siavash summer 2026
                        upper_threshold = np.array([116, 219, 255]),
                        lower_threshold = np.array([96, 21, 81]),
                        debug_image = SC_DEBUG_NONE,
                        tune_thresholds = False,
                        measure_performance = False,
						use_window = SC_USE_STREAM_WINDOW):
        # Initialize PiCar
        self.px = Picarx()
        # Member variables
        self.speed = 0
        self.steer = 0
        self.campan = 0
        self.camtilt = 0
        self.showhelp = True
        self.recording = False
        self.recactivated = 0 # time in s when recording text was started to be displayed
        self.calactivated = 0 # time in s when calibrate servos text was started to be displayed
        self.drive_control = drive_control
        self.steer_control = steer_control
        self.camera_control = camera_control
        self.recording_control = recording_control
        self.use_ultrasonic = use_ultrasonic
        self.distance = 255
        self.last_distance = 255
        self.quit = False
        self.lower_threshold = lower_threshold #lower lane mask values
        self.upper_threshold = upper_threshold #upper lane mask values
        self.debug_image = debug_image # Show special image for debugging
        self.tune_thresholds = tune_thresholds # With additional keys, the color thresholds can be tuned
        self.use_window = use_window # If True, then video and kayboard inputs are tranferred via socket connection
        self.measure_performance = measure_performance # Measure time consumptions and write to file
        self.time_consumptions = {} # Dictionary containing time consumption lists
        # --- Lane controller state (Gruppe 3/8 multi-row adaptive controller) --
        self.lane_seen = False           # True while a lane target was found this frame
        self.lane_count = 0              # 2 = pair found, 1 = single mark, 0 = nothing
        self.lane_target_x = None        # current steering target in pixels, or None
        self.path_mode = "search"        # "straight" | "curve" | "search"
        self.curve_score_x = 0.0         # spread of the target between scan rows
        self.last_left_x = None          # last known left lane mark x (for single-line classification)
        self.last_right_x = None         # last known right lane mark x
        self.last_lane_width = None      # last known lane width in pixels
        self.filtered_error_x = 0.0      # EMA-filtered pixel error used by the controller
        self.auto_error_x = 0.0          # raw (unfiltered) pixel error, for debugging/telemetry
        self.auto_target_steer = 0       # steering angle the controller is moving toward
        self.conn = None
        self.calibration = False # Calibrations mode - changes calibration values for servos
        self.cal_steer = 0
        self.cal_pan = 0
        self.cal_tilt = 0
        # Create a VideoCapture object
        """
        The following resolutions are tested with the original pi cam:
        640x480 (standard)
        1920x1088 (like hd)
        The following resolutions are officially supported by the ELP camera of the smart systems lab:
        2592 (H) x 1944 (V) Pixel MJPEG 15 fps YUY2 3 fps
        2048 (H) x 1536 (V) Pixel MJPEG 15PS YUY2 3 fps
        1600 (H) x 1200 (V) Pixel MJPEG 15 fps YUY2 3 fps
        1920 (H) x 1080 (V) Pixel MJPEG 15 fps YUY2 3 fps
        1280 (H) x 1024 (V) Pixel MJPEG 15 fps YUY2 7.5 FPS
        1280 (H) x 720 (V) Pixeln MJPEG 30 fps YUY2 7.5 FPS
        1024 (H) x 768 (V) Pixeln MJPEG 30 fps YUY2 15 fps
        800 (H) x 600 (V) Pixeln MJPEG 30 fps YUY2 30 fps
        640 (H) x 480 (V) Pixel MJPEG 30 fps YUY2 30 fps
        """
        # Print additional information
        if tune_thresholds:
            print("You can use these keys to tune the color thresholds:")
            print("  Upper HSV threshold +/-")
            print("  7    8    9")
            print("  u    i    o")
            print("  Lower HSV threshold +/-")
            print("  j    k    l")
            print("  m    ,    .")
            print("Current color tresholds")
            print(f"{self.upper_threshold[0]} {self.upper_threshold[1]} {self.upper_threshold[2]}")
            print(f"{self.lower_threshold[0]} {self.lower_threshold[1]} {self.lower_threshold[2]}")
        # Create the video capture object
        self.cap = cv2.VideoCapture(1)
        #self.cap.set(3, 1024)  # Set horizontal resolution
        #self.cap.set(4, 768)  # Set vertical resolution
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Minimize the buffer size to get more up to date images
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
        # Set the encoder to have higher performance (frame rate) while reading images
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Check if camera opened successfully
        if (self.cap.isOpened() == False): 
            print("Unable to read camera feed")
        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        # Video writer initially None
        self.out = None
        # Read calibration values and apply servo values
        try:
            f = open("/home/mendel/SmartCarX/calibration.txt", "r")
            self.cal_steer = int(f.readline())
            self.cal_pan = int(f.readline())
            self.cal_tilt = int(f.readline())
            f.close()
            self.handle_servos()
        except FileNotFoundError:
            pass
        # On demand, open the socket connection (server)
        if self.use_window == SC_USE_STREAM_WINDOW:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', 5001))
            self.server_socket.listen(5)
            print("Server listening on 0.0.0.0:5001")
            print("Please start your client program on remote computer and connect to the SmartCar IP Address.")
            while True:
                self.conn, self.addr = self.server_socket.accept()
                print(f"Connection from: {self.addr}")
                #self.conn.settimeout(0.1)  # Prevent blocking - original value 0.1 which means seconds
                break

    # Destructor
    def __del__(self):
        print("Destructor called")
        # Write the time consumptions to a csv file
        if self.measure_performance:
            # Create the file
            now = time.time() # get snapshot of current time
            lt = time.localtime(now) # convert to local time struct
            filename = "time_consumption-" + time.strftime("%Y-%m-%d-%H-%M-%S", lt) + ".csv"
            time_consumption_file = open(filename, "w+")
            # Write the header
            firstkey = True
            for key in self.time_consumptions:
                if not firstkey:
                    time_consumption_file.write(";")
                firstkey = False
                time_consumption_file.write(key)
            time_consumption_file.write("\n")
            # Write the data lines
            line_index = 0
            data_found = True
            while data_found:
                data_found = False
                # Take data for each existing key
                firstkey = True
                for key in self.time_consumptions:
                    if not firstkey:
                        time_consumption_file.write(";")
                    firstkey = False
                    datastr2 = "end"
                    if len(self.time_consumptions[key]) > line_index:
                        data_found = True
                        datastr = str(self.time_consumptions[key][line_index])
                        datastr2 = datastr.replace(".", ",")
                    time_consumption_file.write(datastr2)
                time_consumption_file.write("\n")
                line_index += 1
            time_consumption_file.close()
        # When everything done, release the video capture and video write objects
        self.cap.release()
        if self.use_window == SC_USE_STREAM_WINDOW and self.conn:
            self.conn.close()
        if self.out != None:
            self.out.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        # Cleanup GPIO at the end
        #GPIO.cleanup()
        #reset the hardware
        self.px.stop()
        self.steer = 0
        self.campan = 0
        self.camtilt = 0
        self.handle_servos()

    # Adds the help text to the frame
    # cv2frame: an image as a numpy array in which the text will be drawn
    # text: The text that shall be displayed (list of strings)
    def addhelp(self):
        # Define the help text depending on available controls
        text = ["Usage of the car control:",
                "show or hide this help: ?"]
        if self.drive_control or self.steer_control or self.camera_control:
            text.append("reset actuators: x")
        text.append("quit program: q")
        if self.drive_control:
            text.append("speed up: w")
            text.append("speed down: s")
        calib_printed = False
        if self.steer_control:
            text.append("left/right: a/d")
            text.append("servo calibration: c")
            calib_printed = True
        if self.camera_control:
            text.append("camera up: t")
            text.append("camera left/right: f/h")
            text.append("camera down: g")
            if not calib_printed:
                text.append("servo calibration: c")
        if self.recording_control:
            text.append("start or stop recording: r")
        if self.use_ultrasonic:
            text.append(f"us distance: {self.distance:.1f}cm")
        calib_printed = False
        # Render the text into the frame
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,20)
        fontScale              = 0.5
        fontColor              = (180,0,0)
        lineType               = 2
        linepos = 0
        for line in text:
            bottomLeftCornerOfText = (10,20+linepos*20)
            cv2.putText(self.frame, line, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            linepos = linepos + 1

    # Adds the "blinking" recording text in red color to a frame
    # Must be called cyclically to work
    # cv2frame: an image as a numpy array in which the text will be drawn
    def addrecording(self):
        # show text for 0.5s
        if self.recactivated > time.time()-0.5:
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,self.frame_height-40)
            fontScale              = 0.5
            fontColor              = (0,0,255)
            lineType               = 2
            linepos = 0
            cv2.putText(self.frame, "recording", 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
        else:
            if self.recactivated < time.time()-1:
                self.recactivated = time.time()

    # Adds the "blinking" calibrate servos text in red color to a frame
    # Must be called cyclically to work
    # cv2frame: an image as a numpy array in which the text will be drawn
    def addcalibrating(self):
        # show text for 0.5s
        if self.calactivated > time.time()-0.5:
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,self.frame_height-60)
            fontScale              = 0.5
            fontColor              = (0,0,255)
            lineType               = 2
            linepos = 0
            cv2.putText(self.frame, "calibrate servos", 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
        else:
            if self.calactivated < time.time()-1:
                self.calactivated = time.time()

    def write_calibfile(self):
            f = open("/home/mendel/SmartCarX/calibration.txt", "w+")
            f.write(str(self.cal_steer) + "\n")
            f.write(str(self.cal_pan) + "\n")
            f.write(str(self.cal_tilt) + "\n")
            f.close()

    # Adds additional text to the window
    def add_text(self, text):
        if text != "":
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 0.5
            fontColor              = (255,100,100)
            lineType               = 2
            bottomLeftCornerOfText = (150,self.frame_height-40)
            cv2.putText(self.frame, text, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

    # Measures the ultrasonic sensor distance
    # Stores it in self.distance
    # self.distance is backed up before in self.last_distance
    def measure_us_distance(self):
        # Read distance in cm
        d = round(self.px.ultrasonic.read(), 2)
        # d=-1 means error in signal. d=-2 means timeout (max distance)
        if d <= -2:
            d = 255 # Max distance
        if d >= 0:
            # Store last measurement
            self.last_distance = self.distance
            # Store measurement
            self.distance = d

    # Read camera stream and us sensor
    def handle_sensors(self):
        self.start_timer("handle_sensors.capture_image")
        self.cap.read()
        self.ret, self.frame = self.cap.read()
        self.stop_timer("handle_sensors.capture_image")
        self.start_timer("handle_sensors.measure_us_distance")
        if self.use_ultrasonic:
            self.measure_us_distance()
        self.stop_timer("handle_sensors.measure_us_distance")
        
    # Show the window
    def handle_window(self, additional_text = ""):
        if self.ret == True:
            self.start_timer("handle_window.add_text")
            # Add the optional text to the image
            self.add_text(additional_text)
            self.stop_timer("handle_window.add_text")
            self.start_timer("handle_window.copy_frame")
            # copy the frame for showing text only in window and not in recording
            self.recordingframe = copy.copy(self.frame)
            self.stop_timer("handle_window.copy_frame")
            self.start_timer("handle_window.showhelp")
            # on demand, show the help text
            if self.showhelp and (self.use_window == SC_USE_LOCAL_WINDOW or self.use_window == SC_USE_STREAM_WINDOW):
                self.addhelp()
            self.stop_timer("handle_window.showhelp")
            self.start_timer("handle_window.addrecording")
            # on demand, show the recording text
            if self.recording:
                self.addrecording()
            if self.calibration:
                self.addcalibrating()
            self.stop_timer("handle_window.addrecording")
            self.start_timer("handle_window.imshow")
            # Display the resulting frame
            if self.use_window == SC_USE_STREAM_WINDOW:
                # Send the stream to client
                try:
                    _, buffer = cv2.imencode('.jpg', self.frame)
                    data = pickle.dumps(buffer)
                    size = struct.pack(">L", len(data))
                    self.conn.sendall(b'IMG' + size + data)  # Send frame
                except BrokenPipeError:
                    # The client is not connected any longer
                    print("Server listening on 0.0.0.0:5001")
                    print("Please start your client program on remote computer and connect to the SmartCar IP Address.")
                    while True:
                        self.conn, self.addr = self.server_socket.accept()
                        print(f"Connection from: {self.addr}")
                        #self.conn.settimeout(0.1)  # Prevent blocking - original value 0.1 which means seconds
                        break
            elif self.use_window == SC_USE_LOCAL_WINDOW:
                cv2.imshow('SmartCar camera',self.frame)
            self.stop_timer("handle_window.imshow")

    # Handle the user input
    def user_command(self):
        self.start_timer("user_command")
        key = None
        calibrate_done = False # Needed because in two lines the key "c" is handled
        if self.use_window == SC_USE_STREAM_WINDOW:
            # Use select to check if data is available (non-blocking)
            readable, _, _ = select.select([self.conn], [], [], 0.1)
            if readable:
                try:
                    key = self.conn.recv(1).decode('utf-8')  # Read 1-byte key input
                    if key:
                        print(f"Received key: {key}")  # Process key input here
                        key = ord(key)
                except:
                    pass  # Ignore small errors
        elif self.use_window == SC_USE_LOCAL_WINDOW:
            # catch the key
            key = cv2.waitKey(1) & 0xFF
        # Press q on keyboard to stop program
        if key == ord('q'):
            self.quit = True
        # driving commands
        if self.drive_control:
            if key == ord('w'):
                self.speed = self.speed + 10
                if self.speed > 100:
                    self.speed = 100
                print("speed:", self.speed)
            elif key == ord('s'):
                self.speed = self.speed - 10
                if self.speed < -100:
                    self.speed = -100
                print("speed:", self.speed)
        # steering commands
        if self.steer_control:
            if key == ord('a'):
                if self.calibration:
                    self.cal_steer -=1
                    self.write_calibfile()
                else:
                    self.steer -= 5
                if self.steer < -35:
                    self.steer = -35
                print("steer:", self.steer)
            elif key == ord('d'):
                if self.calibration:
                    self.cal_steer +=1
                    self.write_calibfile()
                else:
                    self.steer += 5
                if self.steer > 35:
                    self.steer = 35
                print("steer:", self.steer)
            elif key == ord('c'):
                self.calibration = not self.calibration
                if self.calibration:
                    self.center_servos()
                calibrate_done = True # Key already handled here, suppress for camera handling
        # camera commands
        if self.camera_control:
            if key == ord('h'):
                if self.calibration:
                    self.cal_pan +=1
                    self.write_calibfile()
                else:
                    self.campan += 5
                if self.campan > 35:
                    self.campan = 35
                print("campan:", self.campan)
            elif key == ord('f'):
                if self.calibration:
                    self.cal_pan -=1
                    self.write_calibfile()
                else:
                    self.campan -= 5
                if self.campan < -35:
                    self.campan = -35
                print("campan:", self.campan)
            elif key == ord('g'):
                if self.calibration:
                    self.cal_tilt -=1
                    self.write_calibfile()
                else:
                    self.camtilt -= 5
                if self.camtilt < -35:
                    self.camtilt = -35
                print("camtilt:", self.camtilt)
            elif key == ord('t'):
                if self.calibration:
                    self.cal_tilt +=1
                    self.write_calibfile()
                else:
                    self.camtilt += 5
                if self.camtilt > 35:
                    self.camtilt = 35
                print("camtilt:", self.camtilt)
            elif key == ord('c') and not calibrate_done:
                self.calibration = not self.calibration
                if self.calibration:
                    self.center_servos()
        if self.drive_control or self.steer_control or self.camera_control:
            if key == ord('x'):
                self.speed = 0
                self.center_servos()
        if key == ord('?'):
            self.showhelp = not self.showhelp
        if self.recording_control:
            if key == ord('r'):
                if self.recording:
                    if self.out != None:
                        self.out.release()
                else:
                    now = time.time() # get snapshot of current time
                    lt = time.localtime(now) # convert to local time struct
                    filename = time.strftime("%Y-%m-%d-%H-%M-%S", lt) + ".avi"
                    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
                    self.out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (self.frame_width,self.frame_height))
                    textactivated = time.time()
                self.recording = not self.recording
        if self.tune_thresholds:
            print_it = False
            if key == ord('7'):
                self.upper_threshold[0] = self.upper_threshold[0] + 1
                print_it = True
            elif key == ord('u'):
                self.upper_threshold[0] = self.upper_threshold[0] - 1
                print_it = True
            elif key == ord('j'):
                self.lower_threshold[0] = self.lower_threshold[0] + 1
                print_it = True
            elif key == ord('m'):
                self.lower_threshold[0] = self.lower_threshold[0] - 1
                print_it = True
            elif key == ord('8'):
                self.upper_threshold[1] = self.upper_threshold[1] + 1
                print_it = True
            elif key == ord('i'):
                self.upper_threshold[1] = self.upper_threshold[1] - 1
                print_it = True
            elif key == ord('k'):
                self.lower_threshold[1] = self.lower_threshold[1] + 1
                print_it = True
            elif key == ord(','):
                self.lower_threshold[1] = self.lower_threshold[1] - 1
                print_it = True
            elif key == ord('9'):
                self.upper_threshold[2] = self.upper_threshold[2] + 1
                print_it = True
            elif key == ord('o'):
                self.upper_threshold[2] = self.upper_threshold[2] - 1
                print_it = True
            elif key == ord('l'):
                self.lower_threshold[2] = self.lower_threshold[2] + 1
                print_it = True
            elif key == ord('.'):
                self.lower_threshold[2] = self.lower_threshold[2] - 1
                print_it = True
            if print_it:
                print(f"{self.upper_threshold[0]} {self.upper_threshold[1]} {self.upper_threshold[2]}")
                print(f"{self.lower_threshold[0]} {self.lower_threshold[1]} {self.lower_threshold[2]}")
        # Secrect commands for switching debug image modes
        if key == ord('y'):
            if self.debug_image == SC_DEBUG_NONE:
                self.debug_image = SC_DEBUG_MASK
            elif self.debug_image == SC_DEBUG_MASK:
                self.debug_image = SC_DEBUG_OVERLAY
            elif self.debug_image == SC_DEBUG_OVERLAY:
                self.debug_image = SC_DEBUG_NONE
        self.stop_timer("user_command")

    def handle_servos(self):
        self.px.set_dir_servo_angle(min(max(self.steer+self.cal_steer, -35), 35))
        self.px.set_cam_pan_angle(min(max(self.campan+self.cal_pan, -35), 35))
        self.px.set_cam_tilt_angle(min(max(self.camtilt+self.cal_tilt, -35), 35))

    def center_servos(self):
        # Center servos softly
        orig_values = [self.steer, self.campan, self.camtilt]
        steps = 100
        for i in range(steps):
            self.steer = int(orig_values[0] + (0-orig_values[0])*i/steps)
            self.campan = int(orig_values[1] + (0-orig_values[1])*i/steps)
            self.camtilt = int(orig_values[2] + (0-orig_values[2])*i/steps)
            self.handle_servos()
            #time.sleep(0.2)
        self.steer = 0
        self.campan = 0
        self.camtilt = 0
        self.handle_servos()

    # Apply the values to the PiCar actuators
    def handle_actuators(self):
        self.start_timer("handle_actuators")
        # Stop the vehicle if:
        # a) US distance is smaller than 20cm AND
        # b) Car drives forward
        # c) Measurement is trustful (no steps > 100cm)
        if self.distance < 20 and self.speed > 0 and abs(self.last_distance-self.distance) < 100 and self.use_ultrasonic:
            print(f"stopped because d = {self.distance}")
            print(f"Last distance was d = {self.last_distance}")
            self.speed = 0
        # apply the values to the hardware
        if self.speed < 0:
            self.px.backward(abs(self.speed))
            #self.gpio_p13.write(True)
        else:
            self.px.forward(self.speed)
            #self.gpio_p13.write(False)
        self.handle_servos()
        # Write the frame into the file
        if self.recording:
            self.out.write(self.recordingframe)
        self.stop_timer("handle_actuators")

    # Detects the lanes and calculates steering angle automatically
    def lane_detection(self):
        """Detect the blue lane markings and steer/pan the car automatically.

        This is the Gruppe 3/8 multi-row adaptive controller: several scan
        rows are evaluated per frame instead of stopping at the first hit,
        the path is classified as straight/curve from how much the target
        moves between rows, and the steering step size is limited to avoid
        jerky movement. Camera pan is coupled to the steering angle at the
        end of every call (Kamera an Spurhaltung verkuppelt).

        Vision always runs so the debug image / stream stays live, but the
        steering angle is only updated while the car is moving (self.speed
        != 0), mirroring the old PID controller's behaviour. When stopped,
        the last steering angle is held.
        """
        self.start_timer("lane_detection.hsv")
        original = self.frame
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_threshold, self.upper_threshold)
        self.stop_timer("lane_detection.hsv")

        self.start_timer("lane_detection.mask")
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.stop_timer("lane_detection.mask")

        self.start_timer("lane_detection.debugimage")
        if self.debug_image == SC_DEBUG_MASK:
            self.frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif self.debug_image == SC_DEBUG_OVERLAY:
            # New and wrong code:
            #overlay = np.zeros_like(original)
            #overlay[:, :, 0] = mask
            #self.frame = cv2.addWeighted(original, 0.75, overlay, 0.45, 0)
            # Old and working code:
            line_color = (0, 0, 200)
            mask_as_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.frame = np.bitwise_or(self.frame, mask_as_bgr)
        else:
            self.frame = original
        self.stop_timer("lane_detection.debugimage")

        self.start_timer("lane_detection.detect_lanes")
        height, width = self.frame.shape[:2]
        rows = [
            int(height * 0.80),
            int(height * 0.73),
            int(height * 0.66),
            int(height * 0.59),
            int(height * 0.52),
        ]

        min_pair_gap = int(width * 0.28)
        pair_targets = []
        best_marks = []
        best_row = rows[0]

        for index, row in enumerate(rows):
            row = int(sc_clamp(row, 0, height - 1))
            marks = self._extract_lane_marks(mask, row)
            cv2.line(self.frame, (0, row), (width - 1, row), (70, 70, 70), 1)
            for mark in marks:
                cv2.circle(self.frame, (mark, row), 4, (0, 255, 255), -1)

            pair = self._choose_lane_pair(marks, min_pair_gap)
            if pair:
                left, right = pair
                target = (float(left) + float(right)) / 2.0
                pair_targets.append((index, target, left, right))
                if not best_marks:
                    best_marks = marks
                    best_row = row
            elif marks and not best_marks:
                best_marks = marks
                best_row = row

        self.lane_seen = False
        self.lane_count = 0
        target = None

        if pair_targets:
            # Lower rows are close to the robot; upper rows look ahead.
            # Blend both: stable on straights, earlier response in curves.
            first_i, first_t, first_l, first_r = pair_targets[0]
            last_i, last_t, last_l, last_r = pair_targets[-1]
            avg_t = sum(t for _, t, _, _ in pair_targets) / float(len(pair_targets))
            self.curve_score_x = abs(last_t - first_t)

            if self.curve_score_x >= self.SC_CURVE_SCORE_PX:
                target = 0.45 * first_t + 0.35 * avg_t + 0.20 * last_t
                self.path_mode = "curve"
            else:
                target = 0.70 * first_t + 0.30 * avg_t
                self.path_mode = "straight"

            self.lane_seen = True
            self.lane_count = 2
            self.last_left_x = int(first_l)
            self.last_right_x = int(first_r)
            self.last_lane_width = float(max(1, first_r - first_l))
        elif best_marks:
            marks_sorted = sorted(best_marks)
            self.lane_seen = True
            self.lane_count = len(marks_sorted)
            self.path_mode = "search"
            self.curve_score_x = 0.0

            selected_x = float(marks_sorted[0])
            side = self._classify_single_line(int(selected_x), width)
            lane_width = self.last_lane_width or float(width) * self.SC_DEFAULT_LANE_WIDTH_FRACTION
            if side == "left":
                target = selected_x + lane_width / 2.0
            else:
                target = selected_x - lane_width / 2.0
            target = sc_clamp(target, 0.0, float(width - 1))
        else:
            self.path_mode = "search"
            self.curve_score_x = 0.0

        self.lane_target_x = target

        # Draw center line + chosen target line for the stream.
        center_x = int(width / 2)
        top = int(height * 0.46)
        cv2.line(self.frame, (center_x, height), (center_x, top), (255, 0, 0), 1)
        if self.lane_target_x is not None:
            cv2.line(self.frame, (int(self.lane_target_x), height), (int(self.lane_target_x), top), (0, 255, 0), 2)
        cv2.circle(self.frame, (center_x, best_row), 5, (255, 0, 0), -1)
        self.stop_timer("lane_detection.detect_lanes")

        self.start_timer("lane_detection.controller")
        # Controller shall only be active while car is driving; when stopped
        # the last steering angle (and therefore camera pan) is held.
        if self.speed != 0:
            self._lane_auto_steer(width)
        self.stop_timer("lane_detection.controller")

    def _extract_lane_marks(self, mask, row):
        """Find blue mark centers along one horizontal scan row."""
        height, width = mask.shape[:2]
        row_top = max(0, row - 1)
        row_bottom = min(height, row + 2)
        row_mask = np.max(mask[row_top:row_bottom], axis=0) > 0
        padded = np.pad(row_mask.astype(np.int8), (1, 1), mode="constant")
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        scale = width / 640.0
        min_width = max(3, int(7 * scale))
        max_width = max(min_width + 1, int(95 * scale))
        marks = []
        for start, end in zip(starts, ends):
            mark_width = end - start
            if min_width <= mark_width <= max_width:
                marks.append(int((start + end) / 2))
        return sorted(marks)

    def _choose_lane_pair(self, marks, min_pair_gap):
        """Choose the widest valid pair of lane marks in one scan row."""
        if len(marks) < 2:
            return None
        best_pair = None
        best_gap = 0
        for i in range(len(marks) - 1):
            for j in range(i + 1, len(marks)):
                gap = marks[j] - marks[i]
                if gap >= min_pair_gap and gap > best_gap:
                    best_pair = (marks[i], marks[j])
                    best_gap = gap
        return best_pair

    def _classify_single_line(self, x, width):
        """Estimate whether a single visible line is the left or right lane line."""
        if self.last_left_x is not None and self.last_right_x is not None:
            if abs(x - self.last_left_x) <= abs(x - self.last_right_x):
                return "left"
            return "right"
        if x < width / 2:
            return "left"
        return "right"

    def _lane_auto_steer(self, frame_width):
        """Fast curve response, damped straight response.

        Classifies the path as straight/curve from the multi-row lane
        target. On straights it avoids chasing tiny pixel errors; in curves
        it uses less filtering and larger allowed steering steps. Updates
        self.steer and then couples the camera pan to it.
        """
        center_x = frame_width / 2.0

        if self.lane_target_x is None or self.lane_count < 2:
            self.path_mode = "search"
            self._update_camera_pan()
            return

        error = float(self.lane_target_x) - center_x
        self.auto_error_x = error

        is_curve = (
            self.path_mode == "curve"
            or abs(error) >= self.SC_CURVE_ERROR_PX
            or self.curve_score_x >= self.SC_CURVE_SCORE_PX
        )
        alpha = self.SC_CURVE_FILTER_ALPHA if is_curve else self.SC_STRAIGHT_FILTER_ALPHA
        self.filtered_error_x = alpha * self.filtered_error_x + (1.0 - alpha) * error
        control_error = self.filtered_error_x

        if not is_curve and abs(control_error) <= self.SC_AUTO_TARGET_DEADBAND_PX:
            # Straight and close enough: do not zig-zag around the center.
            target_steer = self.SC_STEER_CENTER
            step_limit = self.SC_STRAIGHT_STEP_LIMIT
        else:
            gain_factor = self.SC_CURVE_GAIN_FACTOR if is_curve else self.SC_STRAIGHT_GAIN_FACTOR
            steer_delta = control_error * self.SC_AUTO_STEER_GAIN * gain_factor
            steer_delta = sc_clamp(steer_delta, -self.SC_AUTO_MAX_STEER_DELTA, self.SC_AUTO_MAX_STEER_DELTA)
            target_steer = sc_clamp(self.SC_STEER_CENTER + steer_delta, self.SC_STEER_MIN, self.SC_STEER_MAX)
            step_limit = self.SC_CURVE_STEP_LIMIT if is_curve else self.SC_MID_STEP_LIMIT

        self.auto_target_steer = target_steer

        requested_change = self.auto_target_steer - self.steer
        requested_change = sc_clamp(requested_change, -step_limit, step_limit)
        self.steer = sc_clamp(self.steer + requested_change, self.SC_STEER_MIN, self.SC_STEER_MAX)

        self._update_camera_pan()

    def _update_camera_pan(self):
        """Couple camera pan to the current steering angle (Kamera verkuppelt).

        Deliberately absolute, not incremental: for every steering angle
        there is exactly one matching camera-pan angle, matching the Gruppe
        3/8 implementation (gain 0.95, inverted).
        """
        steer_delta = float(self.steer) - self.SC_STEER_CENTER
        self.campan = sc_clamp(
            self.SC_PAN_CENTER + steer_delta * self.SC_PAN_COUPLING_GAIN,
            self.SC_PAN_MIN,
            self.SC_PAN_MAX,
        )

    # Starts the timer for a specified code block
    # Stores the start time in the dictionary
    def start_timer(self, blockname):
        if self.measure_performance:
            if blockname in self.time_consumptions:
                self.time_consumptions[blockname].append(time.time())
            else:
                self.time_consumptions[blockname] = [time.time()]
        
    # Stops the timer for a specified code block
    # Replaces the start time with the elapsed time in the dictionary
    def stop_timer(self, blockname):
        if self.measure_performance:
            if blockname in self.time_consumptions:
                self.time_consumptions[blockname][-1] = time.time() - self.time_consumptions[blockname][-1]
            else:
                self.time_consumptions[blockname] = ["error"]
    
    # Only shows a camera window including recording option
    def camera_window_loop(self):
        # Redefine avaiable controls
        self.drive_control = False
        self.steer_control = False
        self.camera_control = True
        self.recording_control = True
        self.use_ultrasonic = False
        # Main loop for image processing
        while(True): 
            self.handle_sensors()
            self.handle_window()
            self.user_command()
            if self.quit:
                break
            self.handle_actuators()

    # Drive the car manually while showing a camera window
    # The user can interact with the car using the keyboard:
    # See variable 'text' for key assignments
    def drive_and_record_loop(self):
        # Redefine avaiable controls
        self.drive_control = True
        self.steer_control = True
        self.camera_control = True
        self.recording_control = True
        self.use_ultrasonic = True
        while(True):
            self.handle_sensors()
            self.handle_window()
            self.user_command()
            if self.quit:
                break
            self.handle_actuators()

    # Drive the car manually while showing a camera window
    # The user can interact with the car using the keyboard:
    # See variable 'text' for key assignments
    def lane_detection_loop(self):
        # Redefine avaiable controls
        self.drive_control = True
        self.steer_control = False
        self.camera_control = True
        self.recording_control = True
        self.use_ultrasonic = True
        while(True):
            self.handle_sensors()
            self.lane_detection()
            self.user_command()
            if self.quit:
                break
            self.handle_actuators()
            self.handle_window()

