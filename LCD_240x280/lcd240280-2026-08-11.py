import spidev
import numpy as np
import time
import os
import gpiod
from PIL import Image, ImageDraw, ImageFont
import socket, subprocess, psutil
from robot_hat import ADC
import paho.mqtt.client as mqtt

# MQTT connection for sending vcc
client = mqtt.Client()
client.connect("localhost", 1883, 60)
client.loop_start()

# Update cycle (could be slower depending on needed system services)
UPDATE_CYCLE = 1.0 # seconds

# IMPORTANT NOTICE:
# The wiring used here would match like follows on the PicarX:
# RST: Pin D2 on robot hat
# DC: BSY on robot hat (chip 0, line 8)

# --- Hardware Konfiguration ---
WIDTH, HEIGHT = 280, 240
SPI_SPEED = 64000000 # Worked fine with 64000000 (max)
#SPI_SPEED = 8000000 # Worked fine with 64000000 (max)

# Pin Mapping für Coral (Chip 0 oder 1 je nach Pin)
# DC: Pin 22 -> GPIO 73 (Chip 2, Line 9 auf vielen Coral Images)
# RST: Pin 13 -> GPIO 138
# BL: Pin 15 -> GPIO 140
# CS: Pin 24 -> GPIO 74

class CoralDisplayGpiod:
    def __init__(self):
        # 1. SPI Setup
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = SPI_SPEED
        self.spi.mode = 0b11

        # 2. GPIO Setup via libgpiod
        # Wir greifen auf die Pins direkt über das System-Interface zu
        self.chip0 = gpiod.Chip('gpiochip0', gpiod.Chip.OPEN_BY_NAME)
        self.chip4 = gpiod.Chip('gpiochip4', gpiod.Chip.OPEN_BY_NAME)
        
        # Leitungen anfordern (DC: 9, CS: 10, RST: 138-128?, BL: 140-128?)
        #self.dc = self.chip4.get_line(12)
        self.dc = self.chip0.get_line(8)
        self.rst = self.chip0.get_line(6)

        # Pins als Ausgang konfigurieren
        self.dc.request(consumer="ST7789", type=gpiod.LINE_REQ_DIR_OUT)
        self.rst.request(consumer="ST7789", type=gpiod.LINE_REQ_DIR_OUT)

        #self.bl.set_value(1)
        self.init_display()

        # Font
        '''
        These fonts are available:
        DejaVuSans-Bold.ttf      DejaVuSansMono.ttf  DejaVuSerif-Bold.ttf
        DejaVuSansMono-Bold.ttf  DejaVuSans.ttf      DejaVuSerif.ttf
        '''
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
        
        # ADC for voltage measurement
        self.adc = ADC('A3')
        self.vcclist = []

    def measure_vcc(self):
        # Measurement and conversion
        u = self.adc.read()/4095*3.3*3
        self.vcclist.append(u)
        # Take mean value
        if len(self.vcclist) > 5:
            self.vcclist.pop(0)
        self.vcc = sum(self.vcclist)/len(self.vcclist)
        client.publish("SMARTCAR_status/vcc", self.vcc, retain=True)

    def send_cmd(self, cmd):
        self.dc.set_value(0)
        self.spi.writebytes([cmd])

    def send_data(self, data):
        self.dc.set_value(1)
        if isinstance(data, list):
            self.spi.writebytes(data)
        else:
            self.spi.writebytes([data])

    def init_display(self):
        print("Führe harten Reset durch...")
        # RST Pin mehrmals toggeln um sicherzugehen
        self.rst.set_value(0)
        time.sleep(0.2) # Längere Pause
        self.rst.set_value(1)
        time.sleep(0.2)
        # Important: Some ST7789 need software reset
        self.send_cmd(0x01) # SWRESET
        time.sleep(0.15) 
        self.send_cmd(0x11) # Sleep Out
        time.sleep(0.2)
        # ... Remainder of initialization (0x36, 0x3A, 0x21, 0x29)
        self.send_cmd(0x36); self.send_data(0x70) # 0x00: Normal, 0x70: 90 Grad Drehung
        self.send_cmd(0x3A); self.send_data(0x05)
        self.send_cmd(0x21)
        self.send_cmd(0x29)

    def show(self, pil_img):
            # Scale image to landscape
            img = pil_img.resize((WIDTH, HEIGHT)).convert("RGB")
            data = np.asarray(img)
            
            # RGB888 to RGB565 conversion
            r = (data[:,:,0].astype(np.uint16) & 0xF8) << 8
            g = (data[:,:,1].astype(np.uint16) & 0xFC) << 3
            b = (data[:,:,2].astype(np.uint16) >> 3)
            pixel_bytes = (r | g | b).byteswap().tobytes()

            # Offsets for ST7789 240x280 in landscape Mode:
            # Often the X-Offset is 0 and Y-Offset 20 or 35, 
            # depends on display mapping.
            x_off = 20
            y_off = 0

            # Column area (X)
            self.send_cmd(0x2A) 
            self.send_data([0, x_off, (WIDTH+x_off-1)>>8, (WIDTH+x_off-1)&0xFF])
            
            # Lines area (Y)
            self.send_cmd(0x2B) 
            self.send_data([0, y_off, (HEIGHT+y_off-1)>>8, (HEIGHT+y_off-1)&0xFF])
            
            # RAM Write
            self.send_cmd(0x2C)
            self.dc.set_value(1)
            self.spi.writebytes2(pixel_bytes)
    
    def draw_battery(self, draw: ImageDraw):
        # Draw a battery symbol at x, y and w, h with different colors (red, orange, green)
        x = 10
        y = 15
        w = 50
        h = 15
        k = 3 # Plus-Knob
        tx = 2*x+w # Text position of "x.xV"
        # Delete old area
        draw.rectangle([(x,y), (tx+4*11, y+h)], fill = "black")
        # Battery frame
        draw.polygon([(x,y), (x+w,y), (x+w,y+h//3), (x+w+k,y+h//3), (x+w+k,y+h//3*2), (x+w,y+h//3*2), (x+w,y+h), (x, y+h), (x,y)], outline = "white")
        # Voltage in percent (6...8V)
        percent = int((self.vcc - 6) / 2 * 100)
        percent = max(min(percent, 100), 0)
        # Battery fill
        wb = int(w/100*percent)
        # wb minimum 3
        if wb < 3:
            wb = 3
        # Color battery fill
        color = "red"
        if self.vcc > 6.6 and self.vcc <= 7.5:
            color = "orange"
        elif self.vcc > 7.5:
            color = "LimeGreen"
        draw.rectangle([(x+1, y+1), (x+wb-2, y+h-1)], fill = color)
        # Voltage with 2 digits after comma
        voltage = str(round(self.vcc, 2)) + "V"
        draw.text((tx, y-2), voltage, font=self.font, fill="white")

    def draw_signal(self, image: Image, draw: ImageDraw, signal):
        # Draws symbols and text showing strength of WIFI
        txt_x = 180
        txt_y = 25
        h = 15
        draw.text((txt_x, txt_y-2), signal, font=self.font, fill="white")
        # Icon
        percent = int(signal[0:-1])
        icon_name = "icons/wifi_100.jpg"
        if percent <= 20:
            icon_name = "icons/wifi_20.jpg"
        elif percent <= 40:
            icon_name = "icons/wifi_40.jpg"
        elif percent <= 60:
            icon_name = "icons/wifi_60.jpg"
        elif percent <= 80:
            icon_name = "icons/wifi_80.jpg"
        icon = Image.open(icon_name)
        iconpos = (225, 10)
        image.paste(icon, iconpos)

    def get_stats(self):
        # 1. WLAN SSID
        ssid = "N/A"
        try:
            # Trial 1: iwgetid (fast)
            #ssid = subprocess.check_output(["iwgetid", "-r"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            
            # Trial 2: If iwgetid empty, try with nmcli (NetworkManager)
            #if not ssid:
            cmd = "nmcli -t -f active,ssid,signal dev wifi | grep '^yes'"
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode("utf-8").strip()
            if output:
                parts = output.split(':')
                if len(parts) >= 3:
                    ssid = parts[1]
                    signal = f"{parts[2]}%"
        except:
            ssid = "Disconnected"        
        # 2. IP Address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except: 
            ip = "127.0.0.1"
        finally: 
            s.close()
        
        # 3. CPU Temp
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = f"{float(f.read()) / 1000.0:.1f}°C"
        except: 
            temp = "N/A"

        # --- NEUE METRIKEN ---

        # 4. Prozessor load (Interval of 0.1s for current value)
        cpu_load = f"{psutil.cpu_percent(interval=0.1)}%"

        # 5. RAM usage
        mem = psutil.virtual_memory()
        ram_usage = f"{mem.percent}%" # Alternatively: f"{mem.used // 1024**2}MB / {mem.total // 1024**2}MB"

        # 6. Free disk space (home partition /)
        disk = psutil.disk_usage('/')
        disk_free = f"{disk.free // 1024**3}GB" # In Gigabyte
        # Ermittelt den Platz dort, wo der User tatsächlich arbeitet
        disk_home = psutil.disk_usage('/home')
        disk_free_home = f"{disk_home.free / 1024**3:.1f}GB" 
        disk_usage_pct = f"{disk_home.percent}%"
        disk_free = disk_free_home + "/" + disk_usage_pct

        return {
            "ssid": ssid,
            "signal": signal,
            "ip": ip,
            "hostname": socket.gethostname(),
            "temp": temp,
            "cpu_load": cpu_load,
            "ram_usage": ram_usage,
            "disk_free": disk_free
        }

    # Checks, if USB Stick is found and reacts on content
    def show_stat_usb(self):
        # Black background
        img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        while True:
            # Start time
            start = time.time()
            # Show all info in endless loop
            stats = self.get_stats()
            # Battery voltage
            self.measure_vcc()
            # Delete screen
            draw.rectangle([0, 0, WIDTH-1, HEIGHT-1], fill="black")
            # Current computer values
            x = 10 # Passender Start für erstes Zeichen
            y = 10 # Passender Start für erste Zeile
            d = 25 # Passender Zeilenabstand
            y = y + 2*d # Statusbar freilassen
            draw.text((x, y), "WLAN: " + stats["ssid"], fill="white", font=self.font)
            y += d
            draw.text((x, y), "IP: " + stats["ip"], fill="white", font=self.font)
            # Battery
            disp.draw_battery(draw)
            # Signal strength
            disp.draw_signal(img, draw, stats["signal"])

            # Check if a USB stick is present with new WiFi connection data
            media = os.popen("ls /media/").read()
            # No media directory found - create one inside
            if media == "":
                os.popen("sudo mkdir /media/USB").read()
            # No media mounted - check if USB device plugged in
            media_usb = os.popen("ls /media/USB/").read()
            if media_usb == "":
                fdisk = os.popen("lsblk -f").read()
                # USB Sticks with letter and number assigned, example └─sda1
                pos = fdisk.find("└─sd")
                if pos > 0:
                    print(f"Device found: {fdisk[pos+2:pos+7]}")
                    letter = fdisk[pos+4:pos+7]
                    # Mount device
                    os.popen("sudo mount /dev/sd" + letter + " /media/USB").read()
                    print("Device mounted (at least tried)")
                # USB Sticks only with letter
                else:
                    pos = fdisk.find("sd")
                    if pos > 0:
                        print(f"Device found: {fdisk[pos:pos+4]}")
                        letter = fdisk[pos+2]
                        # Mount device
                        os.popen("sudo mount /dev/sd" + letter + " /media/USB").read()
                        print("Device mounted (at least tried)")
                    else:
                        print("Device /dev/sdx not found.")
            else:
                print("/media/USB has content (already mounted?)")
            # Check again if a USB stick is present with new WiFi connection data
            media_usb = os.popen("ls /media/USB/").read().split('\n')
            if "wifi.txt" in media_usb:
                disp.show(img)
                #debug_file = open("/home/mendel/debug_wifi.log", "w+")
                y += d
                draw.text((x, y), "wifi.txt found", fill="LimeGreen", font=self.font)
                wifi_file = open("/media/USB/wifi.txt", "r")
                l1 = wifi_file.readline()
                l2 = wifi_file.readline()
                wifi_file.close()
                # Do we read two lines containing text?
                if len(l1)>1 and len(l2)>1:
                    # Remove potential CRs and LFs
                    l1 = l1.replace("\n", "")
                    l1 = l1.replace("\r", "")
                    l2 = l2.replace("\n", "")
                    l2 = l2.replace("\r", "")
                #debug_file.write(l1 + " " + l2 + "\n")
                # Delete the wifi.txt file in order not to drive into an endless loop
                os.popen("sudo rm /media/USB/wifi.txt")
                y += d
                draw.text((x, y), "read and removed", fill="LimeGreen", font=self.font)
                disp.show(img)
                # List the available networks
                result = os.popen("sudo nmcli dev wifi list").read()
                time.sleep(2)
                y += d
                draw.text((x, y), l1, fill="blue", font=self.font)
                disp.show(img)
                l2_hidden = ""
                for i in range(len(l2)):
                    if i > 1 and i < len(l2)-2:
                        l2_hidden = l2_hidden + "*"
                    else:
                        l2_hidden = l2_hidden + l2[i] 
                y += d
                draw.text((x, y), l2_hidden, fill="blue", font=self.font)
                disp.show(img)
                time.sleep(2)
                if result.find(l1):
                    y += d
                    draw.text((x, y), "AP found", fill="LimeGreen", font=self.font)
                else:
                    y += d
                    draw.text((x, y), "AP not found", fill="orange", font=self.font)
                disp.show(img)
                #debug_file.write("sudo nmcli dev wifi list\n")
                # Connect to the WiFi network
                time.sleep(2)
                os.popen("sudo nmcli dev wifi connect \"" + l1 + "\" password \"" + l2 + "\" ifname wlan0")
                #debug_file.write("sudo nmcli dev wifi connect \"" + l1 + "\" password \"" + l2 + "\" ifname wlan0")
                #debug_file.close()
                draw.text((x + 140, y), "connect...", fill="yellow", font=self.font)
                disp.show(img)
                time.sleep(2)
            # No wifi.txt in USB drive
            else:
                # Remainder of stats
                y += d
                draw.text((x, y), "Hostname: " + stats["hostname"], fill="white", font=self.font)
                y += d
                draw.text((x, y), "CPU temperature: " + stats["temp"], fill="white", font=self.font)
                y += d
                draw.text((x, y), "CPU load: " + stats["cpu_load"], fill="white", font=self.font)
                y += d
                draw.text((x, y), "RAM usage: " + stats["ram_usage"], fill="white", font=self.font)
                y += d
                draw.text((x, y), "Free disk: " + stats["disk_free"], fill="white", font=self.font)
                disp.show(img)
            end = time.time()
            diff = UPDATE_CYCLE - (end - start)
            if diff > 0:
                time.sleep(diff)

if __name__ == "__main__":
    disp = CoralDisplayGpiod()
    try:
        disp.show_stat_usb()
    finally:
        client.loop_stop()
        client.disconnect()
