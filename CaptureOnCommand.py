from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()

#Should be hi-res for capure, low-res forpreview
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")

# Configure camera
picam2.configure(camera_config)

# Start the camera and preview (use Preview.QTGL for remote desktop compatibility)
picam2.start_preview(Preview.QTGL)
picam2.start()

#picam2.set_controls({"AfMode": 0})  # manual/infinity

time.sleep(2)  # allow exposure + autofocus to settle
picnum = 0
while picnum < 10:
    filename = './calibrationpics/720p' + str(picnum) + '.jpg'
    i = input('Hit enter to take picture')
    picam2.capture_file(filename)
    picnum += 1
    print(f'Captured: {filename}')
    time.sleep(5)

