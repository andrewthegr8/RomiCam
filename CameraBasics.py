from picamera2 import Picamera2
import time

picam2 = Picamera2()

config = picam2.create_still_configuration(
    main={"size": (4608, 2592)}
)

picam2.configure(config)
#picam2.set_controls({"AfMode": 0})  # manual/infinity
picam2.start()

time.sleep(1)  # allow exposure + autofocus to settle
while True:
    filename = input('Enter filename: ')
    picam2.capture_file(filename)
