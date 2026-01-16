#Code extracted from FDR
import cv2
import numpy as np
from picamera2 import Picamera2, MappedArray
import time
import math
import os
import sys
import threading
import argparse
from cv2 import aruco

# Initialize the camera
class Camera:
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.start()

    def capture_frame(self):
        frame = self.picam2.capture_array()
        return frame

# Marker detection and pose estimation
def detect_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector()
    markers, ids, _ = detector.detectMarkers(gray)
    return markers, ids

# Main program loop
if __name__ == "__main__":
    camera = Camera()

    while True:
        frame = camera.capture_frame()
        markers, ids = detect_markers(frame)

        if markers:
            for marker in markers:
                cv2.drawContours(frame, marker, -1, (0, 255, 0), 2)

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
