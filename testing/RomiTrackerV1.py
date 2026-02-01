#Initialization code
import cv2
from cv2 import aruco
import json
import numpy as np
import sys
from picamera2 import Picamera2
import time


#Read K and D from JSON files
with open('K.json', 'r') as f:
    data = json.load(f)
K = np.array(data)

with open('D.json', 'r') as f:
    data = json.load(f)
D = np.array(data)

print(f'Loaded K and D from JSON files.')

#Init Camera
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (4608, 2592)})
picam2.configure(config)
picam2.start()
print('Initializing Camera...')
time.sleep(3)  # allow exposure + autofocus to settle

scaledown_factor = 1 #Scale down factor for faster processing

#Capture a test image to get dimensions
original_img = picam2.capture_array()
img = cv2.resize(original_img, None, fx=scaledown_factor, fy=scaledown_factor, interpolation=cv2.INTER_AREA)
h, w = img.shape[:2]
dim = (w, h)
print(f'Captured image with dimensions: {dim}')

if dim != (4608, 2592): #If dimensions not as expected, we need to scale K
    scale = dim[0] / 4608
    K = K * scale
    print(f"Scaled K by {scale}")

#Init map for undistortion
R = np.eye(3)
#New intrinsic matrix
Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, R, balance=0, new_size=dim)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, Knew, dim, cv2.CV_16SC2)
print(f'Initialized undistortion maps.')

#Preallocate arrays to store latest Pos data
num_markers = 20 #Assuming max 20 markers
pose_data = np.zeros((num_markers, 4)) #ID, center x, center y, heading


#Setup ArUco detector
#Setup aruco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict)

#Run detection on image to find known points for homography
#Convert to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Flatten the Image
flattened = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

#Detect the markers
corners, ids, _ = detector.detectMarkers(flattened)

#Check and make sure we found the known markers
if not(1 in ids and 2 in ids and 3 in ids and 4 in ids):
    raise ValueError("Not all required markers (1,2,3,4) were detected in the image.")

#Unpack array of tag corner coordinate
for idx, tag in enumerate(corners):
    #Get marker ID
    pose_data[idx][0] = ids[idx][0]

    #Corners is a tuple of arrays
    #Tag is an array which contains exactly one element: an array of four arrays eahc with a corner coordinate
    #So we have to double index tag to get to the actual coordinates
    #I don't know why there's this extra array in here and I think it's kinda dumb
    #Uppack corner coordinates
    x_tl, y_tl = tag[0][0]  # Top-left corner
    x_tr, y_tr = tag[0][1]  # Top-right corner
    x_br, y_br = tag[0][2]  # Bottom-right corner
    x_bl, y_bl = tag[0][3]  # Bottom-left corner
    
    #Calculate marker center point
    cX = int((x_tl + x_br) / 2.0)
    cY = int((y_tl + y_br) / 2.0)
    #store center points
    pose_data[idx][1] = cX
    pose_data[idx][2] = cY

#Extract pixel coordinates from pose_data based on marker IDs
ids = pose_data[:,0].astype(int).flatten() #extract IDs as 1D array
pixel_coords = np.zeros((4,2), dtype=np.float32) #preallocate array for pixel coords
for i in range(4): #Extract pixel coords for markers 1,2,3,4 using ids to index into pose_data
    pixel_coords[i,0] = pose_data[np.where(ids==i+1)[0],1][0]
    pixel_coords[i,1] = pose_data[np.where(ids==i+1)[0],2][0]


#Now, so we only look at the game track going forwad, crop the image based on the center points of the 4 markers
x1 = int(pixel_coords[3,0]) #Bottom-right marker x-pos
y1 = int(pixel_coords[3,1]) #Bottom-right marker y-pos
x2 = int(pixel_coords[2,0]) #bottm-left marker x-pos
y2 = int(pixel_coords[1,1]) #Top-left marker y-pos
print(f"Cropping image to x:[{x1}:{x2}], y:[{y1}:{y2}]")

#Translate pixel coords to correspond to cropped image
pixel_coords[:,0] = pixel_coords[:,0] - x1
pixel_coords[:,1] = pixel_coords[:,1] - y1

#Define the real-world coordinates of the 4 markers on the table (in mm)
w = 3.375 #marker width in inches
world_coords = np.array([[0,2*12+5.5+w],
                        #[6*12+3-2*12+8,2*12+5.5+w], #From when marker was in the middle
                         [6*12+3+w,2*12+5.5+w],
                         [6*12+3+w,0],
                         [0,0]], dtype=np.float32)*25.4 #Convert to mm


#Calculate Homography matrix
H = cv2.getPerspectiveTransform(pixel_coords, world_coords) #Compute the homography matrix from pixel coordinates to world coordinates
print("Found Homography Matrix (H)")



#Function to print latest pos data
def display_pose_table(pose_data: np.ndarray, telpased, n: int) -> None:

    #Sort by Marker ID
    sorted_idxs = np.argsort(pose_data[0:n,0])
    
    # Fixed column widths so table never shifts
    W_ID = 9
    W_POS = 21
    W_HEAD = 8

    #Helper Functions
    def hline():
        return f"+{'-'*(W_ID+2)}+{'-'*(W_POS+2)}+{'-'*(W_HEAD+2)}+"

    def row(a, b, c):
        return f"| {a:<{W_ID}} | {b:<{W_POS}} | {c:<{W_HEAD}} |"

    lines = [
        hline(),
        row("Marker ID", "Position", "Heading"),
        hline(),
    ]

    for idx in sorted_idxs:
        id, x, y, heading = pose_data[idx]
        lines.append(
            row(
                str(int(id)),
                f"({x:07.3f}, {y:07.3f})",
                f"{heading:07.3f}",
            )
        )

    lines.append(hline())

    # Clear screen and redraw
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write(f"Elapsed Time: {telpased:.4f} seconds\n")
    sys.stdout.flush()

@profile
def hotloopfnc():
    #Loop Code
    tstart = time.perf_counter()
    #Take a picture
    original_img = picam2.capture_array()
    img = cv2.resize(original_img, None, fx=scaledown_factor, fy=scaledown_factor, interpolation=cv2.INTER_LINEAR)

    #Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Flatten the image
    flattened = cv2.remap(gray_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    #Crop the image
    img_cropped = flattened[y1:y2, x1:x2]

    # Detect the markers
    corners, ids, _ = detector.detectMarkers(img_cropped)

    if ids is not None:    #Display results    
    
        #Unpack array of tag corner coordinate
        for idx, tag in enumerate(corners):
            #Get marker ID
            pose_data[idx][0] = ids[idx][0]
            
            #Transfrom to world coords
            #It makes me sad that we have to do it here, bcause now we have a bunch of floats instead of ints
            #No actually they were floats all along idiot
            mapped = cv2.perspectiveTransform(tag, H)

            #Corners is a tuple of arrays
            #Tag is an array which contains exactly one element: an array of four arrays eahc with a corner coordinate
            #So we have to double index tag to get to the actual coordinates
            #I don't know why there's this extra array in here and I think it's kinda dumb
            #Actually this was probably intentioanl b/c cv2 requires an extra dimension so we can just throw tag straight into it
            #Uppack corner coordinates
            x_tl, y_tl = mapped[0][0]  # Top-left corner
            x_tr, y_tr = mapped[0][1]  # Top-right corner
            x_br, y_br = mapped[0][2]  # Bottom-right corner
            x_bl, y_bl = mapped[0][3]  # Bottom-left corner
            
            #Calculate tag center point and translate to cropped image coords
            cX = (x_tl + x_br) / 2.0
            cY = (y_tl + y_br) / 2.0
            #Remao to world coords
            #store center points
            pose_data[idx][1] = cX
            pose_data[idx][2] = cY
            
            #Calcuate heading angle
            #Look at vector from bottom-right to bottom-left
            deltaX = x_bl - x_br
            deltaY = y_bl - y_br
            angle_rad = np.arctan2(deltaY, deltaX)
            angle_deg = np.degrees(angle_rad)
            #Store heading angle
            pose_data[idx][3] = angle_deg
        
        #Compute elapsed time
        telapsed = time.perf_counter() - tstart

        #Print info
        display_pose_table(pose_data, telapsed, len(ids))

hotloopfnc()
#while True:
#    try:
#        hotloopfnc()
#    except KeyboardInterrupt:
#        break

        
