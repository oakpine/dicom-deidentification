from itertools import count
import cv2
from matplotlib import scale
from numpy.core.arrayprint import array2string
from pydicom import dcmread
import numpy as np
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
from dateutil.parser import parse
import re

from imutils.object_detection import non_max_suppression

def east_detect(image):
    layerNames = [
    	"feature_fusion/Conv_7/Sigmoid",
    	"feature_fusion/concat_3"]
    
    orig = image.copy()
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    (H, W) = image.shape[:2]
    
    # Set new width and height
    (newW, newH) = (800, 800)
    
    rW = W / float(newW)
    rH = H / float(newH)
    
    # Resize image
    image = cv2.resize(image, (newW, newH))
    
    (H, W) = image.shape[:2]
    
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
    	(123.68, 116.78, 103.94), swapRB=True, crop=False)
    
    net.setInput(blob)
    
    (scores, geometry) = net.forward(layerNames)
    
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # Loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
    
        for x in range(0, numCols):
    		# If our score does not have sufficient probability, ignore it
            # Set minimum confidence as required
            if scoresData[x] < 0.5:
                continue
    		# Compute the offset factor as our resulting feature maps will
            #  x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # Extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # Use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # Compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # Add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
                        
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    #boxes= cv2.dnn.NMSBoxes(np.array(rects), confidences, 0.3, 0.3)

    coords = []
    
    # Loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
    	# Scale the bounding box coordinates based on the respective ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # Draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), 0, 2)
        coords.append((startX, startY, endX, endY))

    # plt.imshow(orig, cmap="gray")
    # plt.show()
    # cv2.imwrite("sample_output.jpg", orig) # Save image with bounding boxes
    
    return coords

def overlap(rect1, rect2):
    ''' Check if rectangles overlap (with a buffer)
    rect1 and rect2: [x1, y1, x2, y2]'''

    buffer = 20

    lr = False # Left and right
    tb = False # Top and bottom

    # Check left and right
    if rect1[0] < rect2[0]:
        if rect1[2] + buffer >= rect2[0]:
            lr = True
    else:
        if rect2[2] + buffer >= rect1[0]:
            lr = True

    # Check top and bottom
    if rect1[1] < rect2[1]:
        if rect1[3] + buffer >= rect2[1]:
            tb = True
    else:
        if rect2[3] + buffer >= rect1[1]:
            tb = True

    if lr and tb:
        return True
    return False

def combine_rects(coords):
    rects = []
    for c in coords:
        matched = False
        for r in range(len(rects)):
            if overlap(rects[r], c):
                matched = True

                # Use coordinates that give the biggest rectangle
                rects[r][0] = min(rects[r][0], c[0])
                rects[r][1] = min(rects[r][1], c[1])
                rects[r][2] = max(rects[r][2], c[2])
                rects[r][3] = max(rects[r][3], c[3])

                c = rects[r]

        if not matched:
            rects.append(list(c))

    return rects


# Original dcm files
# path = 'img/ID_0006.dcm'
# path = 'img/nema_img11.dcm'
path = 'img/CT-MONO2-8-abdo'

ds = dcmread(path)
# `img` is a numpy.ndarray
img = ds.pixel_array

# convert to uint8 if necessary
if type(img[0][0]) == np.uint16:
    factor = img.max() / 255
    img =(img/factor).astype('uint8')
    # img = (img/256).astype('uint8')
elif type(img[0][0]) == np.int16:
    print('y int')
    img = img.astype(np.double)
    img = img + 32768
    img = (img/256).astype('uint8')

# Get counts of pixel values
unique, counts = np.unique(img, return_counts=True)
counts = dict(zip(unique, counts))
counts.pop(img.min(), None)
counts.pop(img.max(), None)
# print(counts)
# plt.bar(counts.keys(), counts.values())
# plt.show()

# Calculate thresholds based on pixel value distribution
count_list = [key for key, val in counts.items() for _ in range(val)]
# print(img[0])
mean = np.mean(count_list)
std = np.std(count_list)
# thresholds = [int(mean - 2*std), int(mean - std), int(mean + std), int(mean + 2*std)]
thresholds = [int(mean - std), int(mean + std)]
print(thresholds)

# print(ds)
plt.imshow(img, cmap="gray")
plt.show()

# Create a copy of image
im2 = img.copy()

# Convert to gray scale if necessary
try:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
except:
    pass

# thresholds = [80, 127, 170] # Thresholds to test
# thresholds = [126, 127, 128]
# thresholds = [126.5, 127, 127.5]

max_text = -1
max_img = None # Image with most text
max_coords = None
for t in thresholds:
    # Apply thresholding
    ret,thresh = cv2.threshold(img,t,255,cv2.THRESH_BINARY)
    coords = east_detect(thresh)
    print(len(coords))
    plt.imshow(thresh, cmap="gray")
    plt.show()

    # Save image if it has the most text so far
    if len(coords) > max_text:
        max_text = len(coords)
        max_img = thresh
        max_coords = coords

# Continue using image with the most text
thresh = max_img
coords = max_coords

# coords = east_detect(img)
rects = combine_rects(coords)

plt.imshow(thresh, cmap="gray")
plt.show()