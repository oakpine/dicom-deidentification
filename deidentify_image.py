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

def find_sensitive(data, img, rect):
    # Loop through detected text
    for i in range(len(data['text'])):
        if data['text'][i] == '': # Continue if text is empty
            continue
        for text in re.split('[, ]', data['text'][i]): # Split on comma or space
            print(text)

            # Use datetime comparison for dates
            for f in date_fields:
                try:
                    if parse(text) == parse(ds[f].value):
                        img = censor_text(img, data, rect, i)
                        break # Test is already censored
                except: # Catch if field doesn't exist
                    continue

            text = re.sub(r'[^\w\s]', '', text) # Remove punctuation

            # Loop through sensitive fields
            for f in fields:
                try:
                    if ds[f].value == '': # Continue if the field is empty
                        continue 

                    ftext = re.sub(r'\^', ' ', str(ds[f].value)) # Split on ^ in field text
                    ftext = re.sub(r'[^\w\s]', '', str(ds[f].value)) # Remove punctuation in field text

                    # Remove trailing 0 if field is Patient's Age
                    if f == (0x0010,0x1010):
                        if text.lstrip('0').rstrip('Y') == ftext.lstrip('0').rstrip('Y') or \
                           text.lstrip('O').rstrip('Y') == ftext.lstrip('0').rstrip('Y'):
                            img = censor_text(img, data, rect, i)
                            break # Text is already censored
                    elif text == ftext or text.strip() in str(ftext).strip().split(): # Check for text match
                        try:
                            img = censor_text(img, data, rect, i)
                        except Exception as e:
                            print(e)
                        break # Text is already censored
                except: # Catch if field doesn't exist
                    continue
    return img

def censor_text(img, data, rect, i):
    e = 10
    y = max(0, rect[1]-e)
    x = max(0, rect[0]-e)

    left = data['left'][i]//scale 
    top = data['top'][i]//scale 
    right = left + data['width'][i]//scale 
    bottom = top +data['height'][i]//scale

    img = np.ascontiguousarray(img, dtype=np.uint8)
    try:
        cv2.rectangle(img, (int(left+x), int(top+y)), (int(right+x), int(bottom+y)), 255, -1)
    except Exception as e:
        print(e)

    return img


# Original dcm files
path = 'img/ID_0006.dcm'
# path = 'img/nema_img11.dcm'
# path = 'img/CT-MONO2-8-abdo'

# Sensitive attributes
fields = [(0x0008,0x0014), (0x0008,0x0018), (0x0008,0x0050), (0x0008,0x0080), (0x0008,0x0081), (0x0008,0x0090), (0x0008,0x0092),
          (0x0008,0x0094), (0x0008,0x1010), (0x0008,0x1030), (0x0008,0x103E), (0x0008,0x1040), (0x0008,0x1048), (0x0008,0x1050),
          (0x0008,0x1060), (0x0008,0x1070), (0x0008,0x1080), (0x0008,0x1155), (0x0008,0x2111), (0x0010,0x0010), (0x0010,0x0020),
          (0x0010,0x0040), (0x0010,0x1000), (0x0010,0x1001), (0x0010,0x1010), (0x0010,0x1020), (0x0010,0x1030), (0x0010,0x1090), 
          (0x0010,0x2160), (0x0010,0x2180), (0x0010,0x21B0), (0x0010,0x4000), (0x0018,0x1000), (0x0018,0x1030), (0x0020,0x000D), 
          (0x0020,0x000E), (0x0020,0x0010), (0x0020,0x0052), (0x0020,0x0200), (0x0020,0x4000), (0x0040,0x0275), (0x0040,0xA124), 
          (0x0040,0xA730), (0x0088,0x0140), (0x3006,0x0024), (0x3006,0x00C2)]
date_fields = [(0x0010,0x0030), (0x0010,0x0032)]

ds = dcmread(path)
# `img` is a numpy.ndarray
img = ds.pixel_array

# convert to uint8 if necessary
if type(img[0][0]) == np.uint16:
    factor = img.max() / 255
    img =(img/factor).astype('uint8')
elif type(img[0][0]) == np.int16:
    print('y int')
    img = img.astype(np.double)
    img = img + 32768
    img = (img/256).astype('uint8')

# print(ds)
# plt.imshow(img, cmap="gray")
# plt.show()

# Create a copy of image
im2 = img.copy()

coords = east_detect(img) # Detect text with EAST
rects = combine_rects(coords) # Find sections of text
 
# Looping through text sections
for rect in rects:
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
     
    # Crop the text block for giving input to OCR
    e = 10 # Increase cropped dimensions
    y1 = max(0, y1-e)
    x1 = max(0, x1-e)
    y2 = min(img.shape[0], y2+e)
    x2 = min(img.shape[1], x2+e)

    cropped = im2[y1:y2, x1:x2]

    # Resize image
    scale = max(2000//img.shape[0], 2000//img.shape[1])
    cropped = cv2.resize(cropped, None, fx=scale, fy=scale)

    # Convert to gray scale if necessary
    try:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    except:
        pass

    # Apply OCR to unthresholded
    data_nothresh = pytesseract.image_to_data(cropped, output_type=Output.DICT)
    # plt.imshow(cropped, cmap="gray")
    # plt.show()

    ret, cropped = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # plt.imshow(cropped, cmap="gray")
    # plt.show()

    # Apply OCR 
    data = pytesseract.image_to_data(cropped, output_type=Output.DICT)

    # Apply OCR to inverted
    cropped_inv = cv2.bitwise_not(cropped)
    data_inv = pytesseract.image_to_data(cropped_inv, output_type=Output.DICT)
    # plt.imshow(cropped_inv, cmap="gray")
    # plt.show()

    # Censor sensitive data
    img = find_sensitive(data_nothresh, img, rect)
    # img = find_sensitive(data_inv_nothresh, img, file, rect)
    img = find_sensitive(data, img, rect)
    img = find_sensitive(data_inv, img, rect)

plt.imshow(img, cmap="gray")
# plt.savefig("image4_text.png", dpi=2000)
plt.show()