import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from sklearn.decomposition import PCA
from ultralytics import YOLO

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
from utils import *
from detection_utils import *

import cv2

import numpy as np

# Network Data
source= os.path.abspath('/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/segment/images/image1.png')
model= YOLO("/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/segment/YOLOv8_trained.pt")
save_dir = '/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/segment/masks/'

segmentation = model.predict(source, save=False, save_txt=False, box=True, imgsz=640, line_thickness=1, retina_masks=True)
# Initialize an empty list to store segmented masks
segmented_pixels = []

# Iterate over each Results object in the segmentation list
for result in segmentation:
    # Access the masks attribute from the current Results object
    masks = result.masks
    
    # Access the xy attribute to get the segments in pixel coordinates
    segments = masks.xy
    
    # Append each segment to the list of segmented pixels
    segmented_pixels.extend(segments)
print(len(segmented_pixels))

min_segment_size = 400  # You can adjust this threshold as per your requirement

# Filter out large segments
large_segments = []
for segment_vertices in segmented_pixels:
    # Calculate the area of the segment (you can also use perimeter or any other metric)
    segment_area = cv2.contourArea(segment_vertices)
    
    # Check if the segment meets the minimum size threshold
    if segment_area >= min_segment_size:
        large_segments.append(segment_vertices)
print(len(large_segments))

original_image = cv2.imread(source)

# Create a copy of the original image to draw segments on
image_with_segments = original_image.copy()

# Iterate through each segment and draw it on the image
for segment_vertices in large_segments:
    segment_vertices = segment_vertices.astype(np.int32)  # Convert vertices to integer
    cv2.polylines(image_with_segments, [segment_vertices], isClosed=True, color=(0, 255, 0), thickness=2)

# Display the image with segments
cv2.imshow('Image with Segments', image_with_segments)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Initialize a list to store ROIs
ROIs = []

# Iterate through each segment and extract ROI
for i, segment_vertices in enumerate(large_segments):
    # Convert vertices to integer
    segment_vertices = segment_vertices.astype(np.int32)
    
    # Find the bounding box of the segment
    x, y, w, h = cv2.boundingRect(segment_vertices)
    
    # Crop the ROI from the original image
    ROI = original_image[y:y+h, x:x+w]
    
    # Append the ROI to the list
    ROIs.append(ROI)
    
    # Display the ROI
    cv2.imshow(f'ROI {i+1}', ROI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()