import spectral
from spectral import imshow, get_rgb
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
import pandas as pd
import dask.array as da
from utils import *






# Example usage:
# Path to the ENVI HSI file
raw = "/mnt/c/Users/mahmo/Desktop/Github_Dump/hsi_images_qualicrop_fx10e_29-01-24/anom_1_2024-01-26_15-21-55/capture/anom_1_2024-01-26_15-21-55.hdr"
dark = '/mnt/c/Users/mahmo/Desktop/Github_Dump/hsi_images_qualicrop_fx10e_29-01-24/dark_ref_shutter_cap_on/capture/dark_ref_shutter_2024-01-26_16-03-56.hdr'
white = '/mnt/c/Users/mahmo/Desktop/Github_Dump/hsi_images_qualicrop_fx10e_29-01-24/white_ref_2024-01-26_16-00-30/capture/white_ref_2024-01-26_16-00-30.hdr'

# Specify the range of bands to load
start_wl = 440.18
end_wl = 700.00

#  Load the HSI data cube with the specified range of bands
hsi_data_raw, bandss = load_envi_hsi_by_wavelength(raw, start_wl, end_wl)

#print(bandss)


# Band Selected by manual inspection (Red: Around 620-750 nanometers (nm) /Green: Around 495-570 nm /Blue: Around 450-495 nm)
R = get_band_index(bandss,650.45)
G = get_band_index(bandss,540.62)
B = get_band_index(bandss,460.27)

#  Load the HSI data cube with the specified range of bands
hsi_data_white, _ = load_envi_hsi_by_wavelength(white, start_wl, end_wl)

#  Load the HSI data cube with the specified range of bands
hsi_data_dark, _ = load_envi_hsi_by_wavelength(dark, start_wl, end_wl)

#bands =  hsi_data_raw.bands.centers
###
#print(bands)

corrected_data = data_correction(hsi_data_raw, hsi_data_dark, hsi_data_white)
print("Corrected HSI shape:", corrected_data.shape)

#Free some space (Use if you want)
del hsi_data_raw, hsi_data_white, hsi_data_dark


#Get RGB Image (BGR)
img = get_rgb(corrected_data, bands=[B,G,R])

#Select single band
#sel = 70
#img = data_ref[:,:,sel]

image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
image = image.astype(np.uint8)
#image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)

#Uncomment to view the RGB Image
#cv2.namedWindow("main", cv2.WINDOW_NORMAL)
#cv2.imshow('main', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

clicks = get_right_clicks(image)


box_size = (5, 5)

# Extract ROIs
rois = extract_rois(corrected_data, clicks, box_size)

# Draw bounding boxes
for i, point in enumerate(clicks):
    x, y = point
    x1 = max(0, x - box_size[0] // 2)
    y1 = max(0, y - box_size[1] // 2)
    x2 = min(image.shape[1], x1 + box_size[0])
    y2 = min(image.shape[0], y1 + box_size[1])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, str(i + 1), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the shape of the loaded HSI data cube
#print("Loaded HSI shape:", hsi_data_subset.shape)


#lib = envi.open('/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/hsi_images_qualicrop_fx10e_29-01-24/anom_1_2024-01-26_15-21-55/capture/anom_1_2024-01-26_15-21-55.hdr')
#img = open_image('/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/hsi_images_qualicrop_fx10e_29-01-24/anom_1_2024-01-26_15-21-55/capture/anom_1_2024-01-26_15-21-55.hdr')

#white_ref = envi.open('/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/hsi_images_qualicrop_fx10e_29-01-24/white_ref_2024-01-26_16-00-30/capture/white_ref_2024-01-26_16-00-30.hdr','/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/hsi_images_qualicrop_fx10e_29-01-24/white_ref_2024-01-26_16-00-30/capture/white_ref_2024-01-26_16-00-30.raw')
#dark_ref = envi.open('/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/hsi_images_qualicrop_fx10e_29-01-24/dark_ref_shutter_cap_on/capture/dark_ref_shutter_2024-01-26_16-03-56.hdr','/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/hsi_images_qualicrop_fx10e_29-01-24/dark_ref_shutter_cap_on/capture/dark_ref_shutter_2024-01-26_16-03-56.raw')
#raw_ref = envi.open('/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/hsi_images_qualicrop_fx10e_29-01-24/anom_1_2024-01-26_15-21-55/capture/anom_1_2024-01-26_15-21-55.hdr','/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/hsi_images_qualicrop_fx10e_29-01-24/anom_1_2024-01-26_15-21-55/capture/anom_1_2024-01-26_15-21-55.raw')



#result_np = np.array(corrected)

#print(result_np)


#print(num.size)
#corrected_data = np.divide(num,denom

#Get RGB Image
#img = get_rgb(corrected_data, bands=None)

#Select single band
#sel = 70
#img = raw_data[:,:,sel]

#image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
#image = image.astype(np.uint8)
#image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)

#Uncomment to view the RGB Image
#cv2.namedWindow("main", cv2.WINDOW_NORMAL)
#cv2.imshow('main', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# Calculate average reflectance for each ROI (replace rois with your actual ROIs)

wavelengths = bandss


plot_reflectance_data(rois, wavelengths)

# Calculate statistics for each ROI
roi_statistics = calculate_statistics_for_rois(rois)

# Print results for each ROI
for i, (mean, std_deviation, variance) in enumerate(roi_statistics):
    print(f"ROI {i+1}:")
    print("Mean:", mean)
    print("Standard Deviation:", std_deviation)
    print("Variance:", variance)
    print()
    
plot_statistics_for_rois(rois)