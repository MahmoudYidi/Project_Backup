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

# Specify the range of bands to load (visible)
start_wl = 460.27
end_wl = 750.89

# Specify the range of bands to load (IR)
#start_wl = 700.00
#end_wl = 998.66

# Specify the range of bands to load (In between)
#start_wl = 650.45
#end_wl = 850.00

# Specify the range of bands to load (Going Longer)
#start_wl = 650.45
#end_wl = 998.66

#  Load the HSI data cube with the specified range of bands
hsi_data_raw, bandss = load_envi_hsi_by_wavelength(raw, start_wl, end_wl)

#print(bandss)

# Band Selected by manual inspection (Red: Around 620-750 nanometers (nm) /Green: Around 495-570 nm /Blue: Around 450-495 nm)

R = get_band_index(bandss,650.45)
G = get_band_index(bandss,540.62)
B = get_band_index(bandss,460.27)


# Intermediate
#R = get_band_index(bandss,850.00)
#G = get_band_index(bandss,775.00)
#B = get_band_index(bandss,650.45)

# IR Trial Image
#R = get_band_index(bandss,650.45)
#G = get_band_index(bandss,775.00)
#B = get_band_index(bandss,700.00)

#  Load the HSI data cube with the specified range of bands
hsi_data_white, _ = load_envi_hsi_by_wavelength(white, start_wl, end_wl)

#  Load the HSI data cube with the specified range of bands
hsi_data_dark, _ = load_envi_hsi_by_wavelength(dark, start_wl, end_wl)

#bands =  hsi_data_raw.bands.centers
###
#print(bands)

corrected_data = data_correction(hsi_data_raw, hsi_data_dark, hsi_data_white)
#print("Corrected HSI shape:", corrected_data1.shape)

# Standardisation
#corrected_data=standardize_per_wavelength(corrected_data1)
#corrected_data=min_max_normalization_per_wavelength(corrected_data1)

#Free some space (Use if you want)
del hsi_data_raw, hsi_data_white, hsi_data_dark


#Get RGB Image (BGR)
img = get_rgb(corrected_data, bands=[B,G,R])

#View wavelength
view_image_at_wavelength(corrected_data, get_band_index(bandss,900.89), 'image_at_wave900.png' )

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

box_size = (40, 40)

# Extract ROIs
rois = extract_rois(corrected_data, clicks, box_size)
len(rois)

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

wavelengths = bandss


plot_reflectance_data(rois, wavelengths)

# Calculate statistics for each ROI
roi_statistics = calculate_statistics_for_rois(rois)

# Print results for each ROI
#for i, (mean, std_deviation, variance) in enumerate(roi_statistics):
#    print(f"ROI {i+1}:")
#    print("Mean:", mean)
#    print("Standard Deviation:", std_deviation)
#   print("Variance:", variance)
#   print()
    

##Plotting section ##################


#plot_statistics_for_rois(rois)

#plot_statistics_for_rois_per(rois, wavelengths,roi_index=0)
#plot_statistics_metric(rois, wavelengths,roi_index=None)
#mean1, variance1 = compute_mean_and_variance_between_selected_rois(rois,[0,1])
#mean2, variance2 = compute_mean_and_variance_between_selected_rois(rois,[0,2])

#mse1, cos1 = compute_mse_cos(rois,[0,1])
#mse2, cos2 = compute_mse_cos(rois,[0,2])
#mse3, cos3 = compute_mse_cos(rois,[0,3])
#mse3, cos4 = compute_mse_cos(rois,[1,3])
#print(mse1.shape)

#mse_1, rmse_1, ssi_1, cross_1 = compute_similarity_metrics_between_signatures(rois, 0, 1)
#mse_2, rmse_2, ssi_2, cross_2 = compute_similarity_metrics_between_signatures(rois, 0, 2)

#cos1, euc1 = compute_SAM(rois, [0,1]) #per wavelength
#cos2, euc2 = compute_SAM(rois, [0,2])
#cos3, euc3 = compute_SAM(rois, [0,3])

cos3, euc3 = compute_SAM_all(rois, [0,1])

print(cos3,euc3)


#plot_3_cosine_similarities(cos1, cos2, cos3, 'Scores', wavelengths)
#plot_3_cosine_similarities(euc1, euc2, euc3, 'Angles', wavelengths)

plot_SAM_all(cos3,euc3)

# Print the size of the mean list



# Call the plotting function for each statistic
