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


raw = "/mnt/c/Users/mahmo/Desktop/Github_Dump/hsi_images_qualicrop_fx10e_29-01-24/anom_1_2024-01-26_15-21-55/capture/anom_1_2024-01-26_15-21-55.hdr"
dark = '/mnt/c/Users/mahmo/Desktop/Github_Dump/hsi_images_qualicrop_fx10e_29-01-24/dark_ref_shutter_cap_on/capture/dark_ref_shutter_2024-01-26_16-03-56.hdr'
white = '/mnt/c/Users/mahmo/Desktop/Github_Dump/hsi_images_qualicrop_fx10e_29-01-24/white_ref_2024-01-26_16-00-30/capture/white_ref_2024-01-26_16-00-30.hdr'


start_wl = 529.91
end_wl = 580.80
hsi_data_raw, bandss = load_envi_hsi_by_wavelength(raw, start_wl, end_wl)
print(hsi_data_raw.shape)
#  Load the HSI data cube with the specified range of bands
hsi_data_white, _ = load_envi_hsi_by_wavelength(white, start_wl, end_wl)

#  Load the HSI data cube with the specified range of bands
hsi_data_dark, _ = load_envi_hsi_by_wavelength(dark, start_wl, end_wl)

#  Load the HSI data for flat surface correction
hsi_data_flat_= load_envi_hsi_2D(white, 529.91)
print(hsi_data_flat_.shape)
print("Data Loaded!!")

###############################################################################

corrected_data = data_correction(hsi_data_raw, hsi_data_dark, hsi_data_white)

view_image_at_wavelength(corrected_data, get_band_index(bandss,580.80), 'image_uncorrect.png' )
corrected_data1 = flat_field_correction(corrected_data, hsi_data_flat_)
view_image_at_wavelength(corrected_data1, get_band_index(bandss,580.80), 'image_correct.png' )
#corrected_data3 = histogram_equalization_hsi_cube(corrected_data1)
#view_image_at_wavelength(corrected_data3, get_band_index(bandss,580.80), 'image_correct_hist2.png' )
corrected_data4 = gradient_based_correction(corrected_data)
view_image_at_wavelength(corrected_data4, get_band_index(bandss,580.80), 'image_correct_hist2.png' )

plot_reflectance_data([corrected_data,corrected_data1,corrected_data4], bandss)






