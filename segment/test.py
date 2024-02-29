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
source= os.path.abspath('/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/segment/images')
model= YOLO("/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/segment/YOLOv8_trained.pt")
save_dir = '/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/segment/masks/'



# Path to the ENVI HSI file (1st data)
raw = "/mnt/c/Users/mahmo/Desktop/Github_Dump/hsi_images_qualicrop_fx10e_29-01-24/anom_1_2024-01-26_15-21-55/capture/anom_1_2024-01-26_15-21-55.hdr"
dark = '/mnt/c/Users/mahmo/Desktop/Github_Dump/hsi_images_qualicrop_fx10e_29-01-24/dark_ref_shutter_cap_on/capture/dark_ref_shutter_2024-01-26_16-03-56.hdr'
white = '/mnt/c/Users/mahmo/Desktop/Github_Dump/hsi_images_qualicrop_fx10e_29-01-24/white_ref_2024-01-26_16-00-30/capture/white_ref_2024-01-26_16-00-30.hdr'

# Specify the range of bands to load (visible analysis)
start_wl = 529.91
end_wl = 580.80

# Loading HSI Cube 
hsi_data_raw, bandss = load_envi_hsi_by_wavelength(raw, start_wl, end_wl)
hsi_data_white, _ = load_envi_hsi_by_wavelength(white, start_wl, end_wl)
hsi_data_dark, _ = load_envi_hsi_by_wavelength(dark, start_wl, end_wl)
wavelengths = bandss
#RGB Image formation
RGB_former = load_envi_RGB_wave(raw, [460.27,540.62,650.45]) #BGR
img = get_rgb(RGB_former, bands=[0,1,2])
image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
image = image.astype(np.uint8)

#Calibration (Tc-Td/Tw-Td)
corrected_data = data_correction(hsi_data_raw, hsi_data_dark, hsi_data_white)
#corrected_data= min_max_normalization_per_wavelength(corrected_data)

#print("Corrected HSI shape:", corrected_data1.shape)
del hsi_data_raw, hsi_data_white, hsi_data_dark #Trying to save space. LOL

#Segmentation
segmentation = model.predict(image, save=False, save_txt=False, box=True, imgsz=640, line_thickness=1, retina_masks=True)
ROIS = gather_segmented_pixels(segmentation, save_dir, min_width_threshold=50, min_height_threshold=50)
HSI_rois =  extract_rois_from_hsi(corrected_data, ROIS)
plot_reflectance_data(HSI_rois, wavelengths)


    
####Try Kmeans Operation ######################
#clustter_labels = kmeans_clustering(HSI_rois[0],3)
#visualize_clusters(HSI_rois[0],clustter_labels,save_path='cluster_means.png')
#plot_reflectance_per_cluster(HSI_rois[0],bandss,clustter_labels,save_path='reflectance_cluster.png')
#scatter_plot_clusters(HSI_rois[0],clustter_labels,save_path='scatter_cluster_means.png')
#visualize_clustered_hsi(HSI_rois[0],3,save_path='cluster_combine.png')
#plot_clustered_pixels_pca(HSI_rois[0],clustter_labels,2,save_path='scatter_clusterPCAmeans.png')

# No memory for medoids LOL
#clustter_labels2 = medoids_clustering(HSI_rois[0],3)
#visualize_clusters(HSI_rois[0],clustter_labels2,save_path='cluster_medoids.png')
#scatter_plot_clusters(HSI_rois[0],clustter_labels2,save_path='scatter_cluster_medoids.png')

#Now theres on function to do all!!!!####
cluster_folder = "/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/segment/clusters"
process_hsi_rois_cluster(HSI_rois, 4, bandss, cluster_folder, view_wavelength=580.80)