import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from utils import *
from ultralytics import YOLO
from sklearn_extra.cluster import KMedoids

from IPython import display
display.clear_output()
from sklearn.preprocessing import StandardScaler

import ultralytics
ultralytics.checks()
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import cv2
import os
import numpy as np

def gather_segmented_pixels(segmentation, save_dir, min_width_threshold=50, min_height_threshold=50):
    # Access the first (and only) element of the segmentation list
    result = segmentation[0]
    
    # Access the boxes attribute containing bounding box coordinates
    boxes = result.boxes

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Initialize a list to store ROIs
    rois = []

    # Iterate over bounding boxes
    for i, box in enumerate(boxes.xyxy):
        # Extract coordinates of the bounding box
        x1, y1, x2, y2 = map(int, box)
        
        # Calculate the width and height of the bounding box
        width = x2 - x1
        height = y2 - y1

        # Extract the segmented region from the original image
        segmented_region = result.orig_img[y1:y2, x1:x2]

        # Check if the segmented region meets the criteria
        if width >= min_width_threshold and height >= min_height_threshold:
            # Save the segmented region to a file
            file_path = os.path.join(save_dir, f'segmented_region_{i}.png')
            cv2.imwrite(file_path, segmented_region)
            
            # Append the bounding box coordinates as ROI
            rois.append((x1, y1, x2, y2))

    return rois

def extract_rois_from_hsi(hsi_cube, rois):
    """
    Extracts regions of interest (ROIs) from a hyperspectral image (HSI) cube based on the provided ROIs.

    Parameters:
        hsi_cube (numpy.ndarray): HSI cube data.
        rois (list): List of ROIs, where each ROI is a tuple (x1, y1, x2, y2).

    Returns:
        list: List of extracted ROIs from the HSI cube.
    """
    extracted_rois = []

    # Iterate over ROIs
    for roi in rois:
        x1, y1, x2, y2 = roi
        
        # Extract the ROI from the HSI cube
        roi_data = hsi_cube[y1:y2, x1:x2, :]

        extracted_rois.append(roi_data)

    return extracted_rois


######KMEANS UTILSSSSS######################################
def kmeans_clustering(hsi_cube, n_clusters):
    """
    Apply K-means clustering to a hyperspectral image cube.
    
    Parameters:
        hsi_cube (numpy.ndarray): HSI data cube (3D array).
        n_clusters (int): Number of clusters for K-means.
    
    Returns:
        numpy.ndarray: Cluster labels for each pixel.
    """
    # Reshape the HSI cube into a 2D array
    num_pixels = hsi_cube.shape[0] * hsi_cube.shape[1]
    num_bands = hsi_cube.shape[2]
    data = np.reshape(hsi_cube, (num_pixels, num_bands))
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    
    # Reshape the cluster labels back into the original image shape
    cluster_labels = np.reshape(cluster_labels, (hsi_cube.shape[0], hsi_cube.shape[1]))
    
    return cluster_labels

def visualize_clusters(hsi_cube, cluster_labels, save_path=None):
    """
    Visualize the clustered regions on the original hyperspectral image with color masks.
    
    Parameters:
        hsi_cube (numpy.ndarray): Original hyperspectral image cube (3D array).
        cluster_labels (numpy.ndarray): Cluster labels for each pixel.
        save_path (str, optional): Path to save the visualization plot. If None, the plot will be displayed but not saved.
    """
    # Create a colormap with a unique color for each cluster label
    num_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.Set1(np.linspace(0, 1, num_clusters))  # Using Set1 colormap for distinctive colors
    cmap = ListedColormap(colors)
    
    # Display the original image
    plt.figure(figsize=(10, 8))
    plt.imshow(np.sum(hsi_cube, axis=2), cmap='gray')  # Display grayscale sum of spectral bands
    
    # Overlay color masks for each cluster
    plt.imshow(cluster_labels, cmap=cmap, alpha=0.5, vmin=0, vmax=num_clusters - 1)
    
    # Add legend with cluster labels
    legend_elements = [plt.Rectangle((0,0),1,1, color=colors[i], label=f'Cluster {i}') for i in range(num_clusters)]
    plt.legend(handles=legend_elements, loc='upper right', fontsize='medium')

    plt.title('Clustered Regions')
    plt.axis('off')
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def scatter_plot_clusters(hsi_cube, cluster_labels, save_path=None):
    """
    Plot a scatter plot of pixels in the hyperspectral image colored by their cluster labels.
    
    Parameters:
        hsi_cube (numpy.ndarray): Original hyperspectral image cube (3D array).
        cluster_labels (numpy.ndarray): Cluster labels for each pixel.
        save_path (str, optional): Path to save the visualization plot. If None, the plot will be displayed but not saved.
    """
    # Reshape the HSI cube into a 2D array of pixels
    num_pixels = hsi_cube.shape[0] * hsi_cube.shape[1]
    pixels = np.reshape(hsi_cube, (num_pixels, hsi_cube.shape[2]))
    
    # Generate scatter plot
    plt.figure(figsize=(10, 8))
    for cluster_label in np.unique(cluster_labels):
        cluster_pixels = pixels[cluster_labels.flatten() == cluster_label]
        plt.scatter(cluster_pixels[:, 0], cluster_pixels[:, 1], label=f'Cluster {cluster_label}', s=10)
    
    plt.title('Scatter Plot of Clusters')
    plt.xlabel('Band 1')
    plt.ylabel('Band 2')
    plt.legend()
    plt.grid(True)
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



def medoids_clustering(hsi_cube, num_clusters):
    """
    Perform medoids clustering on the hyperspectral image cube.
    
    Parameters:
        hsi_cube (numpy.ndarray): Original hyperspectral image cube (3D array).
        num_clusters (int): Number of clusters.
    
    Returns:
        numpy.ndarray: Cluster labels for each pixel.
    """
    # Reshape the HSI cube into a 2D array of pixels
    num_pixels = hsi_cube.shape[0] * hsi_cube.shape[1]
    pixels = np.reshape(hsi_cube, (num_pixels, hsi_cube.shape[2]))
    
    # Perform medoids clustering
    kmedoids = KMedoids(n_clusters=num_clusters, random_state=0).fit(pixels)
    
    # Get cluster labels
    cluster_labels = kmedoids.labels_
    
    # Reshape cluster labels to match the original image dimensions
    cluster_labels = np.reshape(cluster_labels, (hsi_cube.shape[0], hsi_cube.shape[1]))
    
    return cluster_labels

def plot_clustered_pixels_pca(hsi_cube, labels, n_clusters, save_path=None):
    """
    Plot clustered pixels in a scatter plot after applying PCA for dimensionality reduction.

    Parameters:
        hsi_cube (numpy.ndarray): Hyperspectral image cube (M x N x L).
        labels (numpy.ndarray): Array of cluster labels for each pixel.
        n_clusters (int): Number of clusters.
        save_path (str): Optional path to save the plot as an image file.
    """
    # Flatten the image cube
    n_pixels, n_bands = hsi_cube.shape[:2]
    X = hsi_cube.reshape(-1, n_bands)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # Ensure the shape of the labels array matches the number of pixels after PCA
    if len(labels) != X_pca.shape[0]:
        raise ValueError("Number of labels does not match the number of pixels after PCA.")

    # Plot scatter of clustered pixels
    plt.figure(figsize=(10, 8))
    for cluster_label in range(n_clusters):
        cluster_pixels = X_pca[labels == cluster_label]
        plt.scatter(cluster_pixels[:, 0], cluster_pixels[:, 1], label=f'Cluster {cluster_label}', s=10)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Scatter Plot of Clustered Pixels (PCA)')
    plt.legend()

    if save_path:
        plt.savefig(save_path)  # Save the plot as an image file

    plt.show()
    
def visualize_clustered_hsi(hsi_cube, n_clusters, save_path):
    """
    Perform k-means clustering on the HSI cube, overlay a distinctive colormap on the clustered image, 
    and save the visualization as an image file.

    Parameters:
        hsi_cube (numpy.ndarray): Hyperspectral image cube (M x N x L).
        n_clusters (int): Number of clusters for k-means clustering.
        save_path (str): Path to save the visualization image.
    """
    # Flatten the HSI cube
    n_pixels, n_bands = hsi_cube.shape[:2]
    X = hsi_cube.reshape(-1, n_bands)

    # Normalize the feature matrix
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X_normalized)

    # Reshape the labels to the original HSI shape for visualization
    clustered_hsi = labels.reshape(n_pixels, -1)

    # Define a distinctive colormap
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    cluster_cmap = ListedColormap(colors)

    # Visualize the clustered HSI with the distinctive colormap
    plt.figure(figsize=(10, 8))
    plt.imshow(clustered_hsi, cmap=cluster_cmap, aspect='auto')

    # Add colorbar
    plt.colorbar(label='Cluster')

    # Add annotations for cluster labels
    for i in range(n_clusters):
        # Find the centroid of the cluster and label it with the cluster number
        centroid = np.mean(X[labels == i], axis=0)
        plt.text(centroid[0], centroid[1], str(i), color='black', fontsize=8)

    plt.title('Clustered Hyperspectral Image')
    plt.axis('off')

    # Save the visualization as an image file
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
def plot_reflectance_per_cluster(hsi_cube, wavelengths, cluster_labels, save_path=None):
    """
    Plot the reflectance per wavelength of each clustered region and optionally save the plot.

    Parameters:
        hsi_cube (numpy.ndarray): Original hyperspectral image cube (3D array).
        wavelengths (list): List containing the wavelengths corresponding to each band.
        cluster_labels (numpy.ndarray): Cluster labels for each pixel.
        save_path (str, optional): Path to save the visualization plot. If None, the plot will not be saved.
    """
    # Number of clusters
    num_clusters = len(np.unique(cluster_labels))

    # Initialize a list to store the average reflectance spectra for each cluster
    average_reflectance_per_cluster = []

    # Iterate through each cluster
    for cluster in range(num_clusters):
        # Mask for pixels belonging to the current cluster
        cluster_mask = (cluster_labels == cluster)

        # Extract reflectance spectra for pixels in the current cluster
        reflectance_values = hsi_cube[cluster_mask]

        # Calculate the average reflectance spectrum for the current cluster
        average_reflectance_spectrum = np.mean(reflectance_values, axis=0)

        # Add the average reflectance spectrum to the list
        average_reflectance_per_cluster.append(average_reflectance_spectrum)

    # Plot reflectance per wavelength for each cluster
    plt.figure(figsize=(10, 6))
    for cluster, reflectance_spectrum in enumerate(average_reflectance_per_cluster):
        plt.plot(wavelengths, reflectance_spectrum, label=f'Cluster {cluster + 1}')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Reflectance per Wavelength of Each Clustered Region')
    plt.legend()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
def process_hsi_rois_cluster(HSI_rois, num_clusters, bands_list, save_folder, view_wavelength=None):
    """
    Process each HSI ROI: perform k-means clustering, visualize the clustered regions,
    plot the reflectance per cluster, and view/save the image at a specific wavelength.

    Parameters:
        HSI_rois (list): List of HSI ROIs.
        num_clusters (int): Number of clusters for k-means clustering.
        bands_list (list): List of wavelengths corresponding to each band.
        save_folder (str): Folder path to save the results.
        view_wavelength (int or float, optional): Wavelength at which to view the image. Default is None.
    """
    for idx, roi in enumerate(HSI_rois):
        # Perform k-means clustering
        cluster_labels = kmeans_clustering(roi, num_clusters)
        
        # Visualize clustered regions
        visualize_clusters(roi, cluster_labels, save_path=f'{save_folder}/clustered_regions_{idx}.png')
        
        # Plot reflectance per cluster
        plot_reflectance_per_cluster(roi, bands_list, cluster_labels, save_path=f'{save_folder}/reflectance_per_cluster_{idx}.png')

        # View/save the image at a specific wavelength
        if view_wavelength is not None:
            view_image_at_wavelength(roi, get_band_index(bands_list, view_wavelength), f'{save_folder}/image_at_wavelength_{idx}.png')