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



def load_envi_hsi_by_wavelength(filepath, start_wavelength, end_wavelength):
    """
    Load an ENVI hyperspectral image (HSI) from a specific range of wavelengths.
    
    Parameters:
        filepath (str): Path to the ENVI HSI file.
        start_wavelength (float): Starting wavelength (in nanometers).
        end_wavelength (float): Ending wavelength (in nanometers).
    
    Returns:
        numpy.ndarray: HSI data cube containing the specified range of wavelengths.
    """
    # Open the HSI file
    hsi_data = spectral.envi.open(filepath)
    
    # Get wavelength information
    wavelengths = hsi_data.bands.centers
    
    # Find bands within the specified range of wavelengths
    selected_bands = [i for i, w in enumerate(wavelengths) if start_wavelength <= w <= end_wavelength]
    
    if not selected_bands:
        raise ValueError("No bands found within the specified range of wavelengths.")
    
    # Read the selected bands
    hsi_subset = hsi_data.read_bands(selected_bands)
    
    selected_wavelengths = [wavelengths[i] for i in selected_bands]

    return hsi_subset, selected_wavelengths


def data_correction(raw_data, dark_data, white_data):
    """"
    Performing Calibration of HSI data
     
    """
    
    corrected_data = np.divide(
            np.subtract(raw_data, dark_data),
            np.subtract(white_data, dark_data))
    return corrected_data

def get_band_index(wavelengths, target_wavelength):
    """
    Get the index position of a particular wavelength in the list of wavelengths.
    
    Parameters:
        wavelengths (list): List of wavelengths.
        target_wavelength (float): Wavelength to search for.
    
    Returns:
        int: Index position of the target wavelength in the list of wavelengths, or -1 if not found.
    """
    try:
        index = wavelengths.index(target_wavelength)
        return index
    except ValueError:
        return -1


def get_right_clicks(image):
    """
    Get right-click events and store the pixel locations of the clicks.

    Args:
    - image: The input image (numpy array).

    Returns:
    - List of pixel locations of the right-click events.
    """

    clicks = []  # Initialize clicks list to store pixel locations

    def mouse_callback(event, x, y, flags, params):
        """
        Callback function to handle mouse events.
        """
        if event == cv2.EVENT_RBUTTONDOWN:  # Right-click event
            clicks.append((x, y))  # Store the pixel location of the click

    # Create a window and set the mouse callback function
    cv2.namedWindow('Click the points')
    cv2.setMouseCallback('Click the points', mouse_callback)

    # Display the image and wait for right-click events
    while True:
        cv2.imshow('Click the points', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit capturing clicks
            break

    # Destroy the window and return the list of clicks
    cv2.destroyAllWindows()
    return clicks




def extract_rois(image, clicked_points, box_size):
    """
    Extract ROIs (Regions of Interest) from an image based on clicked points
    and assign a bounding box of specific size for each click.

    Args:
    - image: The input image (numpy array).
    - clicked_points: List of tuples containing (x, y) coordinates of clicked points.
    - box_size: Tuple containing (width, height) of the bounding box.

    Returns:
    - List of extracted ROIs.
    """

    rois = []

    # Iterate over clicked points
    for point in clicked_points:
        x, y = point

        # Calculate bounding box coordinates
        x1 = max(0, x - box_size[0] // 2)
        y1 = max(0, y - box_size[1] // 2)
        x2 = min(image.shape[1], x1 + box_size[0])
        y2 = min(image.shape[0], y1 + box_size[1])

        # Extract ROI
        roi = image[y1:y2, x1:x2]

        rois.append(roi)

    return rois

def calculate_average_reflectance(roi):
    """
    Calculate the average reflectance for an ROI.

    Args:
    - roi: The ROI (numpy array).

    Returns:
    - The average reflectance for the ROI.
    """
    # Assuming each channel represents reflectance at a different wavelength,
    # calculate the average pixel value across all channels
    average_reflectance = np.mean(roi, axis=(0, 1))
    return average_reflectance


def plot_reflectance_data(rois, wavelengths, output_filename='reflectance_plot.png'):
    """
    Plot each ROI's average reflectance versus the wavelength along with the reflectance data for reference.

    Args:
    - rois: List of ROIs.
    - wavelengths: Array of wavelengths.
    - reflectance_data: Reflectance data for reference.
    - output_filename: Output filename to save the plot as an image file. Default is 'reflectance_plot.png'.
    """

    # Calculate average reflectance for each ROI
    average_reflectances = [calculate_average_reflectance(roi) for roi in rois]

    # Plot each ROI's average reflectance versus the wavelength
    for i, average_reflectance in enumerate(average_reflectances):
        plt.plot(wavelengths, average_reflectance, label=f'ROI {i+1}')



    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Average Reflectance vs. Wavelength')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig(output_filename)

    # Display a message indicating where the plot is saved
    print(f"Plot saved as '{output_filename}'")

def calculate_mean(roi):
    """
    Calculate the mean of an ROI.

    Args:
    - roi: The ROI (numpy array).

    Returns:
    - The mean of the ROI.
    """
    return np.mean(roi)

def calculate_standard_deviation(roi):
    """
    Calculate the standard deviation of an ROI.

    Args:
    - roi: The ROI (numpy array).

    Returns:
    - The standard deviation of the ROI.
    """
    return np.std(roi)

def calculate_variance(roi):
    """
    Calculate the variance of an ROI.

    Args:
    - roi: The ROI (numpy array).

    Returns:
    - The variance of the ROI.
    """
    return np.var(roi)

def calculate_statistics_for_rois(rois):
    """
    Calculate mean, standard deviation, and variance for each ROI.

    Args:
    - rois: List of ROIs (each ROI is a numpy array).

    Returns:
    - List of tuples (mean, standard deviation, variance) for each ROI.
    """
    statistics = []
    for roi in rois:
        mean = calculate_mean(roi)
        std_deviation = calculate_standard_deviation(roi)
        variance = calculate_variance(roi)
        statistics.append((mean, std_deviation, variance))
    
    
    return statistics

def plot_statistics_for_rois(rois, output_filename='statistics_plot.png'):
    """
    Plot and save the statistics (mean, standard deviation, variance) for each ROI.

    Args:
    - rois: List of ROIs (each ROI is a numpy array).
    - output_filename: Output filename to save the plot as an image file. Default is 'statistics_plot.png'.
    """
    # Calculate statistics for each ROI
    roi_statistics = []
    for roi in rois:
        mean = calculate_mean(roi)
        std_deviation = calculate_standard_deviation(roi)
        variance = calculate_variance(roi)
        roi_statistics.append((mean, std_deviation, variance))

    # Extract statistics for plotting
    means = [stat[0] for stat in roi_statistics]
    std_deviations = [stat[1] for stat in roi_statistics]
    variances = [stat[2] for stat in roi_statistics]

    # Plotting
    labels = [f"ROI {i+1}" for i in range(len(rois))]
    x = np.arange(len(rois))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, means, width, label='Mean')
    ax.bar(x, std_deviations, width, label='Standard Deviation')
    ax.bar(x + width, variances, width, label='Variance')

    ax.set_ylabel('Values')
    ax.set_title('Statistics Comparison Across ROIs')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(output_filename)

    # Display a message indicating where the plot is saved
    print(f"Plot saved as '{output_filename}'")
    
def envi_loading(raw,dark,white,start_wl, end_wl):
    start_wl=float(start_wl)
    end_wl=float(end_wl)
    hsi_data_raw, bandss = load_envi_hsi_by_wavelength(raw, start_wl, end_wl)
    print("Done loading raw")
    hsi_data_white, _ = load_envi_hsi_by_wavelength(white, start_wl, end_wl)
    print("Done loading white")
    hsi_data_dark, _ = load_envi_hsi_by_wavelength(dark, start_wl, end_wl)
    print("Done loading dark")
    
    return hsi_data_raw, hsi_data_dark, hsi_data_white, bandss