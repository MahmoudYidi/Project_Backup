
import spectral
import sys
import os



def read_file_paths_from_txt(file_paths_txt):
    """
    Read file paths from a text file and return them as a list.

    Parameters:
        file_paths_txt (str): Path to the text file containing file paths.

    Returns:
        list of str: List of file paths read from the text file.
    """
    # Initialize an empty list to store the file paths
    file_paths = []

    # Read file paths from the text file
    with open(file_paths_txt, 'r') as file:
        for line in file:
            # Remove leading and trailing whitespaces and newline characters
            line = line.strip()
            # Append the file path to the list
            file_paths.append(line)

    return file_paths

def load_envi_hsi_by_wavelengths_net(filepath_txt, start_wavelength, end_wavelength):
    """
    Load ENVI hyperspectral images (HSIs) from a specific range of wavelengths.
    
    Parameters:
        filepaths (list of str): List of file paths to the ENVI HSI files.
        start_wavelength (float): Starting wavelength (in nanometers).
        end_wavelength (float): Ending wavelength (in nanometers).
    
    Returns:
        list of numpy.ndarray: List of HSI data cubes containing the specified range of wavelengths.
        list of numpy.ndarray: List of selected wavelengths for each HSI data cube.
    """
    all_hsi_data = []
    all_selected_wavelengths = []
    filepaths = read_file_paths_from_txt(filepath_txt)
    
    for filepath in filepaths:
        # Open the HSI file
        hsi_data = spectral.envi.open(filepath)
        
        # Get wavelength information
        wavelengths = hsi_data.bands.centers
        
        # Find bands within the specified range of wavelengths
        selected_bands = [i for i, w in enumerate(wavelengths) if start_wavelength <= w <= end_wavelength]
        
        if not selected_bands:
            raise ValueError("No bands found within the specified range of wavelengths for file:", filepath)
        
        # Read the selected bands
        hsi_subset = hsi_data.read_bands(selected_bands)
        
        selected_wavelengths = [wavelengths[i] for i in selected_bands]
        
        # Append loaded data to lists
        all_hsi_data.append(hsi_subset)
        all_selected_wavelengths.append(selected_wavelengths)
    
    return all_hsi_data, all_selected_wavelengths