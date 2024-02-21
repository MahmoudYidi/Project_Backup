import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import cv2
from utils import *

# Create the main application window
app = tk.Tk()
app.title("HSI Analysis Tool")

# Function to load HSI data
def load_hsi_data():
    filepath = filedialog.askopenfilename(title="Select HSI data file")
    if filepath:
        messagebox.showinfo("Information", f"HSI data loaded from {filepath}")
        # Add your HSI data loading and processing code here
        # Example: hsi_data_raw, bandss = load_envi_hsi_by_wavelength(filepath, start_wl, end_wl)

# Function to process HSI data
def process_hsi_data():
    # Add your HSI data processing code here
    # Example: process_data(hsi_data_raw, bandss)
    messagebox.showinfo("Information", "HSI data processed successfully")

# Function to display HSI image
def display_hsi_image():
    # Add your HSI image display code here
    # Example: cv2.imshow('HSI Image', image)
    messagebox.showinfo("Information", "HSI image displayed")

# Function to display statistics
def display_statistics():
    # Add your statistics calculation and display code here
    # Example: calculate_statistics_for_rois(rois)
    messagebox.showinfo("Information", "Statistics displayed")

# Button to load HSI data
load_button = tk.Button(app, text="Load HSI Data", command=load_hsi_data)
load_button.pack()

# Button to process HSI data
process_button = tk.Button(app, text="Process HSI Data", command=process_hsi_data)
process_button.pack()

# Button to display HSI image
display_image_button = tk.Button(app, text="Display HSI Image", command=display_hsi_image)
display_image_button.pack()

# Button to display statistics
display_statistics_button = tk.Button(app, text="Display Statistics", command=display_statistics)
display_statistics_button.pack()

# Run the application
app.mainloop()