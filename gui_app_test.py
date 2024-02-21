import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import cv2

from utils import *

def create_hsi_analysis_app():
    root = tk.Tk()
    app = HSIAnalysisApp(root)
    return app

class HSIAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("HSI Analysis Tool")

        # Initialize references
        self.original_hsi_data = None
        self.dark_ref_data = None
        self.white_ref_data = None
        self.box_size_x = None
        self.box_size_y = None
        
        self.hsi_data_raw = []
        self.hsi_data_white = []
        self.hsi_data_dark = []
        self.bandss = []
        self.corrected_data =[]
        self.image = []
        self.boxes = ()
        self.rois =[]

        # Create the "Data Upload" section
        self.create_data_upload_section()

        # Create the "Data Preprocess" section
        self.create_data_preprocess_section()

    def load_original_hsi_data(self):
        filepath1 = filedialog.askopenfilename(title="Select original HSI data file")
        if filepath1:
            messagebox.showinfo("Information", f"Original HSI data loaded from {filepath1}")
            # Load and process original HSI data
            self.original_hsi_data = filepath1  # Replace None with your loaded data

    def load_dark_ref_data(self):
        filepath2 = filedialog.askopenfilename(title="Select dark reference data file")
        if filepath2:
            messagebox.showinfo("Information", f"Dark reference data loaded from {filepath2}")
            # Load and process dark reference data
            self.dark_ref_data = filepath2  # Replace None with your loaded data

    def load_white_ref_data(self):
        filepath3 = filedialog.askopenfilename(title="Select white reference data file")
        if filepath3:
            messagebox.showinfo("Information", f"White reference data loaded from {filepath3}")
            # Load and process white reference data
            self.white_ref_data = filepath3 # Replace None with your loaded data

    def create_data_upload_section(self):
        self.data_upload_frame = tk.Frame(self.master)
        self.data_upload_frame.pack(padx=20, pady=20)

        # Button to load original HSI data
        load_original_hsi_button = tk.Button(self.data_upload_frame, text="Load Original HSI Data", command=self.load_original_hsi_data)
        load_original_hsi_button.pack(side="left", padx=10)

        # Button to load dark reference data
        load_dark_ref_button = tk.Button(self.data_upload_frame, text="Load Dark Reference Data", command=self.load_dark_ref_data)
        load_dark_ref_button.pack(side="left", padx=10)

        # Button to load white reference data
        load_white_ref_button = tk.Button(self.data_upload_frame, text="Load White Reference Data", command=self.load_white_ref_data)
        load_white_ref_button.pack(side="left", padx=10)

    def load_hsi_data_by_wavelength(self, start_wl, end_wl):
        # Functionality to load HSI data by wavelength range using self.original_hsi_data, self.dark_ref_data, self.white_ref_data
        print(self.original_hsi_data)
        hsi_data_raw, hsi_data_dark, hsi_data_white, bandss = envi_loading(self.original_hsi_data, self.dark_ref_data, self.white_ref_data,start_wl, end_wl )
        print("Done loading all")
        #return hsi_data_raw, hsi_data_white, hsi_data_dark, bandss 
        
        self.hsi_data_raw = hsi_data_raw 
        self.hsi_data_white = hsi_data_white
        self.hsi_data_dark = hsi_data_dark
        self.bandss = bandss

    def preprocess_data(self):
        print("*****************************************")
        print("Calibrating HSI")
        print("*****************************************")
        
        # Functionality to preprocess data using self.original_hsi_data, self.dark_ref_data, self.white_ref_data
        corrected_data = data_correction(self.hsi_data_raw, self.hsi_data_dark, self.hsi_data_white)
        print("Done Corrected HSI shape:", corrected_data.shape)
        self.corrected_data = corrected_data
        
    def catch_boxsize(self, box_size_x, box_size_y):
        boxes = (int(box_size_x), int(box_size_y))
        self.boxes = boxes
   
    def point_selector(self):
        R = get_band_index(self.bandss,650.45)
        G = get_band_index(self.bandss,540.62)
        B = get_band_index(self.bandss,460.27)
        #Get RGB Image (BGR)
        img = get_rgb(self.corrected_data, bands=[B,G,R])
        image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        self.image = image.astype(np.uint8)
        
        clicks = get_right_clicks(self.image)
        box_size = self.boxes

        print(box_size)
        # Extract ROIs
        rois = extract_rois(self.corrected_data, clicks, box_size)

        # Draw bounding boxes
        for i, point in enumerate(clicks):
            x, y = point
            x1 = max(0, x - box_size[0] // 2)
            y1 = max(0, y - box_size[1] // 2)
            x2 = min(self.image.shape[1], x1 + box_size[0])
            y2 = min(self.image.shape[0], y1 + box_size[1])
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.image, str(i + 1), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the image with bounding boxes
        cv2.imshow('Image with Bounding Boxes', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        self.rois = rois
        # Functionality to preprocess data using self.original_hsi_data, self.dark_ref_data, self.white_ref_data
        #corrected_data = data_correction(self.hsi_data_raw, self.hsi_data_dark, self.hsi_data_white)
        #print("Done Corrected HSI shape:", corrected_data.shape)
       # self.corrected_data = corrected_data
    
    def get_reflectance(self):
        plot_reflectance_data(self.rois,self.bandss)
        plot_statistics_for_rois(self.rois)
        print('Please, Check your path for file')
        
    
    def create_data_preprocess_section(self):
        self.data_preprocess_frame = tk.Frame(self.master)
        self.data_preprocess_frame.pack(padx=20, pady=20)

        # Entry widgets for start and end wavelengths
        self.start_wl_entry = tk.Entry(self.data_preprocess_frame)
        self.start_wl_entry.grid(row=0, column=0, padx=5, pady=5)
        self.end_wl_entry = tk.Entry(self.data_preprocess_frame)
        self.end_wl_entry.grid(row=0, column=2, padx=5, pady=5)

        # Label for start and end wavelengths
        start_wl_label = tk.Label(self.data_preprocess_frame, text="Start Wavelength")
        start_wl_label.grid(row=1, column=0, padx=5, pady=5)
        end_wl_label = tk.Label(self.data_preprocess_frame, text="End Wavelength")
        end_wl_label.grid(row=1, column=2, padx=5, pady=5)

        # Button to load HSI data
        load_data_button = tk.Button(self.data_preprocess_frame, text="Load Data", command=lambda: self.load_hsi_data_by_wavelength(self.start_wl_entry.get(), self.end_wl_entry.get()))
        load_data_button.grid(row=2, column=0, columnspan=3, pady=10)

        # Button to preprocess data
        preprocess_data_button = tk.Button(self.data_preprocess_frame, text="Preprocess Data", command=self.preprocess_data)
        preprocess_data_button.grid(row=3, column=0, columnspan=3, pady=10)
        
        # Entry widgets for start and end wavelengths
        self.box_size_x_entry = tk.Entry(self.data_preprocess_frame)
        self.box_size_x_entry.grid(row=4, column=0, padx=5, pady=5)
        self.box_size_y_entry = tk.Entry(self.data_preprocess_frame)
        self.box_size_y_entry.grid(row=4, column=2, padx=5, pady=5)
        
        # Label for boxes
        start_X= tk.Label(self.data_preprocess_frame, text="Box Size (x)")
        start_X.grid(row=5, column=0, padx=5, pady=5)
        start_Y = tk.Label(self.data_preprocess_frame, text="Box Size (y)")
        start_Y.grid(row=5, column=2, padx=5, pady=5)
        
        # Button to boxes
        load_data_button = tk.Button(self.data_preprocess_frame, text="Accept", command=lambda: self.catch_boxsize(self.box_size_x_entry.get(), self.box_size_y_entry.get()))
        load_data_button.grid(row=6, column=0, columnspan=3, pady=10)
        
        # Button to preprocess data
        point_selector_button = tk.Button(self.data_preprocess_frame, text="ROI Selection", command=self.point_selector)
        point_selector_button.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Button to save Reflectance vs Wavelength plot
        point_selector_button = tk.Button(self.data_preprocess_frame, text="Get Reflectance and Stats", command=self.get_reflectance)
        point_selector_button.grid(row=8, column=0, columnspan=3, pady=10)


if __name__ == "__main__":
    app = create_hsi_analysis_app()
    app.master.mainloop()