import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import re

os.makedirs("Dataset", exist_ok=True)
canvas_size = 512

with open("Mapping Automation Completions.txt", "r") as file:
    lines = file.readlines()
    for file_name in tqdm(lines):
        file_name = file_name.strip()
        try:
            file_name = file_name.replace("/", "_")

            # file_name = "Mangshilla Landslide"

            true_color_path = "-True_Color-After.png"
            ndvi_before_path = "-NDVI-Before.png"
            slope_path = "-Slope.png"
            elevation_path = "-Elevation.png"
            NDWI_before_path = "-NDWI-Before.png"
            LSM_not_masked_path = "-LSM_Only-After.png"
            mask_path = "-dNDVI-masked.png"

            lsm_not_masked = plt.imread(f"{file_name}/{file_name}{LSM_not_masked_path}")
            mask = plt.imread(f"{file_name}/{file_name}{mask_path}")
            lsm_masked = lsm_not_masked * mask

            assert canvas_size <= lsm_masked.shape[0]

            canvas = np.zeros((canvas_size, canvas_size*6, lsm_masked.shape[-1]))
            canvas[:, :512, :] = cv2.resize(plt.imread(f"{file_name}/{file_name}{true_color_path}"), (canvas_size, canvas_size)) # True Color
            canvas[:, 512:512*2, :] = cv2.resize(plt.imread(f"{file_name}/{file_name}{ndvi_before_path}"), (canvas_size, canvas_size)) # NDVI
            canvas[:, 512*2:512*3, :] = cv2.resize(plt.imread(f"{file_name}/{file_name}{slope_path}"), (canvas_size, canvas_size)) # Slope
            canvas[:, 512*3:512*4, :] = cv2.resize(plt.imread(f"{file_name}/{file_name}{elevation_path}"), (canvas_size, canvas_size)) # Elevation
            canvas[:, 512*4:512*5, :] = cv2.resize(plt.imread(f"{file_name}/{file_name}{NDWI_before_path}"), (canvas_size, canvas_size)) # NDWI
            canvas[:, 512*5:, :] = cv2.resize(lsm_masked, (canvas_size, canvas_size)) # LSM
            canvas = (canvas * 255).astype(np.uint8)

            plt.imsave(f"Dataset/{file_name}.jpg", canvas)
        except:
            print("Skip")
            pass



