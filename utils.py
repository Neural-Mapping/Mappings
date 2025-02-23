import math
import os
from sentinelhub import SHConfig
from sentinelhub import (
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
    CRS,
    BBox,
)
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sentinelhub import SHConfig
config_sentinel = SHConfig(sh_client_id=os.environ.get("sh_client_id"), sh_client_secret=os.environ.get("sh_client_secret"))
from dotenv import load_dotenv
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def generate_grid(top_left_lat, top_left_lon, grid_side = 9, distance=400000):
    R = 6371000  # Earth's radius in meters
    grid = []
    
    # Convert top-left latitude to radians
    top_left_lat_rad = math.radians(top_left_lat)

    # Compute shifts in degrees
    delta_lat = (distance / R) * (180 / math.pi)
    delta_lon = (distance / (R * math.cos(top_left_lat_rad))) * (180 / math.pi)

    # Generate grid (9x9)
    for row in range(grid_side):  # Move downward
        for col in range(grid_side):  # Move right
            min_lat = top_left_lat - (row * delta_lat)  # Move south
            min_lon = top_left_lon + (col * delta_lon)  # Move east
            max_lat = min_lat + delta_lat
            max_lon = min_lon + delta_lon
            grid.append([min_lat, min_lon, max_lat, max_lon])
    
    return grid


def get_suseptibility_mapping(cordinates, script, box_dim=400, date_start = "2024-04-12", date_end = "2024-04-12", res=2100):
    min_lat, min_lon, max_lat, max_lon  = cordinates

    cords = [min_lon, min_lat, max_lon, max_lat]

    bbox = BBox(bbox=cords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=box_dim*1000/res)

    request_lms_color = SentinelHubRequest(
            evalscript=script,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(date_start, date_end),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=bbox,
            size=size,
            config=config_sentinel,
        )

    lms_response = request_lms_color.get_data()
    return lms_response[0]

def get_images(km, grid, grid_dim, script, box_dim=400, date_start = "2024-04-12", date_end = "2024-04-12", res=2100, file_name=None):
    _box_dim = 1000 if km < 100 else km
    canvas = np.zeros(((grid_dim) * _box_dim, (grid_dim) * _box_dim, 3), dtype=np.uint8)

    row = 0
    col = 0

    for idx, i in enumerate(range(len(grid))):
        y_start = _box_dim * col
        y_end = _box_dim * (col + 1)
        x_start = _box_dim * row
        x_end = _box_dim * (row + 1)

        print(col, row, grid[idx], "->", y_start, y_end, x_start, x_end) 

        image_rgba = cv2.resize(
        get_suseptibility_mapping(grid[idx], script, date_start=date_start, date_end=date_end, res=res, box_dim=box_dim), (_box_dim,_box_dim)
        )
        if image_rgba.shape[-1] == 4:
            image_rgb = image_rgba[..., :3]
        else: image_rgb = image_rgba
        canvas[y_start:y_end, x_start:x_end] = image_rgb

        if file_name: 
            plt.imsave(f"{file_name}.png", canvas)
            print(f"Saved: {file_name}.png")

        row += 1  # Move to the next column
        if (idx + 1) % math.sqrt(len(grid)) == 0:
            print("----") 
            col += 1  # Move to the next row
            row = 0  # Reset column position
    return canvas

def combine_images_based_on_mask(image1, image2, mask, threshold=0, blur = 1):
    """
    Combine two images based on a binary mask with thresholding.
    If the mask value is greater than the threshold, overwrite image2 with image1.

    Args:
        image1 (numpy.ndarray): The first image (will overwrite image2 where mask > threshold).
        image2 (numpy.ndarray): The second image.
        mask (numpy.ndarray): The binary mask (same size as the images).
        threshold (int, float): The threshold to control when image1 overwrites image2.
    
    Returns:
        numpy.ndarray: The combined image.
    """
    # Ensure mask is binary and has the same dimensions as the images
    mask = mask.astype(np.uint8)

    # Apply the threshold to the mask (values above threshold will be 1, others 0)
    # binary_mask = (mask > threshold).astype(np.uint8) * 255

    # Combine the images based on the binary mask
    mask = cv2.blur(mask, (blur,blur))
    combined_image = np.where(mask[:, :, None] > threshold, image1, image2)
    
    return combined_image
