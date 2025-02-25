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

def merge_images(image1, image2, mask):
    """
    Merges two images using a binary mask. 
    - image1 is placed where the mask is white (255).
    - image2 is used where the mask is black (0).
    - If a pixel in image1 is nearly white (≥ [150,150,150]), use image2.
    - If a pixel in image2 is black, replace it with image1.

    Args:
        image1 (numpy.ndarray): The top image.
        image2 (numpy.ndarray): The bottom image.
        mask (numpy.ndarray): Binary mask (255 for image1, 0 for image2).

    Returns:
        numpy.ndarray: The merged image.
    """
    # Normalize mask to binary (1 for white, 0 for black)
    mask = (mask == 255).astype(np.uint8)  

    # Expand mask to match the number of channels if needed
    if len(image1.shape) == 3 and len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)

    # Identify nearly white pixels in image1
    if len(image1.shape) == 3:  # RGB Image
        nearly_white_pixels = np.all(image1 >= [150, 150, 150], axis=-1)
        black_pixels_image2 = np.all(image2 == [0, 0, 0], axis=-1)  # Find black pixels in image2
    else:  # Grayscale Image
        nearly_white_pixels = image1 >= 150
        black_pixels_image2 = image2 == 0  # Find black pixels in image2

    # Update mask: if a pixel in image1 is nearly white, use image2 instead
    mask[nearly_white_pixels] = 0

    # Blend images based on mask
    merged = (image1 * mask) + (image2 * (1 - mask))

    # Replace black pixels in image2 with image1
    merged[black_pixels_image2] = image1[black_pixels_image2]

    return merged.astype(np.uint8)
