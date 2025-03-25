import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sentinelhub import SHConfig
from sentinelhub import (
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
)
import folium
from tqdm import tqdm
import math

from sentinelhub import SHConfig
from dotenv import load_dotenv
import os
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from sentinelhub import CRS, BBox, bbox_to_dimensions
import cv2
import ee

config_sentinel = SHConfig(sh_client_id=os.environ.get("sh_client_id"), sh_client_secret=os.environ.get("sh_client_secret"))

from mapping_automation import find_best_date, subtract_km_from_coordinates, get_cloud_coverage, search_available_dates, get_access_token, get_slope_elevation
from scripts import *
from utils import *
import os
from dotenv import load_dotenv
load_dotenv()


def str_to_bool(value):
    """Convert 'true'/'false' string to boolean."""
    if value.lower() in ["true", "1", "yes"]:
        return True
    elif value.lower() in ["false", "0", "no"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

true_color_path = "-True_Color-After.png"
ndvi_before_path = "-NDVI-Before.png"
ndvi_after_path = "-NDVI-After.png"
slope_path = "-Slope.png"
elevation_path = "-Elevation.png"
NDWI_before_path = "-NDWI-Before.png"
LSM_masked_path = "-LSM_True_Color-After.png"
mask_path = "-dNDVI-masked.png"

if __name__ == "__main__":

    print(r"""
         ooo
        / : \
       / o0o \
 _____"~~~~~~~"_____
 \+###|U * * U|###+/
  \...!(.>..<)!.../
   ^^^^o|   |o^^^^
+=====}:^^^^^:{=====+#
.____  .|!!!|.  ____.
|#####:/" " "\:#####|
|#####=|  O  |=#####|
|#####>\_____/<#####|
 ^^^^^   | |   ^^^^^
         o o
    """)
        
    parser = argparse.ArgumentParser(description="Run grid scan with parameters.")

    # Define required arguments
    parser.add_argument("-pn", "--project_name", type=str, required=True, help="Name of the project (a new folder will be created)")
    parser.add_argument("-d", "--date", type=str, required=True, help="Date around which the grid scan is to happen")
    parser.add_argument("-lat", "--latitude", type=float, required=True, help="Latitude cordinate)")
    parser.add_argument("-lon", "--longitude", type=float, required=True, help="Longitude cordinate")
    parser.add_argument("-dim", "--dimention", type=int, required=True, help="Length of each side of each grid (in kilometer)")
    parser.add_argument("-grids", "--num_grids", type=int, required=True, help="Number of grids (nxn)")
    parser.add_argument("-res", "--resolution", type=int, required=True, help="Resolution of images being fetched")
    parser.add_argument("-new_dates", "--get_new_dates_for_each_grid", type=str_to_bool, required=True, help="Use all resources (true/false)")

    # Parse arguments
    args = parser.parse_args()

    proj_name = args.project_name
    lat = args.latitude
    lon = args.longitude
    box_dim = args.dimention
    grid = args.num_grids
    date = args.date
    res = args.resolution

    g = generate_grid(lat, lon, distance=box_dim*1000, grid_side=grid)

    row = 0
    col = 0
    get_new_dates_for_each = True

    _box_dim = 1000 if box_dim < 100 else box_dim
    canvas_NDVI_Before = np.zeros(((grid) * _box_dim, (grid) * _box_dim, 3), dtype=np.uint8)
    canvas_NDVI_After = np.zeros(((grid) * _box_dim, (grid) * _box_dim, 3), dtype=np.uint8)
    canvas_True_Color_After = np.zeros(((grid) * _box_dim, (grid) * _box_dim, 3), dtype=np.uint8)
    canvas_LSM_Only_After = np.zeros(((grid) * _box_dim, (grid) * _box_dim, 3), dtype=np.uint8)

    for idx, i in enumerate(range(len(g))):

        if get_new_dates_for_each or idx==0:
            lat, lon = g[idx][0], g[idx][1]
            print("Getting dates for new lat lon")
            available_dates = search_available_dates(target_date=date,
                                                    lat = lat,
                                                    lon = lon)
            available_dates_cloud_coverage = get_cloud_coverage(lat=lat, lon=lon, date_list=available_dates)
            min_before_date, min_before_cc, min_after_date, min_after_cc = find_best_date({k: v for k, v in available_dates_cloud_coverage.items() if v is not None}, target_date=date)
            print(min_before_date,":" , min_before_cc, min_after_date,":" ,min_after_cc)

        y_start = _box_dim * col
        y_end = _box_dim * (col + 1)
        x_start = _box_dim * row
        x_end = _box_dim * (row + 1)

        print("NDVI_Before")
        NDVI_Before = get_images(km=box_dim, grid=[g[i]], grid_dim=1, script=evalscript_NDVI, 
            date_start=min_before_date, date_end=min_before_date, res=res, 
            box_dim=box_dim)

        canvas_NDVI_Before[y_start:y_end, x_start:x_end] = NDVI_Before

        print("NDVI_After")
        NDVI_After = get_images(km=box_dim, grid=[g[i]], grid_dim=1, script=evalscript_NDVI, 
                    date_start=min_after_date, date_end=min_after_date, res=res, 
                    box_dim=box_dim)
        canvas_NDVI_After[y_start:y_end, x_start:x_end] = NDVI_After

        print("True_Color_After")
        True_Color_After = get_images(km=box_dim, grid=[g[i]], grid_dim=1, script=evalscript_True_Color, 
                date_start=min_after_date, date_end=min_after_date, res=res, 
                box_dim=box_dim)
        canvas_True_Color_After[y_start:y_end, x_start:x_end] = True_Color_After

        print("LSM_Only_After")
        LSM_Only_After = get_images(km=box_dim, grid=[g[i]], grid_dim=1, script=evalscript_lsm_only, 
                    date_start=min_after_date, date_end=min_after_date, res=res, 
                    box_dim=box_dim)
        canvas_LSM_Only_After[y_start:y_end, x_start:x_end] = LSM_Only_After

        row += 1  # Move to the next column
        if (idx + 1) % math.sqrt(len(g)) == 0:
            print("----") 
            col += 1  # Move to the next row
            row = 0  # Reset column position

    diff = canvas_NDVI_Before - canvas_NDVI_After
    threshold = 0
    tolerance = 60
    mask = ((canvas_NDVI_Before > canvas_NDVI_After + tolerance) & (diff > threshold)).astype(np.uint8) * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask_resized = cv2.resize(mask, (mask.shape[0]//4, mask.shape[0]//4))
    mask = cv2.resize(mask_resized, mask.shape)
    mask = cv2.threshold(mask, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    no_mask = np.ones_like(canvas_True_Color_After) * 255
    LMS_True_Color_dNDVI_not_Masked = canvas_True_Color_After.copy()
    mask_condition = no_mask == 255
    LMS_True_Color_dNDVI_not_Masked[mask_condition] = canvas_LSM_Only_After[mask_condition]
    black_pixels = np.all(LMS_True_Color_dNDVI_not_Masked == [0, 0, 0], axis=-1)
    LMS_True_Color_dNDVI_not_Masked[black_pixels] = canvas_True_Color_After[black_pixels]

    LMS_True_Color_dNDVI_Masked = canvas_True_Color_After.copy()
    LMS_True_Color_dNDVI_Masked[mask[:,:, None].repeat(3, -1) == 255] = canvas_LSM_Only_After[mask[:,:, None].repeat(3, -1) == 255]

    black_mask = np.all(LMS_True_Color_dNDVI_Masked == [0, 0, 0], axis=-1)
    output = LMS_True_Color_dNDVI_Masked.copy()
    LMS_True_Color_dNDVI_Masked[black_mask] = canvas_True_Color_After[black_mask]

    os.makedirs(proj_name, exist_ok=True)
    plt.imsave(f"{proj_name}/{proj_name}{true_color_path}", canvas_True_Color_After)
    plt.imsave(f"{proj_name}/{proj_name}{ndvi_before_path}", canvas_NDVI_Before)
    plt.imsave(f"{proj_name}/{proj_name}{ndvi_after_path}", canvas_NDVI_After)
    plt.imsave(f"{proj_name}/{proj_name}-dNDVI_Mask.png", mask, cmap="gray")
    plt.imsave(f"{proj_name}/{proj_name}{LSM_masked_path}", LMS_True_Color_dNDVI_Masked)
    plt.imsave(f"{proj_name}/{proj_name}-LSM_Only.png", canvas_LSM_Only_After)
    plt.imsave(f"{proj_name}/{proj_name}-LSM_Masked_Only.png", canvas_LSM_Only_After*mask[:, :, None])