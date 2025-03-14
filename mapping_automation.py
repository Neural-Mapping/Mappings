import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()
import ssl
# from utils import get_access_token
import time
from typing import Dict, Tuple
import pandas as pd
import math
from scripts import *
from utils import *
import folium
import ee
import geemap
import shutil
import re

CLIENT_ID = os.environ.get("sh_client_id")
CLIENT_SECRET = os.environ.get("sh_client_secret")

def get_access_token():
    url = "https://services.sentinel-hub.com/oauth/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None



# Search for available Sentinel-2 images
def search_available_dates(lat, lon, target_date, max_days=30):
    access_token = get_access_token()
    
    target_date = datetime.strptime(target_date, "%Y-%m-%d")
    start_date = (target_date - timedelta(days=max_days)).strftime("%Y-%m-%d")
    end_date = (target_date + timedelta(days=max_days)).strftime("%Y-%m-%d")
    
    url = "https://services.sentinel-hub.com/api/v1/catalog/search"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "collections": ["sentinel-2-l2a"],
        "limit": 100,
        "intersects": {
            "type": "Point",
            "coordinates": [lon, lat]
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        print("Error:", response.json())
        return None
    
    data = response.json().get("features", [])
    
    if not data:
        print("No available data found.")
        return None
    
    # Extract dates from the results
    available_dates = sorted([datetime.strptime(item["properties"]["datetime"], "%Y-%m-%dT%H:%M:%SZ") for item in data])
    return [i.strftime("%Y-%m-%d") for i in available_dates]



def get_cloud_coverage(lat, lon, date_list):
    access_token = get_access_token()
    
    url = "https://services.sentinel-hub.com/api/v1/catalog/search"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    cloud_coverage_results = {}
    for date in date_list:
        start_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%dT00:00:00Z")
        end_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%dT23:59:59Z")
        
        payload = {
            "datetime": f"{start_date}/{end_date}",
            "collections": ["sentinel-2-l2a"],
            "limit": 1,  # Get only the first available image for that date
            "intersects": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            print(f"Error fetching data for {date}: {response.json()}")
            cloud_coverage_results[date] = None
            continue
        
        data = response.json().get("features", [])
        
        if not data:
            cloud_coverage_results[date] = None
        else:
            cloud_coverage = data[0]["properties"].get("eo:cloud_cover", "Unknown")
            cloud_coverage_results[date] = cloud_coverage
    
    return cloud_coverage_results

def find_best_date(available_dates_cloud_coverage: Dict[str, float], target_date: str) -> Tuple[str, float, str, float]:
    min_before, min_after = ("", float("inf")), ("", float("inf"))

    for date, cc in available_dates_cloud_coverage.items():
        parsed_date = time.strptime(date, "%Y-%m-%d")
        target_parsed = time.strptime(target_date, "%Y-%m-%d")

        if parsed_date < target_parsed and cc <= min_before[1]:
            min_before = (date, cc)
        elif parsed_date >= target_parsed and cc < min_after[1]:
            min_after = (date, cc)

    return min_before[0], min_before[1], min_after[0], min_after[1]

def subtract_km_from_coordinates(lat: float, lon: float, km: float) -> Tuple[float, float]:
    # Earth radius in km
    earth_radius = 6371

    # Degree change per km
    delta_lat = km / earth_radius
    delta_lon = km / (earth_radius * math.cos(math.radians(lat)))

    # Subtract the distance
    new_lat = lat - delta_lat
    new_lon = lon - delta_lon

    return new_lat, new_lon

def get_slope_elevation(g):
    ee.Authenticate()
    ee.Initialize()

    for cords in g:
        srtm = ee.Image("USGS/SRTMGL1_003")
        terrain = ee.Algorithms.Terrain(srtm)
        elevation = srtm.select('elevation')  # Elevation data
        slope = terrain.select('slope')
        roi = ee.Geometry.Rectangle([cords[1], cords[0], cords[3], cords[2]])

        Map = geemap.Map()
        Map.centerObject(roi, 16)
        Map.addLayer(elevation.clip(roi), {'min': 0, 'max': 3000, 'palette': ['black', 'white', 'gray']}, 'Elevation')
        Map.addLayer(slope.clip(roi), {'min': 0, 'max': 60, 'palette': ['black', 'white', 'gray']}, 'Slope')

        elevation_resampled = elevation.reproject(crs='EPSG:4326', scale=30)
        slope_resampled = slope.reproject(crs='EPSG:4326', scale=30)

        # Convert to NumPy arrays
        elevation_arr = geemap.ee_to_numpy(elevation_resampled, region=roi)
        slope_arr = geemap.ee_to_numpy(slope_resampled, region=roi)
        
    return slope_arr, elevation_arr

#### START #### 

if __name__ == "__main__":
    print(r"""
            ,-.
            / \  `.  __..-,O
        :   \ --''_..-'.'
        |    . .-' `. '.
        :     .     .`.'
            \     `.  /  ..
            \      `.   ' .
            `,       `.   \
            ,|,`.        `-.\
            '.||  ``-...__..-`
            |  |
            |__|
            /||\
            //||\\
        // || \\
        __//__||__\\__
    '--------------' 
    """)

    grid = 1
    box_dim = 5 # km

    columns_to_keep = ["event_date", "event_title", "latitude", "longitude"]
    nasa_landslides = pd.read_csv("nasa_global_landslide_catalog_point.csv", parse_dates=["event_date"], usecols=columns_to_keep)
    nasa_landslides["event_date"] = pd.to_datetime(nasa_landslides["event_date"], errors="coerce")
    df = nasa_landslides.dropna(subset=["event_date"])
    ten_days_ago = datetime.today() - timedelta(days=10)
    nasa_landslides = nasa_landslides[(nasa_landslides["event_date"] > "2018-01-01") & (nasa_landslides["event_date"] < ten_days_ago.date().strftime("%Y-%m-%d"))]
    nasa_landslides["event_date"] = nasa_landslides["event_date"].dt.date

    for landlside_row in range(nasa_landslides.shape[0]):
        project_name = nasa_landslides.iloc[landlside_row].event_title
        project_name = re.sub(r'[\/:*?"<>|]', '_', str(project_name)).strip()
        print("*"*20)

        with open("do_not_attend.txt", "r") as log_file:
            content = [line.strip() for line in log_file]
            if project_name in content: 
                print(f"{project_name} in DO NOT ATTEND: reason: Unknown") # havent figured this out yet lol
                continue

        # with open("Mapping Automation Completions.txt", "r") as log_file:
        #     content = [line.strip() for line in log_file]
        #     if project_name in content: 
        #         print(f"Skipping: {project_name}: Already exists")
        #         continue
        
        print("GETTING BEST DATES\n")
        date = nasa_landslides.iloc[landlside_row].event_date.strftime("%Y-%m-%d")
        lat, lon = subtract_km_from_coordinates(nasa_landslides.iloc[1].latitude, nasa_landslides.iloc[landlside_row].longitude, 1.5)
        available_dates = search_available_dates(target_date=nasa_landslides.iloc[landlside_row].event_date.strftime("%Y-%m-%d"),
                                                lat = lat,
                                                lon = lon)
        if available_dates == None: 
            print(f"Skipping: {project_name}: Dates not available")
            with open("do_not_attend.txt", "a") as log_file:
                log_file.write(f"{project_name}\n")
            continue
        available_dates_cloud_coverage = get_cloud_coverage(lat=lat, lon=lon, date_list=available_dates)
        
        min_before_date, min_before_cc, min_after_date, min_after_cc = find_best_date(available_dates_cloud_coverage, target_date=date)

        min_lat, min_lon  = lat, lon
        start_date = min_before_date
        end_date = min_after_date
        # project_name = project_name
        if start_date == "" or end_date == "":
            print(f"Skipping: {project_name}: start date or end date not found")
            # shutil.rmtree(project_name)
            with open("do_not_attend.txt", "a") as log_file:
                log_file.write(f"{project_name}\n")
            continue
        
        os.makedirs(project_name, exist_ok=True)
        with open(f'{project_name}/log.txt', 'w') as f:
            f.write(f'Project Name: {project_name}\n')
            f.write(f'Grid size: {grid} x {grid}\n')
            f.write(f'Box Dimension: {box_dim} km\n')
            f.write(f'Bounding Box: {min_lat}, {min_lon}\n')
            f.write(f'Start Date: {start_date}\n')
            f.write(f'End Date: {end_date}\n')
        
        g = generate_grid(min_lat, min_lon, distance=box_dim*1000, grid_side=grid)
        
        print("GETTING DIFFERENT FEATURES\n")
        NDVI_Before = get_images(km=box_dim, grid=g, grid_dim=grid, script=evalscript_NDVI, 
            date_start=start_date, date_end=start_date, res=1000, 
            box_dim=box_dim, file_name=f"{project_name}/{project_name}-NDVI-Before")

        NDVI_After = get_images(km=box_dim, grid=g, grid_dim=grid, script=evalscript_NDVI, 
                date_start=end_date, date_end=end_date, res=1000, 
                box_dim=box_dim, file_name=f"{project_name}/{project_name}-NDVI-After")

        True_Color_After = get_images(km=box_dim, grid=g, grid_dim=grid, script=evalscript_True_Color, 
                date_start=end_date, date_end=end_date, res=1000, 
                box_dim=box_dim, file_name=f"{project_name}/{project_name}-True_Color-After")

        LSM_Only_After = get_images(km=box_dim, grid=g, grid_dim=grid, script=evalscript_lsm_only, 
                date_start=end_date, date_end=end_date, res=1000, 
                box_dim=box_dim, file_name=f"{project_name}/{project_name}-LSM_Only-After")

        NDWI_Before  = get_images(km=box_dim, grid=g, grid_dim=grid, script=evalscript_NDWI, 
                date_start=end_date, date_end=end_date, res=1000, 
                box_dim=box_dim, file_name=f"{project_name}/{project_name}-NDWI-Before")

        slope_arr, elevation_arr = get_slope_elevation(g)
        slope_arr = cv2.resize(slope_arr, (1000,1000))
        elevation_arr = cv2.resize(elevation_arr, (1000,1000))

        plt.imsave(f"{project_name}/{project_name}-Slope.png", slope_arr)
        plt.imsave(f"{project_name}/{project_name}-Elevation.png", elevation_arr)

        diff = NDVI_Before - NDVI_After
        threshold = 0
        tolerance = 60
        mask = ((NDVI_Before > NDVI_After + tolerance) & (diff > threshold)).astype(np.uint8) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask_resized = cv2.resize(mask, (mask.shape[0]//4, mask.shape[0]//4))
        mask = cv2.resize(mask_resized, mask.shape)
        mask = cv2.threshold(mask, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        plt.imsave(f"{project_name}/{project_name}-dNDVI-masked.png", mask, cmap='gray')

        no_mask = np.ones_like(True_Color_After) * 255
        LMS_True_Color_dNDVI_not_Masked = True_Color_After.copy()
        mask_condition = no_mask == 255
        LMS_True_Color_dNDVI_not_Masked[mask_condition] = LSM_Only_After[mask_condition]
        black_pixels = np.all(LMS_True_Color_dNDVI_not_Masked == [0, 0, 0], axis=-1)
        LMS_True_Color_dNDVI_not_Masked[black_pixels] = True_Color_After[black_pixels]

        # plt.imshow(LMS_True_Color_dNDVI_not_Masked); plt.axis("off")
        plt.title("LMS True Color Not Masked")
        plt.imsave(f"{project_name}/{project_name}-LMS_True_Color_dNDVI_Not_Masked.png", LMS_True_Color_dNDVI_not_Masked)

        LMS_True_Color_dNDVI_Masked = True_Color_After.copy()
        LMS_True_Color_dNDVI_Masked[mask[:,:, None].repeat(3, -1) == 255] = LSM_Only_After[mask[:,:, None].repeat(3, -1) == 255]

        # replace black spots with true color (i should have figured this out since the start lol)
        black_mask = np.all(LMS_True_Color_dNDVI_Masked == [0, 0, 0], axis=-1)
        output = LMS_True_Color_dNDVI_Masked.copy()
        LMS_True_Color_dNDVI_Masked[black_mask] = True_Color_After[black_mask]

        # plt.imshow(LMS_True_Color_dNDVI_Masked); plt.axis("off")
        plt.title("LMS True Color Masked")
        plt.imsave(f"{project_name}/{project_name}-LMS_True_Color_dNDVI_Masked.png", LMS_True_Color_dNDVI_Masked)

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        # First row
        axes[0, 0].imshow(NDVI_Before)
        axes[0, 0].set_title("NDVI Before")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(NDVI_After)
        axes[0, 1].set_title("NDVI After")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(True_Color_After)
        axes[0, 2].set_title("True Color After")
        axes[0, 2].axis("off")

        # Second row
        axes[1, 0].imshow(LSM_Only_After)
        axes[1, 0].set_title("LSM Only After")
        axes[1, 0].axis("off")

        # Add the mask visualization in grayscale
        axes[1, 1].imshow(mask, cmap="gray")
        axes[1, 1].set_title("Mask")
        axes[1, 1].axis("off")

        # Compute masked image
        axes[1, 2].imshow(LMS_True_Color_dNDVI_Masked, cmap="gray")
        axes[1, 2].set_title("LMS True Color Masked")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(f"{project_name}/{project_name}-combined_image.png")
        # plt.show()

        with open("Mapping Automation Completions.txt", "a") as log_file:
            log_file.write(f"{project_name}\n")
        with open("do_not_attend.txt", "a") as log_file:
            log_file.write(f"{project_name}\n")
                        
        plt.close()

        