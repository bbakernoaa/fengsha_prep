import os
import glob
import datetime
import pandas as pd
import numpy as np
import xarray as xr
from satpy import Scene
from sklearn.cluster import DBSCAN
from dask.diagnostics import ProgressBar

# ==========================================
# CONFIGURATION
# ==========================================
# Select Satellite: 'goes16', 'goes17', 'goes18', 'himawari8', 'himawari9', 'seviri'
SAT_ID = 'goes16' 
REGION = 'meso'  # 'meso' (Mesoscale - high res), 'conus' (USA), 'full' (Full Disk)
START_TIME = datetime.datetime(2023, 4, 1, 18, 0) # Example: April 1, 2023 (Spring Dust Season)
END_TIME = datetime.datetime(2023, 4, 1, 20, 0)   # Short window for testing
OUTPUT_CSV = 'dust_events_catalog.csv'

# Dust Detection Thresholds (Based on standard EUMETSAT/NASA Dust RGB recipes)
# These may need fine-tuning for specific regions (e.g., US SW vs Sahara)
THRESHOLDS = {
    'diff_12_10': -0.5,   # Brightness Temp Difference (12.0 - 10.8 microns) -> Negative for dust
    'diff_10_8': 2.0,     # Brightness Temp Difference (10.8 - 8.7 microns) -> High for dust
    'temp_10': 280        # Minimum temperature (Kelvin) to filter out cold high clouds
}

# ==========================================
# 1. AUTOMATED DATA FETCHING (Using Satpy + S3)
# ==========================================
def get_file_pattern(sat_id):
    if 'goes' in sat_id:
        return 'abi_l1b' # Reader for GOES ABI
    elif 'himawari' in sat_id:
        return 'ahi_hsd' # Reader for Himawari AHI
    elif 'seviri' in sat_id:
        return 'seviri_l1b_native'
    else:
        raise ValueError("Unsupported satellite.")

def process_scene(scn_time):
    """
    Loads, processes, and detects dust for a single timestamp.
    """
    try:
        # Satpy can automatically find files in S3 buckets if configured, 
        # but for simplicity, we assume local files or use satpy's 'find_files_and_readers'
        # In a real automated pipeline, you would use s3fs to download the specific timestep here.
        
        # NOTE: For this snippet to run immediately, you need actual files. 
        # Automated downloading is complex to script concisely, but Satpy has tools for it.
        # Here we assume you point to a directory of downloaded files:
        files = glob.glob(f'data/*{scn_time.strftime("%Y%j%H%M")}*.nc') 
        
        if not files:
            print(f"No files found for {scn_time}")
            return None

        reader = get_file_pattern(SAT_ID)
        scn = Scene(filenames=files, reader=reader)

        # Load standard Dust RGB bands (8.7, 10.8, 12.0 micron)
        # Satpy channel names vary: 'C11', 'C13', 'C15' for GOES roughly match 8.4, 10.3, 12.3
        if 'goes' in SAT_ID:
             bands = ['C11', 'C13', 'C15'] # 8.4um, 10.3um, 12.3um
        elif 'himawari' in SAT_ID:
             bands = ['B11', 'B13', 'B15'] # Similar mapping
        
        scn.load(bands)
        
        # Resample to common grid (performant)
        # 'native' is fastest, but '2km' might be needed if bands differ
        scn = scn.resample(resampler='native') 

        # ==========================================
        # 2. PHYSICAL DUST DETECTION ALGORITHM
        # ==========================================
        # Extract DataArrays
        if 'goes' in SAT_ID:
            # Band mapping for GOES ABI
            # C11 ~ 8.4um, C13 ~ 10.3um, C15 ~ 12.3um
            b08 = scn['C11']
            b10 = scn['C13']
            b12 = scn['C15']
        
        # Calculate Split Window Differences (The core physics)
        # Dust typically has NEGATIVE (12-10) and HIGH (10-8)
        diff_12_10 = b12 - b10
        diff_10_8 = b10 - b08
        
        # Create Binary Mask (1 = Dust, 0 = No Dust)
        # Optimized with Dask (lazy evaluation)
        dust_mask = (
            (diff_12_10 < THRESHOLDS['diff_12_10']) & 
            (diff_10_8 > THRESHOLDS['diff_10_8']) & 
            (b10 > THRESHOLDS['temp_10'])
        )

        # ==========================================
        # 3. CLUSTERING (Object Identification)
        # ==========================================
        # We must compute() here to get numpy array for DBSCAN
        # limiting to valid data to save memory
        valid_pixels = dust_mask.where(dust_mask, drop=True)
        
        if valid_pixels.size == 0:
            return []

        # Get coordinates of dust pixels
        # stack lat/lon into a list of (lat, lon) points
        coords = np.column_stack((valid_pixels.y.values, valid_pixels.x.values))
        
        # Run DBSCAN
        # eps=0.05 degrees (~5km), min_samples=10 pixels (~40km2 area)
        # This groups nearby pixels into "Events"
        db = DBSCAN(eps=0.05, min_samples=10, metric='euclidean').fit(coords)
        
        labels = db.labels_
        unique_labels = set(labels)
        
        events = []
        for k in unique_labels:
            if k == -1: continue # Noise
            
            # Extract points for this cluster
            class_member_mask = (labels == k)
            xy = coords[class_member_mask]
            
            # Centroid
            lat_mean = np.mean(xy[:, 0])
            lon_mean = np.mean(xy[:, 1])
            area_px = len(xy)
            
            # Get actual Lat/Lon from projection y/x if needed (Satpy handles this)
            # For simplicity, assuming x/y are already lat/lon or easily converted
            # If projection coordinates, use scn[band].attrs['area'].get_lonlat(row, col)
            
            events.append({
                'datetime': scn_time,
                'latitude': float(lat_mean),
                'longitude': float(lon_mean),
                'area_pixels': int(area_px),
                'satellite': SAT_ID
            })
            
        return events

    except Exception as e:
        print(f"Error processing {scn_time}: {e}")
        return None

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
all_events = []
current_time = START_TIME

print(f"Starting analysis for {SAT_ID}...")

while current_time <= END_TIME:
    print(f"Processing {current_time}...")
    events = process_scene(current_time)
    
    if events:
        all_events.extend(events)
        print(f"  Found {len(events)} dust plumes.")
    
    # GOES provides data every 10-15 mins
    current_time += datetime.timedelta(minutes=15)

# Save Results
if all_events:
    df = pd.DataFrame(all_events)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Saved {len(df)} events to {OUTPUT_CSV}")
else:
    print("No dust events detected in this period.")
