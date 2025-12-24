import earthaccess
import xarray as xr
import numpy as np

# 1. AUTHENTICATION
# This will prompt for your Earthdata credentials if not found in environment variables
auth = earthaccess.login(strategy="interactive")

def get_modis_data(product, start_date, end_date):
    """Retrieves MODIS CMG data from NASA DAAC using earthaccess."""
    results = earthaccess.search_data(
        short_name=product,
        cloud_hosted=True,
        temporal=(start_date, end_date)
    )
    # Open the multi-file dataset using xarray
    ds = earthaccess.open_xr(results)
    return ds

def process_hybrid_drag(start_date, end_date, u10_wind):
    """
    Automated pipeline to fetch data and calculate the Drag Partition.
    
    References:
    - Chappell & Webb (2016): DOI 10.1016/j.aeolia.2015.11.001
    - Leung et al. (2023): DOI 10.5194/acp-23-11235-2023
    """
    
    print("Fetching MODIS Albedo (MCD43C3)...")
    ds_alb = get_modis_data("MCD43C3", start_date, end_date)
    
    print("Fetching MODIS LAI (MCD15A2H)...")
    ds_lai = get_modis_data("MCD15A2H", start_date, end_date)

    # Align LAI to Albedo grid (0.05 degree CMG)
    ds_lai = ds_lai.interp_like(ds_alb, method='nearest')

    # --- DRAG PARTITION CALCULATION ---
    
    # A. Bare Surface (Chappell & Webb)
    # Using Band 1 (Red) for structural shadowing
    omega_n = 1.0 - (ds_alb['Albedo_BSW_Band1'] / ds_alb['BRDF_Albedo_Parameter_Isotropic_Band1'])
    # Rescale omega following Hennen et al. (2023) DOI 10.1016/j.aeolia.2022.100852
    omega_ns = ((0.0001 - 0.1) * (omega_n - 35.0) / (0.0 - 35.0)) + 0.1
    ra_bare = 0.0311 * np.exp(-omega_ns / 1.131) + 0.007

    # B. Green Vegetation (Leung et al., 2023) DOI 10.5194/acp-23-11235-2023
    lai = ds_lai['Lai']
    sigma_g = 1.0 - np.exp(-0.5 * lai)
    lambda_g = lai / 2.0 # Effective lateral cover

    # C. Brown Vegetation (Guerschman et al., 2009) DOI 10.1016/j.rse.2009.01.006
    # NDTI using SWIR Band 6 and 7
    ndti = (ds_alb['Albedo_BSW_Band6'] - ds_alb['Albedo_BSW_Band7']) / \
           (ds_alb['Albedo_BSW_Band6'] + ds_alb['Albedo_BSW_Band7'] + 1e-6)
    sigma_b = ((ndti - 0.05) / (0.25 - 0.05)).clip(0, 1)
    lambda_b = sigma_b * 0.1

    # D. Final Integration
    f_veg = (1.0 - (sigma_g + sigma_b).clip(0, 1)) * np.exp(-(lambda_g + lambda_b) / 0.1)
    
    # Calculate Surface Friction Velocity (us*)
    us_star = u10_wind * (ra_bare * f_veg)
    
    return us_star

# --- EXECUTION ---
# Example: Process for a specific week in 2024
# u10_wind could be a constant or an xarray matching the MODIS grid
# result = process_hybrid_drag("2024-03-01", "2024-03-07", u10_wind=7.5)
