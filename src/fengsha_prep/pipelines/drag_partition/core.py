from typing import Union

import earthaccess
import numpy as np
import xarray as xr


def get_modis_data(product: str, start_date: str, end_date: str) -> xr.Dataset:
    """Retrieves MODIS CMG data from a NASA DAAC using earthaccess.

    Parameters
    ----------
    product : str
        The short name of the MODIS product (e.g., "MCD43C3", "MCD15A2H").
    start_date : str
        The start date for the data search in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data search in 'YYYY-MM-DD' format.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the downloaded and consolidated MODIS data.
    """
    results = earthaccess.search_data(
        short_name=product, cloud_hosted=True, temporal=(start_date, end_date)
    )
    # Open the multi-file dataset using xarray
    ds = earthaccess.open_xr(results)
    return ds


def _calculate_drag_partition(
    ds_alb: xr.Dataset, ds_lai: xr.Dataset, u10_wind: Union[float, xr.DataArray]
) -> xr.DataArray:
    """Calculates the Drag Partition using a hybrid model.

    This is a pure function that encapsulates the scientific logic for
    estimating surface friction velocity (us*).

    Parameters
    ----------
    ds_alb : xr.Dataset
        The MODIS Albedo dataset (e.g., MCD43C3).
    ds_lai : xr.Dataset
        The MODIS LAI dataset (e.g., MCD15A2H).
    u10_wind : Union[float, xr.DataArray]
        The 10-meter wind speed.

    Returns
    -------
    xr.DataArray
        A DataArray representing the calculated surface friction velocity (us*).
    """
    # Align LAI to Albedo grid (0.05 degree CMG)
    ds_lai = ds_lai.interp_like(ds_alb, method="nearest")

    # --- DRAG PARTITION CALCULATION ---
    # A. Bare Surface (Chappell & Webb)
    omega_n = 1.0 - (
        ds_alb["Albedo_BSW_Band1"] / ds_alb["BRDF_Albedo_Parameter_Isotropic_Band1"]
    )
    omega_ns = ((0.0001 - 0.1) * (omega_n - 35.0) / (0.0 - 35.0)) + 0.1
    ra_bare = 0.0311 * np.exp(-omega_ns / 1.131) + 0.007

    # B. Green Vegetation (Leung et al., 2023)
    lai = ds_lai["Lai"]
    sigma_g = 1.0 - np.exp(-0.5 * lai)
    lambda_g = lai / 2.0

    # C. Brown Vegetation (Guerschman et al., 2009)
    ndti = (ds_alb["Albedo_BSW_Band6"] - ds_alb["Albedo_BSW_Band7"]) / (
        ds_alb["Albedo_BSW_Band6"] + ds_alb["Albedo_BSW_Band7"] + 1e-6
    )
    sigma_b = ((ndti - 0.05) / (0.25 - 0.05)).clip(0, 1)
    lambda_b = sigma_b * 0.1

    # D. Final Integration
    f_veg = (1.0 - (sigma_g + sigma_b).clip(0, 1)) * np.exp(
        -(lambda_g + lambda_b) / 0.1
    )

    # Calculate Surface Friction Velocity (us*)
    us_star = u10_wind * (ra_bare * f_veg)
    us_star.attrs["long_name"] = "Surface Friction Velocity"
    us_star.attrs["units"] = "m s-1"
    us_star.attrs["history"] = (
        f"Calculated at {np.datetime64('now')} using the hybrid drag partition model."
    )

    return us_star


def process_hybrid_drag(
    start_date: str, end_date: str, u10_wind: Union[float, xr.DataArray]
) -> xr.DataArray:
    """Automated pipeline to fetch data and calculate the Drag Partition.

    This function implements a hybrid model to estimate the surface friction
    velocity (us*) by partitioning drag between bare soil, green vegetation,
    and non-photosynthetic (brown) vegetation.

    Parameters
    ----------
    start_date : str
        The start date for the analysis in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the analysis in 'YYYY-MM-DD' format.
    u10_wind : Union[float, xr.DataArray]
        The 10-meter wind speed. Can be a constant (float) or a DataArray
        with dimensions matching the MODIS grid.

    Returns
    -------
    xr.DataArray
        A DataArray representing the calculated surface friction velocity (us*).

    References
    ----------
    - Chappell & Webb (2016): DOI 10.1016/j.aeolia.2015.11.001
    - Leung et al. (2023): DOI 10.5194/acp-23-11235-2023
    - Hennen et al. (2023): DOI 10.1016/j.aeolia.2022.100852
    - Guerschman et al. (2009): DOI 10.1016/j.rse.2009.01.006
    """
    print("Fetching MODIS Albedo (MCD43C3)...")
    ds_alb = get_modis_data("MCD43C3", start_date, end_date)

    print("Fetching MODIS LAI (MCD15A2H)...")
    ds_lai = get_modis_data("MCD15A2H", start_date, end_date)

    return _calculate_drag_partition(ds_alb, ds_lai, u10_wind)


if __name__ == "__main__":
    # --- AUTHENTICATION ---
    # This will prompt for your Earthdata credentials if not found in environment variables
    # For non-interactive environments, it's recommended to have credentials in a .netrc file.
    auth = earthaccess.login(strategy="interactive")
    # --- EXECUTION EXAMPLE ---
    # This example demonstrates how to run the pipeline for a specific week in 2024.
    # The `u10_wind` can be a constant value (as shown here) or another xarray
    # DataArray that aligns with the MODIS grid for more complex scenarios.
    start_date_example = "2024-03-01"
    end_date_example = "2024-03-07"
    wind_speed_example = 7.5  # Constant wind speed in m/s

    print(
        f"Running hybrid drag partition model for {start_date_example} to {end_date_example}..."
    )
    result_us_star = process_hybrid_drag(
        start_date=start_date_example,
        end_date=end_date_example,
        u10_wind=wind_speed_example,
    )
    print("\n--- Processing Complete ---")
    print(result_us_star)

    # Example of how to save the output to a NetCDF file
    # result_us_star.to_netcdf("surface_friction_velocity.nc")
    # print("\nResult saved to surface_friction_velocity.nc")
