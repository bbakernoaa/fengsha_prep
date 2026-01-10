import numpy as np
import xarray as xr


def calculate_drag_partition(
    ds_alb: xr.Dataset, ds_lai: xr.Dataset, u10_wind: float | xr.DataArray
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
