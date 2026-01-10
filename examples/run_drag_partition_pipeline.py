import logging

import earthaccess

from fengsha_prep.pipelines.drag_partition.pipeline import run_drag_partition_pipeline

# --- LOGGING CONFIGURATION ---
# Configure basic logging to see the output when the script is run directly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- AUTHENTICATION ---
# This will prompt for your Earthdata credentials if not found in
# environment variables. For non-interactive environments, it's recommended to
# have credentials in a .netrc file.
auth = earthaccess.login(strategy="interactive")

# --- EXECUTION EXAMPLE ---
# This example demonstrates how to run the pipeline for a specific week in 2024.
# The `u10_wind` can be a constant value (as shown here) or another xarray
# DataArray that aligns with the MODIS grid for more complex scenarios.
start_date_example = "2024-03-01"
end_date_example = "2024-03-07"
wind_speed_example = 7.5  # Constant wind speed in m/s

logging.info(
    "Running hybrid drag partition model for %s to %s...",
    start_date_example,
    end_date_example,
)
result_us_star = run_drag_partition_pipeline(
    start_date=start_date_example,
    end_date=end_date_example,
    u10_wind=wind_speed_example,
)
logging.info("\n--- Processing Complete ---")
logging.info(result_us_star)

# Example of how to save the output to a NetCDF file
# result_us_star.to_netcdf("surface_friction_velocity.nc")
# logging.info("\nResult saved to surface_friction_velocity.nc")
