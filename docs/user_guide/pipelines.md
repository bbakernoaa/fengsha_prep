# Pipelines

This guide explains the data processing pipelines available in `fengsha_prep`.

## Drag Partition Pipeline

The drag partition pipeline implements a hybrid model to estimate the surface friction velocity (`us*`). This is a key parameter in determining how much wind energy is transferred to the ground, which is critical for predicting dust emissions.

The pipeline works by partitioning the drag forces between:
- **Bare Soil:** Using the model from Chappell & Webb (2016).
- **Green Vegetation:** Based on the work of Leung et al. (2023).
- **Brown (Non-Photosynthetic) Vegetation:** Modeled according to Guerschman et al. (2009).

### Usage

The main function, `process_hybrid_drag`, automates the entire workflow. It fetches the required MODIS Albedo (MCD43C3) and Leaf Area Index (MCD15A2H) data from NASA's Earthdata cloud, and then calculates the surface friction velocity.

```python
from fengsha_prep.pipelines.drag_partition import core

# Define the time period for the analysis
start_date = "2024-03-01"
end_date = "2024-03-07"

# Provide the 10-meter wind speed (can be a constant or an xarray DataArray)
wind_speed = 7.5  # in m/s

# Run the pipeline
surface_friction_velocity = core.process_hybrid_drag(
    start_date=start_date,
    end_date=end_date,
    u10_wind=wind_speed
)

print(surface_friction_velocity)
```

## Dust Scan Pipeline

The dust scan pipeline is an asynchronous tool designed to detect and cluster dust plumes from satellite imagery over a specified time period. It can process data from GOES satellites (via AWS S3) or other satellites with local data.

The pipeline performs the following steps:
1.  **Loads Satellite Data:** Asynchronously loads satellite scenes, handling both S3 and local file access.
2.  **Detects Dust:** Applies a physical dust detection algorithm based on brightness temperature differences between infrared channels.
3.  **Clusters Events:** Uses the DBSCAN clustering algorithm to group dusty pixels into distinct plumes, calculating properties like the centroid and area for each event.

### Usage

The pipeline is typically run from the command line. You need to provide the satellite ID, start and end times, and an output file path.

```bash
python -m src.fengsha_prep.pipelines.dust_scan.core \
    --sat goes16 \
    --start 2024-04-01T18:00 \
    --end 2024-04-01T22:00 \
    --output dust_events_report.csv
```

This will generate a CSV file (`dust_events_report.csv`) listing all the dust plumes detected in the specified four-hour window.

## Uthresh Pipeline (GDE-PIML)

The `uthresh` pipeline is a comprehensive, end-to-end suite for modeling global dust emissions, referred to as the **Global Dust Emission Physics-Informed Machine Learning (GDE-PIML) Suite**. It integrates data retrieval, physics-based calculations, and a machine learning model to produce global dust flux maps.

The pipeline is divided into four main stages:

1.  **Data Retrieval:** Fetches meteorological data (from UFS Replay on S3) and soil properties (from SoilGrids) for a given location.
2.  **Physics Engine:** Calculates two key physical parameters:
    - The **hybrid drag partition ratio**, which determines how wind stress is partitioned between bare soil and vegetation.
    - The **moisture inhibition factor**, which models how soil moisture suppresses dust emission.
3.  **PIML Model:** Uses a trained XGBoost model to predict the threshold friction velocity required to initiate dust emission, based on the physical parameters and soil properties.
4.  **Flux Projection:** Combines the predicted threshold velocity with meteorological data to calculate the vertical dust flux using the Marticorena-Bergametti (1995) flux equation.

Due to its complexity and focus on large-scale global modeling, the `uthresh` pipeline is not typically used for simple, small-scale examples. It is designed to be integrated into larger climate and weather modeling workflows.
