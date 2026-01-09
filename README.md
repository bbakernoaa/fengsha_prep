# fengsha_prep

Tools to retrieve and process inputs needed for the NOAA fengsha dust emission model.

## Installation

To install the package, clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/fengsha_prep.git
cd fengsha_prep
pip install .
```

## Configuration

The `fengsha_prep` package requires a configuration file located at `src/fengsha_prep/config.toml`. This file contains the URLs for the BNU soil dataset. A sample `config.toml` file is provided with placeholder URLs. You will need to replace these with the actual download links.

```toml
[bnu_data]
# Please replace these placeholder URLs with the actual download links.
sand_urls = [
  "http://example.com/path/to/sand/data1.nc",
  "http://example.com/path/to/sand/data2.nc",
]
silt_urls = [
  "http://example.com/path/to/silt/data1.nc",
  "http://example.com/path/to/silt/data2.nc",
]
clay_urls = [
  "http://example.com/path/to/clay/data1.nc",
  "http://example.com/path/to/clay/data2.nc",
]
```

## Usage

### Drag Partition Pipeline

The `drag_partition` pipeline implements a hybrid model to estimate the surface friction velocity (us*) by partitioning drag between bare soil, green vegetation, and non-photosynthetic (brown) vegetation. It fetches the required MODIS Albedo (MCD43C3) and LAI (MCD15A2H) data from the NASA Earthdata repository.

**Note on Authentication:** This pipeline uses the `earthaccess` library, which requires authentication with your NASA Earthdata login. For non-interactive environments, it is recommended to have your credentials stored in a `.netrc` file.

```python
import earthaccess
from fengsha_prep.pipelines.drag_partition.pipeline import run_drag_partition_pipeline

# Authenticate with Earthdata
# This will prompt for credentials if not found in a .netrc file
auth = earthaccess.login(strategy="interactive")

# Define the analysis period and wind speed
start_date = "2024-03-01"
end_date = "2024-03-07"
wind_speed = 7.5  # Constant wind speed in m/s

# Run the pipeline
result_us_star = run_drag_partition_pipeline(
    start_date=start_date,
    end_date=end_date,
    u10_wind=wind_speed,
)

print(result_us_star)

# The result can be saved to a NetCDF file
# result_us_star.to_netcdf("surface_friction_velocity.nc")
```

For a complete, runnable script, please see `examples/run_drag_partition_pipeline.py`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
