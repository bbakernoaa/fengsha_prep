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

### SoilGrids

To retrieve data from SoilGrids, use the `get_soilgrids_data` function. The data will be saved as a compressed NetCDF file.

```python
from fengsha_prep.soilgrids import get_soilgrids_data

data = get_soilgrids_data(
    service_id='sand',
    coverage_id='sand_0-5cm_mean',
    west=-10,
    south=-10,
    east=10,
    north=10,
    crs='urn:ogc:def:crs:EPSG::4326',
    output_path='sand_0-5cm_mean.nc'
)
```

### BNU

To retrieve data from the BNU soil dataset, use the `get_bnu_data` function. The `data_type` argument should correspond to an entry in the `config.toml` file (e.g., 'sand', 'silt', 'clay').

```python
from fengsha_prep.bnu import get_bnu_data

downloaded_files = get_bnu_data('sand', output_dir='bnu_sand_data')
```

### Regridding

To regrid a dataset from a MODIS sinusoidal grid to a rectilinear Gaussian grid, use the `regrid_modis_to_rectilinear` function.

```python
import xarray as xr
import numpy as np
from fengsha_prep.regrid import regrid_modis_to_rectilinear

# Create a dummy sinusoidal dataset
ds_sinu = xr.Dataset({
    'foo': (('y', 'x'), np.random.rand(10, 20)),
    'lat': (('y', 'x'), np.random.uniform(20, 50, size=(10, 20))),
    'lon': (('y', 'x'), np.random.uniform(-120, -70, size=(10, 20))),
})

# Define the output grid
lon_min, lon_max, d_lon = -110, -80, 1.0
lat_min, lat_max, d_lat = 30, 45, 1.0

# Regrid the dataset
ds_regridded = regrid_modis_to_rectilinear(
    ds_sinu, 'foo', lon_min, lon_max, d_lon, lat_min, lat_max, d_lat
)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
