# Data Downloaders

This guide provides detailed information on how to use the data downloaders available in `fengsha_prep`.

## BNU Downloader

The BNU downloader retrieves soil data from the BNU dataset. The available data types are `sand`, `silt`, and `clay`.

### Configuration

Before using the downloader, you need to configure the download URLs in the `src/fengsha_prep/data_downloaders/config.toml` file.

```toml
[bnu_data]
sand_urls = ["http://example.com/sand.nc"]
silt_urls = ["http://example.com/silt.nc"]
clay_urls = ["http://example.com/clay.nc"]
```

### Usage

Here's how to use the `get_bnu_data` function:

```python
from fengsha_prep.data_downloaders import bnu

bnu.get_bnu_data(data_type="sand", output_dir="bnu_data")
```

## SoilGrids Downloader

The SoilGrids downloader fetches soil data from the SoilGrids service.

### Usage

The `get_soilgrids_data` function allows you to specify the service, coverage, and geographic bounding box.

```python
from fengsha_prep.data_downloaders import soilgrids

soilgrids.get_soilgrids_data(
    service_id="sand",
    coverage_id="sand_0-5cm_mean",
    west=-10,
    south=-10,
    east=10,
    north=10,
    output_path="soilgrids_sand.nc",
)
```
