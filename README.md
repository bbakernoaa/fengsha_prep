# fengsha_prep

Tools to retrieve and process inputs needed for the NOAA fengsha dust emission model.

## Installation

To install the package, clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/fengsha_prep.git
cd fengsha_prep
pip install .
```

## Usage

The `fengsha_prep` package provides tools to retrieve soil data from SoilGrids and the BNU soil dataset.

### SoilGrids

To retrieve data from SoilGrids, use the `get_soilgrids_data` function:

```python
from src.fengsha_prep.soilgrids import get_soilgrids_data

data = get_soilgrids_data(
    service_id='sand',
    coverage_id='sand_0-5cm_mean',
    west=-10,
    south=-10,
    east=10,
    north=10,
    crs='urn:ogc:def:crs:EPSG::4326',
    output='sand_0-5cm_mean.tif'
)
```

### BNU

To retrieve data from the BNU soil dataset, use the `get_bnu_data` function:

```python
from src.fengsha_prep.bnu import get_bnu_data

data = get_bnu_data()
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
