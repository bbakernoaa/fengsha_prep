from typing import Literal

import s3fs
import earthaccess
import xarray as xr
import numpy as np
import os

from datetime import datetime, timedelta
import re
import logging

# Set up logging
logger = logging.getLogger(__name__)

Sensor = Literal["MODIS", "VIIRS", "NESDIS"]

PRODUCT_MAP = {
    "MODIS": {
        "brdf": "MCD43C1",
        "albedo": "MCD43C3",
        "nbar": "MCD43C4",
        "lai": "MCD15A2H",
    },
    "VIIRS": {
        "brdf": "VJ143C1",
        "albedo": "VJ143C3",
        "nbar": "VJ143C4",
        "lai": "VNP15A2H",
    },
}

NESDIS_BUCKET = "noaa-nesdis-n20-pds"
NESDIS_PRODUCTS = {
    "gvf": "GVF_GLB",
    "lai": "WKL-LAI-GLB",
}


def get_nesdis_data(
    product_type: str, start_date: str, end_date: str, optional: bool = False
) -> xr.Dataset | None:
    """Retrieves NESDIS 4km grid data from AWS S3.

    Parameters
    ----------
    product_type : str
        The type of product to retrieve ('gvf', 'lai', 'albedo').
    start_date : str
        The start date for the data search in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data search in 'YYYY-MM-DD' format.
    optional : bool, optional
        If True, return None if no data is found instead of raising an error.
        Defaults to False.

    Returns
    -------
    xr.Dataset | None
        An xarray Dataset containing the data, or None if optional=True and
        no data was found.
    """
    s3 = s3fs.S3FileSystem(anon=True)
    product = NESDIS_PRODUCTS.get(product_type)
    if not product:
        if optional: return None
        raise ValueError(f"Product type {product_type} not supported for NESDIS.")

    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    logger.info(f"Searching NESDIS {product} data for {start_date}...")

    # Simple search strategy: search in the year/month of start_date
    # NESDIS JPSS data is often weekly, so we look in a window
    search_path = f"{NESDIS_BUCKET}/{product}/{dt_start.year}/{dt_start.month:02d}/"
    files = s3.glob(f"{search_path}*/*.nc")

    if not files:
        logger.info(f"No NESDIS files found in {search_path}, trying previous month...")
        # Fallback to previous month
        prev_month = dt_start - timedelta(days=28)
        search_path = f"{NESDIS_BUCKET}/{product}/{prev_month.year}/{prev_month.month:02d}/"
        files = s3.glob(f"{search_path}*/*.nc")

    if not files:
        if optional:
            logger.warning(f"No NESDIS data found for {product} around {start_date}. Returning None.")
            return None
        raise FileNotFoundError(f"No NESDIS data found for {product} around {start_date} in s3://{search_path}")

    # For now, we take the first matching file in the month.
    # A more sophisticated version would parse sYYYYMMDD and eYYYYMMDD from filenames.
    # To keep it simple and match the 4km readiness requirement:
    remote_path = files[0]
    local_filename = os.path.basename(remote_path)
    cache_dir = os.path.join(os.getcwd(), "data", "nesdis")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, local_filename)

    if not os.path.exists(local_path):
        logger.info(f"Downloading NESDIS file: s3://{remote_path} to {local_path}")
        s3.get(remote_path, local_path)
    else:
        logger.info(f"Using existing NESDIS file: {local_path}")

    # Use dask chunking for large NESDIS files to improve performance.
    # We open with chunks='auto' first to handle varying dimension names (lat/latitude).
    ds = xr.open_dataset(
        local_path,
        engine="h5netcdf",
        chunks="auto"
    )

    # Normalize coordinates to 'lat' and 'lon' and interp to standard CMG grid
    # Target: 0.05 deg global (3600x7200)
    logger.debug(f"Normalizing NESDIS coordinates and interpolating to CMG grid...")
    target_lat = np.linspace(89.975, -89.975, 3600)
    target_lon = np.linspace(-179.975, 179.975, 7200)

    rename_dict = {}
    if "latitude" in ds.dims: rename_dict["latitude"] = "lat"
    elif "y" in ds.dims: rename_dict["y"] = "lat"

    if "longitude" in ds.dims: rename_dict["longitude"] = "lon"
    elif "x" in ds.dims: rename_dict["x"] = "lon"

    if rename_dict:
        # Drop conflicting coords
        for new_name in rename_dict.values():
            if new_name in ds.coords and new_name not in ds.dims:
                ds = ds.drop_vars(new_name)
        ds = ds.rename(rename_dict)

    # Re-chunk to consistent block sizes after renaming
    ds = ds.chunk({"lat": 1800, "lon": 3600})

    # If coordinates are missing or different size, assign them before interp
    # Using accurate pixel centers for global rectilinear grids
    if "lat" in ds.dims and ( "lat" not in ds.coords or ds.lat.size != ds.dims["lat"] ):
        res_lat = 180.0 / ds.dims["lat"]
        logger.debug(f"Assigning lat coords with resolution {res_lat}")
        ds = ds.assign_coords(lat=np.linspace(90.0 - res_lat/2, -90.0 + res_lat/2, ds.dims["lat"]))
    if "lon" in ds.dims and ( "lon" not in ds.coords or ds.lon.size != ds.dims["lon"] ):
        res_lon = 360.0 / ds.dims["lon"]
        logger.debug(f"Assigning lon coords with resolution {res_lon}")
        ds = ds.assign_coords(lon=np.linspace(-180.0 + res_lon/2, 180.0 - res_lon/2, ds.dims["lon"]))

    logger.info(f"Interpolating NESDIS {product} from {ds.dims['lat']}x{ds.dims['lon']} to 3600x7200...")
    ds = ds.interp(lat=target_lat, lon=target_lon, method="nearest")

    # Add time dimension to align with NASA CMG datasets
    t_start = datetime.strptime(start_date, "%Y-%m-%d")
    ds = ds.expand_dims(time=[t_start])

    return ds


def get_cmg_data(
    product_type: str, start_date: str, end_date: str, sensor: Sensor = "MODIS", optional: bool = False
) -> xr.Dataset | None:
    """Retrieves MODIS, VIIRS, or NESDIS CMG data.

    Parameters
    ----------
    product_type : str
        The type of product to retrieve (e.g., "albedo", "lai", "gvf").
    start_date : str
        The start date for the data search in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data search in 'YYYY-MM-DD' format.
    sensor : Sensor, optional
        The sensor/source to use ('MODIS', 'VIIRS', or 'NESDIS'). Defaults to 'MODIS'.
    optional : bool, optional
        If True, return None if no data is found instead of raising an error.
        Defaults to False.

    Returns
    -------
    xr.Dataset | None
        An xarray Dataset containing the downloaded and consolidated data, or
        None if optional=True and no data was found.
    """
    if sensor == "NESDIS":
        if product_type in ["brdf", "nbar", "albedo"]:
            # NESDIS doesn't have required BRDF/NBAR parameters, use NASA VIIRS
            sensor = "VIIRS"
        else:
            return get_nesdis_data(product_type, start_date, end_date, optional=optional)

    short_name = PRODUCT_MAP[sensor][product_type]
    logger.info(f"Searching Earthdata for {short_name} ({sensor}) from {start_date} to {end_date}...")
    results = earthaccess.search_data(
        short_name=short_name, cloud_hosted=True, temporal=(start_date, end_date)
    )

    # Filter results to ensure we don't grab redundant days due to metadata overlap.
    # We strictly match the daily granules to the requested window.
    logger.debug(f"Found {len(results)} granules, filtering for exact date range...")
    filtered_results = []
    dt_s = datetime.strptime(start_date, "%Y-%m-%d")
    dt_e = datetime.strptime(end_date, "%Y-%m-%d")
    for r in results:
        g_date = _parse_vnp_doy_from_name(r.data_links()[0])
        if g_date and dt_s <= g_date <= dt_e:
            filtered_results.append(r)

    if filtered_results:
        results = filtered_results
    logger.info(f"Using {len(results)} granules after filtering.")

    if len(results) == 0:
        if optional:
            logger.warning(f"No data found for {short_name} between {start_date} and {end_date}. Returning None.")
            return None
        raise ValueError(f"No data found for {short_name} between {start_date} and {end_date}.")

    # Open the multi-file dataset using xarray
    r1 = results[0]
    extension = os.path.splitext(r1.data_links()[0])[1]

    # Download files instead of streaming to improve stability and performance
    nasa_cache_dir = os.path.join(os.getcwd(), "data", "nasa")
    os.makedirs(nasa_cache_dir, exist_ok=True)

    logger.info(f"Downloading {len(results)} NASA granules to {nasa_cache_dir}...")
    local_files = earthaccess.download(results, nasa_cache_dir)

    logger.info(f"Opening {len(local_files)} local files (type: {extension})...")
    if extension in [".hdf", ".h4"]:
        # For HDF4 files, use the appropriate engine
        logger.debug("Using rasterio engine for HDF4/HDF files.")
        ds = xr.open_mfdataset(
            local_files,
            combine="by_coords",
            preprocess=lambda ds: ds.sortby("time"),
            engine="rasterio",
            chunks={"lat": 1800, "lon": 3600},
        )
        return ds

    elif sensor == "VIIRS":
        if "VNP43" in short_name or "VJ143" in short_name:
            # Primary group for NASA VIIRS CMG products is often BRDF regardless of sub-product.
            # We try common candidates and re-open fileobjects for each attempt.
            groups = ["/HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields"]
            if "C4" in short_name:
                groups.insert(0, "/HDFEOS/GRIDS/VIIRS_CMG_NBAR/Data Fields")
            elif "C3" in short_name:
                groups.insert(0, "/HDFEOS/GRIDS/VIIRS_CMG_Albedo/Data Fields")

            groups = list(dict.fromkeys(groups))

            last_err = None
            for group in groups:
                try:
                    logger.info(f"Attempting to open VIIRS CMG with group: {group}")
                    ds = xr.open_mfdataset(
                        local_files,
                        combine="by_coords",
                        preprocess=_preprocess_vnp43,
                        group=group,
                        parallel=True,
                        chunks={"lat": 1800, "lon": 3600},
                    )
                    logger.info(f"Successfully opened with group: {group}")
                    return ds
                except (OSError, KeyError, ValueError) as e:
                    logger.debug(f"Failed to open with group {group}: {e}")
                    last_err = e
                    continue

            if last_err:
                logger.error(f"Failed to open VIIRS CMG after trying all groups: {groups}")
                raise last_err

    logger.debug("Opening files with default concat engine.")
    ds = xr.open_mfdataset(
        local_files,
        combine="by_coords",
        preprocess=lambda ds: ds.sortby("time"),
        parallel=True,
        chunks={"lat": 1800, "lon": 3600},
    )
    return ds

def _parse_vnp_doy_from_name(name: str) -> datetime | None:
    # Extract DOY from filename (e.g., .A2021365.)
    m = re.search(r"\.A(\d{7})\.", name)
    if not m:
        return None
    y = int(m.group(1)[:4])
    j = int(m.group(1)[4:])
    dt = datetime.strptime(f"{y}{j:03d}", "%Y%j")
    return dt

def _preprocess_vnp43(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess VNP43 dataset by renaming variables and adjusting coordinates."""
    source_file = ds.encoding.get("source", "unknown")
    logger.debug(f"Preprocessing {source_file}...")

    # 0. Define Target Grid (0.05 deg global)
    target_lat = np.linspace(89.975, -89.975, 3600)
    target_lon = np.linspace(-179.975, 179.975, 7200)

    # 1. Standardize dimensions to 'lat' and 'lon'
    rename_map = {}
    for d, s in ds.dims.items():
        if d in ["phony_dim_0", "y"] or (s == 3600 and d != "lat"):
            rename_map[d] = "lat"
        if d in ["phony_dim_1", "x"] or (s == 7200 and d != "lon"):
            rename_map[d] = "lon"

    if rename_map:
        logger.debug(f"Renaming dimensions: {rename_map}")
        # Avoid naming conflicts with existing coords
        for v in ["lat", "lon"]:
            if v in ds.coords and v not in ds.dims:
                ds = ds.drop_vars(v)
        ds = ds.rename(rename_map)

    # 2. Reconstruct/Ensure coordinates are correctly indexed
    # Even if they have the right size, we must assign values so interp/merge works cleanly.
    if "lat" in ds.dims and "lon" in ds.dims:
        curr_nlat = ds.dims["lat"]
        curr_nlon = ds.dims["lon"]

        # Assign best-guess coordinates based on current size if they don't look like degrees
        if "lat" not in ds.coords or ds.coords["lat"].max() < 1: # Guessing indices
             logger.debug(f"Assigning lat coords for {curr_nlat} points")
             ds = ds.assign_coords(lat=np.linspace(89.975, -89.975, curr_nlat))
        if "lon" not in ds.coords or ds.coords["lon"].min() > -1:
             logger.debug(f"Assigning lon coords for {curr_nlon} points")
             ds = ds.assign_coords(lon=np.linspace(-179.975, 179.975, curr_nlon))

        # Force everything to the target 3600x7200 grid
        if curr_nlat != 3600 or curr_nlon != 7200:
            logger.info(f"Interpolating {source_file} from ({curr_nlat}, {curr_nlon}) to (3600, 7200)...")
            ds = ds.interp(lat=target_lat, lon=target_lon, method="nearest")
        else:
            # Already right size, just ensure standard values to prevent floating point alignment issues
            ds = ds.assign_coords(lat=target_lat, lon=target_lon)

    # 3. Parse and set time coordinate with index
    time_val = _parse_vnp_doy_from_name(source_file)
    if time_val:
        ds = ds.expand_dims("time")
        ds = ds.assign_coords(time=[time_val])
        logger.debug(f"Set time coordinate to {time_val}")

    return ds
