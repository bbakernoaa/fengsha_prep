from typing import Literal

import s3fs
import earthaccess
import xarray as xr
import numpy as np
import os

from datetime import datetime, timedelta
import re
import logging
import warnings
import h5py
from pathlib import Path
import json

# Set up logging
logger = logging.getLogger(__name__)

Sensor = Literal["MODIS", "VNP", "VJ1", "NESDIS"]

PRODUCT_MAP = {
    "MODIS": {
        "brdf": "MCD43C1",
        "albedo": "MCD43C3",
        "nbar": "MCD43C4",
        "lai": "MCD15A2H",
    },
    "VNP": {
        "brdf": "VNP43C1",
        "albedo": "VNP43C3",
        "nbar": "VNP43C4",
        "lai": "VNP15A2H",
        "ndvi": "VNP13C1",
    },
    "VJ1": {
        "brdf": "VJ143C1",
        "albedo": "VJ143C3",
        "nbar": "VJ143C4",
        "lai": "VNP15A2H",
        "ndvi": "VJ113C1",
    },
}

NESDIS_BUCKET = "noaa-nesdis-n20-pds"
NESDIS_PRODUCTS = {
    "gvf": "GVF_GLB",
    "lai": "WKL-LAI-GLB",
}


def get_nesdis_data(
    product_type: str, start_date: str, end_date: str, optional: bool = False, cache_dir: str | None = None
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
    if cache_dir is None:
        nesdis_cache_dir = os.path.join(os.getcwd(), "data", "nesdis")
    else:
        nesdis_cache_dir = cache_dir
    os.makedirs(nesdis_cache_dir, exist_ok=True)
    local_path = os.path.join(nesdis_cache_dir, local_filename)

    if not os.path.exists(local_path):
        logger.info(f"Downloading NESDIS file: s3://{remote_path} to {local_path}")
        s3.get(remote_path, local_path)
    else:
        logger.info(f"Using existing NESDIS file: {local_path}")

    # Use dask chunking for large NESDIS files to improve performance.
    # We open with chunks='auto' first to handle varying dimension names (lat/latitude).
    ds = xr.open_dataset(
        local_path,
        engine="netcdf4",
        chunks="auto"
    )

    # Normalize coordinates to 'lat' and 'lon' and interp to standard CMG grid
    # Target: 0.05 deg global (3600x7200)
    logger.debug(f"Normalizing NESDIS coordinates and interpolating to CMG grid...")
    target_lat = np.linspace(89.975, -89.975, 3600)
    target_lon = np.linspace(-179.975, 179.975, 7200)

    rename_dict = {}
    if "latitude" in ds.sizes: rename_dict["latitude"] = "lat"
    elif "y" in ds.sizes: rename_dict["y"] = "lat"

    if "longitude" in ds.sizes: rename_dict["longitude"] = "lon"
    elif "x" in ds.sizes: rename_dict["x"] = "lon"

    if rename_dict:
        # Drop conflicting coords
        for new_name in rename_dict.values():
            if new_name in ds.coords and new_name not in ds.sizes:
                ds = ds.drop_vars(new_name)
        ds = ds.rename(rename_dict)

    # Re-chunk to consistent block sizes after renaming
    ds = ds.chunk({"lat": 1800, "lon": 3600})

    # If coordinates are missing or different size, assign them before interp
    # Using accurate pixel centers for global rectilinear grids
    if "lat" in ds.sizes and ( "lat" not in ds.coords or ds.lat.size != ds.sizes["lat"] ):
        res_lat = 180.0 / ds.sizes["lat"]
        logger.debug(f"Assigning lat coords with resolution {res_lat}")
        ds = ds.assign_coords(lat=np.linspace(90.0 - res_lat/2, -90.0 + res_lat/2, ds.sizes["lat"]))
    if "lon" in ds.sizes and ( "lon" not in ds.coords or ds.lon.size != ds.sizes["lon"] ):
        res_lon = 360.0 / ds.sizes["lon"]
        logger.debug(f"Assigning lon coords with resolution {res_lon}")
        ds = ds.assign_coords(lon=np.linspace(-180.0 + res_lon/2, 180.0 - res_lon/2, ds.sizes["lon"]))

    logger.info(f"Interpolating NESDIS {product} from {ds.sizes['lat']}x{ds.sizes['lon']} to 3600x7200...")
    ds = ds.interp(lat=target_lat, lon=target_lon, method="nearest")

    # Add time dimension to align with NASA CMG datasets
    t_start = datetime.strptime(start_date, "%Y-%m-%d")
    ds = ds.expand_dims(time=[t_start])

    return ds


_EARTHDATA_LOGGED_IN = False

def _ensure_earthdata_login():
    """Ensures that we are logged into Earthdata.

    This is especially important for Dask workers which may not share
    the session state of the local process. Uses a global flag to
    minimize redundant calls within the same process.
    """
    global _EARTHDATA_LOGGED_IN
    if _EARTHDATA_LOGGED_IN:
        return True

    try:
        # Try to login non-interactively first (for workers)
        auth = earthaccess.login(persist=True)
        if auth:
            _EARTHDATA_LOGGED_IN = True
            return True
    except Exception as e:
        logger.debug(f"Non-interactive login attempt failed: {e}")

    # Final fallback attempt
    auth = earthaccess.login()
    if auth:
        _EARTHDATA_LOGGED_IN = True
        return True

    logger.warning("Earthdata login failed. Downloads may fail if not already authenticated via environment or netrc.")
    return False

def load_prigent_drag_partition(file_path: str | Path = "PRIGENT_ET_AL_DRAGPARTITION.nc") -> xr.Dataset:
    """Loads the Prigent et al. Drag Partition dataset.

    Parameters
    ----------
    file_path : str | Path, optional
        Path to the NetCDF file. Defaults to "PRIGENT_ET_AL_DRAGPARTITION.nc".

    Returns
    -------
    xr.Dataset
        The loaded dataset.
    """
    logger.info(f"Loading Prigent drag partition from {file_path}...")
    if not Path(file_path).exists():
        logger.error(f"Prigent drag partition file not found: {file_path}")
        raise FileNotFoundError(f"Prigent drag partition file not found: {file_path}")

    ds = xr.open_dataset(file_path)

    # Rename the static partition field to a canonical name. The published file
    # stores it as "PRIGENT_RDRAG", but earlier variants used "PRIGENT_DRAG" and
    # "drag_partition", so match on any known token instead of an exact name.
    if "drag_partition" not in ds.data_vars:
        canonical = next(
            (v for v in ds.data_vars if any(t in str(v).lower() for t in ("drag", "prigent"))),
            None,
        )
        if canonical is not None:
            logger.debug(f"Renaming Prigent variable {canonical!r} to 'drag_partition'.")
            ds = ds.rename({canonical: "drag_partition"})
        else:
            logger.warning(
                f"Could not identify the drag partition variable in {file_path}. "
                f"Available: {list(ds.data_vars)}"
            )

    # Ensure latitude is descending (North to South) to match NASA CMG standards
    if "lat" in ds.coords and ds.lat.size > 0:
        if ds.lat.values[0] < ds.lat.values[-1]:
            logger.debug("Flipping Prigent latitude to descending orientation.")
            ds = ds.sortby("lat", ascending=False)

    return ds


def get_cmg_data(
    product_type: str, start_date: str, end_date: str, sensor: Sensor = "MODIS", optional: bool = False, cache_dir: str | None = None
) -> xr.Dataset | None:
    """Retrieves MODIS, VNP, VJ1, or NESDIS CMG data.

    Parameters
    ----------
    product_type : str
        The type of product to retrieve (e.g., "albedo", "lai", "gvf", "ndvi").
    start_date : str
        The start date for the data search in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data search in 'YYYY-MM-DD' format.
    sensor : Sensor, optional
        The sensor/source to use ('MODIS', 'VNP', 'VJ1', or 'NESDIS'). Defaults to 'MODIS'.
    optional : bool, optional
        If True, return None if no data is found instead of raising an error.
        Defaults to False.
    cache_dir : str, optional
        Directory to store downloaded files. Defaults to 'data/nasa' or 'data/nesdis'.

    Returns
    -------
    xr.Dataset | None
        An xarray Dataset containing the downloaded and consolidated data, or
        None if optional=True and no data was found.
    """
    if sensor == "NESDIS":
        if product_type in ["brdf", "nbar", "albedo"]:
            # NESDIS doesn't have required BRDF/NBAR parameters, use NASA VIIRS (VJ1)
            sensor = "VJ1"
        else:
            return get_nesdis_data(product_type, start_date, end_date, optional=optional, cache_dir=cache_dir)

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
    if cache_dir is None:
        nasa_cache_dir = os.path.join(os.getcwd(), "data", "nasa")
    else:
        nasa_cache_dir = cache_dir
    os.makedirs(nasa_cache_dir, exist_ok=True)

    # Ensure we are logged in before downloading (crucial for Dask workers)
    _ensure_earthdata_login()

    logger.info(f"Downloading {len(results)} NASA granules to {nasa_cache_dir} (sequential)...")
    # Set threads=1 to disable parallel downloading, which can cause auth/instability issues.
    local_files = earthaccess.download(results, nasa_cache_dir, threads=1)

    # Filter out empty or non-existent files
    local_files = [f for f in local_files if f and os.path.exists(f) and os.path.getsize(f) > 0]

    if not local_files:
        if optional:
            logger.warning(f"Download failed for {short_name} or no files returned. Returning None.")
            return None
        raise FileNotFoundError(f"Failed to download any valid files for {short_name} to {nasa_cache_dir}.")

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

    elif sensor in ["VNP", "VJ1"]:
        if any(x in short_name for x in ["VNP43", "VJ143", "VNP13", "VJ11", "VNP15"]):
            # For NASA VIIRS CMG products, the group name can vary.
            # We probe the first file to find the correct HDFEOS group.
            def find_group(filename):
                try:
                    with h5py.File(filename, "r") as f:
                        if "HDFEOS" in f and "GRIDS" in f["HDFEOS"]:
                            grids = f["HDFEOS/GRIDS"]
                            for grid_name in grids.keys():
                                group_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
                                if group_path in f:
                                    return group_path
                except Exception as e:
                    logger.warning(f"Failed to probe HDFEOS group in {filename}: {e}")
                return None

            group = find_group(local_files[0])
            if not group:
                logger.warning(f"Could not automatically find HDFEOS group in {local_files[0]}. Falling back to defaults.")
                group = "/HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields"
                if "C4" in short_name: group = "/HDFEOS/GRIDS/VIIRS_CMG_NBAR/Data Fields"
                elif "C3" in short_name: group = "/HDFEOS/GRIDS/VIIRS_CMG_Albedo/Data Fields"
                elif "13C1" in short_name: group = "/HDFEOS/GRIDS/VIIRS_CMG_VegIndices/Data Fields"

            logger.info(f"Opening VIIRS CMG with group: {group} (engine: h5netcdf)")

            # Use h5netcdf engine with phony_dims='sort' for better stability and coordinate alignment.
            # We disable parallel=True here to avoid malloc/segfault issues on certain systems
            # with HDF5/netCDF4 C library interactions.
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*multiple fill values.*")
                    ds = xr.open_mfdataset(
                        local_files,
                        combine="by_coords",
                        preprocess=_preprocess_vnp43,
                        group=group,
                        parallel=False,
                        engine="h5netcdf",
                        backend_kwargs={"phony_dims": "sort"},
                        chunks={"lat": 1800, "lon": 3600},
                    )

                # DIAGNOSTIC: Log all available variables before filtering
                logger.info(f"Available variables in {product_type} ({group}): {list(ds.data_vars)}")

                # Filter to only necessary variables based on product type
                keep_vars = []
                if product_type == "ndvi":
                    keep_vars = ["NDVI", "pixel_reliability"]
                elif product_type == "brdf":
                    # NASA VIIRS BRDF parameters often use "Parameter" in the name
                    keep_vars = [
                        "Isotropic",
                        "Geometric",
                        "Volumetric",
                        "Parameter1",
                        "Parameter2",
                        "Parameter3",
                        "Percent_Snow",
                        "LandWater",
                        "Land_Water",
                        "Land Water",
                    ]
                elif product_type == "nbar":
                    keep_vars = ["Nadir", "NBAR"]
                elif product_type == "albedo":
                    keep_vars = ["Albedo", "BSA", "WSA", "Percent_Snow"]

                if keep_vars:
                    # Match variables exactly if possible, or by substring
                    actual_vars = []
                    for v in ds.data_vars:
                        v_low = v.lower()
                        # Case-insensitive substring match for any of the keep_vars
                        if any(k.lower() in v_low for k in keep_vars):
                            # For VIIRS BRDF/NBAR/Albedo, strictly enforce M7 band if multiple bands are present
                            if product_type in ["brdf", "nbar", "albedo"] and sensor in ["VNP", "VJ1"]:
                                if "m" in v_low and "m7" not in v_low:
                                    continue
                            actual_vars.append(v)

                    if actual_vars:
                        ds = ds[actual_vars]
                        logger.debug(f"Reduced {product_type} dataset to variables: {list(ds.data_vars)}")
                    else:
                        logger.warning(f"No variables matched keep_vars {keep_vars} (filtering for M7 if VIIRS). Keeping ALL variables.")

                return ds
            except Exception as e:
                logger.error(f"Failed to open VIIRS CMG with h5netcdf: {e}. Retrying with default engine...")
                # Last resort fallback with minimal features
                return xr.open_mfdataset(
                    local_files,
                    combine="by_coords",
                    preprocess=_preprocess_vnp43,
                    group=group,
                    parallel=False,
                    chunks={"lat": 1800, "lon": 3600},
                )

    logger.debug("Opening files with default concat engine.")
    ds = xr.open_mfdataset(
        local_files,
        combine="by_coords",
        preprocess=lambda ds: ds.sortby("time"),
        parallel=False,
        chunks={"lat": 1800, "lon": 3600},
    )
    return ds


def build_earthaccess_virtual_refs(
    product_type: str,
    start_date: str,
    end_date: str,
    sensor: Sensor = "VJ1",
    output_refs: str | Path = "data/refs/drag_partition.parquet",
    cache_dir: str | None = None,
    combine: str = "nested",
) -> Path:
    """Build a Kerchunk reference dataset using Earthaccess + VirtualiZarr.

    This utility is intended for large date ranges where repeatedly opening
    many native HDF/netCDF granules is expensive. It downloads granules once,
    creates virtual references, and writes them to a Kerchunk JSON/Parquet
    reference file for fast, lazy re-opening.
    """
    if sensor == "NESDIS":
        raise ValueError("Virtual references are currently supported for NASA products only (MODIS/VNP/VJ1).")

    if product_type not in PRODUCT_MAP[sensor]:
        raise ValueError(f"Unsupported product_type '{product_type}' for sensor '{sensor}'.")

    short_name = PRODUCT_MAP[sensor][product_type]
    logger.info(
        "Building virtual refs for %s (%s) from %s to %s...",
        short_name,
        sensor,
        start_date,
        end_date,
    )

    _ensure_earthdata_login()
    results = earthaccess.search_data(
        short_name=short_name,
        cloud_hosted=True,
        temporal=(start_date, end_date),
    )

    dt_s = datetime.strptime(start_date, "%Y-%m-%d")
    dt_e = datetime.strptime(end_date, "%Y-%m-%d")
    filtered_results = []
    for r in results:
        links = r.data_links()
        g_date = _parse_vnp_doy_from_name(links[0]) if links else None
        if g_date is None or dt_s <= g_date <= dt_e:
            filtered_results.append(r)
    results = filtered_results

    if not results:
        raise ValueError(f"No data found for {short_name} between {start_date} and {end_date}.")

    local_cache = cache_dir or os.path.join(os.getcwd(), "data", "nasa")
    os.makedirs(local_cache, exist_ok=True)
    local_files = earthaccess.download(results, local_cache, threads=1)
    local_files = [f for f in local_files if f and os.path.exists(f) and os.path.getsize(f) > 0]

    if not local_files:
        raise FileNotFoundError(f"Failed to download any valid files for {short_name}.")

    try:
        from virtualizarr import open_virtual_dataset
        from virtualizarr.parsers import HDFParser
        from virtualizarr.manifests import ManifestArray
        from obstore.store import from_url
        from obspec_utils.registry import ObjectStoreRegistry
    except ImportError as e:
        raise ImportError(
            "VirtualiZarr dependencies are missing. Install extras with: "
            "pip install -e '.[virtual]'"
        ) from e

    # VirtualiZarr expects URLs plus an object-store registry.
    common_root = Path(os.path.commonpath([str(Path(f).resolve()) for f in local_files]))
    root_url = common_root.as_uri()
    store = from_url(root_url)
    registry = ObjectStoreRegistry({root_url: store})
    local_urls = [Path(f).resolve().as_uri() for f in local_files]

    parser = HDFParser()
    with warnings.catch_warnings():
        # Known warning from zarr-v3 + numcodecs codec metadata during reference writing.
        warnings.filterwarnings(
            "ignore",
            message="Numcodecs codecs are not in the Zarr version 3 specification*",
            category=UserWarning,
        )

        vds_list = [
            open_virtual_dataset(
                url=u,
                registry=registry,
                parser=parser,
                loadable_variables=["time"],
                decode_times=True,
            )
            for u in local_urls
        ]

        if combine == "nested":
            vds = xr.concat(
                vds_list,
                dim="time",
                coords="minimal",
                data_vars="all",
                compat="override",
            )
        else:
            vds = xr.combine_by_coords(vds_list, combine_attrs="drop_conflicts")

    has_virtual_data = any(
        isinstance(v.data, ManifestArray) for v in vds.data_vars.values()
    )
    if not has_virtual_data:
        logger.warning(
            "VirtualiZarr produced no ManifestArray variables for %s. "
            "Falling back to Kerchunk HDF5 translation.",
            short_name,
        )
        try:
            from kerchunk.hdf import SingleHdf5ToZarr
            from kerchunk.combine import MultiZarrToZarr
        except ImportError as e:
            raise RuntimeError(
                "Virtual reference creation produced no ManifestArray variables, "
                "and Kerchunk fallback is unavailable. Install with: pip install kerchunk"
            ) from e

        ordered_files = sorted(
            local_files,
            key=lambda p: (_parse_vnp_doy_from_name(str(p)) or datetime.min, str(p)),
        )
        time_values = [
            (_parse_vnp_doy_from_name(str(f)) or datetime.min).strftime("%Y-%m-%d")
            for f in ordered_files
        ]

        keep_tokens_map = {
            "brdf": ["Parameter1", "Parameter3", "Percent_Snow", "Land_Water"],
            "nbar": ["Nadir", "NBAR"],
            "albedo": ["Albedo", "BSA", "WSA", "Percent_Snow"],
            "ndvi": ["NDVI", "EVI", "pixel_reliability"],
            "lai": ["LAI", "Fpar", "FPAR"],
        }
        keep_tokens = keep_tokens_map.get(product_type, [])

        def _filter_refs(ref_doc: dict) -> dict:
            refs = ref_doc.get("refs", {})
            if not keep_tokens:
                return ref_doc

            # Identify array roots under Data Fields that match desired science variables.
            selected_array_roots = set()
            for k in refs:
                if not k.endswith("/.zarray"):
                    continue
                if "Data Fields/" not in k:
                    continue
                if any(tok in k for tok in keep_tokens):
                    selected_array_roots.add(k[: -len("/.zarray")])

            filtered_refs = {}
            for k, v in refs.items():
                # Keep only global root metadata plus selected arrays and their chunks/attrs.
                if k in {".zgroup", ".zattrs"}:
                    filtered_refs[k] = v
                    continue

                if any(
                    k == f"{root}/.zarray"
                    or k == f"{root}/.zattrs"
                    or k.startswith(f"{root}/")
                    for root in selected_array_roots
                ):
                    filtered_refs[k] = v

            ref_doc["refs"] = filtered_refs
            return ref_doc

        single_refs = [
            _filter_refs(SingleHdf5ToZarr(f, inline_threshold=0).translate())
            for f in ordered_files
        ]
        combined_refs = MultiZarrToZarr(
            path=ordered_files,
            indicts=single_refs,
            concat_dims=["time"],
            coo_map={"time": time_values},
        ).translate()

        output_refs = Path(output_refs)
        output_refs.parent.mkdir(parents=True, exist_ok=True)
        if output_refs.suffix.lower() == ".parquet":
            output_refs = output_refs.with_suffix(".json")
            logger.warning(
                "Kerchunk fallback currently writes JSON refs. Using %s instead of parquet.",
                output_refs,
            )

        with output_refs.open("w") as f:
            json.dump(combined_refs, f)

        logger.info("Wrote Kerchunk fallback references to %s", output_refs)
        return output_refs

    output_refs = Path(output_refs)
    output_refs.parent.mkdir(parents=True, exist_ok=True)
    fmt = "parquet" if output_refs.suffix.lower() == ".parquet" else "json"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Numcodecs codecs are not in the Zarr version 3 specification*",
            category=UserWarning,
        )
        vds.vz.to_kerchunk(str(output_refs), format=fmt)
    logger.info("Wrote virtual references to %s", output_refs)
    return output_refs

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
    for d, s in ds.sizes.items():
        if d in ["phony_dim_0", "y"] or (s == 3600 and d != "lat"):
            rename_map[d] = "lat"
        if d in ["phony_dim_1", "x"] or (s == 7200 and d != "lon"):
            rename_map[d] = "lon"

    if rename_map:
        logger.debug(f"Renaming dimensions: {rename_map}")
        # Avoid naming conflicts with existing coords
        for v in ["lat", "lon"]:
            if v in ds.coords and v not in ds.sizes:
                ds = ds.drop_vars(v)
        ds = ds.rename(rename_map)

    # 2. Reconstruct/Ensure coordinates are correctly indexed
    # Even if they have the right size, we must assign values so interp/merge works cleanly.
    if "lat" in ds.sizes and "lon" in ds.sizes:
        curr_nlat = ds.sizes["lat"]
        curr_nlon = ds.sizes["lon"]

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

    # 4. Standardize variable names (remove long descriptive prefixes)
    new_data_vars = {}
    for v in ds.data_vars:
        new_name = None
        if "std dev" in v.lower():
            continue

        v_low = v.lower()
        if "ndvi" in v_low:
            new_name = "NDVI"
        elif "evi2" in v_low:
            new_name = "EVI2"
        elif "evi" in v_low:
            new_name = "EVI"
        elif "pixel reliability" in v_low:
            new_name = "pixel_reliability"
        elif "parameter1" in v_low or "isotropic" in v_low:
            # Extract band if present (e.g. M5, Band1, nir, vis)
            band_match = re.search(r"_(M\d+|Band\d+|nir|vis|shortwave)$", v_low)
            band = band_match.group(1) if band_match else ""
            if band and band.lower() != "m7": # Only keep M7 for NASA VIIRS
                 continue
            new_name = f"Isotropic_{band}" if band else "Isotropic"
        elif "parameter2" in v_low or "volumetric" in v_low:
            band_match = re.search(r"_(M\d+|Band\d+|nir|vis|shortwave)$", v_low)
            band = band_match.group(1) if band_match else ""
            if band and band.lower() != "m7":
                 continue
            new_name = f"Volumetric_{band}" if band else "Volumetric"
        elif "parameter3" in v_low or "geometric" in v_low:
            band_match = re.search(r"_(M\d+|Band\d+|nir|vis|shortwave)$", v_low)
            band = band_match.group(1) if band_match else ""
            if band and band.lower() != "m7":
                 continue
            new_name = f"Geometric_{band}" if band else "Geometric"
        elif "percent_snow" in v_low:
            new_name = "Percent_Snow"
        elif "land_water" in v_low or "landwater" in v_low or "land water" in v_low:
            new_name = "Land_Water_Type"

        if new_name and new_name != v:
            # Check for name collisions with variables, coordinates, or newly assigned names
            if new_name in ds.variables or new_name in new_data_vars.values():
                logger.warning(f"Name collision for '{new_name}'. Keeping original name '{v}'.")
            else:
                logger.debug(f"Renaming variable '{v}' to '{new_name}'")
                new_data_vars[v] = new_name

    if new_data_vars:
        ds = ds.rename(new_data_vars)

    return ds
