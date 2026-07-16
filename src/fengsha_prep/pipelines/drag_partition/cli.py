import argparse
import logging
from pathlib import Path

from .pipeline import run_drag_partition_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FENGSHA drag-partition processing for a date or date range."
    )
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", nargs="?", help="End date (YYYY-MM-DD). Defaults to start_date")
    parser.add_argument(
        "--sensor",
        choices=["MODIS", "VNP", "VJ1", "NESDIS"],
        default="VJ1",
        help="Input sensor family.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for daily NetCDF outputs.",
    )
    parser.add_argument(
        "--ndvi-threshold",
        type=float,
        default=0.15,
        help="NDVI threshold used when NDVI adjustment is enabled.",
    )
    parser.add_argument(
        "--use-lai",
        action="store_true",
        help="Enable LAI fetching/usage.",
    )
    parser.add_argument(
        "--use-gvf-adjustment",
        action="store_true",
        help="Enable GVF attenuation (default off).",
    )
    parser.add_argument(
        "--use-ndvi-adjustment",
        action="store_true",
        help="Enable NDVI attenuation (default off).",
    )
    parser.add_argument(
        "--vegetation-gamma",
        type=float,
        default=1.0,
        help="Gamma exponent for non-linear washout protection.",
    )
    parser.add_argument(
        "--bare-threshold",
        type=float,
        default=0.80,
        help="Upper threshold for non-linear washout protection.",
    )
    parser.add_argument(
        "--no-cleanup-downloads",
        action="store_true",
        help="Keep downloaded source files under data/ after run.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory for fetched inputs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    end_date = args.end_date or args.start_date

    result = run_drag_partition_pipeline(
        start_date=args.start_date,
        end_date=end_date,
        sensor=args.sensor,
        use_lai=args.use_lai,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        ndvi_threshold=args.ndvi_threshold,
        use_gvf_adjustment=args.use_gvf_adjustment,
        use_ndvi_adjustment=args.use_ndvi_adjustment,
        vegetation_gamma=args.vegetation_gamma,
        bare_threshold=args.bare_threshold,
        cleanup_downloads=not args.no_cleanup_downloads,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )

    if isinstance(result, list):
        print(f"Wrote {len(result)} file(s).")
        for path in result:
            print(path)
    else:
        print(result)


if __name__ == "__main__":
    main()
