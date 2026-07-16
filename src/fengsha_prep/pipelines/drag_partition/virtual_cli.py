import argparse
import logging

from .io import build_earthaccess_virtual_refs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build VirtualiZarr/Kerchunk references from Earthaccess granules."
    )
    parser.add_argument("product_type", choices=["brdf", "albedo", "nbar", "lai", "ndvi"])
    parser.add_argument("start_date", help="YYYY-MM-DD")
    parser.add_argument("end_date", help="YYYY-MM-DD")
    parser.add_argument("--sensor", default="VJ1", choices=["MODIS", "VNP", "VJ1"])
    parser.add_argument("--output-refs", default="data/refs/drag_partition.parquet")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--combine", default="nested", choices=["nested", "by_coords"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ref_path = build_earthaccess_virtual_refs(
        product_type=args.product_type,
        start_date=args.start_date,
        end_date=args.end_date,
        sensor=args.sensor,
        output_refs=args.output_refs,
        cache_dir=args.cache_dir,
        combine=args.combine,
    )
    print(ref_path)


if __name__ == "__main__":
    main()
