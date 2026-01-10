import argparse
import asyncio
import datetime
import logging

from fengsha_prep.pipelines.dust_scan.pipeline import run_dust_scan_in_period

# Configure basic logging for the example script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to parse arguments and run the dust scan pipeline."""
    parser = argparse.ArgumentParser(description="Scan satellite data for dust events.")
    parser.add_argument(
        "--sat", type=str, default="goes16", help="Satellite ID (e.g., 'goes16')."
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.datetime.fromisoformat(s),
        required=True,
        help="Start time in YYYY-MM-DDTHH:MM format.",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.datetime.fromisoformat(s),
        required=True,
        help="End time in YYYY-MM-DDTHH:MM format.",
    )
    parser.add_argument(
        "--output", type=str, default="dust_events.csv", help="Output CSV file path."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory for local satellite data. Used for non-S3 satellites.",
    )
    args = parser.parse_args()

    asyncio.run(
        run_dust_scan_in_period(
            start_time=args.start,
            end_time=args.end,
            sat_id=args.sat,
            output_csv=args.output,
            data_dir=args.data_dir,
        )
    )


if __name__ == "__main__":
    main()
