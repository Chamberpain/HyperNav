"""Compatibility entry point for current archives; implementation lives in GeneralUtilities."""
import argparse
from pathlib import Path
from GeneralUtilities.Data.Download import current_archive as _shared

globals().update({name: value for name, value in vars(_shared).items()
                  if not name.startswith('__')})

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", choices=("copernicus", "wcofs"))
    parser.add_argument("--start", required=True, help="Inclusive UTC start")
    parser.add_argument("--end", required=True, help="Exclusive UTC end")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--worker-index", type=int, help="Partition index for one of N terminals"
    )
    parser.add_argument("--chunk-days", type=int)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--max-depth", type=float, default=800, help="Physical meters positive down"
    )
    parser.add_argument(
        "--dataset-id", help="Copernicus dataset override for historical records"
    )
    parser.add_argument(
        "--time-step", help="Copernicus custom fixed cadence, e.g. 6h or 1d"
    )
    parser.add_argument(
        "--source",
        choices=("roms", "regulargrid"),
        help="WCOFS source (default: historical ROMS)",
    )
    parser.add_argument(
        "--url-template", help="WCOFS archive or local file path template"
    )
    parser.add_argument(
        "--depth-levels",
        type=float,
        nargs="+",
        help="WCOFS ROMS physical depths in meters",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        help="WCOFS ROMS geographic spacing in degrees (default: 0.05)",
    )
    parser.add_argument(
        "--source-engine",
        help="WCOFS xarray reader (default: pydap; use scipy/netcdf4 for local files)",
    )
    args = parser.parse_args(argv)
    options = {
        "workers": args.workers,
        "worker_index": args.worker_index,
        "retries": args.retries,
        "max_depth": args.max_depth,
    }
    if args.chunk_days is not None:
        options["chunk_days"] = args.chunk_days
    if args.model == "copernicus":
        if any(
            value is not None
            for value in (
                args.source,
                args.url_template,
                args.depth_levels,
                args.grid_spacing,
                args.source_engine,
            )
        ):
            parser.error("WCOFS source/regridding options apply only to wcofs.")
        from .CopernicusGlobal import HumboldtCopernicus

        if args.dataset_id:
            options["dataset_id"] = args.dataset_id
        if args.time_step:
            options["time_step"] = args.time_step
        model = HumboldtCopernicus
    else:
        if args.dataset_id:
            parser.error("--dataset-id applies only to Copernicus.")
        if args.time_step:
            parser.error("--time-step applies only to Copernicus.")
        from .WCOFS import WCOFSHumboldt

        options["source"] = args.source or "roms"
        for key, value in (
            ("url_template", args.url_template),
            ("depth_levels", args.depth_levels),
            ("grid_spacing", args.grid_spacing),
            ("engine", args.source_engine),
        ):
            if value is not None:
                options[key] = value
        model = WCOFSHumboldt
    manifest = model.download_record(args.start, args.end, args.output_dir, **options)
    selected = (
        manifest
        if args.worker_index is None
        else manifest.iloc[args.worker_index :: args.workers]
    )
    return int(selected.status.ne("downloaded").any())


if __name__ == "__main__":
    raise SystemExit(main())
