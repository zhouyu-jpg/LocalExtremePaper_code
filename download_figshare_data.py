"""Download the GHCNd station files archived on Figshare."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


ARTICLE_ID = 30370084
ARTICLE_API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
BUFFER_SIZE = 1024 * 1024


def fetch_manifest() -> dict:
    """Fetch the public Figshare article manifest."""
    request = urllib.request.Request(
        ARTICLE_API_URL,
        headers={"User-Agent": "local-temperature-extremes-data-downloader/1.0"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def data_files(manifest: dict, stations: set[str]) -> list[dict]:
    """Select safe file entries assigned to the Figshare Data folder."""
    folders = {str(key): value for key, value in manifest["folder_structure"].items()}
    selected = []
    for entry in manifest["files"]:
        if folders.get(str(entry["id"])) != "Data":
            continue
        name = entry["name"]
        if Path(name).name != name:
            raise ValueError(f"Unsafe filename in Figshare manifest: {name!r}")
        if stations and Path(name).stem not in stations:
            continue
        selected.append(entry)
    return sorted(selected, key=lambda item: item["name"])


def download(entry: dict, destination: Path, overwrite: bool) -> str:
    """Download one file atomically and confirm its published size."""
    expected_size = int(entry["size"])
    if destination.exists() and not overwrite and destination.stat().st_size == expected_size:
        return "existing file confirmed"

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        entry["download_url"],
        headers={"User-Agent": "local-temperature-extremes-data-downloader/1.0"},
    )
    bytes_written = 0
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as stream:
            while block := response.read(BUFFER_SIZE):
                stream.write(block)
                bytes_written += len(block)
        if bytes_written != expected_size:
            raise ValueError(
                f"Size mismatch for {entry['name']}: expected {expected_size}, got {bytes_written}"
            )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return "downloaded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GHCNd station CSVs from Figshare article 30370084 v3."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Data",
        help="destination directory (default: repository Data directory)",
    )
    parser.add_argument(
        "--station",
        action="append",
        default=[],
        metavar="ID",
        help="download only this station ID; may be repeated",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="redownload files even when an existing file has the expected size",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list selected files without downloading",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stations = {station.removesuffix(".csv") for station in args.station}
    try:
        manifest = fetch_manifest()
        selected = data_files(manifest, stations)
        if stations:
            found = {Path(entry["name"]).stem for entry in selected}
            missing = sorted(stations - found)
            if missing:
                raise ValueError(f"Station(s) not present in the record: {', '.join(missing)}")
        if not selected:
            raise ValueError("No data files selected")

        total_bytes = sum(entry["size"] for entry in selected)
        print(f"Selected {len(selected)} file(s), {total_bytes / 1_000_000:.1f} MB total")
        if args.list:
            for entry in selected:
                print(f"{entry['name']}\t{entry['size']}")
            return 0

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for index, entry in enumerate(selected, start=1):
            destination = args.output_dir / entry["name"]
            status = download(entry, destination, args.overwrite)
            print(f"[{index}/{len(selected)}] {entry['name']}: {status}")
    except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
