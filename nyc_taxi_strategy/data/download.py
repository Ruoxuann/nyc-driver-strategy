"""Download NYC TLC trip record parquet files."""

import logging
from pathlib import Path

import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def get_parquet_url(month: str, taxi_type: str = "yellow") -> str:
    """Construct the TLC download URL for a given month.

    Args:
        month: Month string in 'YYYY-MM' format.
        taxi_type: One of 'yellow', 'green', 'fhv', 'fhvhv'.

    Returns:
        Full URL to the parquet file.
    """
    return f"{TLC_BASE_URL}/{taxi_type}_tripdata_{month}.parquet"


def download_month(month: str, output_dir: str | Path, taxi_type: str = "yellow") -> Path:
    """Download a single month of trip data.

    Args:
        month: Month string in 'YYYY-MM' format.
        output_dir: Directory to save the file.
        taxi_type: Type of taxi data.

    Returns:
        Path to the downloaded file.

    Raises:
        requests.HTTPError: If the download fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{taxi_type}_tripdata_{month}.parquet"
    filepath = output_dir / filename

    if filepath.exists():
        logger.info(f"Already exists: {filepath}")
        return filepath

    url = get_parquet_url(month, taxi_type)
    logger.info(f"Downloading {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(filepath, "wb") as f:
        if HAS_TQDM:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    logger.info(f"Saved to {filepath}")
    return filepath


def download_all(months: list[str], output_dir: str | Path, taxi_type: str = "yellow") -> list[Path]:
    """Download multiple months of trip data.

    Args:
        months: List of month strings in 'YYYY-MM' format.
        output_dir: Directory to save files.
        taxi_type: Type of taxi data.

    Returns:
        List of paths to downloaded files.
    """
    paths = []
    for month in months:
        path = download_month(month, output_dir, taxi_type)
        paths.append(path)
    return paths


if __name__ == "__main__":
    import argparse
    from nyc_taxi_strategy.utils.config import load_config

    parser = argparse.ArgumentParser(description="Download NYC TLC parquet files")
    parser.add_argument("--months", nargs="+", required=True, help="e.g. 2024-01 2024-02")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    download_all(args.months, cfg.data.raw_dir)
