"""Tests for TLC data download functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nyc_taxi_strategy.data.download import download_all, download_month, get_parquet_url


class TestGetParquetUrl:
    def test_yellow_url(self):
        url = get_parquet_url("2024-01")
        assert "yellow_tripdata_2024-01.parquet" in url

    def test_custom_taxi_type(self):
        url = get_parquet_url("2024-01", taxi_type="green")
        assert "green_tripdata" in url

    def test_url_contains_base(self):
        url = get_parquet_url("2024-03")
        assert url.startswith("https://")


class TestDownloadMonth:
    def test_skips_existing_file(self, tmp_path):
        existing = tmp_path / "yellow_tripdata_2024-01.parquet"
        existing.write_bytes(b"fake")
        result = download_month("2024-01", tmp_path)
        assert result == existing

    def test_creates_output_dir(self, tmp_path):
        new_dir = tmp_path / "subdir"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content.return_value = [b"fake data"]
        mock_response.raise_for_status = MagicMock()

        with patch("nyc_taxi_strategy.data.download.requests.get", return_value=mock_response):
            result = download_month("2024-01", new_dir)
        assert new_dir.exists()

    def test_downloads_file(self, tmp_path):
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "8"}
        mock_response.iter_content.return_value = [b"fakefile"]
        mock_response.raise_for_status = MagicMock()

        with patch("nyc_taxi_strategy.data.download.requests.get", return_value=mock_response):
            result = download_month("2024-02", tmp_path)

        assert result.exists()
        assert result.read_bytes() == b"fakefile"

    def test_raises_on_http_error(self, tmp_path):
        import requests as req_lib
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req_lib.HTTPError("404")

        with patch("nyc_taxi_strategy.data.download.requests.get", return_value=mock_response):
            with pytest.raises(req_lib.HTTPError):
                download_month("2024-99", tmp_path)


class TestDownloadAll:
    def test_returns_list_of_paths(self, tmp_path):
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "4"}
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status = MagicMock()

        with patch("nyc_taxi_strategy.data.download.requests.get", return_value=mock_response):
            paths = download_all(["2024-01", "2024-02"], tmp_path)

        assert len(paths) == 2
        assert all(isinstance(p, Path) for p in paths)
