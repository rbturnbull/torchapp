import io
from pathlib import Path
from unittest.mock import patch
import pytest

from torchapp.download import download_file, cached_download, DownloadError


@pytest.fixture
def dummy_url():
    return "https://example.com/fakefile.txt"


@pytest.fixture
def dummy_data():
    return b"Test content for the file."


def make_mock_response(data):
    stream = io.BytesIO(data)
    stream.length = len(data)
    return stream


def test_download_file_success(tmp_path: Path, dummy_url, dummy_data):
    local_file = tmp_path / "downloaded.txt"

    with patch("urllib.request.urlopen", return_value=make_mock_response(dummy_data)):
        result_path = download_file(dummy_url, local_file)
    
    assert result_path.exists()
    assert result_path.read_bytes() == dummy_data


def test_cached_download_skips_if_exists(tmp_path: Path, dummy_url, dummy_data):
    local_file = tmp_path / "cached.txt"
    local_file.write_bytes(dummy_data)

    with patch("urllib.request.urlopen") as mock_urlopen:
        cached_download(dummy_url, local_file)
    
    # Since file exists and force=False, it should not download
    mock_urlopen.assert_not_called()


def test_cached_download_force(tmp_path: Path, dummy_url, dummy_data):
    local_file = tmp_path / "force.txt"

    with patch("urllib.request.urlopen", return_value=make_mock_response(dummy_data)) as mock_urlopen:
        cached_download(dummy_url, local_file, force=True)

    mock_urlopen.assert_called()
    assert local_file.exists()
    assert local_file.read_bytes() == dummy_data


def test_cached_download_fallback_context(tmp_path: Path, dummy_url, dummy_data):
    local_file = tmp_path / "fallback.txt"

    # First call fails, second with context succeeds
    with patch("urllib.request.urlopen", side_effect=[
        Exception("Initial failure"),
        make_mock_response(dummy_data)
    ]) as mock_urlopen:
        cached_download(dummy_url, local_file, force=True)

    assert mock_urlopen.call_count == 2
    assert local_file.exists()
    assert local_file.read_bytes() == dummy_data


def test_cached_download_failure(tmp_path: Path, dummy_url):
    local_file = tmp_path / "fail.txt"

    with patch("urllib.request.urlopen", side_effect=Exception("fail")):
        with pytest.raises(DownloadError):
            cached_download(dummy_url, local_file, force=True)


def test_cached_download_empty_file(tmp_path: Path, dummy_url):
    local_file = tmp_path / "empty.txt"
    local_file.write_bytes(b"")

    with patch("urllib.request.urlopen", return_value=make_mock_response(b"")):
        with pytest.raises(IOError):
            cached_download(dummy_url, local_file, force=True)
