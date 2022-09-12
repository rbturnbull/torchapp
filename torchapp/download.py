from pathlib import Path
from typing import Union
import urllib.request

class DownloadError(Exception):
    pass


def cached_download(url: str, local_path: Union[str, Path], force: bool = False) -> None:
    """
    Downloads a file if a local file does not already exist.

    Args:
        url (str): The url of the file to download.
        local_path (str, Path): The local path of where the file should be.
            If this file isn't there or the file size is zero then this function downloads it to this location.
        force (bool): Whether or not the file should be forced to download again even if present in the local path.
            Default False.

    Raises:
        DownloadError: Raises an exception if it cannot download the file.
        IOError: Raises an exception if the file does not exist or is empty after downloading.
    """
    local_path = Path(local_path)
    if (not local_path.exists() or local_path.stat().st_size == 0) or force:
        try:
            print(f"Downloading {url} to {local_path}")
            urllib.request.urlretrieve(url, local_path)
        except:
            raise DownloadError(f"Error downloading {url}")

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise IOError(f"Error reading {local_path}")
