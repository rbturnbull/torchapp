from pathlib import Path
from typing import Union
import ssl, certifi
import urllib.request
from rich.progress import Progress

class DownloadError(Exception):
    pass


def download_file(url:str, local_path:Path, chunk_size:int=16 * 1024, context=None):
    """ adapted from https://stackoverflow.com/a/1517728 """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url, context=context) as response, open(local_path, 'wb') as f, Progress() as progress:
        task = progress.add_task("[red]Downloading", total=response.length)
        while True:
            chunk = response.read(chunk_size)
            progress.update(task, advance=chunk_size)
            if not chunk:
                break
            f.write(chunk)
        
    return local_path


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
            download_file(url, local_path)
        except Exception:
            try:
                download_file(url, local_path, context=ssl.create_default_context(cafile=certifi.where()))
            except Exception as err:                    
                raise DownloadError(f"Error downloading {url} to {local_path}:\n{err}")

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise IOError(f"Error reading {local_path}")
