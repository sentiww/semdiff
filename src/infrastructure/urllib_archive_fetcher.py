import urllib.request
from pathlib import Path


class UrlLibArchiveFetcher:
    def fetch(self, url: str, destination: Path) -> None:
        urllib.request.urlretrieve(url, destination)
