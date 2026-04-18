from pathlib import Path
import tarfile


class TarArchiveExtractor:
    def extract(self, archive_path: Path, destination: Path) -> None:
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(destination)
