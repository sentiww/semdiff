from __future__ import annotations

import logging
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from .common import IMAGENET_O_ROOT

IMAGENET_O_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar"


def init_dataset(*, logger: logging.Logger) -> None:
    logger.info("Downloading imagenet-o archive")
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "imagenet-o.tar"
        urllib.request.urlretrieve(IMAGENET_O_URL, archive_path)

        logger.info("Extract archive into %s", IMAGENET_O_ROOT)
        subprocess.run(
            [
                "tar",
                "--strip-components=1",
                "--exclude=imagenet-o/README.txt",
                "-xf",
                str(archive_path),
                "-C",
                str(IMAGENET_O_ROOT),
            ],
            check=True,
        )

    logger.info("Done")
