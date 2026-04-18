from __future__ import annotations

import logging
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from .common import IMAGENET_1K_ROOT
from .imagenet_mappings import load_validation_wnids


def init_dataset(*, logger: logging.Logger) -> None:
    root = IMAGENET_1K_ROOT
    val_tar = root / "ILSVRC2012_img_val.tar"
    ground_truth = root / "ILSVRC2012_validation_ground_truth.txt"
    meta_mat = root / "meta.mat"

    logger.info("Loading metadata")
    validation_wnids = load_validation_wnids(meta_mat, ground_truth)

    logger.info("Extracting archive")
    count = 0
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(val_tar, "r") as archive:
            archive.extractall(tmp)

        for image in sorted(Path(tmp).glob("*.JPEG")):
            match = re.fullmatch(r"ILSVRC2012_val_(\d+)", image.stem)
            if match is None:
                raise RuntimeError(f"Unexpected ImageNet validation filename: {image.name}")

            index = int(match.group(1))
            target = root / validation_wnids[index - 1]
            target.mkdir(exist_ok=True)
            shutil.move(str(image), str(target / image.name))
            count += 1

    logger.info("Done: %s images", count)
