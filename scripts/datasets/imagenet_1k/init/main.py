from __future__ import annotations

import logging
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from scipy.io import loadmat

logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class ImageNetSynset:
    imagenet_id: int
    wnid: str
    labels: tuple[str, ...]


def run(output: Path, archive_path: Path, ground_truth: Path, meta: Path) -> None:
    logger.info("Loading metadata")

    if not meta.exists():
        raise FileNotFoundError(f"Missing ImageNet metadata at {meta}")

    synsets: list[ImageNetSynset] = []
    for entry in loadmat(meta, squeeze_me=True)["synsets"]:
        imagenet_id = int(entry[0])
        wnid = str(entry[1])
        num_children = int(entry[4])
        if num_children != 0 or not wnid.startswith("n"):
            continue

        labels = tuple(part.strip() for part in str(entry[2]).split(", "))
        synsets.append(
            ImageNetSynset(
                imagenet_id=imagenet_id,
                wnid=wnid,
                labels=labels,
            )
        )
    imagenet_id_to_wnid: dict[int, str] = {}
    for synset in synsets:
        imagenet_id_to_wnid[synset.imagenet_id] = synset.wnid

    validation_ids: list[int] = []
    for line in ground_truth.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if stripped_line:
            validation_ids.append(int(stripped_line))

    validation_wnids: list[str] = []
    for imagenet_id in validation_ids:
        validation_wnids.append(imagenet_id_to_wnid[imagenet_id])

    if len(validation_wnids) != 50000:
        raise RuntimeError(
            "Expected 50000 ImageNet validation labels in "
            f"{ground_truth}, found {len(validation_wnids)}"
        )

    logger.info("Extracting archive")
    count = 0
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(tmp)

        for image in sorted(Path(tmp).glob("*.JPEG")):
            match = re.fullmatch(r"ILSVRC2012_val_(\d+)", image.stem)
            if match is None:
                raise RuntimeError(
                    f"Unexpected ImageNet validation filename: {image.name}"
                )

            index = int(match.group(1))
            target = output / validation_wnids[index - 1]
            target.mkdir(exist_ok=True)
            shutil.move(str(image), str(target / image.name))
            count += 1

    logger.info("Done: %s images", count)
