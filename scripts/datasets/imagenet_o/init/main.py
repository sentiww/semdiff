from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from nltk.corpus import wordnet as wn

from scripts.common.synset import (
    _normalize_label,
    is_synset_id,
    decode_image_stem_to_synset_id,
)

logger = logging.getLogger("imagenet-o.init")

_IMAGE_SUFFIXES = {".jpeg"}


def run(url: str, output: Path) -> None:
    logger.info("Downloading imagenet-o archive")
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "imagenet-o.tar"
        urllib.request.urlretrieve(url, archive_path)

        extraction_root = Path(tmp) / "imagenet-o"
        extraction_root.mkdir()
        logger.info("Extract archive into temporary directory")
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(extraction_root)

        logger.info("Building WordNet synset decoder")
        synset_lookup: dict[str, list[str]] = {}
        for synset in wn.all_synsets(pos="n"):
            synset_id = f"{synset.pos()}{synset.offset():08d}"
            keys = {_normalize_label(lemma.name()) for lemma in synset.lemmas()}
            keys.add(_normalize_label(synset.name().split(".", 1)[0]))
            keys.add(
                _normalize_label("_".join(lemma.name() for lemma in synset.lemmas()))
            )
            for key in keys:
                if not key:
                    continue
                values = synset_lookup.setdefault(key, [])
                if synset_id not in values:
                    values.append(synset_id)

        for path in output.iterdir():
            if path.is_dir() and is_synset_id(path.name):
                shutil.rmtree(path)

        count = 0

        logger.info("Reorganizing imagenet-o images into synset folders at %s", output)
        for image_path in sorted(extraction_root.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.name == "README.txt":
                continue
            if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue

            synset_id = decode_image_stem_to_synset_id(image_path.stem, synset_lookup)
            target_dir = output / synset_id
            target_dir.mkdir(exist_ok=True)
            shutil.move(str(image_path), str(target_dir / image_path.name))
            count += 1

    logger.info("Done: %s images", count)
