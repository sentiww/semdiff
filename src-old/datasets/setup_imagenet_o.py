from __future__ import annotations

import logging
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from nltk.corpus import wordnet as wn

from .common import IMAGENET_O_ROOT

IMAGENET_O_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar"
_IMAGE_SUFFIXES = {".jpeg"}
_SYNSET_DIR_PATTERN = re.compile(r"n\d+$")
_PREDICTION_SCORE_SUFFIX = re.compile(r"_[0-9]+(?:\.[0-9]+)?$")


def init_dataset(*, logger: logging.Logger) -> None:
    logger.info("Downloading imagenet-o archive")
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "imagenet-o.tar"
        urllib.request.urlretrieve(IMAGENET_O_URL, archive_path)

        extraction_root = Path(tmp) / "imagenet-o"
        extraction_root.mkdir()
        logger.info("Extract archive into temporary directory")
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(extraction_root)

        logger.info("Building WordNet synset decoder")
        synset_lookup = _build_synset_lookup()
        _clear_generated_synset_dirs()
        count = 0

        logger.info(
            "Reorganizing imagenet-o images into synset folders at %s", IMAGENET_O_ROOT
        )
        for image_path in sorted(extraction_root.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.name == "README.txt":
                continue
            if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue

            synset_id = _decode_image_name_to_synset_id(
                image_path.stem,
                synset_lookup,
                logger=logger,
            )
            target_dir = IMAGENET_O_ROOT / synset_id
            target_dir.mkdir(exist_ok=True)
            shutil.move(str(image_path), str(target_dir / image_path.name))
            count += 1

    logger.info("Done: %s images", count)


def _build_synset_lookup() -> dict[str, list[str]]:
    synsets = list(wn.all_synsets(pos="n"))

    lookup: dict[str, list[str]] = {}
    for synset in synsets:
        synset_id = f"{synset.pos()}{synset.offset():08d}"
        keys = _build_lookup_keys(synset)
        keys.add(_normalize_label(synset.name().split(".", 1)[0]))
        for key in keys:
            if not key:
                continue
            if key not in lookup:
                lookup[key] = []
            values = lookup[key]
            if synset_id not in values:
                values.append(synset_id)
    return lookup


def _build_lookup_keys(synset: object) -> set[str]:
    keys: set[str] = set()
    lemma_names: list[str] = []
    for lemma in synset.lemmas():
        lemma_name = lemma.name()
        lemma_names.append(lemma_name)
        keys.add(_normalize_label(lemma_name))

    combined_lemma_name = "_".join(lemma_names)
    keys.add(_normalize_label(combined_lemma_name))
    return keys


def _decode_image_name_to_synset_id(
    stem: str,
    synset_lookup: dict[str, list[str]],
    *,
    logger: logging.Logger,
) -> str:
    normalized_label = _normalize_label(_strip_prediction_score(stem))
    direct_matches = synset_lookup.get(normalized_label, [])
    if len(direct_matches) == 1:
        return direct_matches[0]

    ranked_matches = wn.synsets(normalized_label.replace(" ", "_"), pos="n")
    if ranked_matches:
        synset_id = f"n{ranked_matches[0].offset():08d}"
        logger.info(
            "Non-exact synset match for %r: used WordNet ranked fallback %s",
            stem,
            synset_id,
        )
        return synset_id

    if direct_matches:
        synset_id = direct_matches[0]
        logger.info(
            "Non-exact synset match for %r: used first direct match %s from %s matches",
            stem,
            synset_id,
            len(direct_matches),
        )
        return synset_id
    raise RuntimeError(
        f"Could not decode synset id from imagenet-o filename stem: {stem!r}"
    )


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _strip_prediction_score(stem: str) -> str:
    return _PREDICTION_SCORE_SUFFIX.sub("", stem)


def _clear_generated_synset_dirs() -> None:
    for path in IMAGENET_O_ROOT.iterdir():
        if not path.is_dir():
            continue
        if not _SYNSET_DIR_PATTERN.fullmatch(path.name):
            continue
        shutil.rmtree(path)
