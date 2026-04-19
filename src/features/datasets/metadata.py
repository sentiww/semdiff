from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from scipy.io import loadmat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageNetSynset:
    imagenet_id: int
    wnid: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class ImageNetClassIndexMaps:
    wnid_to_index: dict[str, int]
    index_to_wnid: dict[int, str]
    index_to_label: dict[int, str]
    wnid_to_label: dict[str, str]


@dataclass(frozen=True)
class SynsetLabelMap:
    normalized_label_to_wnid: dict[str, str]


class ImageNetMetadataService:
    def load_synsets(self, meta_path: Path) -> list[ImageNetSynset]:
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing ImageNet metadata at {meta_path}")

        logger.info("Loading ImageNet synset metadata from %s", meta_path)
        raw_payload = loadmat(meta_path, squeeze_me=True)
        if "synsets" not in raw_payload:
            raise RuntimeError(
                f"Missing 'synsets' entry in ImageNet metadata at {meta_path}"
            )

        synsets: list[ImageNetSynset] = []
        for entry in raw_payload["synsets"]:
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

        if len(synsets) != 1000:
            raise RuntimeError(
                f"Expected 1000 ImageNet leaf synsets in {meta_path}, found {len(synsets)}"
            )

        logger.info("Loaded %s ImageNet leaf synsets from %s", len(synsets), meta_path)
        return synsets

    def load_imagenet_id_to_wnid(self, meta_path: Path) -> dict[int, str]:
        imagenet_id_to_wnid: dict[int, str] = {}
        for synset in self.load_synsets(meta_path):
            imagenet_id_to_wnid[synset.imagenet_id] = synset.wnid
        return imagenet_id_to_wnid

    def load_validation_wnids(
        self,
        meta_path: Path,
        ground_truth_path: Path,
    ) -> list[str]:
        logger.info(
            "Loading ImageNet validation wnids from meta=%s ground_truth=%s",
            meta_path,
            ground_truth_path,
        )
        imagenet_id_to_wnid = self.load_imagenet_id_to_wnid(meta_path)

        validation_ids: list[int] = []
        for line in ground_truth_path.read_text(encoding="utf-8").splitlines():
            stripped_line = line.strip()
            if stripped_line:
                validation_ids.append(int(stripped_line))

        validation_wnids = [
            imagenet_id_to_wnid[imagenet_id] for imagenet_id in validation_ids
        ]
        if len(validation_wnids) != 50000:
            raise RuntimeError(
                "Expected 50000 ImageNet validation labels in "
                f"{ground_truth_path}, found {len(validation_wnids)}"
            )

        logger.info(
            "Loaded %s ImageNet validation wnids from %s",
            len(validation_wnids),
            ground_truth_path,
        )
        return validation_wnids

    def load_class_index_maps(
        self,
        class_map_path: Path,
        index_to_wnid_path: Path,
    ) -> ImageNetClassIndexMaps:
        logger.info("Loading ImageNet class map from %s", class_map_path)
        raw_class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
        if not isinstance(raw_class_map, dict):
            raise RuntimeError(f"Expected a JSON object in {class_map_path}")

        wnid_to_label: dict[str, str] = {}
        for raw_wnid, raw_label in cast(dict[str, object], raw_class_map).items():
            if not isinstance(raw_wnid, str) or not isinstance(raw_label, str):
                raise RuntimeError(
                    f"Expected string wnid-to-label entries in {class_map_path}: {raw_wnid!r}"
                )
            wnid_to_label[raw_wnid] = raw_label

        logger.info("Loading ImageNet index-to-wnid map from %s", index_to_wnid_path)
        raw_index_map = json.loads(index_to_wnid_path.read_text(encoding="utf-8"))
        if not isinstance(raw_index_map, dict):
            raise RuntimeError(f"Expected a JSON object in {index_to_wnid_path}")

        index_to_wnid: dict[int, str] = {}
        wnid_to_index: dict[str, int] = {}
        index_to_label: dict[int, str] = {}

        for raw_index, raw_wnid in cast(dict[str, object], raw_index_map).items():
            index = int(raw_index)
            if not isinstance(raw_wnid, str):
                raise RuntimeError(
                    f"Expected string wnids in {index_to_wnid_path}: index={raw_index!r}"
                )

            wnid = raw_wnid
            try:
                label = wnid_to_label[wnid]
            except KeyError as exc:
                raise RuntimeError(
                    f"Missing label for ImageNet class {wnid!r} from {class_map_path}"
                ) from exc
            if wnid in wnid_to_index:
                raise RuntimeError(f"Duplicate synset {wnid} in {index_to_wnid_path}")

            index_to_wnid[index] = wnid
            wnid_to_index[wnid] = index
            index_to_label[index] = label

        if len(wnid_to_label) != 1000:
            raise RuntimeError(
                f"Expected 1000 ImageNet class-map entries in {class_map_path}, "
                f"found {len(wnid_to_label)}"
            )
        if len(index_to_wnid) != 1000:
            raise RuntimeError(
                f"Expected 1000 ImageNet class-index entries in {index_to_wnid_path}, "
                f"found {len(index_to_wnid)}"
            )

        logger.info(
            "Loaded %s ImageNet class-index entries from class_map=%s and index_to_wnid=%s",
            len(index_to_wnid),
            class_map_path,
            index_to_wnid_path,
        )
        return ImageNetClassIndexMaps(
            wnid_to_index=wnid_to_index,
            index_to_wnid=index_to_wnid,
            index_to_label=index_to_label,
            wnid_to_label=wnid_to_label,
        )

    def load_synset_label_map(self, path: Path) -> SynsetLabelMap:
        logger.info("Loading synset label map from %s", path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected a JSON object in {path}")

        normalized_label_to_wnid: dict[str, str] = {}
        for raw_wnid, raw_label in payload.items():
            if not isinstance(raw_wnid, str) or not isinstance(raw_label, str):
                raise RuntimeError(
                    f"Expected string wnid-to-label entries in {path}: {raw_wnid!r}"
                )
            normalized_label = _normalize_label(raw_label)
            if not normalized_label:
                raise RuntimeError(f"Empty label entry in {path}: {raw_label!r}")
            existing = normalized_label_to_wnid.get(normalized_label)
            if existing is not None and existing != raw_wnid:
                raise RuntimeError(
                    f"Conflicting wnids for normalized label {normalized_label!r} in {path}: "
                    f"{existing!r} vs {raw_wnid!r}"
                )
            normalized_label_to_wnid[normalized_label] = raw_wnid

        if not normalized_label_to_wnid:
            raise RuntimeError(f"Expected at least one label entry in {path}")

        logger.info(
            "Loaded %s normalized synset label entries from %s",
            len(normalized_label_to_wnid),
            path,
        )
        return SynsetLabelMap(normalized_label_to_wnid=normalized_label_to_wnid)

    def resolve_synset_id_from_label(
        self,
        label: str,
        synset_label_map: SynsetLabelMap,
    ) -> str:
        normalized_label = _normalize_label(label)
        if not normalized_label:
            raise RuntimeError(f"Could not resolve synset id from empty label: {label!r}")
        try:
            return synset_label_map.normalized_label_to_wnid[normalized_label]
        except KeyError as exc:
            raise RuntimeError(f"Unknown synset label {label!r}") from exc


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
