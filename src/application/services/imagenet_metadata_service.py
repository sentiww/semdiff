from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scipy.io import loadmat


@dataclass(frozen=True)
class ImageNetSynset:
    imagenet_id: int
    wnid: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class ImageNetClassIndexMaps:
    wnid_to_index: dict[str, int]
    index_to_wnid: dict[int, str]


class ImageNetMetadataService:
    def load_synsets(self, meta_path: Path) -> list[ImageNetSynset]:
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing ImageNet metadata at {meta_path}")

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

        return validation_wnids

    def load_wnid_to_index(self, class_index_path: Path) -> dict[str, int]:
        return self.load_class_index_maps(class_index_path).wnid_to_index

    def load_index_to_wnid(self, class_index_path: Path) -> dict[int, str]:
        return self.load_class_index_maps(class_index_path).index_to_wnid

    def load_class_index_maps(
        self,
        class_index_path: Path,
    ) -> ImageNetClassIndexMaps:
        payload = json.loads(class_index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected a JSON object in {class_index_path}")

        index_to_wnid: dict[int, str] = {}
        wnid_to_index: dict[str, int] = {}

        for raw_index, item in payload.items():
            index = int(raw_index)
            if not isinstance(item, list) or not item or not isinstance(item[0], str):
                raise RuntimeError(
                    f"Expected [wnid, ...] entries in {class_index_path}: index={raw_index!r}"
                )

            wnid = item[0]
            if wnid in wnid_to_index:
                raise RuntimeError(f"Duplicate synset {wnid} in {class_index_path}")

            index_to_wnid[index] = wnid
            wnid_to_index[wnid] = index

        if len(index_to_wnid) != 1000:
            raise RuntimeError(
                f"Expected 1000 ImageNet class-index entries in {class_index_path}, "
                f"found {len(index_to_wnid)}"
            )

        return ImageNetClassIndexMaps(
            wnid_to_index=wnid_to_index,
            index_to_wnid=index_to_wnid,
        )
