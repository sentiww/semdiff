from __future__ import annotations

from pathlib import Path

from application.services.imagenet_metadata_service import ImageNetMetadataService


class DefaultValidationLabelLoader:
    def __init__(self, wordnet_service: ImageNetMetadataService) -> None:
        self._wordnet_service = wordnet_service

    def load_validation_wnids(
        self,
        meta_path: Path,
        ground_truth_path: Path,
    ) -> list[str]:
        imagenet_id_to_wnid = self._wordnet_service.load_imagenet_id_to_wnid(meta_path)

        validation_ids: list[int] = []
        for line in ground_truth_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                validation_ids.append(int(stripped))

        validation_wnids = [
            imagenet_id_to_wnid[imagenet_id] for imagenet_id in validation_ids
        ]
        if len(validation_wnids) != 50000:
            raise RuntimeError(
                "Expected 50000 ImageNet validation labels in "
                f"{ground_truth_path}, found {len(validation_wnids)}"
            )
        return validation_wnids
