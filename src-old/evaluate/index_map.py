from __future__ import annotations

from datasets.common import dataset_root
from datasets.imagenet_mappings import load_index_to_wnid
from datasets.imagenet_mappings import load_wnid_to_index


def load_imagenet_synset_index_map(
    categories: list[str],
) -> tuple[dict[str, int], dict[int, str]]:
    class_index_path = dataset_root("imagenet-1k") / "torchvision_class_index.json"
    if not class_index_path.exists():
        raise FileNotFoundError(
            f"Missing ImageNet class index at {class_index_path}. "
            "The evaluator needs datasets/imagenet-1k/torchvision_class_index.json."
        )

    synset_to_index = load_wnid_to_index(class_index_path)
    index_to_synset = load_index_to_wnid(class_index_path)
    if len(index_to_synset) != len(categories):
        raise RuntimeError(
            "ImageNet class index does not match the model categories: "
            f"{len(index_to_synset)} vs {len(categories)}"
        )

    return synset_to_index, index_to_synset
