from typing import Protocol

from torch.utils.data import DataLoader

from application.services.synset_image_folder import SynsetImageFolder


class DataLoaderFactory(Protocol):
    def create(self, image_dataset: SynsetImageFolder) -> DataLoader: ...
