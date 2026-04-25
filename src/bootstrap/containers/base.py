from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from features.files import FileStore


class _FileStoreMixin:
    @cached_property
    def _file_codecs(self):
        from features.files import build_default_codecs

        return build_default_codecs()

    @cached_property
    def _file_store(self) -> FileStore:
        from features.files import FileStore

        return FileStore(codecs=self._file_codecs)


class _WordNetMixin:
    @cached_property
    def _wordnet(self):
        from features.wordnet.service import WordNetService

        return WordNetService()


class _ImageNetMetadataMixin:
    @cached_property
    def _imagenet_metadata_service(self):
        from features.datasets.metadata import ImageNetMetadataService

        return ImageNetMetadataService()
