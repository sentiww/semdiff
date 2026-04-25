from __future__ import annotations

from functools import cached_property

from bootstrap.containers.base import _ImageNetMetadataMixin


class DatasetContainer(_ImageNetMetadataMixin):
    @cached_property
    def archive_fetcher(self):
        from features.datasets.filesystem import UrlLibArchiveFetcher

        return UrlLibArchiveFetcher()

    @cached_property
    def archive_extractor(self):
        from features.datasets.filesystem import TarArchiveExtractor

        return TarArchiveExtractor()

    @cached_property
    def output_preparer(self):
        from features.datasets.filesystem import GeneratedSynsetDirectoryPreparer

        return GeneratedSynsetDirectoryPreparer()

    @cached_property
    def file_mover(self):
        from features.datasets.filesystem import ShutilFileMover

        return ShutilFileMover()

    @cached_property
    def index_parser(self):
        from features.datasets.filesystem import ImageNetValidationImageIndexParser

        return ImageNetValidationImageIndexParser()
