from __future__ import annotations

from functools import cached_property

from bootstrap.containers.base import _FileStoreMixin, _WordNetMixin


class AnalysisContainer(_FileStoreMixin, _WordNetMixin):
    @cached_property
    def semantic_analysis_service(self):
        from features.wordnet.analysis import SemanticAnalysisService

        return SemanticAnalysisService(file_store=self._file_store)
