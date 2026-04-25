from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset

    from semdiff.core.codecs import EntityCodec
    from semdiff.files.store import EntitySink, EntitySource


@runtime_checkable
class IFileStore(Protocol):
    def open_sink(
        self,
        entity_type: type[object],
        path: Path,
    ) -> EntitySink[object]: ...

    def open_source(
        self,
        entity_type: type[object],
        path: Path,
    ) -> EntitySource[object]: ...


@runtime_checkable
class IWordNetService(Protocol):
    def initialize(self) -> bool: ...

    def resolve_synset_id(self, stem: str) -> str: ...

    def lookup_synset_ids(self, query: str) -> list[str]: ...

    def lookup_labels(self, synset_id: str) -> list[str]: ...

    def lookup_definition(self, synset_id: str) -> str | None: ...

    def create_metric(self, name: str) -> ISemanticMetric: ...

    def available_metric_names(self) -> tuple[str, ...]: ...


@runtime_checkable
class ISemanticMetric(Protocol):
    name: str

    def calculate(self, a: str, b: str) -> int | float | None: ...


@runtime_checkable
class IImageNetMetadataService(Protocol):
    def load_synsets(self) -> dict[str, str]: ...

    def load_imagenet_id_to_wnid(self) -> dict[str, str]: ...

    def load_validation_wnids(self) -> list[str]: ...

    def load_class_index_maps(
        self,
        class_map_path: Path,
        index_to_wnid_path: Path,
    ) -> object: ...

    def load_synset_label_map(self, path: Path) -> dict[int, str]: ...

    def resolve_synset_id_from_label(self, label: str) -> str | None: ...


@runtime_checkable
class ISynsetImageFolder(Protocol):
    root: Path
    classes: list[str]
    class_to_idx: dict[str, int]
    samples: list[tuple[str, int]]
    targets: list[int]

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[object, int]: ...


@runtime_checkable
class ISynsetImageFolderFactory(Protocol):
    def create(
        self,
        root: Path,
        *,
        transform: object | None = None,
    ) -> ISynsetImageFolder: ...


@runtime_checkable
class IVisualizationRenderer(Protocol):
    def save_distribution(
        self,
        *,
        plot_spec: object,
        output_path: Path,
    ) -> None: ...


@runtime_checkable
class ISemanticAnalysisService(Protocol):
    def analyze(
        self,
        *,
        predictions_path: Path,
        output_directory: Path,
        metric: ISemanticMetric,
        analysis_filter: object,
    ) -> object: ...

    def analyze_confusions(
        self,
        *,
        predictions_path: Path,
        output_path: Path,
        reverse: bool = False,
    ) -> int: ...


@runtime_checkable
class IEvaluationRuntime(Protocol):
    def create_model_spec(
        self,
        model_name: str,
        *,
        class_index_maps: object | None = None,
    ) -> object: ...

    def create_dataloader(self, dataset: Dataset) -> object: ...

    def resolve_device(self) -> torch.device: ...