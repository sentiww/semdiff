from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from config.datasets import DatasetSettings

if TYPE_CHECKING:
    from bootstrap import DatasetContainer

dataset_app = typer.Typer(help="Dataset operations")
init_app = typer.Typer(help="Initialize datasets")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _PROJECT_ROOT / "datasets"
_MAPPINGS_ROOT = _PROJECT_ROOT / "mappings"
_IMAGENET_1K_ROOT = _DATASETS_ROOT / "imagenet-1k"
_IMAGENET_O_CLASS_MAP = _MAPPINGS_ROOT / "imagenet-o" / "class_map.json"


def _resolve_path(value: Path | None, *, default: Path) -> Path:
    return value if value is not None else default


def _normalize_image_suffixes(image_suffixes: Sequence[str] | None) -> tuple[str, ...]:
    settings = DatasetSettings.DEFAULT
    suffixes = image_suffixes or settings.default_image_suffixes
    return tuple(suffix.lower() for suffix in suffixes)


def register(container_factory: Callable[[], DatasetContainer]) -> typer.Typer:
    @init_app.command("imagenet-o")
    def init_imagenet_o(
        input: str = typer.Option(
            "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar",
            "--input",
            help="Archive URL or local archive file",
        ),
        output: Path = typer.Option(..., "--output"),
        image_suffixes: list[str] | None = typer.Option(None, "--image-suffixes"),
        class_map_path: Path | None = typer.Option(None, "--class-map-path"),
    ) -> None:
        from features.datasets.handlers import DatasetHandlers, ImageNetOInitInput

        container = container_factory()
        handlers = DatasetHandlers(
            archive_fetcher=container.archive_fetcher,
            archive_extractor=container.archive_extractor,
            output_preparer=container.output_preparer,
            file_mover=container.file_mover,
            index_parser=container.index_parser,
            imagenet_metadata_service=container._imagenet_metadata_service,
        )
        handler = handlers.create_imagenet_o_init()
        handler(
            ImageNetOInitInput(
                archive_source=input,
                output_directory=output,
                image_suffixes=_normalize_image_suffixes(image_suffixes),
                class_map_path=_resolve_path(
                    class_map_path,
                    default=_IMAGENET_O_CLASS_MAP,
                ),
            )
        )

    @init_app.command("imagenet-1k")
    def init_imagenet_1k(
        input: str | None = typer.Option(
            None,
            "--input",
            help="Archive URL or local archive file",
        ),
        output: Path = typer.Option(..., "--output"),
        image_suffixes: list[str] | None = typer.Option(None, "--image-suffixes"),
        meta_path: Path | None = typer.Option(None, "--meta-filename"),
        ground_truth_path: Path | None = typer.Option(None, "--ground-truth-filename"),
    ) -> None:
        from features.datasets.handlers import DatasetHandlers, ImageNet1KInitInput

        container = container_factory()
        handlers = DatasetHandlers(
            archive_fetcher=container.archive_fetcher,
            archive_extractor=container.archive_extractor,
            output_preparer=container.output_preparer,
            file_mover=container.file_mover,
            index_parser=container.index_parser,
            imagenet_metadata_service=container._imagenet_metadata_service,
        )
        handler = handlers.create_imagenet_1k_init()
        handler(
            ImageNet1KInitInput(
                archive_source=input
                if input is not None
                else str(_IMAGENET_1K_ROOT / "ILSVRC2012_img_val.tar"),
                output_directory=output,
                image_suffixes=_normalize_image_suffixes(image_suffixes),
                meta_path=_resolve_path(
                    meta_path,
                    default=_IMAGENET_1K_ROOT / "meta.mat",
                ),
                ground_truth_path=_resolve_path(
                    ground_truth_path,
                    default=_IMAGENET_1K_ROOT / "ILSVRC2012_validation_ground_truth.txt",
                ),
            )
        )

    dataset_app.add_typer(init_app, name="init")
    return dataset_app
