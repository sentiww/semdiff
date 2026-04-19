from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from cli.groups.analysis import register as register_analysis
from cli.groups.dataset import register as register_dataset
from cli.groups.evaluate import register as register_evaluate
from cli.groups.synset import register as register_synset
from cli.groups.visualization import register as register_visualization
from cli.groups.wordnet import register as register_wordnet

if TYPE_CHECKING:
    from bootstrap.container import (
        AnalysisContainer,
        AppConfig,
        DatasetContainer,
        EvaluationContainer,
        SynsetContainer,
        VisualizationContainer,
        WordNetContainer,
    )


@lru_cache(maxsize=1)
def build_app_config() -> AppConfig:
    from bootstrap.container import AppConfig

    project_root = Path(__file__).resolve().parents[2]
    return AppConfig(
        imagenet_class_map=(project_root / "mappings" / "imagenet-1k" / "class_map.json"),
        torchvision_index_to_wnid=(project_root / "mappings" / "torchvision_index_to_wnid.json"),
        evaluation_batch_size=32,
        evaluation_num_workers=0,
        evaluation_progress_log_every_batches=10,
    )


@lru_cache(maxsize=1)
def build_dataset_container() -> DatasetContainer:
    from bootstrap.container import DatasetContainer

    return DatasetContainer()


@lru_cache(maxsize=1)
def build_analysis_container() -> AnalysisContainer:
    from bootstrap.container import AnalysisContainer

    return AnalysisContainer()


@lru_cache(maxsize=1)
def build_evaluation_container() -> EvaluationContainer:
    from bootstrap.container import EvaluationContainer

    return EvaluationContainer(build_app_config())


@lru_cache(maxsize=1)
def build_wordnet_container() -> WordNetContainer:
    from bootstrap.container import WordNetContainer

    return WordNetContainer()


@lru_cache(maxsize=1)
def build_synset_container() -> SynsetContainer:
    from bootstrap.container import SynsetContainer

    return SynsetContainer()


@lru_cache(maxsize=1)
def build_visualization_container() -> VisualizationContainer:
    from bootstrap.container import VisualizationContainer

    return VisualizationContainer()


def build_app() -> typer.Typer:
    app = typer.Typer(
        help="Semantic difference CLI",
        no_args_is_help=True,
    )

    app.add_typer(
        register_dataset(build_dataset_container),
        name="dataset",
        help="Dataset management commands",
    )

    app.add_typer(
        register_analysis(build_analysis_container),
        name="analysis",
        help="Analysis management commands",
    )

    app.add_typer(
        register_evaluate(build_evaluation_container),
        name="evaluate",
        help="Model evaluation commands",
    )

    app.add_typer(
        register_wordnet(build_wordnet_container),
        name="wordnet",
        help="WordNet management commands",
    )

    app.add_typer(
        register_synset(build_synset_container),
        name="synset",
        help="Synset lookup commands",
    )

    app.add_typer(
        register_visualization(build_visualization_container),
        name="visualization",
        help="Visualization commands",
    )

    return app
