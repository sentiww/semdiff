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
    from bootstrap import (
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
    from bootstrap import AppConfig

    project_root = Path(__file__).resolve().parents[2]
    return AppConfig.default(project_root)


@lru_cache(maxsize=1)
def build_dataset_container() -> DatasetContainer:
    from bootstrap import DatasetContainer

    return DatasetContainer()


@lru_cache(maxsize=1)
def build_analysis_container() -> AnalysisContainer:
    from bootstrap import AnalysisContainer

    return AnalysisContainer()


@lru_cache(maxsize=1)
def build_evaluation_container() -> EvaluationContainer:
    from bootstrap import EvaluationContainer

    return EvaluationContainer(build_app_config())


@lru_cache(maxsize=1)
def build_wordnet_container() -> WordNetContainer:
    from bootstrap import WordNetContainer

    return WordNetContainer()


@lru_cache(maxsize=1)
def build_synset_container() -> SynsetContainer:
    from bootstrap import SynsetContainer

    return SynsetContainer()


@lru_cache(maxsize=1)
def build_visualization_container() -> VisualizationContainer:
    from bootstrap import VisualizationContainer

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
