from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from bootstrap.container import VisualizationContainer

visualization_app = typer.Typer(help="Visualization commands")


def register(container_factory: Callable[[], VisualizationContainer]) -> typer.Typer:
    @visualization_app.command("distribution")
    def distribution(
        input: list[Path] = typer.Option(
            ...,
            "--input",
            help="Repeat to compare multiple AnalysisResult JSON or annotated analysis JSONL files",
        ),
        output: Path = typer.Option(
            ...,
            "--output",
            help="Image path or output directory for the matplotlib visualization",
        ),
        mode: str = typer.Option(
            "overlay",
            "--mode",
            help="Comparison mode: overlay histograms or plot series side-by-side",
        ),
        series: str | None = typer.Option(
            None,
            "--series",
            help="Series name to visualize; defaults to the first declared series",
        ),
        field: str | None = typer.Option(
            None,
            "--field",
            help="Field name for matrix series; defaults to the first numeric field",
        ),
        label: list[str] | None = typer.Option(
            None,
            "--label",
            help="Repeat to provide custom legend labels matching each --input value",
        ),
        title: str | None = typer.Option(
            None,
            "--title",
            help="Override the plot title",
        ),
        x_label: str | None = typer.Option(
            None,
            "--x-label",
            help="Override the X axis label",
        ),
        y_label: str | None = typer.Option(
            None,
            "--y-label",
            help="Override the Y axis label",
        ),
    ) -> None:
        from features.visualization.command import DistributionInput

        container = container_factory()
        handler = container.distribution_handler()
        result = handler(
            DistributionInput(
                analysis_results=tuple(input),
                output=output,
                mode=mode,
                series=series,
                field=field,
                labels=tuple(label) if label is not None else None,
                title=title,
                x_label=x_label,
                y_label=y_label,
            )
        )
        typer.echo(
            f"Saved {result.analysis_type} distribution comparison for {result.series}\n"
            f"image: {result.output}"
        )

    return visualization_app
