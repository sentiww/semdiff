from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semdiff.files.models import (
        AnalysisResult,
        EvaluationSummary,
        PredictionRecord,
        SemanticAnalysisRecord,
    )

OUTPUT_PATHS: dict[type, str] = {}


def register_output_path(
    entity_type: type,
    filename: str,
) -> None:
    OUTPUT_PATHS[entity_type] = filename


from semdiff.files.models import (
    AnalysisResult,
    EvaluationSummary,
    PredictionRecord,
    SemanticAnalysisRecord,
)

register_output_path(PredictionRecord, "predictions.jsonl")
register_output_path(EvaluationSummary, "summary.json")
register_output_path(SemanticAnalysisRecord, "annotated.jsonl")
register_output_path(AnalysisResult, "analysis.json")


@dataclass(frozen=True)
class OutputPath:
    directory: Path
    filename: str

    @property
    def path(self) -> Path:
        return self.directory / self.filename


class OutputPathResolver:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)

    def resolve(
        self,
        entity_type: type,
        filename: str | None = None,
        *,
        check_collision: bool = False,
        overwrite: bool = False,
    ) -> OutputPath:
        if filename is None:
            filename = self._get_default_filename(entity_type)
        output_path = OutputPath(directory=self._base_dir, filename=filename)
        if check_collision and output_path.path.exists() and not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_path.path}. "
                f"Use overwrite=True to overwrite."
            )
        return output_path

    def _get_default_filename(self, entity_type: type) -> str:
        if entity_type not in OUTPUT_PATHS:
            raise ValueError(
                f"No default output path registered for {entity_type.__name__}. "
                f"Provide explicit filename or register one."
            )
        return OUTPUT_PATHS[entity_type]

    def create_directory(self, entity_type: type, filename: str | None = None) -> Path:
        output_path = self.resolve(entity_type, filename)
        output_path.directory.mkdir(parents=True, exist_ok=True)
        return output_path.path
