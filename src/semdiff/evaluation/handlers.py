from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from semdiff.core.handlers import CommandInput, CommandOutput, Handler, HandlerFactory
from semdiff.evaluation.service import ModelEvaluationService


@dataclass(frozen=True)
class EvaluateInput(CommandInput):
    model: str
    dataset: Path
    output: Path


@dataclass(frozen=True)
class EvaluateOutput(CommandOutput):
    model: str
    dataset: Path
    predictions_path: Path
    summary_path: Path
    num_samples: int
    device: str


class EvaluateHandler(Handler[EvaluateInput, EvaluateOutput]):
    def __init__(
        self,
        model_evaluation_service: ModelEvaluationService,
        class_map_path: Path,
        index_to_wnid_path: Path,
    ) -> None:
        self._model_evaluation_service = model_evaluation_service
        self._class_map_path = class_map_path
        self._index_to_wnid_path = index_to_wnid_path

    def __call__(self, cmd: EvaluateInput) -> EvaluateOutput:
        report = self._model_evaluation_service.evaluate(
            dataset_path=cmd.dataset,
            class_map_path=self._class_map_path,
            index_to_wnid_path=self._index_to_wnid_path,
            output_directory=cmd.output,
            model_name=cmd.model,
        )
        return EvaluateOutput(
            model=report.model_name,
            dataset=report.dataset_path,
            predictions_path=report.predictions_path,
            summary_path=report.summary_path,
            num_samples=report.num_samples,
            device=report.device,
        )


class EvaluationHandlers(HandlerFactory):
    def __init__(
        self,
        evaluation_service: ModelEvaluationService,
        class_map_path: Path,
        index_to_wnid_path: Path,
    ) -> None:
        self._evaluation_service = evaluation_service
        self._class_map_path = class_map_path
        self._index_to_wnid_path = index_to_wnid_path

    def create_evaluate(self) -> EvaluateHandler:
        return EvaluateHandler(
            model_evaluation_service=self._evaluation_service,
            class_map_path=self._class_map_path,
            index_to_wnid_path=self._index_to_wnid_path,
        )
