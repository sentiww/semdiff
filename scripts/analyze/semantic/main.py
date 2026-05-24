from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Literal
from typing import Callable
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset

logger = logging.getLogger(__file__)


def _parse_synset(synset_id: str) -> Synset:
    pos = synset_id[0]
    offset = int(synset_id[1:])
    return wordnet.synset_from_pos_and_offset(pos, offset)  # type: ignore


def _load_synset_mapping(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as mapping_file:
        data = json.load(mapping_file)
    return {str(k): str(v) for k, v in data.items()}


def _build_reverse_mapping(mapping: dict[str, str]) -> dict[str, str]:
    reverse: dict[str, str] = {}
    for synset_id, label in mapping.items():
        reverse.setdefault(label, synset_id)
    return reverse


def _resolve_synset_id(
    raw_value: str, mapping: dict[str, str], reverse_mapping: dict[str, str]
) -> str:
    if raw_value in mapping:
        return raw_value
    if raw_value in reverse_mapping:
        return reverse_mapping[raw_value]
    raise KeyError(f"Could not resolve synset for value: {raw_value}")


def run(
    input: Path,
    output: Path,
    metric_name: str,
    target_synset_mapping: Path,
    predicted_synset_mapping: Path,
) -> None:
    output.mkdir(exist_ok=True)

    metric = get_metric(metric_name)
    target_mapping = _load_synset_mapping(target_synset_mapping)
    predicted_mapping = _load_synset_mapping(predicted_synset_mapping)
    target_reverse_mapping = _build_reverse_mapping(target_mapping)
    predicted_reverse_mapping = _build_reverse_mapping(predicted_mapping)

    with (
        input.open("r", encoding="utf-8") as input_file,
        output.open("w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            record = json.loads(line)

            target = _resolve_synset_id(
                str(record["target"]),
                target_mapping,
                target_reverse_mapping,
            )
            predicted = _resolve_synset_id(
                str(record["predicted"]),
                predicted_mapping,
                predicted_reverse_mapping,
            )

            target_synset = _parse_synset(target)
            predicted_synset = _parse_synset(predicted)

            metric_value = metric(target_synset, predicted_synset)

            metric_record = {
                "target": target,
                "predicted": predicted,
                "metric": metric_value,
            }

            output_file.write(json.dumps(metric_record) + "\n")


def get_metric(metric_name: str) -> Callable[..., int | float | None]:
    match metric_name:
        case "path_distance":
            return path_distance
        case "path_similarity":
            return path_similarity
        case "wup_similarity":
            return wup_similarity
        case _:
            raise Exception()


def path_distance(a: Synset, b: Synset) -> float | Literal[1] | None:
    return a.shortest_path_distance(b, simulate_root=True)


def path_similarity(a: Synset, b: Synset) -> float | Literal[1] | None:
    return a.path_similarity(b, simulate_root=True)


def wup_similarity(a: Synset, b: Synset) -> float | Literal[1] | None:
    return a.wup_similarity(b, simulate_root=True)
