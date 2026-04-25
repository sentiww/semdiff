from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Generic, TextIO, TypeVar, cast

from semdiff.core.codecs import EntityCodec
from semdiff.files.models import JsonObject

logger = logging.getLogger(__name__)

StoreEntity = TypeVar("StoreEntity")


class EntitySink(Generic[StoreEntity]):
    def __init__(
        self,
        file_handle: TextIO,
        codec: EntityCodec[StoreEntity],
    ) -> None:
        self._file_handle = file_handle
        self._codec = codec
        self._write_count = 0

    def __enter__(self) -> EntitySink[StoreEntity]:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, entity: StoreEntity) -> None:
        payload = self._codec.serialize(entity)
        if self._codec.format_name == "json":
            if self._write_count > 0:
                raise RuntimeError(
                    f"{self._codec.entity_type.__name__} is stored as a single JSON document"
                )
            self._file_handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=self._codec.ensure_ascii,
                    indent=self._codec.indent,
                )
                + "\n"
            )
        else:
            self._file_handle.write(
                json.dumps(payload, ensure_ascii=self._codec.ensure_ascii) + "\n"
            )
        self._write_count += 1

    def close(self) -> None:
        self._file_handle.close()


class EntitySource(Generic[StoreEntity]):
    def __init__(
        self,
        path: Path,
        codec: EntityCodec[StoreEntity],
    ) -> None:
        self._path = path
        self._codec = codec

    def __iter__(self) -> Iterator[StoreEntity]:
        if not self._path.exists():
            raise FileNotFoundError(f"Missing file: {self._path}")

        logger.info(
            "Reading %s entities from %s",
            self._codec.entity_type.__name__,
            self._path,
        )
        read_count = 0
        with self._path.open(encoding="utf-8") as input_file:
            try:
                if self._codec.format_name == "json":
                    payload = json.loads(input_file.read())
                    if not isinstance(payload, dict):
                        raise RuntimeError(f"Expected a JSON object in {self._path}")
                    read_count = 1
                    yield self._codec.deserialize(cast(JsonObject, payload))
                    return

                for line in input_file:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise RuntimeError(f"Expected JSON object records in {self._path}")
                    read_count += 1
                    yield self._codec.deserialize(cast(JsonObject, payload))
            finally:
                logger.info(
                    "Finished reading %s %s entities from %s",
                    read_count,
                    self._codec.entity_type.__name__,
                    self._path,
                )


class FileStore:
    def __init__(
        self,
        codecs: Mapping[type[object], EntityCodec[object]],
    ) -> None:
        self._codecs = dict(codecs)

    def open_sink(
        self,
        entity_type: type[StoreEntity],
        path: Path,
    ) -> EntitySink[StoreEntity]:
        codec = self._get_codec(entity_type)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Opening %s sink at %s", entity_type.__name__, path)
        return EntitySink(path.open("w", encoding="utf-8"), codec)

    def open_source(
        self,
        entity_type: type[StoreEntity],
        path: Path,
    ) -> EntitySource[StoreEntity]:
        logger.info("Opening %s source at %s", entity_type.__name__, path)
        return EntitySource(Path(path), self._get_codec(entity_type))

    def _get_codec(
        self,
        entity_type: type[StoreEntity],
    ) -> EntityCodec[StoreEntity]:
        try:
            codec = self._codecs[entity_type]
        except KeyError as exc:
            raise ValueError(
                f"No file codec registered for {entity_type.__name__}"
            ) from exc
        return cast(EntityCodec[StoreEntity], codec)
