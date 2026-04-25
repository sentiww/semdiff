from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEMDIFF_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    imagenet_class_map: Path = Path("mappings/imagenet-1k/class_map.json")
    torchvision_index_to_wnid: Path = Path("mappings/torchvision_index_to_wnid.json")
    evaluation_batch_size: int = Field(default=32)
    evaluation_num_workers: int = Field(default=0)
    evaluation_progress_log_every_batches: int = Field(default=10)

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> AppSettings:
        env_file = self.model_config.get("env_file")
        if env_file and isinstance(env_file, str):
            env_path = Path(env_file)
            if not self.imagenet_class_map.is_absolute():
                self.imagenet_class_map = env_path.parent / self.imagenet_class_map
            if not self.torchvision_index_to_wnid.is_absolute():
                self.torchvision_index_to_wnid = env_path.parent / self.torchvision_index_to_wnid
        return self