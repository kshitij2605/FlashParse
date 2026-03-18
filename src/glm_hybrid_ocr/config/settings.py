from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class VLMSettings(BaseSettings):
    endpoint: str = "http://iitgpu12.hon.otsuka-shokai.co.jp:8008/v1/chat/completions"
    api_key: str = "dummy"
    model: str = "Qwen3.5-35B-A3B"
    temperature: float = 0.0
    top_p: float = 0.1
    max_tokens_classification: int = 50
    max_tokens_caption: int = 2048
    max_concurrency: int = 20
    timeout: float = 120.0

    model_config = {"env_prefix": "VLM_"}


class GLMOCRPipelineSettings(BaseSettings):
    config_path: str = "glmocr_config.yaml"

    model_config = {"env_prefix": "GLMOCR_"}


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "API_"}


class Settings(BaseSettings):
    vlm: VLMSettings = Field(default_factory=VLMSettings)
    glmocr: GLMOCRPipelineSettings = Field(default_factory=GLMOCRPipelineSettings)
    api: APISettings = Field(default_factory=APISettings)

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            vlm=VLMSettings(),
            glmocr=GLMOCRPipelineSettings(),
            api=APISettings(),
        )
