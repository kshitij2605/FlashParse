from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLMSettings(BaseSettings):
    """Settings for the VLM (Vision Language Model) used for image captioning.

    Supports any OpenAI-compatible API endpoint (vLLM, SGLang, TGI, OpenAI, etc.).
    Configure via environment variables with VLM_ prefix or .env file.
    """

    endpoint: str = "http://localhost:8008/v1/chat/completions"
    api_key: str = ""
    model: str = ""
    temperature: float = 0.0
    top_p: float = 0.1
    max_tokens_classification: int = 50
    max_tokens_caption: int = 2048
    max_concurrency: int = 64
    timeout: float = 120.0

    model_config = SettingsConfigDict(env_prefix="VLM_", env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def validate_required_fields(self):
        if not self.model:
            raise ValueError(
                "VLM_MODEL is required. Set it via environment variable or .env file. "
                "Example: VLM_MODEL=Qwen/Qwen2.5-VL-72B-Instruct"
            )
        return self


class GLMOCRPipelineSettings(BaseSettings):
    config_path: str = "glmocr_config.yaml"

    model_config = SettingsConfigDict(env_prefix="GLMOCR_", env_file=".env", extra="ignore")


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")


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
