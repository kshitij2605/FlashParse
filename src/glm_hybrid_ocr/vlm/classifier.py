import logging

from PIL import Image

from ..clients.vlm_client import AsyncVLMClient
from ..config.constants import DEFAULT_CATEGORY, ImageCategory
from ..config.prompts import CLASSIFICATION_PROMPT
from ..config.settings import VLMSettings
from ..models.types import ClassificationResult

logger = logging.getLogger(__name__)


class AsyncImageClassifier:
    def __init__(self, client: AsyncVLMClient, settings: VLMSettings):
        self.client = client
        self.settings = settings

    async def classify(self, image: Image.Image) -> ClassificationResult:
        try:
            response = await self.client.call(
                image=image,
                prompt=CLASSIFICATION_PROMPT,
                max_tokens=self.settings.max_tokens_classification,
                temperature=0.0,
            )
            category = self._parse_response(response)
            return ClassificationResult(category=category, raw_response=response)
        except Exception as e:
            logger.warning("Classification error: %s, defaulting to %s", e, DEFAULT_CATEGORY)
            return ClassificationResult(category=DEFAULT_CATEGORY, raw_response=str(e))

    def _parse_response(self, response: str) -> ImageCategory:
        answer = response.lower().strip()
        if "chart" in answer:
            return "chart"
        elif "scanned_text" in answer or "scanned" in answer:
            return "scanned_text"
        elif "figure" in answer:
            return "figure"
        elif "miscellaneous" in answer or "misc" in answer:
            return "miscellaneous"
        else:
            logger.warning("Unclear VLM response: '%s', defaulting to %s", answer, DEFAULT_CATEGORY)
            return DEFAULT_CATEGORY
