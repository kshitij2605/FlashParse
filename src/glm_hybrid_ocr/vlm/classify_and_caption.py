import logging
import re

from PIL import Image

from ..clients.vlm_client import AsyncVLMClient
from ..config.constants import DEFAULT_CATEGORY, ImageCategory
from ..config.prompts import (
    CLASSIFY_AND_CAPTION_PROMPT,
    CLASSIFY_AND_CAPTION_SINGLE_IMAGE_PROMPT,
    VLM_JAPANESE_SYSTEM_MESSAGE,
)
from ..config.settings import VLMSettings

logger = logging.getLogger(__name__)

_CATEGORY_RE = re.compile(r"\[category\]\s*(\w+)\s*\[/category\]")
_CAPTION_RE = re.compile(r"\[caption\](.*?)\[/caption\]", re.DOTALL)


class AsyncClassifyAndCaption:
    def __init__(self, client: AsyncVLMClient, settings: VLMSettings):
        self.client = client
        self.settings = settings

    async def classify_only(
        self,
        image: Image.Image,
        image_b64: str | None = None,
    ) -> ImageCategory:
        """Fast classification only — single image, short response."""
        from ..config.prompts import CLASSIFICATION_PROMPT
        try:
            response = await self.client.call(
                image=image,
                prompt=CLASSIFICATION_PROMPT,
                max_tokens=self.settings.max_tokens_classification,
                system_message=None,
                image_b64=image_b64,
            )
            lower = response.strip().lower()
            if "chart" in lower:
                return "chart"
            elif "figure" in lower:
                return "figure"
            elif "scanned" in lower:
                return "scanned_text"
            else:
                return "miscellaneous"
        except Exception as e:
            logger.warning("Classify error: %s", e)
            return DEFAULT_CATEGORY

    async def classify_and_caption(
        self,
        image: Image.Image,
        page_image: Image.Image | None = None,
        image_b64: str | None = None,
        page_image_b64: str | None = None,
    ) -> tuple[ImageCategory, str]:
        try:
            if page_image is not None:
                images = [page_image, image]
                b64_list = None
                if page_image_b64 and image_b64:
                    b64_list = [page_image_b64, image_b64]
                elif image_b64:
                    b64_list = [image_b64]

                response = await self.client.call_multi_image(
                    images=images,
                    prompt=CLASSIFY_AND_CAPTION_PROMPT,
                    max_tokens=self.settings.max_tokens_caption,
                    system_message=VLM_JAPANESE_SYSTEM_MESSAGE,
                    images_b64=b64_list,
                )
            else:
                response = await self.client.call(
                    image=image,
                    prompt=CLASSIFY_AND_CAPTION_SINGLE_IMAGE_PROMPT,
                    max_tokens=self.settings.max_tokens_caption,
                    system_message=VLM_JAPANESE_SYSTEM_MESSAGE,
                    image_b64=image_b64,
                )

            category, caption = self._parse_response(response)
            return category, caption

        except Exception as e:
            logger.warning("Classify+caption error: %s", e)
            return DEFAULT_CATEGORY, f"[Caption generation failed: {e}]"

    def _parse_response(self, response: str) -> tuple[ImageCategory, str]:
        cat_match = _CATEGORY_RE.search(response)
        cap_match = _CAPTION_RE.search(response)

        if cat_match:
            raw_cat = cat_match.group(1).lower().strip()
            if "chart" in raw_cat:
                category = "chart"
            elif "scanned" in raw_cat:
                category = "scanned_text"
            elif "figure" in raw_cat:
                category = "figure"
            elif "misc" in raw_cat:
                category = "miscellaneous"
            else:
                logger.warning("Unknown category '%s', defaulting to %s", raw_cat, DEFAULT_CATEGORY)
                category = DEFAULT_CATEGORY
        else:
            # Fallback: try to detect category from the raw response text
            lower = response.lower()
            if "chart" in lower and ("グラフ" in response or "chart" in lower):
                category = "chart"
            elif "figure" in lower or "フロー" in response or "ダイアグラム" in response or "図" in response[:50]:
                category = "figure"
            elif "scanned" in lower or "テキスト" in response[:50]:
                category = "scanned_text"
            else:
                category = DEFAULT_CATEGORY
            logger.warning("No [category] tag found, inferred '%s'", category)

        if cap_match:
            caption = cap_match.group(1).strip()
        else:
            # Fallback: use everything after [/category] or the full response
            caption = response
            if "[/category]" in caption:
                caption = caption.split("[/category]", 1)[1].strip()
            # Strip any remaining tags
            caption = re.sub(r"\[/?(?:category|caption)\]", "", caption).strip()
            if not caption:
                caption = response.strip()

        return category, caption
