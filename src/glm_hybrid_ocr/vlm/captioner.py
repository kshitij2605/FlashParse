import logging

from PIL import Image

from ..clients.vlm_client import AsyncVLMClient
from ..config.constants import ImageCategory
from ..config.prompts import (
    VLM_CHART_CAPTION_PROMPT,
    VLM_FIGURE_CAPTION_PROMPT,
    VLM_JAPANESE_SYSTEM_MESSAGE,
    VLM_MISC_CAPTION_PROMPT,
    VLM_SCANNED_TEXT_CAPTION_PROMPT,
)
from ..config.settings import VLMSettings

logger = logging.getLogger(__name__)


class AsyncCaptioner:
    def __init__(self, client: AsyncVLMClient, settings: VLMSettings):
        self.client = client
        self.settings = settings

    async def caption(
        self,
        image: Image.Image,
        category: ImageCategory,
        page_image: Image.Image | None = None,
        image_b64: str | None = None,
        page_image_b64: str | None = None,
    ) -> str:
        try:
            if category == "scanned_text":
                return await self.client.call(
                    image=image,
                    prompt=VLM_SCANNED_TEXT_CAPTION_PROMPT,
                    max_tokens=self.settings.max_tokens_caption,
                    system_message=VLM_JAPANESE_SYSTEM_MESSAGE,
                    image_b64=image_b64,
                )

            prompt = {
                "chart": VLM_CHART_CAPTION_PROMPT,
                "figure": VLM_FIGURE_CAPTION_PROMPT,
                "miscellaneous": VLM_MISC_CAPTION_PROMPT,
            }.get(category, VLM_MISC_CAPTION_PROMPT)

            images = [page_image, image] if page_image is not None else [image]
            b64_list = None
            if page_image_b64 and image_b64:
                b64_list = [page_image_b64, image_b64]
            elif image_b64:
                b64_list = [image_b64]

            return await self.client.call_multi_image(
                images=images,
                prompt=prompt,
                max_tokens=self.settings.max_tokens_caption,
                system_message=VLM_JAPANESE_SYSTEM_MESSAGE,
                images_b64=b64_list,
            )
        except Exception as e:
            logger.warning("Caption error for %s image: %s", category, e)
            return f"[Caption generation failed: {e}]"
