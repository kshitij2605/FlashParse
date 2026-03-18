from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from ..config.constants import ImageCategory


@dataclass
class ClassificationResult:
    category: ImageCategory
    raw_response: str


@dataclass
class ImageInfo:
    page_idx: int
    region_idx: int
    bbox_2d: list[int]
    cropped: Image.Image
    label: str
    category: Optional[ImageCategory] = None
    caption: Optional[str] = None
    image_filename: Optional[str] = None


@dataclass
class PageResult:
    page_idx: int
    regions: list[dict]
    images: list[ImageInfo] = field(default_factory=list)


@dataclass
class PipelineResult:
    markdown: str
    pages_processed: int
    images_extracted: int
    tables_extracted: int
    image_infos: list[ImageInfo] = field(default_factory=list)
    processing_times: dict = field(default_factory=dict)
