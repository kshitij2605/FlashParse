from typing import Literal

ImageCategory = Literal["chart", "figure", "scanned_text", "miscellaneous"]

IMAGE_CATEGORIES: list[ImageCategory] = ["chart", "figure", "scanned_text", "miscellaneous"]
DEFAULT_CATEGORY: ImageCategory = "miscellaneous"
