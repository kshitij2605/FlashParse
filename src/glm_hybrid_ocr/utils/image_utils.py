import base64
import io

from PIL import Image


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def crop_region_from_page(
    page_image: Image.Image, bbox_2d: list[int]
) -> Image.Image:
    """Crop a region from a page image using normalized bbox (0-1000)."""
    w, h = page_image.size
    x1 = int(bbox_2d[0] * w / 1000)
    y1 = int(bbox_2d[1] * h / 1000)
    x2 = int(bbox_2d[2] * w / 1000)
    y2 = int(bbox_2d[3] * h / 1000)
    return page_image.crop((x1, y1, x2, y2))


def render_pdf_pages(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    """Render all PDF pages to PIL Images using pypdfium2."""
    from glmocr.utils.image_utils import pdf_to_images_pil

    return pdf_to_images_pil(pdf_path, dpi=dpi)
