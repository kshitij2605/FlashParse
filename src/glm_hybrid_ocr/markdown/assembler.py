"""Build final markdown from glmocr JSON output + VLM captions."""

from ..config.constants import ImageCategory
from ..models.types import ImageInfo


def format_caption(caption: str, category: ImageCategory) -> str:
    """Format caption based on image category."""
    if category == "scanned_text":
        return f"```\n{caption}\n```"
    elif category == "chart":
        return f"> **Chart Description:** {caption}"
    else:
        # figure, miscellaneous
        return f"*{caption}*"


def assemble_markdown(
    json_result: list[list[dict]],
    image_infos: list[ImageInfo],
) -> str:
    """Assemble final markdown from glmocr JSON result and image captions.

    Args:
        json_result: Parsed JSON from glmocr - list of pages, each page is a list of regions.
            Each region has: index, label, content, bbox_2d, native_label
        image_infos: List of ImageInfo with category and caption set.

    Returns:
        Final markdown string.
    """
    # Build lookup: (page_idx, region_idx) -> ImageInfo
    image_lookup: dict[tuple[int, int], ImageInfo] = {}
    for info in image_infos:
        image_lookup[(info.page_idx, info.region_idx)] = info

    page_markdowns = []
    table_count = 0

    for page_idx, page_regions in enumerate(json_result):
        page_parts = []

        for region in page_regions:
            label = region.get("label", "text")
            content = region.get("content")
            region_idx = region.get("index", 0)

            if label == "image":
                # Check if we have a caption for this image
                info = image_lookup.get((page_idx, region_idx))
                if info and info.image_filename:
                    page_parts.append(f"![](images/{info.image_filename})")
                    if info.caption:
                        if info.category == "miscellaneous":
                            page_parts.append(info.caption)
                        else:
                            page_parts.append(format_caption(info.caption, info.category))
                else:
                    # Fallback: placeholder
                    bbox = region.get("bbox_2d", [])
                    page_parts.append(f"![](page={page_idx},bbox={bbox})")

            elif label == "table":
                table_count += 1
                table_filename = f"{page_idx}_{region_idx}.jpg"
                page_parts.append(f"![](tables/{table_filename})")
                if content:
                    page_parts.append(content)

            elif content:
                page_parts.append(content)

        page_markdowns.append("\n\n".join(page_parts))

    return "\n\n---\n\n".join(page_markdowns)
