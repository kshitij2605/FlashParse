import re


def deduplicate_repeated_lines(text: str, threshold: int = 3) -> str:
    """Remove lines that repeat more than `threshold` times consecutively."""
    lines = text.split("\n")
    result = []
    count = 1
    for i, line in enumerate(lines):
        if i > 0 and line == lines[i - 1]:
            count += 1
        else:
            count = 1
        if count <= threshold:
            result.append(line)
    return "\n".join(result)


def clean_markdown_artifacts(text: str) -> str:
    """Remove common OCR markdown artifacts."""
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()
