"""Parser for plain text and Markdown files."""


def parse_text(content: bytes) -> str:
    """Decode text bytes, falling back to latin-1 if they are not valid UTF-8."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        # latin-1 maps every byte to a character, so this never fails
        text = content.decode("latin-1")
    return text.replace("\r\n", "\n")
