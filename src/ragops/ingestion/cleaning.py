import re
from html.parser import HTMLParser

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|$)", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


class _HTMLTextExtractor(HTMLParser):
    
    ignored_tags = {"script", "style", "nav", "footer", "header", "noscript"}

    
    block_tags = {
        "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
        "figcaption", "figure", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "main", "ol", "p", "pre", "section", "table", "td", "th",
        "tr", "ul",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts = []
        self._ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.ignored_tags:
            self._ignored_depth += 1
            return
        if self._ignored_depth == 0 and tag in self.block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.ignored_tags and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if self._ignored_depth == 0 and tag in self.block_tags:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._ignored_depth == 0:
            self._parts.append(data)

    def text(self):
        return "".join(self._parts)


def normalize_whitespace(text):
    """Collapse noisy whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_code_text(text):
    """Normalize newlines in code without changing indentation."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def strip_frontmatter(text):
    """Remove YAML frontmatter from the top of Markdown-like files."""
    return _FRONTMATTER_RE.sub("", text, count=1)


def strip_mdx_import_export_lines(text):
    """Remove top-level MDX import/export statements outside code fences."""
    cleaned_lines = []
    in_code_fence = False
    skipping_multiline_statement = False

    for line in text.splitlines():
        stripped = line.strip() 

        if _FENCE_RE.match(stripped):
            in_code_fence = not in_code_fence
            cleaned_lines.append(line)
            continue

        if in_code_fence:
            cleaned_lines.append(line)
            continue

        if skipping_multiline_statement:
            if stripped.endswith(";") or " from " in stripped:
                skipping_multiline_statement = False
            continue

        if stripped.startswith(("import ", "export ")):
            if not stripped.endswith(";") and " from " not in stripped:
                skipping_multiline_statement = True
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def strip_html_tags(text):
    """Convert basic HTML to plain text."""
    parser = _HTMLTextExtractor()
    parser.feed(text)
    parser.close()
    return parser.text()


def clean_text(text, extension):
    """Clean raw text according to its source file extension."""
    extension = extension.lower()
    if not extension.startswith("."):
        extension = f".{extension}"

    if extension in {".md", ".rst"}:
        return normalize_whitespace(strip_frontmatter(text))

    if extension == ".mdx":
        text = strip_frontmatter(text)
        text = strip_mdx_import_export_lines(text)
        return normalize_whitespace(text)

    if extension == ".html":
        return normalize_whitespace(strip_html_tags(text))

    if extension == ".py":
        return normalize_code_text(text)

    return normalize_whitespace(text)