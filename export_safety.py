"""Allow only inert formatting and explicit safe links in HTML exports."""

import html
from html.parser import HTMLParser


class _SafeHTML(HTMLParser):
    tags = frozenset({
        "p", "br", "hr", "strong", "em", "b", "i", "s", "del", "blockquote",
        "pre", "code", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
        "table", "thead", "tbody", "tr", "th", "td", "a",
    })

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.tags:
            return
        attributes = ""
        if tag == "a":
            href = dict(attrs).get("href", "") or ""
            # Relative URLs, script/data protocols, and control characters are rejected.
            if (href.lower().startswith(("https://", "http://", "mailto:", "#"))
                    and not any(ord(c) < 32 or ord(c) == 127 for c in href)):
                attributes = f' href="{html.escape(href, quote=True)}" rel="noopener noreferrer"'
        self.parts.append(f"<{tag}{attributes}>")

    def handle_endtag(self, tag):
        if tag in self.tags and tag not in {"br", "hr"}:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        self.parts.append(html.escape(data))


def safe_markdown(text):
    import markdown
    # Raw HTML is displayed literally; Markdown-generated tags are then allowlisted.
    rendered = markdown.markdown(html.escape(str(text)), extensions=["tables", "fenced_code"])
    parser = _SafeHTML()
    parser.feed(rendered)
    parser.close()
    return "".join(parser.parts)
