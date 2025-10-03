#!/usr/bin/env python3
"""
extract_main_from_epub_strict.py

Stricter EPUB extractor that:
- Walks the spine in order
- Starts at first "Chapter N" (or --include-intro to start at Introduction/Prologue)
- Removes whole sections by heading (Dedication, Acknowledgments, References, Bibliography, Notes, Index, Glossary, Appendix, About the Author, etc.)
- Hard-stops at first back-matter heading (References/Bibliography/Appendix/Notes/etc.)
- Strips nav/header/footer/aside/roles
- Uses Readability for the core content block
- Optional: strip bracketed numeric citations like [12]

Usage:
  python extract_main_from_epub_strict.py book.epub -o main.txt
  python extract_main_from_epub_strict.py book.epub -o main.md --markdown --strip-citations
  python extract_main_from_epub_strict.py book.epub -o main.txt --include-intro
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning
from readability import Document

# Suppress XML parsing warning for EPUB content
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# === Heading patterns ===
# Start of main content
START_PATTERNS = [
    r"^\s*chapter\s+[0-9ivxlcdm]+\b",  # Chapter 1 / Chapter IV
]
INTRO_ALSO = [r"^\s*introduction\b", r"^\s*prologue\b"]

# Back matter (stop when we first see any of these)
END_PATTERNS = [
    r"^\s*references\b",
    r"^\s*bibliograph(y|ies)\b",
    r"^\s*works\s+cited\b",
    r"^\s*notes\b",
    r"^\s*endnotes\b",
    r"^\s*appendix(?:es)?\b",
    r"^\s*glossary\b",
    r"^\s*index\b",
    r"^\s*about\s+the\s+author\b",
    r"^\s*about\s+this\s+book\b",
]

# Sections to drop anywhere they appear (front/back matter that sometimes sits inside chapter docs)
SKIP_SECTION_PATTERNS = [
    r"^\s*dedication\b",
    r"^\s*acknowledg(e)?ments?\b",
    r"^\s*foreword\b",
    r"^\s*preface\b",
    r"^\s*epigraph\b",
    r"^\s*table\s*of\s*contents\b",
    r"^\s*toc\b",
    r"^\s*contributors?\b",
    r"^\s*colophon\b",
    r"^\s*part\s+[0-9ivxlcdm]+\b",  # Part I, Part 1
] + END_PATTERNS  # end headings are also skip headings when embedded

HEAD_SELECTORS = "h1,h2,h3,h4,h5,h6,[role='heading']"

JUNK_SELECTORS = [
    "header",
    "footer",
    "nav",
    "aside",
    "[role='banner']",
    "[role='navigation']",
    "[role='contentinfo']",
    "[role='doc-toc']",
    "[role='doc-index']",
    "[role='doc-bibliography']",
    "[epub\\:type~='frontmatter']",
    "[epub\\:type~='backmatter']",
    "[epub\\:type~='toc']",
    "[epub\\:type~='index']",
    "[epub\\:type~='glossary']",
    ".header",
    ".footer",
    ".nav",
    ".toc",
    ".table-of-contents",
    ".index",
    ".glossary",
    ".copyright",
    ".footnotes",
    "[role='doc-endnotes']",
]

START_RE = re.compile("|".join(START_PATTERNS), re.I)
INTRO_RE = re.compile("|".join(INTRO_ALSO), re.I)
END_RE = re.compile("|".join(END_PATTERNS), re.I)
SKIP_RE = re.compile("|".join(SKIP_SECTION_PATTERNS), re.I)

CITE_BRACKET_RE = re.compile(
    r"\s?\[(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\]"
)  # [12], [1-3], [2, 5]


def simplify_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def first_heading_match(soup: BeautifulSoup, pattern: re.Pattern):
    for h in soup.select(HEAD_SELECTORS):
        if pattern.search(simplify_text(h.get_text())):
            return h
    return None


def any_heading_matches(soup: BeautifulSoup, pattern: re.Pattern) -> bool:
    return first_heading_match(soup, pattern) is not None


def drop_region_from_heading_to_next(soup: BeautifulSoup, heading) -> None:
    """
    Remove heading and all following siblings until the next heading of same or higher level.
    """
    if heading is None:
        return
    # determine level (h1..h6 => 1..6; default to 6 if unknown)
    lvl = 6
    if heading.name and heading.name.lower().startswith("h"):
        try:
            lvl = int(heading.name[1])
        except Exception:
            lvl = 6

    # collect nodes to remove
    to_remove = [heading]
    sib = heading.next_sibling
    while sib:
        nxt = sib.next_sibling
        if isinstance(sib, NavigableString):
            to_remove.append(sib)
        else:
            # another heading?
            if getattr(sib, "name", "") and re.match(r"h[1-6]$", sib.name or "", re.I):
                other_lvl = int(sib.name[1])
                if other_lvl <= lvl:
                    break
            to_remove.append(sib)
        sib = nxt
    for n in to_remove:
        try:
            n.extract()
        except Exception:
            pass


def drop_sections_by_heading(soup: BeautifulSoup, regex: re.Pattern) -> None:
    while True:
        h = first_heading_match(soup, regex)
        if not h:
            return
        drop_region_from_heading_to_next(soup, h)


def apply_readability(html: str) -> BeautifulSoup:
    try:
        readable_html = Document(html).summary(html_partial=True)
        return BeautifulSoup(readable_html, "lxml")
    except Exception:
        return BeautifulSoup(html, "lxml")


def strip_junk(soup: BeautifulSoup) -> None:
    for sel in JUNK_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    for tag in soup(["script", "style", "svg", "canvas", "noscript"]):
        tag.decompose()


def soup_to_text(soup: BeautifulSoup, markdown: bool = False) -> str:
    if markdown:
        for level in range(1, 4):
            for h in soup.find_all(f"h{level}"):
                txt = h.get_text(" ", strip=True)
                h.replace_with(soup.new_string(f"{'#'*level} {txt}\n"))
        for p in soup.find_all("p"):
            p.replace_with(soup.new_string(p.get_text(" ", strip=True) + "\n\n"))
        for li in soup.find_all("li"):
            li.replace_with(soup.new_string(f"- {li.get_text(' ', strip=True)}\n"))
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("epub_path")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument(
        "--include-intro",
        action="store_true",
        help="Also allow 'Introduction'/'Prologue' to start main content",
    )
    ap.add_argument(
        "--strip-citations",
        action="store_true",
        help="Remove bracketed numeric citations like [12]",
    )
    args = ap.parse_args()

    in_path = Path(args.epub_path)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    try:
        book = epub.read_epub(str(in_path))
    except Exception as e:
        print(f"Failed to read EPUB: {e}", file=sys.stderr)
        sys.exit(2)

    # Build spine
    spine_ids = [sid for (sid, _) in getattr(book, "spine", [])]
    docs = []
    if spine_ids:
        for sid in spine_ids:
            it = book.get_item_with_id(sid)
            if it is not None and it.get_type() == ITEM_DOCUMENT:
                docs.append(it)
    else:
        docs = list(book.get_items_of_type(ITEM_DOCUMENT))

    outputs = []
    started = False
    finished = False

    for it in docs:
        if finished:
            break

        raw = it.get_content()
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8", errors="ignore")
            except Exception:
                raw = raw.decode("latin-1", errors="ignore")

        # Work on the original soup for heading-aware cuts
        orig = BeautifulSoup(raw, "lxml")

        # If we haven't started yet, look for the first start heading
        if not started:
            # Drop obvious skip sections (Dedication/ToC/Part/etc.) that might precede the real start
            drop_sections_by_heading(orig, SKIP_RE)

            # Now find the first start heading
            start_h = first_heading_match(orig, START_RE)
            if not start_h and args.include_intro:
                start_h = first_heading_match(orig, INTRO_RE)

            if start_h:
                # Remove everything before start_h
                # Strategy: wrap the document in a synthetic container, then delete prior siblings
                parent = start_h.parent
                # remove all prior siblings in document order until we hit start_h
                node = parent.contents[0] if parent and parent.contents else None
                # If heading isn't a direct child, just drop everything before its position in the whole doc
                pos = None
                for i, el in enumerate(
                    orig.body.descendants if orig.body else orig.descendants
                ):
                    if el is start_h:
                        pos = i
                        break
                if pos is not None:
                    # crude but effective: slice descendants isn't easy; instead, delete elements before heading top-level
                    for el in list(orig.body.children if orig.body else []):
                        # stop once we see a tree that contains start_h
                        if hasattr(el, "descendants"):
                            # Check if start_h is in this element's descendants
                            if any(desc is start_h for desc in el.descendants):
                                break
                        if el is start_h:
                            break
                        try:
                            el.extract()
                        except Exception:
                            pass
                started = True
            else:
                # still not started; skip this file
                continue

        # If we got here, we're in main content
        # Also chop back-matter sections that might appear inside this file
        drop_sections_by_heading(orig, SKIP_RE)

        # If an end heading exists, cut from it to the end and mark finished
        end_h = first_heading_match(orig, END_RE)
        if end_h:
            drop_region_from_heading_to_next(orig, end_h)
            finished = True  # stop after this doc

        # Run readability on what remains, then strip junk
        soup = apply_readability(str(orig))
        strip_junk(soup)

        # Prefer main/article if present
        main_like = soup.select_one("[role='main'], main, article")
        if main_like:
            soup = BeautifulSoup(str(main_like), "lxml")

        text = soup_to_text(soup, markdown=args.markdown)

        if args.strip_citations:
            text = CITE_BRACKET_RE.sub("", text)

        if len(text) >= args.min_chars:
            outputs.append(text)

    combined = "\n\n".join(outputs).strip()

    if not combined:
        print(
            "No main content extracted. Try --include-intro or lower --min-chars.",
            file=sys.stderr,
        )
        sys.exit(3)

    out_path = Path(args.output)
    out_path.write_text(combined, encoding="utf-8")
    print(f"Wrote {out_path} ({len(combined):,} chars)")


if __name__ == "__main__":
    main()
