"""Lightweight local fallback for the notebook's pdfplumber usage.

This project only needs two pieces of the real pdfplumber API:
- `pdfplumber.open(path)` as a context manager
- `page.extract_words(extra_attrs=[...])`

We implement those on top of the already installed `pypdf` package so the
notebook can run without installing external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader


_WORD_RE = re.compile(r"\S+")


def _is_bold(fontname: str) -> bool:
    return "Bold" in fontname or fontname.endswith("-BoldMT")


@dataclass
class _Word:
    text: str
    fontname: str
    size: float

    def as_dict(self, extra_attrs: Iterable[str] | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {"text": self.text}
        if extra_attrs:
            for attr in extra_attrs:
                if attr == "fontname":
                    data["fontname"] = self.fontname
                elif attr == "size":
                    data["size"] = self.size
        return data


class _Page:
    def __init__(self, page: Any):
        self._page = page

    def extract_words(self, extra_attrs: Iterable[str] | None = None) -> list[dict[str, Any]]:
        words: list[_Word] = []

        def visitor_text(text, cm, tm, font_dict, font_size):
            if not text:
                return

            fontname = ""
            if font_dict:
                fontname = (
                    font_dict.get("/BaseFont")
                    or font_dict.get("/FontName")
                    or ""
                )

            size = round(float(font_size), 1)
            for match in _WORD_RE.finditer(text):
                words.append(
                    _Word(
                        text=match.group(0),
                        fontname=str(fontname),
                        size=size,
                    )
                )

        self._page.extract_text(visitor_text=visitor_text)
        return [word.as_dict(extra_attrs) for word in words]


class _PDF:
    def __init__(self, path: str | Path):
        self._reader = PdfReader(str(path))
        self.pages = [_Page(page) for page in self._reader.pages]

    def __enter__(self) -> "_PDF":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def open(path: str | Path) -> _PDF:
    return _PDF(path)
