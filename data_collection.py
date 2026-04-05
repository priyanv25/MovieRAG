"""
Movie RAG — Data Collection
============================
Builds a structured corpus for RAG evaluation over comedy films.

Sources
-------
Wikipedia   Metadata (director, cast, runtime …) + plot text via MediaWiki API.
subdl.com   Timestamped English dialogue cues via free REST API.
            Sign up at subdl.com → Profile → API Key (free, no credit card).

Input
-----
movies.csv  title, wiki_title, year — one row per film.  Swap this file to
            change the corpus without touching code.

Output
------
movies_corpus.json  — list[MovieRecord]

Usage
-----
    python notebook1_data_collection.py                        # wiki + subs
    python notebook1_data_collection.py --movies my_list.csv   # custom list
    python notebook1_data_collection.py --no-subs              # wiki only

Requirements
------------
    pip install wikipedia-api requests beautifulsoup4 tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Schema
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MovieRecord:
    """One film's worth of collected data."""
    title: str
    wiki_title: str
    year: int
    # metadata — multi-value
    director:     list[str] = field(default_factory=list)
    cast:         list[str] = field(default_factory=list)
    writer:       list[str] = field(default_factory=list)
    producer:     list[str] = field(default_factory=list)
    music:        list[str] = field(default_factory=list)
    # metadata — scalar
    runtime:      str = ""
    country:      str = ""
    language:     str = ""
    budget:       str = ""
    box_office:   str = ""
    release_date: str = ""
    # text
    plot_text:         str = ""
    full_article_text: str = ""
    # dialogue  [{index, start, end, text}, ...]
    subtitles: list[dict] = field(default_factory=list)


def load_movie_list(path: str) -> list[tuple[str, str, int]]:
    """Read (title, wiki_title, year) tuples from a CSV."""
    movies = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movies.append((row["title"], row["wiki_title"], int(row["year"])))
    log.info("Loaded %d movies from %s", len(movies), path)
    return movies


# ═════════════════════════════════════════════════════════════════════════════
# Wikipedia client
# ═════════════════════════════════════════════════════════════════════════════

_WIKI_API = "https://en.wikipedia.org/w/api.php"

_INFOBOX_MAP = {
    "directed by":  "director",
    "starring":     "cast",
    "running time": "runtime",
    "release date": "release_date",
    "produced by":  "producer",
    "written by":   "writer",
    "music by":     "music",
    "country":      "country",
    "language":     "language",
    "budget":       "budget",
    "box office":   "box_office",
}
_LIST_FIELDS = {"director", "cast", "writer", "producer", "music"}
_PLOT_HEADINGS = {"plot", "synopsis", "plot summary", "story", "premise"}


def _strip_refs(text: str) -> str:
    # Remove [1], [a], [note 1], [citation needed], etc.
    text = re.sub(r"\[(?:\d+|[a-z]|note\s*\d+|citation needed)[^\]]*\]", "", text)
    # Remove leftover empty brackets and surrounding commas
    text = re.sub(r",?\s*\[\s*,?\s*\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _td_to_list(td: Tag) -> list[str]:
    """Extract a list of names from an infobox cell.

    Handles three Wikipedia markup patterns:
    1.  <ul><li> plain-list items
    2.  <br>-separated names
    3.  Comma-separated (splits before uppercase to keep 'Robert De Niro' intact)
    """
    lis = td.find_all("li")
    if lis:
        return [_strip_refs(li.get_text(strip=True))
                for li in lis if li.get_text(strip=True)]

    for br in td.find_all("br"):
        br.replace_with("|||")
    raw = _strip_refs(td.get_text(strip=True))
    return [s.strip().strip(",")
            for s in re.split(r"\|\|\||,\s*(?=[A-Z])", raw) if s.strip()]


def _td_to_str(td: Tag) -> str:
    for br in td.find_all("br"):
        br.replace_with(", ")
    return _strip_refs(td.get_text(separator=", ", strip=True))


class WikipediaClient:
    """Fetch and parse Wikipedia film pages via the MediaWiki API."""

    def __init__(self, rate_limit: float = 0.5):
        self._s = requests.Session()
        self._s.headers["User-Agent"] = "MovieRAGCollector/1.0 (research)"
        self._delay = rate_limit

    def fetch(self, wiki_title: str) -> dict:
        html = self._get_html(wiki_title)
        if not html:
            return {}
        soup = BeautifulSoup(html, "html.parser")
        result = self._parse_infobox(soup)
        result["plot_text"] = self._extract_section(soup)
        result["full_article_text"] = self._all_paragraphs(soup)
        if not result["plot_text"]:
            result["plot_text"] = self._plot_fallback(wiki_title)
        time.sleep(self._delay)
        return result

    # -- private ----------------------------------------------------------

    def _get_html(self, title: str) -> Optional[str]:
        try:
            r = self._s.get(_WIKI_API, params={
                "action": "parse", "page": title,
                "prop": "text", "format": "json", "redirects": "",
            }, timeout=15)
            r.raise_for_status()
            return r.json().get("parse", {}).get("text", {}).get("*")
        except Exception:
            log.warning("Wiki fetch failed: %s", title)
            return None

    def _parse_infobox(self, soup: BeautifulSoup) -> dict:
        out: dict = {}
        infobox = soup.find("table", class_="infobox")
        if not infobox:
            return out
        for row in infobox.find_all("tr"):
            th, td = row.find("th"), row.find("td")
            if not th or not td:
                continue
            label = th.get_text(strip=True).lower()
            for wl, fn in _INFOBOX_MAP.items():
                if wl in label:
                    out[fn] = _td_to_list(td) if fn in _LIST_FIELDS else _td_to_str(td)
                    break
        return out

    def _extract_section(self, soup: BeautifulSoup) -> str:
        for h in soup.find_all(["h2", "h3"]):
            span = h.find("span", class_="mw-headline")
            if span and span.get_text(strip=True).lower() in _PLOT_HEADINGS:
                parts = []
                for sib in h.find_next_siblings():
                    if sib.name in ("h2", "h3"):
                        break
                    if sib.name == "p":
                        t = _strip_refs(sib.get_text(strip=True))
                        if t:
                            parts.append(t)
                return "\n\n".join(parts)
        return ""

    @staticmethod
    def _all_paragraphs(soup: BeautifulSoup) -> str:
        return "\n\n".join(
            _strip_refs(p.get_text(strip=True))
            for p in soup.find_all("p") if len(p.get_text(strip=True)) > 30
        )

    @staticmethod
    def _plot_fallback(title: str) -> str:
        try:
            import wikipediaapi
            w = wikipediaapi.Wikipedia("MovieRAGCollector/1.0 (research)", "en")
            page = w.page(title)
            if not page.exists():
                return ""
            for name in ("Plot", "Synopsis", "Plot summary", "Story", "Premise"):
                sec = page.section_by_title(name)
                if sec and sec.text.strip():
                    return sec.text.strip()
        except ImportError:
            log.debug("wikipedia-api not installed — skipping fallback")
        return ""


# ═════════════════════════════════════════════════════════════════════════════
# Subtitle client (subdl.com REST API — free API key, no scraping)
# Get your key: sign up at subdl.com → Profile → API Key
# ═════════════════════════════════════════════════════════════════════════════

_SRT_BLOCK = re.compile(r"\n\s*\n")
_SRT_INDEX = re.compile(r"^(\d+)$")
_SRT_TS    = re.compile(
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})"
)

_SUBDL_API = "https://api.subdl.com/api/v1/subtitles"
_SUBDL_DL  = "https://dl.subdl.com"


class SubtitleClient:
    """Download English subtitles via subdl.com REST API.

    Requires a free API key (env var SUBDL_API_KEY).
    Sign up at subdl.com → Profile → API Key.  No credit card.
    """

    def __init__(self, rate_limit: float = 0.5):
        self._key = os.environ.get("SUBDL_API_KEY")
        self._s = requests.Session()
        self._delay = rate_limit
        if not self._key:
            log.info("SUBDL_API_KEY not set — subtitles disabled. "
                     "Get a free key at subdl.com")

    @property
    def enabled(self) -> bool:
        return self._key is not None

    def fetch(self, title: str, year: int) -> list[dict]:
        if not self._key:
            return []
        try:
            # search
            r = self._s.get(_SUBDL_API, params={
                "api_key": self._key,
                "film_name": title,
                "type": "movie",
                "languages": "EN",
                "year": str(year),
                "subs_per_page": "1",
            }, timeout=15)
            r.raise_for_status()
            data = r.json()

            subs = data.get("subtitles", [])
            if not subs:
                log.debug("No subtitles found: %s (%d)", title, year)
                return []

            # download the zip
            dl_url = subs[0].get("url", "")
            if not dl_url:
                return []
            if dl_url.startswith("/"):
                dl_url = _SUBDL_DL + dl_url

            srt_text = self._download_zip(dl_url)
            if not srt_text:
                return []
            return self._parse_srt(srt_text)

        except Exception as e:
            log.warning("Subtitle fetch failed: %s (%d) — %s", title, year, e)
            return []
        finally:
            time.sleep(self._delay)

    def _download_zip(self, url: str) -> Optional[str]:
        """Download zip and extract the first .srt file."""
        import zipfile, io
        r = self._s.get(url, timeout=30)
        r.raise_for_status()
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".srt"):
                        return zf.read(name).decode("utf-8", errors="replace")
        except zipfile.BadZipFile:
            # might be raw srt
            return r.content.decode("utf-8", errors="replace")
        return None

    @staticmethod
    def _parse_srt(content: str) -> list[dict]:
        """Parse raw SRT into structured cues: [{index, start, end, text}]."""
        cues = []
        for block in _SRT_BLOCK.split(content.strip()):
            lines = block.strip().split("\n")
            if len(lines) < 2:
                continue
            idx = _SRT_INDEX.match(lines[0].strip())
            ts  = _SRT_TS.match(lines[1].strip())
            if not idx or not ts:
                continue
            text = " ".join(
                re.sub(r"<[^>]+>", "", ln).strip()
                for ln in lines[2:] if ln.strip()
            )
            text = re.sub(r"^-\s*", "", text).strip()
            text = re.sub(r"\s+-\s+", " ", text)
            if text:
                cues.append({
                    "index": int(idx.group(1)),
                    "start": ts.group(1), "end": ts.group(2),
                    "text": text,
                })
        return cues


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def collect(
    movies: list[tuple[str, str, int]],
    wiki: WikipediaClient,
    subs: SubtitleClient,
    fetch_subs: bool = True,
) -> list[dict]:
    corpus = []
    for title, wiki_title, year in tqdm(movies, desc="Collecting"):
        rec = MovieRecord(title=title, wiki_title=wiki_title, year=year)
        for k, v in wiki.fetch(wiki_title).items():
            if hasattr(rec, k):
                setattr(rec, k, v)
        if fetch_subs and subs.enabled:
            rec.subtitles = subs.fetch(title, year)
        corpus.append(asdict(rec))
    return corpus


def summarize(corpus: list[dict]) -> None:
    n = len(corpus)
    has = lambda key, min_len=1: sum(
        1 for m in corpus
        if (len(m[key]) >= min_len if isinstance(m[key], (str, list)) else bool(m[key]))
    )
    total_cues = sum(len(m["subtitles"]) for m in corpus)

    log.info("── Collection summary ──")
    log.info("  movies:     %d", n)
    log.info("  director:   %d", has("director"))
    log.info("  cast:       %d", has("cast"))
    log.info("  plot:       %d", has("plot_text", 50))
    log.info("  article:    %d", has("full_article_text", 100))
    log.info("  subtitles:  %d movies, %s cues", has("subtitles"), f"{total_cues:,}")

    sparse = [m["title"] for m in corpus if len(m.get("full_article_text", "")) < 100]
    if sparse:
        log.warning("  sparse data (check wiki_title): %s", sparse)


def save(corpus: list[dict], path: str = "movies_corpus.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    mb = os.path.getsize(path) / 1_048_576
    log.info("Saved %s (%.1f MB, %d records)", path, mb, len(corpus))


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Collect movie data for RAG evaluation")
    parser.add_argument("--movies", default="movies.csv", help="Path to movie list CSV")
    parser.add_argument("--output", default="movies_corpus.json", help="Output JSON path")
    parser.add_argument("--no-subs", action="store_true", help="Skip subtitle download")
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Seconds between wiki requests")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    movies = load_movie_list(args.movies)
    wiki = WikipediaClient(rate_limit=args.rate_limit)
    subs = SubtitleClient()

    corpus = collect(movies, wiki, subs, fetch_subs=not args.no_subs)
    summarize(corpus)
    save(corpus, args.output)

    if corpus:
        s = corpus[0]
        log.info("── Sample: %s (%d) ──", s["title"], s["year"])
        log.info("  director:   %s", s["director"])
        log.info("  cast:       %s", s["cast"][:5])
        log.info("  runtime:    %s", s["runtime"])
        log.info("  plot:       %s…", s["plot_text"][:120])
        log.info("  article:    %d chars", len(s["full_article_text"]))
        log.info("  subtitles:  %d cues", len(s["subtitles"]))


if __name__ == "__main__":
    main()