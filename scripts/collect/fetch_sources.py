"""Fetch open-access IBS medical content from curated sources.

Sources (ranked by license quality):
1. StatPearls IBS chapter (CC-BY 4.0)
2. NHS IBS pages (Open Government Licence v3.0)
3. MedlinePlus IBS (Public Domain)
4. PubMed Central open-access IBS reviews (varies, open access)

Usage:
    uv run python scripts/collect/fetch_sources.py --source all
    uv run python scripts/collect/fetch_sources.py --source statpearls
    uv run python scripts/collect/fetch_sources.py --source nhs
    uv run python scripts/collect/fetch_sources.py --source medlineplus
    uv run python scripts/collect/fetch_sources.py --source pubmed --max-papers 30
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import httpx
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

SOURCES_DIR = Path(__file__).resolve().parents[2] / "datasets" / "sources"
PUBMED_DIR = SOURCES_DIR / "pubmed"

NCBI_DELAY = 0.4  # ~2.5 req/s, under the 3/s limit for keyless access
NHS_DELAY = 1.0
DEFAULT_DELAY = 1.0

HEADERS = {
    "User-Agent": (
        "GutWise-DataCollector/0.1 "
        "(IBS health education research; contact: github.com/y0sif/GutWise)"
    ),
}

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubled each retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SourceFile(NamedTuple):
    filename: str
    content: str
    source: str
    url: str
    license: str


def _header_block(source: str, url: str, license_: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"---\nsource: {source}\nurl: {url}\nlicense: {license_}\nfetched: {now}\n---\n\n"


def _clean_text(text: str) -> str:
    """Collapse whitespace runs and strip leading/trailing blanks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _save(sf: SourceFile) -> Path:
    """Write a SourceFile to disk and return the path."""
    path = SOURCES_DIR / sf.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    full = _header_block(sf.source, sf.url, sf.license) + sf.content
    path.write_text(full, encoding="utf-8")
    logger.info("Saved %s (%d chars)", path, len(sf.content))
    return path


async def _fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    delay: float = DEFAULT_DELAY,
    params: dict | None = None,
) -> str:
    """Fetch a URL with retries and inter-request delay."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(url, params=params, follow_redirects=True)
            resp.raise_for_status()
            await asyncio.sleep(delay)
            return resp.text
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_exc = exc
            wait = RETRY_BACKOFF * (2**attempt)
            logger.warning(
                "Attempt %d/%d for %s failed: %s — retrying in %.1fs",
                attempt + 1,
                MAX_RETRIES,
                url,
                exc,
                wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} retries") from last_exc


# ---------------------------------------------------------------------------
# 1. StatPearls
# ---------------------------------------------------------------------------

STATPEARLS_URL = "https://www.ncbi.nlm.nih.gov/books/NBK534810/"


async def fetch_statpearls(client: httpx.AsyncClient) -> list[SourceFile]:
    """Fetch the StatPearls IBS chapter from NCBI Bookshelf."""
    logger.info("Fetching StatPearls IBS chapter...")
    html = await _fetch(client, STATPEARLS_URL, delay=NCBI_DELAY)
    soup = BeautifulSoup(html, "lxml")

    # The main article content lives in div.body-content or article
    content_div = (
        soup.find("div", class_="body-content")
        or soup.find("div", id="body-content")
        or soup.find("article")
        or soup.find("div", class_="content")
    )

    if content_div is None:
        # Fallback: grab the whole main area
        content_div = soup.find("main") or soup.body
        logger.warning("StatPearls: using fallback content container")

    assert isinstance(content_div, Tag)

    # Remove nav, sidebar, footer elements
    for tag in content_div.find_all(["nav", "footer", "script", "style", "aside"]):
        tag.decompose()

    # Remove "Go to:" links and other navigation helpers
    for a in content_div.find_all("a", class_="goto-link"):
        a.decompose()

    text = _clean_text(content_div.get_text(separator="\n"))
    return [
        SourceFile(
            filename="statpearls_ibs.txt",
            content=text,
            source="StatPearls — Irritable Bowel Syndrome",
            url=STATPEARLS_URL,
            license="CC-BY 4.0",
        )
    ]


# ---------------------------------------------------------------------------
# 2. NHS
# ---------------------------------------------------------------------------

NHS_BASE = "https://www.nhs.uk/conditions/irritable-bowel-syndrome-ibs/"
NHS_PAGES: list[tuple[str, str, str]] = [
    # (subpath, filename, label)
    ("", "nhs_ibs_overview.txt", "overview"),
    ("symptoms/", "nhs_ibs_symptoms.txt", "symptoms"),
    (
        "diet-lifestyle-and-medicines/",
        "nhs_ibs_diet_lifestyle_medicines.txt",
        "diet, lifestyle and medicines",
    ),
]


def _extract_nhs_content(html: str) -> str:
    """Extract article text from an NHS page."""
    soup = BeautifulSoup(html, "lxml")

    # NHS uses <main> with article content inside
    main = soup.find("main") or soup.find("article")
    if main is None:
        main = soup.body
        logger.warning("NHS: using <body> as fallback container")

    assert isinstance(main, Tag)

    # Remove breadcrumbs, review dates, navigation, feedback sections
    for sel in [
        "nav",
        "footer",
        "script",
        "style",
        ".nhsuk-breadcrumb",
        ".nhsuk-review-date",
        ".nhsuk-pagination",
        ".nhsuk-footer",
        ".nhsuk-header",
        ".nhsuk-care-card",
        "#nhsuk-feedback-banner",
        ".nhsuk-related-nav",
        ".beta-hub-related-links",
    ]:
        for tag in main.select(sel):
            tag.decompose()

    return _clean_text(main.get_text(separator="\n"))


async def fetch_nhs(client: httpx.AsyncClient) -> list[SourceFile]:
    """Fetch NHS IBS pages (overview + sub-pages)."""
    logger.info("Fetching NHS IBS pages...")
    results: list[SourceFile] = []

    for subpath, filename, label in NHS_PAGES:
        url = NHS_BASE + subpath
        try:
            html = await _fetch(client, url, delay=NHS_DELAY)
            text = _extract_nhs_content(html)
            results.append(
                SourceFile(
                    filename=filename,
                    content=text,
                    source=f"NHS — IBS {label}",
                    url=url,
                    license="Open Government Licence v3.0",
                )
            )
            logger.info("  ✓ NHS %s (%d chars)", label, len(text))
        except Exception:
            logger.exception("Failed to fetch NHS %s page", label)

    return results


# ---------------------------------------------------------------------------
# 3. MedlinePlus
# ---------------------------------------------------------------------------

MEDLINEPLUS_URL = "https://medlineplus.gov/irritablebowelsyndrome.html"


async def fetch_medlineplus(client: httpx.AsyncClient) -> list[SourceFile]:
    """Fetch MedlinePlus IBS page."""
    logger.info("Fetching MedlinePlus IBS page...")
    html = await _fetch(client, MEDLINEPLUS_URL)
    soup = BeautifulSoup(html, "lxml")

    # Main content area
    main = soup.find("article") or soup.find("div", id="topic-summary") or soup.find("main")
    if main is None:
        main = soup.body
        logger.warning("MedlinePlus: using <body> as fallback container")

    assert isinstance(main, Tag)

    for tag in main.find_all(["nav", "footer", "script", "style", "aside", "header"]):
        tag.decompose()

    text = _clean_text(main.get_text(separator="\n"))
    return [
        SourceFile(
            filename="medlineplus_ibs.txt",
            content=text,
            source="MedlinePlus — Irritable Bowel Syndrome",
            url=MEDLINEPLUS_URL,
            license="Public Domain (U.S. Government)",
        )
    ]


# ---------------------------------------------------------------------------
# 4. PubMed Central
# ---------------------------------------------------------------------------

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"

PUBMED_SEARCH_TERM = "irritable bowel syndrome[Title] AND review[pt]"


class PubMedArticle(NamedTuple):
    pmid: str
    title: str
    authors: str
    journal: str
    year: str
    abstract: str


def _parse_pubmed_articles(xml_text: str) -> list[PubMedArticle]:
    """Parse PubMed efetch XML and extract article metadata + abstracts."""
    root = ET.fromstring(xml_text)
    articles: list[PubMedArticle] = []

    for article_el in root.iter("PubmedArticle"):
        medline = article_el.find(".//MedlineCitation")
        if medline is None:
            continue

        # PMID
        pmid_el = medline.find("PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text

        # Title
        title_el = article_el.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled"
        if not title:
            title = "Untitled"

        # Authors
        authors: list[str] = []
        for author in article_el.findall(".//Author"):
            last = author.find("LastName")
            first = author.find("ForeName")
            if last is not None and last.text:
                name = last.text
                if first is not None and first.text:
                    name = f"{first.text} {name}"
                authors.append(name)
        authors_str = ", ".join(authors[:6])
        if len(authors) > 6:
            authors_str += " et al."

        # Journal
        journal_el = article_el.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None and journal_el.text else "Unknown"

        # Year
        year_el = article_el.find(".//PubDate/Year")
        if year_el is None or not year_el.text:
            medline_date = article_el.find(".//PubDate/MedlineDate")
            year = medline_date.text[:4] if medline_date is not None and medline_date.text else ""
        else:
            year = year_el.text

        # Abstract
        abstract_parts: list[str] = []
        for abs_text in article_el.findall(".//AbstractText"):
            label = abs_text.get("Label", "")
            text = "".join(abs_text.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = "\n\n".join(abstract_parts)

        if abstract:
            articles.append(PubMedArticle(pmid, title, authors_str, journal, year, abstract))

    return articles


def _pubmed_header(article: PubMedArticle, fetched: str) -> str:
    """Build a YAML front-matter header for a PubMed article file."""
    return (
        f"---\n"
        f"source: PubMed\n"
        f"url: https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/\n"
        f"license: PubMed Abstract (fair use)\n"
        f"fetched: {fetched}\n"
        f"pmid: {article.pmid}\n"
        f"title: {article.title}\n"
        f"journal: {article.journal}\n"
        f"year: {article.year}\n"
        f"---\n\n"
    )


async def fetch_pubmed(
    client: httpx.AsyncClient,
    max_papers: int = 25,
) -> list[SourceFile]:
    """Search PubMed for IBS review papers and fetch their abstracts."""
    logger.info("Searching PubMed for IBS review papers (max %d)...", max_papers)

    # Step 1: Search for PMIDs
    search_params = {
        "db": "pubmed",
        "term": PUBMED_SEARCH_TERM,
        "retmax": str(max_papers),
        "sort": "relevance",
        "retmode": "xml",
    }
    search_xml = await _fetch(client, ESEARCH_URL, delay=NCBI_DELAY, params=search_params)
    search_root = ET.fromstring(search_xml)

    pmids: list[str] = []
    id_list = search_root.find("IdList")
    if id_list is not None:
        for id_el in id_list.findall("Id"):
            if id_el.text:
                pmids.append(id_el.text)

    if not pmids:
        logger.warning("PubMed search returned no results")
        return []

    logger.info("Found %d PMIDs, fetching abstracts...", len(pmids))

    # Step 2: Fetch article details in batches of 10
    results: list[SourceFile] = []
    batch_size = 10
    fetched = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "abstract",
            "retmode": "xml",
        }
        xml_text = await _fetch(client, EFETCH_URL, delay=NCBI_DELAY, params=fetch_params)

        for article in _parse_pubmed_articles(xml_text):
            try:
                safe_title = re.sub(r"[^\w\s-]", "", article.title)[:60].strip().replace(" ", "_")
                filename = f"pubmed/pmid_{article.pmid}_{safe_title}.txt"

                # Use custom header with PubMed-specific metadata
                header = _pubmed_header(article, fetched)
                content = article.abstract

                # Write directly instead of using _save (custom header format)
                path = SOURCES_DIR / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(header + content, encoding="utf-8")
                logger.info("Saved %s (%d chars)", path, len(content))

                results.append(
                    SourceFile(
                        filename=filename,
                        content=content,
                        source=f"PubMed — {article.title[:80]}",
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/",
                        license="PubMed Abstract (fair use)",
                    )
                )
            except Exception:
                logger.exception("Failed to process PMID %s", article.pmid)

    logger.info("Fetched %d PubMed abstracts", len(results))
    return results


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

SOURCE_FETCHERS = {
    "statpearls": fetch_statpearls,
    "nhs": fetch_nhs,
    "medlineplus": fetch_medlineplus,
    "pubmed": fetch_pubmed,
}


async def run(sources: list[str], max_papers: int = 25) -> list[Path]:
    """Run the collection pipeline for the specified sources."""
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    PUBMED_DIR.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0) as client:
        for source in sources:
            fetcher = SOURCE_FETCHERS[source]
            try:
                if source == "pubmed":
                    files = await fetcher(client, max_papers=max_papers)
                else:
                    files = await fetcher(client)
                for sf in files:
                    saved.append(_save(sf))
            except Exception:
                logger.exception("Failed to fetch source: %s", source)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch open-access IBS medical content for GutWise training data.",
    )
    parser.add_argument(
        "--source",
        choices=["all", *SOURCE_FETCHERS],
        default="all",
        help="Which source(s) to fetch (default: all)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=25,
        help="Max PubMed papers to fetch (default: 25)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    sources = list(SOURCE_FETCHERS.keys()) if args.source == "all" else [args.source]
    saved = asyncio.run(run(sources, max_papers=args.max_papers))

    print(f"\nDone — saved {len(saved)} files to {SOURCES_DIR}")
    for p in saved:
        print(f"  {p.relative_to(SOURCES_DIR)}")


if __name__ == "__main__":
    main()
