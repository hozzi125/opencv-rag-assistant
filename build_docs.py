import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

#-CONFIG-
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
CHUNKS_PATH = os.path.join(OUT_DIR,"chunks.jsonl")
MANIFEST_PATH = os.path.join(OUT_DIR,"manifest.json")

USER_AGENT = "rag-opencv-corpus-builde/1.0 (+https://example.local)"
TIMEOUT_S = 30
SLEEP_BETWEEN_REQ_S = 0.4

#-CHUNKING-
CHUNK_MAX_CHARS = 2200
CHUNK_OVERLAP_CHARS = 250

@dataclass
class Source:
    name: str
    url: str

def ensure_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

def safe_filename(url:str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^a-zA-Z0-9]+","_",url).strip("_")
    slug = slug[-60:] if len(slug) > 60 else slug
    return f"{slug}_{h}.html"

def load_sources(path: str = "sources.yaml") -> List[Source]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    sources = []
    for item in data.get("sources",[]):
        sources.append(Source(name=item["name"], url=item["url"]))
    return sources

def download_html(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url,headers=headers,timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.text

def strip_noise(soup: BeautifulSoup) -> None:
    # remove present noise
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return

def extract_title(soup: BeautifulSoup) -> str:
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    title = re.sub(r"\s+"," ", title).strip()
    return title

def pick_main_container(soup: BeautifulSoup):
    # content may sit under div with class "contents" or "content"
    main = soup.select_one("div.contents")
    if main:
        return main
    main = soup.select_one("div#content")
    if main:
        return main
    # fallback: body
    return soup.body if soup.body else soup

def normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r","\n")
    #collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def node_text(tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()

def extract_sections(main_container) -> List[Tuple[str, str]]:
    # gather elements in order
    headings = {"h1", "h2", "h3", "h4"}
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Introduction"
    current_buf: List[str] = []

    # iter over direct descendants and nested in reading order
    for el in main_container.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"], recursive=True):
        name = el.name.lower()

        if name in headings:
            if current_buf:
                sections.append((current_title, current_buf))
            current_title = node_text(el) or current_title
            current_buf = []
            continue
        
        if name == "pre":
          code = el.get_text("\n", strip = False)
          code = code.strip("\n")
          if code:
              current_buf.append("```")
              current_buf.append(code)
              current_buf.append("```")
          continue

        if name == "code":
            # avoid double-capturing code inside <pre><code>
            if el.parent and el.parent.name and el.parent.name.lower() == "pre":
                continue
            t = el.get_text(" ", strip=True)
            if t:
                current_buf.append(f"`{t}`")
            continue
        
        if name == "p":
            t = el.get_text(" ", strip=True)
            if t:
                current_buf.append(t)
            continue
        
        if name == "li":
            t = el.get_text(" ", strip=True)
            if t:
                current_buf.append(f"- {t}")
            continue
        
    if current_buf:
        sections.append((current_title, current_buf))

    # join buffers
    out: List[Tuple[str, str]] = []
    for title, buf in sections:
        text = "\n".join(buf)
        text = normalize_ws(text)
        if text:
            out.append((title, text))
    return out

def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    # simple char-based chunking but with split at paragraph bounds
    if len(text) <= max_chars:
        return [text]
    
    paragraphs = text.split("\n\n")
    chunks = []
    cur = ""

    for p in paragraphs:
        candidate = (cur + "\n\n" + p).strip() if cur else p
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
                cur = p
            else:
                # paragraph too large -> hard split
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i + max_chars])
                cur = ""
    if cur:
        chunks.append(cur)

    # overlap by characters
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        prev = ""
        for ch in chunks:
            if prev:
                prefix = prev[-overlap:]
                merged = (prefix + "\n" + ch).strip()
                overlapped.append(merged)
            else:
                overlapped.append(ch)
            prev = ch
        chunks = overlapped
    
    return [normalize_ws(c) for c in chunks if c.strip()]

def make_chunk_id(url: str, section_title: str, idx: int) -> str:
    base = f"{url}::{section_title}::{idx}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:20]

def build_corpus(sources: List[Source]) -> None:
    ensure_dirs()

    manifest = {"sources": [], "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # reset outputs
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)

    with open(CHUNKS_PATH, "a", encoding="utf-8") as out_f:
        for s in tqdm(sources, desc="Downloading + Processing"):
            raw_name = safe_filename(s.url)
            raw_path = os.path.join(RAW_DIR, raw_name)

            # download
            if not os.path.exists(raw_path):
                html = download_html(s.url)
                with open(raw_path, "w", encoding = "utf-8") as f:
                    f.write(html)
                time.sleep(SLEEP_BETWEEN_REQ_S)
            else:
                with open(raw_path, "r", encoding="utf-8") as f:
                    html = f.read() 
            
            soup = BeautifulSoup(html, "lxml")
            strip_noise(soup)
            title = extract_title(soup)
            main = pick_main_container(soup)

            sections = extract_sections(main)

            # write chunks
            written = 0
            for section_title, section_text in sections:
                chunks = chunk_text(section_text, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
                for i, ch in enumerate(chunks):
                    rec = {
                        "id": make_chunk_id(s.url, section_title, i),
                        "text": ch,
                        "metadata": {
                            "source_name": s.name,
                            "source_url": s.url,
                            "doc_title": title,
                            "section_title": section_title,
                            "raw_file": raw_name
                        }
                    }
                    out_f.write(json.dumps(rec,ensure_ascii=False) + "\n")
                    written += 1

            manifest["sources"].append(
                {
                    "name": s.name,
                    "url": s.url,
                    "raw_file": raw_name,
                    "doc_title": title,
                    "sections": len(sections),
                    "chunks_written": written
                }
            )

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"Done! Chunks saved to: {CHUNKS_PATH}")
    print(f"Manifest saved to: {MANIFEST_PATH}")


if __name__ == "__main__":
    sources = load_sources("sources.yaml")
    build_corpus(sources)
