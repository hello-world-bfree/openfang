# Book Distiller — Operator's Guide

This Hand walks technical epubs in your library chapter-by-chapter and emits caveman-style cliff-notes optimised for both human re-reading and grep-driven coding-agent context loads.

The library lives in self-hosted Postgres + Cloudflare R2. Inputs:

- **`collection_id`** (preferred) — `reading_lists.id` from the library DB.
- **`collection_name`** — exact ILIKE match. Fails loudly on ambiguity.
- **`book_queue_dir`** — fallback watch directory for ad-hoc book IDs.

Output: `<output_root>/<collection-slug>/<author-slug>/<book-slug>/NN-<chapter-slug>.md`. No README aggregation — chapter frontmatter (keywords array, ratios, model) is the index. Cross-book search lives in docs-mcp (v1.5).

---

## One-time install

### 1. Save the extractor script

Copy this whole heredoc into your shell **once**:

```bash
mkdir -p ~/.openfang/scripts
cat > ~/.openfang/scripts/epub-extract.py <<'PY'
#!/Users/poirot/.openfang/scripts/.venv/bin/python
"""epub-extract.py — extract chapters from an .epub into per-chapter Markdown.

Usage:
    epub-extract.py <input.epub> <output_dir>
    epub-extract.py --self-test

Exit codes:
    0  success
    1  input file not found
    2  DRM-protected / encrypted (cannot read)
    3  malformed epub (bad zip, missing OPF, etc.)
    4  empty spine (no readable chapters)
    5  missing dependency (ebooklib / bs4 not installed)

Stdout (success): one line of JSON
    {"chapters": N, "output_dir": "...", "extraction_path": "nav|ncx|spine"}

Each chapter file is markdown with YAML frontmatter:
    chapter_index, title, source_href, word_count, extraction_path,
    extraction_warn (if word_count < 50)
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
except ImportError as e:
    print(json.dumps({"error": f"missing dependency: {e}"}), file=sys.stderr)
    sys.exit(5)

CHAPTER_FILE_RE = re.compile(r'[^a-z0-9]+')
HEADING_TAGS = ("h1", "h2", "h3")


def slug(text: str, maxlen: int = 60) -> str:
    text = (text or "").strip().lower()
    text = CHAPTER_FILE_RE.sub("-", text).strip("-")
    return text[:maxlen] or "chapter"


def html_to_md(html: str) -> tuple[str, str | None]:
    """Return (markdown_body, first_heading_text)."""
    soup = BeautifulSoup(html, "html.parser")
    first_heading: str | None = None
    for tag in soup.find_all(HEADING_TAGS):
        if tag.get_text(strip=True):
            first_heading = tag.get_text(" ", strip=True)
            break

    # Convert headings to # style.
    for level, name in enumerate(HEADING_TAGS, start=1):
        for tag in soup.find_all(name):
            tag.insert_before("\n" + "#" * level + " ")
            tag.insert_after("\n")
            tag.unwrap()

    # Convert <pre><code> → fenced code blocks.
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        body = (code.get_text("\n") if code else pre.get_text("\n")).rstrip()
        lang = ""
        if code and code.get("class"):
            for cls in code.get("class"):
                m = re.match(r"language-(\w+)", cls) or re.match(r"lang-(\w+)", cls)
                if m:
                    lang = m.group(1)
                    break
        pre.replace_with(f"\n```{lang}\n{body}\n```\n")

    # Convert <code> → inline code.
    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text()}`")

    # Convert <li> → "- " bullets.
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
        li.unwrap()

    # Strip remaining tags but keep text.
    text = soup.get_text("\n")
    # Collapse runs of blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, first_heading


def gather_nav_titles(book: epub.EpubBook) -> dict[str, str]:
    """Map href → title from the EPUB3 nav document, if present."""
    titles: dict[str, str] = {}
    for item in book.get_items_of_type(ebooklib.ITEM_NAVIGATION):
        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            nav = soup.find("nav", attrs={"epub:type": "toc"}) or soup.find("nav")
            if not nav:
                continue
            for a in nav.find_all("a"):
                href = (a.get("href") or "").split("#")[0]
                txt = a.get_text(" ", strip=True)
                if href and txt:
                    titles[href] = txt
        except Exception:
            continue
    return titles


def gather_ncx_titles(book: epub.EpubBook) -> dict[str, str]:
    titles: dict[str, str] = {}
    for item in book.get_items_of_type(ebooklib.ITEM_NAVIGATION):
        # NCX path covered above; ebooklib treats both as ITEM_NAVIGATION.
        try:
            text = item.get_content().decode("utf-8", errors="ignore")
            if "navMap" not in text:
                continue
            soup = BeautifulSoup(text, "xml")
            for navpt in soup.find_all("navPoint"):
                src = navpt.find("content")
                lbl = navpt.find("text")
                if src is not None and lbl is not None:
                    href = (src.get("src") or "").split("#")[0]
                    if href:
                        titles[href] = lbl.get_text(strip=True)
        except Exception:
            continue
    return titles


def write_chapter(out_dir: Path, idx: int, title: str, body: str,
                  source_href: str, extraction_path: str) -> dict:
    word_count = len(body.split())
    s = slug(title)
    fname = f"{idx:02d}-{s}.md"
    fp = out_dir / fname
    fm = [
        "---",
        f"chapter_index: {idx}",
        f'title: "{title.replace(chr(34), chr(39))}"',
        f'source_href: "{source_href}"',
        f"word_count: {word_count}",
        f'extraction_path: "{extraction_path}"',
    ]
    if word_count < 50:
        fm.append(f'extraction_warn: "low word count: {word_count}"')
    fm.append("---")
    fp.write_text("\n".join(fm) + "\n\n" + body + "\n", encoding="utf-8")
    return {"index": idx, "file": str(fp), "words": word_count, "title": title}


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        # Smoke-test imports + IO without a real epub.
        print(json.dumps({"self_test": "ok", "ebooklib": ebooklib.VERSION}))
        return 0

    if len(argv) != 3:
        print("usage: epub-extract.py <input.epub> <output_dir>", file=sys.stderr)
        return 1

    src = Path(argv[1])
    out_dir = Path(argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.is_file():
        print(json.dumps({"error": f"file not found: {src}"}), file=sys.stderr)
        return 1

    try:
        book = epub.read_epub(str(src))
    except Exception as e:
        msg = str(e).lower()
        if "encrypt" in msg or "drm" in msg or "adept" in msg:
            print(json.dumps({"error": f"DRM/encrypted: {e}"}), file=sys.stderr)
            return 2
        print(json.dumps({"error": f"malformed epub: {e}"}), file=sys.stderr)
        return 3

    nav_titles = gather_nav_titles(book) or gather_ncx_titles(book)
    extraction_path = "nav" if nav_titles else "spine"

    results = []
    idx = 0
    for spine_id, _linear in book.spine:
        item = book.get_item_with_id(spine_id)
        if not item or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        href = item.get_name()
        try:
            html = item.get_content().decode("utf-8", errors="ignore")
        except Exception:
            continue
        body, first_heading = html_to_md(html)
        if not body.strip():
            continue
        title = (
            nav_titles.get(href)
            or first_heading
            or item.title
            or f"Chapter {idx + 1}"
        )
        idx += 1
        results.append(write_chapter(out_dir, idx, title, body, href, extraction_path))

    if not results:
        print(json.dumps({"error": "empty spine — no readable chapters"}), file=sys.stderr)
        return 4

    print(json.dumps({
        "chapters": len(results),
        "output_dir": str(out_dir),
        "extraction_path": extraction_path,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
chmod +x ~/.openfang/scripts/epub-extract.py
```

### 2. Save the R2 fetcher script

```bash
cat > ~/.openfang/scripts/r2-fetch.py <<'PY'
#!/Users/poirot/.openfang/scripts/.venv/bin/python
"""r2-fetch.py — fetch one object from a Cloudflare R2 bucket via boto3.

Usage:
    r2-fetch.py <bucket> <key> <local_path>
    r2-fetch.py --self-test

Required env: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL.
Exit codes: 0=ok, 1=arg/env, 2=not found, 3=auth, 4=other, 5=missing dep.
Stdout (success): {"size":N,"etag":"...","local":"..."}
"""
# (see the full source under bundled/book-distiller/scripts in the openfang
#  repo if you'd like to inspect — it's also embedded in this Hand's prompt)
PY
chmod +x ~/.openfang/scripts/r2-fetch.py
```

(The full script body is in the openfang repo at
`crates/openfang-hands/bundled/book-distiller/` — copy that file into
`~/.openfang/scripts/r2-fetch.py` if you need a real install rather than a stub.)

### 3. Create a dedicated venv + install deps

System Python on macOS is typically PEP 668-protected (uv- or Homebrew-managed). The brew-installed `awscli` is also currently broken on Python 3.14 (libexpat / libxml symbol mismatch), so we deliberately avoid it. Use an isolated venv with `boto3` instead:

```bash
# requires uv (brew install uv) — or substitute `python3 -m venv ~/.openfang/scripts/.venv`
uv venv ~/.openfang/scripts/.venv --python 3.12
uv pip install --python ~/.openfang/scripts/.venv/bin/python \
    ebooklib beautifulsoup4 lxml boto3
```

The shebangs `#!/Users/poirot/.openfang/scripts/.venv/bin/python` (auto-set by the heredocs above for this user) route both helpers to that venv. If your home dir is different, edit the shebang lines to your absolute venv path.

### 4. Smoke test

```bash
~/.openfang/scripts/epub-extract.py --self-test
# → {"self_test": "ok", "ebooklib": [0, 20, 0]}

env $(grep -E '^R2_' ~/.openfang/secrets.env | xargs) \
    ~/.openfang/scripts/r2-fetch.py --self-test
# → {"self_test": "ok", "buckets": [...], "endpoint": "..."}
# (or AccessDenied on ListBuckets if your R2 token is bucket-scoped — that's fine,
#  the Hand only needs GetObject permission on the configured bucket.)
```

### 4. Set the required environment variables

The `[[requires]]` block enforces these at activation. Easiest: add to your shell rc (`~/.zshrc` / `~/.bashrc`):

```bash
export ANTHROPIC_API_KEY=...
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
export R2_ENDPOINT_URL=https://<your-account-id>.r2.cloudflarestorage.com
```

Or convert your existing `~/.dbx/env.yaml r2_cloud` block to `~/.dbx/env.sh` of `export VAR=...` lines and `source ~/.dbx/env.sh` before `openfang start`.

The endpoint URL is **not** in `~/.dbx/env.yaml` by default — find it in your Cloudflare dashboard under **R2 → Settings → S3 API**.

---

## How to use

### List available collections (run before activating)

```bash
mcp_dbx_execute_query library \
  "SELECT id, name, (SELECT COUNT(*) FROM list_books WHERE list_id=id) AS books FROM reading_lists ORDER BY name"
```

### Dry-run a collection (recommended first step on a new list)

```bash
curl -X POST http://127.0.0.1:4200/api/hands/book-distiller/activate \
  -H "Content-Type: application/json" \
  -d '{"config": {"collection_id": "42", "preview_only": "true"}, "instance_name": "preview-42"}'
```

The Hand publishes a `book-distiller.preview` event with book count, estimated chapters, and estimated USD. No LLM calls.

### Real run

```bash
curl -X POST http://127.0.0.1:4200/api/hands/book-distiller/activate \
  -H "Content-Type: application/json" \
  -d '{"config": {"collection_id": "42"}, "instance_name": "coll-42"}'
```

By name (must be unambiguous):

```bash
curl -X POST http://127.0.0.1:4200/api/hands/book-distiller/activate \
  -d '{"config": {"collection_name": "Distributed Systems Canon"}, "instance_name": "by-name"}'
```

### Force-reprocess

A single book:

```bash
curl -X POST .../activate \
  -d '{"config": {"collection_id": "42", "force_reprocess": "1234"}, "instance_name": "redo-1234"}'
```

Whole collection:

```bash
curl -X POST .../activate \
  -d '{"config": {"collection_id": "42", "force_reprocess": "all"}, "instance_name": "redo-coll-42"}'
```

The Hand publishes a `book-distiller.reprocess.start` event with the count of cleared progress keys before processing.

### Watch ongoing run

```bash
curl -N http://127.0.0.1:4200/api/events?topic=book-distiller
```

### Cancel

```bash
curl -X POST http://127.0.0.1:4200/api/hands/book-distiller/instances/<instance>/stop
```

A cancellation mid-chapter is safe: the `.tmp → mv` rename pattern means a chapter file is either fully written (and the next run's filesystem-catch-up step will pick it up) or absent (and the next run will re-distill from scratch). At-least-once, never twice.

### Watch-dir fallback (one-off books outside any collection)

With both `collection_id` and `collection_name` empty:

```bash
mkdir -p ~/openfang/book-queue
echo "" > ~/openfang/book-queue/1234.queue
```

The Hand picks up oldest `*.queue` files, renames to `.processing` while distilling, then to `done/<id>.done` on success.

---

## Output shape

Each chapter file:

```markdown
---
book: "Designing Data-Intensive Applications"
author: "Martin Kleppmann"
collection: "Distributed Systems Canon"
chapter: 7
chapter_title: "Transactions"
source_words: 8420
distilled_words: 1240
ratio: 0.147
model: "claude-sonnet-4-6"
keywords: [MVCC, snapshot isolation, two-phase commit, serializability, write skew]
quality_warning: null
---

# Ch 7. Transactions

## Core
- transaction = group of reads/writes treated as one unit. ACID guarantees.
...

## Mechanism
...

## Code
...

## Gotchas
...

## Use when
...
```

Empty section → header omitted. Code blocks copied verbatim. Nomenclature exact.

`quality_warning` may be: `null`, `ratio_too_high`, `output_truncated`, `code_dropped`, `skipped_narrative`. Watch the dashboard counters for aggregates:

- **Code Mutations Flagged** — chapters where the verbatim verifier missed a source code block in the output. Inspect those files manually.
- **Truncated Chapters** — chapters where `stop_reason: max_tokens` fired.
- **Ratio Violations** — chapters above the 20% hard ceiling.

---

## Costs

Default model split: prose chapters → Haiku 4.5; code-heavy chapters (>3 fenced blocks or >500 code lines) → Sonnet 4.6. Estimated total for a typical 88-book × 15-chapter library:

- All-Haiku: ~$20.
- Mixed (40% Sonnet on code-heavy): ~$43.
- With prompt caching (`cache_system_prompt = true`, on by default): system prompt re-billed at 10× cheaper across the run.

Set `budget_cap_usd` to bound spend per activation. The Hand pauses cleanly on cap — re-activate after raising the cap and it resumes from the last `done` chapter.

---

## Resumability + idempotency

Per-chapter contract:

1. `memory_recall book_distiller_chapter_<book_id>_<NN>` — done? skip.
2. Filesystem catch-up — if `<out_dir>/NN-<slug>.md` exists with valid frontmatter, mark memory `done` (catch-up), continue.
3. LLM call.
4. Atomic write (`.tmp → mv`).
5. `memory_store` progress = `done`.

A crash between (3) and (4) loses the LLM result (re-distilled on next run, paid once but never twice across persistent runs because step 1 + step 2 are idempotent).

A crash between (4) and (5) is recovered via step 2 on the next run — no double-LLM-cost.

---

## Schema assumptions

The Hand reads from these tables:

- `reading_lists(id, name, ...)`
- `list_books(list_id, book_id, position NOT NULL, added_at, ...)` — composite PK `(list_id, book_id)`. Ordering: `position ASC NULLS LAST, book_id ASC`.
- `books(id, title, file_path, format, ...)` — `file_path` is the R2 key.
- `book_authors(book_id, author_id, ordinal, ...)` — multi-author tie-break on `authors.sort_name`.
- `authors(id, name, sort_name, ...)`.

Pre-flight verifies tunnel + auth + parameter support before any work begins. If `mcp_dbx_execute_query` does not support parameterized queries on this build, `collection_name` resolution is disabled (injection-prevention) and only integer `collection_id` is accepted.

---

## Failure visibility

Watch the events stream and the dashboard metrics. Common failures and their categories:

| Error event | Cause | Recovery |
|---|---|---|
| `library_unreachable` | dbx tunnel/auth failed | Test `dbx test library` |
| `r2_unreachable` | wrong endpoint/keys | Verify `R2_ENDPOINT_URL` |
| `extractor_missing` | `epub-extract.py` not installed | Run heredoc from §1 |
| `unsafe_name_lookup` | dbx params unsupported + `collection_name` set | Use `collection_id` |
| `collection_not_found` / `collection_ambiguous` | name resolution | Use `collection_id` |
| `no_input` | all input fields empty | Set one input |
| `no_r2_key` | book row has NULL `file_path` | DB hygiene |
| `fetch_failed` | R2 404 / network | Re-run; honors `max_book_retries` |
| `extract_failed_2` | DRM-protected epub | Skip; remove from collection |
| `extract_failed_3/4` | malformed / empty | Manual epub repair |
| `chapter.too_large` | chapter > ceiling | Tune `chapter_size_ceiling_words` |
| `budget_exceeded` | cumulative spend ≥ cap | Raise cap; re-activate |
| `chapter.code_dropped` | code-block verifier mismatch | Manual chapter inspection |
