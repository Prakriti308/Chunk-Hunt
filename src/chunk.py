# chunk.py

import re
import json
import pathlib

DOCS_DIR = pathlib.Path("data/docs/tutorial")
OUTPUT_FILE = pathlib.Path("data/chunks.json")
MIN_CHUNK_LENGTH = 100  # characters; skip fragments shorter than this


def chunk_file(path: pathlib.Path, base_dir: pathlib.Path):
    """Split one markdown file into chunks at '## ' headings."""
    text = path.read_text(encoding="utf-8")

    # Split right before every line that starts with "## "
    parts = re.split(r"\n(?=## )", text)

    # Path relative to data/docs/tutorial, used to build readable IDs
    rel_path = path.relative_to(base_dir).as_posix()  # e.g. "dependencies/index.md"
    source_id = rel_path.replace("/", "-").replace(".md", "")

    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if len(part) < MIN_CHUNK_LENGTH:
            continue
        chunks.append({
            "id": f"{source_id}-{i}",
            "source": rel_path,
            "text": part,
        })
    return chunks


def main():
    all_chunks = []
    md_files = list(DOCS_DIR.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    for path in md_files:
        file_chunks = chunk_file(path, DOCS_DIR)
        all_chunks.extend(file_chunks)

    OUTPUT_FILE.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Wrote {len(all_chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
