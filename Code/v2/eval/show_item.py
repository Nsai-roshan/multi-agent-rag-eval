"""Display a golden-set item and its corpus evidence for human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = ROOT / "Code" / "v2" / "eval" / "golden.jsonl"
CORPUS_PATH = ROOT / "Data" / "CorpusData" / "natmed_documents.json"


def load_items() -> list[dict]:
    with GOLDEN_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_chunks() -> dict[str, dict]:
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        documents = json.load(handle)
    chunks: dict[str, dict] = {}
    for document in documents:
        document_id = str(document.get("id", ""))
        if document_id.rsplit("-", 1)[-1].isdigit():
            base, index = document_id.rsplit("-", 1)
            chunks[f"{base}:{index}"] = document
    return chunks


def show_item(item_id: str) -> int:
    items = {item["id"]: item for item in load_items()}
    item = items.get(item_id)
    if item is None:
        print(f"Unknown item id: {item_id}")
        return 2

    print(f"id: {item['id']}")
    print(f"question: {item['question']}")
    print(f"complexity: {item.get('complexity')}")
    print(f"tags: {', '.join(item.get('tags', []))}")
    print(f"expected_answer: {item.get('expected_answer', '')}")
    print(f"needs_human_rewrite: {item.get('needs_human_rewrite', False)}")
    if item.get("rewrite_note"):
        print(f"rewrite_note: {item['rewrite_note']}")

    chunks = load_chunks()
    doc_ids = item.get("supporting_doc_ids", [])
    if not doc_ids:
        print("supporting_doc_ids: (none)")
    for doc_id in doc_ids:
        print(f"\n=== {doc_id} ===")
        document = chunks.get(doc_id)
        if document is None:
            print(f"WARNING: could not resolve chunk ID: {doc_id}")
            continue
        print("FULL CHUNK TEXT:")
        print(document.get("text", ""))
    return 0


def list_items() -> int:
    for item in load_items():
        status = "rewrite" if item.get("needs_human_rewrite", False) else "drafted"
        verified = "verified" if item.get("human_verified", False) else "unverified"
        print(f"{item['id']}\trewrite={status}\tverified={verified}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("item_id", nargs="?")
    group.add_argument("--list", action="store_true", dest="list_items")
    args = parser.parse_args()
    return list_items() if args.list_items else show_item(args.item_id)


if __name__ == "__main__":
    raise SystemExit(main())
