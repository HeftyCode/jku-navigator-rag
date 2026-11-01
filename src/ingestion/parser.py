import re
import json
import hashlib
import pymupdf4llm
from pathlib import Path
from typing import List, Dict, Any


class JKUDocumentParser:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clean_doc_title(filename: str) -> str:
        """Converts '2116_Hausordnung_V17_2026_10_08.pdf' into a clean readable title without noise numbers."""
        stem = Path(filename).stem
        stem = re.sub(r"^\d+_", "", stem)
        stem = re.sub(r"_V\d+.*$", "", stem)
        return stem.replace("_", " ").strip()

    def parse_pdfs_to_markdown(self) -> List[Dict[str, Any]]:
        """Deterministically parses sorted PDFs in data/raw into page-level Markdown."""
        parsed_documents = []
        pdf_files = sorted(self.raw_dir.glob("*.pdf"), key=lambda p: p.name.lower())

        if not pdf_files:
            print("No PDF files found in data/raw/!")
            return []

        for pdf_path in pdf_files:
            print(f"Parsing PDF: {pdf_path.name}...")
            page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

            full_md = "\n\n---\n\n".join([p["text"] for p in page_chunks])
            output_md_file = self.processed_dir / f"{pdf_path.stem}.md"
            output_md_file.write_text(full_md, encoding="utf-8")

            parsed_documents.append({
                "source": pdf_path.name,
                "doc_title": self._clean_doc_title(pdf_path.name),
                "pages": page_chunks,
            })

        return parsed_documents

    @staticmethod
    def _split_oversized_line(line: str, max_len: int) -> List[str]:
        """Splits a long single-line paragraph along sentence boundaries so it never exceeds max_len."""
        if len(line) <= max_len:
            return [line]
        sentences = re.split(r"(?<=[.!?])\s+", line)
        parts, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_len:
                current = f"{current} {sent}".strip()
            else:
                if current:
                    parts.append(current)
                # If a single sentence is still huge, hard-slice by words
                while len(sent) > max_len:
                    parts.append(sent[:max_len])
                    sent = sent[max_len:]
                current = sent
        if current:
            parts.append(current)
        return parts

    def create_contextual_chunks(
        self,
        parsed_docs: List[Dict[str, Any]],
        chunk_size: int = 420,
        overlap_chars: int = 80,
    ) -> List[Dict[str, Any]]:
        """
        Structure-aware chunker tuned for 128-token embedding models (~420 chars per chunk):
        - Enforces strict character-budget overlap (preventing multi-line overlap cascades).
        - Repeats Markdown table headers across split table chunks.
        - Deduplicates chunks by content hash.
        """
        all_chunks = []
        seen_hashes = set()
        chunk_id = 0

        for doc in parsed_docs:
            source_name = doc["source"]
            doc_title = doc["doc_title"]
            current_section = "General Overview"

            for page_num, page in enumerate(doc["pages"], start=1):
                raw_lines = page["text"].split("\n")
                units: List[str] = []
                for rl in raw_lines:
                    cleaned_line = rl.strip()
                    if not cleaned_line:
                        continue
                    units.extend(self._split_oversized_line(cleaned_line, chunk_size))

                current_lines: List[str] = []
                current_len = 0
                active_table_header: List[str] = []

                for i, line in enumerate(units):
                    # Track section headings
                    if line.startswith("#"):
                        current_section = line.lstrip("#").strip()[:90]
                        active_table_header = []

                    # Detect Markdown table header (row followed by |---| separator)
                    if line.startswith("|") and i + 1 < len(units) and re.match(r"^\|[\s\-:|]+\|$", units[i + 1]):
                        active_table_header = [line, units[i + 1]]
                    elif not line.startswith("|"):
                        active_table_header = []

                    # Flush chunk BEFORE adding line if it would exceed chunk_size
                    if current_lines and (current_len + len(line) + 1 > chunk_size):
                        chunk_id = self._emit_chunk(
                            all_chunks, seen_hashes, chunk_id, source_name, doc_title, page_num, current_section, current_lines
                        )
                        # Carry over only lines that fit strictly within overlap_chars
                        overlap_buf: List[str] = []
                        overlap_len = 0
                        for prev_line in reversed(current_lines):
                            if overlap_len + len(prev_line) <= overlap_chars:
                                overlap_buf.insert(0, prev_line)
                                overlap_len += len(prev_line) + 1
                            else:
                                break

                        # If continuing inside a Markdown table, prepend the table header so columns stay labeled
                        if line.startswith("|") and active_table_header and overlap_buf[:2] != active_table_header:
                            current_lines = list(active_table_header) + overlap_buf
                        else:
                            current_lines = overlap_buf
                        current_len = sum(len(x) + 1 for x in current_lines)

                    current_lines.append(line)
                    current_len += len(line) + 1

                # Flush remaining non-duplicate content on the page
                if current_lines:
                    chunk_id = self._emit_chunk(
                        all_chunks, seen_hashes, chunk_id, source_name, doc_title, page_num, current_section, current_lines
                    )

        chunks_file = self.processed_dir / "chunks.json"
        chunks_file.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(all_chunks)} deduplicated, table-aware chunks to {chunks_file}")
        return all_chunks

    @staticmethod
    def _emit_chunk(
        all_chunks: List[Dict[str, Any]],
        seen_hashes: set,
        chunk_id: int,
        source_name: str,
        doc_title: str,
        page_num: int,
        section: str,
        lines: List[str],
    ) -> int:
        raw_body = "\n".join(lines).strip()
        if len(raw_body) < 35:
            return chunk_id

        # Deduplicate identical or pure-overlap chunks
        norm_hash = hashlib.sha1(f"{source_name}:{page_num}:{raw_body}".encode("utf-8")).hexdigest()[:16]
        if norm_hash in seen_hashes:
            return chunk_id
        seen_hashes.add(norm_hash)

        contextual_text = f"[{doc_title} | Page {page_num} | {section}]\n{raw_body}"
        all_chunks.append({
            "chunk_id": chunk_id,
            "chunk_hash": norm_hash,
            "source": source_name,
            "doc_title": doc_title,
            "page": page_num,
            "section": section,
            "text": raw_body,
            "contextual_text": contextual_text,
        })
        return chunk_id + 1


if __name__ == "__main__":
    parser = JKUDocumentParser()
    docs = parser.parse_pdfs_to_markdown()
    if docs:
        chunks = parser.create_contextual_chunks(docs)
        lengths = [len(c["text"]) for c in chunks]
        lengths.sort()
        median_len = lengths[len(lengths) // 2] if lengths else 0
        print(f"Chunk Stats -> Total: {len(chunks)} | Median chars: {median_len} | Max chars: {max(lengths) if lengths else 0}")