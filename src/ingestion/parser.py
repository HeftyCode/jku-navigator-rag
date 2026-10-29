import json
import pymupdf4llm
from pathlib import Path
from typing import List, Dict, Any

class JKUDocumentParser:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def parse_pdfs_to_markdown(self) -> List[Dict[str, Any]]:
        """Converts JKU PDFs in data/raw into clean Markdown files in data/processed."""
        parsed_documents = []
        pdf_files = list(self.raw_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found in data/raw/! Please drop at least one PDF there.")
            return []

        for pdf_path in pdf_files:
            print(f"Parsing PDF: {pdf_path.name}...")
            # Extract markdown with page-by-page metadata so we can cite exact page numbers
            page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
            
            # Save full markdown for visual inspection
            full_md = "\n\n---\n\n".join([p["text"] for p in page_chunks])
            output_md_file = self.processed_dir / f"{pdf_path.stem}.md"
            output_md_file.write_text(full_md, encoding="utf-8")
            
            parsed_documents.append({
                "source": pdf_path.name,
                "pages": page_chunks
            })
            
        return parsed_documents

    def create_contextual_chunks(
        self, parsed_docs: List[Dict[str, Any]], chunk_size: int = 600, overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Splits pages into chunks while injecting [Source | Page | Section] headers
        into every single chunk so course IDs and § paragraphs never lose context.
        """
        all_chunks = []
        chunk_id = 0

        for doc in parsed_docs:
            source_name = doc["source"]
            current_section = "General Overview"

            for page_num, page in enumerate(doc["pages"], start=1):
                lines = page["text"].split("\n")
                
                current_lines = []
                current_len = 0

                for line in lines:
                    # Detect markdown headings to track which section we are inside
                    if line.strip().startswith("#"):
                        current_section = line.strip("#").strip()

                    current_lines.append(line)
                    current_len += len(line)

                    if current_len >= chunk_size:
                        raw_text = "\n".join(current_lines).strip()
                        if raw_text:
                            all_chunks.append({
                                "chunk_id": chunk_id,
                                "source": source_name,
                                "page": page_num,
                                "section": current_section,
                                "text": f"\n{raw_text}"
                            })
                            chunk_id += 1
                        
                        # Keep last few lines as overlap so sentences aren't cut in half
                        current_lines = current_lines[-3:]
                        current_len = sum(len(l) for l in current_lines)

                # Save any leftover text on the page
                if current_lines:
                    raw_text = "\n".join(current_lines).strip()
                    if len(raw_text) > 30:  # Ignore tiny page numbers/footers
                        all_chunks.append({
                            "chunk_id": chunk_id,
                            "source": source_name,
                            "page": page_num,
                            "section": current_section,
                            "text": f"\n{raw_text}"
                        })
                        chunk_id += 1

        # Save all chunks to data/processed/chunks.json so our database loader can read them
        chunks_file = self.processed_dir / "chunks.json"
        chunks_file.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(all_chunks)} contextual chunks to {chunks_file}")
        return all_chunks

if __name__ == "__main__":
    parser = JKUDocumentParser()
    docs = parser.parse_pdfs_to_markdown()
    if docs:
        chunks = parser.create_contextual_chunks(docs)
        print("\nSample Chunk #0:")
        print(chunks[0]["text"][:300] + "...\n")