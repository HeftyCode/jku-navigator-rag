import pytest
from src.ingestion.parser import JKUDocumentParser
from src.retrieval.hybrid_search import HybridRetriever
from src.generation.generator import JKUNavigatorGenerator, CitationItem, verify_citations


def test_symmetric_bilingual_stemming_and_paragraph_normalization():
    """Verifies that singular 'Hund' and plural 'Hunde' share a common stem and '§ 14' normalizes to '§14'."""
    tokens_singular = set(HybridRetriever._tokenize("Darf ich meinen Hund mitnehmen gem. § 14?"))
    tokens_plural = set(HybridRetriever._tokenize("Hunde sind laut §14 an der Leine zu führen."))

    assert "§14" in tokens_singular
    assert "§14" in tokens_plural
    assert "hund" in tokens_singular.intersection(tokens_plural)


def test_chunker_splits_oversized_lines_and_cleans_titles():
    """Verifies no single long line exceeds chunk_size and version numbers are stripped from doc titles."""
    long_line = "This is a sentence about JKU AI courses. " * 25
    parts = JKUDocumentParser._split_oversized_line(long_line, max_len=420)
    assert all(len(p) <= 420 for p in parts)
    assert JKUDocumentParser._clean_doc_title("2116_Hausordnung_V17_2026_10_08.pdf") == "Hausordnung"


def test_raw_decode_json_extractor_handles_trailing_braces():
    """Verifies _extract_json_object parses valid JSON even if followed by commentary containing braces."""
    messy_llm_output = '<think>thinking...</think>\n{"detected_language": "de", "bilingual_search_query": "Hund dog"} Note: {see rule 2}'
    extracted = JKUNavigatorGenerator._extract_json_object(messy_llm_output)
    assert '"detected_language": "de"' in extracted
    assert "Note:" not in extracted


def test_programmatic_citation_verifier_filters_hallucinated_pages():
    """Verifies that citations to pages not in retrieved_chunks are stripped and flagged."""
    retrieved = [
        {"source": "Bachelor_AI_Studienleitfaden.pdf", "doc_title": "Bachelor AI", "page": 15, "section": "COURSES"}
    ]
    citations = [
        CitationItem(source="Bachelor_AI_Studienleitfaden.pdf", page=15, section="COURSES", evidence_summary="Valid"),
        CitationItem(source="Bachelor_AI_Studienleitfaden.pdf", page=999, section="FAKE", evidence_summary="Invalid"),
    ]
    ans = "VL is a lecture [Bachelor_AI_Studienleitfaden.pdf, Page 15] and [Bachelor_AI_Studienleitfaden.pdf, Page 999]."
    audit = verify_citations(ans, citations, retrieved)

    assert len(audit["verified_citations"]) == 1
    assert audit["verified_citations"][0]["page"] == 15
    assert audit["hallucinated_citations_removed"] > 0
    assert audit["all_citations_valid"] is False