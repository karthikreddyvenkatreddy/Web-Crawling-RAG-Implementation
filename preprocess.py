# preprocess.py
import re
from typing import Generator, Dict, Any, List

def clean_and_normalize_text(raw_text: str) -> str:
    """Strips layout tags and normalizes spacing fragments."""
    if not raw_text: 
        return ""
    text = re.sub(r'Link in Main Catalog|Apply Now', '', raw_text, flags=re.IGNORECASE)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return " \n ".join(lines)

def token_aware_chunk_splitter(text: str, source_label: str, max_words: int = 500, overlap: int = 50) -> Generator[Dict[str, Any], None, None]:
    """Generates source-tagged, metadata-safe dictionary blocks."""
    words = text.split()
    total_words = len(words)
    if total_words <= max_words:
        yield {"source": source_label, "chunk_index": 0, "text": " ".join(words)}
        return
    chunk_idx, start = 0, 0
    while start < total_words:
        end = start + max_words
        yield {"source": source_label, "chunk_index": chunk_idx, "text": " ".join(words[start:end])}
        chunk_idx += 1
        start += (max_words - overlap)

def sanitize_and_deduplicate_records(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates structural cards across chunk overlap thresholds."""
    cleaned_cards, seen_identifiers = [], set()
    for card in raw_records:
        name = card.get("card_name", "").strip()
        item_url = card.get("product_url", "").strip()
        if not name or "SKIP" in name or name in ["Credit Card", "Credit Card Offer"]: 
            continue
            
        normalized_name = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
        normalized_url = item_url.split('?')[0].lower()
        unique_id = f"{normalized_name}||{normalized_url}"
        
        if unique_id not in seen_identifiers:
            seen_identifiers.add(unique_id)
            card.pop("error", None)
            cleaned_cards.append({k: str(v).strip() for k, v in card.items()})
    return cleaned_cards