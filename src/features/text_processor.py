from typing import List
import re
import unicodedata

class TextProcessor:
    """Process text for content-based filtering, handling Vietnamese diacritics"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercase, keep Vietnamese diacritics, strip extra spaces"""
        if not text:
            return ""
        text = text.lower().strip()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def remove_diacritics(text: str) -> str:
        """Remove Vietnamese diacritics for ASCII-folded tokens"""
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(char for char in nfd if not unicodedata.combining(char))
    
    @staticmethod
    def build_document(row) -> str:
        """Build searchable document from book metadata"""
        parts = [
            str(row.get('title', '')),
            # str(row.get('description', '')),
            str(row.get('authors_text', '')),
            str(row.get('genres_text', '')),
            str(row.get('publisher', '')),
            str(row.get('publication_year', ''))
        ]
        
        text = ' '.join(p for p in parts if p and p != 'None')
        
        # Keep Vietnamese + add ASCII version for recall
        normalized = TextProcessor.normalize_text(text)
        ascii_folded = TextProcessor.remove_diacritics(normalized)
        
        return f"{normalized} {ascii_folded}"