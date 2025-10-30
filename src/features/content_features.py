# ==================== src/features/content_features.py ====================
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from src.features.text_processor import TextProcessor
from src.utils.logging_config import logger

class ContentBasedModel:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            strip_accents=None  # Keep Vietnamese diacritics
        )
        self.feature_matrix = None
        self.book_ids = None
        self.id_to_idx = {}
    
    def fit(self, books_df: pd.DataFrame):
        """Build TF-IDF features"""
        logger.info("Building content-based features...")
        
        # Build documents
        books_df = books_df.copy()
        books_df['document'] = books_df.apply(TextProcessor.build_document, axis=1)
        
        # Fit TF-IDF
        self.feature_matrix = self.vectorizer.fit_transform(books_df['document'])
        self.book_ids = books_df['book_id'].values
        self.id_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}
        
        logger.info(f"Content features: {self.feature_matrix.shape}")
    
    def get_similar(self, book_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get similar books via cosine similarity"""
        if book_id not in self.id_to_idx:
            return []
        
        idx = self.id_to_idx[book_id]
        query_vec = self.feature_matrix[idx]
        
        scores = cosine_similarity(query_vec, self.feature_matrix).flatten()
        
        # Exclude self
        scores[idx] = -1
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.book_ids[i], scores[i]) for i in top_indices if scores[i] > 0]
    
    def save(self, path: Path):
        """Save vectorizer and features"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'feature_matrix': self.feature_matrix,
                'book_ids': self.book_ids,
                'id_to_idx': self.id_to_idx
            }, f)
        logger.info(f"Saved content model to {path}")
    
    @classmethod
    def load(cls, path: Path):
        """Load saved model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.vectorizer = data['vectorizer']
        model.feature_matrix = data['feature_matrix']
        model.book_ids = data['book_ids']
        model.id_to_idx = data['id_to_idx']
        return model