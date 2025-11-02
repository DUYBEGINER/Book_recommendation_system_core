# ==================== src/features/content_features.py ====================
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, util
from src.features.text_processor import TextProcessor
from src.utils.logging_config import logger
from underthesea import word_tokenize

def vi_tokenizer(text: str):
    """Vietnamese tokenizer using underthesea"""
    return word_tokenize(text, format="text").split()


class ContentBasedModel:
    def __init__(
        self,
        model_name: str = "keepitreal/vietnamese-sbert",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[torch.Tensor] = None
        self.feature_matrix: Optional[torch.Tensor] = None
        self.book_ids: Optional[np.ndarray] = None
        self.id_to_idx: Dict[int, int] = {}

    def fit(self, books_df: pd.DataFrame):
        """Build SBERT features"""
        logger.info("Building content-based embeddings...")
        self._ensure_model_loaded()

        # Build documents
        books_df = books_df.copy()
        books_df["document"] = books_df.apply(TextProcessor.build_document, axis=1)

        # Encode documents to dense embeddings
        documents = books_df["document"].fillna("").astype(str).tolist()
        embeddings_np = self.model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = torch.from_numpy(embeddings_np).to(self.device)
        self.feature_matrix = self.embeddings
        self.book_ids = books_df["book_id"].astype(int).values
        self.id_to_idx = {int(bid): idx for idx, bid in enumerate(self.book_ids)}

        logger.info(f"Content embeddings: {self.embeddings.shape}")
        logger.info(f"Content features: {self.embeddings.shape}")

    def get_similar(self, book_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get similar books via cosine similarity"""
        if self.embeddings is None or book_id not in self.id_to_idx:
            return []

        idx = self.id_to_idx[book_id]
        query_vec = self.embeddings[idx].unsqueeze(0)

        scores = util.cos_sim(query_vec, self.embeddings).flatten()

        # Exclude self
        scores[idx] = -1.0

        top_k = min(top_k, scores.shape[0])
        if top_k <= 0:
            return []

        top_values, top_indices = torch.topk(scores, k=top_k)
        similar_items = []
        for score, index in zip(top_values.tolist(), top_indices.tolist()):
            if score <= 0:
                continue
            similar_items.append((int(self.book_ids[index]), float(score)))

        return similar_items

    def save(self, path: Path):
        """Save embeddings and metadata"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "embeddings": self.embeddings.detach().cpu().numpy()
                    if self.embeddings is not None
                    else None,
                    "book_ids": self.book_ids,
                    "id_to_idx": self.id_to_idx,
                    "batch_size": self.batch_size,
                },
                f,
            )
        logger.info(f"Saved content model to {path}")

    @classmethod
    def load(cls, path: Path):
        """Load saved model"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        model = cls(
            model_name=data.get("model_name", "keepitreal/vietnamese-sbert"),
            batch_size=data.get("batch_size", 32),
        )
        embeddings = data.get("embeddings")
        if embeddings is not None:
            model.embeddings = torch.from_numpy(embeddings).to(model.device)
            model.feature_matrix = model.embeddings
        model.book_ids = data.get("book_ids")
        model.id_to_idx = data.get("id_to_idx", {})
        return model

    def _ensure_model_loaded(self):
        """Lazily load the SBERT model to the appropriate device"""
        if self.model is None:
            logger.info(
                f"Loading SentenceTransformer model '{self.model_name}' on {self.device}"
            )
            self.model = SentenceTransformer(self.model_name, device=self.device)
