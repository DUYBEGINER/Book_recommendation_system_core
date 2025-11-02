from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RecommendationItem:
    book_id: int
    rating: float
    score: float
    metadata: Dict[str, float]


class DiversityRecommender:
    """
    Recommend diverse books from a target book using semantic signals.

    Pipeline (mirrors the provided flowchart):
        1. Use provided sentence embeddings (preferred) or build TF-IDF vectors from tag columns.
        2. Compute cosine similarity scores against the target book.
        3. Select candidates with low tag overlap (diverse) and rank them using average rating.

    Parameters
    ----------
    books_df:
        DataFrame containing at least `book_id` and tag/genre columns.
    interactions_df:
        Optional interactions with a `type` column (expects `rating`) and either `rating_value`
        or `strength` to compute per-book mean ratings.
    tag_columns:
        Sequence of column names from `books_df` used to build the TF-IDF corpus.
        Defaults to `["genres_text"]`.
    """

    def __init__(
        self,
        books_df: pd.DataFrame,
        interactions_df: Optional[pd.DataFrame] = None,
        tag_columns: Optional[Sequence[str]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        if "book_id" not in books_df.columns:
            raise ValueError("books_df must include a 'book_id' column.")

        self.books_df = books_df.copy()
        self.tag_columns = list(tag_columns) if tag_columns else ["genres_text"]
        self._ensure_tag_columns()

        self.book_ids = self.books_df["book_id"].astype(int).tolist()
        self.id_to_index = {book_id: idx for idx, book_id in enumerate(self.book_ids)}

        self.embedding_matrix: Optional[np.ndarray] = None
        if embeddings is not None:
            if len(embeddings) != len(self.book_ids):
                raise ValueError("Embeddings length must match number of books.")
            self.embedding_matrix = embeddings.astype(np.float32)
            self.vectorizer = None
            self.tfidf_matrix = None
        else:
            corpus = self._build_corpus()
            self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True)
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        self.tag_sets = self._build_tag_sets()
        self.rating_map, self.global_rating = self._build_rating_map(interactions_df)

    def recommend(
        self,
        book_id: int,
        limit: int = 5,
        oversample: int = 10,
    ) -> List[RecommendationItem]:
        """
        Return a list of diverse recommendation items for the given book ID.
        """
        if book_id not in self.id_to_index:
            raise ValueError(f"Book ID {book_id} not found in the provided books dataframe.")

        idx = self.id_to_index[book_id]
        if self.embedding_matrix is not None:
            query_vec = self.embedding_matrix[idx]
            similarity_vector = self.embedding_matrix @ query_vec
        else:
            similarity_vector = cosine_similarity(
                self.tfidf_matrix[idx],
                self.tfidf_matrix,
            ).flatten()
        similarity_vector[idx] = 0.0  # ignore self

        return self._rank_diverse(idx, similarity_vector, limit, oversample)

    # --- Private helpers -------------------------------------------------

    def _ensure_tag_columns(self) -> None:
        for column in self.tag_columns:
            if column not in self.books_df.columns:
                self.books_df[column] = ""

    def _build_corpus(self) -> List[str]:
        texts: List[str] = []
        for _, row in self.books_df.iterrows():
            parts = [str(row[col]) for col in self.tag_columns if pd.notna(row[col])]
            texts.append(" ".join(parts).strip())
        return texts

    def _build_tag_sets(self) -> Dict[int, List[str]]:
        tag_sets: Dict[int, List[str]] = {}
        for book_id, row in self.books_df.set_index("book_id").iterrows():
            tags: List[str] = []
            for column in self.tag_columns:
                column_tags = str(row[column]).strip()
                if not column_tags:
                    continue
                tags.extend(t.strip() for t in column_tags.split() if t.strip())
            tag_sets[int(book_id)] = tags
        return tag_sets

    def _build_rating_map(
        self,
        interactions_df: Optional[pd.DataFrame],
    ) -> (Dict[int, float], float):
        rating_map: Dict[int, float] = {}

        if interactions_df is not None and not interactions_df.empty:
            ratings_df = interactions_df.copy()
            if "type" in ratings_df.columns:
                ratings_df = ratings_df[ratings_df["type"] == "rating"]

            rating_column = None
            for candidate in ("rating_value", "rating", "strength"):
                if candidate in ratings_df.columns:
                    rating_column = candidate
                    break

            if rating_column:
                grouped = ratings_df.groupby("book_id")[rating_column].mean()
                rating_map = {int(book_id): float(value) for book_id, value in grouped.items()}

        if not rating_map:
            fallback_column = next(
                (column for column in ("average_rating", "mean_rating", "rating") if column in self.books_df.columns),
                None,
            )
            if fallback_column:
                series = self.books_df.set_index("book_id")[fallback_column].dropna()
                rating_map = {int(book_id): float(value) for book_id, value in series.items()}

        if rating_map:
            global_rating = float(np.nanmean(list(rating_map.values())))
        else:
            global_rating = 0.0

        return rating_map, global_rating

    def _lookup_rating(self, book_id: int) -> float:
        rating = self.rating_map.get(book_id, self.global_rating)
        if np.isnan(rating):
            return self.global_rating
        return rating

    def _rank_diverse(
        self,
        target_idx: int,
        similarity_vector: np.ndarray,
        top_k: int,
        oversample: int,
    ) -> List[RecommendationItem]:
        if top_k <= 0:
            return []

        target_id = self.book_ids[target_idx]
        target_tags = set(self.tag_sets.get(target_id, []))

        candidates: List[RecommendationItem] = []
        diversity_bucket: List[Dict[str, float]] = []

        for idx, similarity in enumerate(similarity_vector):
            if idx == target_idx:
                continue

            candidate_id = self.book_ids[idx]
            candidate_tags = set(self.tag_sets.get(candidate_id, []))
            overlap = len(target_tags & candidate_tags)

            diversity_score = 1.0 - float(similarity)
            diversity_bucket.append(
                {
                    "index": idx,
                    "overlap": overlap,
                    "similarity": float(similarity),
                    "diversity": diversity_score,
                }
            )

        diversity_bucket.sort(key=lambda item: (item["overlap"], item["similarity"]))

        for entry in diversity_bucket[: max(top_k, oversample)]:
            idx = int(entry["index"])
            candidate_id = self.book_ids[idx]
            item = RecommendationItem(
                book_id=candidate_id,
                rating=self._lookup_rating(candidate_id),
                score=float(entry["diversity"]),
                metadata={
                    "diversity": float(entry["diversity"]),
                    "tag_overlap": float(entry["overlap"]),
                    "similarity": float(entry["similarity"]),
                },
            )
            candidates.append(item)

        candidates.sort(key=lambda item: (item.rating, item.score), reverse=True)
        return candidates[:top_k]
