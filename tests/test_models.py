import pytest
import pandas as pd
import numpy as np
from src.models.collaborative import CollaborativeModel
from src.features.content_features import ContentBasedModel
from src.models.hybrid import HybridRecommender

@pytest.fixture
def sample_interactions():
    return pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3],
        'book_id': [1, 2, 1, 3, 2],
        'strength': [5, 4, 3, 5, 4],
        'ts': pd.date_range('2024-01-01', periods=5)
    })

@pytest.fixture
def sample_books():
    return pd.DataFrame({
        'book_id': [1, 2, 3],
        'title': ['Book A', 'Book B', 'Book C'],
        'description': ['Fantasy adventure', 'Sci-fi thriller', 'Romance novel'],
        'authors_text': ['Author X', 'Author Y', 'Author Z'],
        'genres_text': ['Fantasy', 'Sci-fi', 'Romance'],
        'publisher': ['Pub A', 'Pub B', 'Pub C'],
        'publication_year': [2020, 2021, 2019]
    })

def test_collaborative_model_train(sample_interactions):
    model = CollaborativeModel(factors=8, iterations=5)
    model.fit(sample_interactions)
    
    assert model.user_item_matrix is not None
    assert len(model.user_ids) == 3
    assert len(model.item_ids) == 3

def test_collaborative_model_recommend(sample_interactions):
    model = CollaborativeModel(factors=8, iterations=5)
    model.fit(sample_interactions)
    
    recs = model.recommend(user_id=1, top_k=2)
    assert len(recs) <= 2
    assert all(isinstance(book_id, int) for book_id, score in recs)

def test_content_model_train(sample_books):
    model = ContentBasedModel(max_features=100)
    model.fit(sample_books)
    
    assert model.feature_matrix is not None
    assert len(model.book_ids) == 3

def test_content_model_similar(sample_books):
    model = ContentBasedModel(max_features=100)
    model.fit(sample_books)
    
    similar = model.get_similar(book_id=1, top_k=2)
    assert len(similar) <= 2

def test_hybrid_model(sample_books, sample_interactions):
    model = HybridRecommender(alpha=0.6)
    model.train(sample_books, sample_interactions)
    
    recs = model.recommend(user_id=1, limit=2)
    assert len(recs) <= 2
    assert all('book_id' in r and 'score' in r for r in recs)