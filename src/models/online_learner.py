# ==================== src/models/online_learner.py ====================
"""
Online Learning cho Collaborative Filtering
Cập nhật model khi có interaction mới mà không cần retrain toàn bộ
"""
import numpy as np
from typing import Dict, List, Tuple
from src.utils.logging_config import logger

class OnlineLearner:
    """
    Incremental update cho recommendation model
    Sử dụng cho cold-start và quick adaptation
    """
    
    def __init__(self):
        self.user_profiles = {}  # Cache user preferences
        self.item_popularity = {}  # Track item popularity
        self.recent_interactions = []  # Buffer cho batch update
        
    def add_interaction(self, user_id: int, book_id: int, strength: float, interaction_type: str):
        """
        Thêm interaction mới vào buffer
        
        Args:
            user_id: ID của user
            book_id: ID của sách
            strength: Độ mạnh tương tác (1-5 cho rating, 3 cho favorite, etc.)
            interaction_type: 'rating', 'favorite', 'view', 'bookmark'
        """
        # Cập nhật user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'books': [],
                'avg_rating': 0.0,
                'count': 0
            }
        
        profile = self.user_profiles[user_id]
        profile['books'].append((book_id, strength, interaction_type))
        profile['count'] += 1
        
        # Cập nhật popularity
        if book_id not in self.item_popularity:
            self.item_popularity[book_id] = {'views': 0, 'favorites': 0, 'ratings': []}
        
        if interaction_type == 'view':
            self.item_popularity[book_id]['views'] += 1
        elif interaction_type == 'favorite':
            self.item_popularity[book_id]['favorites'] += 1
        elif interaction_type == 'rating':
            self.item_popularity[book_id]['ratings'].append(strength)
        
        # Thêm vào buffer
        self.recent_interactions.append({
            'user_id': user_id,
            'book_id': book_id,
            'strength': strength,
            'type': interaction_type
        })
        
        logger.info(f"Added interaction: User {user_id} -> Book {book_id} ({interaction_type})")
        
        # Auto batch update nếu buffer đầy
        if len(self.recent_interactions) >= 100:
            self.flush_to_database()
    
    def get_cold_start_recommendations(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        Recommendations cho user mới dựa trên popularity và recent interactions
        """
        # Sắp xếp sách theo popularity score
        popular_books = []
        
        for book_id, stats in self.item_popularity.items():
            # Tính popularity score
            score = (
                stats['views'] * 0.1 +
                stats['favorites'] * 0.5 +
                (np.mean(stats['ratings']) if stats['ratings'] else 0) * 0.4
            )
            popular_books.append((book_id, score))
        
        # Sort và lấy top K
        popular_books.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {
                'book_id': book_id,
                'score': score,
                'reasons': {'cf': 0.0, 'content': 0.0, 'pop': 1.0}
            }
            for book_id, score in popular_books[:limit]
        ]
    
    def get_user_based_recommendations(self, user_id: int, all_users: Dict, limit: int = 10) -> List[Dict]:
        """
        Simple user-based CF cho user có ít interactions
        Tìm similar users và recommend sách họ thích
        """
        if user_id not in self.user_profiles:
            return []
        
        user_books = set([book_id for book_id, _, _ in self.user_profiles[user_id]['books']])
        
        # Tìm similar users (có chung sách)
        similar_users = []
        for other_id, other_profile in self.user_profiles.items():
            if other_id == user_id:
                continue
            
            other_books = set([book_id for book_id, _, _ in other_profile['books']])
            common = len(user_books & other_books)
            
            if common > 0:
                similarity = common / len(user_books | other_books)  # Jaccard similarity
                similar_users.append((other_id, similarity))
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)
        
        # Recommend sách từ similar users
        recommended_books = {}
        for other_id, similarity in similar_users[:5]:  # Top 5 similar users
            for book_id, strength, _ in self.user_profiles[other_id]['books']:
                if book_id not in user_books:
                    if book_id not in recommended_books:
                        recommended_books[book_id] = 0
                    recommended_books[book_id] += strength * similarity
        
        # Convert to list và sort
        results = [
            {
                'book_id': book_id,
                'score': score,
                'reasons': {'cf': score, 'content': 0.0, 'pop': 0.0}
            }
            for book_id, score in recommended_books.items()
        ]
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:limit]
    
    def flush_to_database(self):
        """
        Flush buffer interactions vào database
        Gọi hàm này định kỳ hoặc khi buffer đầy
        """
        if not self.recent_interactions:
            return
        
        logger.info(f"Flushing {len(self.recent_interactions)} interactions to database")
        
        # TODO: Implement database insert
        # INSERT INTO interactions (user_id, book_id, strength, type, created_at)
        # VALUES (...)
        
        self.recent_interactions.clear()
    
    def get_stats(self) -> Dict:
        """Thống kê về online learning"""
        return {
            'total_users': len(self.user_profiles),
            'total_items_tracked': len(self.item_popularity),
            'pending_interactions': len(self.recent_interactions),
            'avg_interactions_per_user': np.mean([p['count'] for p in self.user_profiles.values()]) if self.user_profiles else 0
        }
