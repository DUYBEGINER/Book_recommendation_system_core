# Interaction Strength Mapping

## 📊 Overview

Hệ thống sử dụng **weighted interactions** để học user preferences. Mỗi loại tương tác có **strength value** khác nhau dựa trên mức độ positive signal.

## 🎯 Strength Values

| Event Type | Strength | Rationale | Database Table |
|------------|----------|-----------|----------------|
| **view** | 1.0 | Low confidence - user chỉ xem, chưa chắc thích | N/A (realtime only) |
| **favorite** | 5.0 | Strong positive - user rất thích | `favorites` |
| **rate** | 1.0-5.0 | Explicit rating - user đánh giá trực tiếp | `ratings` |
| **history** | 1.0-5.0 | Reading progress - tính từ progress % | `reading_history` |

**Note**: Bookmark không được sử dụng trong training data hiện tại.

## 📐 Strength Calculation Details

### 1. View Events
```python
strength = 1.0  # Constant
```
- **Khi nào**: User xem trang chi tiết sách
- **Ý nghĩa**: Implicit signal, có thể do tò mò hoặc thật sự thích
- **Usage**: Chỉ dùng cho online learning, không lưu DB

### 2. Favorite Events
```python
strength = 5.0  # Maximum (equivalent to 5-star rating)
```
- **Khi nào**: User thêm sách vào favorites
- **Ý nghĩa**: Strong positive signal - user rất thích sách này
- **Database**: `favorites` table
  ```sql
  SELECT user_id, book_id, added_at AS ts, 
         5.0 AS strength, 'favorite' AS type
  FROM favorites
  ```

### 3. Rating Events
```python
strength = rating_value  # 1-5 from user input
```
- **Khi nào**: User đánh giá sách (1-5 stars)
- **Ý nghĩa**: Explicit feedback - độ tin cậy cao nhất
- **Database**: `ratings` table
  ```sql
  SELECT user_id, book_id, created_at AS ts, 
         rating_value::float AS strength, 'rating' AS type
  FROM ratings
  ```

### 4. Reading History (Calculated)
```python
strength = max(1.0, (progress / 100.0) * 5.0)
# Example: 80% progress = 4.0 strength
```
- **Khi nào**: User đọc sách (tracked by app)
- **Ý nghĩa**: Implicit signal - reading progress indicates interest
- **Database**: `reading_history` table
  ```sql
  SELECT user_id, book_id, last_read_at AS ts,
         GREATEST(1.0, COALESCE(progress/100.0, 0.5) * 5.0)::float AS strength,
         'history' AS type
  FROM reading_history
  ```
- **Examples**:
  - 20% progress → 1.0 strength (min)
  - 50% progress → 2.5 strength
  - 80% progress → 4.0 strength
  - 100% progress → 5.0 strength (max)

## 🔄 Consistency Rules

### ⚠️ CRITICAL: Strength values MUST be identical across:

1. **Training Data** (`db_loader.py`)
   ```python
   # Load from database with specific strengths
   ```

2. **Online Learning** (`routes.py` `/feedback` endpoint)
   ```python
   strength_map = {
       'view': 1.0,
       'favorite': 5.0,
       'rate': rating_value
   }
   ```

3. **Neural Server** (`routes_neural.py` `/feedback` endpoint)
   ```python
   # Same strength_map as above
   ```

### 🐛 Common Bugs to Avoid:

❌ **Wrong:**
```python
# db_loader.py
'favorite': 5.0

# routes.py
'favorite': 3.0  # INCONSISTENT!
```

✅ **Correct:**
```python
# db_loader.py
'favorite': 5.0

# routes.py
'favorite': 5.0  # CONSISTENT ✓
```

## 📈 Strength Distribution Example

Typical distribution in production (10k interactions):

```
Rating (1-5):    50%  (5,000 interactions, avg strength: 4.2)
Favorites:       30%  (3,000 interactions, strength: 5.0)
History:         20%  (2,000 interactions, avg strength: 3.5)
```

**Weighted Average Strength**: ~4.1

## 🎛️ Tuning Strength Values

### When to Adjust:

1. **User behavior changes**
   - Example: Users bookmark everything → reduce bookmark strength to 1.5

2. **Business requirements**
   - Example: Prioritize favorites → increase to 6.0 or 7.0

3. **Model performance**
   - Example: Too many favorites → normalize to 3.0

### How to Adjust:

1. **Update `db_loader.py`**:
   ```python
   SELECT ..., 3.0 AS strength, 'favorite' AS type  # Changed from 5.0
   ```

2. **Update `routes.py` and `routes_neural.py`**:
   ```python
   strength_map = {
       'favorite': 3.0  # Match db_loader.py
   }
   ```

3. **Retrain models**:
   ```bash
   python train.py --evaluate
   python train_neural.py --evaluate
   ```

4. **Document changes** in this file

## 🧪 Testing Consistency

Run this script to verify consistency:

```python
# test_strength_consistency.py
from src.data.db_loader import DatabaseLoader
from src.api.routes import strength_map as classic_strength_map
from src.api.routes_neural import strength_map as neural_strength_map

# Load training data
loader = DatabaseLoader(...)
interactions_df = loader.load_interactions()

# Check training strengths
training_strengths = {
    'favorite': interactions_df[interactions_df['type']=='favorite']['strength'].iloc[0],
    'bookmark': interactions_df[interactions_df['type']=='bookmark']['strength'].iloc[0],
    # ... etc
}

# Compare
assert classic_strength_map['favorite'] == training_strengths['favorite']
assert neural_strength_map['favorite'] == training_strengths['favorite']
print("✅ Strength values are consistent!")
```

## 📊 Impact on Recommendations

### Example User Profile:

```
User 123:
- Rated 5 books: avg rating 4.5 (strength: 4.5)
- Favorited 3 books: (strength: 5.0 each)
- Read 10 books: avg progress 60% (avg strength: 3.0)

Weighted Average Strength: (5*4.5 + 3*5.0 + 10*3.0) / 18 = 3.92
```

**Model learns**: User 123 has strong positive preferences (avg 3.92/5.0)

### Strength Impact on Recommendations:

- **High strength (4-5)**: Books strongly weighted in user profile
- **Medium strength (2-3)**: Moderate influence
- **Low strength (1)**: Weak signal, easily overridden

## 🔮 Future Enhancements

- [ ] **Time decay**: Reduce strength for old interactions
  ```python
  strength *= exp(-lambda * days_since_interaction)
  ```

- [ ] **Frequency weighting**: Multiple views → higher strength
  ```python
  strength = base_strength * log(1 + view_count)
  ```

- [ ] **Context-aware**: Different strengths for different contexts
  ```python
  strength = base_strength * context_multiplier
  # Morning reads: 1.2x, evening reads: 1.0x
  ```

- [ ] **Personalized strengths**: Learn user-specific strength calibration
  ```python
  strength = base_strength * user_calibration_factor[user_id]
  ```

## 📝 Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2025-11-03 | Removed bookmark from API | Not used in training data |
| 2025-11-03 | Changed favorite from 3.0 → 5.0 | Align with max rating scale |
| 2025-11-03 | Documented history calculation | Clarify progress-based strength |

---

**Last Updated**: November 3, 2025
**Maintainer**: @DUYBEGINER
