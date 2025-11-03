"""
Neural Collaborative Filtering (NCF) Model
Uses neural networks to learn user-item interactions

Architecture:
- GMF (Generalized Matrix Factorization): element-wise product
- MLP (Multi-Layer Perceptron): deep interactions
- NeuMF: Fusion of GMF + MLP
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from itertools import product
from typing import Dict, List, Tuple, Optional, Any
from src.utils.logging_config import logger


class InteractionDataset(Dataset):
    """Dataset for explicit user-item interactions."""

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        user_id_map: Dict[int, int],
        item_id_map: Dict[int, int],
    ):
        """
        Args:
            interactions_df: DataFrame with [user_id, book_id, strength]
            user_id_map: {user_id: idx}
            item_id_map: {book_id: idx}
        """
        self.samples: List[Tuple[int, int, float]] = []

        for _, row in interactions_df.iterrows():
            user_id = row["user_id"]
            book_id = row["book_id"]
            rating = row["strength"]

            if user_id not in user_id_map or book_id not in item_id_map:
                continue

            user_idx = user_id_map[user_id]
            item_idx = item_id_map[book_id]
            self.samples.append((user_idx, item_idx, float(rating)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        user_idx, item_idx, rating = self.samples[idx]
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(item_idx, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float),
        )


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization
    
    Combines:
    1. GMF: user_emb ⊙ item_emb (element-wise product)
    2. MLP: concat(user_emb, item_emb) → deep layers
    3. Fusion: concat(GMF, MLP) → prediction
    """
    
    def __init__(self, n_users: int, n_items: int,
                 gmf_dim: int = 64, mlp_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.2):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            gmf_dim: GMF embedding dimension
            mlp_dims: MLP layer dimensions [dim1, dim2, ...]
            dropout: Dropout rate
        """
        super(NeuMF, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        
        # GMF branch
        self.gmf_user_embedding = nn.Embedding(n_users, gmf_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, gmf_dim)
        
        # MLP branch
        self.mlp_user_embedding = nn.Embedding(n_users, mlp_dims[0] // 2)
        self.mlp_item_embedding = nn.Embedding(n_items, mlp_dims[0] // 2)
        
        mlp_layers = []
        input_dim = mlp_dims[0]
        for output_dim in mlp_dims[1:]:
            mlp_layers.append(nn.Linear(input_dim, output_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Fusion layer
        fusion_input_dim = gmf_dim + mlp_dims[-1]
        self.fusion = nn.Linear(fusion_input_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        nn.init.xavier_uniform_(self.gmf_user_embedding.weight)
        nn.init.xavier_uniform_(self.gmf_item_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_user_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_item_embedding.weight)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        Args:
            user_ids: (batch_size,)
            item_ids: (batch_size,)
        
        Returns:
            predictions: (batch_size,)
        """
        # GMF branch
        gmf_user = self.gmf_user_embedding(user_ids)  # (batch, gmf_dim)
        gmf_item = self.gmf_item_embedding(item_ids)  # (batch, gmf_dim)
        gmf_output = gmf_user * gmf_item  # Element-wise product
        
        # MLP branch
        mlp_user = self.mlp_user_embedding(user_ids)  # (batch, mlp_dim/2)
        mlp_item = self.mlp_item_embedding(item_ids)  # (batch, mlp_dim/2)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)  # (batch, mlp_dim)
        mlp_output = self.mlp(mlp_input)  # (batch, mlp_dims[-1])
        
        # Fusion
        fusion_input = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = self.fusion(fusion_input).squeeze()  # (batch,)
        
        return prediction


class NeuralCFModel:
    """Neural Collaborative Filtering recommender"""

    def __init__(
        self,
        gmf_dim: int = 64,
        mlp_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        device: str = None,
        weight_decay: float = 1e-5,
        random_state: int = 42,
    ):
        """
        Args:
            gmf_dim: GMF embedding dimension
            mlp_dims: MLP layer dimensions
            dropout: Dropout rate
            learning_rate: Adam learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            device: 'cuda' or 'cpu' (auto-detect if None)
            weight_decay: L2 penalty applied during optimization
            random_state: Seed for reproducibility (sampling & initialization)
        """
        self.gmf_dim = gmf_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.training_history: List[float] = []
        self.last_training_loss: Optional[float] = None
        self.last_evaluation_metrics: Optional[Dict[str, float]] = None
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model: Optional[NeuMF] = None
        self.user_id_map: Dict[int, int] = {}
        self.item_id_map: Dict[int, int] = {}
        self.user_id_reverse: Dict[int, int] = {}
        self.item_id_reverse: Dict[int, int] = {}
        self.training_interactions: Dict[int, set] = {}
    
    def fit(self, interactions_df: pd.DataFrame):
        """
        Train Neural CF model
        
        Args:
            interactions_df: DataFrame with [user_id, book_id, strength]
        """
        logger.info("Training Neural Collaborative Filtering...")
        
        # Aggregate interactions (sum strengths)
        agg = (
            interactions_df.groupby(['user_id', 'book_id'])['strength']
            .sum()
            .reset_index()
        )

        # Build ID mappings
        user_ids = sorted(agg['user_id'].unique())
        item_ids = sorted(agg['book_id'].unique())
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}
        self.user_id_reverse = {idx: uid for uid, idx in self.user_id_map.items()}
        self.item_id_reverse = {idx: iid for iid, idx in self.item_id_map.items()}
        
        n_users = len(user_ids)
        n_items = len(item_ids)
        
        logger.info(f"Users: {n_users}, Items: {n_items}")
        if n_users == 0 or n_items == 0:
            logger.warning("Empty interaction set provided; skipping training")
            return

        self.training_interactions = {
            int(user): {int(book) for book in books}
            for user, books in agg.groupby('user_id')['book_id']
        }

        # Prepare dataset of explicit ratings
        dataset = InteractionDataset(
            agg,
            self.user_id_map,
            self.item_id_map,
        )
        if len(dataset) == 0:
            logger.warning("No usable interactions after preprocessing; skipping training")
            return

        self.training_history = []
        self.last_training_loss = None

        # Initialize model
        self.model = NeuMF(
            n_users=n_users,
            n_items=n_items,
            gmf_dim=self.gmf_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )
            total_loss = 0
            n_batches = 0
            
            for user_idx, item_idx, label in dataloader:
                user_idx = user_idx.to(self.device)
                item_idx = item_idx.to(self.device)
                label = label.to(self.device)
                
                # Forward pass
                predictions = self.model(user_idx, item_idx)
                loss = criterion(predictions, label)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches if n_batches else float("nan")
            self.training_history.append(float(avg_loss))
            self.last_training_loss = float(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        logger.info("Neural CF training completed")
    
    def recommend(self, user_id: int, top_k: int = 10, 
                  filter_items: Optional[set] = None) -> List[Tuple[int, float]]:
        """
        Get recommendations for user
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            filter_items: Set of item IDs to exclude
        
        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_id_map:
            logger.debug(f"User {user_id} not in training set")
            return []

        if self.model is None:
            logger.error("Model not trained")
            return []

        predictions = self._score_all_items(user_id)
        if predictions is None:
            return []
        predictions = predictions.copy()

        default_exclusions = self.training_interactions.get(user_id, set())
        if filter_items:
            exclude_items = set(filter_items) | default_exclusions
        else:
            exclude_items = default_exclusions

        # Filter out interacted items
        if exclude_items:
            for item_id in exclude_items:
                if item_id in self.item_id_map:
                    idx = self.item_id_map[item_id]
                    predictions[idx] = -np.inf
        
        # Get top K
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if predictions[idx] > -np.inf:
                item_id = self.item_id_reverse[idx]
                score = float(predictions[idx])
                results.append((item_id, score))
        
        return results

    def _score_all_items(self, user_id: int) -> Optional[np.ndarray]:
        """Return predicted ratings for all items for a given user."""
        if user_id not in self.user_id_map or self.model is None:
            return None

        self.model.eval()

        user_idx = self.user_id_map[user_id]
        n_items = len(self.item_id_map)

        user_tensor = torch.full((n_items,), user_idx, dtype=torch.long).to(self.device)
        item_tensor = torch.arange(n_items, dtype=torch.long).to(self.device)

        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor).cpu().numpy()

        return scores

    def evaluate(self, interactions_df: pd.DataFrame, top_k: int = 10) -> Dict[str, float]:
        """Evaluate the model on a hold-out set using MSE / RMSE and ranking hit-rate."""
        if self.model is None:
            logger.error("Model not trained; cannot evaluate")
            return {"mse_loss": float("nan"), "rmse": float("nan"), f"hit_rate@{top_k}": 0.0}

        mask = interactions_df['user_id'].isin(self.user_id_map) & (
            interactions_df['book_id'].isin(self.item_id_map)
        )
        filtered = interactions_df[mask]

        if filtered.empty:
            logger.warning("No overlapping users/items for evaluation; returning defaults")
            return {"mse_loss": float("nan"), "rmse": float("nan"), f"hit_rate@{top_k}": 0.0}

        dataset = InteractionDataset(
            filtered,
            self.user_id_map,
            self.item_id_map,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        criterion = nn.MSELoss()
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for user_idx, item_idx, label in dataloader:
                user_idx = user_idx.to(self.device)
                item_idx = item_idx.to(self.device)
                label = label.to(self.device)

                predictions = self.model(user_idx, item_idx)
                loss = criterion(predictions, label)

                total_loss += float(loss.item()) * len(label)
                total_samples += len(label)

        mse = total_loss / total_samples if total_samples else float("nan")
        rmse = float(np.sqrt(mse)) if not np.isnan(mse) else float("nan")
        metrics = {"mse_loss": mse, "rmse": rmse}

        if top_k > 0:
            metrics[f"hit_rate@{top_k}"] = self._hit_rate_at_k(filtered, top_k=top_k)

        self.last_evaluation_metrics = metrics
        return metrics

    def cross_validate(
        self,
        interactions_df: pd.DataFrame,
        n_splits: int = 5,
        random_state: int = 42,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run K-fold cross-validation and report metrics per fold."""
        from sklearn.model_selection import KFold

        interactions_df = interactions_df.reset_index(drop=True)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        results: List[Dict[str, Any]] = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(interactions_df)):
            train_df = interactions_df.iloc[train_idx]
            val_df = interactions_df.iloc[val_idx]

            fold_model = NeuralCFModel(
                gmf_dim=self.gmf_dim,
                mlp_dims=self.mlp_dims,
                dropout=self.dropout,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                epochs=self.epochs,
                device=self.device,
                weight_decay=self.weight_decay,
                random_state=random_state + fold,
            )

            fold_model.fit(train_df)
            metrics = fold_model.evaluate(val_df, top_k=top_k)
            metrics['fold'] = fold
            results.append(metrics)

        return results

    @staticmethod
    def grid_search(
        interactions_df: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        n_splits: int = 5,
        metric: str = 'rmse',
        random_state: int = 42,
        device: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Simple grid search over hyperparameters using cross-validation."""
        param_keys = list(param_grid.keys())
        if not param_keys:
            raise ValueError("param_grid must contain at least one hyperparameter")

        search_space = list(product(*[param_grid[key] for key in param_keys]))

        minimize = metric.startswith('rmse') or metric.startswith('mse')
        best_score = float('inf') if minimize else -np.inf
        best_params: Optional[Dict[str, Any]] = None
        history: List[Dict[str, Any]] = []

        for combo in search_space:
            params = dict(zip(param_keys, combo))

            model_kwargs = {
                'gmf_dim': params.get('gmf_dim', 64),
                'mlp_dims': params.get('mlp_dims', [128, 64, 32]),
                'dropout': params.get('dropout', 0.2),
                'learning_rate': params.get('learning_rate', 0.001),
                'batch_size': params.get('batch_size', 256),
                'epochs': params.get('epochs', 20),
                'weight_decay': params.get('weight_decay', 1e-5),
                'random_state': random_state,
                'device': device,
            }

            model = NeuralCFModel(**model_kwargs)
            k = int(metric.split('@')[-1]) if '@' in metric else 10
            cv_results = model.cross_validate(
                interactions_df,
                n_splits=n_splits,
                random_state=random_state,
                top_k=k,
            )

            scores = [fold_metrics.get(metric) for fold_metrics in cv_results if metric in fold_metrics]
            avg_score = float(np.nanmean(scores)) if scores else float("nan")

            history.append({
                'params': params,
                'cv_results': cv_results,
                'avg_score': avg_score,
            })

            if np.isnan(avg_score):
                continue

            if minimize:
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = params
            else:
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params

        return best_params, history

    def _hit_rate_at_k(self, interactions_df: pd.DataFrame, top_k: int = 10) -> float:
        """Compute hit rate@k treating each observed pair as a positive signal."""
        hits = 0
        total = 0

        for user_id, group in interactions_df.groupby('user_id'):
            if user_id not in self.user_id_map:
                continue

            positives = [int(book_id) for book_id in group['book_id'] if book_id in self.item_id_map]
            if not positives:
                continue

            top_items = self._top_k_items(user_id, top_k)
            for book_id in positives:
                total += 1
                if book_id in top_items:
                    hits += 1

        return hits / total if total else 0.0

    def _top_k_items(self, user_id: int, top_k: int) -> List[int]:
        """Helper returning top_k item ids for a user (excluding train positives)."""
        scores = self._score_all_items(user_id)
        if scores is None:
            return []

        scores = scores.copy()
        for item_id in self.training_interactions.get(user_id, set()):
            if item_id in self.item_id_map:
                scores[self.item_id_map[item_id]] = -np.inf

        top_indices = np.argsort(scores)[::-1]
        top_items: List[int] = []
        for idx in top_indices:
            if len(top_items) >= top_k:
                break
            if not np.isfinite(scores[idx]):
                continue
            top_items.append(self.item_id_reverse[idx])

        return top_items

    def save(self, path: Path):
        """Save model"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state': self.model.state_dict() if self.model else None,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'user_id_reverse': self.user_id_reverse,
            'item_id_reverse': self.item_id_reverse,
            'gmf_dim': self.gmf_dim,
            'mlp_dims': self.mlp_dims,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'weight_decay': self.weight_decay,
            'random_state': self.random_state,
            'training_history': self.training_history,
            'last_training_loss': self.last_training_loss,
            'last_evaluation_metrics': self.last_evaluation_metrics,
        }
        
        torch.save(save_dict, path)
        logger.info(f"Saved Neural CF model to {path}")
    
    @classmethod
    def load(cls, path: Path, device: str = None):
        """Load saved model"""
        # weights_only=False is required to load NumPy objects from trusted checkpoints
        save_dict = torch.load(path, map_location='cpu', weights_only=False)
        
        model = cls(
            gmf_dim=save_dict['gmf_dim'],
            mlp_dims=save_dict['mlp_dims'],
            dropout=save_dict['dropout'],
            learning_rate=save_dict['learning_rate'],
            batch_size=save_dict['batch_size'],
            epochs=save_dict['epochs'],
            device=device,
            weight_decay=save_dict.get('weight_decay', 1e-5),
            random_state=save_dict.get('random_state', 42),
        )

        model.user_id_map = save_dict['user_id_map']
        model.item_id_map = save_dict['item_id_map']
        model.user_id_reverse = save_dict['user_id_reverse']
        model.item_id_reverse = save_dict['item_id_reverse']
        model.training_history = save_dict.get('training_history', [])
        model.last_training_loss = save_dict.get('last_training_loss')
        model.last_evaluation_metrics = save_dict.get('last_evaluation_metrics')
        
        if save_dict['model_state']:
            n_users = len(model.user_id_map)
            n_items = len(model.item_id_map)
            
            model.model = NeuMF(
                n_users=n_users,
                n_items=n_items,
                gmf_dim=model.gmf_dim,
                mlp_dims=model.mlp_dims,
                dropout=model.dropout
            ).to(model.device)
            
            model.model.load_state_dict(save_dict['model_state'])
            model.model.eval()
        
        logger.info(f"Loaded Neural CF model from {path}")
        return model
