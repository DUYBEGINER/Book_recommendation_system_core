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
from typing import Dict, List, Tuple, Optional
from src.utils.logging_config import logger


class InteractionDataset(Dataset):
    """Dataset for user-item interactions"""
    
    def __init__(self, interactions_df: pd.DataFrame, 
                 user_id_map: Dict, item_id_map: Dict):
        """
        Args:
            interactions_df: DataFrame with [user_id, book_id, strength]
            user_id_map: {user_id: idx}
            item_id_map: {book_id: idx}
        """
        self.interactions = []
        
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            book_id = row['book_id']
            rating = row['strength']
            
            if user_id in user_id_map and book_id in item_id_map:
                user_idx = user_id_map[user_id]
                item_idx = item_id_map[book_id]
                self.interactions.append((user_idx, item_idx, rating))
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_idx, item_idx, rating = self.interactions[idx]
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(item_idx, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float)
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
    
    def __init__(self, gmf_dim: int = 64, mlp_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.2, learning_rate: float = 0.001,
                 batch_size: int = 256, epochs: int = 20,
                 device: str = None):
        """
        Args:
            gmf_dim: GMF embedding dimension
            mlp_dims: MLP layer dimensions
            dropout: Dropout rate
            learning_rate: Adam learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.gmf_dim = gmf_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model: Optional[NeuMF] = None
        self.user_id_map: Dict[int, int] = {}
        self.item_id_map: Dict[int, int] = {}
        self.user_id_reverse: Dict[int, int] = {}
        self.item_id_reverse: Dict[int, int] = {}
    
    def fit(self, interactions_df: pd.DataFrame):
        """
        Train Neural CF model
        
        Args:
            interactions_df: DataFrame with [user_id, book_id, strength]
        """
        logger.info("Training Neural Collaborative Filtering...")
        
        # Aggregate interactions (sum strengths)
        agg = interactions_df.groupby(['user_id', 'book_id'])['strength'].sum().reset_index()
        
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
        
        # Create dataset and dataloader
        dataset = InteractionDataset(agg, self.user_id_map, self.item_id_map)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                              shuffle=True, num_workers=0)
        
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
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            n_batches = 0
            
            for user_idx, item_idx, rating in dataloader:
                user_idx = user_idx.to(self.device)
                item_idx = item_idx.to(self.device)
                rating = rating.to(self.device)
                
                # Forward pass
                predictions = self.model(user_idx, item_idx)
                loss = criterion(predictions, rating)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            
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
        
        self.model.eval()
        
        user_idx = self.user_id_map[user_id]
        n_items = len(self.item_id_map)
        
        # Predict scores for all items
        user_tensor = torch.tensor([user_idx] * n_items, dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(list(range(n_items)), dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(user_tensor, item_tensor).cpu().numpy()
        
        # Filter out interacted items
        if filter_items:
            for item_id in filter_items:
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
            'epochs': self.epochs
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
            device=device
        )
        
        model.user_id_map = save_dict['user_id_map']
        model.item_id_map = save_dict['item_id_map']
        model.user_id_reverse = save_dict['user_id_reverse']
        model.item_id_reverse = save_dict['item_id_reverse']
        
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
