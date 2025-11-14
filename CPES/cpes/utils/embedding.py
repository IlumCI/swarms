"""
Embedding utilities for CPES.

This module provides embedding functionality for converting text to vectors
for semantic similarity search in episodic memory.
"""

import numpy as np
from typing import List, Union, Optional
from loguru import logger
import os


class EmbeddingModel:
    """
    Embedding model wrapper for CPES.
    
    This class provides a unified interface for different embedding models
    and handles the conversion of text to vector representations.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 model_type: str = "sentence_transformers",
                 device: str = "cpu"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the embedding model to use
            model_type: Type of model ("sentence_transformers", "openai", "custom")
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.model = None
        self.dimension = None
        
        self._load_model()
        logger.info(f"Initialized embedding model: {model_name}")
    
    def _load_model(self) -> None:
        """Load the embedding model."""
        try:
            if self.model_type == "sentence_transformers":
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.dimension = self.model.get_sentence_embedding_dimension()
                
            elif self.model_type == "openai":
                import openai
                # OpenAI embeddings don't need a local model
                self.model = openai
                self.dimension = 1536  # OpenAI text-embedding-ada-002 dimension
                
            elif self.model_type == "custom":
                # For custom models, user should set self.model and self.dimension
                logger.warning("Custom model type selected - ensure model and dimension are set")
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except ImportError as e:
            logger.error(f"Failed to import required library for {self.model_type}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], 
               batch_size: int = 32) -> np.ndarray:
        """
        Encode text(s) to embeddings.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing (for sentence_transformers)
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            if self.model_type == "sentence_transformers":
                embeddings = self.model.encode(texts, batch_size=batch_size, 
                                            convert_to_numpy=True)
                
            elif self.model_type == "openai":
                embeddings = self._encode_openai(texts)
                
            elif self.model_type == "custom":
                embeddings = self._encode_custom(texts)
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Ensure embeddings are numpy arrays
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            logger.debug(f"Encoded {len(texts)} texts to {embeddings.shape} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI API."""
        try:
            import openai
            
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            openai.api_key = api_key
            
            # Make API call
            response = openai.Embedding.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            
            # Extract embeddings
            embeddings = [item["embedding"] for item in response["data"]]
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    def _encode_custom(self, texts: List[str]) -> np.ndarray:
        """Encode texts using custom model."""
        if self.model is None:
            raise ValueError("Custom model not set")
        
        # This is a placeholder - user should implement their custom model logic
        # The model should accept a list of texts and return embeddings
        return self.model(texts)
    
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self.dimension is None:
            raise ValueError("Model dimension not set")
        return self.dimension
    
    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def get_similarities(self, query_embedding: np.ndarray, 
                        candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between a query and multiple candidates.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(candidate_embeddings))
        
        query_normalized = query_embedding / query_norm
        
        # Normalize candidate embeddings
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
        candidate_normalized = candidate_embeddings / candidate_norms[:, np.newaxis]
        
        # Calculate cosine similarities
        similarities = np.dot(candidate_normalized, query_normalized)
        return similarities
    
    def find_most_similar(self, query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5) -> tuple:
        """
        Find the most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            Tuple of (indices, similarities) for top_k results
        """
        similarities = self.get_similarities(query_embedding, candidate_embeddings)
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities
    
    def __str__(self) -> str:
        """String representation."""
        return f"EmbeddingModel({self.model_name}, {self.model_type}, dim={self.dimension})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"EmbeddingModel(model_name='{self.model_name}', model_type='{self.model_type}', dimension={self.dimension})"
