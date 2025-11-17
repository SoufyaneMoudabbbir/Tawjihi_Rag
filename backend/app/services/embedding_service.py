#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Service
Handles text embeddings generation using SentenceTransformers
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import EmbeddingError

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings
    Singleton pattern for model reuse
    """

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load embedding model"""
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Failed to load embedding model: {str(e)}")

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings

        Raises:
            EmbeddingError: If encoding fails
        """
        if self._model is None:
            raise EmbeddingError("Model not loaded")

        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]

            # Generate embeddings
            embeddings = self._model.encode(
                texts,
                show_progress_bar=show_progress,
                batch_size=batch_size,
                convert_to_numpy=True
            )

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """Get embedding vector dimension"""
        if self._model is None:
            raise EmbeddingError("Model not loaded")
        return self._model.get_sentence_embedding_dimension()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None
