"""FAISS Vector Database Manager for RA Project"""
import os
from typing import List, Dict, Optional, Any
import numpy as np
import faiss
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class FAISSManager:
    """Manages FAISS vector database for document embeddings"""
    
    def __init__(self, index_path: str = "./vector_db/embeddings", dimension: int = 1536):
        """
        Initialize FAISS manager
        
        Args:
            index_path: Path to store FAISS index
            dimension: Embedding dimension (1536 for OpenAI text-embedding-ada-002)
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        
        self.index_file = self.index_path / "faiss_index.bin"
        self.metadata_file = self.index_path / "metadata.pkl"
        
        # Initialize or load index
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
        else:
            # Create a new index (using L2 distance)
            self.index = faiss.IndexFlatL2(dimension)
            logger.info(f"Created new FAISS index with dimension {dimension}")
        
        # Load or initialize metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata: List[Dict[str, Any]] = []
    
    def add_embeddings(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Add embeddings to the index
        
        Args:
            embeddings: Numpy array of shape (n, dimension)
            metadata: List of metadata dictionaries for each embedding
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Add metadata
        self.metadata.extend(metadata)
        
        # Save to disk
        self.save()
        
        logger.info(f"Added {len(embeddings)} embeddings to index")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query vector of shape (1, dimension)
            k: Number of nearest neighbors to return
        
        Returns:
            List of dictionaries with 'distance' and 'metadata' keys
        """
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Compile results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    'distance': float(dist),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def save(self) -> None:
        """Save index and metadata to disk"""
        faiss.write_index(self.index, str(self.index_file))
        
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved FAISS index to {self.index_file}")
    
    def clear(self) -> None:
        """Clear the index and metadata"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.save()
        logger.info("Cleared FAISS index")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_path': str(self.index_path),
            'metadata_count': len(self.metadata)
        }


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = FAISSManager()
    
    # Create sample embeddings (normally these would come from an embedding model)
    sample_embeddings = np.random.randn(10, 1536).astype('float32')
    sample_metadata = [
        {'doc_id': i, 'text': f'Sample document {i}', 'category': 'investment'}
        for i in range(10)
    ]
    
    # Add embeddings
    manager.add_embeddings(sample_embeddings, sample_metadata)
    
    # Search
    query = np.random.randn(1, 1536).astype('float32')
    results = manager.search(query, k=3)
    
    print("Search results:")
    for result in results:
        print(f"Distance: {result['distance']:.4f}, Metadata: {result['metadata']}")
    
    # Get stats
    print("\nIndex stats:", manager.get_stats())
