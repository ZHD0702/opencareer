"""
Vector memory implementation using ChromaDB.

This module provides vector-based memory storage using ChromaDB for
semantic search capabilities.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from .memory_manager import BaseMemory, MemoryQuery, MemorySearchResult


class ChromaVectorMemory(BaseMemory):
    """Vector memory implementation using ChromaDB."""

    def __init__(self, collection_name: str = "opencareer_memory",
                 persist_directory: str = "./data/chromadb",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize ChromaDB vector memory.

        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the database
            embedding_model: Name of the embedding model to use
        """
        super().__init__(f"chroma_{collection_name}")
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self._collection = None
        self._embedding_function = None

    async def _ensure_initialized(self) -> None:
        """Ensure ChromaDB is initialized."""
        if self._collection is None:
            await self._initialize_chroma()

    async def _initialize_chroma(self) -> None:
        """Initialize ChromaDB connection and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer

            # Initialize embedding function
            self.logger.info(f"Loading embedding model: {self.embedding_model}")
            model = SentenceTransformer(self.embedding_model)

            def embedding_function(texts):
                return model.encode(texts).tolist()

            # Configure ChromaDB settings
            chroma_settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            )

            # Create ChromaDB client
            client = chromadb.Client(chroma_settings)

            # Get or create collection
            try:
                self._collection = client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function
                )
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self._collection = client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function,
                    metadata={"description": "OpenCareer conversation memory"}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")

            self._embedding_function = embedding_function
            self.logger.info("ChromaDB initialized successfully")

        except ImportError as e:
            self.logger.error(f"Failed to import required packages: {e}")
            raise RuntimeError(
                "Required packages not installed. Install with: "
                "pip install chromadb sentence-transformers"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    async def store(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Store content in vector memory.

        Args:
            content: The content to store
            metadata: Additional metadata

        Returns:
            ID of the stored memory
        """
        await self._ensure_initialized()

        if metadata is None:
            metadata = {}

        # Generate unique ID
        memory_id = str(uuid.uuid4())

        # Prepare metadata for ChromaDB
        chroma_metadata = metadata.copy()

        # Ensure all metadata values are strings (ChromaDB requirement)
        for key, value in chroma_metadata.items():
            if not isinstance(value, str):
                chroma_metadata[key] = str(value)

        try:
            # Add to collection
            self._collection.add(
                documents=[content],
                metadatas=[chroma_metadata],
                ids=[memory_id]
            )

            self.logger.debug(f"Stored content in vector memory, id: {memory_id}")
            return memory_id

        except Exception as e:
            self.logger.error(f"Failed to store content in vector memory: {e}")
            raise

    async def search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """Search for memories matching the query.

        Args:
            query: The search query

        Returns:
            List of search results
        """
        await self._ensure_initialized()

        try:
            # Prepare query
            n_results = query.limit
            where_filter = None

            if query.metadata_filter:
                # Convert metadata filter to ChromaDB format
                where_filter = {}
                for key, value in query.metadata_filter.items():
                    where_filter[key] = value

            # Perform search
            if query.text:
                results = self._collection.query(
                    query_texts=[query.text],
                    n_results=n_results,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                # If no query text, get most recent entries
                # This is a simplified approach - ChromaDB doesn't have native timestamp sorting
                # In a production system, you'd need to handle this differently
                results = self._collection.get(
                    where=where_filter,
                    limit=n_results,
                    include=["documents", "metadatas"]
                )
                # Add dummy distances
                results["distances"] = [[0.0] * len(results.get("documents", []))]

            # Convert results to MemorySearchResult objects
            search_results = []

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                # Calculate similarity score from distance
                similarity_score = None
                if i < len(distances):
                    # Convert distance to similarity (assuming cosine distance)
                    distance = distances[i]
                    similarity_score = 1.0 - distance if distance <= 2.0 else 0.0

                # Parse timestamp from metadata if available
                timestamp = None
                if metadata and "timestamp" in metadata:
                    try:
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(metadata["timestamp"])
                    except (ValueError, TypeError):
                        timestamp = None

                result = MemorySearchResult(
                    content=doc,
                    metadata=metadata or {},
                    similarity_score=similarity_score,
                    timestamp=timestamp
                )
                search_results.append(result)

            self.logger.debug(f"Vector search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            self.logger.error(f"Failed to search vector memory: {e}")
            return []

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if deletion was successful
        """
        await self._ensure_initialized()

        try:
            self._collection.delete(ids=[memory_id])
            self.logger.debug(f"Deleted vector memory: {memory_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete vector memory {memory_id}: {e}")
            return False

    async def clear(self) -> None:
        """Clear all vector memories."""
        await self._ensure_initialized()

        try:
            # Get all IDs
            all_results = self._collection.get(include=[])
            ids = all_results.get("ids", [])

            if ids:
                self._collection.delete(ids=ids)
                self.logger.info(f"Cleared {len(ids)} vector memories")
            else:
                self.logger.info("No vector memories to clear")

        except Exception as e:
            self.logger.error(f"Failed to clear vector memory: {e}")
            raise

    async def get(self, memory_id: str) -> Optional[MemorySearchResult]:
        """Get a memory by ID.

        Args:
            memory_id: ID of the memory

        Returns:
            The memory or None if not found
        """
        await self._ensure_initialized()

        try:
            results = self._collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"]
            )

            if not results.get("documents"):
                return None

            content = results["documents"][0]
            metadata = results["metadatas"][0] if results.get("metadatas") else {}

            # Parse timestamp
            timestamp = None
            if metadata and "timestamp" in metadata:
                try:
                    from datetime import datetime
                    timestamp = datetime.fromisoformat(metadata["timestamp"])
                except (ValueError, TypeError):
                    timestamp = None

            return MemorySearchResult(
                content=content,
                metadata=metadata,
                similarity_score=None,  # No similarity score for direct get
                timestamp=timestamp
            )

        except Exception as e:
            self.logger.error(f"Failed to get vector memory {memory_id}: {e}")
            return None

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.

        Returns:
            Collection information
        """
        await self._ensure_initialized()

        try:
            # Get count
            count = self._collection.count()

            return {
                "collection_name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    async def optimize(self) -> None:
        """Optimize the vector database."""
        self.logger.info("Vector memory optimization not implemented for ChromaDB")
        # ChromaDB handles optimization internally
        # In a production system, you might want to add periodic maintenance tasks