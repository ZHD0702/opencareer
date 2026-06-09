"""
Structured memory implementation using JSON files.

This module provides structured memory storage using JSON files with
optional SQLite backend for production use.
"""

import json
import logging
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .memory_manager import BaseMemory, MemoryQuery, MemorySearchResult


class JSONStructuredMemory(BaseMemory):
    """Structured memory implementation using JSON files."""

    def __init__(self, data_dir: str = "./data/structured_memory"):
        """Initialize JSON structured memory.

        Args:
            data_dir: Directory to store JSON files
        """
        super().__init__("json_structured")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._data_file = self.data_dir / "memories.json"
        self._memories: Dict[str, Dict[str, Any]] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load data from JSON file."""
        try:
            if self._data_file.exists():
                with open(self._data_file, 'r', encoding='utf-8') as f:
                    self._memories = json.load(f)
                self.logger.info(f"Loaded {len(self._memories)} memories from JSON")
            else:
                self._memories = {}
                self.logger.info("No existing JSON data found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load JSON data: {e}")
            self._memories = {}

    def _save_data(self) -> None:
        """Save data to JSON file."""
        try:
            # Create backup if file exists
            if self._data_file.exists():
                backup_file = self._data_file.with_suffix('.json.backup')
                self._data_file.rename(backup_file)

            # Save new data
            with open(self._data_file, 'w', encoding='utf-8') as f:
                json.dump(self._memories, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Saved {len(self._memories)} memories to JSON")
        except Exception as e:
            self.logger.error(f"Failed to save JSON data: {e}")
            raise

    async def store(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Store content in structured memory.

        Args:
            content: The content to store
            metadata: Additional metadata

        Returns:
            ID of the stored memory
        """
        if metadata is None:
            metadata = {}

        memory_id = str(uuid.uuid4())

        # Create memory record
        memory_record = {
            "content": content,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
            "memory_id": memory_id
        }

        # Store in memory
        self._memories[memory_id] = memory_record

        # Save to disk
        self._save_data()

        self.logger.debug(f"Stored structured memory, id: {memory_id}")
        return memory_id

    async def search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """Search for memories matching the query.

        Args:
            query: The search query

        Returns:
            List of search results
        """
        results = []

        for memory_id, record in self._memories.items():
            # Apply metadata filter if specified
            if query.metadata_filter:
                matches_filter = True
                for key, expected_value in query.metadata_filter.items():
                    actual_value = record["metadata"].get(key)
                    if actual_value != expected_value:
                        matches_filter = False
                        break
                if not matches_filter:
                    continue

            # Apply text search if specified
            if query.text:
                # Simple substring search for JSON implementation
                # In a real system, you'd use full-text search
                if query.text.lower() not in record["content"].lower():
                    continue

            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(record["timestamp"])
            except (ValueError, TypeError):
                timestamp = None

            # Create result
            result = MemorySearchResult(
                content=record["content"],
                metadata=record["metadata"],
                similarity_score=None,  # No similarity score for structured search
                timestamp=timestamp,
                source_id=record["memory_id"],
            )
            results.append(result)

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.timestamp if x.timestamp else datetime.min,
                    reverse=True)

        # Apply limit
        results = results[:query.limit]

        self.logger.debug(f"Structured search returned {len(results)} results")
        return results

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if deletion was successful
        """
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._save_data()
            self.logger.debug(f"Deleted structured memory: {memory_id}")
            return True
        else:
            self.logger.warning(f"Memory not found for deletion: {memory_id}")
            return False

    async def clear(self) -> None:
        """Clear all structured memories."""
        self._memories.clear()
        self._save_data()
        self.logger.info("Cleared all structured memories")

    async def get(self, memory_id: str) -> Optional[MemorySearchResult]:
        """Get a memory by ID.

        Args:
            memory_id: ID of the memory

        Returns:
            The memory or None if not found
        """
        record = self._memories.get(memory_id)
        if not record:
            return None

        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(record["timestamp"])
        except (ValueError, TypeError):
            timestamp = None

        return MemorySearchResult(
            content=record["content"],
            metadata=record["metadata"],
            similarity_score=None,
            timestamp=timestamp
        )

    async def update_metadata(self, memory_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """Update metadata for a memory.

        Args:
            memory_id: ID of the memory
            metadata_updates: Metadata updates to apply

        Returns:
            True if update was successful
        """
        if memory_id not in self._memories:
            return False

        record = self._memories[memory_id]
        record["metadata"].update(metadata_updates)
        record["timestamp"] = datetime.now().isoformat()  # Update timestamp

        self._save_data()
        self.logger.debug(f"Updated metadata for memory: {memory_id}")
        return True


class SQLiteStructuredMemory(BaseMemory):
    """Structured memory implementation using SQLite."""

    def __init__(self, db_path: str = "./data/opencareer.db"):
        """Initialize SQLite structured memory.

        Args:
            db_path: Path to SQLite database file
        """
        super().__init__("sqlite_structured")
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create memories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')

            # Create index for metadata filtering
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp
                ON memories(timestamp)
            ''')

            conn.commit()
            conn.close()

            self.logger.info(f"SQLite database initialized: {self.db_path}")

        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite database: {e}")
            raise

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def store(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Store content in structured memory.

        Args:
            content: The content to store
            metadata: Additional metadata

        Returns:
            ID of the stored memory
        """
        if metadata is None:
            metadata = {}

        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO memories (id, content, metadata, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                memory_id,
                content,
                json.dumps(metadata),
                timestamp,
                timestamp
            ))

            conn.commit()
            conn.close()

            self.logger.debug(f"Stored structured memory in SQLite, id: {memory_id}")
            return memory_id

        except Exception as e:
            self.logger.error(f"Failed to store content in SQLite: {e}")
            raise

    async def search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """Search for memories matching the query.

        Args:
            query: The search query

        Returns:
            List of search results
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build WHERE clause
            where_clauses = []
            params = []

            if query.metadata_filter:
                # SQLite doesn't easily support JSON querying
                # For simplicity, we'll do filtering in Python
                pass

            if query.text:
                # Simple full-text search
                where_clauses.append("content LIKE ?")
                params.append(f"%{query.text}%")

            # Build query
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            sql = f'''
                SELECT id, content, metadata, timestamp
                FROM memories
                WHERE {where_sql}
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            params.append(query.limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()

            # Process results
            results = []

            for row in rows:
                # Parse metadata
                try:
                    metadata = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    metadata = {}

                # Apply metadata filter in Python if needed
                if query.metadata_filter:
                    matches_filter = True
                    for key, expected_value in query.metadata_filter.items():
                        if metadata.get(key) != expected_value:
                            matches_filter = False
                            break
                    if not matches_filter:
                        continue

                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(row["timestamp"])
                except (ValueError, TypeError):
                    timestamp = None

                result = MemorySearchResult(
                    content=row["content"],
                    metadata=metadata,
                    similarity_score=None,
                    timestamp=timestamp,
                    source_id=row["id"],
                )
                results.append(result)

            self.logger.debug(f"SQLite search returned {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Failed to search SQLite memory: {e}")
            return []

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if deletion was successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            success = deleted_count > 0
            if success:
                self.logger.debug(f"Deleted structured memory from SQLite: {memory_id}")
            else:
                self.logger.warning(f"Memory not found in SQLite: {memory_id}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to delete memory from SQLite: {e}")
            return False

    async def clear(self) -> None:
        """Clear all structured memories."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM memories")
            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            self.logger.info(f"Cleared {deleted_count} structured memories from SQLite")

        except Exception as e:
            self.logger.error(f"Failed to clear SQLite memory: {e}")
            raise

    async def get(self, memory_id: str) -> Optional[MemorySearchResult]:
        """Get a memory by ID.

        Args:
            memory_id: ID of the memory

        Returns:
            The memory or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT content, metadata, timestamp FROM memories WHERE id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            # Parse metadata
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                metadata = {}

            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(row["timestamp"])
            except (ValueError, TypeError):
                timestamp = None

            return MemorySearchResult(
                content=row["content"],
                metadata=metadata,
                similarity_score=None,
                timestamp=timestamp
            )

        except Exception as e:
            self.logger.error(f"Failed to get memory from SQLite: {e}")
            return None

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Database statistics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get count
            cursor.execute("SELECT COUNT(*) as count FROM memories")
            count = cursor.fetchone()["count"]

            # Get oldest and newest timestamps
            cursor.execute("SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest FROM memories")
            timestamp_row = cursor.fetchone()

            conn.close()

            return {
                "total_memories": count,
                "oldest_timestamp": timestamp_row["oldest"],
                "newest_timestamp": timestamp_row["newest"],
                "db_path": self.db_path
            }

        except Exception as e:
            self.logger.error(f"Failed to get SQLite stats: {e}")
            return {"error": str(e)}


# Factory function to create structured memory based on configuration
def create_structured_memory(
    backend: str = "json",
    **kwargs
) -> BaseMemory:
    """Create a structured memory instance.

    Args:
        backend: Backend to use ("json" or "sqlite")
        **kwargs: Additional arguments for the backend

    Returns:
        Structured memory instance
    """
    if backend == "json":
        return JSONStructuredMemory(**kwargs)
    elif backend == "sqlite":
        return SQLiteStructuredMemory(**kwargs)
    else:
        raise ValueError(f"Unsupported structured memory backend: {backend}")