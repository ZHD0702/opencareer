"""
Resume Store — persistence layer for the Schema + Data architecture.

Saves and loads the full ``resume_data`` dict per user using the existing
structured memory backend (JSON or SQLite). This bridges the gap between
in-memory ``ConversationContext.user_profile["resume_data"]`` and
persistent storage that survives session restarts.

Architecture::

    Schema (resume_schema.py)       → field definitions, validation rules
    Data  (ctx.user_profile)        → per-user values (in-memory)
    Store (this file)               → save/load from structured memory
    Extraction (LLM prompt)         → passive extraction from conversation

Usage::

    store = ResumeStore(structured_memory)
    await store.save(user_id, resume_data)
    data = await store.load(user_id)  # None if no saved data
"""

import json
import logging
from typing import Any, Dict, Optional

from .memory_manager import BaseMemory, MemoryQuery
from .resume_schema import new_empty_resume_data

logger = logging.getLogger("memory.resume_store")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESUME_STORE_TYPE = "resume_data"
"""Value of the ``type`` metadata field used to identify resume records."""


class ResumeStore:
    """Persistence layer for resume data backed by structured memory.

    Each user has exactly one resume record, identified by metadata
    ``{"user_id": <id>, "type": "resume_data"}``.  ``save()`` performs
    an upsert — it finds and replaces the existing record if present,
    otherwise creates a new one.

    Args:
        structured_memory: A ``BaseMemory`` instance (JSON or SQLite).
    """

    def __init__(self, structured_memory: BaseMemory) -> None:
        self._memory = structured_memory
        self._logger = logging.getLogger("memory.resume_store")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save(self, user_id: str, resume_data: Dict[str, Any]) -> None:
        """Upsert *resume_data* for *user_id*.

        If a record already exists for this user it is replaced in full;
        otherwise a new record is created.

        Args:
            user_id: Unique user identifier.
            resume_data: The full ``resume_data`` dict (all schema keys).
        """
        content = json.dumps(
            resume_data, ensure_ascii=False, indent=2, default=str
        )
        metadata: Dict[str, Any] = {
            "user_id": user_id,
            "type": RESUME_STORE_TYPE,
        }

        existing_id = await self._find_record_id(user_id)
        if existing_id is not None:
            await self._memory.update(
                existing_id, content=content, metadata=metadata
            )
            self._logger.info(
                "Updated resume_data for user=%s (record=%s)", user_id, existing_id
            )
        else:
            new_id = await self._memory.store(content=content, metadata=metadata)
            self._logger.info(
                "Created resume_data for user=%s (record=%s)", user_id, new_id
            )

    async def load(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load previously saved resume data for *user_id*.

        Returns:
            The full ``resume_data`` dict, or ``None`` if no saved data exists.
            Missing keys are filled from the schema template to guarantee
            that all expected keys are present.
        """
        query = MemoryQuery(
            metadata_filter={"user_id": user_id, "type": RESUME_STORE_TYPE},
            limit=1,
        )
        results = await self._memory.search(query)
        if not results:
            self._logger.debug("No saved resume_data for user=%s", user_id)
            return None

        raw = results[0].content
        try:
            parsed: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._logger.error(
                "Failed to parse resume_data for user=%s: %s", user_id, exc
            )
            return None

        # Merge into schema template so missing keys (schema evolution) get defaults
        template = new_empty_resume_data()
        template.update(parsed)
        self._logger.debug("Loaded resume_data for user=%s (%d keys)", user_id, len(template))
        return template

    async def delete(self, user_id: str) -> bool:
        """Delete saved resume data for *user_id*.

        Returns:
            ``True`` if a record existed and was deleted, ``False`` otherwise.
        """
        existing_id = await self._find_record_id(user_id)
        if existing_id is None:
            self._logger.debug("No resume_data to delete for user=%s", user_id)
            return False

        ok = await self._memory.delete(existing_id)
        if ok:
            self._logger.info("Deleted resume_data for user=%s", user_id)
        else:
            self._logger.warning("Failed to delete resume_data for user=%s", user_id)
        return ok

    async def exists(self, user_id: str) -> bool:
        """Check whether resume data has been saved for *user_id*.

        This is more efficient than ``load()`` when you only need to know
        *if* data exists, not the data itself.
        """
        existing_id = await self._find_record_id(user_id)
        return existing_id is not None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _find_record_id(self, user_id: str) -> Optional[str]:
        """Return the structured memory ID for *user_id*'s resume record.

        Uses ``source_id`` populated on ``MemorySearchResult`` (added by
        the ``source_id`` field).  Returns ``None`` when no record exists.
        """
        query = MemoryQuery(
            metadata_filter={"user_id": user_id, "type": RESUME_STORE_TYPE},
            limit=1,
        )
        results = await self._memory.search(query)
        if results:
            sid = results[0].source_id
            return sid if sid else None
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_resume_store(
    structured_memory: Optional[BaseMemory] = None,
    data_dir: str = "./data/resume_data",
) -> ResumeStore:
    """Create a ``ResumeStore`` backed by structured memory.

    Args:
        structured_memory: A ``BaseMemory`` instance.  If ``None``, a
            default ``JSONStructuredMemory`` rooted at *data_dir* is created.
        data_dir: Ignored when *structured_memory* is provided.

    Returns:
        A ready-to-use ``ResumeStore``.
    """
    if structured_memory is not None:
        return ResumeStore(structured_memory)

    # Lazy import to avoid circular dependency at module level
    from .structured_memory import JSONStructuredMemory

    return ResumeStore(JSONStructuredMemory(data_dir=data_dir))
