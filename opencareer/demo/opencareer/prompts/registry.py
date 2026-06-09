"""
Prompt Registry for OpenCareer.

This module provides a centralized registry for loading, caching, and
rendering YAML-based prompt templates. It supports dot-notation key
access and string.Template variable substitution.
"""

import logging
import string
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Directory where YAML prompt files are stored
PROMPTS_DIR = Path(__file__).parent.resolve()


class PromptRegistry:
    """Registry for managing YAML-based prompt templates.

    Loads YAML files from the prompts/ directory, caches them,
    and provides dot-notation access to nested keys with
    string.Template rendering support.
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        self._prompts_dir = prompts_dir or PROMPTS_DIR
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._file_index: Dict[str, str] = {}  # dotted_prefix -> filename
        self._logger = logging.getLogger("prompt.registry")

    # ---------------------------------------------------------------
    # Loading / indexing
    # ---------------------------------------------------------------

    def _build_file_index(self) -> Dict[str, str]:
        """Walk prompts/ and build a map: dotted_prefix -> filename.

        For example, ``dialogues/handoff.yaml`` becomes the prefix
        ``dialogues.handoff`` so that keys inside that file are
        addressable as ``dialogues.handoff.first_turn_greeting``.
        """
        index: Dict[str, str] = {}
        if not self._prompts_dir.exists():
            self._logger.warning("Prompts directory not found: %s", self._prompts_dir)
            return index

        for yaml_path in sorted(self._prompts_dir.rglob("*.yaml")):
            relative = yaml_path.relative_to(self._prompts_dir)
            # strip suffix and convert path separators to dots
            prefix = str(relative.with_suffix("")).replace("\\", ".").replace("/", ".")
            index[prefix] = str(yaml_path)

        self._logger.debug("Indexed %d YAML files", len(index))
        return index

    def _load_file(self, file_path: str) -> Dict[str, Any]:
        """Load a single YAML file and return its contents as a dict."""
        if file_path in self._cache:
            return self._cache[file_path]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._cache[file_path] = data
            self._logger.debug("Loaded: %s", file_path)
            return data
        except Exception as e:
            self._logger.error("Failed to load YAML file %s: %s", file_path, e)
            return {}

    def _resolve(self, dotted_key: str) -> Any:
        """Resolve a dotted key to its value.

        The first segment is used to find the YAML file, then each
        subsequent segment traverses the nested dict.

        Returns ``_MISSING`` sentinel when the key does not exist.
        """
        if not self._file_index:
            self._file_index = self._build_file_index()

        parts = dotted_key.split(".")
        # find the longest matching prefix in the file index
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in self._file_index:
                file_path = self._file_index[prefix]
                data = self._load_file(file_path)
                # traverse remaining segments inside the file
                remaining = parts[i:]
                current: Any = data
                for key in remaining:
                    if isinstance(current, dict):
                        current = current.get(key)
                        if current is None:
                            break
                    else:
                        current = None
                        break
                return current

        return _MISSING

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a value by its dotted key.

        Args:
            dotted_key: e.g. ``"dialogues.handoff.first_turn_greeting"``
            default: returned when the key is not found (default: ``None``)

        Returns:
            The value stored under *dotted_key*, or *default*.
        """
        value = self._resolve(dotted_key)
        if value is _MISSING:
            return default
        return value

    def render(self, dotted_key: str, default: Any = None, **variables: Any) -> str:
        """Get a template string and render it with ``string.Template``.

        Args:
            dotted_key: e.g. ``"dialogues.career_advice.learning_plan_template"``
            default: returned when the key is not found (default: ``None``)
            **variables: values to substitute into the template (``$var`` syntax)

        Returns:
            The rendered string, or *default* if the key is missing.
        """
        template_str = self.get(dotted_key, default)
        if template_str is _MISSING or template_str is None:
            return default

        if not isinstance(template_str, str):
            self._logger.warning(
                "Value at '%s' is not a string (type=%s), returning as-is",
                dotted_key, type(template_str).__name__,
            )
            return template_str

        if not variables:
            return template_str

        try:
            return string.Template(template_str).safe_substitute(**variables)
        except Exception as e:
            self._logger.error("Template rendering failed for '%s': %s", dotted_key, e)
            return template_str

    def get_all(self, dotted_prefix: str) -> Dict[str, Any]:
        """Get an entire sub-section of prompts.

        Useful for bulk-loading keyword lists or template collections.

        Args:
            dotted_prefix: e.g. ``"system.demand_analysis.job_keywords"``

        Returns:
            A dict (or list, or whatever YAML node is at that path).
        """
        return self.get(dotted_prefix, {})

    def reload(self) -> None:
        """Clear the in-memory cache and re-index files.

        Call this after modifying YAML files at runtime to pick up
        changes without restarting the application.
        """
        self._cache.clear()
        self._file_index = self._build_file_index()
        self._logger.info("Prompt registry reloaded")

    def clear_cache(self) -> None:
        """Clear the in-memory file cache (retains file index)."""
        self._cache.clear()

    @property
    def prompts_dir(self) -> Path:
        return self._prompts_dir


# Sentinel for missing values (since ``None`` is a valid YAML value)
_MISSING = object()

# -------------------------------------------------------------------
# Global singleton
# -------------------------------------------------------------------

_global_registry: Optional[PromptRegistry] = None


def get_global_registry() -> PromptRegistry:
    """Get the global PromptRegistry singleton.

    Creates the instance on first call; reuses it thereafter.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PromptRegistry()
    return _global_registry


def get_prompt(dotted_key: str, default: Any = None) -> Any:
    """Shortcut: get a prompt value from the global registry."""
    return get_global_registry().get(dotted_key, default)


def render_prompt(dotted_key: str, default: Any = None, **variables: Any) -> str:
    """Shortcut: render a prompt template from the global registry."""
    return get_global_registry().render(dotted_key, default, **variables)
