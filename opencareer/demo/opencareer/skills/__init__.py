"""
SKILLs module for OpenCareer.

Contains all SKILLs that implement career assistance functionality.
All functions (except core dialogue and memory) must be implemented as SKILLs.

Skills are auto-discovered: any subdirectory containing a skill.py file
is automatically imported. No manual registration needed when adding a new skill.
"""

import importlib
import logging
import pkgutil
from pathlib import Path

__all__ = []

_logger = logging.getLogger("skills")

# Auto-discover all skill subdirectories with skill.py
_skills_dir = Path(__file__).parent
for _entry in sorted(_skills_dir.iterdir()):
    if _entry.is_dir() and not _entry.name.startswith("__"):
        _skill_file = _entry / "skill.py"
        if _skill_file.exists():
            _module_name = _entry.name
            try:
                _module = importlib.import_module(f".{_module_name}.skill", __package__)
                globals()[_module_name] = _module
                __all__.append(_module_name)
                _logger.debug(f"Auto-discovered skill: {_module_name}")
            except Exception as e:
                _logger.warning(f"Failed to load skill '{_module_name}': {e}")

# Also auto-discover __init__.py packages (for skills with custom package structure)
for _importer, _module_name, _ispkg in pkgutil.iter_modules([str(_skills_dir)]):
    if _module_name.startswith("__") or _module_name in __all__:
        continue
    try:
        _module = importlib.import_module(f".{_module_name}", __package__)
        globals()[_module_name] = _module
        __all__.append(_module_name)
        _logger.debug(f"Auto-discovered skill package: {_module_name}")
    except Exception as e:
        _logger.warning(f"Failed to load skill package '{_module_name}': {e}")