"""
Extraction package for OpenCareer.

Passive information extraction from natural conversation, using LLM to
identify and record resume-related information without active interrogation.
"""

from .resume_extractor import extract_resume_fields

__all__ = [
    "extract_resume_fields",
]
