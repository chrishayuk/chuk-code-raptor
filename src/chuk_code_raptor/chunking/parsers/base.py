# src/chuk_code_raptor/chunking/parsers/base.py
"""
Base parsing context and utilities for chunking.
"""

from dataclasses import dataclass
from typing import Optional
from ..semantic_chunk import ContentType

@dataclass
class ParseContext:
    """Context for parsing operations"""
    file_path: str
    language: str
    content_type: ContentType
    max_chunk_size: int
    min_chunk_size: int
    enable_semantic_analysis: bool = True
    enable_dependency_tracking: bool = True