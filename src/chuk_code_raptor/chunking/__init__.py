# src/chuk_code_raptor/chunking/__init__.py
"""
Clean Chunking System - Pure Registry Edition
=============================================

Modern tree-sitter based code chunking with semantic understanding.
Everything is configured through YAML - no hardcoding anywhere.

Key Components:
- ChunkingEngine: Main interface for chunking operations
- ParserRegistry: YAML-based parser management system
- BaseParser: Base class for all parsers
- TreeSitterParser: AST-based semantic parsing
- HeuristicParser: Pattern-based parsing for unsupported languages
- ChunkingConfig: Rich configuration system
- SemanticChunk: Unified chunk model with semantic information

Basic Usage:
    from chuk_code_raptor.chunking import ChunkingEngine, ChunkingConfig
    
    config = ChunkingConfig(target_chunk_size=800)
    engine = ChunkingEngine(config)
    
    chunks = engine.chunk_file("example.py")
    for chunk in chunks:
        print(f"{chunk.chunk_type.value}: {chunk.content_preview}")

Registry Usage:
    from chuk_code_raptor.chunking import get_registry
    
    registry = get_registry()
    print(f"Config: {registry.config_path}")
    print(f"Languages: {registry.get_supported_languages()}")
"""

import logging

logger = logging.getLogger(__name__)

# Core configuration and enums
from .config import (
    ChunkingConfig,
    ChunkingStrategy,
    DEFAULT_CONFIG,
    FAST_CONFIG,
    PRECISE_CONFIG,
    SEMANTIC_CONFIG,
    HYBRID_CONFIG,
    LARGE_FILES_CONFIG,
    DOCUMENT_CONFIG
)

# Base classes and interfaces
from .base import (
    BaseParser,
    ParseContext,
    ParserError,
    UnsupportedLanguageError,
    InvalidContentError
)

# Specialized parser bases (always available)
from .tree_sitter_base import TreeSitterParser
from .heuristic_base import HeuristicParser

# Semantic chunk model
from .semantic_chunk import (
    SemanticChunk,
    SemanticTag,
    ChunkRelationship,
    ContentType,
    create_chunk_id,
    calculate_chunk_similarity,
    find_related_chunks
)

# Registry system (always available)
from .registry import (
    ParserRegistry,
    get_registry,
    reload_registry,
    register_custom_parser
)

# Main engine
from .engine import ChunkingEngine

# Version info
try:
    from importlib.metadata import version, metadata
    __version__ = version("chuk_code_raptor")
    pkg_metadata = metadata("chuk_code_raptor")
    __author__ = pkg_metadata.get("Author", "CodeRaptor Team")
except ImportError:
    try:
        import pkg_resources
        __version__ = pkg_resources.get_distribution("chuk_code_raptor").version
        __author__ = "CodeRaptor Team"
    except Exception:
        __version__ = "unknown"
        __author__ = "CodeRaptor Team"
except Exception:
    __version__ = "development"
    __author__ = "CodeRaptor Team"

# Main exports
__all__ = [
    # Main interface
    'ChunkingEngine',
    
    # Configuration
    'ChunkingConfig',
    'ChunkingStrategy',
    'DEFAULT_CONFIG',
    'FAST_CONFIG', 
    'PRECISE_CONFIG',
    'SEMANTIC_CONFIG',
    'HYBRID_CONFIG',
    'LARGE_FILES_CONFIG',
    'DOCUMENT_CONFIG',
    
    # Base classes
    'BaseParser',
    'TreeSitterParser',
    'HeuristicParser',
    'ParseContext',
    
    # Semantic model
    'SemanticChunk',
    'SemanticTag',
    'ChunkRelationship',
    'ContentType',
    'create_chunk_id',
    'calculate_chunk_similarity',
    'find_related_chunks',
    
    # Exceptions
    'ParserError',
    'UnsupportedLanguageError',
    'InvalidContentError',
    
    # Registry system
    'ParserRegistry',
    'get_registry',
    'reload_registry', 
    'register_custom_parser',
    
    # Convenience functions
    'create_engine',
    'chunk_file',
    'chunk_content',
    'get_supported_languages',
    'get_supported_extensions',
    'get_parser_info',
    'get_installation_help',
]

# Convenience functions for common use cases
def create_engine(config: ChunkingConfig = None) -> ChunkingEngine:
    """Create a ChunkingEngine with optional configuration."""
    return ChunkingEngine(config or DEFAULT_CONFIG)

def chunk_file(file_path: str, language: str = None, config: ChunkingConfig = None):
    """Convenience function to chunk a file with default settings."""
    engine = create_engine(config)
    return engine.chunk_file(file_path, language)

def chunk_content(content: str, language: str, file_path: str = "unknown", 
                 config: ChunkingConfig = None):
    """Convenience function to chunk content with default settings."""
    engine = create_engine(config)
    return engine.chunk_content(content, language, file_path)

def get_supported_languages() -> list[str]:
    """Get list of all supported programming languages."""
    registry = get_registry()
    return registry.get_supported_languages()

def get_supported_extensions() -> list[str]:
    """Get list of all supported file extensions."""
    registry = get_registry()
    return registry.get_supported_extensions()

def get_parser_info() -> dict:
    """Get information about available parsers."""
    registry = get_registry()
    stats = registry.get_parser_stats()
    return {
        'total_parsers': stats['total_parsers'],
        'available_parsers': stats['available_parsers'],
        'supported_languages': len(registry.get_supported_languages()),
        'supported_extensions': len(registry.get_supported_extensions()),
        'parser_types': stats['parser_types'],
        'package_availability': stats['package_availability'],
        'config_file': str(registry.config_path)
    }

def get_installation_help() -> str:
    """Get help for installing tree-sitter packages."""
    registry = get_registry()
    return registry.get_installation_help()

# Auto-initialization and status
def _log_initialization_status():
    """Log initialization status"""
    try:
        registry = get_registry()
        stats = registry.get_parser_stats()
        
        logger.info(f"Clean chunking system initialized with YAML registry")
        logger.info(f"Configuration file: {registry.config_path}")
        logger.info(f"Total parsers: {stats['total_parsers']}")
        logger.info(f"Available parsers: {stats['available_parsers']}")
        logger.info(f"Supported languages: {stats['supported_languages']}")
        logger.info(f"Supported extensions: {stats['supported_extensions']}")
        
        if stats['available_parsers'] == 0:
            logger.warning("No parsers available - install tree-sitter packages")
            logger.info("💡 Install tree-sitter packages for full functionality")
        
    except Exception as e:
        logger.debug(f"Error during initialization logging: {e}")
        logger.warning("Parser status check failed - some parsers may not be available")

# Initialize on import
_log_initialization_status()