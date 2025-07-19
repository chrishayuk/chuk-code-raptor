# chuk_code_raptor/chunking/__init__.py
"""
Chunking System
===============

Modern AST-based code chunking using tree-sitter parsers.
Provides semantic code understanding for RAG and AI applications.

Key Components:
- ChunkingEngine: Main interface for chunking operations
- BaseChunker: Abstract base for all chunkers
- ChunkerRegistry: Manages available chunkers
- ChunkingConfig: Configuration for chunking behavior
- Tree-sitter chunkers: Language-specific AST-based chunkers

Basic Usage:
    from chuk_code_raptor.chunking import ChunkingEngine, ChunkingConfig
    
    config = ChunkingConfig(target_chunk_size=800)
    engine = ChunkingEngine(config)
    
    chunks = engine.chunk_file("example.py")
    for chunk in chunks:
        print(f"{chunk.chunk_type.value}: {chunk.content_preview}")

Advanced Usage:
    from chuk_code_raptor.chunking import (
        ChunkingEngine, 
        ChunkingConfig,
        ChunkingStrategy,
        get_registry
    )
    
    # Custom configuration
    config = ChunkingConfig(
        target_chunk_size=1000,
        preserve_atomic_nodes=True,
        primary_strategy=ChunkingStrategy.STRUCTURAL
    )
    
    engine = ChunkingEngine(config)
    
    # Check supported languages
    print(f"Supported: {engine.get_supported_languages()}")
    
    # Chunk content directly
    chunks = engine.chunk_content(code_string, "python", "file.py")
"""

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
    BaseChunker,
    ChunkerError,
    UnsupportedLanguageError,
    InvalidContentError
)

# Main engine
from .engine import ChunkingEngine

# Registry for managing chunkers
from .registry import (
    ChunkerRegistry,
    get_registry,
    register_chunker,
    get_chunker,
    set_global_config
)

# Tree-sitter chunking system (auto-registers chunkers)
try:
    from . import tree_sitter_chunker
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Version info - read from pyproject.toml
try:
    from importlib.metadata import version, metadata
    __version__ = version("chuk_code_raptor")
    
    # Get author from package metadata
    pkg_metadata = metadata("chuk_code_raptor")
    __author__ = pkg_metadata.get("Author", "CodeRaptor Team")
    
except ImportError:
    # Fallback for Python < 3.8
    try:
        import pkg_resources
        __version__ = pkg_resources.get_distribution("chuk_code_raptor").version
        __author__ = "CodeRaptor Team"
    except Exception:
        __version__ = "unknown"
        __author__ = "CodeRaptor Team"
except Exception:
    # Fallback if package not installed or other issues
    __version__ = "development"
    __author__ = "CodeRaptor Team"

# Main exports - what users typically import
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
    
    # Base classes (for extending)
    'BaseChunker',
    
    # Registry functions
    'get_registry',
    'register_chunker',
    'get_chunker',
    'set_global_config',
    
    # Exceptions
    'ChunkerError',
    'UnsupportedLanguageError',
    'InvalidContentError',
]

# Convenience functions for common use cases
def create_engine(config: ChunkingConfig = None) -> ChunkingEngine:
    """
    Create a ChunkingEngine with optional configuration.
    
    Args:
        config: ChunkingConfig to use (uses DEFAULT_CONFIG if None)
        
    Returns:
        Configured ChunkingEngine instance
    """
    return ChunkingEngine(config or DEFAULT_CONFIG)

def chunk_file(file_path: str, language: str = None, config: ChunkingConfig = None):
    """
    Convenience function to chunk a file with default settings.
    
    Args:
        file_path: Path to file to chunk
        language: Programming language (auto-detected if None)
        config: ChunkingConfig to use (uses DEFAULT_CONFIG if None)
        
    Returns:
        List of CodeChunk objects
    """
    engine = create_engine(config)
    return engine.chunk_file(file_path, language)

def chunk_content(content: str, language: str, file_path: str = "unknown", 
                 config: ChunkingConfig = None):
    """
    Convenience function to chunk content with default settings.
    
    Args:
        content: Content to chunk
        language: Programming language
        file_path: Source file path (for metadata)
        config: ChunkingConfig to use (uses DEFAULT_CONFIG if None)
        
    Returns:
        List of CodeChunk objects
    """
    engine = create_engine(config)
    return engine.chunk_content(content, language, file_path)

def get_supported_languages() -> list[str]:
    """Get list of all supported programming languages."""
    return get_registry().get_available_languages()

def get_supported_extensions() -> list[str]:
    """Get list of all supported file extensions."""
    return get_registry().get_available_extensions()

# Auto-initialization
def _initialize_chunking_system():
    """Initialize the chunking system on import."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        registry = get_registry()
        languages = registry.get_available_languages()
        extensions = registry.get_available_extensions()
        
        logger.debug(f"Chunking system initialized with {len(languages)} languages: {languages}")
        logger.debug(f"Supporting {len(extensions)} extensions: {extensions}")
        
        if TREE_SITTER_AVAILABLE:
            logger.debug("Tree-sitter chunking available")
        else:
            logger.debug("Tree-sitter chunking not available")
        
    except Exception as e:
        logger.warning(f"Error initializing chunking system: {e}")

# Initialize on import
_initialize_chunking_system()