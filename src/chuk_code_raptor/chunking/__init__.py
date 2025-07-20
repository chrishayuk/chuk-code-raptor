# src/chuk_code_raptor/chunking/__init__.py
"""
Clean Chunking System
=====================

Modern tree-sitter based code chunking with semantic understanding.
Clean, forward-looking architecture for RAG and AI applications.

Key Components:
- ChunkingEngine: Main interface for chunking operations
- BaseParser: Base class for all parsers
- TreeSitterParser: AST-based semantic parsing
- HeuristicParser: Pattern-based parsing fallback
- ChunkingConfig: Rich configuration system
- SemanticChunk: Unified chunk model with semantic information

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
        PRECISE_CONFIG
    )
    
    # Use predefined configuration
    engine = ChunkingEngine(PRECISE_CONFIG)
    
    # Check supported languages
    print(f"Supported: {engine.get_supported_languages()}")
    
    # Chunk content directly
    chunks = engine.chunk_content(code_string, "python", "file.py")
    
    # Access semantic information
    for chunk in chunks:
        print(f"Tags: {[tag.name for tag in chunk.semantic_tags]}")
        print(f"Dependencies: {chunk.dependencies}")
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

# Specialized parser bases
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

# Check parser availability (RUST REMOVED TO AVOID VERSION CONFLICTS)
PARSERS_AVAILABLE = {}

def _check_parser_availability():
    """Check which parsers are available"""
    parsers_to_check = [
        ('python', 'chuk_code_raptor.chunking.parsers.python', 'PythonParser'),
        ('markdown', 'chuk_code_raptor.chunking.parsers.markdown', 'MarkdownParser'),
        ('javascript', 'chuk_code_raptor.chunking.parsers.javascript', 'JavaScriptParser'),
        ('json', 'chuk_code_raptor.chunking.parsers.json', 'JSONParser'),
        ('html', 'chuk_code_raptor.chunking.parsers.html', 'HTMLParser'),
        # RUST TEMPORARILY REMOVED DUE TO TREE-SITTER VERSION INCOMPATIBILITY
        # ('rust', 'chuk_code_raptor.chunking.parsers.rust', 'RustParser'),
    ]
    
    available = {}
    
    for language, module_path, class_name in parsers_to_check:
        try:
            module = __import__(module_path, fromlist=[class_name])
            parser_class = getattr(module, class_name)
            # Try to create instance to verify dependencies
            from .config import ChunkingConfig
            test_instance = parser_class(ChunkingConfig())
            available[language] = {
                'module': module_path,
                'class': class_name,
                'parser_type': getattr(test_instance, 'parser_type', 'unknown')
            }
            logger.debug(f"Parser available: {language} ({test_instance.parser_type})")
        except ImportError as e:
            logger.debug(f"Parser {language} not available: missing dependencies ({e})")
        except Exception as e:
            logger.debug(f"Parser {language} failed initialization: {e}")
    
    return available

# Check availability on import
PARSERS_AVAILABLE = _check_parser_availability()

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
    
    # Convenience functions
    'create_engine',
    'chunk_file',
    'chunk_content',
    'get_supported_languages',
    'get_supported_extensions',
    'get_parser_info',
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
        List of SemanticChunk objects
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
        List of SemanticChunk objects
    """
    engine = create_engine(config)
    return engine.chunk_content(content, language, file_path)

def get_supported_languages() -> list[str]:
    """Get list of all supported programming languages."""
    engine = create_engine()
    return engine.get_supported_languages()

def get_supported_extensions() -> list[str]:
    """Get list of all supported file extensions."""
    engine = create_engine()
    return engine.get_supported_extensions()

def get_parser_info() -> dict:
    """Get information about available parsers."""
    return {
        'available_parsers': PARSERS_AVAILABLE,
        'total_parsers': len(PARSERS_AVAILABLE),
        'parser_types': list(set(info['parser_type'] for info in PARSERS_AVAILABLE.values())),
        'supported_languages': get_supported_languages(),
    }

# Auto-initialization and status
def _log_initialization_status():
    """Log initialization status"""
    if PARSERS_AVAILABLE:
        available_langs = list(PARSERS_AVAILABLE.keys())
        parser_types = list(set(info['parser_type'] for info in PARSERS_AVAILABLE.values()))
        
        logger.info(f"Clean chunking system initialized")
        logger.info(f"Available parsers: {len(PARSERS_AVAILABLE)} ({', '.join(available_langs)})")
        logger.info(f"Parser types: {', '.join(parser_types)}")
    else:
        logger.warning("No parsers available - check tree-sitter dependencies")

# Initialize on import
_log_initialization_status()