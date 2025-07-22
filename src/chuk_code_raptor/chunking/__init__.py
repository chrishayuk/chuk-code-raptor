# src/chuk_code_raptor/chunking/__init__.py
"""
Clean Chunking System - Practical Edition
==========================================

Modern tree-sitter based code chunking with semantic understanding.
Uses practical parser discovery that automatically adapts to available packages.

Key Components:
- ChunkingEngine: Main interface for chunking operations
- BaseParser: Base class for all parsers
- TreeSitterParser: AST-based semantic parsing
- AvailableTreeSitterParser: Practical parser using available packages
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

Package Requirements:
    For full language support including LaTeX:
        pip install tree-sitter-languages
    
    Alternative comprehensive package:
        pip install tree-sitter-language-pack
    
    Individual packages (LaTeX not available):
        pip install tree-sitter-python tree-sitter-javascript
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

# Try to import heuristic base if available
try:
    from .heuristic_base import HeuristicParser
except ImportError:
    logger.debug("HeuristicParser not available")
    HeuristicParser = None

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

# Use practical parser discovery
def _get_parser_availability():
    """Get parser availability using practical discovery"""
    try:
        from .parsers import discover_available_parsers, get_installation_help
        return discover_available_parsers()
    except ImportError as e:
        logger.debug(f"Parser discovery not available: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Parser discovery failed: {e}")
        return {}

# Get available parsers
PARSERS_AVAILABLE = _get_parser_availability()

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
    'get_installation_help',
]

# Add HeuristicParser to exports if available
if HeuristicParser is not None:
    __all__.append('HeuristicParser')

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
    working_parsers = {k: v for k, v in PARSERS_AVAILABLE.items() 
                      if k != '_package_info' and isinstance(v, dict) and v.get('status') == 'available'}
    
    package_info = PARSERS_AVAILABLE.get('_package_info', {})
    
    return {
        'available_parsers': working_parsers,
        'total_parsers': len(working_parsers),
        'parser_types': list(set(info.get('parser_type', 'unknown') for info in working_parsers.values())),
        'supported_languages': get_supported_languages(),
        'package_info': package_info,
    }

def get_installation_help() -> str:
    """Get help for installing tree-sitter packages."""
    try:
        from .parsers import get_installation_help as get_help_impl
        return get_help_impl()
    except ImportError:
        return """
🌲 Tree-sitter Package Installation:

Install comprehensive package for full language support:
    pip install tree-sitter-languages

Alternative comprehensive package:
    pip install tree-sitter-language-pack

Individual packages (limited language support):
    pip install tree-sitter-python tree-sitter-javascript

Note: LaTeX requires a comprehensive package.
"""
    except Exception as e:
        logger.debug(f"Error getting installation help: {e}")
        return "Installation help not available. Try: pip install tree-sitter-languages"

# Auto-initialization and status
def _log_initialization_status():
    """Log initialization status using practical discovery"""
    try:
        package_info = PARSERS_AVAILABLE.get('_package_info', {})
        working_parsers = {k: v for k, v in PARSERS_AVAILABLE.items() 
                          if k != '_package_info' and isinstance(v, dict) and v.get('status') == 'available'}
        
        if working_parsers:
            available_langs = list(working_parsers.keys())
            parser_types = list(set(info.get('parser_type', 'unknown') for info in working_parsers.values()))
            
            logger.info(f"Clean chunking system initialized")
            logger.info(f"Available parsers: {len(working_parsers)} ({', '.join(available_langs)})")
            logger.info(f"Parser types: {', '.join(parser_types)}")
            
            # Show package info
            if package_info.get('comprehensive_packages'):
                logger.info(f"Using comprehensive packages: {', '.join(package_info['comprehensive_packages'])}")
            elif package_info.get('individual_packages'):
                logger.info(f"Using individual packages: {len(package_info['individual_packages'])}")
            
        else:
            logger.warning("No tree-sitter parsers available")
            
            if not package_info.get('comprehensive_packages') and not package_info.get('individual_packages'):
                logger.info("💡 Install tree-sitter-languages for full support: pip install tree-sitter-languages")
            else:
                logger.info("💡 Some parsers may need additional packages")
    
    except Exception as e:
        logger.debug(f"Error during initialization logging: {e}")
        logger.warning("Parser status check failed - some parsers may not be available")

# Initialize on import
_log_initialization_status()

# Backward compatibility aliases
def _check_parser_availability():
    """Backward compatibility function"""
    logger.warning("_check_parser_availability is deprecated. Use get_parser_info() instead.")
    return get_parser_info()