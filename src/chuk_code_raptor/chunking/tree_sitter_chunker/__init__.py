# chuk_code_raptor/chunking/tree_sitter_chunker/__init__.py
"""
Semantic Tree-sitter Chunking System
====================================

Modern AST-based chunking using tree-sitter parsers that produces SemanticChunk objects.
Each language has its own specialized chunker with full semantic understanding.

Available Chunkers:
- PythonSemanticChunker: Python with full semantic analysis, dependency tracking
- More languages can be added by extending BaseSemanticTreeSitterChunker

Usage:
    from chuk_code_raptor.chunking.tree_sitter_chunker import PythonSemanticChunker
    from chuk_code_raptor.chunking import ChunkingConfig
    
    config = ChunkingConfig()
    chunker = PythonSemanticChunker(config)
    chunks = chunker.chunk_content(python_code, 'python', 'example.py')
    
    # chunks are now SemanticCodeChunk objects with full semantic understanding
"""

from .base import BaseSemanticTreeSitterChunker, SemanticChunkCandidate, LanguageConfig
from .config import TreeSitterConfigLoader, load_language_config, get_config_loader
from .python_chunker import PythonSemanticChunker

# Import other language chunkers as they're implemented
# from .javascript_chunker import JavaScriptSemanticChunker
# from .rust_chunker import RustSemanticChunker
# from .go_chunker import GoSemanticChunker

__all__ = [
    # Base classes
    'BaseSemanticTreeSitterChunker',
    'SemanticChunkCandidate', 
    'LanguageConfig',
    
    # Configuration
    'TreeSitterConfigLoader',
    'load_language_config',
    'get_config_loader',
    
    # Language-specific semantic chunkers
    'PythonSemanticChunker',
    # 'JavaScriptSemanticChunker',
    # 'RustSemanticChunker',
    # 'GoSemanticChunker',
]

# Auto-register semantic tree-sitter chunkers
def register_semantic_tree_sitter_chunkers():
    """Register all available semantic tree-sitter chunkers"""
    from ..registry import register_chunker
    
    # Register available semantic chunkers
    chunkers = [
        PythonSemanticChunker,
        # JavaScriptSemanticChunker,  # When implemented
        # RustSemanticChunker,        # When implemented
        # GoSemanticChunker,          # When implemented
    ]
    
    for chunker_class in chunkers:
        try:
            register_chunker(chunker_class)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Could not register {chunker_class.__name__}: {e}")

# Auto-register when imported
register_semantic_tree_sitter_chunkers()