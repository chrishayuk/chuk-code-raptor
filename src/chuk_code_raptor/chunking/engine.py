# src/chuk_code_raptor/chunking/engine.py
"""
Chunking Engine
===============

Main engine that coordinates chunking operations.
Clean, unified implementation using semantic chunks.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from chuk_code_raptor.core.models import FileInfo
from .config import ChunkingConfig
from .registry import ChunkerRegistry, get_registry
from .base import UnsupportedLanguageError, InvalidContentError
from .semantic_chunk import SemanticChunk, ContentType
logger = logging.getLogger(__name__)

# Simple ParseContext class for chunking
from dataclasses import dataclass

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

class ChunkingEngine:
    """Main engine for chunking operations"""
    
    def __init__(self, config: ChunkingConfig = None, registry: ChunkerRegistry = None):
        """
        Initialize chunking engine.
        
        Args:
            config: ChunkingConfig to use (creates default if None)
            registry: ChunkerRegistry to use (uses global if None)
        """
        self.config = config or ChunkingConfig()
        self.registry = registry or get_registry()
        
        # Set config on registry
        self.registry.set_config(self.config)
        
        # Statistics
        self.stats = {
            'files_chunked': 0,
            'chunks_created': 0,
            'total_processing_time': 0.0,
            'chunker_usage': {},
            'errors': []
        }
        
        logger.info(f"Initialized ChunkingEngine with {len(self.registry._chunkers)} chunkers")
    
    def chunk_file(self, file_path: str, language: str = None, content: str = None) -> List[SemanticChunk]:
        """
        Chunk a file into semantic chunks.
        
        Args:
            file_path: Path to file to chunk
            language: Programming language (auto-detected if None)
            content: File content (read from file if None)
            
        Returns:
            List of SemanticChunk objects
        """
        file_path = Path(file_path)
        
        # Read content if not provided
        if content is None:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Detect language if not provided
        if language is None:
            language = self._detect_language(file_path)
        
        return self.chunk_content(content, language, str(file_path))
    
    def chunk_content(self, content: str, language: str, file_path: str = "unknown") -> List[SemanticChunk]:
        """
        Chunk content string into semantic chunks.
        
        Args:
            content: Content to chunk
            language: Programming language
            file_path: Source file path (for metadata)
            
        Returns:
            List of SemanticChunk objects
        """
        import time
        start_time = time.time()
        
        if not content.strip():
            logger.debug(f"Empty content for {file_path}")
            return []
        
        # Get appropriate chunker
        file_extension = Path(file_path).suffix.lower()
        chunker = self.registry.get_chunker(language, file_extension)
        
        if chunker is None:
            error_msg = f"No chunker available for language '{language}' and extension '{file_extension}'"
            self.stats['errors'].append(error_msg)
            raise UnsupportedLanguageError(error_msg)
        
        # Create parse context
        context = ParseContext(
            file_path=file_path,
            language=language,
            content_type=self._detect_content_type(file_path, language),
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            enable_semantic_analysis=True,
            enable_dependency_tracking=True
        )
        
        # All chunkers use the modern parse_content interface
        chunks = chunker.parse_content(content, context)
        
        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(chunker, chunks, processing_time)
        
        chunker_name = getattr(chunker, 'name', chunker.__class__.__name__)
        logger.debug(f"Chunked {file_path} into {len(chunks)} chunks using {chunker_name}")
        return chunks
    
    def _detect_content_type(self, file_path: str, language: str) -> ContentType:
        """Detect content type from file and language"""
        extension = Path(file_path).suffix.lower()
        
        # Content type mapping
        if extension in {'.md', '.markdown'}:
            return ContentType.MARKDOWN
        elif extension in {'.html', '.htm'}:
            return ContentType.HTML
        elif extension in {'.txt', '.rst'}:
            return ContentType.PLAINTEXT
        elif extension in {'.json'}:
            return ContentType.JSON
        elif extension in {'.yaml', '.yml'}:
            return ContentType.YAML
        elif extension in {'.xml'}:
            return ContentType.XML
        elif language in {'python', 'javascript', 'typescript', 'rust', 'go', 'java', 'cpp', 'c'}:
            return ContentType.CODE
        else:
            return ContentType.PLAINTEXT
    
    def chunk_file_info(self, file_info: FileInfo, content: str = None) -> List[SemanticChunk]:
        """
        Chunk a file using FileInfo metadata.
        
        Args:
            file_info: FileInfo object with file metadata
            content: File content (read from file if None)
            
        Returns:
            List of SemanticChunk objects
        """
        return self.chunk_file(
            file_path=file_info.path,
            language=file_info.language,
            content=content
        )
    
    def get_supported_languages(self) -> List[str]:
        """Get list of all supported languages"""
        return self.registry.get_available_languages()
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions"""
        return self.registry.get_available_extensions()
    
    def can_chunk_language(self, language: str) -> bool:
        """Check if the engine can chunk the given language"""
        return language in self.get_supported_languages()
    
    def can_chunk_file(self, file_path: str) -> bool:
        """Check if the engine can chunk the given file"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        language = self._detect_language(file_path)
        
        return self.registry.get_chunker(language, extension) is not None
    
    def get_chunker_for_language(self, language: str, file_extension: str = None):
        """Get the chunker that would be used for a language/extension"""
        return self.registry.get_chunker(language, file_extension)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        stats = self.stats.copy()
        
        # Add registry information
        stats['available_chunkers'] = len(self.registry._chunkers)
        stats['supported_languages'] = len(self.get_supported_languages())
        stats['supported_extensions'] = len(self.get_supported_extensions())
        
        # Calculate averages
        if stats['files_chunked'] > 0:
            stats['avg_chunks_per_file'] = stats['chunks_created'] / stats['files_chunked']
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['files_chunked']
        else:
            stats['avg_chunks_per_file'] = 0
            stats['avg_processing_time'] = 0
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = {
            'files_chunked': 0,
            'chunks_created': 0,
            'total_processing_time': 0.0,
            'chunker_usage': {},
            'errors': []
        }
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        extension = file_path.suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.pyx': 'python', 
            '.pyi': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.mjs': 'javascript',
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.rst': 'rst',
            '.txt': 'text',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xml': 'xml',
        }
        
        return language_map.get(extension, 'unknown')
    
    def _update_stats(self, chunker, chunks: List[SemanticChunk], processing_time: float):
        """Update internal statistics"""
        self.stats['files_chunked'] += 1
        self.stats['chunks_created'] += len(chunks)
        self.stats['total_processing_time'] += processing_time
        
        # Track chunker usage
        chunker_name = getattr(chunker, 'name', chunker.__class__.__name__)
        if chunker_name not in self.stats['chunker_usage']:
            self.stats['chunker_usage'][chunker_name] = {
                'files_processed': 0,
                'chunks_created': 0,
                'total_time': 0.0
            }
        
        self.stats['chunker_usage'][chunker_name]['files_processed'] += 1
        self.stats['chunker_usage'][chunker_name]['chunks_created'] += len(chunks)
        self.stats['chunker_usage'][chunker_name]['total_time'] += processing_time