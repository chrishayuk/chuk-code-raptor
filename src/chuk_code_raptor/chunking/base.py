# src/chuk_code_raptor/chunking/base.py
"""
Base Parser Classes
==================

Clean base classes for all parsers. Tree-sitter first, with support for alternative strategies.
No legacy support - clean forward-looking implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import logging

from .semantic_chunk import SemanticChunk, ContentType, create_chunk_id
from chuk_code_raptor.core.models import ChunkType
from .config import ChunkingConfig

logger = logging.getLogger(__name__)

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
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseParser(ABC):
    """Base class for all parsers in the chunking system"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.supported_languages: Set[str] = set()
        self.supported_extensions: Set[str] = set()
        self.parser_type: str = "base"
        self.name = self.__class__.__name__
        
        # Strategy for coordination with engine
        self.strategy = getattr(config, 'primary_strategy', 'structural')
    
    @abstractmethod
    def can_parse(self, language: str, file_extension: str) -> bool:
        """Check if this parser can handle the language/extension"""
        pass
    
    @abstractmethod
    def parse(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse content and return semantic chunks"""
        pass
    
    def get_priority(self, language: str, file_extension: str) -> int:
        """Return priority for this parser (higher = more preferred)"""
        if language in self.supported_languages:
            return 100
        if file_extension in self.supported_extensions:
            return 50
        return 0
    
    # Compatibility interface for existing engine
    def can_chunk(self, language: str, file_extension: str) -> bool:
        """Compatibility method"""
        return self.can_parse(language, file_extension)
    
    def chunk_content(self, content: str, language: str, file_path: str) -> List[SemanticChunk]:
        """Compatibility interface - creates ParseContext and calls parse"""
        context = ParseContext(
            file_path=file_path,
            language=language,
            content_type=self._detect_content_type(file_path, language),
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size
        )
        return self.parse(content, context)
    
    def parse_content(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Alternative interface name for consistency"""
        return self.parse(content, context)
    
    def _detect_content_type(self, file_path: str, language: str) -> ContentType:
        """Detect content type from file path and language"""
        from pathlib import Path
        extension = Path(file_path).suffix.lower()
        
        if extension in {'.md', '.markdown'}:
            return ContentType.MARKDOWN
        elif extension in {'.html', '.htm'}:
            return ContentType.HTML
        elif extension == '.json':
            return ContentType.JSON
        elif extension in {'.yaml', '.yml'}:
            return ContentType.YAML
        elif extension == '.xml':
            return ContentType.XML
        elif language in {'python', 'javascript', 'typescript', 'rust', 'go', 'java', 'cpp'}:
            return ContentType.CODE
        else:
            return ContentType.TEXT
    
    def _create_chunk(self, content: str, start_line: int, end_line: int, 
                     chunk_type: ChunkType, language: str, file_path: str,
                     identifier: str = None, metadata: Dict[str, Any] = None) -> SemanticChunk:
        """Helper method to create a SemanticChunk with consistent metadata"""
        chunk_id = create_chunk_id(file_path, start_line, chunk_type, identifier)
        
        # Merge parser metadata
        final_metadata = {
            'parser': self.name,
            'parser_type': self.parser_type,
            'strategy': self.strategy,
            'extraction_method': 'structural',
            **(metadata or {})
        }
        
        return SemanticChunk(
            id=chunk_id,
            file_path=file_path,
            content=content,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            content_type=self._detect_content_type(file_path, language),
            language=language,
            metadata=final_metadata
        )
    
    def _should_include_chunk(self, chunk: SemanticChunk) -> bool:
        """Determine if chunk should be included based on size and importance"""
        # Size filters
        if len(chunk.content) < self.config.min_chunk_size:
            # Allow small chunks for important types
            if chunk.chunk_type not in [ChunkType.IMPORT, ChunkType.COMMENT]:
                return False
        
        if len(chunk.content) > self.config.max_chunk_size:
            # Allow large chunks for atomic types if configured
            if not self.config.preserve_atomic_nodes:
                return False
        
        return True
    
    def _post_process(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Post-process chunks to ensure quality and size constraints"""
        processed_chunks = []
        
        for chunk in chunks:
            # Skip chunks that are too small (unless they're important)
            if (len(chunk.content) < self.config.min_chunk_size and 
                chunk.chunk_type not in {ChunkType.IMPORT, ChunkType.COMMENT}):
                logger.debug(f"Skipping small chunk: {chunk.id}")
                continue
            
            # Handle chunks that are too large
            if len(chunk.content) > self.config.max_chunk_size:
                if self.config.preserve_atomic_nodes and self._is_atomic_chunk(chunk):
                    # Preserve atomic chunks even if large
                    logger.warning(f"Large atomic chunk preserved: {chunk.id} ({len(chunk.content)} chars)")
                    processed_chunks.append(chunk)
                else:
                    # Could implement splitting logic here if needed
                    processed_chunks.append(chunk)
            else:
                processed_chunks.append(chunk)
        
        # Sort by line number
        processed_chunks.sort(key=lambda c: c.start_line)
        return processed_chunks
    
    def _is_atomic_chunk(self, chunk: SemanticChunk) -> bool:
        """Check if chunk represents an atomic unit that shouldn't be split"""
        atomic_types = {ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.CLASS}
        return chunk.chunk_type in atomic_types
    
    def _extract_identifier(self, content: str, chunk_type: ChunkType) -> Optional[str]:
        """
        Extract an identifier (name) from chunk content.
        Override in subclasses for language-specific extraction.
        """
        return None
    
    def _calculate_importance_score(self, chunk: SemanticChunk) -> float:
        """
        Calculate importance score for the chunk.
        Override in subclasses for language-specific scoring.
        """
        score = 0.5  # Base score
        
        # Size score (prefer target size)
        size_ratio = len(chunk.content) / self.config.target_chunk_size
        if 0.5 <= size_ratio <= 1.5:  # Within 50% of target
            score += 0.2
        elif 0.2 <= size_ratio <= 2.0:  # Reasonable size
            score += 0.1
        
        # Semantic unit bonus
        if chunk.chunk_type in {ChunkType.FUNCTION, ChunkType.CLASS, ChunkType.METHOD}:
            score += 0.3
        
        # Dependency bonus
        if chunk.dependencies:
            score += min(len(chunk.dependencies) * 0.05, 0.2)
        
        return min(score, 1.0)  # Cap at 1.0

class ParserError(Exception):
    """Base exception for parser errors"""
    pass

class UnsupportedLanguageError(ParserError):
    """Raised when language is not supported"""
    pass

class InvalidContentError(ParserError):
    """Raised when content cannot be chunked properly"""
    pass