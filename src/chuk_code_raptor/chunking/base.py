# chuk_code_raptor/chunking/base.py
"""
Base Chunker Interface
======================

Abstract base class for all chunkers in the modern AST-based system.
Clean, focused on tree-sitter structural understanding.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
import logging

from chuk_code_raptor.core.models import CodeChunk, ChunkType, create_chunk_id
from .config import ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)

class BaseChunker(ABC):
    """Abstract base class for all chunkers"""
    
    def __init__(self, config: ChunkingConfig):
        """
        Initialize the chunker with configuration.
        
        Args:
            config: ChunkingConfig object with chunking parameters
        """
        self.config = config
        self.supported_extensions: Set[str] = set()
        self.supported_languages: Set[str] = set()
        self.strategy = ChunkingStrategy.STRUCTURAL
        self.name = self.__class__.__name__
    
    @abstractmethod
    def can_chunk(self, language: str, file_extension: str) -> bool:
        """
        Check if this chunker can handle the given language/extension.
        
        Args:
            language: Programming language (e.g., 'python', 'javascript')
            file_extension: File extension (e.g., '.py', '.js')
            
        Returns:
            True if this chunker can handle the language/extension
        """
        pass
    
    @abstractmethod
    def chunk_content(self, content: str, language: str, file_path: str) -> List[CodeChunk]:
        """
        Chunk the content and return list of CodeChunk objects.
        
        Args:
            content: File content to chunk
            language: Programming language
            file_path: Path to the source file
            
        Returns:
            List of CodeChunk objects
        """
        pass
    
    def get_priority(self, language: str, file_extension: str) -> int:
        """
        Return priority for this chunker (higher = more preferred).
        
        Args:
            language: Programming language
            file_extension: File extension
            
        Returns:
            Priority score (0-100, higher is better)
        """
        if language in self.supported_languages:
            return 100
        if file_extension in self.supported_extensions:
            return 50
        return 0
    
    def _create_chunk(self, content: str, start_line: int, end_line: int, 
                     chunk_type: ChunkType, language: str, file_path: str,
                     identifier: str = None, metadata: Dict[str, Any] = None) -> CodeChunk:
        """
        Helper method to create a CodeChunk with consistent metadata.
        
        Args:
            content: Chunk content
            start_line: Starting line number (1-based)
            end_line: Ending line number (1-based)
            chunk_type: Type of chunk
            language: Programming language
            file_path: Source file path
            identifier: Optional identifier (function name, class name, etc.)
            metadata: Additional metadata
            
        Returns:
            CodeChunk object
        """
        chunk_id = create_chunk_id(file_path, start_line, chunk_type, identifier)
        
        # Merge chunker metadata
        final_metadata = {
            'chunker': self.name,
            'strategy': self.strategy.value,
            'extraction_method': 'structural',
            **(metadata or {})
        }
        
        return CodeChunk(
            id=chunk_id,
            file_path=file_path,
            content=content,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            language=language,
            metadata=final_metadata
        )
    
    def _split_into_lines(self, content: str) -> List[str]:
        """Split content into lines, preserving empty lines"""
        return content.split('\n')
    
    def _join_lines(self, lines: List[str], start_idx: int, end_idx: int) -> str:
        """Join a range of lines back into content"""
        return '\n'.join(lines[start_idx:end_idx])
    
    def _is_size_valid(self, content: str) -> bool:
        """Check if chunk size is within acceptable limits"""
        size = len(content)
        return self.config.min_chunk_size <= size <= self.config.max_chunk_size
    
    def _post_process_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """
        Post-process chunks to ensure quality and size constraints.
        
        Args:
            chunks: Raw chunks from the chunker
            
        Returns:
            Processed and validated chunks
        """
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
        
        return processed_chunks
    
    def _is_atomic_chunk(self, chunk: CodeChunk) -> bool:
        """Check if chunk represents an atomic unit that shouldn't be split"""
        atomic_types = {ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.CLASS}
        return chunk.chunk_type in atomic_types
    
    def _extract_identifier(self, content: str, chunk_type: ChunkType) -> Optional[str]:
        """
        Extract an identifier (name) from chunk content.
        Override in subclasses for language-specific extraction.
        
        Args:
            content: Chunk content
            chunk_type: Type of chunk
            
        Returns:
            Identifier string if found, None otherwise
        """
        return None
    
    def _calculate_complexity_score(self, chunk: CodeChunk) -> float:
        """
        Calculate a complexity/quality score for the chunk.
        Override in subclasses for language-specific scoring.
        
        Args:
            chunk: Chunk to score
            
        Returns:
            Complexity score (0.0 to 10.0)
        """
        score = 0.0
        
        # Size score (prefer target size)
        size_ratio = len(chunk.content) / self.config.target_chunk_size
        if 0.5 <= size_ratio <= 1.5:  # Within 50% of target
            score += 2.0
        elif 0.2 <= size_ratio <= 2.0:  # Reasonable size
            score += 1.0
        
        # Content richness
        word_count = len(chunk.content.split())
        if word_count > 10:
            score += 0.5
        if word_count > 50:
            score += 0.5
        
        # Semantic unit bonus
        if chunk.chunk_type in {ChunkType.FUNCTION, ChunkType.CLASS, ChunkType.METHOD}:
            score += 1.0
        
        return min(score, 10.0)  # Cap at 10.0

class ChunkerError(Exception):
    """Base exception for chunker errors"""
    pass

class UnsupportedLanguageError(ChunkerError):
    """Raised when a chunker doesn't support the requested language"""
    pass

class InvalidContentError(ChunkerError):
    """Raised when content cannot be chunked properly"""
    pass