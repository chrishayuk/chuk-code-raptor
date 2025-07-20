# src/chuk_code_raptor/chunking/heuristic_base.py
"""
Heuristic Base Parser
====================

Base class for non-tree-sitter parsers using heuristics, regex, etc.
Clean implementation extending BaseParser.
"""

from abc import abstractmethod
from typing import List
import logging
import re

from .base import BaseParser, ParseContext
from .semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class HeuristicParser(BaseParser):
    """Base class for non-tree-sitter parsers"""
    
    def __init__(self, config):
        super().__init__(config)
        self.parser_type = "heuristic"
    
    @abstractmethod
    def _extract_chunks_heuristically(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Extract chunks using heuristic methods - implement in subclasses"""
        pass
    
    def parse(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse content using heuristic methods"""
        if not content.strip():
            return []
        
        chunks = self._extract_chunks_heuristically(content, context)
        return self._post_process(chunks)
    
    def _create_heuristic_chunk(self, content: str, start_line: int, end_line: int,
                               chunk_type: ChunkType, context: ParseContext,
                               identifier: str = None, **kwargs) -> SemanticChunk:
        """Helper to create chunks in heuristic parsers"""
        chunk_id = create_chunk_id(context.file_path, start_line, chunk_type, identifier)
        
        # Calculate importance based on content
        importance_score = self._calculate_heuristic_importance(content, chunk_type)
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=content.strip(),
            start_line=start_line,
            end_line=end_line,
            content_type=context.content_type,
            chunk_type=chunk_type,
            language=context.language,
            importance_score=importance_score,
            metadata={
                'parser': self.name,
                'parser_type': 'heuristic',
                'identifier': identifier,
                'extraction_method': 'heuristic',
                **kwargs
            }
        )
        
        return chunk
    
    def _calculate_heuristic_importance(self, content: str, chunk_type: ChunkType) -> float:
        """Calculate importance score for heuristic chunks"""
        base_score = 0.5
        
        # Adjust based on chunk type
        type_scores = {
            ChunkType.FUNCTION: 0.9,
            ChunkType.CLASS: 0.9,
            ChunkType.METHOD: 0.8,
            ChunkType.IMPORT: 0.6,
            ChunkType.TEXT_BLOCK: 0.5,
        }
        
        # Handle None or unknown chunk types
        if chunk_type is None:
            base_score = 0.5
        else:
            base_score = type_scores.get(chunk_type, 0.5)
        
        # Adjust based on content characteristics
        word_count = len(content.split())
        line_count = content.count('\n') + 1
        
        # Prefer medium-sized chunks
        if 50 <= word_count <= 200:
            base_score += 0.1
        if 5 <= line_count <= 50:
            base_score += 0.1
        
        # Look for structural indicators
        if any(indicator in content for indicator in ['def ', 'class ', 'function ', 'const ']):
            base_score += 0.1
        
        return min(1.0, max(0.1, base_score))
    
    def _split_into_lines(self, content: str) -> List[str]:
        """Split content into lines, preserving empty lines"""
        return content.split('\n')
    
    def _join_lines(self, lines: List[str], start_idx: int, end_idx: int) -> str:
        """Join a range of lines back into content"""
        return '\n'.join(lines[start_idx:end_idx])
    
    def _find_line_ranges(self, content: str, pattern: str) -> List[tuple]:
        """Find line ranges matching a pattern"""
        lines = self._split_into_lines(content)
        ranges = []
        
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                ranges.append((i + 1, i + 1))  # Convert to 1-based line numbers
        
        return ranges
    
    def _extract_sections_by_pattern(self, content: str, start_pattern: str, 
                                   end_pattern: str = None) -> List[tuple]:
        """Extract sections between start and end patterns"""
        lines = self._split_into_lines(content)
        sections = []
        
        if end_pattern is None:
            # If no end pattern, each start creates a section to the end
            for i, line in enumerate(lines):
                if re.search(start_pattern, line):
                    sections.append((i + 1, len(lines)))  # 1-based line numbers
        else:
            # Look for start-end pairs
            current_start = None
            
            for i, line in enumerate(lines):
                if re.search(start_pattern, line) and current_start is None:
                    current_start = i + 1  # 1-based line number
                elif end_pattern and re.search(end_pattern, line) and current_start is not None:
                    sections.append((current_start, i + 1))
                    current_start = None
            
            # Handle unclosed sections
            if current_start is not None:
                sections.append((current_start, len(lines)))
        
        return sections