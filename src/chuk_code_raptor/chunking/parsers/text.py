# src/chuk_code_raptor/chunking/parsers/text.py
"""
Generic Text Parser
==================

Heuristic-based parser for plain text and unstructured content.
Handles unknown file types and provides basic text chunking.
"""

import logging
import re
from typing import List, Optional

from ..heuristic_base import HeuristicParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from ..base import ParseContext
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class TextParser(HeuristicParser):
    """Generic text parser for unstructured content and unknown file types"""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "TextParser"
        self.supported_languages = {'text', 'plaintext', 'unknown'}
        self.supported_extensions = {'.txt', '.log', '.cfg', '.ini', '.conf', '.properties', '.env'}
        
        # Text parsing patterns
        self.text_patterns = {
            'paragraph_break': r'\n\s*\n',
            'section_header': r'^[A-Z][A-Z\s]{3,}:?\s*$',
            'numbered_section': r'^\d+[\.\)]\s+.+$',
            'bullet_point': r'^[\s]*[-\*\+•]\s+.+$',
            'key_value': r'^([^=:]+)[=:]\s*(.+)$',
            'log_entry': r'^\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://[^\s]+',
            'file_path': r'[/\\]?(?:[a-zA-Z0-9_.-]+[/\\])*[a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+',
        }
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _extract_chunks_heuristically(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Extract chunks from text content using heuristic methods"""
        
        # Determine text type and strategy
        text_type = self._analyze_text_type(content)
        
        if text_type == 'structured':
            return self._extract_structured_text_chunks(content, context)
        elif text_type == 'log':
            return self._extract_log_chunks(content, context)
        elif text_type == 'config':
            return self._extract_config_chunks(content, context)
        else:
            return self._extract_paragraph_chunks(content, context)
    
    def _analyze_text_type(self, content: str) -> str:
        """Analyze text to determine the best chunking strategy"""
        lines = content.split('\n')
        
        # Check for log patterns
        log_lines = sum(1 for line in lines if re.search(self.text_patterns['log_entry'], line))
        if log_lines > len(lines) * 0.3:
            return 'log'
        
        # Check for config patterns
        config_lines = sum(1 for line in lines if re.search(self.text_patterns['key_value'], line))
        if config_lines > len(lines) * 0.4:
            return 'config'
        
        # Check for structured text (headers, sections)
        header_lines = sum(1 for line in lines if re.search(self.text_patterns['section_header'], line))
        numbered_lines = sum(1 for line in lines if re.search(self.text_patterns['numbered_section'], line))
        
        if (header_lines + numbered_lines) > len(lines) * 0.1:
            return 'structured'
        
        return 'plain'
    
    def _extract_structured_text_chunks(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Extract chunks from structured text with headers and sections"""
        chunks = []
        lines = content.split('\n')
        current_section = []
        current_start_line = 1
        section_title = None
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Check for section headers
            if (re.search(self.text_patterns['section_header'], line) or 
                re.search(self.text_patterns['numbered_section'], line)):
                
                # Save previous section if it exists
                if current_section:
                    chunk = self._create_text_chunk(
                        content='\n'.join(current_section),
                        start_line=current_start_line,
                        end_line=i,
                        context=context,
                        text_type='section',
                        title=section_title
                    )
                    if chunk:
                        chunks.append(chunk)
                
                # Start new section
                current_section = [line]
                current_start_line = line_num
                section_title = line.strip()
            else:
                current_section.append(line)
        
        # Handle last section
        if current_section:
            chunk = self._create_text_chunk(
                content='\n'.join(current_section),
                start_line=current_start_line,
                end_line=len(lines),
                context=context,
                text_type='section',
                title=section_title
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _extract_log_chunks(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Extract chunks from log files"""
        chunks = []
        lines = content.split('\n')
        current_chunk_lines = []
        current_start_line = 1
        chunk_count = 0
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Check if this is a new log entry
            if re.search(self.text_patterns['log_entry'], line) and current_chunk_lines:
                # Check if current chunk is large enough
                chunk_content = '\n'.join(current_chunk_lines)
                if len(chunk_content) >= self.config.min_chunk_size:
                    chunk = self._create_text_chunk(
                        content=chunk_content,
                        start_line=current_start_line,
                        end_line=i,
                        context=context,
                        text_type='log_entry',
                        entry_number=chunk_count + 1
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_count += 1
                
                # Start new chunk
                current_chunk_lines = [line]
                current_start_line = line_num
            else:
                current_chunk_lines.append(line)
            
            # Also chunk by size to prevent overly large chunks
            if len('\n'.join(current_chunk_lines)) > self.config.max_chunk_size:
                chunk_content = '\n'.join(current_chunk_lines)
                chunk = self._create_text_chunk(
                    content=chunk_content,
                    start_line=current_start_line,
                    end_line=line_num,
                    context=context,
                    text_type='log_entry',
                    entry_number=chunk_count + 1
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_count += 1
                
                current_chunk_lines = []
                current_start_line = line_num + 1
        
        # Handle remaining content
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            if len(chunk_content) >= self.config.min_chunk_size:
                chunk = self._create_text_chunk(
                    content=chunk_content,
                    start_line=current_start_line,
                    end_line=len(lines),
                    context=context,
                    text_type='log_entry',
                    entry_number=chunk_count + 1
                )
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def _extract_config_chunks(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Extract chunks from configuration files"""
        chunks = []
        lines = content.split('\n')
        current_section = []
        current_start_line = 1
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Group config entries
            if line.strip() and not line.startswith('#') and not line.startswith(';'):
                current_section.append(line)
            else:
                # Empty line or comment - potential section break
                if current_section:
                    section_content = '\n'.join(current_section)
                    if len(section_content) >= self.config.min_chunk_size:
                        chunk = self._create_text_chunk(
                            content=section_content,
                            start_line=current_start_line,
                            end_line=i,
                            context=context,
                            text_type='config_section'
                        )
                        if chunk:
                            chunks.append(chunk)
                    
                    current_section = []
                    current_start_line = line_num + 1
                
                # Include comments as separate chunks if substantial
                if line.strip() and (line.startswith('#') or line.startswith(';')):
                    current_section.append(line)
        
        # Handle remaining content
        if current_section:
            section_content = '\n'.join(current_section)
            if len(section_content) >= self.config.min_chunk_size:
                chunk = self._create_text_chunk(
                    content=section_content,
                    start_line=current_start_line,
                    end_line=len(lines),
                    context=context,
                    text_type='config_section'
                )
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def _extract_paragraph_chunks(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Extract chunks from plain text by paragraphs"""
        chunks = []
        
        # Split by paragraph breaks
        paragraphs = re.split(self.text_patterns['paragraph_break'], content)
        current_line = 1
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_lines = paragraph.count('\n') + 1
            
            # Only create chunk if it meets size requirements
            if len(paragraph) >= self.config.min_chunk_size:
                chunk = self._create_text_chunk(
                    content=paragraph,
                    start_line=current_line,
                    end_line=current_line + paragraph_lines - 1,
                    context=context,
                    text_type='paragraph'
                )
                if chunk:
                    chunks.append(chunk)
            
            current_line += paragraph_lines + 1  # +1 for paragraph break
        
        # If no paragraphs met size requirements, create chunks by size
        if not chunks:
            chunks = self._extract_by_size(content, context)
        
        return chunks
    
    def _extract_by_size(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Fallback: extract chunks purely by size"""
        chunks = []
        lines = content.split('\n')
        current_chunk_lines = []
        current_start_line = 1
        
        for i, line in enumerate(lines):
            current_chunk_lines.append(line)
            
            # Check if chunk is large enough
            chunk_content = '\n'.join(current_chunk_lines)
            if len(chunk_content) >= self.config.target_chunk_size:
                chunk = self._create_text_chunk(
                    content=chunk_content,
                    start_line=current_start_line,
                    end_line=i + 1,
                    context=context,
                    text_type='text_block'
                )
                if chunk:
                    chunks.append(chunk)
                
                current_chunk_lines = []
                current_start_line = i + 2
        
        # Handle remaining content
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            if len(chunk_content) >= self.config.min_chunk_size:
                chunk = self._create_text_chunk(
                    content=chunk_content,
                    start_line=current_start_line,
                    end_line=len(lines),
                    context=context,
                    text_type='text_block'
                )
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def _create_text_chunk(self, content: str, start_line: int, end_line: int,
                          context: ParseContext, text_type: str, **metadata) -> Optional[SemanticChunk]:
        """Create a text chunk with appropriate metadata"""
        
        chunk = self._create_heuristic_chunk(
            content=content,
            start_line=start_line,
            end_line=end_line,
            chunk_type=ChunkType.TEXT_BLOCK,
            context=context,
            identifier=f"{text_type}_{start_line}",
            text_type=text_type,
            **metadata
        )
        
        # Add text-specific tags
        chunk.add_tag('text_content', source='heuristic')
        chunk.add_tag(f'text_{text_type}', source='heuristic')
        
        # Analyze content for additional features
        self._analyze_text_content(chunk)
        
        return chunk
    
    def _analyze_text_content(self, chunk: SemanticChunk):
        """Analyze text content for additional semantic information"""
        content = chunk.content
        
        # Check for various text features
        if re.search(self.text_patterns['email'], content):
            chunk.add_tag('contains_email', source='heuristic')
        
        if re.search(self.text_patterns['url'], content):
            chunk.add_tag('contains_url', source='heuristic')
        
        if re.search(self.text_patterns['file_path'], content):
            chunk.add_tag('contains_filepath', source='heuristic')
        
        # Word and sentence analysis
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        chunk.metadata.update({
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        })
        
        # Content classification
        if chunk.metadata.get('avg_sentence_length', 0) > 20:
            chunk.add_tag('complex_text', source='heuristic')
        elif chunk.metadata.get('avg_sentence_length', 0) < 5:
            chunk.add_tag('simple_text', source='heuristic')
        
        # Check for technical content
        technical_indicators = ['error', 'warning', 'debug', 'info', 'exception', 'stack trace']
        if any(indicator in content.lower() for indicator in technical_indicators):
            chunk.add_tag('technical_content', source='heuristic')