# src/chuk_code_raptor/chunking/parsers/markdown.py
"""
Enhanced Markdown Tree-sitter Parser
====================================

Enhanced tree-sitter based markdown parser with semantic analysis and manual fallback.
Automatically falls back to manual semantic extraction when tree-sitter parsing is insufficient.
"""

from typing import Dict, List, Optional, Any
import logging
import re

from ..tree_sitter_base import TreeSitterParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class MarkdownParser(TreeSitterParser):
    """Enhanced Markdown parser with tree-sitter and manual semantic fallback"""
    
    def __init__(self, config):
        # Initialize parent class first
        super().__init__(config)
        
        # Then set our specific language support
        self.supported_languages = {'markdown'}
        self.supported_extensions = {'.md', '.markdown', '.mdown', '.mkd'}
        
        # Markdown semantic patterns for manual parsing
        self.semantic_patterns = {
            'headings': r'^(#{1,6})\s+(.+)$',
            'code_blocks': r'```(\w*)\n(.*?)\n```',
            'tables': r'\|.*\|',
            'lists': r'^[\s]*[-\*\+]\s+(.+)$',
            'ordered_lists': r'^[\s]*\d+\.\s+(.+)$',
            'blockquotes': r'^>\s+(.+)$',
            'horizontal_rules': r'^[-\*_]{3,}$',
            'links': r'\[([^\]]+)\]\(([^)]+)\)',
            'images': r'!\[([^\]]*)\]\(([^)]+)\)',
            'inline_code': r'`([^`]+)`',
            'bold': r'\*\*([^*]+)\*\*',
            'italic': r'\*([^*]+)\*'
        }
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_tree_sitter_language(self):
        """Get Markdown tree-sitter language"""
        try:
            import tree_sitter
            import tree_sitter_markdown
            return tree_sitter.Language(tree_sitter_markdown.language())
        except ImportError as e:
            raise ImportError("tree-sitter-markdown not installed. Install with: pip install tree-sitter-markdown") from e
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """Markdown AST node types to chunk types mapping"""
        return {
            # Headings
            'atx_heading': ChunkType.TEXT_BLOCK,
            'setext_heading': ChunkType.TEXT_BLOCK,
            
            # Content blocks
            'paragraph': ChunkType.TEXT_BLOCK,
            'fenced_code_block': ChunkType.TEXT_BLOCK,
            'indented_code_block': ChunkType.TEXT_BLOCK,
            'list': ChunkType.TEXT_BLOCK,
            'list_item': ChunkType.TEXT_BLOCK,
            'blockquote': ChunkType.TEXT_BLOCK,
            'table': ChunkType.TEXT_BLOCK,
            'thematic_break': ChunkType.TEXT_BLOCK,
            
            # Sections (if available in grammar)
            'section': ChunkType.TEXT_BLOCK,
        }
    
    def parse(self, content: str, context) -> List[SemanticChunk]:
        """Enhanced parse method with automatic fallback to manual semantic extraction"""
        
        # First try tree-sitter parsing
        tree_sitter_chunks = []
        try:
            tree_sitter_chunks = super().parse(content, context)
            logger.debug(f"Tree-sitter produced {len(tree_sitter_chunks)} chunks")
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed: {e}")
        
        # Check if tree-sitter parsing was successful
        if self._is_parsing_successful(tree_sitter_chunks, content):
            logger.info(f"Using tree-sitter results: {len(tree_sitter_chunks)} chunks")
            return self._post_process(tree_sitter_chunks)
        
        # Fall back to manual semantic element extraction
        logger.info("Tree-sitter parsing insufficient, using manual semantic analysis")
        semantic_chunks = self._process_semantic_elements(content, context)
        
        if semantic_chunks and len(semantic_chunks) > len(tree_sitter_chunks):
            logger.info(f"Manual semantic analysis produced {len(semantic_chunks)} chunks")
            return self._post_process(semantic_chunks)
        
        # Final fallback: return tree-sitter result even if poor
        logger.warning("Manual semantic analysis didn't improve results, using tree-sitter fallback")
        return self._post_process(tree_sitter_chunks)
    
    def _is_parsing_successful(self, chunks: List[SemanticChunk], content: str) -> bool:
        """Determine if tree-sitter parsing was successful"""
        if not chunks:
            logger.debug("No chunks created - parsing failed")
            return False
        
        # If we only got one chunk that's nearly the entire document, it's likely failed
        if len(chunks) == 1:
            chunk = chunks[0]
            content_ratio = len(chunk.content) / len(content)
            if content_ratio > 0.90:  # Single chunk contains >90% of content
                logger.debug(f"Single chunk contains {content_ratio:.1%} of content - likely failed parsing")
                return False
        
        # Check if we have reasonable chunk size distribution
        if chunks:
            avg_chunk_size = sum(len(chunk.content) for chunk in chunks) / len(chunks)
            if avg_chunk_size > self.config.target_chunk_size * 2:
                logger.debug(f"Average chunk size {avg_chunk_size} is too large")
                return False
        
        # If we have multiple chunks with reasonable sizes, consider it successful
        if len(chunks) >= 3:
            return True
        
        return False
    
    def _process_semantic_elements(self, content: str, context) -> List[SemanticChunk]:
        """Process Markdown content to extract semantic elements manually"""
        chunks = []
        lines = content.split('\n')
        current_line = 0
        
        while current_line < len(lines):
            chunk_result = self._extract_next_semantic_chunk(lines, current_line, content, context)
            if chunk_result:
                chunk, end_line = chunk_result
                if chunk:
                    chunks.append(chunk)
                current_line = end_line + 1
            else:
                current_line += 1
        
        return chunks
    
    def _extract_next_semantic_chunk(self, lines: List[str], start_line: int, full_content: str, context) -> Optional[tuple]:
        """Extract the next semantic chunk starting from the given line"""
        
        if start_line >= len(lines):
            return None
        
        # Skip empty lines
        while start_line < len(lines) and not lines[start_line].strip():
            start_line += 1
        
        if start_line >= len(lines):
            return None
        
        # Try to extract different types of semantic elements
        extractors = [
            self._extract_heading_section,
            self._extract_code_block,
            self._extract_table,
            self._extract_list,
            self._extract_blockquote,
            self._extract_paragraph
        ]
        
        for extractor in extractors:
            result = extractor(lines, start_line, full_content, context)
            if result:
                return result
        
        return None
    
    def _extract_heading_section(self, lines: List[str], start_line: int, full_content: str, context) -> Optional[tuple]:
        """Extract a heading and its content until the next heading of equal or higher level"""
        if start_line >= len(lines):
            return None
        
        line = lines[start_line].strip()
        heading_match = re.match(self.semantic_patterns['headings'], line)
        
        if not heading_match:
            return None
        
        heading_level = len(heading_match.group(1))
        heading_text = heading_match.group(2).strip()
        
        # Find the end of this section (next heading of equal or higher level)
        end_line = start_line
        for i in range(start_line + 1, len(lines)):
            next_line = lines[i].strip()
            if not next_line:  # Skip empty lines
                continue
            next_heading = re.match(self.semantic_patterns['headings'], next_line)
            if next_heading and len(next_heading.group(1)) <= heading_level:
                end_line = i - 1
                break
            end_line = i
        
        # Create chunk content
        chunk_lines = lines[start_line:end_line + 1]
        chunk_content = '\n'.join(chunk_lines).strip()
        
        if len(chunk_content) < self.config.min_chunk_size:
            # Still create the chunk if it's a heading, even if small
            if len(chunk_content) < 10:  # But not if it's tiny
                return None
        
        # Create semantic chunk
        chunk_id = create_chunk_id(
            context.file_path,
            start_line + 1,
            ChunkType.TEXT_BLOCK,
            f"heading_{self._clean_identifier(heading_text)}"
        )
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=chunk_content,
            start_line=start_line + 1,
            end_line=end_line + 1,
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=self._calculate_heading_importance(heading_level, heading_text),
            metadata={
                'parser': self.name,
                'parser_type': 'manual_semantic',
                'section_type': 'heading',
                'heading_level': heading_level,
                'heading_text': heading_text,
                'manual_extraction': True
            }
        )
        
        # Add semantic tags
        chunk.add_tag('heading', source='manual_semantic')
        chunk.add_tag(f'heading_level_{heading_level}', source='manual_semantic')
        chunk.add_tag('section', source='manual_semantic')
        
        # Analyze content for additional features
        self._analyze_chunk_content(chunk)
        
        return chunk, end_line
    
    def _extract_code_block(self, lines: List[str], start_line: int, full_content: str, context) -> Optional[tuple]:
        """Extract a fenced code block"""
        if start_line >= len(lines):
            return None
        
        line = lines[start_line].strip()
        if not line.startswith('```'):
            return None
        
        # Extract language if specified
        lang_match = re.match(r'^```(\w*)', line)
        language = lang_match.group(1) if lang_match and lang_match.group(1) else 'text'
        
        # Find the end of the code block
        end_line = start_line
        for i in range(start_line + 1, len(lines)):
            if lines[i].strip().startswith('```'):
                end_line = i
                break
        else:
            # No closing fence found, skip this
            return None
        
        # Create chunk content
        chunk_lines = lines[start_line:end_line + 1]
        chunk_content = '\n'.join(chunk_lines).strip()
        
        # Create semantic chunk
        chunk_id = create_chunk_id(
            context.file_path,
            start_line + 1,
            ChunkType.TEXT_BLOCK,
            f"code_{language}"
        )
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=chunk_content,
            start_line=start_line + 1,
            end_line=end_line + 1,
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=0.7,  # Code blocks are generally important
            metadata={
                'parser': self.name,
                'parser_type': 'manual_semantic',
                'section_type': 'code_block',
                'code_language': language,
                'manual_extraction': True
            }
        )
        
        # Add semantic tags
        chunk.add_tag('code_block', source='manual_semantic')
        if language and language != 'text':
            chunk.add_tag(f'code_{language}', source='manual_semantic')
        
        return chunk, end_line
    
    def _extract_table(self, lines: List[str], start_line: int, full_content: str, context) -> Optional[tuple]:
        """Extract a markdown table"""
        if start_line >= len(lines):
            return None
        
        line = lines[start_line].strip()
        if '|' not in line:
            return None
        
        # Find the extent of the table
        end_line = start_line
        for i in range(start_line, len(lines)):
            if i < len(lines) and '|' in lines[i].strip() and lines[i].strip():
                end_line = i
            else:
                break
        
        if end_line == start_line:
            return None
        
        # Create chunk content
        chunk_lines = lines[start_line:end_line + 1]
        chunk_content = '\n'.join(chunk_lines).strip()
        
        # Analyze table structure
        table_info = self._analyze_table_structure(chunk_content)
        
        # Create semantic chunk
        chunk_id = create_chunk_id(
            context.file_path,
            start_line + 1,
            ChunkType.TEXT_BLOCK,
            f"table_{table_info['rows']}x{table_info['columns']}"
        )
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=chunk_content,
            start_line=start_line + 1,
            end_line=end_line + 1,
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=0.6,
            metadata={
                'parser': self.name,
                'parser_type': 'manual_semantic',
                'section_type': 'table',
                'table_rows': table_info['rows'],
                'table_columns': table_info['columns'],
                'manual_extraction': True
            }
        )
        
        # Add semantic tags
        chunk.add_tag('table', source='manual_semantic')
        chunk.add_tag('tabular_data', source='manual_semantic')
        
        return chunk, end_line
    
    def _extract_list(self, lines: List[str], start_line: int, full_content: str, context) -> Optional[tuple]:
        """Extract a list (ordered or unordered)"""
        if start_line >= len(lines):
            return None
        
        line = lines[start_line].strip()
        is_unordered = re.match(self.semantic_patterns['lists'], line)
        is_ordered = re.match(self.semantic_patterns['ordered_lists'], line)
        
        if not (is_unordered or is_ordered):
            return None
        
        list_type = 'unordered' if is_unordered else 'ordered'
        
        # Find the extent of the list
        end_line = start_line
        for i in range(start_line, len(lines)):
            if i >= len(lines):
                break
            line = lines[i].strip()
            if not line:  # Empty line might continue list
                if i < len(lines) - 1:  # Check next line
                    next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    if (re.match(self.semantic_patterns['lists'], next_line) or 
                        re.match(self.semantic_patterns['ordered_lists'], next_line) or
                        next_line.startswith('  ')):
                        end_line = i
                        continue
                break
            if (re.match(self.semantic_patterns['lists'], line) or 
                re.match(self.semantic_patterns['ordered_lists'], line) or
                line.startswith('  ') or line.startswith('\t')):  # Indented content
                end_line = i
            else:
                break
        
        # Create chunk content
        chunk_lines = lines[start_line:end_line + 1]
        chunk_content = '\n'.join(chunk_lines).strip()
        
        if len(chunk_content) < self.config.min_chunk_size and len(chunk_content) < 20:
            return None
        
        # Create semantic chunk
        chunk_id = create_chunk_id(
            context.file_path,
            start_line + 1,
            ChunkType.TEXT_BLOCK,
            f"list_{list_type}"
        )
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=chunk_content,
            start_line=start_line + 1,
            end_line=end_line + 1,
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=0.5,
            metadata={
                'parser': self.name,
                'parser_type': 'manual_semantic',
                'section_type': 'list',
                'list_type': list_type,
                'manual_extraction': True
            }
        )
        
        # Add semantic tags
        chunk.add_tag('list', source='manual_semantic')
        chunk.add_tag(f'{list_type}_list', source='manual_semantic')
        
        return chunk, end_line
    
    def _extract_blockquote(self, lines: List[str], start_line: int, full_content: str, context) -> Optional[tuple]:
        """Extract a blockquote"""
        if start_line >= len(lines):
            return None
        
        line = lines[start_line].strip()
        if not re.match(self.semantic_patterns['blockquotes'], line):
            return None
        
        # Find the extent of the blockquote
        end_line = start_line
        for i in range(start_line, len(lines)):
            if i >= len(lines):
                break
            line = lines[i].strip()
            if line.startswith('>') or not line:  # Blockquote line or empty
                end_line = i
            else:
                break
        
        # Create chunk content
        chunk_lines = lines[start_line:end_line + 1]
        chunk_content = '\n'.join(chunk_lines).strip()
        
        # Create semantic chunk
        chunk_id = create_chunk_id(
            context.file_path,
            start_line + 1,
            ChunkType.TEXT_BLOCK,
            "blockquote"
        )
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=chunk_content,
            start_line=start_line + 1,
            end_line=end_line + 1,
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=0.6,
            metadata={
                'parser': self.name,
                'parser_type': 'manual_semantic',
                'section_type': 'blockquote',
                'manual_extraction': True
            }
        )
        
        # Add semantic tags
        chunk.add_tag('blockquote', source='manual_semantic')
        chunk.add_tag('quote', source='manual_semantic')
        
        return chunk, end_line
    
    def _extract_paragraph(self, lines: List[str], start_line: int, full_content: str, context) -> Optional[tuple]:
        """Extract a paragraph as fallback"""
        if start_line >= len(lines):
            return None
        
        line = lines[start_line].strip()
        if not line:  # Skip empty lines
            return None
        
        # Find the extent of the paragraph (until empty line or special element)
        end_line = start_line
        for i in range(start_line, len(lines)):
            if i >= len(lines):
                break
            line = lines[i].strip()
            if not line:  # Empty line ends paragraph
                end_line = i - 1
                break
            # Don't include lines that start special elements
            if (line.startswith('#') or line.startswith('```') or 
                line.startswith('>') or line.startswith('|') or
                re.match(r'^[\s]*[-\*\+]\s+', line) or
                re.match(r'^[\s]*\d+\.\s+', line)):
                if i == start_line:
                    return None  # This line is a special element, not a paragraph
                end_line = i - 1
                break
            end_line = i
        
        # Create chunk content
        chunk_lines = lines[start_line:end_line + 1]
        chunk_content = '\n'.join(chunk_lines).strip()
        
        if len(chunk_content) < self.config.min_chunk_size:
            return None
        
        # Create semantic chunk
        chunk_id = create_chunk_id(
            context.file_path,
            start_line + 1,
            ChunkType.TEXT_BLOCK,
            "paragraph"
        )
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=chunk_content,
            start_line=start_line + 1,
            end_line=end_line + 1,
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=0.4,
            metadata={
                'parser': self.name,
                'parser_type': 'manual_semantic',
                'section_type': 'paragraph',
                'manual_extraction': True
            }
        )
        
        # Add semantic tags
        chunk.add_tag('paragraph', source='manual_semantic')
        
        # Analyze content for additional features
        self._analyze_chunk_content(chunk)
        
        return chunk, end_line
    
    def _calculate_heading_importance(self, level: int, text: str) -> float:
        """Calculate importance score for headings"""
        # Higher level headings (h1, h2) are more important
        base_score = 1.0 - (level - 1) * 0.15
        
        # Certain keywords increase importance
        important_keywords = ['api', 'introduction', 'overview', 'getting started', 'configuration']
        text_lower = text.lower()
        
        for keyword in important_keywords:
            if keyword in text_lower:
                base_score += 0.1
                break
        
        return min(1.0, max(0.3, base_score))
    
    def _analyze_table_structure(self, content: str) -> Dict[str, int]:
        """Analyze table structure"""
        lines = [line.strip() for line in content.split('\n') if line.strip() and '|' in line]
        
        if not lines:
            return {'rows': 0, 'columns': 0}
        
        # Count columns from first row
        first_row = lines[0]
        columns = len([cell.strip() for cell in first_row.split('|') if cell.strip()])
        
        # Count data rows (exclude separator rows)
        data_rows = 0
        for line in lines:
            if not re.match(r'^[\s\|:-]+$', line):
                data_rows += 1
        
        return {'rows': data_rows, 'columns': columns}
    
    def _analyze_chunk_content(self, chunk: SemanticChunk):
        """Analyze chunk content for additional semantic features"""
        content = chunk.content
        
        # Check for various markdown features
        if re.search(self.semantic_patterns['links'], content):
            chunk.add_tag('has_links', source='manual_semantic')
        
        if re.search(self.semantic_patterns['images'], content):
            chunk.add_tag('has_images', source='manual_semantic')
        
        if re.search(self.semantic_patterns['inline_code'], content):
            chunk.add_tag('has_inline_code', source='manual_semantic')
        
        if re.search(self.semantic_patterns['bold'], content):
            chunk.add_tag('has_emphasis', source='manual_semantic')
        
        # Detect content type based on keywords
        content_lower = content.lower()
        
        if any(keyword in content_lower for keyword in ['api', 'endpoint', 'method']):
            chunk.add_tag('api_documentation', source='manual_semantic')
        
        if any(keyword in content_lower for keyword in ['config', 'settings', 'yaml', 'json']):
            chunk.add_tag('configuration', source='manual_semantic')
        
        if any(keyword in content_lower for keyword in ['install', 'setup', 'usage']):
            chunk.add_tag('tutorial', source='manual_semantic')
        
        if any(keyword in content_lower for keyword in ['example', 'demo']):
            chunk.add_tag('example', source='manual_semantic')
    
    def _clean_identifier(self, text: str) -> str:
        """Clean text for use as identifier"""
        # Remove special characters and make safe for ID
        clean_text = re.sub(r'[^\w\s-]', '', text)
        clean_text = re.sub(r'\s+', '_', clean_text)
        return clean_text.lower()[:50]  # Limit length
    
    def _should_include_chunk(self, chunk: SemanticChunk) -> bool:
        """Override to be more lenient with markdown chunk sizes"""
        # Always include headings regardless of size
        if chunk.metadata.get('heading_level') is not None:
            return True
        
        # Always include special content types
        section_type = chunk.metadata.get('section_type')
        if section_type in ['code_block', 'table', 'blockquote']:
            return True
        
        # Use parent logic for other chunks
        return super()._should_include_chunk(chunk)

    # Keep compatibility with existing tree-sitter methods
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from Markdown AST node"""
        if node.type in ['atx_heading', 'setext_heading']:
            # Extract heading text
            heading_text = self._extract_heading_text(node, content)
            if heading_text:
                # Create clean identifier from heading
                clean_text = "".join(c for c in heading_text if c.isalnum() or c.isspace())
                words = clean_text.split()[:5]  # First 5 words
                return "_".join(words).lower()
        
        elif node.type == 'fenced_code_block':
            # Try to get language from info string
            language = self._extract_code_language(node, content)
            return f"code_{language}" if language else "code_block"
        
        elif node.type == 'list':
            return "list"
        
        elif node.type == 'table':
            return "table"
        
        elif node.type == 'blockquote':
            return "quote"
        
        return node.type
    
    def _extract_heading_text(self, node, content: str) -> Optional[str]:
        """Extract text content from heading node"""
        # Look for heading_content or text children
        for child in node.children:
            if child.type in ['heading_content', 'text']:
                text = content[child.start_byte:child.end_byte].strip()
                return text
        
        # Fallback: get all text from the node
        node_text = content[node.start_byte:node.end_byte].strip()
        # Remove markdown heading markers
        if node_text.startswith('#'):
            return node_text.lstrip('#').strip()
        
        return node_text
    
    def _extract_code_language(self, node, content: str) -> Optional[str]:
        """Extract language from fenced code block"""
        for child in node.children:
            if child.type == 'info_string':
                info_text = content[child.start_byte:child.end_byte].strip()
                # Take first word as language
                return info_text.split()[0] if info_text else None
        return None
    
    def _extract_heading_level(self, node, content: str) -> int:
        """Extract heading level from node"""
        if node.type == 'atx_heading':
            # Count # characters at start
            node_text = content[node.start_byte:node.end_byte]
            level = 0
            for char in node_text:
                if char == '#':
                    level += 1
                else:
                    break
            return min(level, 6)  # Max level 6
        
        elif node.type == 'setext_heading':
            # Check underline type
            for child in node.children:
                if child.type == 'setext_h1_underline':
                    return 1
                elif child.type == 'setext_h2_underline':
                    return 2
        
        return 1  # Default
    
    def _add_semantic_tags(self, chunk: SemanticChunk, node, content: str):
        """Add Markdown-specific semantic tags"""
        super()._add_semantic_tags(chunk, node, content)
        
        # Heading tags
        if node.type in ['atx_heading', 'setext_heading']:
            chunk.add_tag('heading', source='tree_sitter')
            
            level = self._extract_heading_level(node, content)
            chunk.add_tag(f'heading_level_{level}', source='tree_sitter')
            chunk.metadata['heading_level'] = level
            
            # Analyze heading content for semantic meaning
            heading_text = self._extract_heading_text(node, content)
            if heading_text:
                self._analyze_heading_semantics(chunk, heading_text)
        
        # Code block tags
        elif node.type in ['fenced_code_block', 'indented_code_block']:
            chunk.add_tag('code_block', source='tree_sitter')
            chunk.metadata['section_type'] = 'code_block'
            
            if node.type == 'fenced_code_block':
                language = self._extract_code_language(node, content)
                if language:
                    chunk.add_tag(f'code_{language}', source='tree_sitter')
                    chunk.metadata['code_language'] = language
        
        # List tags
        elif node.type in ['list', 'list_item']:
            chunk.add_tag('list', source='tree_sitter')
            chunk.metadata['section_type'] = 'list'
        
        # Table tags
        elif node.type == 'table':
            chunk.add_tag('table', source='tree_sitter')
            chunk.add_tag('tabular_data', source='tree_sitter')
            chunk.metadata['section_type'] = 'table'
        
        # Quote tags
        elif node.type == 'blockquote':
            chunk.add_tag('blockquote', source='tree_sitter')
            chunk.add_tag('quote', source='tree_sitter')
            chunk.metadata['section_type'] = 'quote'
        
        # Paragraph tags
        elif node.type == 'paragraph':
            chunk.add_tag('paragraph', source='tree_sitter')
            chunk.metadata['section_type'] = 'paragraph'
        
        # General content analysis
        self._analyze_content_patterns(chunk)
    
    def _analyze_heading_semantics(self, chunk: SemanticChunk, heading_text: str):
        """Analyze heading content for semantic meaning"""
        heading_lower = heading_text.lower()
        
        # API documentation
        if any(word in heading_lower for word in ['api', 'endpoint', 'rest', 'graphql']):
            chunk.add_tag('api_documentation', source='tree_sitter')
        
        # Installation/setup
        if any(word in heading_lower for word in ['install', 'setup', 'configuration', 'getting started']):
            chunk.add_tag('setup_guide', source='tree_sitter')
        
        # Tutorial/guide content
        if any(word in heading_lower for word in ['tutorial', 'guide', 'how to', 'example', 'walkthrough']):
            chunk.add_tag('tutorial', source='tree_sitter')
        
        # Troubleshooting
        if any(word in heading_lower for word in ['troubleshoot', 'error', 'problem', 'debug', 'issue']):
            chunk.add_tag('troubleshooting', source='tree_sitter')
        
        # Reference material
        if any(word in heading_lower for word in ['reference', 'specification', 'docs', 'documentation']):
            chunk.add_tag('reference', source='tree_sitter')
        
        # Features/functionality
        if any(word in heading_lower for word in ['feature', 'functionality', 'usage', 'use case']):
            chunk.add_tag('feature_description', source='tree_sitter')
    
    def _analyze_content_patterns(self, chunk: SemanticChunk):
        """Analyze content for patterns and add relevant tags"""
        content = chunk.content
        content_lower = content.lower()
        
        # TODO/FIXME markers
        if any(marker in content for marker in ['TODO', 'FIXME', 'XXX', 'HACK', 'NOTE']):
            chunk.add_tag('todo', source='tree_sitter')
        
        # Callout patterns (Note:, Warning:, etc.)
        if any(pattern in content_lower for pattern in ['note:', 'warning:', 'important:', 'tip:', 'caution:']):
            chunk.add_tag('callout', source='tree_sitter')
        
        # Links
        import re
        if re.search(r'\[([^\]]+)\]\(([^)]+)\)', content):
            chunk.add_tag('has_links', source='tree_sitter')
        
        # Images
        if re.search(r'!\[([^\]]*)\]\(([^)]+)\)', content):
            chunk.add_tag('has_images', source='tree_sitter')
        
        # Code spans (inline code)
        if '`' in content and content.count('`') >= 2:
            chunk.add_tag('has_inline_code', source='tree_sitter')
        
        # Emphasis (bold/italic)
        if any(marker in content for marker in ['**', '__', '*', '_']):
            chunk.add_tag('has_emphasis', source='tree_sitter')
    
    def _extract_dependencies(self, chunk: SemanticChunk, node, content: str):
        """Extract Markdown dependencies (links, references)"""
        import re
        
        # Extract links as dependencies
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, chunk.content)
        
        for link_text, link_url in links:
            chunk.dependencies.append(f"links_to:{link_url}")
        
        # Extract reference-style links
        ref_pattern = r'\[([^\]]+)\]\[([^\]]*)\]'
        refs = re.findall(ref_pattern, chunk.content)
        
        for ref_text, ref_id in refs:
            if ref_id:  # Non-empty reference ID
                chunk.dependencies.append(f"references:{ref_id}")
        
        # Store extracted information
        if links:
            chunk.metadata['external_links'] = [url for _, url in links]
        
        if refs:
            chunk.metadata['reference_links'] = [ref_id for _, ref_id in refs if ref_id]