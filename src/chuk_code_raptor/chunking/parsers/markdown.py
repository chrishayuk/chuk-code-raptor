# src/chuk_code_raptor/chunking/parsers/markdown.py
"""
Markdown Tree-sitter Parser
===========================

Tree-sitter based markdown parser with semantic analysis.
"""

from typing import Dict, List, Optional, Any
import logging

from ..tree_sitter_base import TreeSitterParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class MarkdownParser(TreeSitterParser):
    """Markdown parser using tree-sitter"""
    
    def __init__(self, config):
        # Initialize parent class first
        super().__init__(config)
        
        # Then set our specific language support
        self.supported_languages = {'markdown'}
        self.supported_extensions = {'.md', '.markdown', '.mdown', '.mkd'}
    
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
            
            # Check for ordered vs unordered
            if 'ordered' in content[node.start_byte:node.end_byte][:20].lower():
                chunk.add_tag('ordered_list', source='tree_sitter')
            else:
                chunk.add_tag('unordered_list', source='tree_sitter')
        
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
    
    def _should_include_chunk(self, chunk: SemanticChunk) -> bool:
        """Override to be more lenient with markdown chunk sizes"""
        # Always include headings regardless of size
        if chunk.metadata.get('heading_level') is not None:
            return True
        
        # Always include special content types
        section_type = chunk.metadata.get('section_type')
        if section_type in ['code_block', 'table', 'quote']:
            return True
        
        # Use parent logic for other chunks
        return super()._should_include_chunk(chunk)