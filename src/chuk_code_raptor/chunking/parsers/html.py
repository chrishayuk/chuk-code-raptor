# src/chuk_code_raptor/chunking/parsers/html.py
"""
HTML Parser
===========

HTML parser using tree-sitter-html for AST-based chunking.
Follows the common TreeSitterParser architecture with enhanced semantic element processing.
"""

import logging
from typing import Dict, List, Optional

from ..tree_sitter_base import TreeSitterParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class HTMLParser(TreeSitterParser):
    """HTML parser using tree-sitter-html with enhanced semantic chunking"""
    
    def __init__(self, config):
        # Call parent first
        super().__init__(config)
        
        # Then set our specific supported languages and extensions
        self.supported_languages = {'html'}
        self.supported_extensions = {'.html', '.htm', '.xhtml'}
        
        # Semantic HTML elements we want to chunk
        self.semantic_elements = {
            'article', 'section', 'header', 'footer', 'nav', 'main', 
            'aside', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'pre', 'code', 'blockquote', 'ul', 'ol', 'li', 
            'table', 'thead', 'tbody', 'tr', 'form', 'figure'
        }
        
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_tree_sitter_language(self):
        """Get HTML tree-sitter language"""
        try:
            import tree_sitter
            import tree_sitter_html
            return tree_sitter.Language(tree_sitter_html.language())
        except ImportError as e:
            raise ImportError("tree-sitter-html not installed. Install with: pip install tree-sitter-html") from e
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """HTML AST node types to chunk types mapping - more inclusive"""
        return {
            'element': ChunkType.TEXT_BLOCK,
            'text': ChunkType.TEXT_BLOCK,  # Also include text nodes
        }
    
    def parse(self, content: str, context) -> List[SemanticChunk]:
        """Override parse method to use semantic element extraction"""
        
        # First try semantic element extraction
        semantic_chunks = self._process_semantic_elements(content, context)
        
        # If we got good semantic chunks, use them
        if len(semantic_chunks) > 1:  # More than just the root element
            return self._post_process(semantic_chunks)
        
        # Otherwise fall back to default tree-sitter parsing
        return super().parse(content, context)
    
    def _process_semantic_elements(self, content: str, context) -> List[SemanticChunk]:
        """Process HTML content to extract semantic elements as separate chunks"""
        try:
            import tree_sitter
            import tree_sitter_html
            
            language = tree_sitter.Language(tree_sitter_html.language())
            parser = tree_sitter.Parser(language)
            tree = parser.parse(bytes(content, 'utf8'))
            
            chunks = []
            
            def extract_semantic_chunks(node, depth=0):
                """Extract chunks from semantic HTML elements"""
                if node.type == 'element':
                    # Get tag name
                    tag_name = self._extract_tag_name(node, content)
                    
                    # If this is a semantic element, create a chunk
                    if tag_name and tag_name in self.semantic_elements:
                        chunk_content = content[node.start_byte:node.end_byte]
                        
                        # Only create chunk if it meets size criteria
                        if len(chunk_content) >= self.config.min_chunk_size:
                            chunk = self._create_chunk_from_node(node, content, context, ChunkType.TEXT_BLOCK)
                            if chunk and self._should_include_chunk(chunk):
                                chunks.append(chunk)
                    
                    # Always process children to find nested semantic elements
                    for child in node.children:
                        extract_semantic_chunks(child, depth + 1)
                else:
                    # Process children of non-element nodes
                    for child in node.children:
                        extract_semantic_chunks(child, depth + 1)
            
            extract_semantic_chunks(tree.root_node)
            return chunks
            
        except Exception as e:
            logger.warning(f"Error in semantic element extraction: {e}")
            # Fall back to default parsing if there's an error
            return super().parse(content, context)
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from HTML AST node"""
        if node.type == 'element':
            tag_name = self._extract_tag_name(node, content)
            if tag_name:
                attributes = self._extract_attributes(node, content)
                
                # Use id attribute if available
                if 'id' in attributes:
                    return f"{tag_name}#{attributes['id']}"
                
                # Use class if available
                if 'class' in attributes:
                    classes = attributes['class'].split()
                    if classes:
                        return f"{tag_name}.{classes[0]}"
                
                # Use text content for headings
                if tag_name in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
                    text_content = self._extract_text_content(node, content)
                    if text_content:
                        words = text_content.split()[:3]
                        return f"{tag_name}_{'_'.join(words)}"
                
                return tag_name
        
        return node.type
    
    def _extract_tag_name(self, node, content: str) -> Optional[str]:
        """Extract tag name from an element node"""
        if node.type != 'element':
            return None
        
        for child in node.children:
            if child.type == 'start_tag':
                for grandchild in child.children:
                    if grandchild.type == 'tag_name':
                        return content[grandchild.start_byte:grandchild.end_byte].lower()
        return None
    
    def _extract_attributes(self, node, content: str) -> Dict[str, str]:
        """Extract attributes from an HTML element"""
        attributes = {}
        
        for child in node.children:
            if child.type == 'start_tag':
                for grandchild in child.children:
                    if grandchild.type == 'attribute':
                        attr_name, attr_value = self._parse_attribute(grandchild, content)
                        if attr_name:
                            attributes[attr_name] = attr_value or ''
        return attributes
    
    def _parse_attribute(self, attr_node, content: str) -> tuple:
        """Parse an attribute node to extract name and value"""
        attr_name = None
        attr_value = None
        
        for child in attr_node.children:
            if child.type == 'attribute_name':
                attr_name = content[child.start_byte:child.end_byte]
            elif child.type == 'attribute_value':
                attr_value = content[child.start_byte:child.end_byte]
                # Remove quotes if present
                if attr_value.startswith('"') and attr_value.endswith('"'):
                    attr_value = attr_value[1:-1]
                elif attr_value.startswith("'") and attr_value.endswith("'"):
                    attr_value = attr_value[1:-1]
        
        return attr_name, attr_value
    
    def _extract_text_content(self, node, content: str) -> str:
        """Extract just the text content from an HTML element"""
        text_parts = []
        
        def extract_text(n):
            if n.type == 'text':
                text_parts.append(content[n.start_byte:n.end_byte])
            elif n.type not in ['start_tag', 'end_tag']:
                for child in n.children:
                    extract_text(child)
        
        extract_text(node)
        return ' '.join(text_parts).strip()
    
    def _add_semantic_tags(self, chunk: SemanticChunk, node, content: str):
        """Add HTML-specific semantic tags"""
        super()._add_semantic_tags(chunk, node, content)
        
        if node.type == 'element':
            tag_name = self._extract_tag_name(node, content)
            attributes = self._extract_attributes(node, content)
            
            if tag_name:
                chunk.add_tag('html_element', source='tree_sitter')
                chunk.add_tag(tag_name, source='tree_sitter')
                
                # Add semantic tags based on tag name
                if tag_name in {'article', 'section'}:
                    chunk.add_tag('content_section', source='tree_sitter')
                elif tag_name in {'header', 'footer'}:
                    chunk.add_tag('page_structure', source='tree_sitter')
                elif tag_name == 'nav':
                    chunk.add_tag('navigation', source='tree_sitter')
                elif tag_name in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
                    chunk.add_tag('heading', source='tree_sitter')
                    level = int(tag_name[1])
                    chunk.add_tag(f'heading_level_{level}', source='tree_sitter')
                elif tag_name in {'ul', 'ol', 'li'}:
                    chunk.add_tag('list', source='tree_sitter')
                elif tag_name == 'table':
                    chunk.add_tag('tabular_data', source='tree_sitter')
                elif tag_name == 'form':
                    chunk.add_tag('interactive', source='tree_sitter')
                elif tag_name in {'pre', 'code'}:
                    chunk.add_tag('code', source='tree_sitter')
                
                # Add semantic tags based on attributes
                if 'class' in attributes:
                    classes = attributes['class'].split()
                    for cls in classes:
                        chunk.add_tag(f'class_{cls}', source='tree_sitter')
                
                if 'id' in attributes:
                    chunk.add_tag(f'id_{attributes["id"]}', source='tree_sitter')
                
                # Store metadata
                chunk.metadata.update({
                    'tag_name': tag_name,
                    'attributes': attributes,
                    'css_classes': attributes.get('class', '').split(),
                    'element_id': attributes.get('id')
                })
    
    def _extract_dependencies(self, chunk: SemanticChunk, node, content: str):
        """Extract HTML dependencies (links, references)"""
        import re
        
        # Extract links as dependencies
        link_pattern = r'href=["\']([^"\']+)["\']'
        links = re.findall(link_pattern, chunk.content)
        
        for link in links:
            chunk.dependencies.append(f"links_to:{link}")
        
        # Extract src attributes (images, scripts, etc.)
        src_pattern = r'src=["\']([^"\']+)["\']'
        sources = re.findall(src_pattern, chunk.content)
        
        for src in sources:
            chunk.dependencies.append(f"references:{src}")
        
        if links:
            chunk.metadata['external_links'] = links
        if sources:
            chunk.metadata['referenced_sources'] = sources
    
    def _should_include_chunk(self, chunk: SemanticChunk) -> bool:
        """Override to be more inclusive with HTML semantic elements"""
        
        # Always include if it's a semantic HTML element
        tag_name = chunk.metadata.get('tag_name')
        if tag_name and tag_name in self.semantic_elements:
            return True
        
        # For non-semantic elements, use stricter criteria
        if tag_name in ['html', 'body', 'head']:
            # Only include if it's reasonably sized and not the whole document
            return len(chunk.content) < self.config.target_chunk_size * 3
        
        # Use parent logic for other chunks
        return super()._should_include_chunk(chunk)