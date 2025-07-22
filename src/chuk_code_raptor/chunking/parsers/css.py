# src/chuk_code_raptor/chunking/parsers/css.py
"""
CSS Parser
==========

CSS parser using tree-sitter-css for AST-based chunking.
Handles CSS rules, selectors, and properties with semantic understanding.
"""

import logging
from typing import Dict, List, Optional

from ..tree_sitter_base import TreeSitterParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class CSSParser(TreeSitterParser):
    """CSS parser using tree-sitter-css with semantic rule processing"""
    
    def __init__(self, config):
        # Set supported languages and extensions first
        self.supported_languages = {'css'}
        self.supported_extensions = {'.css', '.scss', '.sass', '.less'}
        
        # Then call parent init
        super().__init__(config)
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (self.parser is not None and 
                (language in self.supported_languages or 
                 file_extension in self.supported_extensions))
    
    def _get_tree_sitter_language(self):
        """Get CSS tree-sitter language with robust package handling"""
        from ..tree_sitter_base import get_tree_sitter_language_robust
        
        # Try different package sources for CSS
        package_candidates = [
            'tree_sitter_languages',      # Comprehensive package (preferred)
            'tree_sitter_language_pack',  # Alternative comprehensive
            'tree_sitter_css'             # Individual package
        ]
        
        language, package_used = get_tree_sitter_language_robust('css', package_candidates)
        
        if language is None:
            raise ImportError("No compatible tree-sitter CSS package found. Try: pip install tree-sitter-languages")
        
        self._package_used = package_used
        return language
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """CSS AST node types to chunk types mapping"""
        return {
            'rule_set': ChunkType.TEXT_BLOCK,
            'at_rule': ChunkType.TEXT_BLOCK,
            'media_statement': ChunkType.TEXT_BLOCK,
            'keyframes_statement': ChunkType.TEXT_BLOCK,
            'import_statement': ChunkType.IMPORT,
            'declaration': ChunkType.TEXT_BLOCK,
        }
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from CSS AST node"""
        if node.type == 'rule_set':
            # Extract selector(s)
            for child in node.children:
                if child.type == 'selectors':
                    selector_text = content[child.start_byte:child.end_byte].strip()
                    # Clean up selector for identifier
                    clean_selector = selector_text.replace('\n', ' ').replace('  ', ' ')
                    if len(clean_selector) > 50:
                        clean_selector = clean_selector[:47] + "..."
                    return f"rule_{clean_selector}"
            return "css_rule"
        
        elif node.type == 'at_rule':
            # Extract at-rule name
            rule_text = content[node.start_byte:node.end_byte].strip()
            if rule_text.startswith('@'):
                rule_name = rule_text.split()[0] if ' ' in rule_text else rule_text
                return f"at_rule_{rule_name[1:]}"  # Remove @
            return "at_rule"
        
        elif node.type == 'media_statement':
            return "media_query"
        
        elif node.type == 'keyframes_statement':
            # Extract keyframes name
            for child in node.children:
                if child.type == 'keyframes_name':
                    name = content[child.start_byte:child.end_byte].strip()
                    return f"keyframes_{name}"
            return "keyframes"
        
        elif node.type == 'import_statement':
            return "import"
        
        elif node.type == 'declaration':
            # Extract property name
            for child in node.children:
                if child.type == 'property_name':
                    prop_name = content[child.start_byte:child.end_byte].strip()
                    return f"prop_{prop_name}"
            return "declaration"
        
        return node.type
    
    def _add_semantic_tags(self, chunk: SemanticChunk, node, content: str):
        """Add CSS-specific semantic tags"""
        super()._add_semantic_tags(chunk, node, content)
        
        chunk.add_tag('css', source='tree_sitter')
        
        if node.type == 'rule_set':
            chunk.add_tag('css_rule', source='tree_sitter')
            
            # Analyze selector types
            selectors = self._extract_selectors(node, content)
            if selectors:
                chunk.metadata['selectors'] = selectors
                chunk.metadata['css_selector'] = ', '.join(selectors)  # Add main selector
                
                # Calculate selector specificity
                total_specificity = sum(self._calculate_selector_specificity(sel) for sel in selectors)
                chunk.metadata['selector_specificity'] = total_specificity / len(selectors)
                
                # Tag based on selector types
                for selector in selectors:
                    if selector.startswith('#'):
                        chunk.add_tag('id_selector', source='tree_sitter')
                    elif selector.startswith('.'):
                        chunk.add_tag('class_selector', source='tree_sitter')
                    elif selector.startswith('@'):
                        chunk.add_tag('at_rule_selector', source='tree_sitter')
                    elif ':' in selector and not selector.startswith('@'):
                        chunk.add_tag('pseudo_selector', source='tree_sitter')
                    elif '[' in selector:
                        chunk.add_tag('attribute_selector', source='tree_sitter')
                    else:
                        chunk.add_tag('element_selector', source='tree_sitter')
            
            # Extract CSS properties
            properties = self._extract_css_properties(node, content)
            if properties:
                chunk.metadata['css_properties'] = properties
                
                # Detect responsive features
                if any('width' in prop or 'flex' in prop or 'grid' in prop for prop in properties):
                    chunk.add_tag('css_layout', source='tree_sitter')
        
        elif node.type == 'at_rule':
            chunk.add_tag('at_rule', source='tree_sitter')
            
            # Specific at-rule types
            rule_content = content[node.start_byte:node.end_byte].lower()
            if rule_content.startswith('@media'):
                chunk.add_tag('media_query', source='tree_sitter')
                chunk.add_tag('breakpoint_rule', source='tree_sitter')
                if 'max-width' in rule_content or 'min-width' in rule_content:
                    chunk.add_tag('responsive_breakpoint', source='tree_sitter')
                if 'prefers-color-scheme' in rule_content:
                    chunk.add_tag('color_scheme_preference', source='tree_sitter')
                if 'prefers-reduced-motion' in rule_content:
                    chunk.add_tag('motion_preference', source='tree_sitter')
            elif rule_content.startswith('@import'):
                chunk.add_tag('import_rule', source='tree_sitter')
            elif rule_content.startswith('@font-face'):
                chunk.add_tag('font_face', source='tree_sitter')
            elif rule_content.startswith('@keyframes'):
                chunk.add_tag('animation', source='tree_sitter')
                chunk.add_tag('css_animation', source='tree_sitter')
        
        elif node.type == 'media_statement':
            chunk.add_tag('media_query', source='tree_sitter')
            chunk.add_tag('responsive_design', source='tree_sitter')
        
        elif node.type == 'keyframes_statement':
            chunk.add_tag('keyframes', source='tree_sitter')
            chunk.add_tag('animation', source='tree_sitter')
            chunk.add_tag('css_animation', source='tree_sitter')
        
        elif node.type == 'import_statement':
            chunk.add_tag('import', source='tree_sitter')
            chunk.add_tag('css_import', source='tree_sitter')
        
        elif node.type == 'declaration':
            chunk.add_tag('css_property', source='tree_sitter')
    
    def _calculate_selector_specificity(self, selector: str) -> int:
        """Calculate CSS selector specificity"""
        import re
        
        # Remove pseudo-elements and pseudo-classes for counting
        clean_selector = re.sub(r'::[a-zA-Z-]+', '', selector)
        clean_selector = re.sub(r':[a-zA-Z-]+(?:\([^)]*\))?', '', clean_selector)
        
        # Count components
        id_count = len(re.findall(r'#[a-zA-Z][\w-]*', clean_selector))
        class_count = len(re.findall(r'\.[a-zA-Z][\w-]*', clean_selector))
        attr_count = len(re.findall(r'\[[^\]]+\]', clean_selector))
        element_count = len(re.findall(r'\b[a-zA-Z][\w-]*(?![#\.\[])', clean_selector))
        
        # CSS specificity calculation: (inline, ids, classes+attrs, elements)
        # We return just the numeric value for comparison
        return (id_count * 100) + ((class_count + attr_count) * 10) + element_count
    
    def _extract_css_properties(self, rule_node, content: str) -> List[str]:
        """Extract CSS property names from a rule"""
        properties = []
        
        for child in rule_node.children:
            if child.type == 'declaration_list' or child.type == 'block':
                for grandchild in child.children:
                    if grandchild.type == 'declaration':
                        for ggchild in grandchild.children:
                            if ggchild.type == 'property_name':
                                prop_name = content[ggchild.start_byte:ggchild.end_byte]
                                properties.append(prop_name)
                                break
        
        return properties
    
    def get_html_css_relationships(self, css_chunks, html_chunks):
        """Analyze relationships between CSS and HTML chunks"""
        relationships = {}
        
        for css_chunk in css_chunks:
            css_id = css_chunk.id
            matching_html = []
            
            # Get selectors from this CSS chunk
            selectors = css_chunk.metadata.get('selectors', [])
            
            for selector in selectors:
                # Find HTML elements that match this selector
                for html_chunk in html_chunks:
                    if self._selector_matches_html(selector, html_chunk):
                        matching_html.append(html_chunk.id)
            
            if matching_html:
                relationships[css_id] = list(set(matching_html))  # Remove duplicates
        
        return relationships
    
    def _selector_matches_html(self, css_selector: str, html_chunk) -> bool:
        """Check if a CSS selector matches an HTML element"""
        # Get HTML element metadata
        tag_name = html_chunk.metadata.get('tag_name', '').lower()
        element_id = html_chunk.metadata.get('element_id', '')
        css_classes = html_chunk.metadata.get('css_classes', [])
        
        # Simple selector matching
        selector = css_selector.strip()
        
        # ID selector (#id)
        if selector.startswith('#'):
            target_id = selector[1:]
            return element_id == target_id
        
        # Class selector (.class)
        elif selector.startswith('.'):
            target_class = selector[1:]
            return target_class in css_classes
        
        # Element selector (tagname)
        elif selector.isalpha():
            return tag_name == selector.lower()
        
        # Simple descendant selectors (basic support)
        elif ' ' in selector:
            parts = selector.split()
            # For now, just check if the last part matches
            last_part = parts[-1]
            return self._selector_matches_html(last_part, html_chunk)
        
        return False
    
    def _extract_selectors(self, rule_node, content: str) -> List[str]:
        """Extract selectors from a CSS rule"""
        selectors = []
        
        for child in rule_node.children:
            if child.type == 'selectors':
                selector_text = content[child.start_byte:child.end_byte]
                # Split on commas and clean up
                for selector in selector_text.split(','):
                    clean_selector = selector.strip()
                    if clean_selector:
                        selectors.append(clean_selector)
                break
        
        return selectors
    
    def _extract_dependencies(self, chunk: SemanticChunk, node, content: str):
        """Extract CSS dependencies (imports, url references)"""
        import re
        
        chunk_content = chunk.content
        
        # Extract @import statements
        import_pattern = r'@import\s+["\']([^"\']+)["\']'
        imports = re.findall(import_pattern, chunk_content)
        for imp in imports:
            chunk.dependencies.append(f"imports:{imp}")
        
        # Extract url() references
        url_pattern = r'url\(["\']?([^"\')\s]+)["\']?\)'
        urls = re.findall(url_pattern, chunk_content)
        for url in urls:
            if not url.startswith('data:'):  # Skip data URLs
                chunk.dependencies.append(f"references:{url}")
        
        # Store metadata
        if imports:
            chunk.metadata['css_imports'] = imports
        if urls:
            chunk.metadata['url_references'] = urls