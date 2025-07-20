# src/chuk_code_raptor/chunking/parsers/css.py
"""
CSS Parser
==========

CSS parser using tree-sitter-css for AST-based chunking.
Supports CSS selectors, rules, media queries, and HTML dependency tracking.
"""

import logging
import re
from typing import Dict, List, Optional, Set

from ..tree_sitter_base import TreeSitterParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class CSSParser(TreeSitterParser):
    """CSS parser using tree-sitter-css with HTML dependency tracking"""
    
    def __init__(self, config):
        # Call parent first
        super().__init__(config)
        
        # Then set our specific supported languages and extensions
        self.supported_languages = {'css'}
        self.supported_extensions = {'.css', '.scss', '.sass', '.less'}
        
        # CSS semantic categories for chunking
        self.css_categories = {
            # Structural elements
            'layout': ['grid', 'flex', 'float', 'position', 'display'],
            'spacing': ['margin', 'padding', 'gap', 'space'],
            'sizing': ['width', 'height', 'min-', 'max-', 'size'],
            'typography': ['font', 'text', 'line-height', 'letter-spacing'],
            'colors': ['color', 'background', 'border-color', 'fill', 'stroke'],
            'effects': ['shadow', 'blur', 'opacity', 'transform', 'transition'],
            'responsive': ['@media', '@container', '@supports'],
            'animation': ['@keyframes', 'animation', 'transition']
        }
        
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_tree_sitter_language(self):
        """Get CSS tree-sitter language"""
        try:
            import tree_sitter
            import tree_sitter_css
            return tree_sitter.Language(tree_sitter_css.language())
        except ImportError as e:
            raise ImportError("tree-sitter-css not installed. Install with: pip install tree-sitter-css") from e
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """CSS AST node types to chunk types mapping"""
        return {
            'rule_set': ChunkType.TEXT_BLOCK,
            'at_rule': ChunkType.TEXT_BLOCK,
            'media_statement': ChunkType.TEXT_BLOCK,
            'keyframes_statement': ChunkType.TEXT_BLOCK,
            'import_statement': ChunkType.IMPORT,
        }
    
    def parse(self, content: str, context) -> List[SemanticChunk]:
        """Override parse method to use CSS-specific element extraction"""
        
        # Try CSS-specific semantic element extraction
        css_chunks = self._process_css_elements(content, context)
        
        # If we got good CSS chunks, use them
        if css_chunks:
            return self._post_process(css_chunks)
        
        # Otherwise fall back to default tree-sitter parsing
        return super().parse(content, context)
    
    def _process_css_elements(self, content: str, context) -> List[SemanticChunk]:
        """Process CSS content to extract meaningful chunks"""
        try:
            import tree_sitter
            import tree_sitter_css
            
            language = tree_sitter.Language(tree_sitter_css.language())
            parser = tree_sitter.Parser(language)
            tree = parser.parse(bytes(content, 'utf8'))
            
            chunks = []
            
            def extract_css_chunks(node, depth=0):
                """Extract chunks from CSS AST nodes"""
                chunk_type = None
                
                if node.type in ['rule_set', 'at_rule', 'media_statement', 'keyframes_statement']:
                    chunk_type = ChunkType.TEXT_BLOCK
                elif node.type == 'import_statement':
                    chunk_type = ChunkType.IMPORT
                
                if chunk_type and node.end_byte > node.start_byte:
                    chunk_content = content[node.start_byte:node.end_byte]
                    
                    # Only create chunk if it meets size criteria or is important
                    if (len(chunk_content) >= self.config.min_chunk_size or 
                        self._is_important_css_rule(node, content)):
                        
                        chunk = self._create_chunk_from_node(node, content, context, chunk_type)
                        if chunk and self._should_include_chunk(chunk):
                            chunks.append(chunk)
                
                # Process children
                for child in node.children:
                    extract_css_chunks(child, depth + 1)
            
            extract_css_chunks(tree.root_node)
            return chunks
            
        except Exception as e:
            logger.warning(f"Error in CSS element extraction: {e}")
            return super().parse(content, context)
    
    def _is_important_css_rule(self, node, content: str) -> bool:
        """Check if a CSS rule is important regardless of size"""
        chunk_content = content[node.start_byte:node.end_byte].lower()
        
        # Important CSS patterns
        important_patterns = [
            '@media', '@keyframes', '@import', '@supports',
            'css-grid', 'flexbox', 'animation', 'transition',
            ':root', ':host', '::before', '::after'
        ]
        
        return any(pattern in chunk_content for pattern in important_patterns)
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from CSS AST node"""
        chunk_content = content[node.start_byte:node.end_byte]
        
        if node.type == 'rule_set':
            # Extract selector
            selector = self._extract_css_selector(node, content)
            if selector:
                return self._normalize_css_selector(selector)
        
        elif node.type == 'media_statement':
            # Extract media query
            media_query = self._extract_media_query(node, content)
            return f"media_{media_query}" if media_query else "media_query"
        
        elif node.type == 'keyframes_statement':
            # Extract keyframes name
            keyframes_name = self._extract_keyframes_name(node, content)
            return f"keyframes_{keyframes_name}" if keyframes_name else "keyframes"
        
        elif node.type == 'import_statement':
            # Extract imported file
            import_path = self._extract_import_path(node, content)
            return f"import_{import_path}" if import_path else "import"
        
        elif node.type == 'at_rule':
            # Extract at-rule name
            at_rule_name = self._extract_at_rule_name(node, content)
            return f"at_rule_{at_rule_name}" if at_rule_name else "at_rule"
        
        return node.type
    
    def _extract_css_selector(self, node, content: str) -> Optional[str]:
        """Extract CSS selector from a rule set"""
        for child in node.children:
            if child.type == 'selectors':
                selector = content[child.start_byte:child.end_byte].strip()
                return selector
        return None
    
    def _normalize_css_selector(self, selector: str) -> str:
        """Normalize CSS selector for use as identifier"""
        # Replace special characters and spaces
        normalized = re.sub(r'[^a-zA-Z0-9_-]', '_', selector)
        # Remove multiple underscores
        normalized = re.sub(r'_+', '_', normalized)
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        return normalized or 'selector'
    
    def _extract_media_query(self, node, content: str) -> Optional[str]:
        """Extract media query condition"""
        for child in node.children:
            if child.type == 'media_query_list':
                query = content[child.start_byte:child.end_byte].strip()
                # Simplify for identifier
                query = re.sub(r'[^\w-]', '_', query)
                return query[:50]  # Limit length
        return None
    
    def _extract_keyframes_name(self, node, content: str) -> Optional[str]:
        """Extract keyframes animation name"""
        for child in node.children:
            if child.type == 'keyframes_name':
                name = content[child.start_byte:child.end_byte].strip()
                return re.sub(r'[^\w-]', '_', name)
        return None
    
    def _extract_import_path(self, node, content: str) -> Optional[str]:
        """Extract imported file path"""
        import_content = content[node.start_byte:node.end_byte]
        # Extract filename from import statement
        match = re.search(r'["\']([^"\']+)["\']', import_content)
        if match:
            path = match.group(1)
            # Extract just filename for identifier
            filename = path.split('/')[-1].replace('.css', '')
            return re.sub(r'[^\w-]', '_', filename)
        return None
    
    def _extract_at_rule_name(self, node, content: str) -> Optional[str]:
        """Extract at-rule name (e.g., @supports, @container)"""
        at_content = content[node.start_byte:node.end_byte]
        match = re.match(r'@(\w+)', at_content)
        if match:
            return match.group(1)
        return None
    
    def _add_semantic_tags(self, chunk: SemanticChunk, node, content: str):
        """Add CSS-specific semantic tags"""
        super()._add_semantic_tags(chunk, node, content)
        
        chunk.add_tag('css', source='tree_sitter')
        
        if node.type == 'rule_set':
            chunk.add_tag('css_rule', source='tree_sitter')
            
            # Analyze selector type
            selector = self._extract_css_selector(node, content)
            if selector:
                self._add_selector_tags(chunk, selector)
                
            # Analyze CSS properties
            properties = self._extract_css_properties(node, content)
            self._add_property_tags(chunk, properties)
            
            # Store metadata
            chunk.metadata.update({
                'css_selector': selector,
                'css_properties': properties,
                'selector_specificity': self._calculate_specificity(selector),
                'html_targets': self._extract_html_targets(selector)
            })
        
        elif node.type == 'media_statement':
            chunk.add_tag('css_media_query', source='tree_sitter')
            chunk.add_tag('responsive_design', source='tree_sitter')
            
            media_query = self._extract_media_query(node, content)
            if media_query:
                chunk.metadata['media_query'] = media_query
                self._add_media_query_tags(chunk, media_query)
        
        elif node.type == 'keyframes_statement':
            chunk.add_tag('css_animation', source='tree_sitter')
            chunk.add_tag('keyframes', source='tree_sitter')
            
            keyframes_name = self._extract_keyframes_name(node, content)
            if keyframes_name:
                chunk.metadata['animation_name'] = keyframes_name
        
        elif node.type == 'import_statement':
            chunk.add_tag('css_import', source='tree_sitter')
            
            import_path = self._extract_import_path(node, content)
            if import_path:
                chunk.metadata['imported_file'] = import_path
        
        elif node.type == 'at_rule':
            chunk.add_tag('css_at_rule', source='tree_sitter')
            
            at_rule_name = self._extract_at_rule_name(node, content)
            if at_rule_name:
                chunk.add_tag(f'at_rule_{at_rule_name}', source='tree_sitter')
                chunk.metadata['at_rule_type'] = at_rule_name
    
    def _add_selector_tags(self, chunk: SemanticChunk, selector: str):
        """Add tags based on CSS selector type"""
        selector_lower = selector.lower()
        
        # Selector type tags
        if selector.startswith('#'):
            chunk.add_tag('id_selector', source='tree_sitter')
        elif selector.startswith('.'):
            chunk.add_tag('class_selector', source='tree_sitter')
        elif selector.startswith('@'):
            chunk.add_tag('at_rule_selector', source='tree_sitter')
        elif re.match(r'^[a-zA-Z]', selector):
            chunk.add_tag('element_selector', source='tree_sitter')
        
        # Pseudo selector tags
        if '::' in selector:
            chunk.add_tag('pseudo_element', source='tree_sitter')
        elif ':' in selector:
            chunk.add_tag('pseudo_class', source='tree_sitter')
        
        # Combinator tags
        if '>' in selector:
            chunk.add_tag('child_combinator', source='tree_sitter')
        elif '+' in selector:
            chunk.add_tag('adjacent_sibling', source='tree_sitter')
        elif '~' in selector:
            chunk.add_tag('general_sibling', source='tree_sitter')
        elif ' ' in selector.strip():
            chunk.add_tag('descendant_combinator', source='tree_sitter')
    
    def _add_property_tags(self, chunk: SemanticChunk, properties: List[str]):
        """Add tags based on CSS properties"""
        for prop in properties:
            prop_lower = prop.lower()
            
            # Categorize properties
            for category, keywords in self.css_categories.items():
                if any(keyword in prop_lower for keyword in keywords):
                    chunk.add_tag(f'css_{category}', source='tree_sitter')
    
    def _add_media_query_tags(self, chunk: SemanticChunk, media_query: str):
        """Add tags based on media query features"""
        query_lower = media_query.lower()
        
        # Device type tags
        if 'screen' in query_lower:
            chunk.add_tag('screen_media', source='tree_sitter')
        elif 'print' in query_lower:
            chunk.add_tag('print_media', source='tree_sitter')
        
        # Responsive breakpoint tags
        if 'min-width' in query_lower or 'max-width' in query_lower:
            chunk.add_tag('breakpoint_rule', source='tree_sitter')
        
        # Feature detection tags
        if 'prefers-color-scheme' in query_lower:
            chunk.add_tag('color_scheme_preference', source='tree_sitter')
        elif 'prefers-reduced-motion' in query_lower:
            chunk.add_tag('motion_preference', source='tree_sitter')
    
    def _extract_css_properties(self, node, content: str) -> List[str]:
        """Extract CSS property names from a rule set"""
        properties = []
        
        def extract_properties(n):
            if n.type == 'declaration':
                for child in n.children:
                    if child.type == 'property_name':
                        prop_name = content[child.start_byte:child.end_byte]
                        properties.append(prop_name)
            
            for child in n.children:
                extract_properties(child)
        
        extract_properties(node)
        return properties
    
    def _calculate_specificity(self, selector: str) -> int:
        """Calculate CSS selector specificity"""
        if not selector:
            return 0
        
        # Simple specificity calculation
        # IDs = 100, classes/attributes/pseudo-classes = 10, elements = 1
        specificity = 0
        specificity += len(re.findall(r'#\w+', selector)) * 100  # IDs
        specificity += len(re.findall(r'\.\w+', selector)) * 10   # Classes
        specificity += len(re.findall(r':\w+', selector)) * 10    # Pseudo-classes
        specificity += len(re.findall(r'\[\w+', selector)) * 10   # Attributes
        specificity += len(re.findall(r'\b[a-z]+\b', selector))   # Elements
        
        return specificity
    
    def _extract_html_targets(self, selector: str) -> List[str]:
        """Extract HTML elements/classes/IDs that this CSS targets"""
        if not selector:
            return []
        
        targets = []
        
        # Extract IDs
        ids = re.findall(r'#([\w-]+)', selector)
        targets.extend([f'id:{id_name}' for id_name in ids])
        
        # Extract classes
        classes = re.findall(r'\.([\w-]+)', selector)
        targets.extend([f'class:{class_name}' for class_name in classes])
        
        # Extract elements
        elements = re.findall(r'\b([a-z]+)\b', selector)
        targets.extend([f'element:{elem}' for elem in elements if elem not in ['and', 'or', 'not']])
        
        return targets
    
    def _extract_dependencies(self, chunk: SemanticChunk, node, content: str):
        """Extract CSS dependencies (imports, HTML targets)"""
        
        # Extract CSS imports
        if node.type == 'import_statement':
            import_path = self._extract_import_path(node, content)
            if import_path:
                chunk.dependencies.append(f"imports:{import_path}.css")
        
        # Extract HTML dependencies from selectors
        if node.type == 'rule_set':
            selector = chunk.metadata.get('css_selector')
            html_targets = chunk.metadata.get('html_targets', [])
            
            for target in html_targets:
                chunk.dependencies.append(f"targets_html:{target}")
        
        # Extract font imports
        font_imports = re.findall(r'url\(["\']?([^"\']+\.(?:woff|woff2|ttf|eot))["\']?\)', chunk.content)
        for font in font_imports:
            chunk.dependencies.append(f"references:{font}")
        
        # Extract image references
        image_refs = re.findall(r'url\(["\']?([^"\']+\.(?:png|jpg|jpeg|gif|svg|webp))["\']?\)', chunk.content)
        for image in image_refs:
            chunk.dependencies.append(f"references:{image}")
    
    def _should_include_chunk(self, chunk: SemanticChunk) -> bool:
        """Override to be more inclusive with important CSS rules"""
        
        # Always include imports and at-rules
        if chunk.chunk_type == ChunkType.IMPORT or any(tag.name.startswith('at_rule_') for tag in chunk.semantic_tags):
            return True
        
        # Always include media queries and animations
        if any(tag.name in ['css_media_query', 'css_animation'] for tag in chunk.semantic_tags):
            return True
        
        # Include high-specificity selectors
        specificity = chunk.metadata.get('selector_specificity', 0)
        if specificity > 50:  # High specificity threshold
            return True
        
        # Use parent logic for other chunks
        return super()._should_include_chunk(chunk)
    
    def get_html_css_relationships(self, css_chunks: List[SemanticChunk], 
                                  html_chunks: List[SemanticChunk]) -> Dict[str, List[str]]:
        """Analyze relationships between HTML and CSS chunks"""
        relationships = {}
        
        # Build HTML element maps
        html_ids = {}
        html_classes = {}
        html_elements = {}
        
        for html_chunk in html_chunks:
            chunk_id = html_chunk.id
            metadata = html_chunk.metadata or {}
            
            # Map HTML IDs
            element_id = metadata.get('element_id')
            if element_id:
                html_ids[element_id] = chunk_id
            
            # Map HTML classes
            css_classes = metadata.get('css_classes', [])
            for css_class in css_classes:
                if css_class not in html_classes:
                    html_classes[css_class] = []
                html_classes[css_class].append(chunk_id)
            
            # Map HTML elements
            tag_name = metadata.get('tag_name')
            if tag_name:
                if tag_name not in html_elements:
                    html_elements[tag_name] = []
                html_elements[tag_name].append(chunk_id)
        
        # Map CSS to HTML relationships
        for css_chunk in css_chunks:
            css_id = css_chunk.id
            html_targets = css_chunk.metadata.get('html_targets', [])
            
            related_html = []
            
            for target in html_targets:
                target_type, target_name = target.split(':', 1)
                
                if target_type == 'id' and target_name in html_ids:
                    related_html.append(html_ids[target_name])
                elif target_type == 'class' and target_name in html_classes:
                    related_html.extend(html_classes[target_name])
                elif target_type == 'element' and target_name in html_elements:
                    related_html.extend(html_elements[target_name])
            
            if related_html:
                relationships[css_id] = list(set(related_html))  # Remove duplicates
        
        return relationships