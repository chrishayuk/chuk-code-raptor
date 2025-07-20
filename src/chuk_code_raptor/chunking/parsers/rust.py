# src/chuk_code_raptor/chunking/parsers/rust.py
"""
Rust Tree-sitter Parser
=======================

Rust parser using tree-sitter-rust for AST-based chunking.
Handles Rust-specific language features like traits, impls, and modules.
"""

import logging
from typing import List, Dict, Any, Optional

from ..tree_sitter_base import TreeSitterParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class RustParser(TreeSitterParser):
    """Rust parser using tree-sitter-rust"""
    
    def __init__(self, config):
        self.supported_languages = {'rust'}
        self.supported_extensions = {'.rs'}
        # Initialize parent class - this will call _initialize_tree_sitter
        try:
            super().__init__(config)
        except Exception as e:
            logger.warning(f"Rust parser failed to initialize tree-sitter: {e}")
            # Fall back to basic parser functionality
            self.parser = None
            self.language = None
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions) and self.parser is not None
    
    def _get_tree_sitter_language(self):
        """Get Rust tree-sitter language"""
        try:
            import tree_sitter
            import tree_sitter_rust
            language = tree_sitter.Language(tree_sitter_rust.language())
            
            # Check if the language version is compatible
            # This is a simple way to detect version issues
            try:
                # Try to create a parser with the language
                test_parser = tree_sitter.Parser(language)
                return language
            except Exception as version_error:
                logger.error(f"Tree-sitter version incompatibility for Rust: {version_error}")
                raise
                
        except ImportError as e:
            raise ImportError("tree-sitter-rust not installed. Install with: pip install tree-sitter-rust") from e
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """Rust AST node types to chunk types mapping"""
        return {
            'function_item': ChunkType.FUNCTION,
            'struct_item': ChunkType.CLASS,
            'enum_item': ChunkType.CLASS,
            'trait_item': ChunkType.CLASS,
            'impl_item': ChunkType.CLASS,
            'mod_item': ChunkType.CLASS,
            'use_declaration': ChunkType.IMPORT,
            'const_item': ChunkType.VARIABLE,
            'static_item': ChunkType.VARIABLE,
            'macro_definition': ChunkType.FUNCTION,
        }
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract Rust identifier from AST node"""
        # Look for identifier or type_identifier children
        for child in node.children:
            if child.type == 'identifier':
                return content[child.start_byte:child.end_byte]
            elif child.type == 'type_identifier':
                return content[child.start_byte:child.end_byte]
        
        # Special handling for different node types
        if node.type == 'use_declaration':
            return self._extract_use_path(node, content)
        
        return None
    
    def _extract_use_path(self, node, content: str) -> Optional[str]:
        """Extract the path from a use declaration"""
        for child in node.children:
            if child.type in ['scoped_identifier', 'identifier']:
                return content[child.start_byte:child.end_byte]
        return None
    
    def _add_semantic_tags(self, chunk: SemanticChunk, node, content: str):
        """Add Rust-specific semantic tags"""
        super()._add_semantic_tags(chunk, node, content)
        
        node_content = content[node.start_byte:node.end_byte]
        
        # Function tags
        if node.type == 'function_item':
            chunk.add_tag('function')
            
            # Check for async functions
            if 'async ' in node_content:
                chunk.add_tag('async')
            
            # Check for visibility
            if 'pub ' in node_content:
                chunk.add_tag('public')
            elif 'pub(crate)' in node_content:
                chunk.add_tag('crate_public')
            else:
                chunk.add_tag('private')
            
            # Check for unsafe
            if 'unsafe ' in node_content:
                chunk.add_tag('unsafe')
            
            # Check for extern
            if 'extern ' in node_content:
                chunk.add_tag('extern')
        
        # Struct tags
        elif node.type == 'struct_item':
            chunk.add_tag('struct')
            
            if 'pub ' in node_content:
                chunk.add_tag('public')
            
            # Check for derive macros
            if '#[derive(' in node_content:
                chunk.add_tag('derives_traits')
            
            # Check if it's a tuple struct
            if '(' in node_content and ')' in node_content:
                chunk.add_tag('tuple_struct')
        
        # Enum tags
        elif node.type == 'enum_item':
            chunk.add_tag('enum')
            
            if 'pub ' in node_content:
                chunk.add_tag('public')
            
            if '#[derive(' in node_content:
                chunk.add_tag('derives_traits')
        
        # Trait tags
        elif node.type == 'trait_item':
            chunk.add_tag('trait')
            
            if 'pub ' in node_content:
                chunk.add_tag('public')
            
            if 'unsafe ' in node_content:
                chunk.add_tag('unsafe')
        
        # Implementation tags
        elif node.type == 'impl_item':
            chunk.add_tag('implementation')
            
            # Check if it's a trait implementation
            if ' for ' in node_content:
                chunk.add_tag('trait_impl')
            else:
                chunk.add_tag('inherent_impl')
            
            if 'unsafe ' in node_content:
                chunk.add_tag('unsafe')
        
        # Module tags
        elif node.type == 'mod_item':
            chunk.add_tag('module')
            
            if 'pub ' in node_content:
                chunk.add_tag('public')
        
        # Use declaration tags
        elif node.type == 'use_declaration':
            chunk.add_tag('import')
            
            if 'pub ' in node_content:
                chunk.add_tag('re_export')
            
            # Check for external crates
            if '::' in node_content:
                chunk.add_tag('external_crate')
            
            # Check for glob imports
            if '*' in node_content:
                chunk.add_tag('glob_import')
        
        # Macro tags
        elif node.type == 'macro_definition':
            chunk.add_tag('macro')
            
            if 'pub ' in node_content:
                chunk.add_tag('public')
        
        # Constant and static tags
        elif node.type == 'const_item':
            chunk.add_tag('constant')
            if 'pub ' in node_content:
                chunk.add_tag('public')
        
        elif node.type == 'static_item':
            chunk.add_tag('static')
            
            if 'pub ' in node_content:
                chunk.add_tag('public')
            if 'mut ' in node_content:
                chunk.add_tag('mutable')
    
    def _extract_dependencies(self, chunk: SemanticChunk, node, content: str):
        """Extract dependencies from Rust nodes"""
        dependencies = []
        
        def find_dependencies(n):
            # Use dependencies
            if n.type == 'use_declaration':
                use_path = self._extract_use_path(n, content)
                if use_path:
                    dependencies.append(f"uses:{use_path}")
            
            # Function call dependencies
            elif n.type == 'call_expression':
                for child in n.children:
                    if child.type in ['identifier', 'scoped_identifier']:
                        func_name = content[child.start_byte:child.end_byte]
                        dependencies.append(f"calls:{func_name}")
                        break
            
            # Macro invocation dependencies
            elif n.type == 'macro_invocation':
                for child in n.children:
                    if child.type == 'identifier':
                        macro_name = content[child.start_byte:child.end_byte]
                        dependencies.append(f"invokes:{macro_name}")
                        break
            
            # Type dependencies
            elif n.type in ['type_identifier', 'generic_type']:
                type_name = content[n.start_byte:n.end_byte]
                dependencies.append(f"uses_type:{type_name}")
            
            # Trait bounds
            elif n.type == 'trait_bounds':
                for child in n.children:
                    if child.type == 'type_identifier':
                        trait_name = content[child.start_byte:child.end_byte]
                        dependencies.append(f"implements:{trait_name}")
            
            # Struct field access
            elif n.type == 'field_expression':
                for child in n.children:
                    if child.type == 'field_identifier':
                        field_name = content[child.start_byte:child.end_byte]
                        dependencies.append(f"accesses:{field_name}")
            
            # Recurse into children
            for child in n.children:
                find_dependencies(child)
        
        find_dependencies(node)
        chunk.dependencies.extend(list(set(dependencies)))  # Remove duplicates