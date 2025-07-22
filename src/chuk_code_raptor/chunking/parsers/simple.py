# src/chuk_code_raptor/chunking/parsers/simple.py
"""
Simple Tree-sitter Parsers
==========================

Simple implementations for languages that don't have dedicated parsers yet.
These use basic tree-sitter parsing with minimal language-specific customization.
"""

import logging
from typing import Dict, List, Optional

from ..tree_sitter_base import TreeSitterParser
from ..semantic_chunk import SemanticChunk
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class SimpleTreeSitterParser(TreeSitterParser):
    """Base class for simple tree-sitter parsers"""
    
    def __init__(self, config, language_name: str, package_name: str = None):
        self.language_name = language_name
        self.package_name = package_name or f"tree_sitter_{language_name}"
        super().__init__(config)
    
    def _get_tree_sitter_language(self):
        """Get tree-sitter language for this parser"""
        try:
            import tree_sitter
            
            # Try to import the language package
            try:
                module = __import__(self.package_name, fromlist=['language'])
                if hasattr(module, 'language'):
                    return tree_sitter.Language(module.language())
                else:
                    raise ImportError(f"No language function in {self.package_name}")
            except ImportError:
                raise ImportError(f"{self.package_name} not installed. Install with: pip install {self.package_name.replace('_', '-')}")
                
        except ImportError as e:
            raise ImportError(f"tree-sitter not available for {self.language_name}") from e
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """Basic chunk node types - override in subclasses for language-specific types"""
        return {
            'function_definition': ChunkType.FUNCTION,
            'method_definition': ChunkType.METHOD,
            'class_definition': ChunkType.CLASS,
            'struct_definition': ChunkType.CLASS,
            'enum_definition': ChunkType.CLASS,
            'interface_definition': ChunkType.CLASS,
            'trait_definition': ChunkType.CLASS,
            'impl_item': ChunkType.CLASS,
            'type_definition': ChunkType.CLASS,
            'import_statement': ChunkType.IMPORT,
            'import_declaration': ChunkType.IMPORT,
            'use_declaration': ChunkType.IMPORT,
            'package_declaration': ChunkType.IMPORT,
            'variable_declaration': ChunkType.VARIABLE,
            'const_declaration': ChunkType.VARIABLE,
            'let_declaration': ChunkType.VARIABLE,
        }
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Basic identifier extraction - works for most languages"""
        # Look for identifier child nodes
        for child in node.children:
            if child.type == 'identifier':
                return content[child.start_byte:child.end_byte]
        
        # Fallback to node type
        return node.type

class RustParser(SimpleTreeSitterParser):
    """Simple Rust parser"""
    
    def __init__(self, config):
        self.supported_languages = {'rust', 'rs'}
        self.supported_extensions = {'.rs'}
        super().__init__(config, 'rust')
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """Rust-specific chunk types"""
        base_types = super()._get_chunk_node_types()
        rust_types = {
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
        base_types.update(rust_types)
        return base_types

class GoParser(SimpleTreeSitterParser):
    """Simple Go parser"""
    
    def __init__(self, config):
        self.supported_languages = {'go'}
        self.supported_extensions = {'.go'}
        super().__init__(config, 'go')
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """Go-specific chunk types"""
        base_types = super()._get_chunk_node_types()
        go_types = {
            'function_declaration': ChunkType.FUNCTION,
            'method_declaration': ChunkType.METHOD,
            'type_declaration': ChunkType.CLASS,
            'import_declaration': ChunkType.IMPORT,
            'var_declaration': ChunkType.VARIABLE,
            'const_declaration': ChunkType.VARIABLE,
        }
        base_types.update(go_types)
        return base_types

class JavaParser(SimpleTreeSitterParser):
    """Simple Java parser"""
    
    def __init__(self, config):
        self.supported_languages = {'java'}
        self.supported_extensions = {'.java'}
        super().__init__(config, 'java')
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """Java-specific chunk types"""
        base_types = super()._get_chunk_node_types()
        java_types = {
            'method_declaration': ChunkType.METHOD,
            'constructor_declaration': ChunkType.METHOD,
            'class_declaration': ChunkType.CLASS,
            'interface_declaration': ChunkType.CLASS,
            'enum_declaration': ChunkType.CLASS,
            'import_declaration': ChunkType.IMPORT,
            'package_declaration': ChunkType.IMPORT,
            'field_declaration': ChunkType.VARIABLE,
        }
        base_types.update(java_types)
        return base_types

class CppParser(SimpleTreeSitterParser):
    """Simple C++ parser"""
    
    def __init__(self, config):
        self.supported_languages = {'cpp', 'c++', 'cxx'}
        self.supported_extensions = {'.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.h++'}
        super().__init__(config, 'cpp')
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """C++-specific chunk types"""
        base_types = super()._get_chunk_node_types()
        cpp_types = {
            'function_definition': ChunkType.FUNCTION,
            'function_declarator': ChunkType.FUNCTION,
            'class_specifier': ChunkType.CLASS,
            'struct_specifier': ChunkType.CLASS,
            'enum_specifier': ChunkType.CLASS,
            'namespace_definition': ChunkType.CLASS,
            'preproc_include': ChunkType.IMPORT,
            'declaration': ChunkType.VARIABLE,
        }
        base_types.update(cpp_types)
        return base_types

class CParser(SimpleTreeSitterParser):
    """Simple C parser"""
    
    def __init__(self, config):
        self.supported_languages = {'c'}
        self.supported_extensions = {'.c', '.h'}
        super().__init__(config, 'c')
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """C-specific chunk types"""
        base_types = super()._get_chunk_node_types()
        c_types = {
            'function_definition': ChunkType.FUNCTION,
            'function_declarator': ChunkType.FUNCTION,
            'struct_specifier': ChunkType.CLASS,
            'enum_specifier': ChunkType.CLASS,
            'union_specifier': ChunkType.CLASS,
            'preproc_include': ChunkType.IMPORT,
            'declaration': ChunkType.VARIABLE,
        }
        base_types.update(c_types)
        return base_types

class YAMLParser(SimpleTreeSitterParser):
    """Simple YAML parser"""
    
    def __init__(self, config):
        self.supported_languages = {'yaml', 'yml'}
        self.supported_extensions = {'.yaml', '.yml'}
        super().__init__(config, 'yaml')
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """YAML-specific chunk types"""
        return {
            'block_mapping': ChunkType.TEXT_BLOCK,
            'block_sequence': ChunkType.TEXT_BLOCK,
            'flow_mapping': ChunkType.TEXT_BLOCK,
            'flow_sequence': ChunkType.TEXT_BLOCK,
            'document': ChunkType.TEXT_BLOCK,
        }

class TOMLParser(SimpleTreeSitterParser):
    """Simple TOML parser"""
    
    def __init__(self, config):
        self.supported_languages = {'toml'}
        self.supported_extensions = {'.toml'}
        super().__init__(config, 'toml')
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """TOML-specific chunk types"""
        return {
            'table': ChunkType.TEXT_BLOCK,
            'array_table': ChunkType.TEXT_BLOCK,
            'pair': ChunkType.TEXT_BLOCK,
        }

class CSSParser(SimpleTreeSitterParser):
    """Simple CSS parser"""
    
    def __init__(self, config):
        self.supported_languages = {'css'}
        self.supported_extensions = {'.css', '.scss', '.sass', '.less'}
        super().__init__(config, 'css')
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """CSS-specific chunk types"""
        return {
            'rule_set': ChunkType.TEXT_BLOCK,
            'at_rule': ChunkType.TEXT_BLOCK,
            'keyframes_statement': ChunkType.TEXT_BLOCK,
            'media_statement': ChunkType.TEXT_BLOCK,
        }

# Export all parser classes
__all__ = [
    'SimpleTreeSitterParser',
    'RustParser',
    'GoParser', 
    'JavaParser',
    'CppParser',
    'CParser',
    'YAMLParser',
    'TOMLParser',
    'CSSParser'
]