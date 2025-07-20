# src/chuk_code_raptor/chunking/parsers/javascript.py
"""
JavaScript/TypeScript Parser
============================

JavaScript parser using tree-sitter-javascript.
Follows the common TreeSitterParser architecture.
"""

import logging
from typing import Dict, List, Optional

from ..tree_sitter_base import TreeSitterParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class JavaScriptParser(TreeSitterParser):
    """JavaScript/TypeScript parser using tree-sitter-javascript"""
    
    def __init__(self, config):
        self.supported_languages = {'javascript', 'typescript'}
        self.supported_extensions = {'.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'}
        super().__init__(config)
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_tree_sitter_language(self):
        """Get JavaScript tree-sitter language"""
        try:
            import tree_sitter
            import tree_sitter_javascript
            return tree_sitter.Language(tree_sitter_javascript.language())
        except ImportError as e:
            raise ImportError("tree-sitter-javascript not installed. Install with: pip install tree-sitter-javascript") from e
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """JavaScript AST node types to chunk types mapping"""
        return {
            'function_declaration': ChunkType.FUNCTION,
            'arrow_function': ChunkType.FUNCTION,
            'function_expression': ChunkType.FUNCTION,
            'method_definition': ChunkType.METHOD,
            'class_declaration': ChunkType.CLASS,
            'class_expression': ChunkType.CLASS,
            'variable_declaration': ChunkType.VARIABLE,
            'lexical_declaration': ChunkType.VARIABLE,
            'import_statement': ChunkType.IMPORT,
            'export_statement': ChunkType.IMPORT,
        }
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from JavaScript AST node"""
        if node.type in ['function_declaration', 'class_declaration']:
            for child in node.children:
                if child.type == 'identifier':
                    identifier = content[child.start_byte:child.end_byte]
                    if identifier and identifier.strip():
                        return identifier.strip()
        
        elif node.type == 'arrow_function':
            # Try to get name from parent assignment
            parent = node.parent
            if parent and parent.type == 'variable_declarator':
                for child in parent.children:
                    if child.type == 'identifier':
                        identifier = content[child.start_byte:child.end_byte]
                        if identifier and identifier.strip():
                            return identifier.strip()
            return 'arrow_function'
        
        elif node.type == 'method_definition':
            for child in node.children:
                if child.type in ['property_name', 'identifier']:
                    identifier = content[child.start_byte:child.end_byte]
                    if identifier and identifier.strip():
                        return identifier.strip()
        
        elif node.type in ['variable_declaration', 'lexical_declaration']:
            # Find the first variable declarator
            for child in node.children:
                if child.type == 'variable_declarator':
                    for grandchild in child.children:
                        if grandchild.type == 'identifier':
                            identifier = content[grandchild.start_byte:grandchild.end_byte]
                            if identifier and identifier.strip():
                                return identifier.strip()
        
        elif node.type in ['import_statement', 'export_statement']:
            if 'import' in content[node.start_byte:node.end_byte]:
                return 'import'
            elif 'export' in content[node.start_byte:node.end_byte]:
                return 'export'
        
        return node.type
    
    def _add_semantic_tags(self, chunk: SemanticChunk, node, content: str):
        """Add JavaScript-specific semantic tags"""
        super()._add_semantic_tags(chunk, node, content)
        
        chunk_content = chunk.content.lower()
        
        if chunk.chunk_type == ChunkType.FUNCTION:
            chunk.add_tag('function', source='tree_sitter')
            
            if 'async' in chunk_content:
                chunk.add_tag('async', source='tree_sitter')
            if '=>' in chunk.content:
                chunk.add_tag('arrow_function', source='tree_sitter')
            if 'function*' in chunk_content:
                chunk.add_tag('generator', source='tree_sitter')
        
        elif chunk.chunk_type == ChunkType.METHOD:
            chunk.add_tag('method', source='tree_sitter')
            
            if 'static' in chunk_content:
                chunk.add_tag('static_method', source='tree_sitter')
            if 'async' in chunk_content:
                chunk.add_tag('async', source='tree_sitter')
            if 'get ' in chunk_content:
                chunk.add_tag('getter', source='tree_sitter')
            elif 'set ' in chunk_content:
                chunk.add_tag('setter', source='tree_sitter')
        
        elif chunk.chunk_type == ChunkType.CLASS:
            chunk.add_tag('class', source='tree_sitter')
            
            if 'extends' in chunk_content:
                chunk.add_tag('inheritance', source='tree_sitter')
        
        elif chunk.chunk_type == ChunkType.VARIABLE:
            chunk.add_tag('variable', source='tree_sitter')
            
            if chunk_content.strip().startswith('const'):
                chunk.add_tag('constant', source='tree_sitter')
            elif chunk_content.strip().startswith('let'):
                chunk.add_tag('let_variable', source='tree_sitter')
            elif chunk_content.strip().startswith('var'):
                chunk.add_tag('var_variable', source='tree_sitter')
        
        elif chunk.chunk_type == ChunkType.IMPORT:
            chunk.add_tag('import', source='tree_sitter')
            
            if 'export' in chunk_content:
                chunk.add_tag('export', source='tree_sitter')
            if 'default' in chunk_content:
                chunk.add_tag('default_export', source='tree_sitter')
        
        # Detect framework-specific patterns
        if 'useState' in chunk.content or 'useEffect' in chunk.content:
            chunk.add_tag('react_hooks', source='tree_sitter')
        if 'return <' in chunk.content:
            chunk.add_tag('jsx_return', source='tree_sitter')
    
    def _extract_dependencies(self, chunk: SemanticChunk, node, content: str):
        """Extract JavaScript dependencies"""
        import re
        
        # Function calls
        call_pattern = r'(\w+)\s*\('
        matches = re.findall(call_pattern, chunk.content)
        for match in matches:
            if match not in {'if', 'for', 'while', 'switch', 'function', 'class'}:
                chunk.dependencies.append(f"calls:{match}")
        
        # Variable declarations
        var_pattern = r'(?:let|const|var)\s+(\w+)'
        matches = re.findall(var_pattern, chunk.content)
        for match in matches:
            chunk.dependencies.append(f"declares:{match}")