# src/chuk_code_raptor/chunking/parsers/json.py
"""
JSON Parser
===========

JSON parser using tree-sitter-json for AST-based chunking.
Follows the common TreeSitterParser architecture.
"""

import logging
from typing import List, Dict, Optional

from ..tree_sitter_base import TreeSitterParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class JSONParser(TreeSitterParser):
    """JSON parser using tree-sitter-json"""
    
    def __init__(self, config):
        # Call parent first
        super().__init__(config)
        
        # Then set our specific supported languages and extensions
        self.supported_languages = {'json'}
        self.supported_extensions = {'.json', '.jsonl', '.ndjson'}
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_tree_sitter_language(self):
        """Get JSON tree-sitter language"""
        try:
            import tree_sitter
            import tree_sitter_json
            return tree_sitter.Language(tree_sitter_json.language())
        except ImportError as e:
            raise ImportError("tree-sitter-json not installed. Install with: pip install tree-sitter-json") from e
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """JSON AST node types to chunk types mapping"""
        return {
            'object': ChunkType.TEXT_BLOCK,
            'array': ChunkType.TEXT_BLOCK,
            'pair': ChunkType.TEXT_BLOCK,
        }
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from JSON AST node"""
        if node.type == 'object':
            return "json_object"
        elif node.type == 'array':
            return "json_array"
        elif node.type == 'pair':
            # Extract key from pair
            for child in node.children:
                if child.type == 'string':
                    key = content[child.start_byte:child.end_byte]
                    # Remove quotes
                    if key.startswith('"') and key.endswith('"'):
                        key = key[1:-1]
                    return f"key_{key}"
            return "json_pair"
        return node.type
    
    def _add_semantic_tags(self, chunk: SemanticChunk, node, content: str):
        """Add JSON-specific semantic tags"""
        super()._add_semantic_tags(chunk, node, content)
        
        chunk.add_tag('json', source='tree_sitter')
        
        if node.type == 'object':
            chunk.add_tag('json_object', source='tree_sitter')
            
            # Analyze object properties
            properties = self._extract_object_properties(node, content)
            if properties:
                chunk.metadata['properties'] = properties
                
                # Add semantic tags based on property patterns
                if any(prop in ['id', 'uuid', '_id'] for prop in properties):
                    chunk.add_tag('has_identifier', source='tree_sitter')
                if any(prop in ['name', 'title', 'label'] for prop in properties):
                    chunk.add_tag('has_name', source='tree_sitter')
                if any(prop in ['created_at', 'updated_at', 'timestamp'] for prop in properties):
                    chunk.add_tag('timestamped', source='tree_sitter')
        
        elif node.type == 'array':
            chunk.add_tag('json_array', source='tree_sitter')
            
            # Analyze array items
            item_types = self._analyze_array_items(node)
            chunk.metadata['item_types'] = item_types
            
            if len(set(item_types)) == 1:
                chunk.add_tag('homogeneous_array', source='tree_sitter')
            else:
                chunk.add_tag('heterogeneous_array', source='tree_sitter')
        
        elif node.type == 'pair':
            chunk.add_tag('json_pair', source='tree_sitter')
    
    def _extract_object_properties(self, node, content: str) -> List[str]:
        """Extract property names from a JSON object"""
        properties = []
        for child in node.children:
            if child.type == 'pair':
                for grandchild in child.children:
                    if grandchild.type == 'string':
                        prop_name = content[grandchild.start_byte:grandchild.end_byte]
                        if prop_name.startswith('"') and prop_name.endswith('"'):
                            prop_name = prop_name[1:-1]
                        properties.append(prop_name)
                        break
        return properties
    
    def _analyze_array_items(self, node) -> List[str]:
        """Analyze types of items in a JSON array"""
        item_types = []
        for child in node.children:
            if child.type not in [',', '[', ']']:
                item_types.append(child.type)
        return item_types
    
    def _extract_dependencies(self, chunk: SemanticChunk, node, content: str):
        """Extract JSON dependencies (references between objects)"""
        # For JSON, dependencies could be references to other objects
        # This is a simplified implementation
        chunk_content = chunk.content.lower()
        
        # Look for common reference patterns
        if '"$ref"' in chunk_content:
            chunk.add_tag('has_reference', source='tree_sitter')
        
        if '"id"' in chunk_content:
            chunk.add_tag('has_id', source='tree_sitter')