# src/chuk_code_raptor/chunking/tree_sitter_base.py
"""
Tree-sitter Base Parser - Robust Edition
========================================

Base class for tree-sitter based parsers with robust package handling.
Handles different tree-sitter package API variations gracefully.
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Any
import logging

from .base import BaseParser, ParseContext
from .semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class TreeSitterParser(BaseParser):
    """Base class for tree-sitter based parsers with robust initialization"""
    
    def __init__(self, config):
        super().__init__(config)
        self.parser_type = "tree_sitter"
        self.parser = None
        self.language = None
        self._package_used = None
        self._initialization_error = None
        self._initialize_tree_sitter()
    
    @abstractmethod
    def _get_tree_sitter_language(self):
        """Get tree-sitter Language object - implement in subclasses"""
        pass
    
    @abstractmethod
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """Get mapping of AST node types to chunk types"""
        pass
    
    @abstractmethod
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from AST node"""
        pass
    
    def _initialize_tree_sitter(self):
        """Initialize tree-sitter parser with robust error handling"""
        try:
            import tree_sitter
            
            self.language = self._get_tree_sitter_language()
            self.parser = tree_sitter.Parser(self.language)
            logger.debug(f"✅ Initialized tree-sitter parser for {self.name}")
            
        except ImportError as e:
            self._initialization_error = f"tree-sitter not available: {e}"
            logger.debug(f"❌ {self.name}: {self._initialization_error}")
        except AttributeError as e:
            self._initialization_error = f"tree-sitter package API incompatible: {e}"
            logger.debug(f"❌ {self.name}: {self._initialization_error}")
        except Exception as e:
            self._initialization_error = f"tree-sitter initialization failed: {e}"
            logger.debug(f"❌ {self.name}: {self._initialization_error}")
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        """Check if this parser can handle the language/extension"""
        # Only return True if tree-sitter is properly initialized
        return (self.parser is not None and self.language is not None and
                (language in self.supported_languages or 
                 file_extension in self.supported_extensions))
    
    def parse(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse content using tree-sitter"""
        if self.parser is None or self.language is None:
            raise ImportError(f"Parser {self.name} not properly initialized: {self._initialization_error}")
        
        if not content.strip():
            return []
        
        tree = self.parser.parse(bytes(content, 'utf8'))
        root_node = tree.root_node
        
        chunks = []
        self._traverse_and_extract(root_node, content, context, chunks)
        
        return self._post_process(chunks)
    
    def _traverse_and_extract(self, node, content: str, context: ParseContext, 
                            chunks: List[SemanticChunk], depth: int = 0):
        """Traverse AST and extract chunks"""
        if depth > 30:
            return
        
        chunk_types = self._get_chunk_node_types()
        
        if node.type in chunk_types:
            chunk = self._create_chunk_from_node(node, content, context, chunk_types[node.type])
            if chunk:
                chunks.append(chunk)
                return  # Don't traverse children of extracted chunks
        
        for child in node.children:
            self._traverse_and_extract(child, content, context, chunks, depth + 1)
    
    def _create_chunk_from_node(self, node, content: str, context: ParseContext, 
                              chunk_type: ChunkType) -> Optional[SemanticChunk]:
        """Create semantic chunk from AST node"""
        chunk_content = content[node.start_byte:node.end_byte].strip()
        if not chunk_content:
            return None
        
        identifier = self._extract_identifier(node, content)
        
        chunk_id = create_chunk_id(
            context.file_path, 
            node.start_point[0] + 1, 
            chunk_type, 
            identifier
        )
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=chunk_content,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            content_type=context.content_type,
            chunk_type=chunk_type,
            language=context.language,
            importance_score=self._calculate_node_importance(node),
            metadata={
                'parser': self.name,
                'parser_type': 'tree_sitter',
                'node_type': node.type,
                'identifier': identifier,
                'byte_range': (node.start_byte, node.end_byte),
                'ast_depth': self._get_node_depth(node),
                'package_used': self._package_used,
            }
        )
        
        # Add semantic tags and extract dependencies
        self._add_semantic_tags(chunk, node, content)
        
        if context.enable_dependency_tracking:
            self._extract_dependencies(chunk, node, content)
        
        return chunk
    
    def _calculate_node_importance(self, node) -> float:
        """Calculate importance score for AST node"""
        # Base importance from node type
        type_importance = {
            'function_definition': 1.0,
            'class_definition': 1.0,
            'method_definition': 0.9,
            'async_function_definition': 1.0,
            'import_statement': 0.7,
            'import_from_statement': 0.7,
            'assignment': 0.5,
        }
        
        base_score = type_importance.get(node.type, 0.5)
        
        # Adjust based on size (larger nodes often more important)
        size_factor = min(1.0, (node.end_byte - node.start_byte) / 1000)
        
        # Adjust based on complexity (number of children)
        complexity_factor = min(1.0, node.child_count / 10)
        
        # Combine factors
        final_score = (base_score * 0.7 + size_factor * 0.15 + complexity_factor * 0.15)
        
        return min(1.0, max(0.1, final_score))
    
    def _get_node_depth(self, node) -> int:
        """Calculate depth of node in AST"""
        depth = 0
        parent = node.parent
        while parent:
            depth += 1
            parent = parent.parent
        return depth
    
    def _add_semantic_tags(self, chunk: SemanticChunk, node, content: str):
        """Add semantic tags - override in subclasses for language-specific tags"""
        chunk.add_tag(chunk.chunk_type.value, source="tree_sitter")
        chunk.add_tag(node.type, source="tree_sitter")
        
        # Add structural tags
        if chunk.importance_score > 0.8:
            chunk.add_tag('high_importance', source="tree_sitter")
        
        # Size-based tags
        if len(chunk.content) > self.config.target_chunk_size:
            chunk.add_tag('large_chunk', source="tree_sitter")
        elif len(chunk.content) < self.config.min_chunk_size:
            chunk.add_tag('small_chunk', source="tree_sitter")
    
    def _extract_dependencies(self, chunk: SemanticChunk, node, content: str):
        """Extract dependencies - override in subclasses for language-specific dependencies"""
        # Basic dependency extraction - subclasses should override
        pass


# Utility functions for robust tree-sitter package handling

def get_tree_sitter_language_robust(language_name: str, package_candidates: List[str] = None):
    """
    Robustly get a tree-sitter language, trying multiple package variations.
    
    Args:
        language_name: Name of the language (e.g., 'python', 'xml')
        package_candidates: List of package names to try, in order of preference
    
    Returns:
        tuple: (language_object, package_used) or (None, None) if failed
    """
    import tree_sitter
    
    if package_candidates is None:
        package_candidates = [
            f'tree_sitter_{language_name}',
            f'tree_sitter_languages',
            f'tree_sitter_language_pack'
        ]
    
    for package_name in package_candidates:
        try:
            if package_name in ['tree_sitter_languages', 'tree_sitter_language_pack']:
                # Try comprehensive packages
                if package_name == 'tree_sitter_languages':
                    from tree_sitter_languages import get_language
                    language_obj = get_language(language_name)
                    return tree_sitter.Language(language_obj), package_name
                elif package_name == 'tree_sitter_language_pack':
                    from tree_sitter_language_pack import get_language
                    language_obj = get_language(language_name)
                    return tree_sitter.Language(language_obj), package_name
            else:
                # Try individual packages
                module = __import__(package_name, fromlist=['language'])
                
                # Try different API variations
                if hasattr(module, 'language') and callable(getattr(module, 'language')):
                    # Most common API: module.language()
                    language_obj = module.language()
                    return tree_sitter.Language(language_obj), package_name
                elif hasattr(module, 'LANGUAGE'):
                    # Alternative API: module.LANGUAGE
                    language_obj = module.LANGUAGE
                    return tree_sitter.Language(language_obj), package_name
                elif hasattr(module, f'{language_name.upper()}_LANGUAGE'):
                    # Alternative API: module.PYTHON_LANGUAGE
                    language_obj = getattr(module, f'{language_name.upper()}_LANGUAGE')
                    return tree_sitter.Language(language_obj), package_name
                else:
                    logger.debug(f"Package {package_name} found but no compatible API")
                    continue
                    
        except ImportError:
            logger.debug(f"Package {package_name} not available")
            continue
        except Exception as e:
            logger.debug(f"Package {package_name} failed: {e}")
            continue
    
    return None, None


def create_robust_tree_sitter_parser(language_name: str, package_candidates: List[str] = None):
    """
    Create a robust tree-sitter parser that handles package variations.
    
    This is a helper for creating parsers that can handle different tree-sitter
    package APIs gracefully.
    """
    language, package_used = get_tree_sitter_language_robust(language_name, package_candidates)
    
    if language is None:
        return None, f"No tree-sitter package found for {language_name}"
    
    try:
        import tree_sitter
        parser = tree_sitter.Parser(language)
        return (parser, language, package_used), None
    except Exception as e:
        return None, f"Failed to create parser: {e}"