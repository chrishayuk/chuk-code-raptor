# src/chuk_code_raptor/chunking/parsers/python.py
"""
Improved Python Parser
======================

Enhanced Python parser with better chunking strategies, quality assessment,
and import analysis. Uses tree-sitter AST analysis exclusively.
"""

import re
from typing import Dict, List, Optional, Any, Set
import logging

from ..tree_sitter_base import TreeSitterParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from ..code_chunk import SemanticCodeChunk, ArchitecturalRole
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class PythonParser(TreeSitterParser):
    """Improved Python parser with sophisticated semantic analysis"""
    
    def __init__(self, config):
        super().__init__(config)
        self.supported_languages = {'python'}
        self.supported_extensions = {'.py', '.pyx', '.pyi'}
        
        # Python standard library modules for import classification
        self.stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'asyncio', 'logging',
            'collections', 'functools', 'itertools', 'operator', 'typing', 'dataclasses',
            'abc', 'contextlib', 'threading', 'multiprocessing', 'subprocess', 'shutil',
            'glob', 'tempfile', 're', 'string', 'math', 'random', 'statistics', 'decimal',
            'fractions', 'sqlite3', 'pickle', 'csv', 'configparser', 'argparse', 'getopt',
            'urllib', 'http', 'email', 'socket', 'ssl', 'hashlib', 'hmac', 'secrets',
            'unittest', 'doctest', 'pdb', 'profile', 'pstats', 'timeit', 'trace',
            'warnings', 'types', 'copy', 'weakref', 'gc', 'inspect', 'importlib'
        }
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_tree_sitter_language(self):
        """Get Python tree-sitter language"""
        try:
            import tree_sitter
            import tree_sitter_python
            return tree_sitter.Language(tree_sitter_python.language())
        except ImportError as e:
            raise ImportError("tree-sitter-python not installed. Install with: pip install tree-sitter-python") from e
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """Python AST node types to chunk types mapping"""
        return {
            'function_definition': ChunkType.FUNCTION,
            'async_function_definition': ChunkType.FUNCTION,
            'class_definition': ChunkType.CLASS,
            'import_statement': ChunkType.IMPORT,
            'import_from_statement': ChunkType.IMPORT,
            'assignment': ChunkType.VARIABLE,
        }
    
    def _create_chunk_from_node(self, node, content: str, context, 
                              chunk_type: ChunkType) -> Optional[SemanticChunk]:
        """Create enhanced semantic chunk from AST node"""
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
        
        # Create enhanced code chunk for code content
        if chunk_type in [ChunkType.FUNCTION, ChunkType.CLASS]:
            chunk = SemanticCodeChunk(
                id=chunk_id,
                file_path=context.file_path,
                content=chunk_content,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content_type=context.content_type,
                chunk_type=chunk_type,
                language=context.language,
                importance_score=self._calculate_enhanced_importance(node, chunk_content, chunk_type),
                metadata={
                    'parser': self.name,
                    'parser_type': 'tree_sitter',
                    'node_type': node.type,
                    'identifier': identifier,
                    'byte_range': (node.start_byte, node.end_byte),
                    'ast_depth': self._get_node_depth(node),
                }
            )
            
            # Analyze code quality using AST
            self._analyze_code_chunk(chunk, node, content)
            
        else:
            # Regular semantic chunk for other types
            chunk = SemanticChunk(
                id=chunk_id,
                file_path=context.file_path,
                content=chunk_content,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content_type=context.content_type,
                chunk_type=chunk_type,
                language=context.language,
                importance_score=self._calculate_enhanced_importance(node, chunk_content, chunk_type),
                metadata={
                    'parser': self.name,
                    'parser_type': 'tree_sitter',
                    'node_type': node.type,
                    'identifier': identifier,
                    'byte_range': (node.start_byte, node.end_byte),
                    'ast_depth': self._get_node_depth(node),
                }
            )
        
        # Add semantic tags and extract dependencies
        self._add_semantic_tags(chunk, node, content)
        
        if context.enable_dependency_tracking:
            self._extract_dependencies(chunk, node, content)
        
        return chunk
    
    def _analyze_code_chunk(self, chunk: SemanticCodeChunk, node, content: str):
        """Analyze code chunk using AST information"""
        # Documentation analysis
        has_docstring, docstring_quality = self._analyze_docstring_ast(node, content)
        
        # Type hints analysis
        has_type_hints, type_coverage = self._analyze_type_hints_ast(node, content)
        
        # Error handling analysis
        has_error_handling = self._analyze_error_handling_ast(node, content)
        
        # Complexity analysis
        cyclomatic_complexity = self._calculate_cyclomatic_complexity_ast(node, content)
        
        # Set code quality indicators
        chunk.set_code_quality_indicators(
            has_docstring=has_docstring,
            docstring_quality=docstring_quality,
            has_type_hints=has_type_hints,
            type_coverage=type_coverage,
            has_error_handling=has_error_handling,
            cyclomatic_complexity=cyclomatic_complexity
        )
        
        # Set architectural role
        chunk.architectural_role = self._detect_architectural_role_ast(node, content)
    
    def _analyze_docstring_ast(self, node, content: str) -> tuple[bool, float]:
        """Analyze docstring using AST"""
        # Look for string literals at the beginning of function/class bodies
        for child in node.children:
            if child.type == 'block':
                for grandchild in child.children:
                    if grandchild.type == 'expression_statement':
                        for ggchild in grandchild.children:
                            if ggchild.type == 'string':
                                docstring_content = content[ggchild.start_byte:ggchild.end_byte]
                                # Remove quotes
                                docstring_text = docstring_content.strip('"""').strip("'''").strip('"').strip("'")
                                
                                if len(docstring_text) > 10:  # Minimum docstring length
                                    quality = self._assess_docstring_quality(docstring_text)
                                    return True, quality
                break
        
        return False, 0.0
    
    def _assess_docstring_quality(self, docstring: str) -> float:
        """Assess docstring quality based on content"""
        if not docstring or len(docstring.strip()) < 10:
            return 0.0
        
        score = 0.0
        
        # Length bonus
        if len(docstring) > 100:
            score += 0.4
        elif len(docstring) > 50:
            score += 0.3
        elif len(docstring) > 20:
            score += 0.2
        else:
            score += 0.1
        
        # Structure bonus
        structure_keywords = [
            'args:', 'arguments:', 'parameters:', 'returns:', 'return:', 
            'yields:', 'raises:', 'examples:', 'note:', 'warning:'
        ]
        if any(keyword in docstring.lower() for keyword in structure_keywords):
            score += 0.3
        
        # Description quality
        sentences = docstring.count('.') + docstring.count('!') + docstring.count('?')
        if sentences >= 2:
            score += 0.3
        elif sentences >= 1:
            score += 0.2
        
        return min(1.0, score)
    
    def _analyze_type_hints_ast(self, node, content: str) -> tuple[bool, float]:
        """Analyze type hints using AST"""
        total_params = 0
        typed_params = 0
        has_return_type = False
        
        # For function definitions, check parameters and return type
        if node.type in ['function_definition', 'async_function_definition']:
            for child in node.children:
                if child.type == 'parameters':
                    for param_child in child.children:
                        if param_child.type in ['identifier', 'typed_parameter']:
                            total_params += 1
                            if param_child.type == 'typed_parameter':
                                typed_params += 1
                
                elif child.type == 'type':
                    has_return_type = True
        
        has_type_hints = typed_params > 0 or has_return_type
        type_coverage = typed_params / max(total_params, 1) if total_params > 0 else 0.0
        
        if has_return_type:
            type_coverage = min(type_coverage + 0.2, 1.0)
        
        return has_type_hints, type_coverage
    
    def _analyze_error_handling_ast(self, node, content: str) -> bool:
        """Analyze error handling using AST"""
        def has_error_handling_node(n):
            if n.type in ['try_statement', 'raise_statement', 'assert_statement']:
                return True
            for child in n.children:
                if has_error_handling_node(child):
                    return True
            return False
        
        return has_error_handling_node(node)
    
    def _calculate_cyclomatic_complexity_ast(self, node, content: str) -> int:
        """Calculate cyclomatic complexity using AST"""
        complexity = 1  # Base complexity
        
        def count_decision_points(n):
            nonlocal complexity
            
            # Decision points that increase complexity
            decision_nodes = {
                'if_statement', 'elif_clause', 'else_clause',
                'for_statement', 'while_statement',
                'try_statement', 'except_clause',
                'boolean_operator'
            }
            
            if n.type in decision_nodes:
                complexity += 1
            
            for child in n.children:
                count_decision_points(child)
        
        count_decision_points(node)
        return complexity
    
    def _detect_architectural_role_ast(self, node, content: str) -> Optional[ArchitecturalRole]:
        """Detect architectural role using AST and naming patterns"""
        identifier = self._extract_identifier(node, content)
        if not identifier:
            return None
        
        identifier_lower = identifier.lower()
        
        if any(term in identifier_lower for term in ['repository', 'dao', 'store']):
            return ArchitecturalRole.DATA_ACCESS
        elif any(term in identifier_lower for term in ['service', 'manager', 'handler']):
            return ArchitecturalRole.BUSINESS_LOGIC
        elif any(term in identifier_lower for term in ['controller', 'view', 'router']):
            return ArchitecturalRole.PRESENTATION
        elif any(term in identifier_lower for term in ['factory', 'builder', 'creator']):
            return ArchitecturalRole.CREATIONAL
        elif any(term in identifier_lower for term in ['config', 'settings', 'options']):
            return ArchitecturalRole.CONFIGURATION
        elif any(term in identifier_lower for term in ['test', 'mock', 'stub']):
            return ArchitecturalRole.TESTING
        
        return None
    
    def _calculate_enhanced_importance(self, node, content: str, chunk_type: ChunkType) -> float:
        """Calculate enhanced importance score"""
        base_score = 0.5
        
        # Type-based importance
        type_weights = {
            ChunkType.FUNCTION: 0.8,
            ChunkType.CLASS: 0.9,
            ChunkType.IMPORT: 0.4,
            ChunkType.VARIABLE: 0.3,
        }
        
        base_score = type_weights.get(chunk_type, 0.5)
        
        # Async functions are often more important
        if node.type == 'async_function_definition':
            base_score += 0.1
        
        # Public APIs (no leading underscore) are more important
        identifier = self._extract_identifier(node, content)
        if identifier and not identifier.startswith('_'):
            base_score += 0.1
        
        # Decorators indicate special functionality
        if self._has_decorators_ast(node, content):
            base_score += 0.1
        
        # Size considerations
        content_size = len(content)
        if content_size > self.config.target_chunk_size * 3:
            base_score -= 0.1  # Very large chunks might be doing too much
        elif content_size > self.config.target_chunk_size:
            pass  # Neutral
        elif content_size < self.config.min_chunk_size:
            base_score -= 0.05  # Very small chunks might be less important
        
        return min(1.0, max(0.1, base_score))
    
    def _has_decorators_ast(self, node, content: str) -> bool:
        """Check for decorators using AST"""
        for child in node.children:
            if child.type == 'decorator':
                return True
        return False
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from Python AST node"""
        if node.type in ['function_definition', 'async_function_definition', 'class_definition']:
            for child in node.children:
                if child.type == 'identifier':
                    return content[child.start_byte:child.end_byte]
        
        elif node.type in ['import_statement', 'import_from_statement']:
            return self._extract_import_name(node, content)
        
        elif node.type == 'assignment':
            for child in node.children:
                if child.type == 'identifier':
                    return content[child.start_byte:child.end_byte]
        
        return None
    
    def _extract_import_name(self, node, content: str) -> str:
        """Extract import name for identifier"""
        import_parts = []
        
        def collect_parts(n):
            if n.type == 'dotted_name':
                import_parts.append(content[n.start_byte:n.end_byte])
            elif n.type == 'identifier':
                import_parts.append(content[n.start_byte:n.end_byte])
            else:
                for child in n.children:
                    collect_parts(child)
        
        collect_parts(node)
        return '.'.join(import_parts) if import_parts else 'import'
    
    def _add_semantic_tags(self, chunk: SemanticChunk, node, content: str):
        """Add Python-specific semantic tags"""
        super()._add_semantic_tags(chunk, node, content)
        
        if chunk.chunk_type == ChunkType.FUNCTION:
            if node.type == 'async_function_definition':
                chunk.add_tag('async', source='tree_sitter')
            if self._has_decorators_ast(node, content):
                chunk.add_tag('decorated', source='tree_sitter')
            if self._is_method(node):
                chunk.add_tag('method', source='tree_sitter')
            if self._is_generator_ast(node, content):
                chunk.add_tag('generator', source='tree_sitter')
        
        elif chunk.chunk_type == ChunkType.CLASS:
            if self._has_inheritance_ast(node, content):
                chunk.add_tag('inherits', source='tree_sitter')
            if self._has_decorators_ast(node, content):
                chunk.add_tag('decorated', source='tree_sitter')
            if self._is_abstract_class_ast(node, content):
                chunk.add_tag('abstract', source='tree_sitter')
        
        elif chunk.chunk_type == ChunkType.IMPORT:
            if node.type == 'import_from_statement':
                chunk.add_tag('from_import', source='tree_sitter')
            
            # Classify import type
            import_name = chunk.metadata.get('identifier', '')
            if self._is_stdlib_import(import_name):
                chunk.add_tag('stdlib_import', source='tree_sitter')
            elif self._is_local_import(import_name):
                chunk.add_tag('local_import', source='tree_sitter')
            else:
                chunk.add_tag('third_party_import', source='tree_sitter')
    
    def _is_method(self, node) -> bool:
        """Check if function is inside a class"""
        parent = node.parent
        while parent:
            if parent.type == 'class_definition':
                return True
            parent = parent.parent
        return False
    
    def _is_generator_ast(self, node, content: str) -> bool:
        """Check if function is a generator using AST"""
        def has_yield(n):
            if n.type == 'yield':
                return True
            for child in n.children:
                if has_yield(child):
                    return True
            return False
        
        return has_yield(node)
    
    def _has_inheritance_ast(self, node, content: str) -> bool:
        """Check if class has inheritance using AST"""
        for child in node.children:
            if child.type == 'argument_list':
                return True
        return False
    
    def _is_abstract_class_ast(self, node, content: str) -> bool:
        """Check if class is abstract using AST"""
        # Look for ABC inheritance or abstractmethod decorators
        def has_abc_reference(n):
            if n.type == 'identifier':
                text = content[n.start_byte:n.end_byte]
                if text in ['ABC', 'abstractmethod']:
                    return True
            for child in n.children:
                if has_abc_reference(child):
                    return True
            return False
        
        return has_abc_reference(node)
    
    def _is_stdlib_import(self, import_name: str) -> bool:
        """Check if import is from standard library"""
        if not import_name:
            return False
        
        # Get the top-level module name
        top_level = import_name.split('.')[0]
        return top_level in self.stdlib_modules
    
    def _is_local_import(self, import_name: str) -> bool:
        """Check if import is local (relative import or same package)"""
        if not import_name:
            return False
        
        # Relative imports start with dots
        if import_name.startswith('.'):
            return True
        
        # Check for package-specific patterns (can be customized)
        return any(pattern in import_name for pattern in ['chuk_', 'src.'])
    
    def _extract_dependencies(self, chunk: SemanticChunk, node, content: str):
        """Extract Python dependencies using AST"""
        if chunk.chunk_type == ChunkType.IMPORT:
            import_name = chunk.metadata.get('identifier')
            if import_name:
                chunk.dependencies.append(f"imports:{import_name}")
        
        # Extract function calls and other dependencies using AST traversal
        self._extract_ast_dependencies(chunk, node, content)
    
    def _extract_ast_dependencies(self, chunk: SemanticChunk, node, content: str):
        """Extract dependencies using AST traversal"""
        calls = []
        variables = []
        
        def extract_from_node(n):
            if n.type == 'call':
                for child in n.children:
                    if child.type in ['identifier', 'attribute']:
                        call_name = content[child.start_byte:child.end_byte]
                        calls.append(call_name)
                        break
            
            elif n.type == 'assignment':
                for child in n.children:
                    if child.type == 'identifier':
                        var_name = content[child.start_byte:child.end_byte]
                        variables.append(var_name)
                        break
            
            for child in n.children:
                extract_from_node(child)
        
        extract_from_node(node)
        
        # Add to chunk
        if isinstance(chunk, SemanticCodeChunk):
            chunk.function_calls.extend(calls)
            chunk.variables_used.extend(variables)
        
        for call in calls:
            chunk.dependencies.append(f"calls:{call}")
        
        for var in variables:
            chunk.dependencies.append(f"declares:{var}")
    
    def _post_process(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Post-process Python chunks"""
        chunks = super()._post_process(chunks)
        
        if self.config.group_imports:
            chunks = self._group_imports_intelligently(chunks)
        
        return chunks
    
    def _group_imports_intelligently(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Group imports by type (stdlib, third-party, local)"""
        import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
        other_chunks = [c for c in chunks if c.chunk_type != ChunkType.IMPORT]
        
        if len(import_chunks) <= 1:
            return chunks
        
        # Group imports by type
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        for chunk in import_chunks:
            if any(tag.name == 'stdlib_import' for tag in chunk.semantic_tags):
                stdlib_imports.append(chunk)
            elif any(tag.name == 'local_import' for tag in chunk.semantic_tags):
                local_imports.append(chunk)
            else:
                third_party_imports.append(chunk)
        
        # Create groups for large collections
        grouped_chunks = []
        
        if len(stdlib_imports) > 3:
            grouped_chunks.append(self._create_import_group(stdlib_imports, "stdlib"))
        else:
            grouped_chunks.extend(stdlib_imports)
        
        if len(third_party_imports) > 2:
            grouped_chunks.append(self._create_import_group(third_party_imports, "third_party"))
        else:
            grouped_chunks.extend(third_party_imports)
        
        # Keep local imports separate
        grouped_chunks.extend(local_imports)
        
        all_chunks = grouped_chunks + other_chunks
        all_chunks.sort(key=lambda c: c.start_line)
        return all_chunks
    
    def _create_import_group(self, import_chunks: List[SemanticChunk], group_type: str) -> SemanticChunk:
        """Create a grouped import chunk"""
        if not import_chunks:
            return None
        
        first = import_chunks[0]
        last = import_chunks[-1]
        
        group_content = '\n'.join(chunk.content for chunk in import_chunks)
        group_id = create_chunk_id(first.file_path, first.start_line, ChunkType.IMPORT, f"{group_type}_imports")
        
        grouped_chunk = SemanticChunk(
            id=group_id,
            file_path=first.file_path,
            content=group_content,
            start_line=first.start_line,
            end_line=last.end_line,
            content_type=first.content_type,
            chunk_type=ChunkType.IMPORT,
            language=first.language,
            metadata={
                'parser': self.name,
                'is_import_group': True,
                'group_type': group_type,
                'import_count': len(import_chunks),
            }
        )
        grouped_chunk.add_tag('import_group', source='tree_sitter')
        grouped_chunk.add_tag(f'{group_type}_imports', source='tree_sitter')
        
        return grouped_chunk