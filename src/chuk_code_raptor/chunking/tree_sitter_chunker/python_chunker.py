# chuk_code_raptor/chunking/tree_sitter_chunker/python_chunker.py
from .config import load_language_config
from chuk_code_raptor.chunking.semantic_chunk import SemanticCodeChunk, SemanticTag
from chuk_code_raptor.core.models import ChunkType# chuk_code_raptor/chunking/tree_sitter_chunker/python_chunker.py
"""
Semantic Python Tree-sitter Chunker
===================================

Specialized tree-sitter chunker for Python that produces SemanticCodeChunk objects
with full semantic understanding, dependency tracking, and Python-specific analysis.
"""

import re
from typing import List, Dict, Any, Optional, Set
import logging

import tree_sitter
from tree_sitter import Language, Node
import tree_sitter_python

from .base import BaseSemanticTreeSitterChunker, SemanticChunkCandidate, LanguageConfig, ParseContext

logger = logging.getLogger(__name__)

class PythonSemanticChunker(BaseSemanticTreeSitterChunker):
    """
    Python-specific semantic tree-sitter chunker that understands:
    - Function and class definitions with full semantic analysis
    - Import dependencies and module relationships
    - Decorators and their semantic meaning
    - Python-specific patterns (async/await, comprehensions, etc.)
    - Variable usage and type analysis
    """
    
    def _load_language_config(self) -> LanguageConfig:
        """Load Python-specific configuration"""
        return load_language_config('python')
    
    def _get_tree_sitter_language(self) -> Language:
        """Get Python tree-sitter language"""
        return Language(tree_sitter_python.language())
    
    def _extract_identifier(self, node: Node, content: str) -> Optional[str]:
        """Extract identifier from Python AST nodes"""
        if node.type in ['function_definition', 'async_function_definition', 'class_definition']:
            # Look for 'name' field
            for child in node.children:
                if child.type == 'identifier':
                    return content[child.start_byte:child.end_byte]
        
        elif node.type in ['import_statement', 'import_from_statement']:
            return self._extract_import_identifier(node, content)
        
        elif node.type == 'assignment':
            return self._extract_assignment_identifier(node, content)
        
        return None
    
    def _extract_semantic_tags(self, node: Node, content: str) -> List[str]:
        """Extract semantic tags from Python nodes"""
        tags = []
        
        # Function/method tags
        if node.type in ['function_definition', 'async_function_definition']:
            tags.append('function')
            if node.type == 'async_function_definition':
                tags.append('async')
            
            # Check for special function types
            node_content = content[node.start_byte:node.end_byte]
            if '__init__' in node_content:
                tags.append('constructor')
            elif '__str__' in node_content or '__repr__' in node_content:
                tags.append('string_method')
            elif 'yield' in node_content:
                tags.append('generator')
            elif '@property' in node_content:
                tags.append('property')
            elif '@staticmethod' in node_content:
                tags.append('static_method')
            elif '@classmethod' in node_content:
                tags.append('class_method')
        
        # Class tags
        elif node.type == 'class_definition':
            tags.append('class')
            node_content = content[node.start_byte:node.end_byte]
            if 'ABC' in node_content or 'abstractmethod' in node_content:
                tags.append('abstract')
            if 'Exception' in node_content or 'Error' in node_content:
                tags.append('exception')
        
        # Import tags
        elif node.type in ['import_statement', 'import_from_statement']:
            tags.append('import')
            node_content = content[node.start_byte:node.end_byte]
            if 'typing' in node_content:
                tags.append('type_import')
            elif any(lib in node_content for lib in ['numpy', 'pandas', 'matplotlib']):
                tags.append('data_science')
            elif any(lib in node_content for lib in ['flask', 'django', 'fastapi']):
                tags.append('web_framework')
        
        # Control structure tags
        elif node.type in ['if_statement', 'for_statement', 'while_statement', 'try_statement']:
            tags.append('control_flow')
            if node.type == 'try_statement':
                tags.append('error_handling')
        
        return tags
    
    def _extract_dependencies(self, node: Node, content: str) -> List[str]:
        """Extract dependencies from Python nodes"""
        dependencies = []
        
        def find_dependencies(n: Node):
            # Import dependencies
            if n.type in ['import_statement', 'import_from_statement']:
                dep = self._extract_import_identifier(n, content)
                if dep:
                    dependencies.append(dep)
            
            # Function call dependencies
            elif n.type == 'call':
                for child in n.children:
                    if child.type == 'identifier':
                        func_name = content[child.start_byte:child.end_byte]
                        dependencies.append(f"calls:{func_name}")
                    elif child.type == 'attribute':
                        attr_name = content[child.start_byte:child.end_byte]
                        dependencies.append(f"calls:{attr_name}")
            
            # Variable usage
            elif n.type == 'identifier':
                var_name = content[n.start_byte:n.end_byte]
                if not var_name.startswith('_') and len(var_name) > 1:
                    dependencies.append(f"uses:{var_name}")
            
            # Inheritance dependencies
            elif n.type == 'argument_list' and n.parent and n.parent.type == 'class_definition':
                for child in n.children:
                    if child.type == 'identifier':
                        base_class = content[child.start_byte:child.end_byte]
                        dependencies.append(f"inherits:{base_class}")
            
            # Recurse into children
            for child in n.children:
                find_dependencies(child)
        
        find_dependencies(node)
        return list(set(dependencies))  # Remove duplicates
    
    def _enhance_candidates(self, candidates: List[SemanticChunkCandidate], 
                          content: str, context: ParseContext) -> List[SemanticChunkCandidate]:
        """Apply Python-specific enhancements"""
        candidates = self._handle_decorators(candidates, content)
        candidates = self._group_imports(candidates, content)
        candidates = self._detect_main_blocks(candidates, content)
        candidates = self._analyze_class_methods(candidates, content)
        return candidates
    
    def _handle_decorators(self, candidates: List[SemanticChunkCandidate], 
                          content: str) -> List[SemanticChunkCandidate]:
        """Include decorators with function/class definitions"""
        lines = content.split('\n')
        enhanced_candidates = []
        
        for candidate in candidates:
            if candidate.node_type in ['function_definition', 'async_function_definition', 'class_definition']:
                # Look backwards for decorators
                start_line = candidate.start_line - 1  # Convert to 0-based
                decorator_start = start_line
                
                for i in range(start_line - 1, -1, -1):
                    line = lines[i].strip()
                    if line.startswith('@'):
                        decorator_start = i
                        # Add decorator semantic tag
                        decorator_name = line[1:].split('(')[0]
                        candidate.semantic_tags.append(f"decorator:{decorator_name}")
                    elif line and not line.startswith('#'):
                        break
                
                if decorator_start < start_line:
                    # Expand content to include decorators
                    new_content = '\n'.join(lines[decorator_start:candidate.end_line])
                    candidate.content = new_content
                    candidate.start_line = decorator_start + 1
                    candidate.metadata['includes_decorators'] = True
                    candidate.metadata['decorator_lines'] = start_line - decorator_start
            
            enhanced_candidates.append(candidate)
        
        return enhanced_candidates
    
    def _group_imports(self, candidates: List[SemanticChunkCandidate], 
                      content: str) -> List[SemanticChunkCandidate]:
        """Group consecutive import statements"""
        import_candidates = [c for c in candidates if c.chunk_type == ChunkType.IMPORT]
        other_candidates = [c for c in candidates if c.chunk_type != ChunkType.IMPORT]
        
        if len(import_candidates) <= 1:
            return candidates
        
        # Sort imports by line number
        import_candidates.sort(key=lambda c: c.start_line)
        
        # Group consecutive imports
        groups = []
        current_group = [import_candidates[0]]
        
        for i in range(1, len(import_candidates)):
            prev_import = import_candidates[i-1]
            curr_import = import_candidates[i]
            
            # Group if within 2 lines of each other
            if curr_import.start_line - prev_import.end_line <= 2:
                current_group.append(curr_import)
            else:
                groups.append(current_group)
                current_group = [curr_import]
        
        groups.append(current_group)
        
        # Create grouped candidates
        grouped_candidates = []
        lines = content.split('\n')
        
        for group in groups:
            if len(group) > 1:
                # Create grouped import
                first_import = group[0]
                last_import = group[-1]
                
                group_content = '\n'.join(lines[first_import.start_line-1:last_import.end_line])
                identifiers = [imp.identifier for imp in group if imp.identifier]
                
                grouped_candidate = SemanticChunkCandidate(
                    node=first_import.node,
                    content=group_content,
                    start_line=first_import.start_line,
                    end_line=last_import.end_line,
                    start_byte=first_import.start_byte,
                    end_byte=last_import.end_byte,
                    node_type='import_group',
                    chunk_type=ChunkType.IMPORT,
                    content_type=first_import.content_type,
                    importance_score=max(imp.importance_score for imp in group),
                    identifier=', '.join(identifiers),
                    semantic_tags=['import', 'import_group'],
                    metadata={
                        'is_import_group': True,
                        'import_count': len(group),
                        'import_identifiers': identifiers
                    }
                )
                grouped_candidates.append(grouped_candidate)
            else:
                grouped_candidates.extend(group)
        
        # Combine with other candidates and sort
        all_candidates = grouped_candidates + other_candidates
        all_candidates.sort(key=lambda c: c.start_line)
        
        return all_candidates
    
    def _detect_main_blocks(self, candidates: List[SemanticChunkCandidate], 
                           content: str) -> List[SemanticChunkCandidate]:
        """Detect if __name__ == "__main__" blocks"""
        lines = content.split('\n')
        main_candidates = []
        
        for i, line in enumerate(lines):
            if '__name__' in line and '__main__' in line:
                # Found potential main block
                start_line = i + 1
                end_line = len(lines)
                
                # Find the actual end of the if block
                indent_level = len(line) - len(line.lstrip())
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if next_line.strip() and (len(next_line) - len(next_line.lstrip())) <= indent_level:
                        end_line = j
                        break
                
                main_content = '\n'.join(lines[i:end_line])
                
                main_candidate = SemanticChunkCandidate(
                    node=None,  # No specific node for this construct
                    content=main_content,
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=0,  # Approximate
                    end_byte=len(main_content),
                    node_type='main_block',
                    chunk_type=ChunkType.TEXT_BLOCK,
                    content_type=candidates[0].content_type if candidates else ContentType.CODE,
                    importance_score=0.9,
                    identifier='__main__',
                    semantic_tags=['main_block', 'entry_point'],
                    metadata={
                        'is_main_block': True,
                        'special_construct': True
                    }
                )
                main_candidates.append(main_candidate)
                break  # Only one main block per file
        
        return candidates + main_candidates
    
    def _analyze_class_methods(self, candidates: List[SemanticChunkCandidate], 
                             content: str) -> List[SemanticChunkCandidate]:
        """Analyze methods within classes and add semantic information"""
        class_candidates = [c for c in candidates if c.chunk_type == ChunkType.CLASS]
        method_candidates = [c for c in candidates if c.chunk_type == ChunkType.FUNCTION]
        
        # Associate methods with their classes
        for method in method_candidates:
            for class_candidate in class_candidates:
                if (class_candidate.start_line <= method.start_line <= class_candidate.end_line):
                    # This method is inside this class
                    method.semantic_tags.append(f"class_method")
                    method.metadata['parent_class'] = class_candidate.identifier
                    
                    # Determine method type
                    if method.identifier == '__init__':
                        method.semantic_tags.append('constructor')
                    elif method.identifier.startswith('__') and method.identifier.endswith('__'):
                        method.semantic_tags.append('magic_method')
                    elif method.identifier.startswith('_'):
                        method.semantic_tags.append('private_method')
                    else:
                        method.semantic_tags.append('public_method')
                    
                    break
        
        return candidates
    
    def _extract_import_identifier(self, node: Node, content: str) -> Optional[str]:
        """Extract identifier from import statements"""
        import_parts = []
        
        def collect_import_parts(n: Node):
            if n.type == 'dotted_name':
                import_parts.append(content[n.start_byte:n.end_byte])
            elif n.type == 'identifier':
                import_parts.append(content[n.start_byte:n.end_byte])
            else:
                for child in n.children:
                    collect_import_parts(child)
        
        collect_import_parts(node)
        return '.'.join(import_parts) if import_parts else None
    
    def _extract_assignment_identifier(self, node: Node, content: str) -> Optional[str]:
        """Extract identifier from assignment statements"""
        for child in node.children:
            if child.type == 'identifier':
                return content[child.start_byte:child.end_byte]
            elif child.type == 'attribute':
                return content[child.start_byte:child.end_byte]
        return None
    
    def _determine_accessibility(self, candidate: SemanticChunkCandidate) -> str:
        """Determine accessibility level for Python code"""
        if candidate.identifier:
            if candidate.identifier.startswith('__') and not candidate.identifier.endswith('__'):
                return "private"
            elif candidate.identifier.startswith('_'):
                return "protected"
        return "public"
    
    def _extract_imports(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract import statements from Python code"""
        imports = []
        
        if candidate.chunk_type == ChunkType.IMPORT:
            if candidate.metadata.get('is_import_group'):
                imports = candidate.metadata.get('import_identifiers', [])
            else:
                imports = [candidate.identifier] if candidate.identifier else []
        else:
            # Look for imports in the content
            import_pattern = r'(?:from\s+[\w.]+\s+)?import\s+([\w.,\s]+)'
            matches = re.findall(import_pattern, candidate.content)
            for match in matches:
                imports.extend([imp.strip() for imp in match.split(',')])
        
        return imports
    
    def _extract_function_calls(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract function calls from Python code"""
        calls = []
        
        # Simple regex pattern for function calls
        call_pattern = r'(\w+)\s*\('
        matches = re.findall(call_pattern, candidate.content)
        
        # Filter out Python keywords and built-ins
        python_keywords = {'if', 'for', 'while', 'with', 'try', 'except', 'def', 'class'}
        
        for match in matches:
            if match not in python_keywords and not match.startswith('__'):
                calls.append(match)
        
        return list(set(calls))
    
    def _extract_variables(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract variables used in Python code"""
        variables = []
        
        # Simple pattern for variable assignments
        var_pattern = r'(\w+)\s*='
        matches = re.findall(var_pattern, candidate.content)
        
        for match in matches:
            if not match.startswith('__') and len(match) > 1:
                variables.append(match)
        
        return list(set(variables))
    
    def _extract_types(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract type annotations from Python code"""
        types = []
        
        # Type annotation patterns
        type_patterns = [
            r':\s*(\w+)',  # Variable annotations
            r'->\s*(\w+)',  # Return type annotations
            r'List\[(\w+)\]',  # List types
            r'Dict\[(\w+),\s*(\w+)\]',  # Dict types
            r'Optional\[(\w+)\]',  # Optional types
        ]
        
        for pattern in type_patterns:
            matches = re.findall(pattern, candidate.content)
            if isinstance(matches[0], tuple) if matches else False:
                for match_tuple in matches:
                    types.extend(match_tuple)
            else:
                types.extend(matches)
        
        return list(set(types))
    
    def _extract_code_patterns(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract Python-specific code patterns"""
        patterns = []
        content = candidate.content
        
        # Async/await patterns
        if 'async def' in content:
            patterns.append('async_function')
        if 'await' in content:
            patterns.append('await_usage')
        
        # Comprehensions
        if '[' in content and 'for' in content and 'in' in content:
            patterns.append('list_comprehension')
        if '{' in content and 'for' in content and 'in' in content:
            patterns.append('dict_comprehension')
        
        # Context managers
        if 'with' in content:
            patterns.append('context_manager')
        
        # Exception handling
        if 'try:' in content:
            patterns.append('exception_handling')
        
        # Decorators
        if '@' in content:
            patterns.append('decorator_usage')
        
        # Lambda functions
        if 'lambda' in content:
            patterns.append('lambda_function')
        
        # Generator patterns
        if 'yield' in content:
            patterns.append('generator')
        
        return patterns