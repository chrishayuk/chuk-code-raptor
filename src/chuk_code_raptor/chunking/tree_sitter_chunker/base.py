# chuk_code_raptor/chunking/tree_sitter_chunker/base.py
# chuk_code_raptor/chunking/tree_sitter_chunker/base.py
"""
Semantic Tree-sitter Base Classes
=================================

Base classes for language-specific tree-sitter chunkers that produce SemanticChunk objects.
Clean implementation using tree-sitter AST analysis with full semantic understanding.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import logging

import tree_sitter
from tree_sitter import Language, Parser, Node

from chuk_code_raptor.chunking.semantic_chunk import SemanticChunk, SemanticCodeChunk, ContentType, ChunkRelationship
from chuk_code_raptor.core.models import ChunkType
from ..base import BaseChunker
from ..config import ChunkingConfig, ChunkingStrategy
from ..parsers.base import ParseContext

logger = logging.getLogger(__name__)

@dataclass
class SemanticChunkCandidate:
    """A candidate semantic chunk extracted from tree-sitter analysis"""
    node: 'Node'
    content: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    node_type: str
    chunk_type: ChunkType
    content_type: ContentType
    importance_score: float
    identifier: Optional[str] = None
    dependencies: List[str] = None
    semantic_tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.semantic_tags is None:
            self.semantic_tags = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class LanguageConfig:
    """Configuration for a specific language's tree-sitter parsing"""
    language_name: str
    file_extensions: Set[str]
    
    # Node types that represent meaningful chunks
    chunk_node_types: Dict[str, ChunkType]
    
    # Node types that should be preserved as complete units
    atomic_node_types: Set[str]
    
    # Node types that can be safely split
    splittable_node_types: Set[str]
    
    # Importance scoring weights for different node types
    importance_weights: Dict[str, float]
    
    # Language-specific configuration
    language_specific: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.language_specific is None:
            self.language_specific = {}

class BaseSemanticTreeSitterChunker(BaseChunker, ABC):
    """
    Base class for language-specific tree-sitter chunkers that produce SemanticChunk objects.
    
    Each language should subclass this and implement language-specific semantic analysis.
    """
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.strategy = ChunkingStrategy.STRUCTURAL
        
        # Language-specific configuration
        self.language_config: LanguageConfig = self._load_language_config()
        
        # Tree-sitter components
        self.language: Optional[Language] = None
        self.parser: Optional[Parser] = None
        
        # Update supported languages/extensions from config
        self.supported_languages = {self.language_config.language_name}
        self.supported_extensions = self.language_config.file_extensions
        
        # Initialize tree-sitter for this language
        self._initialize_tree_sitter()
    
    @abstractmethod
    def _load_language_config(self) -> LanguageConfig:
        """Load language-specific configuration"""
        pass
    
    @abstractmethod
    def _get_tree_sitter_language(self) -> Language:
        """Get the tree-sitter Language object"""
        pass
    
    def _initialize_tree_sitter(self):
        """Initialize tree-sitter parser for this language"""
        self.language = self._get_tree_sitter_language()
        self.parser = Parser(self.language)
        logger.debug(f"Initialized tree-sitter parser for {self.language_config.language_name}")
    
    def can_chunk(self, language: str, file_extension: str) -> bool:
        """Check if this chunker can handle the given language/extension"""
        return (language == self.language_config.language_name or 
                file_extension in self.language_config.file_extensions)
    
    def get_priority(self, language: str, file_extension: str) -> int:
        """Return priority for this chunker"""
        if language == self.language_config.language_name:
            return 1000  # Highest priority for exact language match
        elif file_extension in self.language_config.file_extensions:
            return 800   # High priority for extension match
        return 0
    
    def parse_content(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse content and return SemanticChunk objects"""
        if not content.strip():
            return []
        
        return self._parse_with_tree_sitter(content, context)
    
    def chunk_content(self, content: str, language: str, file_path: str) -> List[SemanticChunk]:
        """Legacy interface - creates ParseContext and calls parse_content"""
        context = ParseContext(
            file_path=file_path,
            language=language,
            content_type=self._detect_content_type(file_path),
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            enable_semantic_analysis=True,
            enable_dependency_tracking=True
        )
        
        return self.parse_content(content, context)
    
    def _detect_content_type(self, file_path: str) -> ContentType:
        """Detect content type from file path"""
        extension = file_path.lower().split('.')[-1] if '.' in file_path else ''
        
        if self.language_config.language_name in {'python', 'javascript', 'typescript', 'rust', 'go', 'java', 'cpp', 'c'}:
            return ContentType.CODE
        elif extension in {'md', 'markdown'}:
            return ContentType.MARKDOWN
        elif extension in {'html', 'htm'}:
            return ContentType.HTML
        elif extension in {'json'}:
            return ContentType.JSON
        elif extension in {'yaml', 'yml'}:
            return ContentType.YAML
        elif extension in {'xml'}:
            return ContentType.XML
        else:
            return ContentType.CODE
    
    def _parse_with_tree_sitter(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Main tree-sitter parsing logic"""
        # Parse the content
        tree = self.parser.parse(bytes(content, 'utf8'))
        root_node = tree.root_node
        
        # Extract semantic chunk candidates
        candidates = self._extract_semantic_candidates(root_node, content, context)
        
        # Apply language-specific enhancements
        candidates = self._enhance_candidates(candidates, content, context)
        
        # Convert to SemanticChunk objects
        chunks = self._candidates_to_semantic_chunks(candidates, context)
        
        # Build relationships between chunks
        if context.enable_dependency_tracking:
            chunks = self._build_chunk_relationships(chunks, content, context)
        
        return chunks
    
    def _extract_semantic_candidates(self, root_node: Node, content: str, 
                                   context: ParseContext) -> List[SemanticChunkCandidate]:
        """Extract semantic chunk candidates from AST"""
        candidates = []
        
        def traverse_node(node: Node, depth: int = 0):
            if depth > 50:  # Prevent infinite recursion
                return
            
            # Check if this node represents a meaningful chunk
            if node.type in self.language_config.chunk_node_types:
                chunk_type = self.language_config.chunk_node_types[node.type]
                
                candidate = SemanticChunkCandidate(
                    node=node,
                    content=content[node.start_byte:node.end_byte],
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    node_type=node.type,
                    chunk_type=chunk_type,
                    content_type=context.content_type,
                    importance_score=self._calculate_node_importance(node),
                    identifier=self._extract_identifier(node, content),
                    semantic_tags=self._extract_semantic_tags(node, content),
                    metadata={
                        'tree_sitter_type': node.type,
                        'ast_depth': depth,
                        'has_children': node.child_count > 0,
                        'byte_size': node.end_byte - node.start_byte,
                        'language': context.language
                    }
                )
                
                # Extract dependencies for this candidate
                if context.enable_dependency_tracking:
                    candidate.dependencies = self._extract_dependencies(node, content)
                
                candidates.append(candidate)
            
            # Continue traversing children
            for child in node.children:
                traverse_node(child, depth + 1)
        
        traverse_node(root_node)
        return candidates
    
    def _calculate_node_importance(self, node: Node) -> float:
        """Calculate importance score for a node"""
        base_score = self.language_config.importance_weights.get(node.type, 0.5)
        
        # Adjust based on size
        size_factor = min(1.0, (node.end_byte - node.start_byte) / 1000)
        
        # Adjust based on complexity (number of children)
        complexity_factor = min(1.0, node.child_count / 10)
        
        # Combine factors
        final_score = (base_score * 0.6 + size_factor * 0.2 + complexity_factor * 0.2)
        
        return min(1.0, max(0.0, final_score))
    
    @abstractmethod
    def _extract_identifier(self, node: Node, content: str) -> Optional[str]:
        """Extract identifier from node - language specific"""
        pass
    
    @abstractmethod
    def _extract_semantic_tags(self, node: Node, content: str) -> List[str]:
        """Extract semantic tags from node - language specific"""
        pass
    
    @abstractmethod
    def _extract_dependencies(self, node: Node, content: str) -> List[str]:
        """Extract dependencies from node - language specific"""
        pass
    
    @abstractmethod
    def _enhance_candidates(self, candidates: List[SemanticChunkCandidate], 
                          content: str, context: ParseContext) -> List[SemanticChunkCandidate]:
        """Apply language-specific enhancements to candidates"""
        pass
    
    def _candidates_to_semantic_chunks(self, candidates: List[SemanticChunkCandidate],
                                     context: ParseContext) -> List[SemanticChunk]:
        """Convert candidates to SemanticChunk objects"""
        chunks = []
        
        for candidate in candidates:
            # Create appropriate chunk type based on content
            if candidate.content_type == ContentType.CODE:
                chunk = SemanticCodeChunk(
                    id=self._create_chunk_id(candidate, context),
                    file_path=context.file_path,
                    content=candidate.content,
                    start_line=candidate.start_line,
                    end_line=candidate.end_line,
                    content_type=candidate.content_type,
                    language=context.language,
                    chunk_type=candidate.chunk_type,
                    accessibility=self._determine_accessibility(candidate),
                    imports=self._extract_imports(candidate),
                    exports=self._extract_exports(candidate),
                    function_calls=self._extract_function_calls(candidate),
                    variables_used=self._extract_variables(candidate),
                    types_used=self._extract_types(candidate),
                    ast_node_path=f"{candidate.node_type}:{candidate.start_line}",
                    code_patterns=self._extract_code_patterns(candidate),
                    dependencies=candidate.dependencies,
                    summary=self._generate_summary(candidate),
                    keywords=self._extract_keywords(candidate),
                    metadata={
                        'extraction_method': 'tree_sitter',
                        'tree_sitter_node_type': candidate.node_type,
                        'importance_score': candidate.importance_score,
                        'byte_range': (candidate.start_byte, candidate.end_byte),
                        **candidate.metadata
                    }
                )
            else:
                chunk = SemanticChunk(
                    id=self._create_chunk_id(candidate, context),
                    file_path=context.file_path,
                    content=candidate.content,
                    start_line=candidate.start_line,
                    end_line=candidate.end_line,
                    content_type=candidate.content_type,
                    language=context.language,
                    dependencies=candidate.dependencies,
                    summary=self._generate_summary(candidate),
                    keywords=self._extract_keywords(candidate),
                    metadata={
                        'extraction_method': 'tree_sitter',
                        'tree_sitter_node_type': candidate.node_type,
                        'importance_score': candidate.importance_score,
                        'byte_range': (candidate.start_byte, candidate.end_byte),
                        **candidate.metadata
                    }
                )
            
            # Add semantic tags
            for tag_name in candidate.semantic_tags:
                chunk.add_semantic_tag(tag_name, confidence=0.9, source="tree_sitter")
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_id(self, candidate: SemanticChunkCandidate, context: ParseContext) -> str:
        """Create unique chunk ID"""
        file_name = context.file_path.split('/')[-1] if '/' in context.file_path else context.file_path
        identifier = candidate.identifier or candidate.node_type
        return f"{file_name}:{candidate.chunk_type.value}:{identifier}:{candidate.start_line}"
    
    def _determine_accessibility(self, candidate: SemanticChunkCandidate) -> str:
        """Determine accessibility level - override in language chunkers"""
        return "public"
    
    def _extract_imports(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract import statements - override in language chunkers"""
        return []
    
    def _extract_exports(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract export statements - override in language chunkers"""
        return []
    
    def _extract_function_calls(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract function calls - override in language chunkers"""
        return []
    
    def _extract_variables(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract variables used - override in language chunkers"""
        return []
    
    def _extract_types(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract types used - override in language chunkers"""
        return []
    
    def _extract_code_patterns(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract code patterns - override in language chunkers"""
        return []
    
    def _generate_summary(self, candidate: SemanticChunkCandidate) -> Optional[str]:
        """Generate summary for chunk - override in language chunkers"""
        if candidate.identifier:
            return f"{candidate.chunk_type.value.title()}: {candidate.identifier}"
        return f"{candidate.chunk_type.value.title()} ({candidate.node_type})"
    
    def _extract_keywords(self, candidate: SemanticChunkCandidate) -> List[str]:
        """Extract keywords from content - basic implementation"""
        # Simple keyword extraction - can be enhanced
        content_words = candidate.content.lower().split()
        keywords = []
        
        # Language-specific keywords
        if candidate.metadata.get('language') == 'python':
            python_keywords = {'def', 'class', 'import', 'from', 'async', 'await', 'lambda', 'yield'}
            keywords.extend([word for word in content_words if word in python_keywords])
        
        # Add identifier as keyword
        if candidate.identifier:
            keywords.append(candidate.identifier.lower())
        
        return list(set(keywords))
    
    def _build_chunk_relationships(self, chunks: List[SemanticChunk], content: str, 
                                 context: ParseContext) -> List[SemanticChunk]:
        """Build relationships between chunks"""
        # Create a mapping of identifiers to chunks
        chunk_map = {}
        for chunk in chunks:
            if hasattr(chunk, 'ast_node_path'):
                chunk_map[chunk.ast_node_path] = chunk
        
        # Analyze dependencies and create relationships
        for chunk in chunks:
            for dep in chunk.dependencies:
                # Find the dependency chunk
                dep_chunk = None
                for other_chunk in chunks:
                    if (hasattr(other_chunk, 'ast_node_path') and 
                        dep in other_chunk.ast_node_path):
                        dep_chunk = other_chunk
                        break
                
                if dep_chunk:
                    chunk.add_dependency(
                        dep_chunk.id, 
                        relationship_type="depends_on",
                        strength=0.8
                    )
        
        return chunks