# src/chuk_code_raptor/chunking/semantic_chunk.py
#!/usr/bin/env python3
"""
Semantic Chunk Model
====================

Base semantic chunk model that can represent any type of content with
semantic understanding, relationships, and quality metrics.

Supports:
- SemanticChunk: Base semantic-aware chunk
- SemanticCodeChunk: Code-specific semantic chunk (extends CodeChunk)
- SemanticDocumentChunk: Document-specific semantic chunk
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# Import base models
from chuk_code_raptor.core.models import CodeChunk, ChunkType

@dataclass
class ChunkRelationship:
    """Represents a relationship between chunks"""
    source_chunk_id: str
    target_chunk_id: str
    relationship_type: str  # depends_on, calls, imports, references, contains, similar, related
    strength: float = 1.0   # Relationship strength (0.0-1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticTag:
    """A semantic tag for chunk categorization"""
    name: str
    confidence: float = 1.0
    source: str = "manual"  # manual, ast, pattern, ml, nlp
    metadata: Dict[str, Any] = field(default_factory=dict)

class QualityMetric(Enum):
    """Quality metrics for chunks"""
    SEMANTIC_COHERENCE = "semantic_coherence"
    COMPLETENESS = "completeness"
    COUPLING = "coupling"
    REUSABILITY = "reusability"
    COMPLEXITY = "complexity"
    READABILITY = "readability"

class ContentType(Enum):
    """Types of content in chunks"""
    CODE = "code"
    DOCUMENTATION = "documentation"
    MARKDOWN = "markdown"
    HTML = "html"
    PLAINTEXT = "text"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"

@dataclass
class SemanticChunk:
    """
    Base semantic chunk with relationships, dependencies, and quality metrics.
    
    This is the foundation for all semantic-aware chunks, supporting any content type.
    """
    
    # === Core Properties ===
    id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    content_type: ContentType
    language: Optional[str] = None
    
    # === Content Analysis ===
    content_hash: str = field(init=False)
    character_count: int = field(init=False)
    word_count: int = field(init=False)
    
    # === Relationships and Dependencies ===
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    relationships: List[ChunkRelationship] = field(default_factory=list)
    
    # === Semantic Information ===
    semantic_tags: List[SemanticTag] = field(default_factory=list)
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # === Context Awareness ===
    context_chunks: List[str] = field(default_factory=list)
    scope_level: int = 0
    namespace: Optional[str] = None
    
    # === Quality Metrics ===
    quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # === Neural Features ===
    semantic_embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    
    # === Metadata ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        self.character_count = len(self.content)
        self.word_count = len(self.content.split())
    
    @property
    def line_count(self) -> int:
        """Get number of lines in chunk"""
        return self.end_line - self.start_line + 1
    
    @property
    def content_preview(self) -> str:
        """Get preview of content"""
        if len(self.content) <= 200:
            return self.content
        return self.content[:197] + "..."
    
    def add_dependency(self, chunk_id: str, relationship_type: str = "depends_on", 
                      strength: float = 1.0, metadata: Dict[str, Any] = None):
        """Add a dependency relationship"""
        if chunk_id not in self.dependencies:
            self.dependencies.append(chunk_id)
        
        relationship = ChunkRelationship(
            source_chunk_id=self.id,
            target_chunk_id=chunk_id,
            relationship_type=relationship_type,
            strength=strength,
            metadata=metadata or {}
        )
        self.relationships.append(relationship)
        self.updated_at = datetime.now()
    
    def add_semantic_tag(self, name: str, confidence: float = 1.0, 
                        source: str = "manual", metadata: Dict[str, Any] = None):
        """Add a semantic tag"""
        tag = SemanticTag(
            name=name,
            confidence=confidence,
            source=source,
            metadata=metadata or {}
        )
        self.semantic_tags.append(tag)
        self.updated_at = datetime.now()
    
    def set_quality_score(self, metric: QualityMetric, score: float):
        """Set a quality score for this chunk"""
        self.quality_scores[metric.value] = max(0.0, min(1.0, score))
        self.updated_at = datetime.now()
    
    def get_quality_score(self, metric: QualityMetric) -> float:
        """Get quality score for a metric"""
        return self.quality_scores.get(metric.value, 0.0)
    
    @property
    def tag_names(self) -> List[str]:
        """Get list of semantic tag names"""
        return [tag.name for tag in self.semantic_tags]
    
    @property
    def has_semantic_embedding(self) -> bool:
        """Check if chunk has semantic embedding"""
        return self.semantic_embedding is not None and len(self.semantic_embedding) > 0

@dataclass  
class SemanticCodeChunk(SemanticChunk):
    """
    Semantic chunk specifically for code content.
    Extends SemanticChunk with code-specific features.
    """
    
    # === Code-Specific Properties ===
    chunk_type: ChunkType = ChunkType.TEXT_BLOCK
    accessibility: str = "public"  # public/private/protected
    
    # === Code Analysis ===
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    variables_used: List[str] = field(default_factory=list)
    types_used: List[str] = field(default_factory=list)
    ast_node_path: Optional[str] = None
    code_patterns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize code chunk"""
        super().__post_init__()
        if not self.content_type:
            self.content_type = ContentType.CODE
    
    def calculate_coupling_score(self) -> float:
        """Calculate coupling score based on dependencies"""
        if not self.dependencies and not self.dependents:
            return 0.0
        
        total_connections = len(self.dependencies) + len(self.dependents)
        return min(total_connections / 20.0, 1.0)
    
    def calculate_reusability_score(self) -> float:
        """Calculate reusability based on coupling and completeness"""
        coupling = self.calculate_coupling_score()
        completeness = self.get_quality_score(QualityMetric.COMPLETENESS)
        return max(0.0, completeness - (coupling * 0.5))
    
    @property
    def is_highly_coupled(self) -> bool:
        """Check if chunk is highly coupled"""
        return self.calculate_coupling_score() > 0.7
    
    @property
    def is_reusable(self) -> bool:
        """Check if chunk is reusable"""
        return self.calculate_reusability_score() > 0.6
    
    @classmethod
    def from_code_chunk(cls, chunk: CodeChunk) -> 'SemanticCodeChunk':
        """Create SemanticCodeChunk from existing CodeChunk"""
        return cls(
            id=chunk.id,
            file_path=chunk.file_path,
            content=chunk.content,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            content_type=ContentType.CODE,
            language=chunk.language,
            chunk_type=chunk.chunk_type,
            summary=chunk.summary,
            keywords=chunk.keywords,
            semantic_embedding=chunk.embedding,
            embedding_model=chunk.embedding_model,
            metadata=chunk.metadata,
            created_at=chunk.created_at,
            updated_at=chunk.updated_at
        )

@dataclass
class SemanticDocumentChunk(SemanticChunk):
    """
    Semantic chunk specifically for document content.
    Extends SemanticChunk with document-specific features.
    """
    
    # === Document-Specific Properties ===
    section_type: str = "paragraph"  # heading, paragraph, list, table, code_block
    heading_level: Optional[int] = None  # For headings (1-6)
    
    # === Document Analysis ===
    entities: List[str] = field(default_factory=list)        # Named entities
    topics: List[str] = field(default_factory=list)          # Extracted topics
    document_structure: Dict[str, Any] = field(default_factory=dict)
    cross_references: List[str] = field(default_factory=list)  # Links to other sections
    
    def __post_init__(self):
        """Initialize document chunk"""
        super().__post_init__()
        if self.content_type == ContentType.CODE:  # Don't override if explicitly set
            self.content_type = ContentType.DOCUMENTATION
    
    def calculate_readability_score(self) -> float:
        """Calculate readability score based on measurable text properties"""
        # Use actual readability metrics, not arbitrary thresholds
        
        # Sentence count
        sentence_endings = self.content.count('.') + self.content.count('!') + self.content.count('?')
        if sentence_endings == 0:
            return 0.0
        
        # Average sentence length (words per sentence)
        avg_sentence_length = self.word_count / sentence_endings
        
        # Simple readability based on sentence complexity
        # Shorter sentences are generally more readable
        if avg_sentence_length <= 15:
            return 0.9  # Very readable
        elif avg_sentence_length <= 25:
            return 0.7  # Moderately readable
        elif avg_sentence_length <= 35:
            return 0.5  # Complex
        else:
            return 0.3  # Very complex
    
    def calculate_structure_consistency(self) -> float:
        """Calculate how well this chunk follows expected structural patterns"""
        if self.is_heading:
            # Check if heading follows common patterns
            content_clean = self.content.strip()
            
            # Good headings are typically 3-80 characters
            if 3 <= len(content_clean) <= 80:
                score = 0.8
            else:
                score = 0.4
            
            # Bonus for proper heading markers in markdown
            if content_clean.startswith('#'):
                score += 0.2
            
            return min(score, 1.0)
        
        elif self.section_type == "code_block":
            # Code blocks should have some language indicators
            if '```' in self.content or any(keyword in self.content.lower() 
                                          for keyword in ['def ', 'function', 'class ', 'import']):
                return 0.9
            else:
                return 0.5
        
        else:
            # For regular content, check for basic structure
            has_sentences = any(end in self.content for end in '.!?')
            has_words = self.word_count > 0
            
            if has_sentences and has_words:
                return 0.7
            elif has_words:
                return 0.5
            else:
                return 0.1
    
    @property
    def is_heading(self) -> bool:
        """Check if this chunk is a heading"""
        return self.section_type == "heading"
    
    @property
    def is_code_block(self) -> bool:
        """Check if this chunk is a code block"""
        return self.section_type == "code_block"

# Utility functions

def calculate_chunk_similarity(chunk1: SemanticChunk, chunk2: SemanticChunk, 
                             method: str = "content") -> float:
    """Calculate similarity between two chunks"""
    
    if method == "embedding" and chunk1.has_semantic_embedding and chunk2.has_semantic_embedding:
        import numpy as np
        vec1 = np.array(chunk1.semantic_embedding)
        vec2 = np.array(chunk2.semantic_embedding)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    elif method == "content":
        words1 = set(chunk1.content.lower().split())
        words2 = set(chunk2.content.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    elif method == "semantic_tags":
        tags1 = set(chunk1.tag_names)
        tags2 = set(chunk2.tag_names)
        
        intersection = len(tags1.intersection(tags2))
        union = len(tags1.union(tags2))
        
        return intersection / union if union > 0 else 0.0
    
    return 0.0

def find_related_chunks(target_chunk: SemanticChunk, all_chunks: List[SemanticChunk],
                       similarity_threshold: float = 0.7, max_results: int = 10) -> List[SemanticChunk]:
    """Find chunks related to the target chunk"""
    similarities = []
    
    for chunk in all_chunks:
        if chunk.id == target_chunk.id:
            continue
        
        similarity = calculate_chunk_similarity(target_chunk, chunk)
        if similarity >= similarity_threshold:
            similarities.append((similarity, chunk))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in similarities[:max_results]]

def create_chunk_for_content_type(content_type: ContentType, **kwargs) -> SemanticChunk:
    """Factory function to create appropriate chunk type based on content"""
    
    if content_type == ContentType.CODE:
        return SemanticCodeChunk(content_type=content_type, **kwargs)
    elif content_type in [ContentType.DOCUMENTATION, ContentType.MARKDOWN, ContentType.HTML]:
        return SemanticDocumentChunk(content_type=content_type, **kwargs)
    else:
        return SemanticChunk(content_type=content_type, **kwargs)

# Quality analysis functions

def calculate_code_quality_metrics(chunk: SemanticCodeChunk) -> Dict[str, float]:
    """Calculate quality metrics specific to code chunks"""
    metrics = {}
    
    # Only calculate objectively measurable metrics
    
    # Dependency coupling (actual measurable coupling)
    metrics['coupling'] = chunk.calculate_coupling_score()
    
    # Import-to-content ratio (how much external dependencies vs internal logic)
    if chunk.imports:
        import_ratio = len(chunk.imports) / max(chunk.line_count, 1)
        metrics['import_density'] = min(import_ratio, 1.0)
    else:
        metrics['import_density'] = 0.0
    
    # Function call density (how much this chunk calls other functions)
    if chunk.function_calls:
        call_density = len(chunk.function_calls) / max(chunk.line_count, 1)
        metrics['call_density'] = min(call_density, 1.0)
    else:
        metrics['call_density'] = 0.0
    
    # Comment-to-code ratio (if we can detect comments)
    comment_lines = chunk.content.count('#')  # Simple heuristic for Python
    if comment_lines > 0:
        comment_ratio = comment_lines / max(chunk.line_count, 1)
        metrics['comment_ratio'] = min(comment_ratio, 1.0)
    else:
        metrics['comment_ratio'] = 0.0
    
    # Pattern complexity (number of detected patterns)
    if chunk.code_patterns:
        metrics['pattern_complexity'] = min(len(chunk.code_patterns) / 5.0, 1.0)
    else:
        metrics['pattern_complexity'] = 0.0
    
    return metrics

def calculate_document_quality_metrics(chunk: SemanticDocumentChunk) -> Dict[str, float]:
    """Calculate quality metrics specific to document chunks"""
    metrics = {}
    
    # Only calculate metrics that can be objectively measured
    
    # Information density (actual keywords to content ratio)
    if chunk.keywords:
        keyword_density = len(chunk.keywords) / max(chunk.word_count, 1)
        metrics['keyword_density'] = min(keyword_density, 1.0)
    else:
        metrics['keyword_density'] = 0.0
    
    # Cross-reference connectivity (how connected this chunk is)
    if chunk.cross_references:
        metrics['connectivity'] = min(len(chunk.cross_references) / 10.0, 1.0)
    else:
        metrics['connectivity'] = 0.0
    
    # Entity density (named entities per word)
    if chunk.entities:
        entity_density = len(chunk.entities) / max(chunk.word_count, 1)
        metrics['entity_density'] = min(entity_density, 1.0)
    else:
        metrics['entity_density'] = 0.0
    
    # Structural consistency (if it follows expected patterns for its type)
    if chunk.is_heading and chunk.heading_level:
        # Heading has appropriate length for its level
        expected_range = {1: (5, 60), 2: (10, 80), 3: (10, 100)}
        if chunk.heading_level in expected_range:
            min_len, max_len = expected_range[chunk.heading_level]
            if min_len <= len(chunk.content) <= max_len:
                metrics['structural_consistency'] = 1.0
            else:
                metrics['structural_consistency'] = 0.5
        else:
            metrics['structural_consistency'] = 0.7
    
    return metrics

# Example usage
if __name__ == "__main__":
    # Test code chunk
    code_chunk = SemanticCodeChunk(
        id="function_example",
        file_path="example.py",
        content="def process_data(data):\n    return [x * 2 for x in data]",
        start_line=1,
        end_line=2,
        chunk_type=ChunkType.FUNCTION,
        language="python"
    )
    
    code_chunk.add_semantic_tag("data_processing", confidence=0.9, source="pattern")
    code_chunk.function_calls = ["list_comprehension"]
    code_chunk.variables_used = ["data", "x"]
    
    print(f"Code chunk: {code_chunk.id}")
    print(f"Content type: {code_chunk.content_type.value}")
    print(f"Tags: {code_chunk.tag_names}")
    print(f"Quality metrics: {calculate_code_quality_metrics(code_chunk)}")
    
    # Test document chunk
    doc_chunk = SemanticDocumentChunk(
        id="heading_example",
        file_path="README.md",
        content="# Data Processing\n\nThis module handles data processing operations.",
        start_line=1,
        end_line=3,
        content_type=ContentType.MARKDOWN,
        section_type="heading",
        heading_level=1
    )
    
    doc_chunk.add_semantic_tag("documentation", confidence=1.0, source="structure")
    doc_chunk.topics = ["data_processing", "module_overview"]
    
    print(f"\nDocument chunk: {doc_chunk.id}")
    print(f"Content type: {doc_chunk.content_type.value}")
    print(f"Section type: {doc_chunk.section_type}")
    print(f"Is heading: {doc_chunk.is_heading}")
    print(f"Topics: {doc_chunk.topics}")
    print(f"Quality metrics: {calculate_document_quality_metrics(doc_chunk)}")