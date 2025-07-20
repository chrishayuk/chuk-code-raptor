# src/chuk_code_raptor/chunking/semantic_chunk.py
"""
Enhanced Semantic Chunk Model
=============================

Enhanced semantic chunk model with future-ready foundations:
- Content-addressable fingerprints for Merkle tree support
- Graph-ready relationships for property graphs
- Embedding support for vector search
- Change tracking for incremental updates

Language-agnostic foundation for all semantic-aware chunks.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

# Import base models
from chuk_code_raptor.core.models import ChunkType

@dataclass
class ChunkRelationship:
    """Enhanced relationship between chunks with graph integration support"""
    source_chunk_id: str
    target_chunk_id: str
    relationship_type: str  # depends_on, calls, imports, references, contains, similar, related
    strength: float = 1.0   # Relationship strength (0.0-1.0)
    context: str = ""       # How/where the relationship occurs
    line_number: Optional[int] = None  # Location of relationship
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_graph_edge(self) -> Dict[str, Any]:
        """Convert to graph edge format for future property graph integration"""
        return {
            'source': self.source_chunk_id,
            'target': self.target_chunk_id,
            'type': self.relationship_type,
            'weight': self.strength,
            'context': self.context,
            'line': self.line_number,
            'metadata': self.metadata
        }

@dataclass
class SemanticTag:
    """Enhanced semantic tag with confidence and source tracking"""
    name: str
    confidence: float = 1.0
    source: str = "manual"  # manual, ast, pattern, ml, nlp, analysis
    category: str = "general"  # general, pattern, architecture, quality, complexity
    metadata: Dict[str, Any] = field(default_factory=dict)

class QualityMetric(Enum):
    """Quality metrics for chunks"""
    SEMANTIC_COHERENCE = "semantic_coherence"
    COMPLETENESS = "completeness"
    COUPLING = "coupling"
    REUSABILITY = "reusability"
    COMPLEXITY = "complexity"
    READABILITY = "readability"
    DOCUMENTATION_QUALITY = "documentation_quality"
    TYPE_SAFETY = "type_safety"
    ERROR_HANDLING = "error_handling"
    MAINTAINABILITY = "maintainability"
    TESTABILITY = "testability"

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

class ChunkComplexity(Enum):
    """Complexity levels for chunks"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

@dataclass
class CodePattern:
    """Detected code pattern with confidence"""
    pattern_name: str
    confidence: float
    evidence: List[str]
    category: str  # design_pattern, functional_pattern, architectural_pattern

@dataclass
class SemanticChunk:
    """
    Enhanced semantic chunk with future-ready foundations.
    
    Key improvements:
    - Content fingerprinting for change detection
    - Graph-ready relationships  
    - Embedding support
    - Quality scoring framework
    """
    
    # === Core Properties ===
    id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    content_type: ContentType
    language: Optional[str] = None
    
    # === Chunk Type and Classification ===
    chunk_type: ChunkType = ChunkType.TEXT_BLOCK
    complexity_level: ChunkComplexity = ChunkComplexity.SIMPLE
    
    # === Content Analysis ===
    content_hash: str = field(init=False)
    character_count: int = field(init=False)
    word_count: int = field(init=False)
    
    # === Enhanced Relationships and Dependencies ===
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    relationships: List[ChunkRelationship] = field(default_factory=list)
    
    # === Semantic Information ===
    semantic_tags: List[SemanticTag] = field(default_factory=list)
    detected_patterns: List[CodePattern] = field(default_factory=list)
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # === Context Awareness ===
    context_chunks: List[str] = field(default_factory=list)
    parent_context: Optional[str] = None  # Parent class/module context
    scope_level: int = 0
    namespace: Optional[str] = None
    
    # === Quality Metrics ===
    quality_scores: Dict[str, float] = field(default_factory=dict)
    importance_score: float = 0.5
    
    # === Future-Ready Features ===
    
    # Content fingerprinting for Merkle trees
    content_fingerprint: Optional[str] = field(init=False)
    dependency_fingerprint: Optional[str] = field(init=False)
    combined_fingerprint: Optional[str] = field(init=False)
    
    # Change tracking
    version: int = 1
    last_modified: datetime = field(default_factory=datetime.now)
    
    # Embedding support
    semantic_embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    embedding_version: int = 1
    
    # Graph integration preparation  
    graph_node_id: Optional[str] = field(init=False)
    
    # === Metadata ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Enhanced post-initialization with fingerprinting"""
        self.content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        self.character_count = len(self.content)
        self.word_count = len(self.content.split())
        
        # Create fingerprints for change detection
        self._create_fingerprints()
        
        # Set graph node ID
        self.graph_node_id = f"node_{self.id}"
        
        # Basic complexity detection
        self._detect_basic_complexity()
    
    def _create_fingerprints(self):
        """Create content-addressable fingerprints for Merkle tree support"""
        # Content fingerprint
        self.content_fingerprint = hashlib.sha256(self.content.encode('utf-8')).hexdigest()
        
        # Dependency fingerprint (for change propagation)
        deps_str = json.dumps(sorted(self.dependencies), sort_keys=True)
        self.dependency_fingerprint = hashlib.sha256(deps_str.encode('utf-8')).hexdigest()
        
        # Combined fingerprint for quick comparison
        combined_data = f"{self.content_fingerprint}:{self.dependency_fingerprint}:{self.chunk_type.value}"
        self.combined_fingerprint = hashlib.sha256(combined_data.encode('utf-8')).hexdigest()
    
    def _detect_basic_complexity(self):
        """Detect basic complexity indicators"""
        complexity_score = 0
        
        # Line count factor
        if self.line_count > 50:
            complexity_score += 2
        elif self.line_count > 20:
            complexity_score += 1
        
        # Nesting indicators (braces, indentation)
        complexity_score += self.content.count('{')
        complexity_score += self.content.count('(')
        complexity_score += len([line for line in self.content.split('\n') if len(line) - len(line.lstrip()) > 8])
        
        # Classify complexity level
        if complexity_score <= 2:
            self.complexity_level = ChunkComplexity.SIMPLE
        elif complexity_score <= 5:
            self.complexity_level = ChunkComplexity.MODERATE
        elif complexity_score <= 10:
            self.complexity_level = ChunkComplexity.COMPLEX
        else:
            self.complexity_level = ChunkComplexity.VERY_COMPLEX
    
    # === Change Detection & Merkle Support ===
    
    def has_changed(self, other_fingerprint: str) -> bool:
        """Check if content has changed (Merkle tree support)"""
        return self.combined_fingerprint != other_fingerprint
    
    def has_content_changed(self, other_content_fingerprint: str) -> bool:
        """Check if just content has changed"""
        return self.content_fingerprint != other_content_fingerprint
    
    def has_dependencies_changed(self, other_dependency_fingerprint: str) -> bool:
        """Check if dependencies have changed"""
        return self.dependency_fingerprint != other_dependency_fingerprint
    
    def update_fingerprints(self):
        """Update fingerprints after changes"""
        self._create_fingerprints()
        self.version += 1
        self.last_modified = datetime.now()
        self.updated_at = datetime.now()
    
    # === Graph Integration Support ===
    
    def add_relationship(self, target_chunk_id: str, relationship_type: str, 
                        strength: float = 1.0, context: str = "", 
                        line_number: Optional[int] = None):
        """Add an enhanced relationship with graph support"""
        if target_chunk_id not in self.dependencies:
            self.dependencies.append(target_chunk_id)
        
        relationship = ChunkRelationship(
            source_chunk_id=self.id,
            target_chunk_id=target_chunk_id,
            relationship_type=relationship_type,
            strength=strength,
            context=context,
            line_number=line_number
        )
        self.relationships.append(relationship)
        self.update_fingerprints()  # Update fingerprints when relationships change
    
    def get_graph_edges(self) -> List[Dict[str, Any]]:
        """Get all relationships as graph edges for property graph integration"""
        return [rel.to_graph_edge() for rel in self.relationships]
    
    # === Embedding Support ===
    
    def set_embedding(self, embedding: List[float], model: str, version: int = 1):
        """Set semantic embedding with versioning"""
        self.semantic_embedding = embedding
        self.embedding_model = model
        self.embedding_version = version
        self.updated_at = datetime.now()
    
    def needs_embedding_update(self, current_model: str, current_version: int) -> bool:
        """Check if embedding needs updating"""
        return (
            self.semantic_embedding is None or
            self.embedding_model != current_model or
            self.embedding_version < current_version
        )
    
    def get_embedding_text(self, include_context: bool = True) -> str:
        """Get text optimized for embedding generation"""
        parts = []
        
        # Add semantic context
        if self.chunk_type:
            parts.append(f"[{self.chunk_type.value.upper()}]")
        
        # Add high-confidence tags
        high_conf_tags = self.high_confidence_tags
        if high_conf_tags:
            parts.append(f"Tags: {', '.join(high_conf_tags[:5])}")  # Top 5 tags
        
        # Add detected patterns
        if self.detected_patterns:
            pattern_names = [p.pattern_name for p in self.detected_patterns]
            parts.append(f"Patterns: {', '.join(pattern_names)}")
        
        # Add the actual content
        parts.append(self.content)
        
        # Add dependency context if requested
        if include_context and self.dependencies:
            deps = [dep for dep in self.dependencies[:3]]  # Top 3
            parts.append(f"Dependencies: {', '.join(deps)}")
        
        return "\n".join(parts)
    
    # === Enhanced Properties ===
    
    @property
    def line_count(self) -> int:
        """Get number of lines in chunk"""
        return self.end_line - self.start_line + 1
    
    @property
    def content_preview(self) -> str:
        """Get enhanced preview of content"""
        if len(self.content) <= 200:
            return self.content
        
        # Try to end at a natural boundary
        preview = self.content[:197]
        
        # Find last complete line or sentence
        last_newline = preview.rfind('\n')
        last_sentence = max(preview.rfind('.'), preview.rfind('!'), preview.rfind('?'))
        
        if last_newline > 150:
            preview = preview[:last_newline]
        elif last_sentence > 150:
            preview = preview[:last_sentence + 1]
        
        return preview + "..."
    
    @property
    def is_complex(self) -> bool:
        """Check if chunk is complex"""
        return self.complexity_level in [ChunkComplexity.COMPLEX, ChunkComplexity.VERY_COMPLEX]
    
    @property
    def has_semantic_embedding(self) -> bool:
        """Check if chunk has semantic embedding"""
        return self.semantic_embedding is not None and len(self.semantic_embedding) > 0
    
    @property
    def tag_names(self) -> List[str]:
        """Get list of semantic tag names"""
        return [tag.name for tag in self.semantic_tags]
    
    @property
    def high_confidence_tags(self) -> List[str]:
        """Get tags with high confidence (>0.8)"""
        return [tag.name for tag in self.semantic_tags if tag.confidence > 0.8]
    
    # === Quality Management ===
    
    def add_semantic_tag(self, name: str, confidence: float = 1.0, 
                        source: str = "manual", category: str = "general",
                        metadata: Dict[str, Any] = None):
        """Add an enhanced semantic tag"""
        tag = SemanticTag(
            name=name,
            confidence=confidence,
            source=source,
            category=category,
            metadata=metadata or {}
        )
        self.semantic_tags.append(tag)
        self.updated_at = datetime.now()
    
    def add_tag(self, name: str, confidence: float = 1.0, source: str = "manual"):
        """Alias for add_semantic_tag for compatibility"""
        self.add_semantic_tag(name, confidence, source)
    
    def set_quality_score(self, metric: QualityMetric, score: float):
        """Set a quality score for this chunk"""
        self.quality_scores[metric.value] = max(0.0, min(1.0, score))
        self.updated_at = datetime.now()
    
    def get_quality_score(self, metric: QualityMetric) -> float:
        """Get quality score for a metric"""
        return self.quality_scores.get(metric.value, 0.0)
    
    def calculate_overall_quality_score(self) -> float:
        """Calculate overall quality score from all metrics"""
        if not self.quality_scores:
            return self.importance_score
        
        # Weight different quality metrics
        weights = {
            QualityMetric.DOCUMENTATION_QUALITY.value: 0.2,
            QualityMetric.COMPLEXITY.value: 0.2,
            QualityMetric.READABILITY.value: 0.2,
            QualityMetric.MAINTAINABILITY.value: 0.2,
            QualityMetric.COMPLETENESS.value: 0.2
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in self.quality_scores:
                weighted_score += self.quality_scores[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return self.importance_score
    
    def get_summary_info(self) -> Dict[str, Any]:
        """Get comprehensive summary information"""
        return {
            'id': self.id,
            'type': self.chunk_type.value,
            'complexity': self.complexity_level.value,
            'quality_score': self.calculate_overall_quality_score(),
            'importance': self.importance_score,
            'size': {
                'characters': self.character_count,
                'lines': self.line_count,
                'words': self.word_count
            },
            'fingerprints': {
                'content': self.content_fingerprint[:8] if self.content_fingerprint else None,
                'combined': self.combined_fingerprint[:8] if self.combined_fingerprint else None,
            },
            'patterns': [p.pattern_name for p in self.detected_patterns],
            'high_confidence_tags': self.high_confidence_tags,
            'dependencies_count': len(self.dependencies),
            'relationships_count': len(self.relationships),
            'has_embedding': self.has_semantic_embedding,
            'version': self.version
        }
    
    def to_dict(self, include_embedding: bool = True) -> Dict[str, Any]:
        """Convert SemanticChunk to dictionary for JSON serialization
        
        Args:
            include_embedding: Whether to include the actual embedding data (default: True)
                            Set to False to save space when embeddings are large
        """
        # Convert enums to their values
        chunk_type_value = self.chunk_type.value if hasattr(self.chunk_type, 'value') else str(self.chunk_type)
        content_type_value = self.content_type.value if hasattr(self.content_type, 'value') else str(self.content_type)
        complexity_value = self.complexity_level.value if hasattr(self.complexity_level, 'value') else str(self.complexity_level)
        
        # Convert relationships to dicts
        relationships_data = []
        for rel in self.relationships:
            relationships_data.append({
                'source_chunk_id': rel.source_chunk_id,
                'target_chunk_id': rel.target_chunk_id,
                'relationship_type': rel.relationship_type,
                'strength': rel.strength,
                'context': rel.context,
                'line_number': rel.line_number,
                'metadata': rel.metadata
            })
        
        # Convert semantic tags to dicts
        tags_data = []
        for tag in self.semantic_tags:
            tags_data.append({
                'name': tag.name,
                'confidence': tag.confidence,
                'source': tag.source,
                'category': tag.category,
                'metadata': tag.metadata
            })
        
        # Convert detected patterns to dicts
        patterns_data = []
        for pattern in self.detected_patterns:
            patterns_data.append({
                'pattern_name': pattern.pattern_name,
                'confidence': pattern.confidence,
                'evidence': pattern.evidence,
                'category': pattern.category
            })
        
        return {
            'id': self.id,
            'file_path': self.file_path,
            'content': self.content,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'content_type': content_type_value,
            'language': self.language,
            'chunk_type': chunk_type_value,
            'complexity_level': complexity_value,
            
            # Content analysis
            'content_hash': self.content_hash,
            'character_count': self.character_count,
            'word_count': self.word_count,
            
            # Dependencies and relationships
            'dependencies': self.dependencies,
            'dependents': self.dependents,
            'references': self.references,
            'relationships': relationships_data,
            
            # Semantic information
            'semantic_tags': tags_data,
            'detected_patterns': patterns_data,
            'summary': self.summary,
            'keywords': self.keywords,
            
            # Context
            'context_chunks': self.context_chunks,
            'parent_context': self.parent_context,
            'scope_level': self.scope_level,
            'namespace': self.namespace,
            
            # Quality metrics
            'quality_scores': self.quality_scores,
            'importance_score': self.importance_score,
            
            # Fingerprints
            'content_fingerprint': self.content_fingerprint,
            'dependency_fingerprint': self.dependency_fingerprint,
            'combined_fingerprint': self.combined_fingerprint,
            
            # Change tracking
            'version': self.version,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
            
            # Embedding - FIX: Include the actual embedding data
            'semantic_embedding': self.semantic_embedding if include_embedding else None,
            'has_semantic_embedding': self.has_semantic_embedding,
            'embedding_model': self.embedding_model,
            'embedding_version': self.embedding_version,
            
            # Graph integration
            'graph_node_id': self.graph_node_id,
            
            # Metadata
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticChunk':
        """Create SemanticChunk from dictionary"""
        from chuk_code_raptor.core.models import ChunkType
        
        # Convert string values back to enums
        chunk_type = ChunkType(data['chunk_type']) if isinstance(data['chunk_type'], str) else data['chunk_type']
        content_type = ContentType(data['content_type']) if isinstance(data['content_type'], str) else data['content_type']
        complexity_level = ChunkComplexity(data['complexity_level']) if isinstance(data['complexity_level'], str) else data['complexity_level']
        
        # Convert datetime strings back to datetime objects
        created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now()
        updated_at = datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.now()
        last_modified = datetime.fromisoformat(data['last_modified']) if data.get('last_modified') else datetime.now()
        
        # Create the chunk
        chunk = cls(
            id=data['id'],
            file_path=data['file_path'],
            content=data['content'],
            start_line=data['start_line'],
            end_line=data['end_line'],
            content_type=content_type,
            language=data.get('language'),
            chunk_type=chunk_type,
            complexity_level=complexity_level,
            dependencies=data.get('dependencies', []),
            dependents=data.get('dependents', []),
            references=data.get('references', []),
            summary=data.get('summary'),
            keywords=data.get('keywords', []),
            context_chunks=data.get('context_chunks', []),
            parent_context=data.get('parent_context'),
            scope_level=data.get('scope_level', 0),
            namespace=data.get('namespace'),
            quality_scores=data.get('quality_scores', {}),
            importance_score=data.get('importance_score', 0.5),
            version=data.get('version', 1),
            # FIX: Properly restore the embedding data
            semantic_embedding=data.get('semantic_embedding'),
            embedding_model=data.get('embedding_model'),
            embedding_version=data.get('embedding_version', 1),
            metadata=data.get('metadata', {}),
            created_at=created_at,
            updated_at=updated_at
        )
        
        # Restore fingerprints
        chunk.content_fingerprint = data.get('content_fingerprint')
        chunk.dependency_fingerprint = data.get('dependency_fingerprint')
        chunk.combined_fingerprint = data.get('combined_fingerprint')
        chunk.last_modified = last_modified
        
        # Restore relationships
        for rel_data in data.get('relationships', []):
            relationship = ChunkRelationship(
                source_chunk_id=rel_data['source_chunk_id'],
                target_chunk_id=rel_data['target_chunk_id'],
                relationship_type=rel_data['relationship_type'],
                strength=rel_data['strength'],
                context=rel_data['context'],
                line_number=rel_data.get('line_number'),
                metadata=rel_data.get('metadata', {})
            )
            chunk.relationships.append(relationship)
        
        # Restore semantic tags
        for tag_data in data.get('semantic_tags', []):
            tag = SemanticTag(
                name=tag_data['name'],
                confidence=tag_data['confidence'],
                source=tag_data['source'],
                category=tag_data['category'],
                metadata=tag_data.get('metadata', {})
            )
            chunk.semantic_tags.append(tag)
        
        # Restore detected patterns
        for pattern_data in data.get('detected_patterns', []):
            pattern = CodePattern(
                pattern_name=pattern_data['pattern_name'],
                confidence=pattern_data['confidence'],
                evidence=pattern_data['evidence'],
                category=pattern_data['category']
            )
            chunk.detected_patterns.append(pattern)
        
        return chunk

# Utility functions

def create_chunk_id(file_path: str, start_line: int, chunk_type: ChunkType, 
                   identifier: str = None) -> str:
    """
    Create unique chunk ID from components.
    """
    from pathlib import Path
    
    # Get just the filename without path
    file_name = Path(file_path).stem
    
    # Get chunk type string
    type_str = chunk_type.value if hasattr(chunk_type, 'value') else str(chunk_type)
    
    # Clean identifier if provided
    if identifier:
        # Remove special characters and make safe for ID
        clean_identifier = "".join(c for c in identifier if c.isalnum() or c in '_-').strip()
        if not clean_identifier:
            clean_identifier = "chunk"
    else:
        clean_identifier = "chunk"
    
    # Create ID in format: filename:type:identifier:line
    return f"{file_name}:{type_str}:{clean_identifier}:{start_line}"

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