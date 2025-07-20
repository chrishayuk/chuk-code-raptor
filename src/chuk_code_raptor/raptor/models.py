#!/usr/bin/env python3
# src/chuk_code_raptor/raptor/models.py
#!/usr/bin/env python3
"""
RAPTOR Models
=============

Data models for Recursive Abstractive Processing for Tree-Organized Retrieval.
Builds hierarchical abstractions from SemanticChunks with incremental updates.

Path: chuk_code_raptor/raptor/models.py
"""

from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

from chuk_code_raptor.chunking.semantic_chunk import SemanticChunk

class HierarchyLevel(Enum):
    """Levels in the RAPTOR hierarchy"""
    CHUNK = 0           # Raw SemanticChunks (functions, classes)
    FILE = 1            # File-level summaries
    MODULE = 2          # Module/package summaries  
    REPOSITORY = 3      # Repository overview
    DOMAIN = 4          # Cross-repository domain summaries

class SummaryType(Enum):
    """Types of summaries"""
    FUNCTIONAL = "functional"           # What does this do?
    ARCHITECTURAL = "architectural"     # How is this structured?
    API = "api"                        # What interfaces are exposed?
    QUALITY = "quality"                # What's the code quality like?
    DEPENDENCIES = "dependencies"       # What does this depend on?
    PATTERNS = "patterns"              # What patterns are used?

@dataclass
class RaptorNode:
    """
    Node in the RAPTOR hierarchy tree
    
    Represents an abstraction at a specific level with content,
    summaries, and relationships to other nodes.
    """
    
    # Core identity
    node_id: str
    level: HierarchyLevel
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    
    # Content and summaries
    raw_content: str = ""
    summaries: Dict[SummaryType, str] = field(default_factory=dict)
    primary_summary: str = ""
    
    # Metadata
    title: str = ""
    file_path: Optional[str] = None
    module_path: Optional[str] = None
    language: Optional[str] = None
    
    # Source chunks (for leaf nodes)
    source_chunk_ids: List[str] = field(default_factory=list)
    
    # Clustering information
    cluster_id: Optional[str] = None
    topic_keywords: List[str] = field(default_factory=list)
    
    # Content fingerprinting for incremental updates
    content_hash: str = field(init=False)
    summary_hashes: Dict[SummaryType, str] = field(default_factory=dict)
    combined_hash: str = field(init=False)
    
    # Embeddings for search
    content_embedding: Optional[List[float]] = None
    summary_embeddings: Dict[SummaryType, List[float]] = field(default_factory=dict)
    embedding_model: Optional[str] = None
    
    # Quality and importance
    importance_score: float = 0.5
    quality_score: float = 0.5
    complexity_score: float = 0.5
    
    # Change tracking
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields"""
        self._update_hashes()
    
    def _update_hashes(self):
        """Update content and summary hashes"""
        # Content hash
        self.content_hash = hashlib.sha256(self.raw_content.encode('utf-8')).hexdigest()
        
        # Summary hashes
        for summary_type, summary_text in self.summaries.items():
            self.summary_hashes[summary_type] = hashlib.sha256(
                summary_text.encode('utf-8')
            ).hexdigest()
        
        # Combined hash for change detection
        # Convert enum keys to strings for JSON serialization
        summary_hashes_serializable = {
            k.value if hasattr(k, 'value') else str(k): v 
            for k, v in self.summary_hashes.items()
        }
        
        combined_content = json.dumps({
            'content': self.content_hash,
            'summaries': summary_hashes_serializable,
            'children': sorted(self.child_ids)
        }, sort_keys=True)
        self.combined_hash = hashlib.sha256(combined_content.encode('utf-8')).hexdigest()
    
    def add_summary(self, summary_type: SummaryType, text: str, 
                   embedding: Optional[List[float]] = None):
        """Add or update a summary"""
        self.summaries[summary_type] = text
        
        if embedding:
            self.summary_embeddings[summary_type] = embedding
        
        # Update primary summary if this is functional
        if summary_type == SummaryType.FUNCTIONAL:
            self.primary_summary = text
        
        self._update_hashes()
        self.updated_at = datetime.now()
        self.version += 1
    
    def add_child(self, child_id: str):
        """Add a child node"""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
            self._update_hashes()
            self.updated_at = datetime.now()
    
    def remove_child(self, child_id: str):
        """Remove a child node"""
        if child_id in self.child_ids:
            self.child_ids.remove(child_id)
            self._update_hashes()
            self.updated_at = datetime.now()
    
    def set_content_embedding(self, embedding: List[float], model: str):
        """Set content embedding"""
        self.content_embedding = embedding
        self.embedding_model = model
        self.updated_at = datetime.now()
    
    def has_changed(self, other_hash: str) -> bool:
        """Check if node has changed since last hash"""
        return self.combined_hash != other_hash
    
    def get_search_text(self, include_summaries: bool = True) -> str:
        """Get text optimized for search/embedding"""
        parts = [self.title] if self.title else []
        
        if self.primary_summary:
            parts.append(self.primary_summary)
        elif self.raw_content:
            # Truncate raw content for embedding
            content_preview = self.raw_content[:500] + "..." if len(self.raw_content) > 500 else self.raw_content
            parts.append(content_preview)
        
        if include_summaries:
            for summary_type in [SummaryType.FUNCTIONAL, SummaryType.ARCHITECTURAL, SummaryType.API]:
                if summary_type in self.summaries:
                    parts.append(f"[{summary_type.value.upper()}] {self.summaries[summary_type]}")
        
        if self.topic_keywords:
            parts.append(f"Keywords: {', '.join(self.topic_keywords)}")
        
        return "\n".join(parts)
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.child_ids) == 0
    
    @property
    def is_root(self) -> bool:
        """Check if this is a root node"""
        return self.parent_id is None
    
    @property
    def depth_from_root(self) -> int:
        """Get depth from root (level enum value)"""
        return self.level.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'node_id': self.node_id,
            'level': self.level.value,
            'parent_id': self.parent_id,
            'child_ids': self.child_ids,
            'raw_content': self.raw_content[:1000] + "..." if len(self.raw_content) > 1000 else self.raw_content,
            'summaries': {k.value: v for k, v in self.summaries.items()},
            'primary_summary': self.primary_summary,
            'title': self.title,
            'file_path': self.file_path,
            'module_path': self.module_path,
            'language': self.language,
            'source_chunk_ids': self.source_chunk_ids,
            'cluster_id': self.cluster_id,
            'topic_keywords': self.topic_keywords,
            'content_hash': self.content_hash,
            'combined_hash': self.combined_hash,
            'importance_score': self.importance_score,
            'quality_score': self.quality_score,
            'complexity_score': self.complexity_score,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class QueryResult:
    """Result from a RAPTOR query"""
    node_id: str
    level: HierarchyLevel
    content: str
    summary: str
    score: float
    
    # Context
    file_path: Optional[str] = None
    module_path: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'node_id': self.node_id,
            'level': self.level.value,
            'content': self.content,
            'summary': self.summary,
            'score': self.score,
            'file_path': self.file_path,
            'module_path': self.module_path,
            'metadata': self.metadata
        }

@dataclass
class ClusterInfo:
    """Information about a content cluster"""
    cluster_id: str
    level: HierarchyLevel
    topic: str
    keywords: List[str]
    member_ids: List[str]
    centroid_embedding: Optional[List[float]] = None
    quality_score: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'cluster_id': self.cluster_id,
            'level': self.level.value,
            'topic': self.topic,
            'keywords': self.keywords,
            'member_ids': self.member_ids,
            'quality_score': self.quality_score
        }

@dataclass
class HierarchyStats:
    """Statistics about the RAPTOR hierarchy"""
    total_nodes: int = 0
    nodes_by_level: Dict[int, int] = field(default_factory=dict)
    total_chunks: int = 0
    
    # Quality metrics
    average_quality_by_level: Dict[int, float] = field(default_factory=dict)
    coverage_by_file: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    build_time_seconds: Optional[float] = None
    last_update_time: Optional[datetime] = None
    
    # Size metrics
    total_content_size: int = 0
    total_summary_size: int = 0
    compression_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_nodes': self.total_nodes,
            'nodes_by_level': self.nodes_by_level,
            'total_chunks': self.total_chunks,
            'average_quality_by_level': self.average_quality_by_level,
            'coverage_by_file': self.coverage_by_file,
            'build_time_seconds': self.build_time_seconds,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'total_content_size': self.total_content_size,
            'total_summary_size': self.total_summary_size,
            'compression_ratio': self.compression_ratio
        }

# Utility functions

def create_raptor_node_id(level: HierarchyLevel, identifier: str, 
                         file_path: Optional[str] = None) -> str:
    """Create a unique RAPTOR node ID"""
    level_prefix = f"L{level.value}"
    
    if file_path:
        from pathlib import Path
        file_part = Path(file_path).stem
        return f"{level_prefix}:{file_part}:{identifier}"
    else:
        return f"{level_prefix}:{identifier}"

def extract_module_path(file_path: str) -> str:
    """Extract module path from file path"""
    from pathlib import Path
    
    path = Path(file_path)
    
    # Remove common source directories
    parts = list(path.parts)
    if 'src' in parts:
        src_index = parts.index('src')
        parts = parts[src_index + 1:]
    elif 'lib' in parts:
        lib_index = parts.index('lib')
        parts = parts[lib_index + 1:]
    
    # Remove file extension
    if parts:
        parts[-1] = Path(parts[-1]).stem
    
    return '.'.join(parts)

def classify_query_type(query: str) -> str:
    """Classify query to determine optimal RAPTOR level"""
    query_lower = query.lower()
    
    # Architectural queries - start at module/repo level
    if any(word in query_lower for word in ['architecture', 'overview', 'structure', 'design', 'how does', 'explain']):
        return 'architectural'
    
    # Relationship queries - use property graph
    if any(word in query_lower for word in ['calls', 'uses', 'depends', 'imports', 'references']):
        return 'relationship'
    
    # Implementation queries - drill down to chunks
    if any(word in query_lower for word in ['implementation', 'code', 'function', 'method', 'show me', 'exact']):
        return 'implementation'
    
    # API queries - file/module level
    if any(word in query_lower for word in ['api', 'interface', 'endpoint', 'method', 'parameter']):
        return 'api'
    
    # Quality queries - cross-cutting
    if any(word in query_lower for word in ['quality', 'refactor', 'improve', 'issue', 'problem']):
        return 'quality'
    
    return 'general'