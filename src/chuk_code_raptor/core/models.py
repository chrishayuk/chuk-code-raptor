#!/usr/bin/env python3
"""
Core Data Models
================

Defines the fundamental data structures used throughout the CodeRaptor system.
These models represent files, chunks, and hierarchical structures.

Usage:
    from chuk_code_raptor.core.models import FileInfo, CodeChunk, HierarchyNode
    
    file_info = FileInfo(
        path="src/main.py",
        language="python",
        content_hash="abc123",
        ...
    )
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json

class ChunkType(Enum):
    """Types of code chunks"""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    SECTION = "section"        # For markdown sections
    TEXT_BLOCK = "text_block"  # Generic text chunks
    COMMENT = "comment"
    IMPORT = "import"
    VARIABLE = "variable"
    SUMMARY = "summary"        # RAPTOR summaries
    METADATA = "metadata"

class ProcessingStatus(Enum):
    """Status of file processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class FileInfo:
    """Complete information about a processed file"""
    # Core file properties
    path: str
    relative_path: str
    language: str
    content_hash: str
    last_modified: datetime
    file_size: int
    encoding: str
    line_count: int
    
    # Processing metadata
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Ensure path is absolute
        if not Path(self.path).is_absolute():
            self.path = str(Path(self.path).resolve())
        
        # Normalize language
        self.language = self.language.lower()
        
        # Validate encoding
        if self.encoding not in ['utf-8', 'utf-16', 'latin-1', 'ascii', 'binary']:
            self.encoding = 'utf-8'
    
    @property
    def file_name(self) -> str:
        """Get just the filename"""
        return Path(self.path).name
    
    @property
    def file_extension(self) -> str:
        """Get file extension"""
        return Path(self.path).suffix.lower()
    
    @property
    def is_text_file(self) -> bool:
        """Check if this is a text file (not binary)"""
        return self.encoding != 'binary'
    
    @property
    def size_mb(self) -> float:
        """Get file size in MB"""
        return self.file_size / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['last_modified'] = self.last_modified.isoformat()
        data['processing_status'] = self.processing_status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileInfo':
        """Create from dictionary"""
        # Handle datetime conversion
        if isinstance(data['last_modified'], str):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        
        # Handle enum conversion
        if isinstance(data['processing_status'], str):
            data['processing_status'] = ProcessingStatus(data['processing_status'])
        
        return cls(**data)
    
    def mark_completed(self, processing_time: float):
        """Mark file as successfully processed"""
        self.processing_status = ProcessingStatus.COMPLETED
        self.processing_time = processing_time
        self.error_message = None
    
    def mark_failed(self, error_message: str):
        """Mark file as failed to process"""
        self.processing_status = ProcessingStatus.FAILED
        self.error_message = error_message

@dataclass
class CodeChunk:
    """A semantic chunk of code or text"""
    # Core properties
    id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: ChunkType
    language: str
    
    # Content analysis
    content_hash: str = field(init=False)
    character_count: int = field(init=False)
    word_count: int = field(init=False)
    
    # Semantic properties
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    complexity_score: Optional[float] = None
    
    # Hierarchy and relationships
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    related_chunk_ids: List[str] = field(default_factory=list)
    
    # Embeddings and search
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate derived properties"""
        # Generate content hash
        self.content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        
        # Calculate text statistics
        self.character_count = len(self.content)
        self.word_count = len(self.content.split())
        
        # Normalize language
        self.language = self.language.lower()
        
        # Ensure chunk type is enum
        if isinstance(self.chunk_type, str):
            self.chunk_type = ChunkType(self.chunk_type)
    
    @property
    def relative_file_path(self) -> str:
        """Get relative file path if available in metadata"""
        return self.metadata.get('relative_path', self.file_path)
    
    @property
    def line_count(self) -> int:
        """Get number of lines in chunk"""
        return self.end_line - self.start_line + 1
    
    @property
    def size_category(self) -> str:
        """Categorize chunk by size"""
        if self.character_count < 100:
            return "small"
        elif self.character_count < 500:
            return "medium"
        elif self.character_count < 2000:
            return "large"
        else:
            return "very_large"
    
    @property
    def content_preview(self) -> str:
        """Get first 200 characters of content"""
        if len(self.content) <= 200:
            return self.content
        return self.content[:197] + "..."
    
    def has_embedding(self) -> bool:
        """Check if chunk has an embedding"""
        return self.embedding is not None and len(self.embedding) > 0
    
    def add_related_chunk(self, chunk_id: str):
        """Add a related chunk ID"""
        if chunk_id not in self.related_chunk_ids:
            self.related_chunk_ids.append(chunk_id)
            self.updated_at = datetime.now()
    
    def add_child_chunk(self, chunk_id: str):
        """Add a child chunk ID"""
        if chunk_id not in self.child_chunk_ids:
            self.child_chunk_ids.append(chunk_id)
            self.updated_at = datetime.now()
    
    def update_embedding(self, embedding: List[float], model: str):
        """Update embedding and model info"""
        self.embedding = embedding
        self.embedding_model = model
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['chunk_type'] = self.chunk_type.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeChunk':
        """Create from dictionary"""
        # Handle datetime conversion
        if isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Handle enum conversion
        if isinstance(data['chunk_type'], str):
            data['chunk_type'] = ChunkType(data['chunk_type'])
        
        return cls(**data)

@dataclass
class HierarchyNode:
    """Node in RAPTOR hierarchical structure"""
    # Core properties
    id: str
    level: int  # 0=leaf chunks, 1=first summary, 2=higher summary, etc.
    content: str
    summary: str
    
    # Relationships
    child_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    sibling_ids: List[str] = field(default_factory=list)
    
    # Content properties
    content_hash: str = field(init=False)
    chunk_count: int = field(init=False)  # Number of leaf chunks represented
    
    # Semantic properties
    primary_language: Optional[str] = None
    file_paths: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    
    # Embeddings
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # Clustering information
    cluster_id: Optional[str] = None
    cluster_score: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate derived properties"""
        # Generate content hash
        combined_content = f"{self.content}\n{self.summary}"
        self.content_hash = hashlib.md5(combined_content.encode('utf-8')).hexdigest()
        
        # Calculate chunk count (will be updated when building hierarchy)
        self.chunk_count = len(self.child_ids) if self.level > 0 else 1
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (level 0)"""
        return self.level == 0
    
    @property
    def is_root(self) -> bool:
        """Check if this is a root node (no parent)"""
        return self.parent_id is None
    
    @property
    def depth_from_leaves(self) -> int:
        """Get depth from leaf nodes"""
        return self.level
    
    @property
    def content_preview(self) -> str:
        """Get preview of content"""
        if len(self.content) <= 300:
            return self.content
        return self.content[:297] + "..."
    
    @property
    def summary_preview(self) -> str:
        """Get preview of summary"""
        if len(self.summary) <= 150:
            return self.summary
        return self.summary[:147] + "..."
    
    def add_child(self, child_id: str):
        """Add a child node"""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
            self.chunk_count = len(self.child_ids)
            self.updated_at = datetime.now()
    
    def add_sibling(self, sibling_id: str):
        """Add a sibling node"""
        if sibling_id not in self.sibling_ids:
            self.sibling_ids.append(sibling_id)
            self.updated_at = datetime.now()
    
    def update_embedding(self, embedding: List[float], model: str):
        """Update embedding information"""
        self.embedding = embedding
        self.embedding_model = model
        self.updated_at = datetime.now()
    
    def add_file_path(self, file_path: str):
        """Add a file path if not already present"""
        if file_path not in self.file_paths:
            self.file_paths.append(file_path)
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HierarchyNode':
        """Create from dictionary"""
        # Handle datetime conversion
        if isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)

@dataclass
class SearchResult:
    """Result from search operation"""
    # Core result data
    chunk_id: str
    content: str
    file_path: str
    chunk_type: ChunkType
    language: str
    
    # Relevance scoring
    similarity_score: float
    rank: int
    
    # Context information
    start_line: int
    end_line: int
    surrounding_context: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def relevance_category(self) -> str:
        """Categorize relevance by score"""
        if self.similarity_score >= 0.9:
            return "excellent"
        elif self.similarity_score >= 0.7:
            return "good"
        elif self.similarity_score >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['chunk_type'] = self.chunk_type.value
        return data

@dataclass 
class ProcessingStats:
    """Statistics from processing operations"""
    # Counts
    files_total: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    
    # Size information  
    total_size_bytes: int = 0
    total_lines: int = 0
    
    # Language breakdown
    languages: Dict[str, int] = field(default_factory=dict)
    
    # Chunk type breakdown
    chunk_types: Dict[str, int] = field(default_factory=dict)
    
    # Timing information
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    
    # Error information
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.files_total == 0:
            return 0.0
        return self.files_processed / self.files_total
    
    @property
    def total_size_mb(self) -> float:
        """Get total size in MB"""
        return self.total_size_bytes / (1024 * 1024)
    
    @property
    def avg_file_size_kb(self) -> float:
        """Get average file size in KB"""
        if self.files_processed == 0:
            return 0.0
        return (self.total_size_bytes / self.files_processed) / 1024
    
    def add_file_processed(self, file_info: FileInfo):
        """Add a successfully processed file"""
        self.files_processed += 1
        self.total_size_bytes += file_info.file_size
        self.total_lines += file_info.line_count
        
        # Update language counts
        self.languages[file_info.language] = self.languages.get(file_info.language, 0) + 1
    
    def add_chunk_created(self, chunk: CodeChunk):
        """Add a created chunk"""
        self.chunks_created += 1
        
        # Update chunk type counts
        chunk_type_str = chunk.chunk_type.value
        self.chunk_types[chunk_type_str] = self.chunk_types.get(chunk_type_str, 0) + 1
    
    def add_error(self, error_message: str):
        """Add an error"""
        self.errors.append(error_message)
        self.files_failed += 1
    
    def finalize(self):
        """Finalize statistics (call when processing complete)"""
        self.end_time = datetime.now()
        if self.start_time:
            self.processing_time_seconds = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data

# Utility functions for working with models

def create_chunk_id(file_path: str, start_line: int, chunk_type: ChunkType, 
                   identifier: Optional[str] = None) -> str:
    """Create a unique chunk ID"""
    # Use relative path for cleaner IDs
    file_part = Path(file_path).name
    
    if identifier:
        return f"{file_part}:{chunk_type.value}:{identifier}:{start_line}"
    else:
        return f"{file_part}:{chunk_type.value}:{start_line}"

def create_hierarchy_node_id(level: int, cluster_id: str, node_index: int = 0) -> str:
    """Create a unique hierarchy node ID"""
    return f"L{level}:C{cluster_id}:N{node_index}"

def validate_chunk(chunk: CodeChunk) -> List[str]:
    """Validate a chunk and return list of issues"""
    issues = []
    
    if not chunk.content.strip():
        issues.append("Chunk content is empty")
    
    if chunk.start_line <= 0:
        issues.append("Start line must be positive")
    
    if chunk.end_line < chunk.start_line:
        issues.append("End line must be >= start line")
    
    if not chunk.file_path:
        issues.append("File path is required")
    
    if chunk.embedding and len(chunk.embedding) == 0:
        issues.append("Embedding is empty list")
    
    return issues

def validate_hierarchy_node(node: HierarchyNode) -> List[str]:
    """Validate a hierarchy node and return list of issues"""
    issues = []
    
    if node.level < 0:
        issues.append("Level must be non-negative")
    
    if not node.content.strip() and not node.summary.strip():
        issues.append("Node must have either content or summary")
    
    if node.level > 0 and not node.child_ids:
        issues.append("Non-leaf nodes must have children")
    
    if node.level == 0 and node.child_ids:
        issues.append("Leaf nodes should not have children")
    
    return issues

# Example usage and testing
if __name__ == "__main__":
    # Demo the models
    print("=== CodeRaptor Models Demo ===")
    
    # Create a FileInfo
    file_info = FileInfo(
        path="/home/user/project/main.py",
        relative_path="main.py",
        language="python",
        content_hash="abc123def456",
        last_modified=datetime.now(),
        file_size=2048,
        encoding="utf-8",
        line_count=85
    )
    
    print(f"FileInfo: {file_info.file_name} ({file_info.language})")
    print(f"  Size: {file_info.size_mb:.2f} MB")
    print(f"  Text file: {file_info.is_text_file}")
    
    # Create a CodeChunk
    chunk = CodeChunk(
        id=create_chunk_id(file_info.path, 10, ChunkType.FUNCTION, "main"),
        file_path=file_info.path,
        content="def main():\n    print('Hello, world!')\n    return 0",
        start_line=10,
        end_line=12,
        chunk_type=ChunkType.FUNCTION,
        language=file_info.language,
        summary="Main entry point function"
    )
    
    print(f"\nCodeChunk: {chunk.id}")
    print(f"  Type: {chunk.chunk_type.value}")
    print(f"  Size: {chunk.size_category}")
    print(f"  Lines: {chunk.line_count}")
    print(f"  Preview: {chunk.content_preview}")
    
    # Validate
    issues = validate_chunk(chunk)
    if issues:
        print(f"  Issues: {issues}")
    else:
        print("  ✅ Valid chunk")
    
    # Create HierarchyNode
    node = HierarchyNode(
        id=create_hierarchy_node_id(1, "python_functions", 0),
        level=1,
        content="Collection of Python functions from main.py",
        summary="Main entry points and utility functions",
        child_ids=[chunk.id],
        primary_language="python",
        file_paths=[file_info.path]
    )
    
    print(f"\nHierarchyNode: {node.id}")
    print(f"  Level: {node.level}")
    print(f"  Children: {len(node.child_ids)}")
    print(f"  Leaf: {node.is_leaf}")
    print(f"  Root: {node.is_root}")
    
    # Test serialization
    chunk_dict = chunk.to_dict()
    chunk_restored = CodeChunk.from_dict(chunk_dict)
    print(f"\n✅ Serialization test passed: {chunk.id == chunk_restored.id}")