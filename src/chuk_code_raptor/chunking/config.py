# src/chuk_code_raptor/chunking/config.py
"""
Chunking Configuration
======================

Configuration classes and enums for the semantic chunking system.
Clean, focused configuration for modern content understanding.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any

class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    STRUCTURAL = "structural"       # AST/structure-based chunking (primary)
    SEMANTIC = "semantic"           # Neural semantic boundary detection  
    HYBRID = "hybrid"               # Structure + neural combination

@dataclass
class ChunkingConfig:
    """Configuration for semantic chunking behavior"""
    
    # Size constraints
    min_chunk_size: int = 50          # Minimum characters
    max_chunk_size: int = 2000        # Maximum characters  
    target_chunk_size: int = 800      # Preferred size
    
    # Structural chunking behavior
    preserve_atomic_nodes: bool = True      # Never split atomic nodes (functions, classes)
    include_decorators: bool = True         # Include decorators with functions/classes
    group_imports: bool = True              # Group consecutive imports
    detect_main_blocks: bool = True         # Find main execution blocks
    include_docstrings: bool = True         # Include docstrings with definitions
    
    # Semantic analysis
    enable_dependency_tracking: bool = True     # Track dependencies between chunks
    enable_similarity_analysis: bool = True     # Calculate chunk similarities
    enable_contextual_linking: bool = True      # Link related chunks
    
    # Quality settings
    respect_boundaries: bool = True             # Respect structural boundaries
    min_importance_threshold: float = 0.3       # Minimum importance to chunk
    quality_threshold: float = 0.5              # Minimum quality score
    
    # Strategy preferences  
    primary_strategy: ChunkingStrategy = ChunkingStrategy.STRUCTURAL
    enable_neural_enhancement: bool = False     # Add neural boundary detection
    
    # Content type specific settings
    markdown_section_chunking: bool = True      # Chunk markdown by sections
    html_semantic_chunking: bool = True         # Use semantic HTML elements
    code_function_chunking: bool = True         # Chunk code by functions/classes
    
    # Language-specific settings
    language_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_language_setting(self, language: str, setting: str, default: Any = None) -> Any:
        """Get a language-specific setting"""
        return self.language_settings.get(language, {}).get(setting, default)
    
    def set_language_setting(self, language: str, setting: str, value: Any):
        """Set a language-specific setting"""
        if language not in self.language_settings:
            self.language_settings[language] = {}
        self.language_settings[language][setting] = value

# Predefined configurations for different use cases

DEFAULT_CONFIG = ChunkingConfig()

FAST_CONFIG = ChunkingConfig(
    target_chunk_size=1200,
    preserve_atomic_nodes=True,
    group_imports=False,
    include_decorators=False,
    enable_dependency_tracking=False,
    enable_similarity_analysis=False,
    primary_strategy=ChunkingStrategy.STRUCTURAL
)

PRECISE_CONFIG = ChunkingConfig(
    target_chunk_size=600,
    min_chunk_size=100,
    preserve_atomic_nodes=True,
    include_decorators=True,
    group_imports=True,
    detect_main_blocks=True,
    include_docstrings=True,
    enable_dependency_tracking=True,
    enable_similarity_analysis=True,
    enable_contextual_linking=True,
    min_importance_threshold=0.5,
    quality_threshold=0.6,
    primary_strategy=ChunkingStrategy.STRUCTURAL
)

SEMANTIC_CONFIG = ChunkingConfig(
    target_chunk_size=800,
    preserve_atomic_nodes=True,
    include_decorators=True,
    group_imports=True,
    enable_dependency_tracking=True,
    enable_similarity_analysis=True,
    enable_contextual_linking=True,
    primary_strategy=ChunkingStrategy.SEMANTIC,
    enable_neural_enhancement=True
)

HYBRID_CONFIG = ChunkingConfig(
    target_chunk_size=800,
    preserve_atomic_nodes=True,
    include_decorators=True,
    group_imports=True,
    enable_dependency_tracking=True,
    enable_similarity_analysis=True,
    enable_contextual_linking=True,
    primary_strategy=ChunkingStrategy.HYBRID,
    enable_neural_enhancement=True
)

LARGE_FILES_CONFIG = ChunkingConfig(
    target_chunk_size=1500,
    max_chunk_size=3000,
    preserve_atomic_nodes=True,
    min_importance_threshold=0.4,
    enable_dependency_tracking=True,
    enable_similarity_analysis=False,  # Skip for performance
    enable_contextual_linking=False
)

DOCUMENT_CONFIG = ChunkingConfig(
    target_chunk_size=1000,
    min_chunk_size=100,
    preserve_atomic_nodes=False,  # Documents can be split more freely
    markdown_section_chunking=True,
    html_semantic_chunking=True,
    enable_dependency_tracking=False,  # Less relevant for documents
    enable_similarity_analysis=True,
    enable_contextual_linking=True,
    primary_strategy=ChunkingStrategy.STRUCTURAL
)