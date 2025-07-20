# src/chuk_code_raptor/chunking/code_chunk.py - Fixed Version
"""
Code-Specific Chunk Model - Fixed Version
==========================================

Specialized semantic chunk for code content. Language-agnostic design
that relies on tree-sitter AST analysis rather than regex patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from .semantic_chunk import SemanticChunk, ContentType, QualityMetric
from chuk_code_raptor.core.models import CodeChunk

class ArchitecturalRole(Enum):
    """Architectural roles for code chunks"""
    DATA_ACCESS = "data_access"
    BUSINESS_LOGIC = "business_logic"
    PRESENTATION = "presentation"
    CREATIONAL = "creational"
    CONFIGURATION = "configuration"
    TESTING = "testing"
    UTILITY = "utility"
    INFRASTRUCTURE = "infrastructure"

@dataclass
class SemanticCodeChunk(SemanticChunk):
    """
    Semantic chunk specifically for code content.
    Language-agnostic design that relies on AST analysis from parsers.
    """
    
    # === Code-Specific Properties ===
    accessibility: str = "public"  # public/private/protected
    architectural_role: Optional[ArchitecturalRole] = None
    
    # === Code Analysis (populated by language-specific parsers) ===
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    variables_used: List[str] = field(default_factory=list)
    types_used: List[str] = field(default_factory=list)
    ast_node_path: Optional[str] = None
    
    # === Code Quality Indicators (set by parsers) ===
    has_docstring: bool = False
    docstring_quality: float = 0.0
    has_type_hints: bool = False
    type_coverage: float = 0.0
    has_error_handling: bool = False
    cyclomatic_complexity: int = 0
    
    # === Quality Metrics ===
    maintainability_index: float = 0.0
    test_coverage_indicator: float = 0.0
    
    # === Pattern and Architecture Analysis ===
    design_patterns: List[str] = field(default_factory=list)
    code_smells: List[str] = field(default_factory=list)
    architectural_concerns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize code chunk"""
        # Set default content type for code if not specified
        if not hasattr(self, '_content_type_explicitly_set'):
            self.content_type = ContentType.CODE
        
        super().__post_init__()
        
        # Calculate initial quality metrics
        self._calculate_quality_metrics()
    
    def set_code_quality_indicators(self, 
                                   has_docstring: bool = False,
                                   docstring_quality: float = 0.0,
                                   has_type_hints: bool = False,
                                   type_coverage: float = 0.0,
                                   has_error_handling: bool = False,
                                   cyclomatic_complexity: int = 0):
        """Set code quality indicators (called by language-specific parsers)"""
        self.has_docstring = has_docstring
        # Clamp docstring_quality to valid range
        self.docstring_quality = max(0.0, min(1.0, docstring_quality))
        self.has_type_hints = has_type_hints
        # Clamp type_coverage to valid range
        self.type_coverage = max(0.0, min(1.0, type_coverage))
        self.has_error_handling = has_error_handling
        self.cyclomatic_complexity = max(0, cyclomatic_complexity)
        
        # Recalculate derived metrics
        self._calculate_quality_metrics()
    
    def _calculate_quality_metrics(self):
        """Calculate derived quality metrics"""
        self.maintainability_index = self._calculate_maintainability()
        self.test_coverage_indicator = self._calculate_test_coverage_indicator()
    
    def _calculate_maintainability(self) -> float:
        """Calculate maintainability index"""
        base_score = 1.0
        
        # Penalty for high complexity
        if self.cyclomatic_complexity > 10:
            base_score -= 0.3
        elif self.cyclomatic_complexity > 5:
            base_score -= 0.1
        
        # Bonus for documentation
        base_score += self.docstring_quality * 0.2
        
        # Bonus for type hints
        base_score += self.type_coverage * 0.1
        
        # Penalty for very long chunks
        if self.line_count > 100:
            base_score -= 0.2
        elif self.line_count > 50:
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_test_coverage_indicator(self) -> float:
        """Calculate test coverage indicator based on patterns"""
        indicators = 0
        total_checks = 3
        
        # Check for error handling
        if self.has_error_handling:
            indicators += 1
        
        # Check for test-related semantic tags
        test_tags = [tag.name for tag in self.semantic_tags if 'test' in tag.name.lower()]
        if test_tags:
            indicators += 1
        
        # Check for validation patterns in dependencies
        validation_deps = [dep for dep in self.dependencies if any(word in dep.lower() 
                          for word in ['validate', 'verify', 'check', 'assert'])]
        if validation_deps:
            indicators += 1
        
        return indicators / total_checks
    
    def calculate_coupling_score(self) -> float:
        """Calculate coupling score based on dependencies"""
        if not self.dependencies and not self.dependents:
            return 0.0
        
        # Simple coupling calculation based on dependency count
        total_connections = len(self.dependencies) + len(self.dependents)
        return min(total_connections / 20.0, 1.0)
    
    def calculate_reusability_score(self) -> float:
        """Calculate reusability based on various factors"""
        coupling = self.calculate_coupling_score()
        completeness = self.get_quality_score(QualityMetric.COMPLETENESS)
        documentation = self.docstring_quality
        type_safety = self.type_coverage
        
        # Combined reusability score
        reusability = (completeness * 0.3 + 
                      documentation * 0.3 + 
                      type_safety * 0.2 + 
                      (1.0 - coupling) * 0.2)
        
        return max(0.0, reusability)
    
    @property
    def is_highly_coupled(self) -> bool:
        """Check if chunk is highly coupled"""
        return self.calculate_coupling_score() > 0.7
    
    @property
    def is_reusable(self) -> bool:
        """Check if chunk is reusable"""
        return self.calculate_reusability_score() > 0.6
    
    @property
    def is_well_tested(self) -> bool:
        """Check if chunk appears to be well tested"""
        return self.test_coverage_indicator > 0.5
    
    @property
    def is_high_quality(self) -> bool:
        """Check if chunk meets high quality standards"""
        quality_checks = [
            self.importance_score > 0.7,
            self.docstring_quality > 0.6,
            self.cyclomatic_complexity <= 10,
            len(self.code_smells) == 0
        ]
        
        return sum(quality_checks) >= 3
    
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

# Quality analysis functions

def calculate_code_quality_metrics(chunk: SemanticCodeChunk) -> Dict[str, float]:
    """Calculate comprehensive quality metrics for code chunks"""
    metrics = {}
    
    # Core metrics
    metrics['coupling'] = chunk.calculate_coupling_score()
    metrics['reusability'] = chunk.calculate_reusability_score()
    metrics['maintainability'] = chunk.maintainability_index
    metrics['test_coverage_indicator'] = chunk.test_coverage_indicator
    
    # Documentation quality (already clamped)
    metrics['documentation_quality'] = chunk.docstring_quality
    
    # Type safety (already clamped)
    metrics['type_safety'] = chunk.type_coverage
    
    # Complexity (inverted - lower complexity is better)
    if chunk.cyclomatic_complexity <= 5:
        metrics['complexity'] = 1.0
    elif chunk.cyclomatic_complexity <= 10:
        metrics['complexity'] = 0.7
    elif chunk.cyclomatic_complexity <= 15:
        metrics['complexity'] = 0.5
    else:
        metrics['complexity'] = 0.3
    
    # Error handling
    metrics['error_handling'] = 1.0 if chunk.has_error_handling else 0.0
    
    return metrics

def create_code_chunk_for_content_type(content_type: ContentType, **kwargs) -> SemanticCodeChunk:
    """Factory function to create code chunk"""
    kwargs['content_type'] = content_type
    return SemanticCodeChunk(**kwargs)