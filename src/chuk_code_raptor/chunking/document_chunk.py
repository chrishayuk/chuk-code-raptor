# src/chuk_code_raptor/chunking/document_chunk.py
"""
Document-Specific Chunk Model - Fixed Version
==============================================

Specialized semantic chunk for document content with document-specific analysis,
readability metrics, and structure assessment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import re

from .semantic_chunk import SemanticChunk, ContentType, QualityMetric

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
    
    # === Readability Metrics ===
    readability_score: float = 0.0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    
    def __post_init__(self):
        """Initialize document chunk"""
        # Set default content type for documents if not explicitly CODE
        if not hasattr(self, '_content_type_set'):
            if self.content_type != ContentType.CODE:
                self.content_type = ContentType.DOCUMENTATION
        
        super().__post_init__()
        
        # Enhanced document analysis
        self._analyze_readability()
        self._analyze_structure()
    
    def _analyze_readability(self):
        """Analyze document readability"""
        # Count sentences more accurately
        sentence_endings = self.content.count('.') + self.content.count('!') + self.content.count('?')
        self.sentence_count = max(sentence_endings, 1)
        
        # Calculate average sentence length
        if self.sentence_count > 0:
            self.avg_sentence_length = self.word_count / self.sentence_count
        else:
            self.avg_sentence_length = 0.0
        
        # Calculate readability score (simplified)
        if self.sentence_count > 0 and self.word_count > 0:
            avg_sentence_length = self.avg_sentence_length
            
            # Simplified readability calculation
            if avg_sentence_length <= 15:
                self.readability_score = 0.9  # Very readable
            elif avg_sentence_length <= 25:
                self.readability_score = 0.7  # Moderately readable
            elif avg_sentence_length <= 35:
                self.readability_score = 0.5  # Complex
            else:
                self.readability_score = 0.3  # Very complex
        else:
            self.readability_score = 0.0
    
    def _analyze_structure(self):
        """Analyze document structure"""
        content_clean = self.content.strip()
        
        # Detect section type if not explicitly set to something other than "paragraph"
        if self.section_type == "paragraph":
            # Check for headings
            if content_clean.startswith('#'):
                self.section_type = "heading"
                # Count the number of # symbols to determine level
                heading_match = re.match(r'^(#{1,6})\s+', content_clean)
                if heading_match:
                    self.heading_level = len(heading_match.group(1))
                else:
                    self.heading_level = 1
            
            # Check for code blocks (use original content for indentation check)
            elif (content_clean.startswith('```') or 
                  content_clean.endswith('```') or
                  '```' in content_clean):
                self.section_type = "code_block"
            
            # Check for indented code blocks (use original content, not stripped)
            elif self._is_indented_code_block(self.content):
                self.section_type = "code_block"
            
            # Check for lists
            elif (content_clean.startswith(('-', '*', '+')) or 
                  re.match(r'^\d+\.', content_clean) or
                  any(line.strip().startswith(('-', '*', '+')) or re.match(r'^\d+\.', line.strip()) 
                      for line in content_clean.split('\n') if line.strip())):
                self.section_type = "list"
            
            # Check for tables
            elif ('|' in content_clean and content_clean.count('|') >= 3 and
                  any('|' in line for line in content_clean.split('\n'))):
                self.section_type = "table"
        
        # Structural consistency score
        self.document_structure['consistency'] = self.calculate_structure_consistency()
    
    def _is_indented_code_block(self, content: str) -> bool:
        """Check if content is an indented code block"""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return False
        
        # Check if all non-empty lines start with at least 4 spaces
        return all(line.startswith('    ') for line in non_empty_lines)
    
    def calculate_readability_score(self) -> float:
        """Calculate readability score based on measurable text properties"""
        return self.readability_score
    
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
            if ('```' in self.content or 
                any(keyword in self.content.lower() for keyword in ['def ', 'function', 'class ', 'import', 'return']) or
                any(line.startswith('    ') for line in self.content.split('\n'))):
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
    
    @property
    def is_highly_readable(self) -> bool:
        """Check if document is highly readable"""
        return self.readability_score > 0.7

# Quality analysis functions

def calculate_document_quality_metrics(chunk: SemanticDocumentChunk) -> Dict[str, float]:
    """Calculate comprehensive quality metrics for document chunks"""
    metrics = {}
    
    # Readability
    metrics['readability'] = chunk.readability_score
    
    # Structure consistency
    metrics['structure_consistency'] = chunk.document_structure.get('consistency', 0.0)
    
    # Information density
    if chunk.keywords:
        keyword_density = len(chunk.keywords) / max(chunk.word_count, 1)
        metrics['keyword_density'] = min(keyword_density, 1.0)
    else:
        metrics['keyword_density'] = 0.0
    
    # Cross-reference connectivity
    if chunk.cross_references:
        metrics['connectivity'] = min(len(chunk.cross_references) / 10.0, 1.0)
    else:
        metrics['connectivity'] = 0.0
    
    # Entity density
    if chunk.entities:
        entity_density = len(chunk.entities) / max(chunk.word_count, 1)
        metrics['entity_density'] = min(entity_density, 1.0)
    else:
        metrics['entity_density'] = 0.0
    
    # Content completeness (based on sentence structure)
    if chunk.sentence_count > 0:
        # More realistic completeness calculation
        base_completeness = min(chunk.sentence_count / 5.0, 1.0)
        # Bonus for reasonable sentence length
        if 5 <= chunk.avg_sentence_length <= 30:
            base_completeness = min(base_completeness + 0.2, 1.0)
        metrics['completeness'] = base_completeness
    else:
        metrics['completeness'] = 0.0
    
    return metrics

def create_document_chunk_for_content_type(content_type: ContentType, **kwargs) -> SemanticDocumentChunk:
    """Factory function to create document chunk"""
    kwargs['content_type'] = content_type
    return SemanticDocumentChunk(**kwargs)