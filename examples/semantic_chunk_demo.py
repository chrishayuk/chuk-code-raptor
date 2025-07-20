#!/usr/bin/env python3
"""
SemanticChunk Feature Demo
=========================

Focused demonstration of SemanticChunk's enhanced features:
- Content fingerprinting and change detection
- Graph-ready relationships
- Embedding support and versioning
- Quality scoring framework
- Semantic tagging with confidence
- Pattern detection
"""

import json
from datetime import datetime
from typing import List, Dict, Any

# Import your actual SemanticChunk classes
from chuk_code_raptor.chunking.semantic_chunk import (
    SemanticChunk, ChunkRelationship, SemanticTag, CodePattern,
    QualityMetric, ContentType, ChunkComplexity, create_chunk_id
)
from chuk_code_raptor.core.models import ChunkType

# Mock sample content for demonstration
SAMPLE_CONTENT = """
class DatabaseManager:
    def __init__(self, connection_string):
        self.connection = connect(connection_string)
        self.cache = {}
    
    def execute_query(self, query, params=None):
        cache_key = self._get_cache_key(query, params)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.connection.execute(query, params)
        self.cache[cache_key] = result
        return result
"""

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title: str):
    """Print a section header"""
    print(f"\n--- {title} ---")

def demo_basic_chunk_creation():
    """Demo basic SemanticChunk creation and properties"""
    print_header("SEMANTIC CHUNK CREATION & BASIC PROPERTIES")
    
    # Create a semantic chunk
    chunk = SemanticChunk(
        id="db_manager:class:DatabaseManager:1",
        file_path="src/database/manager.py",
        content=SAMPLE_CONTENT.strip(),
        start_line=1,
        end_line=15,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    print(f"Chunk ID: {chunk.id}")
    print(f"Content Type: {chunk.content_type}")
    print(f"Chunk Type: {chunk.chunk_type}")
    print(f"Complexity: {chunk.complexity_level}")
    print(f"Character Count: {chunk.character_count}")
    print(f"Line Count: {chunk.line_count}")
    print(f"Word Count: {chunk.word_count}")
    print(f"Content Hash: {chunk.content_hash}")
    print(f"Version: {chunk.version}")
    
    return chunk

def demo_fingerprinting_and_change_detection(chunk):
    """Demo content fingerprinting and change detection"""
    print_header("FINGERPRINTING & CHANGE DETECTION")
    
    print("Original Fingerprints:")
    print(f"  Content: {chunk.content_fingerprint[:16]}...")
    print(f"  Dependency: {chunk.dependency_fingerprint[:16]}...")
    print(f"  Combined: {chunk.combined_fingerprint[:16]}...")
    
    # Store original fingerprints
    original_content_fp = chunk.content_fingerprint
    original_combined_fp = chunk.combined_fingerprint
    
    print_section("Simulating Content Change")
    
    # Modify content
    new_content = chunk.content + "\n    # Added comment"
    old_content = chunk.content
    chunk.content = new_content
    chunk.update_fingerprints()
    
    print(f"Content changed: {chunk.has_content_changed(original_content_fp)}")
    print(f"Overall changed: {chunk.has_changed(original_combined_fp)}")
    print(f"New version: {chunk.version}")
    print(f"Last modified: {chunk.last_modified}")
    
    # Restore original content for other demos
    chunk.content = old_content
    chunk.update_fingerprints()

def demo_semantic_tagging(chunk):
    """Demo semantic tagging with confidence and sources"""
    print_header("SEMANTIC TAGGING")
    
    # Add various semantic tags
    chunk.add_semantic_tag("database", confidence=0.95, source="ast", category="domain")
    chunk.add_semantic_tag("caching", confidence=0.88, source="pattern", category="architecture")
    chunk.add_semantic_tag("manager-pattern", confidence=0.82, source="analysis", category="pattern")
    chunk.add_semantic_tag("sql", confidence=0.75, source="nlp", category="domain")
    chunk.add_semantic_tag("well-structured", confidence=0.90, source="manual", category="quality")
    chunk.add_semantic_tag("performance-optimized", confidence=0.70, source="analysis", category="quality")
    
    print(f"Total tags: {len(chunk.semantic_tags)}")
    print(f"All tag names: {chunk.tag_names}")
    print(f"High confidence tags: {chunk.high_confidence_tags}")
    
    print_section("Tag Details")
    for tag in chunk.semantic_tags:
        print(f"  {tag.name}: {tag.confidence:.2f} (source: {tag.source}, category: {tag.category})")

def demo_quality_scoring(chunk):
    """Demo quality scoring framework"""
    print_header("QUALITY SCORING FRAMEWORK")
    
    # Set various quality scores
    chunk.set_quality_score(QualityMetric.READABILITY, 0.85)
    chunk.set_quality_score(QualityMetric.MAINTAINABILITY, 0.78)
    chunk.set_quality_score(QualityMetric.SEMANTIC_COHERENCE, 0.92)
    chunk.set_quality_score(QualityMetric.COMPLETENESS, 0.88)
    chunk.set_quality_score(QualityMetric.DOCUMENTATION_QUALITY, 0.65)
    chunk.set_quality_score(QualityMetric.COMPLEXITY, 0.72)
    
    print("Individual Quality Scores:")
    for metric_name, score in chunk.quality_scores.items():
        print(f"  {metric_name}: {score:.2f}")
    
    overall_score = chunk.calculate_overall_quality_score()
    print(f"\nOverall Quality Score: {overall_score:.3f}")
    print(f"Importance Score: {chunk.importance_score}")

def demo_relationships_and_graph_integration(chunk):
    """Demo enhanced relationships and graph integration"""
    print_header("RELATIONSHIPS & GRAPH INTEGRATION")
    
    # Add various types of relationships
    chunk.add_relationship(
        target_chunk_id="connection:class:Connection:1",
        relationship_type="depends_on",
        strength=0.9,
        context="uses for database connectivity",
        line_number=3
    )
    
    chunk.add_relationship(
        target_chunk_id="cache:module:cache_utils:1", 
        relationship_type="imports",
        strength=0.7,
        context="caching functionality"
    )
    
    chunk.add_relationship(
        target_chunk_id="query:function:execute_sql:1",
        relationship_type="calls",
        strength=0.85,
        context="delegated query execution",
        line_number=12
    )
    
    print(f"Total relationships: {len(chunk.relationships)}")
    print(f"Dependencies: {chunk.dependencies}")
    print(f"Graph node ID: {chunk.graph_node_id}")
    
    print_section("Relationship Details")
    for rel in chunk.relationships:
        print(f"  {rel.relationship_type}: {rel.target_chunk_id}")
        print(f"    Strength: {rel.strength}, Context: {rel.context}")
        if rel.line_number:
            print(f"    Line: {rel.line_number}")
    
    print_section("Graph Edges")
    edges = chunk.get_graph_edges()
    for edge in edges:
        print(f"  {edge['source']} --[{edge['type']}]--> {edge['target']} (weight: {edge['weight']})")

def demo_embedding_support(chunk):
    """Demo embedding support and versioning"""
    print_header("EMBEDDING SUPPORT")
    
    print(f"Has embedding: {chunk.has_semantic_embedding}")
    print(f"Needs embedding update: {chunk.needs_embedding_update('text-embedding-ada-002', 2)}")
    
    # Set embedding
    mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 50  # 250-dim mock embedding
    chunk.set_embedding(mock_embedding, "text-embedding-ada-002", 2)
    
    print(f"Embedding set: {chunk.has_semantic_embedding}")
    print(f"Embedding model: {chunk.embedding_model}")
    print(f"Embedding version: {chunk.embedding_version}")
    print(f"Embedding dimensions: {len(chunk.semantic_embedding)}")
    
    print_section("Embedding Text Generation")
    embedding_text = chunk.get_embedding_text(include_context=True)
    print("Text for embedding generation:")
    print(embedding_text[:300] + "..." if len(embedding_text) > 300 else embedding_text)

def demo_pattern_detection(chunk):
    """Demo code pattern detection"""
    print_header("PATTERN DETECTION")
    
    # Add detected patterns
    patterns = [
        CodePattern(
            pattern_name="Manager Pattern",
            confidence=0.85,
            evidence=["class name contains 'Manager'", "manages database connections", "encapsulates operations"],
            category="design_pattern"
        ),
        CodePattern(
            pattern_name="Caching Pattern", 
            confidence=0.78,
            evidence=["cache dictionary", "cache key generation", "cache lookup"],
            category="architectural_pattern"
        ),
        CodePattern(
            pattern_name="Facade Pattern",
            confidence=0.65,
            evidence=["simplified interface", "delegates to connection"],
            category="design_pattern"
        )
    ]
    
    chunk.detected_patterns = patterns
    
    print(f"Detected patterns: {len(chunk.detected_patterns)}")
    for pattern in chunk.detected_patterns:
        print(f"\n  {pattern.pattern_name} ({pattern.category})")
        print(f"    Confidence: {pattern.confidence:.2f}")
        print(f"    Evidence: {', '.join(pattern.evidence)}")

def demo_comprehensive_summary(chunk):
    """Demo comprehensive summary information"""
    print_header("COMPREHENSIVE SUMMARY")
    
    summary = chunk.get_summary_info()
    
    print("Complete chunk summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    print_section("Key Metrics")
    print(f"Overall Quality: {chunk.calculate_overall_quality_score():.3f}")
    print(f"Complexity Level: {chunk.complexity_level.value}")
    print(f"Is Complex: {chunk.is_complex}")
    print(f"Content Preview: {chunk.content_preview}")

def demo_serialization(chunk):
    """Demo serialization capabilities"""
    print_header("SERIALIZATION & PERSISTENCE")
    
    # This would be the actual serialization in your implementation
    print("Serialization capabilities:")
    print(f"  Content hash for indexing: {chunk.content_hash}")
    print(f"  Fingerprints for change detection: {len([chunk.content_fingerprint, chunk.dependency_fingerprint, chunk.combined_fingerprint])} fingerprints")
    print(f"  Graph node ID: {chunk.graph_node_id}")
    print(f"  Version tracking: v{chunk.version}")
    print(f"  Timestamp tracking: created {chunk.created_at}, updated {chunk.updated_at}")

def main():
    """Run the complete SemanticChunk demo"""
    print_header("SEMANTIC CHUNK ENHANCED FEATURES DEMO")
    print("Demonstrating the future-ready foundations of SemanticChunk")
    
    # Create and demo basic chunk
    chunk = demo_basic_chunk_creation()
    
    # Demo all enhanced features
    demo_fingerprinting_and_change_detection(chunk)
    demo_semantic_tagging(chunk)
    demo_quality_scoring(chunk)
    demo_relationships_and_graph_integration(chunk)
    demo_embedding_support(chunk)
    demo_pattern_detection(chunk)
    demo_comprehensive_summary(chunk)
    demo_serialization(chunk)
    
    print_header("DEMO COMPLETE")
    print("SemanticChunk showcases:")
    print("✅ Content-addressable fingerprints for Merkle tree support")
    print("✅ Graph-ready relationships for property graphs") 
    print("✅ Embedding support with versioning")
    print("✅ Quality scoring framework")
    print("✅ Enhanced semantic tagging")
    print("✅ Pattern detection capabilities")
    print("✅ Change tracking for incremental updates")
    print("✅ Future-ready architecture")

if __name__ == "__main__":
    main()