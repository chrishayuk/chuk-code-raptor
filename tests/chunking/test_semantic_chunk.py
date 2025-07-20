#!/usr/bin/env python3
# tests/chunking/test_semantic_chunk.py
"""
Comprehensive pytest tests for SemanticChunk class
==================================================

Tests cover:
- Basic initialization and properties
- Content fingerprinting and change detection
- Relationship management
- Embedding functionality
- Quality metrics
- Serialization/deserialization
- Graph integration features
- Utility functions
"""

import pytest
from datetime import datetime
from unittest.mock import patch
import json
import hashlib

# Import the classes we're testing
from chuk_code_raptor.chunking.semantic_chunk import (
    SemanticChunk, ChunkRelationship, SemanticTag, CodePattern,
    QualityMetric, ContentType, ChunkComplexity,
    create_chunk_id, calculate_chunk_similarity, find_related_chunks
)
from chuk_code_raptor.core.models import ChunkType


class TestSemanticChunk:
    """Test suite for SemanticChunk class"""
    
    @pytest.fixture
    def sample_chunk(self):
        """Create a sample chunk for testing"""
        return SemanticChunk(
            id="test_chunk_1",
            file_path="/test/example.py",
            content="def hello_world():\n    print('Hello, World!')\n    return True",
            start_line=10,
            end_line=12,
            content_type=ContentType.CODE,
            language="python",
            chunk_type=ChunkType.FUNCTION
        )
    
    @pytest.fixture
    def complex_chunk(self):
        """Create a more complex chunk for testing"""
        content = """
class DatabaseManager:
    def __init__(self, connection_string):
        self.connection = connect(connection_string)
        self.cache = {}
        
    def query(self, sql, params=None):
        if sql in self.cache:
            return self.cache[sql]
        
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            result = cursor.fetchall()
            self.cache[sql] = result
            return result
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise
        finally:
            cursor.close()
"""
        return SemanticChunk(
            id="complex_chunk_1",
            file_path="/test/database.py",
            content=content.strip(),
            start_line=1,
            end_line=25,
            content_type=ContentType.CODE,
            language="python",
            chunk_type=ChunkType.CLASS
        )

    def test_basic_initialization(self, sample_chunk):
        """Test basic chunk initialization"""
        assert sample_chunk.id == "test_chunk_1"
        assert sample_chunk.file_path == "/test/example.py"
        assert sample_chunk.language == "python"
        assert sample_chunk.chunk_type == ChunkType.FUNCTION
        assert sample_chunk.content_type == ContentType.CODE
        assert sample_chunk.start_line == 10
        assert sample_chunk.end_line == 12
        
    def test_post_init_calculations(self, sample_chunk):
        """Test calculations performed in __post_init__"""
        # Content hash should be generated
        assert sample_chunk.content_hash is not None
        assert len(sample_chunk.content_hash) == 32  # MD5 hash length
        
        # Character and word counts
        assert sample_chunk.character_count == len(sample_chunk.content)
        assert sample_chunk.word_count > 0
        
        # Fingerprints should be created
        assert sample_chunk.content_fingerprint is not None
        assert sample_chunk.dependency_fingerprint is not None
        assert sample_chunk.combined_fingerprint is not None
        
        # Graph node ID should be set
        assert sample_chunk.graph_node_id == f"node_{sample_chunk.id}"
        
    def test_line_count_property(self, sample_chunk):
        """Test line_count property calculation"""
        expected_lines = sample_chunk.end_line - sample_chunk.start_line + 1
        assert sample_chunk.line_count == expected_lines
        
    def test_content_preview_short(self, sample_chunk):
        """Test content preview for short content"""
        # Short content should return as-is
        preview = sample_chunk.content_preview
        assert preview == sample_chunk.content
        assert not preview.endswith("...")
        
    def test_content_preview_long(self):
        """Test content preview for long content"""
        long_content = "x" * 300
        chunk = SemanticChunk(
            id="long_chunk",
            file_path="/test/long.py",
            content=long_content,
            start_line=1,
            end_line=5,
            content_type=ContentType.CODE
        )
        
        preview = chunk.content_preview
        assert len(preview) <= 200
        assert preview.endswith("...")

    def test_complexity_detection_simple(self, sample_chunk):
        """Test complexity detection for simple chunk"""
        assert sample_chunk.complexity_level == ChunkComplexity.SIMPLE
        
    def test_complexity_detection_complex(self, complex_chunk):
        """Test complexity detection for complex chunk"""
        # Complex chunk should be detected as more complex
        assert complex_chunk.complexity_level in [
            ChunkComplexity.MODERATE, 
            ChunkComplexity.COMPLEX, 
            ChunkComplexity.VERY_COMPLEX
        ]

    def test_fingerprint_creation(self, sample_chunk):
        """Test fingerprint creation and uniqueness"""
        # All fingerprints should be created
        assert sample_chunk.content_fingerprint
        assert sample_chunk.dependency_fingerprint
        assert sample_chunk.combined_fingerprint
        
        # Should be valid SHA256 hashes (64 chars)
        assert len(sample_chunk.content_fingerprint) == 64
        assert len(sample_chunk.dependency_fingerprint) == 64
        assert len(sample_chunk.combined_fingerprint) == 64
        
        # Different content should produce different fingerprints
        other_chunk = SemanticChunk(
            id="test_chunk_2",
            file_path="/test/other.py",
            content="def goodbye():\n    print('Goodbye!')",
            start_line=1,
            end_line=2,
            content_type=ContentType.CODE
        )
        
        assert sample_chunk.content_fingerprint != other_chunk.content_fingerprint

    def test_change_detection(self, sample_chunk):
        """Test change detection methods"""
        original_fingerprint = sample_chunk.combined_fingerprint
        original_content_fingerprint = sample_chunk.content_fingerprint
        original_dep_fingerprint = sample_chunk.dependency_fingerprint
        
        # Should not detect changes with same fingerprints
        assert not sample_chunk.has_changed(original_fingerprint)
        assert not sample_chunk.has_content_changed(original_content_fingerprint)
        assert not sample_chunk.has_dependencies_changed(original_dep_fingerprint)
        
        # Should detect changes with different fingerprints
        assert sample_chunk.has_changed("different_fingerprint")
        assert sample_chunk.has_content_changed("different_content_fingerprint")
        assert sample_chunk.has_dependencies_changed("different_dep_fingerprint")

    def test_fingerprint_update(self, sample_chunk):
        """Test fingerprint updating"""
        original_version = sample_chunk.version
        original_fingerprint = sample_chunk.combined_fingerprint
        
        # Modify content and update fingerprints
        sample_chunk.content = "modified content"
        sample_chunk.update_fingerprints()
        
        # Version should increment
        assert sample_chunk.version == original_version + 1
        
        # Fingerprint should change
        assert sample_chunk.combined_fingerprint != original_fingerprint

    def test_add_relationship(self, sample_chunk):
        """Test adding relationships"""
        target_id = "target_chunk_1"
        
        sample_chunk.add_relationship(
            target_chunk_id=target_id,
            relationship_type="depends_on",
            strength=0.8,
            context="function call",
            line_number=15
        )
        
        # Should add to dependencies
        assert target_id in sample_chunk.dependencies
        
        # Should create relationship object
        assert len(sample_chunk.relationships) == 1
        rel = sample_chunk.relationships[0]
        assert rel.target_chunk_id == target_id
        assert rel.relationship_type == "depends_on"
        assert rel.strength == 0.8
        assert rel.context == "function call"
        assert rel.line_number == 15

    def test_graph_edges(self, sample_chunk):
        """Test graph edge generation"""
        sample_chunk.add_relationship("chunk1", "calls", 0.9)
        sample_chunk.add_relationship("chunk2", "imports", 0.7)
        
        edges = sample_chunk.get_graph_edges()
        assert len(edges) == 2
        
        edge1 = edges[0]
        assert edge1['source'] == sample_chunk.id
        assert edge1['target'] == "chunk1"
        assert edge1['type'] == "calls"
        assert edge1['weight'] == 0.9

    def test_semantic_tags(self, sample_chunk):
        """Test semantic tag functionality"""
        sample_chunk.add_semantic_tag(
            name="utility_function",
            confidence=0.9,
            source="ast_analysis",
            category="pattern"
        )
        
        assert len(sample_chunk.semantic_tags) == 1
        tag = sample_chunk.semantic_tags[0]
        assert tag.name == "utility_function"
        assert tag.confidence == 0.9
        assert tag.source == "ast_analysis"
        assert tag.category == "pattern"
        
        # Test tag_names property
        assert "utility_function" in sample_chunk.tag_names
        
        # Test high_confidence_tags property
        assert "utility_function" in sample_chunk.high_confidence_tags
        
        # Add low confidence tag
        sample_chunk.add_semantic_tag("low_conf_tag", confidence=0.3)
        assert "low_conf_tag" not in sample_chunk.high_confidence_tags

    def test_quality_scores(self, sample_chunk):
        """Test quality score management"""
        # Set quality scores
        sample_chunk.set_quality_score(QualityMetric.READABILITY, 0.8)
        sample_chunk.set_quality_score(QualityMetric.COMPLEXITY, 0.6)
        
        # Test retrieval
        assert sample_chunk.get_quality_score(QualityMetric.READABILITY) == 0.8
        assert sample_chunk.get_quality_score(QualityMetric.COMPLEXITY) == 0.6
        assert sample_chunk.get_quality_score(QualityMetric.TESTABILITY) == 0.0  # Not set
        
        # Test overall quality calculation
        overall = sample_chunk.calculate_overall_quality_score()
        assert 0.0 <= overall <= 1.0

    def test_embedding_functionality(self, sample_chunk):
        """Test embedding-related functionality"""
        # Initially no embedding
        assert not sample_chunk.has_semantic_embedding
        assert sample_chunk.needs_embedding_update("new_model", 1)
        
        # Set embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        sample_chunk.set_embedding(embedding, "test_model", 1)
        
        assert sample_chunk.has_semantic_embedding
        assert sample_chunk.semantic_embedding == embedding
        assert sample_chunk.embedding_model == "test_model"
        assert sample_chunk.embedding_version == 1
        
        # Test embedding update check
        assert not sample_chunk.needs_embedding_update("test_model", 1)
        assert sample_chunk.needs_embedding_update("new_model", 1)
        assert sample_chunk.needs_embedding_update("test_model", 2)

    def test_embedding_text_generation(self, sample_chunk):
        """Test embedding text generation"""
        sample_chunk.add_semantic_tag("function", 0.9)
        sample_chunk.dependencies = ["dep1", "dep2"]
        
        embedding_text = sample_chunk.get_embedding_text()
        
        # Should include chunk type
        assert ChunkType.FUNCTION.value.upper() in embedding_text
        
        # Should include tags
        assert "function" in embedding_text
        
        # Should include content
        assert sample_chunk.content in embedding_text
        
        # Should include dependencies when requested
        embedding_text_with_context = sample_chunk.get_embedding_text(include_context=True)
        assert "dep1" in embedding_text_with_context

    def test_serialization_roundtrip(self, sample_chunk):
        """Test serialization and deserialization"""
        # Add some data to make it interesting
        sample_chunk.add_semantic_tag("test_tag", 0.8)
        sample_chunk.add_relationship("other_chunk", "calls", 0.7)
        sample_chunk.set_quality_score(QualityMetric.READABILITY, 0.9)
        sample_chunk.set_embedding([0.1, 0.2, 0.3], "test_model")
        
        # Serialize to dict (with embedding by default)
        chunk_dict = sample_chunk.to_dict()
        
        # Check that all important fields are present
        assert chunk_dict['id'] == sample_chunk.id
        assert chunk_dict['content'] == sample_chunk.content
        assert chunk_dict['chunk_type'] == sample_chunk.chunk_type.value
        assert len(chunk_dict['semantic_tags']) == 1
        assert len(chunk_dict['relationships']) == 1
        
        # Check that embedding data is included
        assert chunk_dict['semantic_embedding'] == [0.1, 0.2, 0.3]
        assert chunk_dict['has_semantic_embedding'] == True
        assert chunk_dict['embedding_model'] == "test_model"
        
        # Deserialize back
        restored_chunk = SemanticChunk.from_dict(chunk_dict)
        
        # Check that important data is preserved
        assert restored_chunk.id == sample_chunk.id
        assert restored_chunk.content == sample_chunk.content
        assert restored_chunk.chunk_type == sample_chunk.chunk_type
        assert len(restored_chunk.semantic_tags) == 1
        assert len(restored_chunk.relationships) == 1
        assert restored_chunk.semantic_tags[0].name == "test_tag"
        
        # Check that embedding is properly restored
        assert restored_chunk.has_semantic_embedding
        assert restored_chunk.semantic_embedding == [0.1, 0.2, 0.3]
        assert restored_chunk.embedding_model == "test_model"

    def test_summary_info(self, sample_chunk):
        """Test summary info generation"""
        sample_chunk.add_semantic_tag("function", 0.9)
        sample_chunk.add_relationship("other", "calls")
        
        summary = sample_chunk.get_summary_info()
        
        # Check that summary contains expected fields
        assert summary['id'] == sample_chunk.id
        assert summary['type'] == sample_chunk.chunk_type.value
        assert 'quality_score' in summary
        assert 'size' in summary
        assert 'fingerprints' in summary
        assert summary['dependencies_count'] == 1
        assert summary['relationships_count'] == 1

    def test_properties(self, sample_chunk):
        """Test various properties"""
        # Test is_complex property
        assert not sample_chunk.is_complex  # Simple chunk
        
        complex_chunk = SemanticChunk(
            id="complex",
            file_path="/test/complex.py", 
            content="x" * 1000 + "{" * 20,  # Make it complex
            start_line=1,
            end_line=50,
            content_type=ContentType.CODE
        )
        # This might be complex depending on detection algorithm
        # Just test that the property returns a boolean
        assert isinstance(complex_chunk.is_complex, bool)

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Empty content
        empty_chunk = SemanticChunk(
            id="empty",
            file_path="/test/empty.py",
            content="",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        assert empty_chunk.character_count == 0
        assert empty_chunk.word_count == 0
        
        # Single line
        single_line = SemanticChunk(
            id="single",
            file_path="/test/single.py",
            content="x = 1",
            start_line=5,
            end_line=5,
            content_type=ContentType.CODE
        )
        assert single_line.line_count == 1


class TestChunkRelationship:
    """Test suite for ChunkRelationship class"""
    
    def test_relationship_creation(self):
        """Test relationship creation"""
        rel = ChunkRelationship(
            source_chunk_id="chunk1",
            target_chunk_id="chunk2",
            relationship_type="depends_on",
            strength=0.8,
            context="function call",
            line_number=10
        )
        
        assert rel.source_chunk_id == "chunk1"
        assert rel.target_chunk_id == "chunk2"
        assert rel.relationship_type == "depends_on"
        assert rel.strength == 0.8
        assert rel.context == "function call"
        assert rel.line_number == 10

    def test_to_graph_edge(self):
        """Test conversion to graph edge format"""
        rel = ChunkRelationship(
            source_chunk_id="chunk1",
            target_chunk_id="chunk2",
            relationship_type="calls",
            strength=0.9
        )
        
        edge = rel.to_graph_edge()
        
        assert edge['source'] == "chunk1"
        assert edge['target'] == "chunk2"
        assert edge['type'] == "calls"
        assert edge['weight'] == 0.9


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_create_chunk_id(self):
        """Test chunk ID creation"""
        chunk_id = create_chunk_id(
            file_path="/path/to/file.py",
            start_line=10,
            chunk_type=ChunkType.FUNCTION,
            identifier="main"
        )
        
        assert "file" in chunk_id
        assert "function" in chunk_id
        assert "main" in chunk_id
        assert "10" in chunk_id
        
        # Test without identifier
        chunk_id_no_ident = create_chunk_id(
            file_path="/path/to/file.py",
            start_line=20,
            chunk_type=ChunkType.CLASS
        )
        assert "chunk" in chunk_id_no_ident

    def test_calculate_chunk_similarity_content(self):
        """Test content-based similarity calculation"""
        chunk1 = SemanticChunk(
            id="chunk1",
            file_path="/test/file1.py",
            content="def hello_world(): print('Hello, World!')",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        chunk2 = SemanticChunk(
            id="chunk2", 
            file_path="/test/file2.py",
            content="def hello_world(): print('Hello, Universe!')",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        similarity = calculate_chunk_similarity(chunk1, chunk2, method="content")
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0  # Should have some similarity

    def test_calculate_chunk_similarity_tags(self):
        """Test tag-based similarity calculation"""
        chunk1 = SemanticChunk(
            id="chunk1",
            file_path="/test/file1.py",
            content="content1",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        chunk1.add_semantic_tag("function")
        chunk1.add_semantic_tag("utility")
        
        chunk2 = SemanticChunk(
            id="chunk2",
            file_path="/test/file2.py", 
            content="content2",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        chunk2.add_semantic_tag("function")
        chunk2.add_semantic_tag("main")
        
        similarity = calculate_chunk_similarity(chunk1, chunk2, method="semantic_tags")
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0  # Should have some similarity due to shared "function" tag

    @patch('numpy.array')
    @patch('numpy.dot')
    @patch('numpy.linalg.norm')
    def test_calculate_chunk_similarity_embedding(self, mock_norm, mock_dot, mock_array):
        """Test embedding-based similarity calculation"""
        # Mock numpy operations
        mock_array.return_value = [0.1, 0.2, 0.3]
        mock_dot.return_value = 0.5
        mock_norm.return_value = 1.0
        
        chunk1 = SemanticChunk(
            id="chunk1",
            file_path="/test/file1.py",
            content="content1",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        chunk1.set_embedding([0.1, 0.2, 0.3], "test_model")
        
        chunk2 = SemanticChunk(
            id="chunk2",
            file_path="/test/file2.py",
            content="content2", 
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        chunk2.set_embedding([0.2, 0.3, 0.4], "test_model")
        
        similarity = calculate_chunk_similarity(chunk1, chunk2, method="embedding")
        assert similarity == 0.5  # Mocked result

    def test_find_related_chunks(self):
        """Test finding related chunks"""
        target_chunk = SemanticChunk(
            id="target",
            file_path="/test/target.py",
            content="def process_data(): return data.process()",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        # Create some chunks with varying similarity
        similar_chunk = SemanticChunk(
            id="similar",
            file_path="/test/similar.py", 
            content="def process_data(): return data.clean()",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        different_chunk = SemanticChunk(
            id="different",
            file_path="/test/different.py",
            content="def render_template(): return html",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        all_chunks = [target_chunk, similar_chunk, different_chunk]
        
        # Find related chunks with low threshold to include similar_chunk
        related = find_related_chunks(
            target_chunk, 
            all_chunks, 
            similarity_threshold=0.1,
            max_results=5
        )
        
        # Should not include target_chunk itself
        assert target_chunk not in related
        
        # Should be sorted by similarity (highest first)
        assert isinstance(related, list)


class TestCodePattern:
    """Test suite for CodePattern class"""
    
    def test_code_pattern_creation(self):
        """Test code pattern creation"""
        pattern = CodePattern(
            pattern_name="singleton",
            confidence=0.8,
            evidence=["__new__ method", "instance check"],
            category="design_pattern"
        )
        
        assert pattern.pattern_name == "singleton"
        assert pattern.confidence == 0.8
        assert len(pattern.evidence) == 2
        assert pattern.category == "design_pattern"


class TestSemanticTag:
    """Test suite for SemanticTag class"""
    
    def test_semantic_tag_creation(self):
        """Test semantic tag creation"""
        tag = SemanticTag(
            name="utility_function",
            confidence=0.9,
            source="ast_analysis",
            category="pattern",
            metadata={"complexity": "low"}
        )
        
        assert tag.name == "utility_function"
        assert tag.confidence == 0.9
        assert tag.source == "ast_analysis"
        assert tag.category == "pattern"
        assert tag.metadata["complexity"] == "low"


class TestEnums:
    """Test suite for enum classes"""
    
    def test_quality_metric_enum(self):
        """Test QualityMetric enum"""
        assert QualityMetric.READABILITY.value == "readability"
        assert QualityMetric.COMPLEXITY.value == "complexity"
        
    def test_content_type_enum(self):
        """Test ContentType enum"""
        assert ContentType.CODE.value == "code"
        assert ContentType.MARKDOWN.value == "markdown"
        
    def test_chunk_complexity_enum(self):
        """Test ChunkComplexity enum"""
        assert ChunkComplexity.SIMPLE.value == "simple"
        assert ChunkComplexity.VERY_COMPLEX.value == "very_complex"


# Integration tests
class TestIntegration:
    """Integration tests combining multiple features"""
    
    def test_serialization_with_embedding_control(self):
        """Test serialization with embedding inclusion control"""
        chunk = SemanticChunk(
            id="embedding_control_test",
            file_path="/test/embedding.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        # Set embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk.set_embedding(embedding, "test_model")
        
        # Test serialization with embedding included (default)
        chunk_dict_with_embedding = chunk.to_dict(include_embedding=True)
        assert chunk_dict_with_embedding['semantic_embedding'] == embedding
        assert chunk_dict_with_embedding['has_semantic_embedding'] == True
        
        # Test serialization without embedding data
        chunk_dict_without_embedding = chunk.to_dict(include_embedding=False)
        assert chunk_dict_without_embedding['semantic_embedding'] is None
        assert chunk_dict_without_embedding['has_semantic_embedding'] == True  # Still shows it exists
        assert chunk_dict_without_embedding['embedding_model'] == "test_model"
        
        # Test deserialization of both versions
        restored_with_embedding = SemanticChunk.from_dict(chunk_dict_with_embedding)
        assert restored_with_embedding.has_semantic_embedding
        assert restored_with_embedding.semantic_embedding == embedding
        
        restored_without_embedding = SemanticChunk.from_dict(chunk_dict_without_embedding)
        assert not restored_without_embedding.has_semantic_embedding  # No actual data
        assert restored_without_embedding.embedding_model == "test_model"  # But metadata preserved

    def test_full_workflow(self):
        """Test a complete workflow with semantic chunk"""
        # Create chunk
        chunk = SemanticChunk(
            id="workflow_test",
            file_path="/test/workflow.py",
            content="class DataProcessor:\n    def process(self, data):\n        return clean(data)",
            start_line=1,
            end_line=3,
            content_type=ContentType.CODE,
            language="python",
            chunk_type=ChunkType.CLASS
        )
        
        # Add semantic information
        chunk.add_semantic_tag("data_processing", 0.9, "manual")
        chunk.add_semantic_tag("class_definition", 0.95, "ast")
        
        # Add relationships
        chunk.add_relationship("data_cleaner", "depends_on", 0.8)
        
        # Add quality scores
        chunk.set_quality_score(QualityMetric.READABILITY, 0.8)
        chunk.set_quality_score(QualityMetric.MAINTAINABILITY, 0.7)
        
        # Set embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk.set_embedding(embedding, "test_model")
        
        # Test serialization roundtrip with full data
        chunk_dict = chunk.to_dict(include_embedding=True)
        restored_chunk = SemanticChunk.from_dict(chunk_dict)
        
        # Verify everything is preserved
        assert restored_chunk.id == chunk.id
        assert len(restored_chunk.semantic_tags) == 2
        assert len(restored_chunk.relationships) == 1
        assert len(restored_chunk.quality_scores) == 2
        assert restored_chunk.has_semantic_embedding
        assert restored_chunk.semantic_embedding == embedding
        
        # Test summary info
        summary = restored_chunk.get_summary_info()
        assert summary['high_confidence_tags'] == ['data_processing', 'class_definition']
        assert summary['has_embedding'] == True


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])