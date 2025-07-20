#!/usr/bin/env python3
# tests/chunking/test_integration.py
"""
Integration tests for the complete chunking system
=================================================

These tests verify that the chunking system works end-to-end,
matching the functionality demonstrated in the demo scripts.
Tests the actual integration between all components.
"""

import pytest
import tempfile
import os
from pathlib import Path

# Import the main chunking module
import chuk_code_raptor.chunking as chunking
from chuk_code_raptor.core.models import ChunkType
from chuk_code_raptor.chunking.semantic_chunk import QualityMetric


class TestChunkingSystemIntegration:
    """Integration tests for the complete chunking system"""
    
    def test_basic_chunking_workflow(self):
        """Test basic end-to-end chunking workflow"""
        # Create a simple Python file
        python_code = '''def hello_world():
    """A simple greeting function."""
    print("Hello, World!")
    return "greeting"

class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        return a * b
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            f.flush()
            temp_path = f.name
        
        try:
            # Test chunking the file
            chunks = chunking.chunk_file(temp_path, "python")
            
            # Should return a list of chunks
            assert isinstance(chunks, list)
            
            # Test chunking content directly
            chunks_direct = chunking.chunk_content(python_code, "python", "test.py")
            assert isinstance(chunks_direct, list)
            
        finally:
            os.unlink(temp_path)
    
    def test_engine_creation_and_usage(self):
        """Test creating and using the chunking engine"""
        # Test default engine creation
        engine = chunking.create_engine()
        assert engine is not None
        
        # Test with custom config
        config = chunking.ChunkingConfig(
            max_chunk_size=1000,
            min_chunk_size=50,
            target_chunk_size=500
        )
        engine_custom = chunking.create_engine(config)
        assert engine_custom is not None
        assert engine_custom.config.max_chunk_size == 1000
    
    def test_configuration_presets(self):
        """Test that all configuration presets work"""
        configs = [
            chunking.DEFAULT_CONFIG,
            chunking.FAST_CONFIG,
            chunking.PRECISE_CONFIG,
            chunking.SEMANTIC_CONFIG,
            chunking.HYBRID_CONFIG,
            chunking.LARGE_FILES_CONFIG,
            chunking.DOCUMENT_CONFIG
        ]
        
        for config in configs:
            # Should be able to create engine with each preset
            engine = chunking.create_engine(config)
            assert engine is not None
            
            # Basic properties should be set
            assert hasattr(config, 'max_chunk_size')
            assert hasattr(config, 'min_chunk_size')
            assert hasattr(config, 'target_chunk_size')
    
    def test_supported_languages_and_extensions(self):
        """Test language and extension support"""
        languages = chunking.get_supported_languages()
        extensions = chunking.get_supported_extensions()
        
        # Should return lists
        assert isinstance(languages, list)
        assert isinstance(extensions, list)
        
        # Get parser info
        parser_info = chunking.get_parser_info()
        assert isinstance(parser_info, dict)
        assert 'available_parsers' in parser_info
        assert 'total_parsers' in parser_info
        assert 'supported_languages' in parser_info
    
    def test_semantic_chunk_creation(self):
        """Test SemanticChunk creation and basic functionality"""
        # Test creating a semantic chunk
        chunk = chunking.SemanticChunk(
            id="test_chunk",
            file_path="test.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            content_type=chunking.ContentType.CODE,
            chunk_type=ChunkType.FUNCTION
        )
        
        # Basic properties should be set
        assert chunk.id == "test_chunk"
        assert chunk.file_path == "test.py"
        assert chunk.content == "def test(): pass"
        assert chunk.content_type == chunking.ContentType.CODE
        
        # Should have computed properties
        assert hasattr(chunk, 'content_hash')
        assert hasattr(chunk, 'character_count')
        assert hasattr(chunk, 'word_count')
        assert hasattr(chunk, 'content_fingerprint')
        
        # Test adding semantic tags
        chunk.add_semantic_tag("function", confidence=0.9, source="ast")
        assert len(chunk.semantic_tags) == 1
        assert chunk.semantic_tags[0].name == "function"
        assert chunk.semantic_tags[0].confidence == 0.9
    
    def test_chunk_relationships(self):
        """Test chunk relationship functionality"""
        chunk1 = chunking.SemanticChunk(
            id="chunk1",
            file_path="test.py",
            content="def function1(): pass",
            start_line=1,
            end_line=1,
            content_type=chunking.ContentType.CODE
        )
        
        chunk2 = chunking.SemanticChunk(
            id="chunk2",
            file_path="test.py",
            content="def function2(): function1()",
            start_line=3,
            end_line=3,
            content_type=chunking.ContentType.CODE
        )
        
        # Add relationship
        chunk2.add_relationship("chunk1", "calls", strength=0.8, context="function call")
        
        # Should have relationship
        assert len(chunk2.relationships) == 1
        assert chunk2.relationships[0].target_chunk_id == "chunk1"
        assert chunk2.relationships[0].relationship_type == "calls"
        assert chunk2.relationships[0].strength == 0.8
        
        # Test graph edge conversion
        edges = chunk2.get_graph_edges()
        assert len(edges) == 1
        assert edges[0]['source'] == "chunk2"
        assert edges[0]['target'] == "chunk1"
        assert edges[0]['type'] == "calls"
    
    def test_chunk_similarity_calculation(self):
        """Test chunk similarity calculation"""
        chunk1 = chunking.SemanticChunk(
            id="chunk1",
            file_path="test1.py",
            content="def add(a, b): return a + b",
            start_line=1,
            end_line=1,
            content_type=chunking.ContentType.CODE
        )
        
        chunk2 = chunking.SemanticChunk(
            id="chunk2", 
            file_path="test2.py",
            content="def subtract(a, b): return a - b",
            start_line=1,
            end_line=1,
            content_type=chunking.ContentType.CODE
        )
        
        # Test content similarity
        similarity = chunking.calculate_chunk_similarity(chunk1, chunk2, method="content")
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        
        # Add some common tags
        chunk1.add_semantic_tag("math")
        chunk1.add_semantic_tag("function")
        chunk2.add_semantic_tag("math")
        chunk2.add_semantic_tag("operation")
        
        # Test tag similarity
        tag_similarity = chunking.calculate_chunk_similarity(chunk1, chunk2, method="semantic_tags")
        assert isinstance(tag_similarity, float)
        assert 0.0 <= tag_similarity <= 1.0
    
    def test_chunk_serialization(self):
        """Test chunk serialization and deserialization"""
        chunk = chunking.SemanticChunk(
            id="test_chunk",
            file_path="test.py",
            content="def test(): return 'hello'",
            start_line=1,
            end_line=1,
            content_type=chunking.ContentType.CODE,
            language="python"
        )
        
        # Add some semantic information
        chunk.add_semantic_tag("function", confidence=0.9)
        chunk.add_relationship("other_chunk", "calls", strength=0.7)
        
        # Test serialization
        serialized = chunk.to_dict()
        assert isinstance(serialized, dict)
        assert serialized['id'] == "test_chunk"
        assert serialized['content'] == "def test(): return 'hello'"
        assert len(serialized['semantic_tags']) == 1
        assert len(serialized['relationships']) == 1
        
        # Test deserialization
        restored_chunk = chunking.SemanticChunk.from_dict(serialized)
        assert restored_chunk.id == chunk.id
        assert restored_chunk.content == chunk.content
        assert len(restored_chunk.semantic_tags) == len(chunk.semantic_tags)
        assert len(restored_chunk.relationships) == len(chunk.relationships)
    
    def test_content_types_and_detection(self):
        """Test different content types and their detection"""
        # Test different content types
        content_types = [
            chunking.ContentType.CODE,
            chunking.ContentType.DOCUMENTATION,
            chunking.ContentType.MARKDOWN,
            chunking.ContentType.HTML,
            chunking.ContentType.PLAINTEXT,
            chunking.ContentType.JSON,
            chunking.ContentType.YAML
        ]
        
        for content_type in content_types:
            chunk = chunking.SemanticChunk(
                id=f"chunk_{content_type.value}",
                file_path=f"test.{content_type.value}",
                content=f"Sample {content_type.value} content",
                start_line=1,
                end_line=1,
                content_type=content_type
            )
            
            assert chunk.content_type == content_type
    
    def test_error_handling(self):
        """Test error handling in the chunking system"""
        # Test with nonexistent file
        with pytest.raises(FileNotFoundError):
            chunking.chunk_file("/nonexistent/file.py", "python")
        
        # Test with empty content
        chunks = chunking.chunk_content("", "python", "empty.py")
        assert isinstance(chunks, list)
        assert len(chunks) == 0
        
        # Test with whitespace-only content
        chunks = chunking.chunk_content("   \n\t  ", "python", "whitespace.py")
        assert isinstance(chunks, list)
        assert len(chunks) == 0
    
    def test_markdown_content_chunking(self):
        """Test chunking markdown content"""
        markdown_content = '''# Main Heading

This is a paragraph with some text.

## Subheading

- List item 1
- List item 2
- List item 3

```python
def example():
    return "code block"
```

Another paragraph.
'''
        
        chunks = chunking.chunk_content(markdown_content, "markdown", "test.md")
        assert isinstance(chunks, list)
        
        # Should handle markdown content without errors
    
    def test_quality_scoring_integration(self):
        """Test quality scoring functionality"""
        chunk = chunking.SemanticChunk(
            id="quality_test",
            file_path="test.py",
            content="def well_documented_function():\n    '''This function is well documented.'''\n    return True",
            start_line=1,
            end_line=3,
            content_type=chunking.ContentType.CODE,
            language="python"
        )
        
        # Test setting quality scores
        chunk.set_quality_score(QualityMetric.READABILITY, 0.9)
        chunk.set_quality_score(QualityMetric.DOCUMENTATION_QUALITY, 0.8)
        
        # Test getting quality scores
        readability = chunk.get_quality_score(QualityMetric.READABILITY)
        assert readability == 0.9
        
        # Test overall quality calculation
        overall_quality = chunk.calculate_overall_quality_score()
        assert isinstance(overall_quality, float)
        assert 0.0 <= overall_quality <= 1.0
    
    def test_change_detection(self):
        """Test change detection and fingerprinting"""
        chunk = chunking.SemanticChunk(
            id="change_test",
            file_path="test.py",
            content="original content",
            start_line=1,
            end_line=1,
            content_type=chunking.ContentType.CODE
        )
        
        # Get original fingerprints
        original_content_fp = chunk.content_fingerprint
        original_combined_fp = chunk.combined_fingerprint
        original_version = chunk.version
        
        # Simulate content change
        chunk.content = "modified content"
        chunk.update_fingerprints()
        
        # Should detect changes
        assert chunk.has_content_changed(original_content_fp)
        assert chunk.has_changed(original_combined_fp)
        assert chunk.version > original_version
    
    def test_utility_functions(self):
        """Test utility functions"""
        # Test chunk ID creation
        chunk_id = chunking.create_chunk_id(
            "/path/to/file.py", 
            10, 
            ChunkType.FUNCTION, 
            "test_function"
        )
        
        assert isinstance(chunk_id, str)
        assert "file" in chunk_id  # Should contain filename
        assert "function" in chunk_id  # Should contain chunk type
        assert "test_function" in chunk_id  # Should contain identifier
        assert "10" in chunk_id  # Should contain line number


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""
    
    def test_code_review_scenario(self):
        """Test code review automation scenario"""
        # Sample code that might be reviewed
        code_sample = '''
def process_user_data(user_data):
    # TODO: Add input validation
    if user_data:
        result = []
        for item in user_data:
            if item.get('active'):
                processed = {
                    'id': item['id'],
                    'name': item['name'],
                    'score': item.get('score', 0) * 1.5
                }
                result.append(processed)
        return result
    return []
'''
        
        chunks = chunking.chunk_content(code_sample, "python", "review.py")
        
        # Should be able to analyze for code review
        assert isinstance(chunks, list)
        
        # If chunks are created, they should have basic properties
        for chunk in chunks:
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'chunk_type')
            assert hasattr(chunk, 'content_fingerprint')
    
    def test_documentation_processing_scenario(self):
        """Test documentation processing scenario"""
        doc_content = '''
# API Documentation

## Authentication

All API requests require authentication using an API key.

### Getting an API Key

1. Register for an account
2. Navigate to the API section
3. Generate a new key

## Endpoints

### GET /users

Returns a list of users.

**Parameters:**
- limit (integer): Maximum number of users to return
- offset (integer): Number of users to skip

**Response:**
```json
{
  "users": [...],
  "total": 100
}
```
'''
        
        chunks = chunking.chunk_content(doc_content, "markdown", "api_docs.md")
        
        # Should handle documentation content
        assert isinstance(chunks, list)
    
    def test_large_file_processing(self):
        """Test processing larger content"""
        # Generate a larger code sample
        large_content = '''
class UserManager:
    """Manages user operations and data persistence."""
    
    def __init__(self, database_url):
        self.db = connect(database_url)
        self.cache = {}
    
    def create_user(self, user_data):
        """Create a new user."""
        if not self.validate_user_data(user_data):
            raise ValueError("Invalid user data")
        
        user_id = self.generate_user_id()
        user = {
            'id': user_id,
            'email': user_data['email'],
            'name': user_data['name'],
            'created_at': datetime.now(),
            'active': True
        }
        
        self.db.insert('users', user)
        self.cache[user_id] = user
        return user
    
    def get_user(self, user_id):
        """Retrieve a user by ID."""
        if user_id in self.cache:
            return self.cache[user_id]
        
        user = self.db.query('users', {'id': user_id})
        if user:
            self.cache[user_id] = user
        return user
    
    def update_user(self, user_id, updates):
        """Update user information."""
        user = self.get_user(user_id)
        if not user:
            raise ValueError("User not found")
        
        for key, value in updates.items():
            if key in ['email', 'name', 'active']:
                user[key] = value
        
        user['updated_at'] = datetime.now()
        self.db.update('users', {'id': user_id}, user)
        self.cache[user_id] = user
        return user
    
    def delete_user(self, user_id):
        """Delete a user."""
        if user_id in self.cache:
            del self.cache[user_id]
        return self.db.delete('users', {'id': user_id})
    
    def validate_user_data(self, data):
        """Validate user data."""
        required_fields = ['email', 'name']
        return all(field in data for field in required_fields)
    
    def generate_user_id(self):
        """Generate a unique user ID."""
        import uuid
        return str(uuid.uuid4())
'''
        
        chunks = chunking.chunk_content(large_content, "python", "user_manager.py")
        
        # Should handle larger content
        assert isinstance(chunks, list)
        
        # Should create multiple chunks for a class this large (if parsers are available)
        # The number depends on available parsers and configuration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])