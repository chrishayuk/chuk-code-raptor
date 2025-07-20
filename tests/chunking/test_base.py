#!/usr/bin/env python3
# tests/chunking/test_base.py
"""
Comprehensive pytest tests for BaseParser and related classes
=============================================================

Tests cover:
- ParseContext initialization and validation
- BaseParser abstract interface and concrete implementations
- Content type detection across different file types
- Chunk creation with proper metadata
- Size filtering and post-processing logic
- Importance scoring algorithms
- Error handling and edge cases
- Compatibility interfaces
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

# Import the classes we're testing
from chuk_code_raptor.chunking.base import (
    BaseParser, ParseContext, ParserError, UnsupportedLanguageError,
    InvalidContentError
)
from chuk_code_raptor.chunking.semantic_chunk import SemanticChunk, ContentType
from chuk_code_raptor.chunking.config import ChunkingConfig
from chuk_code_raptor.core.models import ChunkType


class TestParseContext:
    """Test suite for ParseContext class"""
    
    def test_basic_initialization(self):
        """Test basic ParseContext initialization"""
        context = ParseContext(
            file_path="/src/test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=2000,
            min_chunk_size=50
        )
        
        assert context.file_path == "/src/test.py"
        assert context.language == "python"
        assert context.content_type == ContentType.CODE
        assert context.max_chunk_size == 2000
        assert context.min_chunk_size == 50
        assert context.enable_semantic_analysis == True  # Default
        assert context.enable_dependency_tracking == True  # Default
        assert context.metadata == {}  # Default empty dict
    
    def test_initialization_with_optional_parameters(self):
        """Test ParseContext with all optional parameters"""
        metadata = {"source": "test", "version": "1.0"}
        
        context = ParseContext(
            file_path="/src/test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=2000,
            min_chunk_size=50,
            enable_semantic_analysis=False,
            enable_dependency_tracking=False,
            metadata=metadata
        )
        
        assert context.enable_semantic_analysis == False
        assert context.enable_dependency_tracking == False
        assert context.metadata == metadata
    
    def test_post_init_metadata_handling(self):
        """Test __post_init__ method handles None metadata"""
        context = ParseContext(
            file_path="/src/test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=2000,
            min_chunk_size=50,
            metadata=None
        )
        
        assert context.metadata == {}


class TestBaseParser:
    """Test suite for BaseParser abstract class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 50
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'structural'
        return config
    
    @pytest.fixture
    def concrete_parser(self, mock_config):
        """Create a concrete implementation of BaseParser for testing"""
        class ConcreteParser(BaseParser):
            def __init__(self, config):
                super().__init__(config)
                self.supported_languages = {"python", "javascript"}
                self.supported_extensions = {".py", ".js"}
                self.parser_type = "test_parser"
            
            def can_parse(self, language: str, file_extension: str) -> bool:
                return (language in self.supported_languages or 
                        file_extension in self.supported_extensions)
            
            def parse(self, content: str, context: ParseContext) -> list:
                # Simple mock implementation
                return [self._create_chunk(
                    content=content,
                    start_line=1,
                    end_line=content.count('\n') + 1,
                    chunk_type=ChunkType.FUNCTION,
                    language=context.language,
                    file_path=context.file_path,
                    identifier="test_function"
                )]
        
        return ConcreteParser(mock_config)
    
    def test_abstract_methods_require_implementation(self, mock_config):
        """Test that BaseParser cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseParser(mock_config)
    
    def test_initialization(self, concrete_parser, mock_config):
        """Test BaseParser initialization"""
        assert concrete_parser.config == mock_config
        assert concrete_parser.supported_languages == {"python", "javascript"}
        assert concrete_parser.supported_extensions == {".py", ".js"}
        assert concrete_parser.parser_type == "test_parser"
        assert concrete_parser.name == "ConcreteParser"
        assert concrete_parser.strategy == "structural"
    
    def test_can_parse_implementation(self, concrete_parser):
        """Test can_parse method implementation"""
        # Test supported language
        assert concrete_parser.can_parse("python", ".py") == True
        assert concrete_parser.can_parse("javascript", ".js") == True
        
        # Test supported extension only
        assert concrete_parser.can_parse("unknown", ".py") == True
        assert concrete_parser.can_parse("unknown", ".js") == True
        
        # Test unsupported
        assert concrete_parser.can_parse("unknown", ".txt") == False
    
    def test_get_priority(self, concrete_parser):
        """Test priority calculation"""
        # Supported language should get highest priority
        assert concrete_parser.get_priority("python", ".py") == 100
        assert concrete_parser.get_priority("javascript", ".js") == 100
        
        # Supported extension only should get medium priority
        assert concrete_parser.get_priority("unknown", ".py") == 50
        assert concrete_parser.get_priority("unknown", ".js") == 50
        
        # Unsupported should get zero priority
        assert concrete_parser.get_priority("unknown", ".txt") == 0
    
    def test_compatibility_methods(self, concrete_parser):
        """Test compatibility interface methods"""
        # Test can_chunk (compatibility for can_parse)
        assert concrete_parser.can_chunk("python", ".py") == True
        assert concrete_parser.can_chunk("unknown", ".txt") == False
    
    def test_chunk_content_compatibility(self, concrete_parser):
        """Test chunk_content compatibility method"""
        content = "def test_function():\n    pass"
        chunks = concrete_parser.chunk_content(content, "python", "/src/test.py")
        
        assert len(chunks) == 1
        assert isinstance(chunks[0], SemanticChunk)
        assert chunks[0].content == content
        assert chunks[0].language == "python"
        assert chunks[0].file_path == "/src/test.py"
    
    def test_parse_content_alias(self, concrete_parser):
        """Test parse_content alias method"""
        content = "def test_function():\n    pass"
        context = ParseContext(
            file_path="/src/test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=2000,
            min_chunk_size=50
        )
        
        chunks = concrete_parser.parse_content(content, context)
        assert len(chunks) == 1
        assert isinstance(chunks[0], SemanticChunk)


class TestContentTypeDetection:
    """Test suite for content type detection"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 50
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'structural'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser for content type testing"""
        class TestParser(BaseParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def parse(self, content: str, context: ParseContext) -> list:
                return []
        
        return TestParser(mock_config)
    
    def test_markdown_detection(self, parser):
        """Test markdown file detection"""
        assert parser._detect_content_type("README.md", "markdown") == ContentType.MARKDOWN
        assert parser._detect_content_type("doc.markdown", "markdown") == ContentType.MARKDOWN
    
    def test_html_detection(self, parser):
        """Test HTML file detection"""
        assert parser._detect_content_type("index.html", "html") == ContentType.HTML
        assert parser._detect_content_type("page.htm", "html") == ContentType.HTML
    
    def test_json_detection(self, parser):
        """Test JSON file detection"""
        assert parser._detect_content_type("config.json", "json") == ContentType.JSON
    
    def test_yaml_detection(self, parser):
        """Test YAML file detection"""
        assert parser._detect_content_type("config.yaml", "yaml") == ContentType.YAML
        assert parser._detect_content_type("docker-compose.yml", "yaml") == ContentType.YAML
    
    def test_xml_detection(self, parser):
        """Test XML file detection"""
        assert parser._detect_content_type("config.xml", "xml") == ContentType.XML
    
    def test_code_detection(self, parser):
        """Test code language detection"""
        code_languages = ["python", "javascript", "typescript", "rust", "go", "java", "cpp"]
        for language in code_languages:
            assert parser._detect_content_type(f"test.{language}", language) == ContentType.CODE
    
    def test_default_text_detection(self, parser):
        """Test default text detection"""
        assert parser._detect_content_type("README.txt", "text") == ContentType.PLAINTEXT
        assert parser._detect_content_type("unknown.xyz", "unknown") == ContentType.PLAINTEXT


class TestChunkCreation:
    """Test suite for chunk creation helper methods"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 50
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'structural'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser for chunk creation testing"""
        class TestParser(BaseParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def parse(self, content: str, context: ParseContext) -> list:
                return []
        
        return TestParser(mock_config)
    
    def test_create_chunk_basic(self, parser):
        """Test basic chunk creation"""
        content = "def test_function():\n    pass"
        chunk = parser._create_chunk(
            content=content,
            start_line=10,
            end_line=12,
            chunk_type=ChunkType.FUNCTION,
            language="python",
            file_path="/src/test.py",
            identifier="test_function"
        )
        
        assert isinstance(chunk, SemanticChunk)
        assert chunk.content == content
        assert chunk.start_line == 10
        assert chunk.end_line == 12
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.language == "python"
        assert chunk.file_path == "/src/test.py"
        assert chunk.content_type == ContentType.CODE
        
        # Check metadata
        assert chunk.metadata['parser'] == 'TestParser'
        assert chunk.metadata['parser_type'] == 'base'
        assert chunk.metadata['strategy'] == 'structural'
        assert chunk.metadata['extraction_method'] == 'structural'
    
    def test_create_chunk_with_custom_metadata(self, parser):
        """Test chunk creation with custom metadata"""
        custom_metadata = {"complexity": "high", "importance": 0.8}
        
        chunk = parser._create_chunk(
            content="test content",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language="python",
            file_path="/src/test.py",
            metadata=custom_metadata
        )
        
        # Check that custom metadata is merged
        assert chunk.metadata['complexity'] == "high"
        assert chunk.metadata['importance'] == 0.8
        assert chunk.metadata['parser'] == 'TestParser'  # Still has parser metadata
    
    def test_create_chunk_id_generation(self, parser):
        """Test that chunk ID is properly generated"""
        chunk = parser._create_chunk(
            content="test content",
            start_line=5,
            end_line=7,
            chunk_type=ChunkType.FUNCTION,
            language="python",
            file_path="/src/test.py",
            identifier="my_function"
        )
        
        # ID should include file name, type, identifier, and line
        assert "test" in chunk.id  # filename
        assert "function" in chunk.id  # chunk type
        assert "my_function" in chunk.id  # identifier
        assert "5" in chunk.id  # start line


class TestSizeFiltering:
    """Test suite for size filtering and validation"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 100
        config.min_chunk_size = 20
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'structural'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser with specific size constraints"""
        mock_config.min_chunk_size = 20
        mock_config.max_chunk_size = 100
        mock_config.preserve_atomic_nodes = True
        
        class TestParser(BaseParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def parse(self, content: str, context: ParseContext) -> list:
                return []
        
        return TestParser(mock_config)
    
    def test_should_include_chunk_normal_size(self, parser):
        """Test chunk inclusion for normal-sized chunks"""
        chunk = SemanticChunk(
            id="test",
            file_path="/test.py",
            content="x" * 50,  # Within size limits
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            content_type=ContentType.CODE
        )
        
        assert parser._should_include_chunk(chunk) == True
    
    def test_should_include_chunk_too_small(self, parser):
        """Test chunk exclusion for chunks that are too small"""
        chunk = SemanticChunk(
            id="test",
            file_path="/test.py",
            content="x" * 10,  # Below min_chunk_size (20)
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            content_type=ContentType.CODE
        )
        
        assert parser._should_include_chunk(chunk) == False
    
    def test_should_include_small_important_chunks(self, parser):
        """Test that small but important chunks are included"""
        # Import chunks should be included even if small
        import_chunk = SemanticChunk(
            id="test",
            file_path="/test.py",
            content="import os",  # Small but important
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.IMPORT,
            content_type=ContentType.CODE
        )
        
        assert parser._should_include_chunk(import_chunk) == True
        
        # Comment chunks should also be included even if small
        comment_chunk = SemanticChunk(
            id="test",
            file_path="/test.py",
            content="# Short comment",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.COMMENT,
            content_type=ContentType.CODE
        )
        
        assert parser._should_include_chunk(comment_chunk) == True
    
    def test_should_include_chunk_too_large(self, parser):
        """Test chunk handling for chunks that are too large"""
        chunk = SemanticChunk(
            id="test",
            file_path="/test.py",
            content="x" * 200,  # Above max_chunk_size (100)
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            content_type=ContentType.CODE
        )
        
        # Large chunks are included if preserve_atomic_nodes is True
        assert parser._should_include_chunk(chunk) == True
        
        # Test with preserve_atomic_nodes = False
        parser.config.preserve_atomic_nodes = False
        assert parser._should_include_chunk(chunk) == False


class TestPostProcessing:
    """Test suite for post-processing logic"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 100
        config.min_chunk_size = 20
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'structural'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser for post-processing tests"""
        mock_config.min_chunk_size = 20
        mock_config.max_chunk_size = 100
        mock_config.preserve_atomic_nodes = True
        
        class TestParser(BaseParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def parse(self, content: str, context: ParseContext) -> list:
                return []
        
        return TestParser(mock_config)
    
    def test_post_process_filtering(self, parser):
        """Test that post-processing filters chunks appropriately"""
        chunks = [
            # Normal chunk - should be included
            SemanticChunk(
                id="normal", file_path="/test.py", content="x" * 50,
                start_line=1, end_line=1, chunk_type=ChunkType.FUNCTION,
                content_type=ContentType.CODE
            ),
            # Too small chunk - should be excluded
            SemanticChunk(
                id="small", file_path="/test.py", content="x" * 10,
                start_line=2, end_line=2, chunk_type=ChunkType.FUNCTION,
                content_type=ContentType.CODE
            ),
            # Small but important - should be included
            SemanticChunk(
                id="import", file_path="/test.py", content="import os",
                start_line=3, end_line=3, chunk_type=ChunkType.IMPORT,
                content_type=ContentType.CODE
            ),
            # Large atomic chunk - should be included with warning
            SemanticChunk(
                id="large", file_path="/test.py", content="x" * 200,
                start_line=4, end_line=4, chunk_type=ChunkType.CLASS,
                content_type=ContentType.CODE
            )
        ]
        
        with patch('chuk_code_raptor.chunking.base.logger') as mock_logger:
            processed = parser._post_process(chunks)
            
            # Should have 3 chunks (normal, import, large)
            assert len(processed) == 3
            
            # Check that large chunk warning was logged
            mock_logger.warning.assert_called_once()
            
            # Verify chunks are sorted by line number
            assert processed[0].id == "normal"    # line 1
            assert processed[1].id == "import"    # line 3
            assert processed[2].id == "large"     # line 4
    
    def test_is_atomic_chunk(self, parser):
        """Test atomic chunk detection"""
        atomic_types = [ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.CLASS]
        non_atomic_types = [ChunkType.IMPORT, ChunkType.COMMENT, ChunkType.VARIABLE]
        
        for chunk_type in atomic_types:
            chunk = SemanticChunk(
                id="test", file_path="/test.py", content="test",
                start_line=1, end_line=1, chunk_type=chunk_type,
                content_type=ContentType.CODE
            )
            assert parser._is_atomic_chunk(chunk) == True
        
        for chunk_type in non_atomic_types:
            chunk = SemanticChunk(
                id="test", file_path="/test.py", content="test",
                start_line=1, end_line=1, chunk_type=chunk_type,
                content_type=ContentType.CODE
            )
            assert parser._is_atomic_chunk(chunk) == False


class TestImportanceScoring:
    """Test suite for importance scoring algorithms"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 50
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'structural'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser for importance scoring tests"""
        mock_config.target_chunk_size = 500
        
        class TestParser(BaseParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def parse(self, content: str, context: ParseContext) -> list:
                return []
        
        return TestParser(mock_config)
    
    def test_calculate_importance_score_base(self, parser):
        """Test base importance score calculation"""
        chunk = SemanticChunk(
            id="test", file_path="/test.py", content="x" * 500,  # Target size
            start_line=1, end_line=1, chunk_type=ChunkType.FUNCTION,
            content_type=ContentType.CODE
        )
        
        score = parser._calculate_importance_score(chunk)
        
        # Should get base score (0.5) + size bonus (0.2) + semantic bonus (0.3)
        expected = 0.5 + 0.2 + 0.3
        assert abs(score - expected) < 0.01
    
    def test_calculate_importance_score_with_dependencies(self, parser):
        """Test importance score with dependencies"""
        chunk = SemanticChunk(
            id="test", file_path="/test.py", content="x" * 500,
            start_line=1, end_line=1, chunk_type=ChunkType.FUNCTION,
            content_type=ContentType.CODE
        )
        chunk.dependencies = ["dep1", "dep2", "dep3"]  # 3 dependencies
        
        score = parser._calculate_importance_score(chunk)
        
        # Should get additional dependency bonus: min(3 * 0.05, 0.2) = 0.15
        # But score is capped at 1.0, so: min(0.5 + 0.2 + 0.3 + 0.15, 1.0) = 1.0
        expected = 1.0  # Capped at 1.0
        assert score == expected
    
    def test_calculate_importance_score_size_penalties(self, parser):
        """Test importance score with different sizes"""
        # Very small chunk
        small_chunk = SemanticChunk(
            id="small", file_path="/test.py", content="x" * 50,  # 10% of target
            start_line=1, end_line=1, chunk_type=ChunkType.COMMENT,
            content_type=ContentType.CODE
        )
        
        small_score = parser._calculate_importance_score(small_chunk)
        assert small_score == 0.5  # Just base score, no bonuses
        
        # Very large chunk
        large_chunk = SemanticChunk(
            id="large", file_path="/test.py", content="x" * 2000,  # 400% of target
            start_line=1, end_line=1, chunk_type=ChunkType.FUNCTION,
            content_type=ContentType.CODE
        )
        
        large_score = parser._calculate_importance_score(large_chunk)
        assert large_score == 0.8  # Base + semantic bonus only
    
    def test_calculate_importance_score_capped(self, parser):
        """Test that importance score is capped at 1.0"""
        chunk = SemanticChunk(
            id="test", file_path="/test.py", content="x" * 500,
            start_line=1, end_line=1, chunk_type=ChunkType.FUNCTION,
            content_type=ContentType.CODE
        )
        # Add many dependencies to test capping
        chunk.dependencies = ["dep" + str(i) for i in range(20)]
        
        score = parser._calculate_importance_score(chunk)
        assert score <= 1.0


class TestHelperMethods:
    """Test suite for various helper methods"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 50
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'structural'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser for helper method tests"""
        class TestParser(BaseParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def parse(self, content: str, context: ParseContext) -> list:
                return []
            
            def _extract_identifier(self, content: str, chunk_type: ChunkType) -> str:
                # Override for testing
                if "def test_function" in content:
                    return "test_function"
                return None
        
        return TestParser(mock_config)
    
    def test_extract_identifier_base_implementation(self, parser):
        """Test base implementation of extract_identifier"""
        # Create a minimal concrete parser to test the base implementation
        class MinimalParser(BaseParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def parse(self, content: str, context: ParseContext) -> list:
                return []
            
            # Don't override _extract_identifier to test base implementation
        
        minimal_parser = MinimalParser(parser.config)
        result = minimal_parser._extract_identifier("def test():", ChunkType.FUNCTION)
        assert result is None
    
    def test_extract_identifier_override(self, parser):
        """Test overridden extract_identifier implementation"""
        result = parser._extract_identifier("def test_function():", ChunkType.FUNCTION)
        assert result == "test_function"
        
        result = parser._extract_identifier("def other_function():", ChunkType.FUNCTION)
        assert result is None


class TestErrorHandling:
    """Test suite for error handling"""
    
    def test_parser_error_hierarchy(self):
        """Test exception hierarchy"""
        assert issubclass(UnsupportedLanguageError, ParserError)
        assert issubclass(InvalidContentError, ParserError)
        assert issubclass(ParserError, Exception)
    
    def test_unsupported_language_error(self):
        """Test UnsupportedLanguageError"""
        with pytest.raises(UnsupportedLanguageError):
            raise UnsupportedLanguageError("Language not supported")
    
    def test_invalid_content_error(self):
        """Test InvalidContentError"""
        with pytest.raises(InvalidContentError):
            raise InvalidContentError("Content cannot be parsed")


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 50
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'structural'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser for edge case testing"""
        class TestParser(BaseParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def parse(self, content: str, context: ParseContext) -> list:
                return []
        
        return TestParser(mock_config)
    
    def test_empty_content(self, parser):
        """Test handling of empty content"""
        chunk = parser._create_chunk(
            content="",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language="python",
            file_path="/test.py"
        )
        
        assert chunk.content == ""
        assert len(chunk.content) == 0
    
    def test_very_long_file_path(self, parser):
        """Test handling of very long file paths"""
        long_path = "/very/long/path/" + "directory/" * 50 + "file.py"
        
        chunk = parser._create_chunk(
            content="test",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language="python",
            file_path=long_path
        )
        
        assert chunk.file_path == long_path
    
    def test_zero_line_numbers(self, parser):
        """Test handling of zero or negative line numbers"""
        chunk = parser._create_chunk(
            content="test",
            start_line=0,
            end_line=0,
            chunk_type=ChunkType.FUNCTION,
            language="python",
            file_path="/test.py"
        )
        
        assert chunk.start_line == 0
        assert chunk.end_line == 0
    
    def test_invalid_chunk_type(self, parser):
        """Test handling of invalid chunk types"""
        # This should work as ChunkType is an enum
        chunk = parser._create_chunk(
            content="test",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language="python",
            file_path="/test.py"
        )
        
        assert chunk.chunk_type == ChunkType.FUNCTION


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])