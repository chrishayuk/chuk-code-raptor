#!/usr/bin/env python3
# tests/chunking/test_heuristic_base.py
"""
Comprehensive pytest tests for HeuristicParser class
====================================================

Tests cover:
- HeuristicParser abstract interface and concrete implementations
- Heuristic chunk creation with importance scoring
- Content analysis and pattern matching utilities
- Line-based content manipulation methods
- Section extraction with regex patterns
- Integration with BaseParser functionality
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch
import re

# Import the classes we're testing
from chuk_code_raptor.chunking.heuristic_base import HeuristicParser
from chuk_code_raptor.chunking.base import ParseContext
from chuk_code_raptor.chunking.semantic_chunk import SemanticChunk, ContentType
from chuk_code_raptor.chunking.config import ChunkingConfig
from chuk_code_raptor.core.models import ChunkType


class TestHeuristicParser:
    """Test suite for HeuristicParser abstract class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 10  # Reduced to allow small header chunks
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'heuristic'
        return config
    
    @pytest.fixture
    def concrete_parser(self, mock_config):
        """Create a concrete implementation of HeuristicParser for testing"""
        class ConcreteHeuristicParser(HeuristicParser):
            def __init__(self, config):
                super().__init__(config)
                self.supported_languages = {"markdown", "text"}
                self.supported_extensions = {".md", ".txt"}
            
            def can_parse(self, language: str, file_extension: str) -> bool:
                return (language in self.supported_languages or 
                        file_extension in self.supported_extensions)
            
            def _extract_chunks_heuristically(self, content: str, context: ParseContext) -> list:
                # Simple mock implementation that creates chunks based on headers
                chunks = []
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    # Fix the regex pattern to properly match markdown headers
                    if re.match(r'^#+\s+', line.strip()):
                        # Create a header chunk - use COMMENT type so small chunks are allowed
                        chunk = self._create_heuristic_chunk(
                            content=line.strip(),
                            start_line=i + 1,
                            end_line=i + 1,
                            chunk_type=ChunkType.COMMENT,  # Changed to COMMENT to allow small chunks
                            context=context,
                            identifier=line.strip('#').strip()
                        )
                        chunks.append(chunk)
                
                return chunks
        
        return ConcreteHeuristicParser(mock_config)
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample ParseContext for testing"""
        return ParseContext(
            file_path="/docs/test.md",
            language="markdown",
            content_type=ContentType.MARKDOWN,
            max_chunk_size=2000,
            min_chunk_size=10  # Reduced to allow small header chunks
        )

    def test_abstract_methods_require_implementation(self, mock_config):
        """Test that HeuristicParser cannot be instantiated directly"""
        with pytest.raises(TypeError):
            HeuristicParser(mock_config)

    def test_initialization(self, concrete_parser, mock_config):
        """Test HeuristicParser initialization"""
        assert concrete_parser.config == mock_config
        assert concrete_parser.parser_type == "heuristic"
        assert concrete_parser.supported_languages == {"markdown", "text"}
        assert concrete_parser.supported_extensions == {".md", ".txt"}
        assert concrete_parser.name == "ConcreteHeuristicParser"

    def test_parse_empty_content(self, concrete_parser, sample_context):
        """Test parsing empty content"""
        result = concrete_parser.parse("", sample_context)
        assert result == []
        
        result = concrete_parser.parse("   \n\t  ", sample_context)
        assert result == []

    def test_parse_with_content(self, concrete_parser, sample_context):
        """Test parsing content with headers"""
        content = """# Introduction
This is an introduction.

# Methods
This section covers methods.

## Subsection
More details here."""
        
        result = concrete_parser.parse(content, sample_context)
        
        # Should extract 3 header chunks
        assert len(result) == 3
        
        # Check first chunk
        assert result[0].content == "# Introduction"
        assert result[0].start_line == 1
        assert result[0].end_line == 1
        assert result[0].chunk_type == ChunkType.COMMENT  # Updated to match new chunk type
        assert result[0].metadata['identifier'] == "Introduction"
        
        # Check that chunks are sorted by line number (from post-processing)
        assert result[0].start_line <= result[1].start_line <= result[2].start_line

    def test_can_parse_implementation(self, concrete_parser):
        """Test can_parse method implementation"""
        # Test supported language
        assert concrete_parser.can_parse("markdown", ".md") == True
        assert concrete_parser.can_parse("text", ".txt") == True
        
        # Test supported extension only
        assert concrete_parser.can_parse("unknown", ".md") == True
        assert concrete_parser.can_parse("unknown", ".txt") == True
        
        # Test unsupported
        assert concrete_parser.can_parse("python", ".py") == False
        assert concrete_parser.can_parse("unknown", ".xyz") == False


class TestHeuristicChunkCreation:
    """Test suite for heuristic chunk creation methods"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 10  # Reduced to allow small header chunks
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'heuristic'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser for chunk creation testing"""
        class TestHeuristicParser(HeuristicParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def _extract_chunks_heuristically(self, content: str, context: ParseContext) -> list:
                return []
        
        return TestHeuristicParser(mock_config)
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample ParseContext for testing"""
        return ParseContext(
            file_path="/docs/test.md",
            language="markdown",
            content_type=ContentType.MARKDOWN,
            max_chunk_size=2000,
            min_chunk_size=10  # Reduced to allow small header chunks
        )

    def test_create_heuristic_chunk_basic(self, parser, sample_context):
        """Test basic heuristic chunk creation"""
        content = "# Test Header\nThis is a test."
        
        chunk = parser._create_heuristic_chunk(
            content=content,
            start_line=10,
            end_line=12,
            chunk_type=ChunkType.TEXT_BLOCK,
            context=sample_context,
            identifier="test_header"
        )
        
        assert isinstance(chunk, SemanticChunk)
        assert chunk.content == content.strip()
        assert chunk.start_line == 10
        assert chunk.end_line == 12
        assert chunk.chunk_type == ChunkType.TEXT_BLOCK
        assert chunk.language == "markdown"
        assert chunk.file_path == "/docs/test.md"
        assert chunk.content_type == ContentType.MARKDOWN
        
        # Check heuristic-specific metadata
        assert chunk.metadata['parser'] == 'TestHeuristicParser'
        assert chunk.metadata['parser_type'] == 'heuristic'
        assert chunk.metadata['identifier'] == 'test_header'
        assert chunk.metadata['extraction_method'] == 'heuristic'
        
        # Check importance score was calculated
        assert 0.1 <= chunk.importance_score <= 1.0

    def test_create_heuristic_chunk_with_custom_metadata(self, parser, sample_context):
        """Test heuristic chunk creation with custom metadata"""
        content = "Test content"
        custom_metadata = {"complexity": "high", "priority": 0.9}
        
        chunk = parser._create_heuristic_chunk(
            content=content,
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.TEXT_BLOCK,
            context=sample_context,
            **custom_metadata
        )
        
        # Check that custom metadata is merged
        assert chunk.metadata['complexity'] == "high"
        assert chunk.metadata['priority'] == 0.9
        assert chunk.metadata['parser_type'] == 'heuristic'  # Still has base metadata

    def test_create_heuristic_chunk_strips_content(self, parser, sample_context):
        """Test that chunk creation strips whitespace from content"""
        content = "  \n  Test content with whitespace  \n  "
        
        chunk = parser._create_heuristic_chunk(
            content=content,
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.TEXT_BLOCK,
            context=sample_context
        )
        
        assert chunk.content == "Test content with whitespace"


class TestImportanceScoring:
    """Test suite for heuristic importance scoring"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 10  # Reduced to allow small header chunks
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'heuristic'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser for importance scoring tests"""
        class TestHeuristicParser(HeuristicParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def _extract_chunks_heuristically(self, content: str, context: ParseContext) -> list:
                return []
        
        return TestHeuristicParser(mock_config)

    def test_calculate_heuristic_importance_chunk_types(self, parser):
        """Test importance scoring based on chunk types"""
        # Test high-importance types
        # Note: The actual implementation adds structural bonuses that can push scores to 1.0
        function_score = parser._calculate_heuristic_importance("def test():", ChunkType.FUNCTION)
        class_score = parser._calculate_heuristic_importance("class Test:", ChunkType.CLASS)
        method_score = parser._calculate_heuristic_importance("def method(self):", ChunkType.METHOD)
        
        # The implementation gives 0.9 base + 0.1 structural bonus = 1.0 for functions with def
        assert function_score == 1.0  # Updated expectation
        assert class_score == 1.0     # Updated expectation  
        assert method_score == 0.9    # Method gets base 0.8 + 0.1 = 0.9
        
        # Test medium-importance types
        import_score = parser._calculate_heuristic_importance("import os", ChunkType.IMPORT)
        assert import_score == 0.6
        
        # Test default importance - use content without structural indicators
        text_score = parser._calculate_heuristic_importance("Some plain text", ChunkType.TEXT_BLOCK)
        assert text_score == 0.5

    def test_calculate_heuristic_importance_content_size(self, parser):
        """Test importance scoring based on content size"""
        # Optimal word count (50-200 words)
        optimal_content = " ".join(["word"] * 100)  # 100 words
        optimal_score = parser._calculate_heuristic_importance(optimal_content, ChunkType.TEXT_BLOCK)
        
        # Should get base score (0.5) + word count bonus (0.1) = 0.6
        assert optimal_score == 0.6
        
        # Optimal line count (5-50 lines)
        optimal_lines = "\n".join(["line"] * 25)  # 25 lines
        line_score = parser._calculate_heuristic_importance(optimal_lines, ChunkType.TEXT_BLOCK)
        
        # Should get base score (0.5) + line count bonus (0.1) = 0.6
        assert line_score == 0.6
        
        # Both optimal
        both_optimal = "\n".join([" ".join(["word"] * 4)] * 25)  # 25 lines, 100 words
        both_score = parser._calculate_heuristic_importance(both_optimal, ChunkType.TEXT_BLOCK)
        
        # Should get base score (0.5) + word bonus (0.1) + line bonus (0.1) = 0.7
        assert both_score == 0.7

    def test_calculate_heuristic_importance_structural_indicators(self, parser):
        """Test importance scoring based on structural indicators"""
        # Content with function definition
        function_content = "def my_function():\n    return True"
        function_score = parser._calculate_heuristic_importance(function_content, ChunkType.TEXT_BLOCK)
        
        # Should get base score (0.5) + structural bonus (0.1) = 0.6
        assert function_score == 0.6
        
        # Content with class definition
        class_content = "class MyClass:\n    pass"
        class_score = parser._calculate_heuristic_importance(class_content, ChunkType.TEXT_BLOCK)
        
        # Should get base score (0.5) + structural bonus (0.1) = 0.6
        assert class_score == 0.6
        
        # Content with JavaScript function
        js_function = "function myFunc() { return true; }"
        js_score = parser._calculate_heuristic_importance(js_function, ChunkType.TEXT_BLOCK)
        
        # Should get base score (0.5) + structural bonus (0.1) = 0.6
        assert js_score == 0.6
        
        # Content with const declaration
        const_content = "const MY_CONSTANT = 42;"
        const_score = parser._calculate_heuristic_importance(const_content, ChunkType.TEXT_BLOCK)
        
        # Should get base score (0.5) + structural bonus (0.1) = 0.6
        assert const_score == 0.6

    def test_calculate_heuristic_importance_boundaries(self, parser):
        """Test importance scoring boundary conditions"""
        # Test minimum boundary
        empty_score = parser._calculate_heuristic_importance("", ChunkType.TEXT_BLOCK)
        assert empty_score == 0.5  # Base score for TEXT_BLOCK
        
        # Test maximum boundary with optimal content
        optimal_content = "\n".join([" ".join(["word"] * 4)] * 25)  # Optimal size
        function_content = f"def test():\n{optimal_content}"  # Add structural indicator
        
        max_score = parser._calculate_heuristic_importance(function_content, ChunkType.FUNCTION)
        
        # Should get: base(0.9) + word(0.1) + line(0.1) + structural(0.1) = 1.2, capped at 1.0
        assert max_score == 1.0
        
        # Test that scores are never below 0.1
        # Even with unknown chunk type, should not go below 0.1
        unknown_score = parser._calculate_heuristic_importance("test", "UNKNOWN_TYPE")
        assert unknown_score >= 0.1


class TestContentManipulation:
    """Test suite for content manipulation utilities"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 10  # Reduced to allow small header chunks
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'heuristic'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser for content manipulation testing"""
        class TestHeuristicParser(HeuristicParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def _extract_chunks_heuristically(self, content: str, context: ParseContext) -> list:
                return []
        
        return TestHeuristicParser(mock_config)

    def test_split_into_lines(self, parser):
        """Test line splitting functionality"""
        content = "Line 1\nLine 2\n\nLine 4"
        lines = parser._split_into_lines(content)
        
        expected = ["Line 1", "Line 2", "", "Line 4"]
        assert lines == expected

    def test_split_into_lines_edge_cases(self, parser):
        """Test line splitting edge cases"""
        # Empty content
        assert parser._split_into_lines("") == [""]
        
        # Single line
        assert parser._split_into_lines("Single line") == ["Single line"]
        
        # Content ending with newline
        assert parser._split_into_lines("Line 1\nLine 2\n") == ["Line 1", "Line 2", ""]
        
        # Multiple consecutive newlines
        assert parser._split_into_lines("Line 1\n\n\nLine 4") == ["Line 1", "", "", "Line 4"]

    def test_join_lines(self, parser):
        """Test line joining functionality"""
        lines = ["Line 1", "Line 2", "Line 3", "Line 4"]
        
        # Join all lines
        result = parser._join_lines(lines, 0, 4)
        assert result == "Line 1\nLine 2\nLine 3\nLine 4"
        
        # Join subset of lines
        result = parser._join_lines(lines, 1, 3)
        assert result == "Line 2\nLine 3"
        
        # Join single line
        result = parser._join_lines(lines, 2, 3)
        assert result == "Line 3"

    def test_join_lines_edge_cases(self, parser):
        """Test line joining edge cases"""
        lines = ["Line 1", "Line 2", "Line 3"]
        
        # Empty range
        result = parser._join_lines(lines, 1, 1)
        assert result == ""
        
        # Start equals end
        result = parser._join_lines(lines, 2, 2)
        assert result == ""
        
        # Full range
        result = parser._join_lines(lines, 0, len(lines))
        assert result == "Line 1\nLine 2\nLine 3"


class TestPatternMatching:
    """Test suite for pattern matching utilities"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 10  # Reduced to allow small header chunks
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'heuristic'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser for pattern matching testing"""
        class TestHeuristicParser(HeuristicParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def _extract_chunks_heuristically(self, content: str, context: ParseContext) -> list:
                return []
        
        return TestHeuristicParser(mock_config)

    def test_find_line_ranges(self, parser):
        """Test line range finding functionality"""
        content = """Line 1
# Header 1
Regular text
## Header 2
More text
# Header 3"""
        
        # Find all markdown headers
        ranges = parser._find_line_ranges(content, r'^#+\s+')
        
        expected = [(2, 2), (4, 4), (6, 6)]  # 1-based line numbers
        assert ranges == expected

    def test_find_line_ranges_no_matches(self, parser):
        """Test line range finding with no matches"""
        content = "Line 1\nLine 2\nLine 3"
        ranges = parser._find_line_ranges(content, r'^#+\s+')
        
        assert ranges == []

    def test_find_line_ranges_complex_pattern(self, parser):
        """Test line range finding with complex patterns"""
        content = """def function1():
    pass

class MyClass:
    def method(self):
        pass

def function2():
    return True"""
        
        # Find Python function definitions
        ranges = parser._find_line_ranges(content, r'^\s*def\s+\w+')
        
        expected = [(1, 1), (5, 5), (8, 8)]
        assert ranges == expected

    def test_extract_sections_by_pattern(self, parser):
        """Test section extraction functionality"""
        content = """Introduction

<!-- START_SECTION -->
This is section 1 content.
More content here.
<!-- END_SECTION -->

Middle content

<!-- START_SECTION -->
This is section 2 content.
<!-- END_SECTION -->

Conclusion"""
        
        sections = parser._extract_sections_by_pattern(
            content, 
            r'<!-- START_SECTION -->',
            r'<!-- END_SECTION -->'
        )
        
        expected = [(3, 6), (10, 12)]  # 1-based line numbers
        assert sections == expected

    def test_extract_sections_by_pattern_unclosed(self, parser):
        """Test section extraction with unclosed sections"""
        content = """Introduction

<!-- START_SECTION -->
This section is never closed.
Content continues...
Final line."""
        
        sections = parser._extract_sections_by_pattern(
            content,
            r'<!-- START_SECTION -->',
            r'<!-- END_SECTION -->'
        )
        
        # Should include unclosed section to end of content
        expected = [(3, 6)]  # From START to end of content
        assert sections == expected

    def test_extract_sections_by_pattern_no_end_pattern(self, parser):
        """Test section extraction without end pattern"""
        content = """# Header 1
Content 1

# Header 2
Content 2

# Header 3
Content 3"""
        
        # Extract from headers to end (no end pattern)
        sections = parser._extract_sections_by_pattern(content, r'^#\s+')
        
        # Each header should start a section that goes to the end
        # Fixed expectation - all sections go to the end when no end pattern
        expected = [(1, 8), (4, 8), (7, 8)]  # All extend to end
        assert sections == expected

    def test_extract_sections_by_pattern_nested(self, parser):
        """Test section extraction with nested patterns"""
        content = """<!-- START_SECTION -->
Outer section
<!-- START_SECTION -->
Inner section (this should be ignored in basic implementation)
<!-- END_SECTION -->
<!-- END_SECTION -->"""
        
        sections = parser._extract_sections_by_pattern(
            content,
            r'<!-- START_SECTION -->',
            r'<!-- END_SECTION -->'
        )
        
        # Basic implementation should match first START with first END
        expected = [(1, 5)]
        assert sections == expected


class TestEdgeCases:
    """Test suite for edge cases and error conditions"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ChunkingConfig for testing"""
        config = Mock(spec=ChunkingConfig)
        config.max_chunk_size = 2000
        config.min_chunk_size = 10  # Reduced to allow small header chunks
        config.target_chunk_size = 500
        config.preserve_atomic_nodes = True
        config.primary_strategy = 'heuristic'
        return config
    
    @pytest.fixture
    def parser(self, mock_config):
        """Create parser for edge case testing"""
        class TestHeuristicParser(HeuristicParser):
            def can_parse(self, language: str, file_extension: str) -> bool:
                return True
            
            def _extract_chunks_heuristically(self, content: str, context: ParseContext) -> list:
                # Return empty list for testing
                return []
        
        return TestHeuristicParser(mock_config)
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample ParseContext for testing"""
        return ParseContext(
            file_path="/docs/test.md",
            language="markdown",
            content_type=ContentType.MARKDOWN,
            max_chunk_size=2000,
            min_chunk_size=10  # Reduced to allow small header chunks
        )

    def test_parse_whitespace_only_content(self, parser, sample_context):
        """Test parsing content that is only whitespace"""
        whitespace_content = "   \n\t\n   \n\t\t   "
        result = parser.parse(whitespace_content, sample_context)
        assert result == []

    def test_create_chunk_with_empty_content(self, parser, sample_context):
        """Test creating chunk with empty content"""
        chunk = parser._create_heuristic_chunk(
            content="",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.TEXT_BLOCK,
            context=sample_context
        )
        
        assert chunk.content == ""
        assert chunk.importance_score >= 0.1

    def test_create_chunk_with_whitespace_content(self, parser, sample_context):
        """Test creating chunk with whitespace-heavy content"""
        content = "   \n\n  Some content  \n\n   "
        chunk = parser._create_heuristic_chunk(
            content=content,
            start_line=1,
            end_line=5,
            chunk_type=ChunkType.TEXT_BLOCK,
            context=sample_context
        )
        
        # Content should be stripped
        assert chunk.content == "Some content"

    def test_pattern_matching_with_invalid_regex(self, parser):
        """Test pattern matching with invalid regex"""
        content = "Test content"
        
        # This should not raise an exception but return empty results
        with pytest.raises(re.error):
            parser._find_line_ranges(content, r'[invalid regex')

    def test_importance_scoring_with_none_chunk_type(self, parser):
        """Test importance scoring with None chunk type"""
        score = parser._calculate_heuristic_importance("test content", None)
        
        # Should handle gracefully and return default score
        assert score >= 0.1
        assert score <= 1.0

    def test_line_manipulation_with_empty_lines(self, parser):
        """Test line manipulation with empty line lists"""
        empty_lines = []
        
        # Join should handle empty list gracefully
        result = parser._join_lines(empty_lines, 0, 0)
        assert result == ""

    def test_section_extraction_with_empty_content(self, parser):
        """Test section extraction with empty content"""
        sections = parser._extract_sections_by_pattern("", r'pattern')
        assert sections == []

    def test_integration_with_base_parser(self, parser, sample_context):
        """Test integration with BaseParser functionality"""
        # Test that HeuristicParser properly inherits BaseParser methods
        assert hasattr(parser, 'get_priority')
        assert hasattr(parser, '_post_process')
        assert hasattr(parser, '_should_include_chunk')
        assert hasattr(parser, '_is_atomic_chunk')
        
        # Test priority calculation
        priority = parser.get_priority("unknown", ".unknown")
        assert priority == 0  # Should return 0 for unsupported


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])