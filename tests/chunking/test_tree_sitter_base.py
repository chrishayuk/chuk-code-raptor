#!/usr/bin/env python3
# tests/chunking/test_tree_sitter_base.py
"""
Comprehensive pytest tests for TreeSitterParser base class
=========================================================

Tests cover:
- Abstract base class initialization and setup
- Tree-sitter parser initialization and error handling
- AST traversal and chunk extraction
- Node importance calculation and depth measurement
- Semantic tag assignment and dependency extraction
- Chunk creation from AST nodes
- Error handling and edge cases
- Configuration integration
- Mock tree-sitter integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

# Import the classes we're testing
from chuk_code_raptor.chunking.tree_sitter_base import TreeSitterParser
from chuk_code_raptor.chunking.config import ChunkingConfig
from chuk_code_raptor.chunking.base import ParseContext
from chuk_code_raptor.chunking.semantic_chunk import SemanticChunk, ContentType
from chuk_code_raptor.core.models import ChunkType


class MockTreeSitterNode:
    """Mock tree-sitter node for testing"""
    
    def __init__(self, node_type="function_definition", start_point=(0, 0), end_point=(5, 0),
                 start_byte=0, end_byte=100, children=None, parent=None, child_count=0):
        self.type = node_type
        self.start_point = start_point
        self.end_point = end_point
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.children = children or []
        self.parent = parent
        self.child_count = child_count or len(self.children)
        
        # Set parent references for children
        for child in self.children:
            child.parent = self


class MockTreeSitterTree:
    """Mock tree-sitter tree for testing"""
    
    def __init__(self, root_node):
        self.root_node = root_node


class MockTreeSitterParser:
    """Mock tree-sitter parser for testing"""
    
    def __init__(self, language):
        self.language = language
        self.parse_calls = []
    
    def parse(self, content_bytes):
        self.parse_calls.append(content_bytes)
        # Return a mock tree with simple structure
        root_node = MockTreeSitterNode(
            node_type="module",
            start_point=(0, 0),
            end_point=(10, 0),
            start_byte=0,
            end_byte=len(content_bytes),
            children=[
                MockTreeSitterNode(
                    node_type="function_definition",
                    start_point=(1, 0),
                    end_point=(5, 0),
                    start_byte=0,
                    end_byte=50,
                    child_count=3
                ),
                MockTreeSitterNode(
                    node_type="class_definition",
                    start_point=(6, 0),
                    end_point=(10, 0),
                    start_byte=51,
                    end_byte=len(content_bytes),
                    child_count=5
                )
            ]
        )
        return MockTreeSitterTree(root_node)


class ConcreteTreeSitterParser(TreeSitterParser):
    """Concrete implementation of TreeSitterParser for testing"""
    
    def __init__(self, config, mock_tree_sitter=None):
        self.name = "TestTreeSitterParser"
        self.supported_languages = {'test_language'}
        self.supported_extensions = {'.test'}
        self._mock_language = Mock()
        self._chunk_node_types = {
            'function_definition': ChunkType.FUNCTION,
            'class_definition': ChunkType.CLASS,
            'method_definition': ChunkType.FUNCTION,
            'async_function_definition': ChunkType.FUNCTION,
        }
        self._mock_tree_sitter = mock_tree_sitter
        super().__init__(config)
    
    def _initialize_tree_sitter(self):
        """Initialize tree-sitter parser with mocking support"""
        try:
            if self._mock_tree_sitter:
                # Use provided mock for testing
                tree_sitter = self._mock_tree_sitter
            else:
                # Normal import for production - simulate ImportError for testing
                raise ImportError("tree-sitter not found")
            
            self.language = self._get_tree_sitter_language()
            self.parser = tree_sitter.Parser(self.language)
            
        except ImportError:
            raise ImportError(f"tree-sitter not available for {self.name}")
        except Exception as e:
            raise
    
    def _get_tree_sitter_language(self):
        """Return mock language for testing"""
        return self._mock_language
    
    def _get_chunk_node_types(self):
        """Return chunk node types mapping"""
        return self._chunk_node_types
        """Return chunk node types mapping"""
        return self._chunk_node_types
    
    def _extract_identifier(self, node, content: str):
        """Extract identifier from node for testing"""
        if node.type in ['function_definition', 'class_definition']:
            # Mock identifier extraction - return type + position
            return f"{node.type}_{node.start_point[0]}"
        return None
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)


class TestTreeSitterParserInitialization:
    """Test suite for TreeSitterParser initialization"""
    
    def test_successful_initialization(self):
        """Test successful tree-sitter parser initialization"""
        config = ChunkingConfig()
        mock_tree_sitter = Mock()
        mock_parser = Mock()
        mock_tree_sitter.Parser.return_value = mock_parser
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        
        assert parser.parser_type == "tree_sitter"
        assert parser.parser == mock_parser
        assert parser.language == parser._mock_language
        mock_tree_sitter.Parser.assert_called_once_with(parser._mock_language)

    def test_tree_sitter_import_error(self):
        """Test initialization when tree-sitter is not available"""
        config = ChunkingConfig()
        
        # Pass None to trigger ImportError in initialization
        with pytest.raises(ImportError) as exc_info:
            ConcreteTreeSitterParser(config, None)
        
        assert "tree-sitter not available" in str(exc_info.value)

    def test_tree_sitter_initialization_error(self):
        """Test initialization when tree-sitter parser creation fails"""
        config = ChunkingConfig()
        mock_tree_sitter = Mock()
        mock_tree_sitter.Parser.side_effect = Exception("Parser creation failed")
        
        with pytest.raises(Exception) as exc_info:
            ConcreteTreeSitterParser(config, mock_tree_sitter)
        
        assert "Parser creation failed" in str(exc_info.value)

    def test_abstract_methods_implementation_required(self):
        """Test that abstract methods must be implemented"""
        config = ChunkingConfig()
        
        # Try to instantiate base class directly (should fail)
        with pytest.raises(TypeError):
            TreeSitterParser(config)


class TestTreeSitterParsing:
    """Test suite for tree-sitter parsing functionality"""
    
    @pytest.fixture
    def parser_with_mock_tree_sitter(self):
        """Create parser with mocked tree-sitter"""
        config = ChunkingConfig(min_chunk_size=10, max_chunk_size=1000)
        
        mock_tree_sitter = Mock()
        mock_parser = MockTreeSitterParser(Mock())
        mock_tree_sitter.Parser.return_value = mock_parser
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        parser.parser = mock_parser
        
        return parser, mock_parser

    def test_parse_empty_content(self, parser_with_mock_tree_sitter):
        """Test parsing empty content"""
        parser, _ = parser_with_mock_tree_sitter
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=10
        )
        
        # Empty content should return empty list
        chunks = parser.parse("", context)
        assert chunks == []
        
        # Whitespace-only content should return empty list
        chunks = parser.parse("   \n\t  ", context)
        assert chunks == []

    def test_parse_valid_content(self, parser_with_mock_tree_sitter):
        """Test parsing valid content"""
        parser, mock_ts_parser = parser_with_mock_tree_sitter
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=10
        )
        
        content = "def hello():\n    print('Hello')\n\nclass TestClass:\n    pass"
        chunks = parser.parse(content, context)
        
        # Should create chunks for function and class
        assert len(chunks) >= 1  # At least one chunk should be created
        
        # Parser should have been called with bytes
        assert len(mock_ts_parser.parse_calls) == 1
        assert mock_ts_parser.parse_calls[0] == content.encode('utf8')

    def test_parse_with_post_processing(self, parser_with_mock_tree_sitter):
        """Test that parse calls post-processing"""
        parser, _ = parser_with_mock_tree_sitter
        
        # Mock post-processing to verify it's called
        original_post_process = parser._post_process
        parser._post_process = Mock(side_effect=original_post_process)
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=10
        )
        
        content = "def test(): pass"
        parser.parse(content, context)
        
        # Post-processing should have been called
        parser._post_process.assert_called_once()


class TestASTTraversal:
    """Test suite for AST traversal functionality"""
    
    @pytest.fixture
    def parser_setup(self):
        """Setup parser for traversal testing"""
        config = ChunkingConfig(min_chunk_size=1, max_chunk_size=1000)
        
        mock_tree_sitter = Mock()
        mock_tree_sitter.Parser.return_value = Mock()
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        return parser

    def test_traverse_and_extract_basic(self, parser_setup):
        """Test basic AST traversal and extraction"""
        parser = parser_setup
        
        # Create a simple AST structure
        root_node = MockTreeSitterNode(
            node_type="module",
            children=[
                MockTreeSitterNode(
                    node_type="function_definition",
                    start_point=(1, 0),
                    end_point=(3, 0),
                    start_byte=0,
                    end_byte=30
                )
            ]
        )
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=1
        )
        
        content = "def test_function():\n    return True"
        chunks = []
        
        parser._traverse_and_extract(root_node, content, context, chunks)
        
        # Should extract the function
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.FUNCTION

    def test_traverse_with_depth_limit(self, parser_setup):
        """Test traversal respects depth limit"""
        parser = parser_setup
        
        # Create deeply nested structure
        def create_nested_node(depth):
            if depth > 35:  # Beyond the limit of 30
                return MockTreeSitterNode(node_type="statement")
            return MockTreeSitterNode(
                node_type="block",
                children=[create_nested_node(depth + 1)]
            )
        
        root_node = create_nested_node(0)
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=1
        )
        
        chunks = []
        
        # Should not crash due to deep recursion
        parser._traverse_and_extract(root_node, "content", context, chunks, depth=0)
        
        # Should handle gracefully
        assert isinstance(chunks, list)

    def test_traverse_skips_children_of_extracted_chunks(self, parser_setup):
        """Test that traversal doesn't descend into extracted chunks"""
        parser = parser_setup
        
        # Function with nested function inside
        function_node = MockTreeSitterNode(
            node_type="function_definition",
            start_point=(1, 0),
            end_point=(5, 0),
            start_byte=0,
            end_byte=100,
            children=[
                MockTreeSitterNode(
                    node_type="function_definition",  # Nested function
                    start_point=(2, 4),
                    end_point=(4, 4),
                    start_byte=20,
                    end_byte=80
                )
            ]
        )
        
        root_node = MockTreeSitterNode(
            node_type="module",
            children=[function_node]
        )
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=1
        )
        
        content = "def outer():\n    def inner():\n        pass\n    pass"
        chunks = []
        
        parser._traverse_and_extract(root_node, content, context, chunks)
        
        # Should only extract the outer function, not the nested one
        assert len(chunks) == 1
        assert chunks[0].start_line == 2  # Outer function


class TestChunkCreation:
    """Test suite for chunk creation from AST nodes"""
    
    @pytest.fixture
    def parser_setup(self):
        """Setup parser for chunk creation testing"""
        config = ChunkingConfig(min_chunk_size=1, max_chunk_size=1000, target_chunk_size=500)
        
        mock_tree_sitter = Mock()
        mock_tree_sitter.Parser.return_value = Mock()
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        return parser

    def test_create_chunk_from_node_basic(self, parser_setup):
        """Test basic chunk creation from AST node"""
        parser = parser_setup
        
        content = "def test_function():\n    return True\n"
        
        node = MockTreeSitterNode(
            node_type="function_definition",
            start_point=(0, 0),
            end_point=(2, 0),
            start_byte=0,
            end_byte=len(content.strip()),  # Use actual content length
            child_count=3
        )
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=1
        )
        
        chunk = parser._create_chunk_from_node(node, content, context, ChunkType.FUNCTION)
        
        assert chunk is not None
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.content == content.strip()  # Content should be stripped
        assert chunk.start_line == 1
        assert chunk.end_line == 3  # end_point[0] + 1
        assert chunk.language == "python"
        assert chunk.file_path == "test.py"

    def test_create_chunk_with_identifier(self, parser_setup):
        """Test chunk creation includes identifier"""
        parser = parser_setup
        
        content = "def my_function(): pass"
        
        node = MockTreeSitterNode(
            node_type="function_definition",
            start_point=(0, 0),
            end_point=(0, len(content)),
            start_byte=0,
            end_byte=len(content)
        )
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=1
        )
        
        chunk = parser._create_chunk_from_node(node, content, context, ChunkType.FUNCTION)
        
        assert chunk.metadata['identifier'] == "function_definition_0"

    def test_create_chunk_from_empty_node(self, parser_setup):
        """Test chunk creation from node with empty content"""
        parser = parser_setup
        
        node = MockTreeSitterNode(
            node_type="function_definition",
            start_point=(0, 0),
            end_point=(0, 0),
            start_byte=5,
            end_byte=5  # Empty range
        )
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=1
        )
        
        content = "     "  # Whitespace only
        chunk = parser._create_chunk_from_node(node, content, context, ChunkType.FUNCTION)
        
        # Should return None for empty content
        assert chunk is None

    def test_chunk_metadata_population(self, parser_setup):
        """Test that chunk metadata is properly populated"""
        parser = parser_setup
        
        content = "class TestClass:\n    def method(self): pass"
        
        node = MockTreeSitterNode(
            node_type="class_definition",
            start_point=(5, 0),
            end_point=(6, 0),
            start_byte=0,  # Start from beginning of content
            end_byte=len(content),  # Use actual content length
            child_count=7
        )
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=1
        )
        
        chunk = parser._create_chunk_from_node(node, content, context, ChunkType.CLASS)
        
        assert chunk is not None
        # The parser should use the 'name' attribute if available, otherwise class name
        expected_parser_name = getattr(parser, 'name', parser.__class__.__name__)
        assert chunk.metadata['parser'] == expected_parser_name
        assert chunk.metadata['parser_type'] == 'tree_sitter'
        assert chunk.metadata['node_type'] == 'class_definition'
        assert chunk.metadata['byte_range'] == (0, len(content))
        assert chunk.metadata['ast_depth'] == 0  # Root level node
        assert isinstance(chunk.metadata['identifier'], str)

    def test_chunk_with_dependency_tracking(self, parser_setup):
        """Test chunk creation with dependency tracking enabled"""
        parser = parser_setup
        
        # Mock dependency extraction
        parser._extract_dependencies = Mock()
        
        content = "def test(): pass"
        
        node = MockTreeSitterNode(
            node_type="function_definition",
            start_point=(0, 0),
            end_point=(0, len(content)),
            start_byte=0,
            end_byte=len(content)
        )
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=1,
            enable_dependency_tracking=True
        )
        
        chunk = parser._create_chunk_from_node(node, content, context, ChunkType.FUNCTION)
        
        # Dependency extraction should have been called
        parser._extract_dependencies.assert_called_once_with(chunk, node, content)


class TestImportanceCalculation:
    """Test suite for node importance calculation"""
    
    @pytest.fixture
    def parser_setup(self):
        """Setup parser for importance testing"""
        config = ChunkingConfig()
        
        mock_tree_sitter = Mock()
        mock_tree_sitter.Parser.return_value = Mock()
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        return parser

    def test_calculate_importance_function_definition(self, parser_setup):
        """Test importance calculation for function definitions"""
        parser = parser_setup
        
        node = MockTreeSitterNode(
            node_type="function_definition",
            start_byte=0,
            end_byte=100,
            child_count=5
        )
        
        importance = parser._calculate_node_importance(node)
        
        assert 0.1 <= importance <= 1.0
        assert importance > 0.5  # Functions should have high importance

    def test_calculate_importance_class_definition(self, parser_setup):
        """Test importance calculation for class definitions"""
        parser = parser_setup
        
        node = MockTreeSitterNode(
            node_type="class_definition",
            start_byte=0,
            end_byte=500,
            child_count=10
        )
        
        importance = parser._calculate_node_importance(node)
        
        assert 0.1 <= importance <= 1.0
        assert importance > 0.5  # Classes should have high importance

    def test_calculate_importance_unknown_type(self, parser_setup):
        """Test importance calculation for unknown node types"""
        parser = parser_setup
        
        node = MockTreeSitterNode(
            node_type="unknown_type",
            start_byte=0,
            end_byte=50,
            child_count=2
        )
        
        importance = parser._calculate_node_importance(node)
        
        assert 0.1 <= importance <= 1.0
        # Should use default importance of 0.5

    def test_calculate_importance_size_factor(self, parser_setup):
        """Test that importance calculation considers size"""
        parser = parser_setup
        
        small_node = MockTreeSitterNode(
            node_type="function_definition",
            start_byte=0,
            end_byte=50,  # Small
            child_count=1
        )
        
        large_node = MockTreeSitterNode(
            node_type="function_definition",
            start_byte=0,
            end_byte=2000,  # Large
            child_count=1
        )
        
        small_importance = parser._calculate_node_importance(small_node)
        large_importance = parser._calculate_node_importance(large_node)
        
        # Larger nodes should generally have higher importance
        assert large_importance >= small_importance

    def test_calculate_importance_complexity_factor(self, parser_setup):
        """Test that importance calculation considers complexity"""
        parser = parser_setup
        
        simple_node = MockTreeSitterNode(
            node_type="function_definition",
            start_byte=0,
            end_byte=100,
            child_count=1  # Simple
        )
        
        complex_node = MockTreeSitterNode(
            node_type="function_definition",
            start_byte=0,
            end_byte=100,
            child_count=15  # Complex
        )
        
        simple_importance = parser._calculate_node_importance(simple_node)
        complex_importance = parser._calculate_node_importance(complex_node)
        
        # More complex nodes should have higher importance
        assert complex_importance >= simple_importance


class TestNodeDepthCalculation:
    """Test suite for AST node depth calculation"""
    
    @pytest.fixture
    def parser_setup(self):
        """Setup parser for depth testing"""
        config = ChunkingConfig()
        
        mock_tree_sitter = Mock()
        mock_tree_sitter.Parser.return_value = Mock()
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        return parser

    def test_get_node_depth_root(self, parser_setup):
        """Test depth calculation for root node"""
        parser = parser_setup
        
        root_node = MockTreeSitterNode(node_type="module", parent=None)
        
        depth = parser._get_node_depth(root_node)
        assert depth == 0

    def test_get_node_depth_nested(self, parser_setup):
        """Test depth calculation for nested nodes"""
        parser = parser_setup
        
        # Create nested structure: root -> class -> method
        root_node = MockTreeSitterNode(node_type="module", parent=None)
        class_node = MockTreeSitterNode(node_type="class_definition", parent=root_node)
        method_node = MockTreeSitterNode(node_type="method_definition", parent=class_node)
        
        assert parser._get_node_depth(root_node) == 0
        assert parser._get_node_depth(class_node) == 1
        assert parser._get_node_depth(method_node) == 2

    def test_get_node_depth_deep_nesting(self, parser_setup):
        """Test depth calculation for deeply nested nodes"""
        parser = parser_setup
        
        # Create deeply nested structure
        current_node = MockTreeSitterNode(node_type="root", parent=None)
        target_depth = 10
        
        for i in range(target_depth):
            child_node = MockTreeSitterNode(node_type=f"level_{i}", parent=current_node)
            current_node = child_node
        
        depth = parser._get_node_depth(current_node)
        assert depth == target_depth


class TestSemanticTags:
    """Test suite for semantic tag assignment"""
    
    @pytest.fixture
    def parser_setup(self):
        """Setup parser for semantic tag testing"""
        config = ChunkingConfig(min_chunk_size=50, target_chunk_size=200, max_chunk_size=1000)
        
        mock_tree_sitter = Mock()
        mock_tree_sitter.Parser.return_value = Mock()
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        return parser

    def test_add_semantic_tags_basic(self, parser_setup):
        """Test basic semantic tag assignment"""
        parser = parser_setup
        
        node = MockTreeSitterNode(node_type="function_definition")
        chunk = SemanticChunk(
            id="test_chunk",
            file_path="test.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE,
            chunk_type=ChunkType.FUNCTION,
            importance_score=0.5
        )
        
        parser._add_semantic_tags(chunk, node, "def test(): pass")
        
        # Should have basic tags
        tags = [tag.name for tag in chunk.semantic_tags]  # Extract tag names from SemanticTag objects
        assert 'function' in tags  # chunk_type tag
        assert 'function_definition' in tags  # node_type tag

    def test_add_semantic_tags_high_importance(self, parser_setup):
        """Test semantic tags for high importance chunks"""
        parser = parser_setup
        
        node = MockTreeSitterNode(node_type="class_definition")
        chunk = SemanticChunk(
            id="test_chunk",
            file_path="test.py",
            content="class Test: pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE,
            chunk_type=ChunkType.CLASS,
            importance_score=0.9  # High importance
        )
        
        parser._add_semantic_tags(chunk, node, "class Test: pass")
        
        # Should have high importance tag
        tags = [tag.name for tag in chunk.semantic_tags]  # Extract tag names from SemanticTag objects
        assert 'high_importance' in tags

    def test_add_semantic_tags_large_chunk(self, parser_setup):
        """Test semantic tags for large chunks"""
        parser = parser_setup
        
        node = MockTreeSitterNode(node_type="function_definition")
        large_content = "def test():\n" + "    print('line')\n" * 50  # Large content
        chunk = SemanticChunk(
            id="test_chunk",
            file_path="test.py",
            content=large_content,
            start_line=1,
            end_line=51,
            content_type=ContentType.CODE,
            chunk_type=ChunkType.FUNCTION,
            importance_score=0.5
        )
        
        parser._add_semantic_tags(chunk, node, large_content)
        
        # Should have large chunk tag
        tags = [tag.name for tag in chunk.semantic_tags]  # Extract tag names from SemanticTag objects
        assert 'large_chunk' in tags

    def test_add_semantic_tags_small_chunk(self, parser_setup):
        """Test semantic tags for small chunks"""
        parser = parser_setup
        
        node = MockTreeSitterNode(node_type="function_definition")
        small_content = "def f(): pass"  # Small content (< min_chunk_size)
        chunk = SemanticChunk(
            id="test_chunk",
            file_path="test.py",
            content=small_content,
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE,
            chunk_type=ChunkType.FUNCTION,
            importance_score=0.5
        )
        
        parser._add_semantic_tags(chunk, node, small_content)
        
        # Should have small chunk tag
        tags = [tag.name for tag in chunk.semantic_tags]  # Extract tag names from SemanticTag objects
        assert 'small_chunk' in tags


class TestDependencyExtraction:
    """Test suite for dependency extraction"""
    
    @pytest.fixture
    def parser_setup(self):
        """Setup parser for dependency testing"""
        config = ChunkingConfig()
        
        mock_tree_sitter = Mock()
        mock_tree_sitter.Parser.return_value = Mock()
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        return parser

    def test_extract_dependencies_base_implementation(self, parser_setup):
        """Test base implementation of dependency extraction"""
        parser = parser_setup
        
        chunk = SemanticChunk(
            id="test_chunk",
            file_path="test.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE,
            chunk_type=ChunkType.FUNCTION
        )
        
        node = MockTreeSitterNode(node_type="function_definition")
        
        # Base implementation should not crash
        parser._extract_dependencies(chunk, node, "def test(): pass")
        
        # Base implementation is a no-op, so no specific assertions needed
        # Subclasses should override this method


class TestErrorHandling:
    """Test suite for error handling in tree-sitter operations"""
    
    def test_parse_with_invalid_utf8(self):
        """Test parsing with invalid UTF-8 content"""
        config = ChunkingConfig()
        
        mock_tree_sitter = Mock()
        mock_parser = Mock()
        mock_parser.parse.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
        mock_tree_sitter.Parser.return_value = mock_parser
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        parser.parser = mock_parser
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=10
        )
        
        # Should handle encoding errors gracefully
        with pytest.raises(UnicodeDecodeError):
            parser.parse("content", context)

    def test_parse_with_tree_sitter_error(self):
        """Test parsing when tree-sitter raises an error"""
        config = ChunkingConfig()
        mock_tree_sitter = Mock()
        mock_parser = Mock()
        mock_parser.parse.side_effect = Exception("Tree-sitter parsing error")
        mock_tree_sitter.Parser.return_value = mock_parser
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        parser.parser = mock_parser
        
        context = ParseContext(
            file_path="test.py",
            language="python",
            content_type=ContentType.CODE,
            max_chunk_size=1000,
            min_chunk_size=10
        )
        
        # Should propagate tree-sitter errors
        with pytest.raises(Exception) as exc_info:
            parser.parse("content", context)
        
        assert "Tree-sitter parsing error" in str(exc_info.value)


class TestConfigurationIntegration:
    """Test suite for configuration integration"""
    
    def test_config_used_in_chunk_creation(self):
        """Test that configuration is properly used in chunk creation"""
        config = ChunkingConfig(
            min_chunk_size=100,
            max_chunk_size=500,
            target_chunk_size=300
        )
        
        mock_tree_sitter = Mock()
        mock_tree_sitter.Parser.return_value = Mock()
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        
        # Config should be accessible
        assert parser.config.min_chunk_size == 100
        assert parser.config.max_chunk_size == 500
        assert parser.config.target_chunk_size == 300

    def test_semantic_tags_use_config(self):
        """Test that semantic tag assignment uses configuration"""
        config = ChunkingConfig(
            min_chunk_size=20,
            target_chunk_size=50,
            max_chunk_size=100
        )
        
        mock_tree_sitter = Mock()
        mock_tree_sitter.Parser.return_value = Mock()
        
        parser = ConcreteTreeSitterParser(config, mock_tree_sitter)
        
        node = MockTreeSitterNode(node_type="function_definition")
        
        # Test small chunk detection
        small_chunk = SemanticChunk(
            id="small_chunk",
            file_path="test.py",
            content="def f(): pass",  # 13 chars < 20 min_chunk_size
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE,
            chunk_type=ChunkType.FUNCTION,
            importance_score=0.5
        )
        
        parser._add_semantic_tags(small_chunk, node, small_chunk.content)
        small_tags = [tag.name for tag in small_chunk.semantic_tags]  # Extract tag names
        assert 'small_chunk' in small_tags
        
        # Test large chunk detection
        large_content = "def large_function():\n" + "    print('line')\n" * 20  # > 100 max
        large_chunk = SemanticChunk(
            id="large_chunk",
            file_path="test.py",
            content=large_content,
            start_line=1,
            end_line=21,
            content_type=ContentType.CODE,
            chunk_type=ChunkType.FUNCTION,
            importance_score=0.5
        )
        
        parser._add_semantic_tags(large_chunk, node, large_content)
        large_tags = [tag.name for tag in large_chunk.semantic_tags]  # Extract tag names
        assert 'large_chunk' in large_tags


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])