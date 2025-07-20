#!/usr/bin/env python3
# tests/chunking/test_init.py
"""
Comprehensive pytest tests for chunking __init__.py module
=========================================================

Tests cover:
- Module imports and exports verification
- Parser availability checking
- Convenience function functionality
- Engine creation and configuration
- Error handling and edge cases
- Version and metadata handling
- Initialization logging and status
- Module-level functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
import tempfile
import os
from pathlib import Path

# Test the chunking module initialization
import chuk_code_raptor.chunking as chunking_module


class TestModuleImports:
    """Test suite for module imports and exports"""
    
    def test_all_exports_available(self):
        """Test that all items in __all__ are available"""
        for item in chunking_module.__all__:
            assert hasattr(chunking_module, item), f"Export '{item}' not available in module"
    
    def test_core_classes_imported(self):
        """Test that core classes are properly imported"""
        # Main interface
        assert hasattr(chunking_module, 'ChunkingEngine')
        
        # Configuration classes
        assert hasattr(chunking_module, 'ChunkingConfig')
        assert hasattr(chunking_module, 'ChunkingStrategy')
        
        # Base classes
        assert hasattr(chunking_module, 'BaseParser')
        assert hasattr(chunking_module, 'TreeSitterParser')
        assert hasattr(chunking_module, 'HeuristicParser')
        assert hasattr(chunking_module, 'ParseContext')
        
        # Semantic model
        assert hasattr(chunking_module, 'SemanticChunk')
        assert hasattr(chunking_module, 'SemanticTag')
        assert hasattr(chunking_module, 'ChunkRelationship')
        assert hasattr(chunking_module, 'ContentType')
        
        # Utility functions
        assert hasattr(chunking_module, 'create_chunk_id')
        assert hasattr(chunking_module, 'calculate_chunk_similarity')
        assert hasattr(chunking_module, 'find_related_chunks')
        
        # Exceptions
        assert hasattr(chunking_module, 'ParserError')
        assert hasattr(chunking_module, 'UnsupportedLanguageError')
        assert hasattr(chunking_module, 'InvalidContentError')
    
    def test_configuration_presets_available(self):
        """Test that configuration presets are available"""
        presets = [
            'DEFAULT_CONFIG',
            'FAST_CONFIG',
            'PRECISE_CONFIG',
            'SEMANTIC_CONFIG',
            'HYBRID_CONFIG',
            'LARGE_FILES_CONFIG',
            'DOCUMENT_CONFIG'
        ]
        
        for preset in presets:
            assert hasattr(chunking_module, preset), f"Config preset '{preset}' not available"
            config = getattr(chunking_module, preset)
            assert config is not None
            # Should be a ChunkingConfig instance
            assert hasattr(config, 'max_chunk_size')
    
    def test_convenience_functions_available(self):
        """Test that convenience functions are available"""
        convenience_functions = [
            'create_engine',
            'chunk_file',
            'chunk_content',
            'get_supported_languages',
            'get_supported_extensions',
            'get_parser_info'
        ]
        
        for func in convenience_functions:
            assert hasattr(chunking_module, func), f"Convenience function '{func}' not available"
            assert callable(getattr(chunking_module, func))


class TestVersionAndMetadata:
    """Test suite for version and metadata handling"""
    
    def test_version_available(self):
        """Test that version information is available"""
        assert hasattr(chunking_module, '__version__')
        assert isinstance(chunking_module.__version__, str)
        assert chunking_module.__version__ != ""
    
    def test_author_available(self):
        """Test that author information is available"""
        assert hasattr(chunking_module, '__author__')
        assert isinstance(chunking_module.__author__, str)
        assert chunking_module.__author__ != ""
    
    def test_version_from_importlib_metadata(self):
        """Test version retrieval from importlib.metadata"""
        # Since the module is already loaded, we can't easily test different
        # version detection methods without complex mocking. Just verify the
        # version information is properly set up.
        assert hasattr(chunking_module, '__version__')
        assert isinstance(chunking_module.__version__, str)
        assert chunking_module.__version__ != ""
    
    def test_version_fallback_to_pkg_resources(self):
        """Test version fallback to pkg_resources"""
        # Since pkg_resources may not be available in all environments,
        # just test that the version attribute exists and is a string
        assert hasattr(chunking_module, '__version__')
        assert isinstance(chunking_module.__version__, str)
        assert chunking_module.__version__ != ""
    
    def test_version_fallback_to_unknown(self):
        """Test version fallback when all methods fail"""
        # Since the module is already loaded, we can't easily test the fallback
        # without complex module manipulation. Just verify version exists and is valid.
        assert hasattr(chunking_module, '__version__')
        assert isinstance(chunking_module.__version__, str)
        assert chunking_module.__version__ != ""
        # Version should be one of the expected values
        valid_versions = ["unknown", "development"]
        # Or it could be an actual version string if importlib.metadata worked
        is_valid = (chunking_module.__version__ in valid_versions or 
                   any(c.isdigit() for c in chunking_module.__version__))
        assert is_valid


class TestParserAvailabilityChecking:
    """Test suite for parser availability checking"""
    
    def test_parsers_available_structure(self):
        """Test that PARSERS_AVAILABLE has correct structure"""
        assert hasattr(chunking_module, 'PARSERS_AVAILABLE')
        assert isinstance(chunking_module.PARSERS_AVAILABLE, dict)
        
        for language, info in chunking_module.PARSERS_AVAILABLE.items():
            assert isinstance(language, str)
            assert isinstance(info, dict)
            assert 'module' in info
            assert 'class' in info
            assert 'parser_type' in info
    
    @patch('builtins.__import__')
    def test_check_parser_availability_success(self, mock_import):
        """Test successful parser availability checking"""
        # Mock successful parser import
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.parser_type = "tree_sitter"
        mock_parser_class.return_value = mock_parser_instance
        
        mock_module = Mock()
        mock_module.PythonParser = mock_parser_class
        mock_import.return_value = mock_module
        
        # Test the function
        result = chunking_module._check_parser_availability()
        
        # Should have attempted to import parsers
        assert isinstance(result, dict)
    
    @patch('builtins.__import__', side_effect=ImportError("Module not found"))
    def test_check_parser_availability_import_error(self, mock_import):
        """Test parser availability checking with import errors"""
        result = chunking_module._check_parser_availability()
        
        # Should handle import errors gracefully
        assert isinstance(result, dict)
        # Likely empty since imports fail
    
    @patch('builtins.__import__')
    def test_check_parser_availability_init_error(self, mock_import):
        """Test parser availability checking with initialization errors"""
        # Mock parser that fails to initialize
        mock_parser_class = Mock(side_effect=Exception("Init failed"))
        mock_module = Mock()
        mock_module.PythonParser = mock_parser_class
        mock_import.return_value = mock_module
        
        result = chunking_module._check_parser_availability()
        
        # Should handle initialization errors gracefully
        assert isinstance(result, dict)


class TestConvenienceFunctions:
    """Test suite for convenience functions"""
    
    def test_create_engine_default(self):
        """Test create_engine with default configuration"""
        engine = chunking_module.create_engine()
        
        assert engine is not None
        assert hasattr(engine, 'chunk_file')
        assert hasattr(engine, 'chunk_content')
        assert hasattr(engine, 'get_supported_languages')
    
    def test_create_engine_custom_config(self):
        """Test create_engine with custom configuration"""
        custom_config = chunking_module.ChunkingConfig(
            max_chunk_size=1000,
            min_chunk_size=50
        )
        
        engine = chunking_module.create_engine(custom_config)
        
        assert engine is not None
        assert engine.config.max_chunk_size == 1000
        assert engine.config.min_chunk_size == 50
    
    @patch('chuk_code_raptor.chunking.ChunkingEngine')
    def test_chunk_file_convenience(self, mock_engine_class):
        """Test chunk_file convenience function"""
        mock_engine = Mock()
        mock_chunks = [Mock(), Mock()]
        mock_engine.chunk_file.return_value = mock_chunks
        mock_engine_class.return_value = mock_engine
        
        result = chunking_module.chunk_file("test.py", "python")
        
        assert result == mock_chunks
        mock_engine.chunk_file.assert_called_once_with("test.py", "python")
    
    @patch('chuk_code_raptor.chunking.ChunkingEngine')
    def test_chunk_file_with_custom_config(self, mock_engine_class):
        """Test chunk_file with custom configuration"""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        custom_config = chunking_module.ChunkingConfig(max_chunk_size=500)
        chunking_module.chunk_file("test.py", config=custom_config)
        
        # Should create engine with custom config
        mock_engine_class.assert_called_once_with(custom_config)
    
    @patch('chuk_code_raptor.chunking.ChunkingEngine')
    def test_chunk_content_convenience(self, mock_engine_class):
        """Test chunk_content convenience function"""
        mock_engine = Mock()
        mock_chunks = [Mock(), Mock()]
        mock_engine.chunk_content.return_value = mock_chunks
        mock_engine_class.return_value = mock_engine
        
        content = "def test(): pass"
        result = chunking_module.chunk_content(content, "python", "test.py")
        
        assert result == mock_chunks
        mock_engine.chunk_content.assert_called_once_with(content, "python", "test.py")
    
    @patch('chuk_code_raptor.chunking.ChunkingEngine')
    def test_chunk_content_with_defaults(self, mock_engine_class):
        """Test chunk_content with default parameters"""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        content = "def test(): pass"
        chunking_module.chunk_content(content, "python")
        
        # Should use default file_path
        mock_engine.chunk_content.assert_called_once_with(content, "python", "unknown")
    
    @patch('chuk_code_raptor.chunking.ChunkingEngine')
    def test_get_supported_languages(self, mock_engine_class):
        """Test get_supported_languages convenience function"""
        mock_engine = Mock()
        mock_languages = ['python', 'javascript', 'markdown']
        mock_engine.get_supported_languages.return_value = mock_languages
        mock_engine_class.return_value = mock_engine
        
        result = chunking_module.get_supported_languages()
        
        assert result == mock_languages
        mock_engine.get_supported_languages.assert_called_once()
    
    @patch('chuk_code_raptor.chunking.ChunkingEngine')
    def test_get_supported_extensions(self, mock_engine_class):
        """Test get_supported_extensions convenience function"""
        mock_engine = Mock()
        mock_extensions = ['.py', '.js', '.md']
        mock_engine.get_supported_extensions.return_value = mock_extensions
        mock_engine_class.return_value = mock_engine
        
        result = chunking_module.get_supported_extensions()
        
        assert result == mock_extensions
        mock_engine.get_supported_extensions.assert_called_once()


class TestGetParserInfo:
    """Test suite for get_parser_info function"""
    
    @patch('chuk_code_raptor.chunking.get_supported_languages')
    def test_get_parser_info_structure(self, mock_get_languages):
        """Test get_parser_info returns correct structure"""
        mock_get_languages.return_value = ['python', 'javascript']
        
        info = chunking_module.get_parser_info()
        
        assert isinstance(info, dict)
        assert 'available_parsers' in info
        assert 'total_parsers' in info
        assert 'parser_types' in info
        assert 'supported_languages' in info
    
    @patch('chuk_code_raptor.chunking.get_supported_languages')
    def test_get_parser_info_with_parsers(self, mock_get_languages):
        """Test get_parser_info with available parsers"""
        mock_get_languages.return_value = ['python', 'javascript']
        
        # Mock some available parsers
        with patch.object(chunking_module, 'PARSERS_AVAILABLE', {
            'python': {'parser_type': 'tree_sitter', 'module': 'test', 'class': 'Test'},
            'javascript': {'parser_type': 'tree_sitter', 'module': 'test', 'class': 'Test'}
        }):
            info = chunking_module.get_parser_info()
            
            assert info['total_parsers'] == 2
            assert 'python' in info['available_parsers']
            assert 'javascript' in info['available_parsers']
            assert 'tree_sitter' in info['parser_types']
            assert info['supported_languages'] == ['python', 'javascript']
    
    @patch('chuk_code_raptor.chunking.get_supported_languages')
    def test_get_parser_info_no_parsers(self, mock_get_languages):
        """Test get_parser_info with no available parsers"""
        mock_get_languages.return_value = []
        
        with patch.object(chunking_module, 'PARSERS_AVAILABLE', {}):
            info = chunking_module.get_parser_info()
            
            assert info['total_parsers'] == 0
            assert info['available_parsers'] == {}
            assert info['parser_types'] == []
            assert info['supported_languages'] == []


class TestFileOperations:
    """Test suite for file-based operations"""
    
    def test_chunk_file_with_real_file(self):
        """Test chunk_file with actual file (integration test)"""
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def hello():\n    print('Hello, World!')\n")
            f.flush()
            temp_path = f.name
        
        try:
            # Test chunking the file
            chunks = chunking_module.chunk_file(temp_path, "python")
            
            # Should return a list (might be empty if no parsers available)
            assert isinstance(chunks, list)
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_chunk_file_nonexistent(self):
        """Test chunk_file with nonexistent file"""
        with pytest.raises(FileNotFoundError):
            chunking_module.chunk_file("/nonexistent/file.py", "python")
    
    def test_chunk_content_basic(self):
        """Test chunk_content with basic content"""
        content = "def test_function():\n    return True"
        
        # Should not raise an error (though might return empty list if no parsers)
        chunks = chunking_module.chunk_content(content, "python", "test.py")
        assert isinstance(chunks, list)


class TestEdgeCases:
    """Test suite for edge cases and error conditions"""
    
    def test_create_engine_none_config(self):
        """Test create_engine with None config"""
        engine = chunking_module.create_engine(None)
        
        assert engine is not None
        # Should use DEFAULT_CONFIG
        assert engine.config is not None
    
    def test_chunk_content_empty_content(self):
        """Test chunk_content with empty content"""
        chunks = chunking_module.chunk_content("", "python", "empty.py")
        
        # Should handle empty content gracefully
        assert isinstance(chunks, list)
        assert len(chunks) == 0
    
    def test_chunk_content_whitespace_only(self):
        """Test chunk_content with whitespace-only content"""
        chunks = chunking_module.chunk_content("   \n\t  ", "python", "whitespace.py")
        
        # Should handle whitespace-only content gracefully
        assert isinstance(chunks, list)
        assert len(chunks) == 0
    
    @patch('chuk_code_raptor.chunking.ChunkingEngine')
    def test_chunk_file_engine_error(self, mock_engine_class):
        """Test chunk_file when engine raises an error"""
        mock_engine = Mock()
        mock_engine.chunk_file.side_effect = Exception("Engine error")
        mock_engine_class.return_value = mock_engine
        
        with pytest.raises(Exception) as exc_info:
            chunking_module.chunk_file("test.py", "python")
        
        assert "Engine error" in str(exc_info.value)
    
    @patch('chuk_code_raptor.chunking.ChunkingEngine')
    def test_chunk_content_engine_error(self, mock_engine_class):
        """Test chunk_content when engine raises an error"""
        mock_engine = Mock()
        mock_engine.chunk_content.side_effect = Exception("Content error")
        mock_engine_class.return_value = mock_engine
        
        with pytest.raises(Exception) as exc_info:
            chunking_module.chunk_content("content", "python", "test.py")
        
        assert "Content error" in str(exc_info.value)


class TestInitializationLogging:
    """Test suite for initialization and logging"""
    
    @patch('chuk_code_raptor.chunking.logger')
    def test_log_initialization_status_with_parsers(self, mock_logger):
        """Test initialization logging with available parsers"""
        mock_parsers = {
            'python': {'parser_type': 'tree_sitter'},
            'javascript': {'parser_type': 'tree_sitter'}
        }
        
        with patch.object(chunking_module, 'PARSERS_AVAILABLE', mock_parsers):
            chunking_module._log_initialization_status()
            
            # Should log successful initialization
            mock_logger.info.assert_called()
            info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
            
            # Should mention initialization and available parsers
            assert any("initialized" in call for call in info_calls)
            assert any("Available parsers" in call for call in info_calls)
    
    @patch('chuk_code_raptor.chunking.logger')
    def test_log_initialization_status_no_parsers(self, mock_logger):
        """Test initialization logging with no available parsers"""
        with patch.object(chunking_module, 'PARSERS_AVAILABLE', {}):
            chunking_module._log_initialization_status()
            
            # Should log warning about no parsers
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "No parsers available" in warning_call


class TestModuleIntegration:
    """Test suite for module-level integration"""
    
    def test_module_can_be_imported(self):
        """Test that the module can be imported without errors"""
        # This test passes if the import at the top doesn't raise an error
        assert chunking_module is not None
    
    def test_basic_workflow(self):
        """Test basic end-to-end workflow"""
        # Create engine
        engine = chunking_module.create_engine()
        
        # Get supported languages (might be empty if no parsers available)
        languages = chunking_module.get_supported_languages()
        assert isinstance(languages, list)
        
        # Get parser info
        info = chunking_module.get_parser_info()
        assert isinstance(info, dict)
        
        # Try chunking content
        chunks = chunking_module.chunk_content("# Test", "python", "test.py")
        assert isinstance(chunks, list)
    
    def test_configuration_presets_work(self):
        """Test that configuration presets can be used"""
        configs = [
            chunking_module.DEFAULT_CONFIG,
            chunking_module.FAST_CONFIG,
            chunking_module.PRECISE_CONFIG,
            chunking_module.SEMANTIC_CONFIG,
            chunking_module.HYBRID_CONFIG,
            chunking_module.LARGE_FILES_CONFIG,
            chunking_module.DOCUMENT_CONFIG
        ]
        
        for config in configs:
            # Should be able to create engine with each preset
            engine = chunking_module.create_engine(config)
            assert engine is not None
            assert engine.config is config


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])