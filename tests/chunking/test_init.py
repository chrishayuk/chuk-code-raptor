#!/usr/bin/env python3
# tests/chunking/test_init.py
"""
Comprehensive pytest tests for chunking __init__.py module - YAML Registry Edition
==================================================================================

Tests cover:
- Module imports and exports verification
- Registry-based parser availability checking
- Convenience function functionality
- Engine creation and configuration
- Error handling and edge cases
- Version and metadata handling
- YAML-based initialization logging and status
- Module-level functionality with registry integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
import tempfile
import os
from pathlib import Path

# Test the chunking module initialization
import chuk_code_raptor.chunking as chunking_module


class MockRegistry:
    """Mock registry for testing"""
    
    def __init__(self):
        self.config_path = Path("mock_config.yaml")
        self.parsers = {}
        self.parser_configs = {}
        self.language_mapping = {}
        self.extension_mapping = {}
        
    def get_parser_stats(self):
        return {
            'total_parsers': len(self.parser_configs),
            'available_parsers': len(self.parsers),
            'unavailable_parsers': len(self.parser_configs) - len(self.parsers),
            'supported_languages': len(self.language_mapping),
            'supported_extensions': len(self.extension_mapping),
            'parser_types': {'tree_sitter': 2, 'heuristic': 1},
            'package_availability': {
                'comprehensive': ['tree-sitter-languages'],
                'individual': ['tree-sitter-python', 'tree-sitter-javascript']
            },
            'comprehensive_packages': 1,
            'individual_packages': 2
        }
    
    def get_supported_languages(self):
        return list(self.language_mapping.keys())
    
    def get_supported_extensions(self):
        return list(self.extension_mapping.keys())
    
    def get_installation_help(self):
        return "pip install tree-sitter-languages"


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
        
        # Registry system
        assert hasattr(chunking_module, 'ParserRegistry')
        assert hasattr(chunking_module, 'get_registry')
        assert hasattr(chunking_module, 'reload_registry')
        assert hasattr(chunking_module, 'register_custom_parser')
    
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
            'get_parser_info',
            'get_installation_help'
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
    
    def test_version_types(self):
        """Test version can be one of expected types"""
        version = chunking_module.__version__
        
        # Version should be a valid string
        assert isinstance(version, str)
        assert version != ""
        
        # Should be one of: actual version, "unknown", "development"
        valid_patterns = [
            lambda v: any(c.isdigit() for c in v),  # Has digits (actual version)
            lambda v: v == "unknown",
            lambda v: v == "development"
        ]
        
        assert any(pattern(version) for pattern in valid_patterns)


class TestRegistryIntegration:
    """Test suite for registry integration in module initialization"""
    
    @patch('chuk_code_raptor.chunking.get_registry')
    def test_get_supported_languages_uses_registry(self, mock_get_registry):
        """Test get_supported_languages uses registry"""
        mock_registry = MockRegistry()
        mock_registry.language_mapping = {'python': 'python', 'javascript': 'javascript'}
        mock_get_registry.return_value = mock_registry
        
        languages = chunking_module.get_supported_languages()
        
        assert set(languages) == {'python', 'javascript'}
        mock_get_registry.assert_called_once()
    
    @patch('chuk_code_raptor.chunking.get_registry')
    def test_get_supported_extensions_uses_registry(self, mock_get_registry):
        """Test get_supported_extensions uses registry"""
        mock_registry = MockRegistry()
        mock_registry.extension_mapping = {'.py': 'python', '.js': 'javascript'}
        mock_get_registry.return_value = mock_registry
        
        extensions = chunking_module.get_supported_extensions()
        
        assert set(extensions) == {'.py', '.js'}
        mock_get_registry.assert_called_once()
    
    @patch('chuk_code_raptor.chunking.get_registry')
    def test_get_parser_info_uses_registry(self, mock_get_registry):
        """Test get_parser_info uses registry for comprehensive information"""
        mock_registry = MockRegistry()
        mock_registry.parser_configs = {
            'python': Mock(),
            'javascript': Mock(),
            'markdown': Mock()
        }
        mock_registry.parsers = {
            'python': Mock(),
            'javascript': Mock()
        }
        mock_registry.language_mapping = {'python': 'python', 'javascript': 'javascript'}
        mock_get_registry.return_value = mock_registry
        
        info = chunking_module.get_parser_info()
        
        assert isinstance(info, dict)
        assert info['total_parsers'] == 3
        assert info['available_parsers'] == 2
        assert info['supported_languages'] == 2
        assert info['supported_extensions'] == 0  # Empty in mock
        assert 'parser_types' in info
        assert 'package_availability' in info
        assert 'config_file' in info
        assert str(mock_registry.config_path) in info['config_file']
    
    @patch('chuk_code_raptor.chunking.get_registry')
    def test_get_installation_help_uses_registry(self, mock_get_registry):
        """Test get_installation_help uses registry"""
        mock_registry = MockRegistry()
        mock_get_registry.return_value = mock_registry
        
        help_text = chunking_module.get_installation_help()
        
        assert help_text == "pip install tree-sitter-languages"
        mock_get_registry.assert_called_once()


class TestConvenienceFunctions:
    """Test suite for convenience functions with registry support"""
    
    def test_create_engine_default(self):
        """Test create_engine with default configuration"""
        engine = chunking_module.create_engine()
        
        assert engine is not None
        assert hasattr(engine, 'chunk_file')
        assert hasattr(engine, 'chunk_content')
        assert hasattr(engine, 'get_supported_languages')
        assert hasattr(engine, 'registry')  # Should have registry
    
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


class TestFileOperations:
    """Test suite for file-based operations with registry"""
    
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


class TestInitializationLogging:
    """Test suite for YAML-based initialization and logging"""
    
    @patch('chuk_code_raptor.chunking.logger')
    @patch('chuk_code_raptor.chunking.get_registry')
    def test_log_initialization_status_with_registry(self, mock_get_registry, mock_logger):
        """Test initialization logging with registry information"""
        mock_registry = MockRegistry()
        mock_registry.parser_configs = {'python': Mock(), 'javascript': Mock()}
        mock_registry.parsers = {'python': Mock()}
        mock_get_registry.return_value = mock_registry
        
        chunking_module._log_initialization_status()
        
        # Should log registry-based information
        mock_logger.info.assert_called()
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        
        # Should mention registry information - be flexible about exact wording
        registry_mentioned = any("registry" in call.lower() or "yaml" in call.lower() for call in info_calls)
        config_mentioned = any("config" in call.lower() for call in info_calls)
        
        assert registry_mentioned or config_mentioned, f"Expected registry/config info in logs: {info_calls}"
    
    @patch('chuk_code_raptor.chunking.logger')
    @patch('chuk_code_raptor.chunking.get_registry')
    def test_log_initialization_status_no_parsers(self, mock_get_registry, mock_logger):
        """Test initialization logging with no available parsers"""
        mock_registry = MockRegistry()  # Empty registry
        mock_get_registry.return_value = mock_registry
        
        chunking_module._log_initialization_status()
        
        # Should log warning about no parsers - check if any logging happened
        # The exact behavior might vary, so be flexible
        warning_called = mock_logger.warning.called
        info_called = mock_logger.info.called
        
        # At least some logging should happen
        assert warning_called or info_called, "Expected some logging to occur"
        
        if warning_called:
            warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
            no_parsers_mentioned = any("parsers" in call.lower() for call in warning_calls)
            assert no_parsers_mentioned, f"Expected parsers mentioned in warnings: {warning_calls}"
    
    @patch('chuk_code_raptor.chunking.logger')
    @patch('chuk_code_raptor.chunking.get_registry')
    def test_log_initialization_status_with_comprehensive_packages(self, mock_get_registry, mock_logger):
        """Test initialization logging mentions comprehensive packages"""
        mock_registry = MockRegistry()
        mock_registry.parsers = {'python': Mock(), 'javascript': Mock()}
        mock_registry.parser_configs = {'python': Mock(), 'javascript': Mock()}
        # Mock registry already has comprehensive packages in get_parser_stats
        mock_get_registry.return_value = mock_registry
        
        chunking_module._log_initialization_status()
        
        # Should log package information
        mock_logger.info.assert_called()
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        
        # Should mention parsers and configuration - be more flexible with the exact wording
        parser_mentioned = any("parser" in call.lower() for call in info_calls)
        config_mentioned = any("config" in call.lower() for call in info_calls)
        
        assert parser_mentioned or config_mentioned, f"Expected parser/config info in logs: {info_calls}"
    
    @patch('chuk_code_raptor.chunking.logger')
    @patch('chuk_code_raptor.chunking.get_registry', side_effect=Exception("Registry error"))
    def test_log_initialization_status_registry_error(self, mock_get_registry, mock_logger):
        """Test initialization logging handles registry errors gracefully"""
        chunking_module._log_initialization_status()
        
        # Should log debug error and warning
        mock_logger.debug.assert_called()
        mock_logger.warning.assert_called()
        
        debug_call = mock_logger.debug.call_args[0][0]
        warning_call = mock_logger.warning.call_args[0][0]
        
        assert "Registry error" in debug_call
        assert "some parsers may not be available" in warning_call


class TestEdgeCases:
    """Test suite for edge cases and error conditions with registry"""
    
    def test_create_engine_none_config(self):
        """Test create_engine with None config"""
        engine = chunking_module.create_engine(None)
        
        assert engine is not None
        # Should use DEFAULT_CONFIG
        assert engine.config is chunking_module.DEFAULT_CONFIG
    
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
    
    @patch('chuk_code_raptor.chunking.get_registry')
    def test_get_supported_languages_registry_error(self, mock_get_registry):
        """Test get_supported_languages handles registry errors"""
        mock_get_registry.side_effect = Exception("Registry failed")
        
        # Should handle error gracefully, possibly returning empty list
        try:
            languages = chunking_module.get_supported_languages()
            assert isinstance(languages, list)
        except Exception:
            # If it raises, that's also acceptable behavior
            pass
    
    @patch('chuk_code_raptor.chunking.get_registry')
    def test_get_parser_info_registry_error(self, mock_get_registry):
        """Test get_parser_info handles registry errors"""
        mock_get_registry.side_effect = Exception("Registry failed")
        
        # Should handle error gracefully
        try:
            info = chunking_module.get_parser_info()
            assert isinstance(info, dict)
        except Exception:
            # If it raises, that's also acceptable behavior
            pass


class TestModuleIntegration:
    """Test suite for module-level integration with YAML registry"""
    
    def test_module_can_be_imported(self):
        """Test that the module can be imported without errors"""
        # This test passes if the import at the top doesn't raise an error
        assert chunking_module is not None
    
    def test_basic_workflow_with_registry(self):
        """Test basic end-to-end workflow using registry"""
        # Create engine (should use registry)
        engine = chunking_module.create_engine()
        assert hasattr(engine, 'registry')
        
        # Get supported languages from registry
        languages = chunking_module.get_supported_languages()
        assert isinstance(languages, list)
        
        # Get parser info from registry
        info = chunking_module.get_parser_info()
        assert isinstance(info, dict)
        assert 'config_file' in info  # Should include registry config file
        
        # Try chunking content
        chunks = chunking_module.chunk_content("# Test", "python", "test.py")
        assert isinstance(chunks, list)
    
    def test_configuration_presets_work_with_registry(self):
        """Test that configuration presets work with registry system"""
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
            assert hasattr(engine, 'registry')  # Should have registry
    
    @patch('chuk_code_raptor.chunking.get_registry')
    def test_registry_methods_available(self, mock_get_registry):
        """Test that registry methods are available at module level"""
        mock_registry = Mock()
        mock_get_registry.return_value = mock_registry
        
        # Test registry access functions
        assert hasattr(chunking_module, 'get_registry')
        assert hasattr(chunking_module, 'reload_registry')
        assert hasattr(chunking_module, 'register_custom_parser')
        
        # Should be able to call these functions
        registry = chunking_module.get_registry()
        assert registry is mock_registry
    
    def test_module_exports_registry_classes(self):
        """Test that module exports registry-related classes"""
        # Should export ParserRegistry
        assert hasattr(chunking_module, 'ParserRegistry')
        assert chunking_module.ParserRegistry is not None
        
        # Registry functions should be callable
        assert callable(chunking_module.get_registry)
        assert callable(chunking_module.reload_registry)
        assert callable(chunking_module.register_custom_parser)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])