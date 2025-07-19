# src/chuk_code_raptor/chunking/registry.py
"""
Chunker Registry
================

Registry for managing and selecting chunkers based on language and file type.
Supports automatic chunker discovery and priority-based selection.
"""

from typing import List, Dict, Optional, Type, Set
import logging

from .base import BaseChunker
from .config import ChunkingConfig

logger = logging.getLogger(__name__)

class ChunkerRegistry:
    """Registry for managing available chunkers"""
    
    def __init__(self):
        """Initialize empty registry"""
        self._chunkers: List[Type] = []
        self._instances: Dict[Type, any] = {}
        self._config: Optional[ChunkingConfig] = None
    
    def register(self, chunker_class: Type):
        """
        Register a chunker class.
        
        Args:
            chunker_class: Chunker class to register
        """
        if chunker_class not in self._chunkers:
            self._chunkers.append(chunker_class)
            logger.debug(f"Registered chunker: {chunker_class.__name__}")
        else:
            logger.warning(f"Chunker already registered: {chunker_class.__name__}")
    
    def unregister(self, chunker_class: Type):
        """
        Unregister a chunker class.
        
        Args:
            chunker_class: Class to unregister
        """
        if chunker_class in self._chunkers:
            self._chunkers.remove(chunker_class)
            if chunker_class in self._instances:
                del self._instances[chunker_class]
            logger.debug(f"Unregistered chunker: {chunker_class.__name__}")
    
    def set_config(self, config: ChunkingConfig):
        """
        Set configuration for all chunkers.
        
        Args:
            config: ChunkingConfig to use for all chunkers
        """
        self._config = config
        # Clear existing instances to force recreation with new config
        self._instances.clear()
    
    def get_chunker(self, language: str, file_extension: str = None):
        """
        Get the best chunker for the given language and file extension.
        
        Args:
            language: Programming language (e.g., 'python', 'javascript')
            file_extension: File extension (e.g., '.py', '.js')
            
        Returns:
            Best matching chunker instance, or None if no suitable chunker found
        """
        if not self._chunkers:
            logger.warning("No chunkers registered")
            return None
        
        # Find the best chunker based on priority
        best_chunker_class = None
        best_priority = -1
        
        for chunker_class in self._chunkers:
            instance = self._get_instance(chunker_class)
            
            if self._can_handle(instance, language, file_extension):
                priority = self._get_priority(instance, language, file_extension)
                
                if priority > best_priority:
                    best_priority = priority
                    best_chunker_class = chunker_class
        
        if best_chunker_class:
            logger.debug(f"Selected chunker: {best_chunker_class.__name__} "
                        f"(priority: {best_priority}) for {language}{file_extension or ''}")
            return self._get_instance(best_chunker_class)
        
        logger.warning(f"No suitable chunker found for {language}{file_extension or ''}")
        return None
    
    def get_available_languages(self) -> List[str]:
        """
        Get list of all supported languages.
        
        Returns:
            List of supported language names
        """
        languages = set()
        
        for chunker_class in self._chunkers:
            instance = self._get_instance(chunker_class)
            langs = self._get_supported_languages(instance)
            languages.update(langs)
        
        return sorted(list(languages))
    
    def get_available_extensions(self) -> List[str]:
        """
        Get list of all supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        extensions = set()
        
        for chunker_class in self._chunkers:
            instance = self._get_instance(chunker_class)
            exts = self._get_supported_extensions(instance)
            extensions.update(exts)
        
        return sorted(list(extensions))
    
    def get_chunkers_for_language(self, language: str) -> List:
        """
        Get all chunkers that support a specific language.
        
        Args:
            language: Programming language
            
        Returns:
            List of chunkers that support the language, sorted by priority
        """
        compatible_chunkers = []
        
        for chunker_class in self._chunkers:
            instance = self._get_instance(chunker_class)
            
            if self._can_handle(instance, language, ''):
                priority = self._get_priority(instance, language, '')
                compatible_chunkers.append((priority, instance))
        
        # Sort by priority (highest first)
        compatible_chunkers.sort(key=lambda x: x[0], reverse=True)
        
        return [chunker for priority, chunker in compatible_chunkers]
    
    def list_chunkers(self) -> List[Dict[str, any]]:
        """
        Get information about all registered chunkers.
        
        Returns:
            List of dictionaries with chunker information
        """
        chunker_info = []
        
        for chunker_class in self._chunkers:
            instance = self._get_instance(chunker_class)
            
            info = {
                'name': chunker_class.__name__,
                'class': chunker_class,
                'supported_languages': list(self._get_supported_languages(instance)),
                'supported_extensions': list(self._get_supported_extensions(instance))
            }
            
            # Add strategy if it exists
            if hasattr(instance, 'strategy'):
                info['strategy'] = instance.strategy.value
            
            chunker_info.append(info)
        
        return chunker_info
    
    def _get_instance(self, chunker_class: Type):
        """
        Get or create chunker instance.
        
        Args:
            chunker_class: Chunker class
            
        Returns:
            Chunker instance
        """
        if chunker_class not in self._instances:
            # Try with config first, fallback to no config
            config = self._config or ChunkingConfig()
            try:
                self._instances[chunker_class] = chunker_class(config)
            except TypeError:
                # Chunker doesn't take config
                self._instances[chunker_class] = chunker_class()
        
        return self._instances[chunker_class]
    
    def _can_handle(self, instance, language: str, file_extension: str) -> bool:
        """Check if chunker can handle the given content"""
        if hasattr(instance, 'can_chunk'):
            return instance.can_chunk(language, file_extension or '')
        elif hasattr(instance, 'can_parse'):
            # Different interface - adapt as needed
            return (language in self._get_supported_languages(instance) or
                   file_extension in self._get_supported_extensions(instance))
        return False
    
    def _get_priority(self, instance, language: str, file_extension: str) -> int:
        """Get priority for chunker"""
        if hasattr(instance, 'get_priority'):
            return instance.get_priority(language, file_extension or '')
        return 50  # Default priority
    
    def _get_supported_languages(self, instance) -> Set[str]:
        """Get supported languages from chunker"""
        if hasattr(instance, 'supported_languages'):
            return instance.supported_languages
        elif hasattr(instance, 'capabilities') and hasattr(instance.capabilities, 'languages'):
            return instance.capabilities.languages
        return set()
    
    def _get_supported_extensions(self, instance) -> Set[str]:
        """Get supported extensions from chunker"""
        if hasattr(instance, 'supported_extensions'):
            return instance.supported_extensions
        elif hasattr(instance, 'capabilities') and hasattr(instance.capabilities, 'file_extensions'):
            return instance.capabilities.file_extensions
        return set()
    
    def clear(self):
        """Clear all registered chunkers"""
        self._chunkers.clear()
        self._instances.clear()
        logger.debug("Cleared all chunkers from registry")

# Global registry instance
_global_registry = ChunkerRegistry()

def get_registry() -> ChunkerRegistry:
    """Get the global chunker registry instance"""
    return _global_registry

def register_chunker(chunker_class: Type):
    """
    Convenience function to register a chunker with the global registry.
    
    Args:
        chunker_class: Chunker class to register
    """
    _global_registry.register(chunker_class)

def get_chunker(language: str, file_extension: str = None):
    """
    Convenience function to get a chunker from the global registry.
    
    Args:
        language: Programming language
        file_extension: File extension
        
    Returns:
        Best matching chunker or None
    """
    return _global_registry.get_chunker(language, file_extension)

def set_global_config(config: ChunkingConfig):
    """
    Set configuration for the global registry.
    
    Args:
        config: ChunkingConfig to use
    """
    _global_registry.set_config(config)