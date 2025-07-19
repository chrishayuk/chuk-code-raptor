# chuk_code_raptor/chunking/tree_sitter_chunker/config.py
"""
Tree-sitter Configuration Loading
=================================

Loads language-specific configurations from YAML files.
Each language has its own configuration defining how to parse and chunk.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from .base import LanguageConfig
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class TreeSitterConfigLoader:
    """Loads tree-sitter configurations from YAML files"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config loader.
        
        Args:
            config_dir: Directory containing language config files
                       If None, uses default location
        """
        if config_dir is None:
            # Default to configs in the same directory as this file
            config_dir = Path(__file__).parent / "configs"
        
        self.config_dir = Path(config_dir)
        logger.debug(f"Tree-sitter config directory: {self.config_dir}")
    
    def load_language_config(self, language: str) -> LanguageConfig:
        """
        Load configuration for a specific language.
        
        Args:
            language: Language name (e.g., 'python', 'javascript')
            
        Returns:
            LanguageConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_file = self.config_dir / f"{language}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"No tree-sitter config found for {language}: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            return self._parse_config(raw_config, language)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config for {language}: {e}")
    
    def _parse_config(self, raw_config: Dict[str, Any], language: str) -> LanguageConfig:
        """Parse raw YAML config into LanguageConfig object"""
        
        # Parse chunk node types
        chunk_node_types = {}
        for node_type, chunk_type_str in raw_config.get('chunk_node_types', {}).items():
            try:
                chunk_node_types[node_type] = ChunkType(chunk_type_str)
            except ValueError:
                logger.warning(f"Invalid chunk type '{chunk_type_str}' for node '{node_type}' in {language}")
        
        # Parse sets
        atomic_node_types = set(raw_config.get('atomic_node_types', []))
        splittable_node_types = set(raw_config.get('splittable_node_types', []))
        file_extensions = set(raw_config.get('file_extensions', []))
        
        # Parse importance weights
        importance_weights = raw_config.get('importance_weights', {})
        
        # Parse language-specific config
        language_specific = raw_config.get('language_specific', {})
        
        # Validate required fields
        if not file_extensions:
            raise ValueError(f"No file extensions defined for {language}")
        
        if not chunk_node_types:
            raise ValueError(f"No chunk node types defined for {language}")
        
        return LanguageConfig(
            language_name=language,
            file_extensions=file_extensions,
            chunk_node_types=chunk_node_types,
            atomic_node_types=atomic_node_types,
            splittable_node_types=splittable_node_types,
            importance_weights=importance_weights,
            language_specific=language_specific
        )
    
    def list_available_languages(self) -> List[str]:
        """List all available language configurations"""
        if not self.config_dir.exists():
            return []
        
        languages = []
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.stem not in ['_base', '_template']:  # Skip template files
                languages.append(config_file.stem)
        
        return sorted(languages)
    
    def validate_config(self, language: str) -> List[str]:
        """
        Validate a language configuration and return list of issues.
        
        Args:
            language: Language to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        try:
            config = self.load_language_config(language)
            
            # Check for required chunk types
            required_chunk_types = {'function_definition', 'class_definition'}
            found_chunk_types = set(config.chunk_node_types.keys())
            
            missing_types = required_chunk_types - found_chunk_types
            if missing_types:
                issues.append(f"Missing important chunk types: {missing_types}")
            
            # Check importance weights
            if not config.importance_weights:
                issues.append("No importance weights defined")
            
            # Validate weight values
            for node_type, weight in config.importance_weights.items():
                if not 0.0 <= weight <= 1.0:
                    issues.append(f"Invalid importance weight for {node_type}: {weight} (should be 0.0-1.0)")
            
            # Check for conflicts between atomic and splittable
            conflicts = config.atomic_node_types & config.splittable_node_types
            if conflicts:
                issues.append(f"Node types marked as both atomic and splittable: {conflicts}")
            
        except Exception as e:
            issues.append(f"Failed to load config: {e}")
        
        return issues

# Global config loader instance
_config_loader: Optional[TreeSitterConfigLoader] = None

def get_config_loader() -> TreeSitterConfigLoader:
    """Get the global config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = TreeSitterConfigLoader()
    return _config_loader

def load_language_config(language: str) -> LanguageConfig:
    """Convenience function to load language config"""
    return get_config_loader().load_language_config(language)