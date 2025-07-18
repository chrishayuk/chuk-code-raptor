#!/usr/bin/env python3
# chuk_code_raptor/file_processor/config.py
"""
Configuration Management Module
===============================

Simple YAML configuration loader with validation.
Configuration is defined in external YAML files, not in Python code.

Usage:
    config = ConfigLoader.load_config("config.yaml")
    ConfigLoader.save_config(config, "config.yaml")
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessorConfig:
    """Configuration for FileProcessor components"""
    languages: Dict[str, str]
    ignore_patterns: Dict[str, List[str]]
    encoding: Dict[str, Any]
    processing: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessorConfig':
        """Create config from dictionary with validation"""
        return cls(
            languages=data.get('languages', {}),
            ignore_patterns=data.get('ignore_patterns', {'directories': [], 'files': []}),
            encoding=data.get('encoding', {'priority': ['utf-8'], 'fallback': 'binary'}),
            processing=data.get('processing', {'max_file_size_mb': 10, 'chunk_size_bytes': 8192})
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if not self.languages:
            issues.append("No language mappings defined")
        
        if not isinstance(self.ignore_patterns.get('directories'), list):
            issues.append("ignore_patterns.directories must be a list")
            
        if not isinstance(self.ignore_patterns.get('files'), list):
            issues.append("ignore_patterns.files must be a list")
        
        encoding_priority = self.encoding.get('priority', [])
        if not isinstance(encoding_priority, list) or not encoding_priority:
            issues.append("encoding.priority must be a non-empty list")
        
        max_size = self.processing.get('max_file_size_mb')
        if not isinstance(max_size, (int, float)) or max_size <= 0:
            issues.append("processing.max_file_size_mb must be a positive number")
        
        return issues

class ConfigLoader:
    """Loads and validates configuration from YAML files"""
    
    @staticmethod
    def load_config(config_path: Path) -> ProcessorConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ProcessorConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
            ValueError: If config validation fails
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data:
                raise ValueError("Empty configuration file")
            
            config = ProcessorConfig.from_dict(data)
            
            # Validate configuration
            issues = config.validate()
            if issues:
                raise ValueError(f"Configuration validation failed: {issues}")
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config from {config_path}: {e}")
    
    @staticmethod
    def save_config(config: ProcessorConfig, config_path: Path) -> None:
        """Save configuration to YAML file"""
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False, indent=2)
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            raise
    
    @staticmethod
    def merge_configs(base_config: ProcessorConfig, override_config: ProcessorConfig) -> ProcessorConfig:
        """Merge two configurations, with override taking precedence"""
        return ProcessorConfig(
            languages={**base_config.languages, **override_config.languages},
            ignore_patterns={
                'directories': list(set(
                    base_config.ignore_patterns.get('directories', []) + 
                    override_config.ignore_patterns.get('directories', [])
                )),
                'files': list(set(
                    base_config.ignore_patterns.get('files', []) + 
                    override_config.ignore_patterns.get('files', [])
                ))
            },
            encoding={**base_config.encoding, **override_config.encoding},
            processing={**base_config.processing, **override_config.processing}
        )

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Demo usage
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path("file_processor_config.yaml")
    
    print("=== ConfigLoader Demo ===")
    print(f"Loading configuration from: {config_path}")
    
    try:
        # Load and validate config
        config = ConfigLoader.load_config(config_path)
        
        print(f"\nConfiguration loaded successfully:")
        print(f"  Languages: {len(config.languages)} mappings")
        print(f"  Ignore directories: {len(config.ignore_patterns['directories'])}")
        print(f"  Ignore files: {len(config.ignore_patterns['files'])}")
        print(f"  Encoding priority: {config.encoding['priority']}")
        print(f"  Max file size: {config.processing['max_file_size_mb']} MB")
        
        # Show some example mappings
        print(f"\nExample language mappings:")
        for ext, lang in list(config.languages.items())[:5]:
            print(f"  {ext} -> {lang}")
            
        print(f"\nExample ignore patterns:")
        for pattern in config.ignore_patterns['directories'][:5]:
            print(f"  directory: {pattern}")
        for pattern in config.ignore_patterns['files'][:5]:
            print(f"  file: {pattern}")
        
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found")
        print("Please create a YAML configuration file with the required structure")
    except Exception as e:
        print(f"Error: {e}")