# src/chuk_code_raptor/chunking/parsers/registry.py
"""
Dynamic Parser Registry System
==============================

YAML-driven parser configuration system that eliminates hard-coding and makes
adding new parsers as simple as updating a YAML file.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Type
from dataclasses import dataclass
from importlib import import_module

logger = logging.getLogger(__name__)

@dataclass
class ParserConfig:
    """Configuration for a single parser"""
    name: str
    type: str  # tree_sitter_available, tree_sitter_dedicated, heuristic
    module: str
    class_name: str
    languages: List[str]
    extensions: List[str]
    description: str
    package_priority: List[str]
    fallback: Optional[str] = None
    config: Dict[str, Any] = None

class ParserRegistry:
    """
    Dynamic parser registry that loads configuration from YAML and manages
    parser discovery, instantiation, and fallback strategies.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = {}
        self.parsers = {}
        self.language_map = {}
        self.extension_map = {}
        self.package_availability = {}
        self._loaded = False
        
        # Load configuration
        self._load_config()
    
    def _get_default_config_path(self) -> Path:
        """Get the default path to the parser configuration YAML"""
        current_dir = Path(__file__).parent
        return current_dir / "config.yaml"
    
    def _load_config(self):
        """Load parser configuration from YAML file"""
        try:
            if not self.config_path.exists():
                logger.error(f"Parser config file not found: {self.config_path}")
                self.config = self._get_fallback_config()
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Build language and extension mappings
            self._build_mappings()
            
            # Check package availability
            self._check_package_availability()
            
            self._loaded = True
            logger.info(f"Loaded parser configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load parser config: {e}")
            self.config = self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict:
        """Get minimal fallback configuration when YAML loading fails"""
        return {
            'parsers': {
                'text': {
                    'type': 'heuristic',
                    'module': 'chuk_code_raptor.chunking.parsers.text',
                    'class': 'TextParser',
                    'languages': ['text', 'unknown'],
                    'extensions': ['.txt'],
                    'description': 'Fallback text parser',
                    'package_priority': [],
                    'fallback': None
                }
            },
            'language_aliases': {
                'unknown': 'text'
            },
            'defaults': {
                'tree_sitter_timeout': 30,
                'fallback_enabled': True,
                'enable_heuristic_fallback': True
            }
        }
    
    def _build_mappings(self):
        """Build language and extension mappings from parser configs"""
        parsers_config = self.config.get('parsers', {})
        aliases = self.config.get('language_aliases', {})
        
        # Clear existing mappings
        self.language_map.clear()
        self.extension_map.clear()
        
        # Build mappings from parser definitions
        for parser_name, parser_info in parsers_config.items():
            languages = parser_info.get('languages', [])
            extensions = parser_info.get('extensions', [])
            
            # Map languages to parser
            for language in languages:
                self.language_map[language] = parser_name
            
            # Map extensions to parser
            for extension in extensions:
                self.extension_map[extension] = parser_name
        
        # Add language aliases
        for alias, target in aliases.items():
            if target in self.language_map:
                self.language_map[alias] = self.language_map[target]
        
        logger.debug(f"Built mappings: {len(self.language_map)} languages, {len(self.extension_map)} extensions")
    
    def _check_package_availability(self):
        """Check which tree-sitter packages are available"""
        self.package_availability = {}
        
        # Check comprehensive packages
        comprehensive_available = []
        
        try:
            from tree_sitter_languages import get_language
            comprehensive_available.append('tree-sitter-languages')
            logger.debug("tree-sitter-languages available")
        except ImportError:
            pass
        
        try:
            from tree_sitter_language_pack import get_language
            comprehensive_available.append('tree-sitter-language-pack')
            logger.debug("tree-sitter-language-pack available")
        except ImportError:
            pass
        
        self.package_availability['comprehensive'] = comprehensive_available
        
        # Check individual packages
        individual_packages = {}
        parsers_config = self.config.get('parsers', {})
        
        for parser_name, parser_info in parsers_config.items():
            if parser_info.get('type') == 'tree_sitter_available':
                languages = parser_info.get('languages', [])
                for language in languages:
                    package_name = f"tree_sitter_{language}"
                    try:
                        module = import_module(package_name)
                        if hasattr(module, 'language'):
                            individual_packages[language] = package_name
                    except ImportError:
                        pass
        
        self.package_availability['individual'] = individual_packages
        
        logger.info(f"Package availability: {len(comprehensive_available)} comprehensive, {len(individual_packages)} individual")
    
    def get_parser_config(self, parser_name: str) -> Optional[ParserConfig]:
        """Get configuration for a specific parser"""
        if not self._loaded:
            return None
        
        parsers_config = self.config.get('parsers', {})
        parser_info = parsers_config.get(parser_name)
        
        if not parser_info:
            return None
        
        return ParserConfig(
            name=parser_name,
            type=parser_info['type'],
            module=parser_info['module'],
            class_name=parser_info['class'],
            languages=parser_info['languages'],
            extensions=parser_info['extensions'],
            description=parser_info['description'],
            package_priority=parser_info.get('package_priority', []),
            fallback=parser_info.get('fallback'),
            config=self.config.get('parser_configs', {}).get(parser_name, {})
        )
    
    def get_parser_for_language(self, language: str) -> Optional[str]:
        """Get parser name for a given language"""
        return self.language_map.get(language.lower())
    
    def get_parser_for_extension(self, extension: str) -> Optional[str]:
        """Get parser name for a given file extension"""
        return self.extension_map.get(extension.lower())
    
    def get_supported_languages(self) -> List[str]:
        """Get all supported languages"""
        return list(self.language_map.keys())
    
    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions"""
        return list(self.extension_map.keys())
    
    def get_all_parser_names(self) -> List[str]:
        """Get names of all configured parsers"""
        return list(self.config.get('parsers', {}).keys())
    
    def create_parser(self, parser_name: str, chunking_config) -> Optional[Any]:
        """Create a parser instance by name"""
        parser_config = self.get_parser_config(parser_name)
        if not parser_config:
            logger.warning(f"No configuration found for parser: {parser_name}")
            return None
        
        # Check if parser type is available
        if not self._is_parser_available(parser_config):
            logger.warning(f"Parser {parser_name} not available (missing dependencies)")
            return None
        
        try:
            # Import the parser module
            module = import_module(parser_config.module)
            parser_class = getattr(module, parser_config.class_name)
            
            # Apply parser-specific configuration
            self._apply_parser_config(chunking_config, parser_config)
            
            # Create parser instance
            parser = parser_class(chunking_config)
            
            # Set metadata from config
            parser.supported_languages = set(parser_config.languages)
            parser.supported_extensions = set(parser_config.extensions)
            parser.parser_type = parser_config.type
            parser.name = parser_config.class_name
            parser._config_name = parser_name
            
            logger.debug(f"Created parser: {parser_name} ({parser_config.type})")
            return parser
            
        except Exception as e:
            logger.error(f"Failed to create parser {parser_name}: {e}")
            return None
    
    def _is_parser_available(self, parser_config: ParserConfig) -> bool:
        """Check if a parser is available (has required dependencies)"""
        if parser_config.type == 'heuristic':
            # Heuristic parsers don't need tree-sitter
            return True
        
        elif parser_config.type in ['tree_sitter_dedicated', 'tree_sitter_simple']:
            # For dedicated and simple parsers, try to import the module to check availability
            try:
                module = import_module(parser_config.module)
                parser_class = getattr(module, parser_config.class_name)
                
                # Try to create a test instance to verify dependencies
                from ..config import ChunkingConfig
                test_config = ChunkingConfig()
                test_instance = parser_class(test_config)
                
                return True
            except ImportError:
                logger.debug(f"Parser {parser_config.name} not available: module not found")
                return False
            except Exception as e:
                logger.debug(f"Parser {parser_config.name} not available: {e}")
                return False
        
        return False
    
    def _apply_parser_config(self, chunking_config, parser_config: ParserConfig):
        """Apply parser-specific configuration to chunking config"""
        if not parser_config.config:
            return
        
        # Apply parser-specific settings as attributes
        for key, value in parser_config.config.items():
            setattr(chunking_config, f"{parser_config.name}_{key}", value)
    
    def discover_available_parsers(self, chunking_config) -> Dict[str, Any]:
        """Discover all available parsers and create instances"""
        available_parsers = {}
        
        for parser_name in self.get_all_parser_names():
            parser = self.create_parser(parser_name, chunking_config)
            if parser:
                # Map by all supported languages
                parser_config = self.get_parser_config(parser_name)
                for language in parser_config.languages:
                    available_parsers[language] = parser
        
        logger.info(f"Discovered {len(available_parsers)} available parser instances")
        return available_parsers
    
    def get_installation_help(self) -> str:
        """Generate installation help based on current availability"""
        comprehensive = self.package_availability.get('comprehensive', [])
        individual = self.package_availability.get('individual', {})
        
        recommendations = self.config.get('package_recommendations', {})
        
        if comprehensive:
            available_langs = self._get_comprehensive_languages(comprehensive[0])
            return f"""
✅ Comprehensive tree-sitter support detected!
📦 Using: {comprehensive[0]}
🚀 Available languages: {len(available_langs)}

{', '.join(sorted(available_langs))}

All parsers should work correctly.
"""
        elif individual:
            return f"""
⚠️  Individual tree-sitter packages detected
📦 Available: {len(individual)} languages
Languages: {', '.join(sorted(individual.keys()))}

💡 For better support, install a comprehensive package:
    pip install tree-sitter-languages

This will enable support for all languages including LaTeX and specialized formats.
"""
        else:
            comprehensive_options = recommendations.get('comprehensive', [])
            suggestion = comprehensive_options[0]['name'] if comprehensive_options else 'tree-sitter-languages'
            
            return f"""
❌ No tree-sitter packages found

Install comprehensive support (recommended):
    pip install {suggestion}

Or install individual packages:
    pip install tree-sitter-python tree-sitter-javascript
    pip install tree-sitter-html tree-sitter-css tree-sitter-json

Note: Some specialized parsers (like LaTeX) require comprehensive packages.
"""
    
    def _get_comprehensive_languages(self, package_name: str) -> List[str]:
        """Get languages supported by a comprehensive package"""
        recommendations = self.config.get('package_recommendations', {})
        for pkg_info in recommendations.get('comprehensive', []):
            if pkg_info['name'] == package_name:
                return pkg_info.get('languages', [])
        
        # Fallback: return all configured languages
        return list(set(lang for parser_config in self.config.get('parsers', {}).values() 
                       for lang in parser_config.get('languages', [])))
    
    def add_parser_runtime(self, parser_name: str, parser_instance: Any, languages: List[str], extensions: List[str]):
        """Add a parser at runtime (for dynamic registration)"""
        # Update mappings
        for language in languages:
            self.language_map[language] = parser_name
        
        for extension in extensions:
            self.extension_map[extension] = parser_name
        
        # Store instance if needed
        self.parsers[parser_name] = parser_instance
        
        logger.info(f"Registered runtime parser: {parser_name}")
    
    def reload_config(self):
        """Reload configuration from YAML file"""
        self._loaded = False
        self.parsers.clear()
        self._load_config()
        logger.info("Parser configuration reloaded")
    
    def get_parser_stats(self) -> Dict[str, Any]:
        """Get statistics about parser availability"""
        total_parsers = len(self.config.get('parsers', {}))
        available_parsers = sum(1 for name in self.get_all_parser_names() 
                               if self._is_parser_available(self.get_parser_config(name)))
        
        parser_types = {}
        for parser_name in self.get_all_parser_names():
            parser_config = self.get_parser_config(parser_name)
            parser_type = parser_config.type
            parser_types[parser_type] = parser_types.get(parser_type, 0) + 1
        
        return {
            'total_parsers': total_parsers,
            'available_parsers': available_parsers,
            'unavailable_parsers': total_parsers - available_parsers,
            'supported_languages': len(self.get_supported_languages()),
            'supported_extensions': len(self.get_supported_extensions()),
            'parser_types': parser_types,
            'package_availability': self.package_availability,
            'comprehensive_packages': len(self.package_availability.get('comprehensive', [])),
            'individual_packages': len(self.package_availability.get('individual', {}))
        }

# Global registry instance
_registry = None

def get_registry() -> ParserRegistry:
    """Get the global parser registry instance"""
    global _registry
    if _registry is None:
        _registry = ParserRegistry()
    return _registry

def reload_registry():
    """Reload the global parser registry"""
    global _registry
    _registry = None
    _registry = ParserRegistry()

# Convenience functions for backward compatibility
def discover_available_parsers(config=None):
    """Discover available parsers using the registry"""
    from ..config import ChunkingConfig
    if config is None:
        config = ChunkingConfig()
    
    registry = get_registry()
    return registry.discover_available_parsers(config)

def get_installation_help():
    """Get installation help using the registry"""
    registry = get_registry()
    return registry.get_installation_help()

def check_parser_availability():
    """Check parser availability using the registry"""
    registry = get_registry()
    return registry.package_availability

def get_supported_languages():
    """Get supported languages using the registry"""
    registry = get_registry()
    return registry.get_supported_languages()

def get_supported_extensions():
    """Get supported extensions using the registry"""
    registry = get_registry()
    return registry.get_supported_extensions()

# Example of adding a new parser at runtime
def register_custom_parser(name: str, parser_class: Type, languages: List[str], 
                          extensions: List[str], config_dict: Dict[str, Any]):
    """
    Register a custom parser at runtime
    
    Example:
        register_custom_parser(
            name="my_custom_lang",
            parser_class=MyCustomParser,
            languages=["mylang"],
            extensions=[".mylang"],
            config_dict={
                "type": "tree_sitter_dedicated",
                "description": "My custom language parser"
            }
        )
    """
    registry = get_registry()
    
    # Create a runtime config
    from ..config import ChunkingConfig
    config = ChunkingConfig()
    
    # Create parser instance
    parser = parser_class(config)
    
    # Register with registry
    registry.add_parser_runtime(name, parser, languages, extensions)
    
    logger.info(f"Custom parser '{name}' registered successfully")