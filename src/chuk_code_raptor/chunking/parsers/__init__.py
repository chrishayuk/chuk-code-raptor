# src/chuk_code_raptor/chunking/parsers/__init__.py
"""
Parsers Package - Pure Registry Edition
=======================================

Clean parser implementations with pure YAML configuration.
No hardcoding - everything is managed through the registry system.
"""

import logging

logger = logging.getLogger(__name__)

# Import registry system for dynamic parser management
from ..registry import (
    ParserRegistry,
    get_registry,
    reload_registry,
    register_custom_parser
)

def get_parser_info():
    """Get information about all parsers from YAML configuration"""
    registry = get_registry()
    stats = registry.get_parser_stats()
    return {
        'total_parsers': stats['total_parsers'],
        'available_parsers': stats['available_parsers'],
        'parser_configs': {name: registry.get_parser_config(name).__dict__ 
                          for name in registry.get_all_parser_names()},
        'config_file': str(registry.config_path)
    }

def check_parser_availability():
    """Check availability of all configured parsers"""
    registry = get_registry()
    availability = {}
    
    for parser_name in registry.get_all_parser_names():
        parser_config = registry.get_parser_config(parser_name)
        availability[parser_name] = registry._is_parser_available(parser_config)
    
    return availability

def load_parser(language: str, config=None):
    """Load a parser instance for the given language using YAML config"""
    registry = get_registry()
    
    # Get parser name for language
    parser_name = registry.get_parser_for_language(language)
    if not parser_name:
        return None
    
    # Create parser instance
    return registry.create_parser(parser_name, config)

def list_available_parsers():
    """List all available parser names from YAML config"""
    registry = get_registry()
    return [name for name in registry.get_all_parser_names() 
            if registry._is_parser_available(registry.get_parser_config(name))]

def print_parser_status():
    """Print comprehensive parser status"""
    print("📦 Parser Package Status (Pure Registry)")
    print("=" * 45)
    
    registry = get_registry()
    stats = registry.get_parser_stats()
    
    print(f"Configuration file: {registry.config_path}")
    print(f"Total configured parsers: {stats['total_parsers']}")
    print(f"Available parsers: {stats['available_parsers']}")
    print(f"Unavailable parsers: {stats['unavailable_parsers']}")
    print(f"Supported languages: {stats['supported_languages']}")
    print(f"Supported extensions: {stats['supported_extensions']}")
    
    if stats['parser_types']:
        print(f"\nParser types:")
        for parser_type, count in stats['parser_types'].items():
            print(f"  {parser_type}: {count}")
    
    # Show availability details
    availability = check_parser_availability()
    available_parsers = [name for name, avail in availability.items() if avail]
    unavailable_parsers = [name for name, avail in availability.items() if not avail]
    
    if available_parsers:
        print(f"\nAvailable parsers:")
        for name in available_parsers:
            config = registry.get_parser_config(name)
            print(f"  ✅ {name} ({config.type}) - {', '.join(config.languages)}")
    
    if unavailable_parsers:
        print(f"\nUnavailable parsers:")
        for name in unavailable_parsers:
            config = registry.get_parser_config(name)
            print(f"  ❌ {name} ({config.type}) - {', '.join(config.languages)}")
    
    if unavailable_parsers:
        print(f"\n💡 Installation help:")
        help_text = registry.get_installation_help()
        for line in help_text.split('\n'):
            if line.strip():
                print(f"   {line}")

# Convenience functions using registry
def get_supported_languages():
    """Get supported languages from registry"""
    return get_registry().get_supported_languages()

def get_supported_extensions():
    """Get supported extensions from registry"""
    return get_registry().get_supported_extensions()

def get_installation_help():
    """Get installation help from registry"""
    return get_registry().get_installation_help()

def discover_available_parsers(config=None):
    """Discover available parsers using registry"""
    return get_registry().discover_available_parsers(config)

# Log parser package status on import
def _log_parser_package_status():
    """Log the status of the parser package"""
    try:
        registry = get_registry()
        stats = registry.get_parser_stats()
        
        logger.info(f"📦 Parser package loaded (pure registry)")
        logger.info(f"   Configuration: {registry.config_path}")
        logger.info(f"   Total parsers: {stats['total_parsers']}")
        logger.info(f"   Available: {stats['available_parsers']}")
        logger.info(f"   Languages: {stats['supported_languages']}")
        logger.info(f"   Extensions: {stats['supported_extensions']}")
        
        if stats['available_parsers'] == 0:
            logger.warning("⚠️  No parsers available - check tree-sitter packages")
        elif stats['unavailable_parsers'] > 0:
            logger.info(f"💡 {stats['unavailable_parsers']} parsers need dependencies")
    
    except Exception as e:
        logger.debug(f"Error checking parser package status: {e}")

# Initialize on import
_log_parser_package_status()

# Export commonly used functions
__all__ = [
    # Registry system
    'ParserRegistry',
    'get_registry',
    'reload_registry', 
    'register_custom_parser',
    
    # Discovery functions
    'discover_available_parsers',
    'get_installation_help',
    'check_parser_availability',
    'get_supported_languages',
    'get_supported_extensions',
    
    # Package-specific functions
    'get_parser_info',
    'load_parser',
    'list_available_parsers',
    'print_parser_status',
]