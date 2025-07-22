# src/chuk_code_raptor/chunking/parsers/__init__.py
"""
Parsers Package - Practical Edition
===================================

Clean parser implementations using available tree-sitter packages.
Automatically discovers and uses tree-sitter-languages, tree-sitter-language-pack,
or individual packages without requiring specific installations.
"""

import logging
from typing import Dict, List, Type, Optional

logger = logging.getLogger(__name__)

# Available parser configurations using practical packages
AVAILABLE_PARSERS = {
    'python': {
        'module': 'chuk_code_raptor.chunking.parsers.available_parsers',
        'class': 'PythonParser',
        'type': 'tree_sitter_available',
        'description': 'Python parser using available tree-sitter packages',
        'package_priority': ['tree-sitter-languages', 'tree-sitter-language-pack', 'tree-sitter-python']
    },
    'javascript': {
        'module': 'chuk_code_raptor.chunking.parsers.available_parsers',
        'class': 'JavaScriptParser',
        'type': 'tree_sitter_available',
        'description': 'JavaScript parser using available tree-sitter packages',
        'package_priority': ['tree-sitter-languages', 'tree-sitter-language-pack', 'tree-sitter-javascript']
    },
    'typescript': {
        'module': 'chuk_code_raptor.chunking.parsers.available_parsers',
        'class': 'TypeScriptParser',
        'type': 'tree_sitter_available',
        'description': 'TypeScript parser using available tree-sitter packages',
        'package_priority': ['tree-sitter-languages', 'tree-sitter-language-pack', 'tree-sitter-typescript']
    },
    'markdown': {
        'module': 'chuk_code_raptor.chunking.parsers.available_parsers',
        'class': 'MarkdownParser',
        'type': 'tree_sitter_available',
        'description': 'Markdown parser using available tree-sitter packages',
        'package_priority': ['tree-sitter-languages', 'tree-sitter-language-pack']
    },
    'latex': {
        'module': 'chuk_code_raptor.chunking.parsers.available_parsers',
        'class': 'LaTeXParser',
        'type': 'tree_sitter_available',
        'description': 'LaTeX parser using available tree-sitter packages (requires comprehensive package)',
        'package_priority': ['tree-sitter-languages', 'tree-sitter-language-pack']
    },
    'html': {
        'module': 'chuk_code_raptor.chunking.parsers.available_parsers',
        'class': 'HTMLParser',
        'type': 'tree_sitter_available',
        'description': 'HTML parser using available tree-sitter packages',
        'package_priority': ['tree-sitter-languages', 'tree-sitter-language-pack', 'tree-sitter-html']
    },
    'css': {
        'module': 'chuk_code_raptor.chunking.parsers.available_parsers',
        'class': 'CSSParser',
        'type': 'tree_sitter_available',
        'description': 'CSS parser using available tree-sitter packages',
        'package_priority': ['tree-sitter-languages', 'tree-sitter-language-pack', 'tree-sitter-css']
    },
    'json': {
        'module': 'chuk_code_raptor.chunking.parsers.available_parsers',
        'class': 'JSONParser',
        'type': 'tree_sitter_available',
        'description': 'JSON parser using available tree-sitter packages',
        'package_priority': ['tree-sitter-languages', 'tree-sitter-language-pack', 'tree-sitter-json']
    },
}

def get_parser_info() -> Dict[str, Dict]:
    """Get information about all available parsers"""
    return AVAILABLE_PARSERS.copy()

def get_supported_languages() -> List[str]:
    """Get list of languages with available parsers"""
    return list(AVAILABLE_PARSERS.keys())

def check_parser_availability(language: str) -> bool:
    """Check if a parser is available for the given language"""
    if language not in AVAILABLE_PARSERS:
        return False
    
    parser_info = AVAILABLE_PARSERS[language]
    
    try:
        # Try to import the parser module
        module = __import__(parser_info['module'], fromlist=[parser_info['class']])
        parser_class = getattr(module, parser_info['class'])
        
        # Try to create an instance to verify dependencies
        from ..config import ChunkingConfig
        test_instance = parser_class(ChunkingConfig())
        
        # Check if it actually has a working language parser
        return hasattr(test_instance, '_language')
        
    except ImportError as e:
        logger.debug(f"Parser {language} not available: missing dependencies ({e})")
        return False
    except Exception as e:
        logger.debug(f"Parser {language} failed initialization: {e}")
        return False

def load_parser(language: str, config=None):
    """Load a parser instance for the given language"""
    if language not in AVAILABLE_PARSERS:
        return None
    
    if not check_parser_availability(language):
        return None
    
    parser_info = AVAILABLE_PARSERS[language]
    
    try:
        from ..config import ChunkingConfig
        module = __import__(parser_info['module'], fromlist=[parser_info['class']])
        parser_class = getattr(module, parser_info['class'])
        
        config = config or ChunkingConfig()
        parser = parser_class(config)
        
        # Only return if successfully initialized
        if hasattr(parser, '_language'):
            return parser
        else:
            logger.debug(f"Parser {language} created but no language available")
            return None
        
    except Exception as e:
        logger.error(f"Failed to load parser for {language}: {e}")
        return None

def discover_available_parsers() -> Dict[str, Dict]:
    """Discover which parsers are actually available"""
    from .available_parsers import discover_available_parsers as discover_impl
    from .available_parsers import check_parser_availability as check_availability
    
    # Get runtime information
    runtime_parsers = discover_impl()
    
    # Get package availability info
    package_info = check_availability()
    
    available = {}
    
    for language, parser_info in AVAILABLE_PARSERS.items():
        if language in runtime_parsers:
            parser = runtime_parsers[language]
            available[language] = parser_info.copy()
            
            # Add runtime information
            available[language].update({
                'supported_languages': list(parser.supported_languages),
                'supported_extensions': list(parser.supported_extensions),
                'parser_name': parser.name,
                'parser_type': parser.parser_type,
                'package_used': getattr(parser, '_package_used', 'unknown'),
                'status': 'available'
            })
        else:
            available[language] = parser_info.copy()
            available[language].update({
                'status': 'unavailable',
                'reason': 'No tree-sitter package found'
            })
    
    # Add package availability info
    available['_package_info'] = package_info
    
    return available

def get_installation_help() -> str:
    """Get help for installing tree-sitter packages"""
    from .available_parsers import get_installation_recommendations, check_parser_availability
    
    package_info = check_parser_availability()
    
    if package_info['has_comprehensive']:
        missing_parsers = []
        available_parsers = discover_available_parsers()
        
        for language in AVAILABLE_PARSERS:
            if language not in available_parsers or available_parsers[language]['status'] == 'unavailable':
                missing_parsers.append(language)
        
        if missing_parsers:
            return f"""
🌲 Some parsers are not available: {', '.join(missing_parsers)}

This might be because:
1. The language is not included in your tree-sitter package
2. The language name mapping is different
3. Package needs to be updated

Currently working parsers: {len([p for p in available_parsers.values() if isinstance(p, dict) and p.get('status') == 'available'])}

You have: {', '.join(package_info['comprehensive_packages'])}
"""
        else:
            return "✅ All parsers are working correctly!"
    else:
        return get_installation_recommendations()

# Log available parsers on import
def _log_parser_status():
    """Log the status of available parsers"""
    try:
        available = discover_available_parsers()
        package_info = available.get('_package_info', {})
        
        working_parsers = [lang for lang, info in available.items() 
                          if lang != '_package_info' and isinstance(info, dict) and info.get('status') == 'available']
        
        failed_parsers = [lang for lang, info in available.items() 
                         if lang != '_package_info' and isinstance(info, dict) and info.get('status') == 'unavailable']
        
        if working_parsers:
            logger.info(f"✅ Parsers package initialized with {len(working_parsers)} working parsers")
            
            # Group by package used
            package_usage = {}
            for lang in working_parsers:
                package = available[lang].get('package_used', 'unknown')
                if package not in package_usage:
                    package_usage[package] = []
                package_usage[package].append(lang)
            
            for package, languages in package_usage.items():
                logger.info(f"   📦 {package}: {', '.join(languages)}")
        
        if failed_parsers:
            logger.info(f"❌ {len(failed_parsers)} parsers unavailable: {', '.join(failed_parsers)}")
        
        # Show package status
        if package_info.get('comprehensive_packages'):
            logger.info(f"📦 Comprehensive packages available: {', '.join(package_info['comprehensive_packages'])}")
        elif package_info.get('individual_packages'):
            logger.info(f"📦 Individual packages available: {len(package_info['individual_packages'])}")
        else:
            logger.warning("⚠️  No tree-sitter packages found")
            logger.info("💡 Install tree-sitter-languages for best support: pip install tree-sitter-languages")
        
    except Exception as e:
        logger.debug(f"Error checking parser status: {e}")
        logger.warning("⚠️  Parser status check failed - some parsers may not be available")

# Initialize on import
_log_parser_status()

# Export commonly used functions
__all__ = [
    'get_parser_info',
    'get_supported_languages', 
    'check_parser_availability',
    'load_parser',
    'discover_available_parsers',
    'get_installation_help',
    'AVAILABLE_PARSERS'
]