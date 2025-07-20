# src/chuk_code_raptor/chunking/parsers/__init__.py
"""
Parsers Package
===============

Clean parser implementations for the chunking system.
Each parser uses either tree-sitter or heuristic approaches based on what works best.
"""

import logging
from typing import Dict, List, Type, Optional

logger = logging.getLogger(__name__)

# Available parser configurations - RUST TEMPORARILY DISABLED DUE TO VERSION ISSUE
AVAILABLE_PARSERS = {
    'python': {
        'module': 'chuk_code_raptor.chunking.parsers.python',
        'class': 'PythonParser',
        'type': 'tree_sitter',
        'description': 'Python parser using tree-sitter AST analysis'
    },
    'markdown': {
        'module': 'chuk_code_raptor.chunking.parsers.markdown', 
        'class': 'MarkdownParser',
        'type': 'tree_sitter',
        'description': 'Markdown parser using tree-sitter with semantic analysis'
    },
    'javascript': {
        'module': 'chuk_code_raptor.chunking.parsers.javascript',
        'class': 'JavaScriptParser', 
        'type': 'tree_sitter',
        'description': 'JavaScript/TypeScript parser using tree-sitter'
    },
    # 'rust': {
    #     'module': 'chuk_code_raptor.chunking.parsers.rust',
    #     'class': 'RustParser',
    #     'type': 'tree_sitter', 
    #     'description': 'Rust parser using tree-sitter AST analysis'
    # },
    'json': {
        'module': 'chuk_code_raptor.chunking.parsers.json',
        'class': 'JSONParser',
        'type': 'tree_sitter',
        'description': 'JSON parser using tree-sitter'
    },
    'html': {
        'module': 'chuk_code_raptor.chunking.parsers.html',
        'class': 'HTMLParser', 
        'type': 'tree_sitter',
        'description': 'HTML parser using tree-sitter'
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
        
        return True
        
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
        return parser_class(config)
        
    except Exception as e:
        logger.error(f"Failed to load parser for {language}: {e}")
        return None

def discover_available_parsers() -> Dict[str, Dict]:
    """Discover which parsers are actually available"""
    available = {}
    
    for language, parser_info in AVAILABLE_PARSERS.items():
        if check_parser_availability(language):
            available[language] = parser_info.copy()
            
            # Add runtime information
            try:
                parser = load_parser(language)
                if parser:
                    available[language].update({
                        'supported_languages': list(parser.supported_languages),
                        'supported_extensions': list(parser.supported_extensions),
                        'parser_name': parser.name,
                        'parser_type': parser.parser_type
                    })
            except Exception as e:
                logger.debug(f"Could not get runtime info for {language}: {e}")
    
    return available

# Log available parsers on import
def _log_parser_status():
    """Log the status of available parsers"""
    available = discover_available_parsers()
    
    if available:
        logger.info(f"Parsers package initialized with {len(available)} available parsers")
        for language, info in available.items():
            logger.debug(f"  {language}: {info['type']} parser ({info.get('parser_name', 'unknown')})")
    else:
        logger.warning("No parsers available - check tree-sitter dependencies")

# Initialize on import
_log_parser_status()

# Export commonly used functions
__all__ = [
    'get_parser_info',
    'get_supported_languages', 
    'check_parser_availability',
    'load_parser',
    'discover_available_parsers',
    'AVAILABLE_PARSERS'
]