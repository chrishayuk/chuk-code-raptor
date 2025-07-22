# src/chuk_code_raptor/chunking/parsers/available_parsers.py
"""
Practical Parser Discovery - Enhanced Edition
=============================================

Discovers and creates parsers using available tree-sitter packages.
Now includes heuristic parsers for languages without tree-sitter support.

Auto-adapts to:
- tree-sitter-languages (comprehensive)
- tree-sitter-language-pack (alternative comprehensive)  
- Individual tree-sitter packages
- Heuristic parsers for unsupported languages (like LaTeX)
"""

import logging
from typing import Dict, Optional, Any, Set, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass 
class ParserInfo:
    """Information about an available parser"""
    parser_class: type
    package_used: str
    parser_type: str  # 'tree_sitter', 'heuristic'
    languages: Set[str]
    extensions: Set[str]
    description: str

def check_parser_availability() -> Dict[str, Any]:
    """Check what tree-sitter packages are available"""
    package_info = {
        'comprehensive_packages': [],
        'individual_packages': [],
        'has_comprehensive': False,
        'total_languages': 0
    }
    
    # Check comprehensive packages
    comprehensive_langs = set()
    
    # Try tree-sitter-languages
    try:
        from tree_sitter_languages import get_language
        test_languages = ['python', 'javascript', 'html', 'css', 'json', 'markdown']
        working_langs = []
        
        for lang in test_languages:
            try:
                get_language(lang)
                working_langs.append(lang)
            except Exception:
                pass
        
        if working_langs:
            package_info['comprehensive_packages'].append('tree-sitter-languages')
            comprehensive_langs.update(working_langs)
            logger.debug(f"tree-sitter-languages: {len(working_langs)} languages")
    except ImportError:
        pass
    
    # Try tree-sitter-language-pack
    if not comprehensive_langs:
        try:
            from tree_sitter_language_pack import get_language
            test_languages = [
                'python', 'javascript', 'typescript', 'html', 'css', 'json', 
                'markdown', 'rust', 'go', 'java', 'cpp', 'c', 'latex'
            ]
            working_langs = []
            
            for lang in test_languages:
                try:
                    get_language(lang)
                    working_langs.append(lang)
                except Exception:
                    pass
            
            if working_langs:
                package_info['comprehensive_packages'].append('tree-sitter-language-pack')
                comprehensive_langs.update(working_langs)
                logger.debug(f"tree-sitter-language-pack: {len(working_langs)} languages")
        except ImportError:
            pass
    
    # Check individual packages
    individual_packages = []
    test_individual = [
        'python', 'javascript', 'typescript', 'html', 'css', 'json', 
        'markdown', 'rust', 'go', 'java', 'cpp', 'c'
    ]
    
    for lang in test_individual:
        try:
            package_name = f"tree_sitter_{lang}"
            module = __import__(package_name, fromlist=['language'])
            if hasattr(module, 'language') and callable(getattr(module, 'language')):
                individual_packages.append(lang)
        except ImportError:
            pass
    
    package_info['individual_packages'] = individual_packages
    package_info['has_comprehensive'] = len(package_info['comprehensive_packages']) > 0
    package_info['total_languages'] = len(comprehensive_langs.union(set(individual_packages)))
    
    return package_info

def discover_available_parsers(config=None) -> Dict[str, Any]:
    """Discover all available parsers (tree-sitter + heuristic)"""
    from ..config import ChunkingConfig
    if config is None:
        config = ChunkingConfig()
    
    available_parsers = {}
    
    # Get package availability info
    package_info = check_parser_availability()
    
    # 1. Tree-sitter parsers
    tree_sitter_parsers = _discover_tree_sitter_parsers(config, package_info)
    available_parsers.update(tree_sitter_parsers)
    
    # 2. Heuristic parsers for languages without tree-sitter support
    heuristic_parsers = _discover_heuristic_parsers(config, available_parsers.keys())
    available_parsers.update(heuristic_parsers)
    
    logger.info(f"Discovered {len(available_parsers)} total parsers:")
    logger.info(f"  Tree-sitter: {len(tree_sitter_parsers)}")
    logger.info(f"  Heuristic: {len(heuristic_parsers)}")
    
    return available_parsers

def _discover_tree_sitter_parsers(config, package_info: Dict[str, Any]) -> Dict[str, Any]:
    """Discover tree-sitter based parsers"""
    parsers = {}
    
    # Language configurations for tree-sitter parsers
    language_configs = {
        'python': {
            'extensions': {'.py', '.pyx', '.pyi'},
            'class_name': 'PythonParser'
        },
        'javascript': {
            'extensions': {'.js', '.jsx', '.mjs', '.cjs'},
            'class_name': 'JavaScriptParser'
        },
        'typescript': {
            'extensions': {'.ts', '.tsx'},
            'class_name': 'TypeScriptParser'
        },
        'html': {
            'extensions': {'.html', '.htm'},
            'class_name': 'HTMLParser'
        },
        'css': {
            'extensions': {'.css', '.scss', '.sass', '.less'},
            'class_name': 'CSSParser'
        },
        'json': {
            'extensions': {'.json'},
            'class_name': 'JSONParser'
        },
        'markdown': {
            'extensions': {'.md', '.markdown', '.mdown', '.mkd'},
            'class_name': 'MarkdownParser'
        },
        'rust': {
            'extensions': {'.rs'},
            'class_name': 'RustParser'
        },
        'go': {
            'extensions': {'.go'},
            'class_name': 'GoParser'
        },
        'java': {
            'extensions': {'.java'},
            'class_name': 'JavaParser'
        },
        'cpp': {
            'extensions': {'.cpp', '.cxx', '.cc', '.hpp'},
            'class_name': 'CppParser'
        },
        'c': {
            'extensions': {'.c', '.h'},
            'class_name': 'CParser'
        }
    }
    
    # Try to create parsers for available languages
    for language, config_info in language_configs.items():
        parser = _create_tree_sitter_parser(language, config_info, config, package_info)
        if parser:
            parsers[language] = parser
    
    return parsers

def _create_tree_sitter_parser(language: str, lang_config: Dict, config, package_info: Dict) -> Optional[Any]:
    """Create a tree-sitter parser for a specific language"""
    
    # Check if language is available
    available_via_comprehensive = (package_info['has_comprehensive'] and 
                                 _test_comprehensive_language(language))
    available_via_individual = language in package_info.get('individual_packages', [])
    
    if not (available_via_comprehensive or available_via_individual):
        return None
    
    try:
        # Create the parser class dynamically
        parser_class = _create_available_tree_sitter_parser_class(
            language, 
            lang_config['class_name'],
            lang_config['extensions']
        )
        
        # Try to instantiate it
        parser = parser_class(config)
        
        # Verify it has a working language
        if hasattr(parser, '_language') and parser._language:
            return parser
        else:
            logger.debug(f"Parser {language} created but no working language")
            return None
            
    except Exception as e:
        logger.debug(f"Failed to create {language} parser: {e}")
        return None

def _test_comprehensive_language(language: str) -> bool:
    """Test if a language is available via comprehensive packages"""
    # Try tree-sitter-languages first
    try:
        from tree_sitter_languages import get_language
        get_language(language)
        return True
    except:
        pass
    
    # Try tree-sitter-language-pack
    try:
        from tree_sitter_language_pack import get_language
        get_language(language)
        return True
    except:
        pass
    
    return False

def _create_available_tree_sitter_parser_class(language: str, class_name: str, extensions: Set[str]):
    """Create a parser class that uses available packages"""
    
    class AvailableTreeSitterParser:
        """Parser that uses whichever tree-sitter package is available"""
        
        def __init__(self, config):
            self.config = config
            self.name = class_name
            self.parser_type = "tree_sitter_available"
            self.supported_languages = {language}
            self.supported_extensions = extensions
            
            # Initialize tree-sitter
            self._language = None
            self._package_used = None
            self._initialize_language()
        
        def _initialize_language(self):
            """Initialize with available package"""
            import tree_sitter
            
            # Try comprehensive packages first
            language_obj = None
            
            # Method 1: tree-sitter-languages
            try:
                from tree_sitter_languages import get_language
                language_obj = get_language(language)
                self._package_used = "tree-sitter-languages"
                logger.debug(f"{language}: using tree-sitter-languages")
            except Exception as e:
                logger.debug(f"{language}: tree-sitter-languages failed: {e}")
            
            # Method 2: tree-sitter-language-pack
            if not language_obj:
                try:
                    from tree_sitter_language_pack import get_language
                    language_obj = get_language(language)
                    self._package_used = "tree-sitter-language-pack"
                    logger.debug(f"{language}: using tree-sitter-language-pack")
                except Exception as e:
                    logger.debug(f"{language}: tree-sitter-language-pack failed: {e}")
            
            # Method 3: Individual package
            if not language_obj:
                try:
                    package_name = f"tree_sitter_{language}"
                    module = __import__(package_name, fromlist=['language'])
                    language_func = getattr(module, 'language')
                    language_obj = tree_sitter.Language(language_func())
                    self._package_used = package_name
                    logger.debug(f"{language}: using {package_name}")
                except Exception as e:
                    logger.debug(f"{language}: individual package failed: {e}")
            
            if language_obj:
                self._language = language_obj
                self.parser = tree_sitter.Parser(language_obj)
                logger.debug(f"✅ {language} parser ready via {self._package_used}")
            else:
                logger.warning(f"❌ Could not initialize {language} parser")
                raise ImportError(f"No working tree-sitter package for {language}")
        
        def can_parse(self, lang: str, file_extension: str) -> bool:
            """Check if this parser can handle the language/extension"""
            return (lang in self.supported_languages or 
                    file_extension in self.supported_extensions)
        
        def parse(self, content: str, context) -> List:
            """Parse content - basic implementation"""
            if not self._language or not content.strip():
                return []
            
            try:
                tree = self.parser.parse(bytes(content, 'utf8'))
                # For now, just create a single chunk
                # This could be enhanced with proper AST traversal
                from ..semantic_chunk import create_chunk_id, SemanticChunk
                from chuk_code_raptor.core.models import ChunkType
                from ..semantic_chunk import ContentType
                
                chunk_id = create_chunk_id(context.file_path, 1, ChunkType.TEXT_BLOCK, "content")
                
                chunk = SemanticChunk(
                    id=chunk_id,
                    file_path=context.file_path,
                    content=content,
                    start_line=1,
                    end_line=len(content.split('\n')),
                    content_type=context.content_type,
                    chunk_type=ChunkType.TEXT_BLOCK,
                    language=context.language,
                    importance_score=0.5,
                    metadata={
                        'parser': self.name,
                        'parser_type': self.parser_type,
                        'package_used': self._package_used,
                        'tree_sitter_available': True
                    }
                )
                
                chunk.add_tag('tree_sitter_parsed', source='tree_sitter_available')
                chunk.add_tag(f'language_{language}', source='tree_sitter_available')
                
                return [chunk]
                
            except Exception as e:
                logger.error(f"Error parsing {language} content: {e}")
                return []
    
    return AvailableTreeSitterParser

def _discover_heuristic_parsers(config, existing_languages: Set[str]) -> Dict[str, Any]:
    """Discover heuristic parsers for languages without tree-sitter support"""
    parsers = {}
    
    # LaTeX heuristic parser (since tree-sitter LaTeX isn't available as Python package)
    if 'latex' not in existing_languages:
        try:
            from .latex import HeuristicLaTeXParser
            latex_parser = HeuristicLaTeXParser(config)
            parsers['latex'] = latex_parser
            logger.debug("✅ LaTeX heuristic parser available")
        except ImportError as e:
            logger.debug(f"LaTeX heuristic parser not available: {e}")
    
    # Could add more heuristic parsers here for other languages
    # For example: YAML, TOML, XML parsers if tree-sitter versions aren't available
    
    return parsers

def get_installation_recommendations() -> str:
    """Get recommendations for installing tree-sitter packages"""
    package_info = check_parser_availability()
    
    if package_info['has_comprehensive']:
        return f"""
✅ You have comprehensive tree-sitter support!
📦 Available: {', '.join(package_info['comprehensive_packages'])}
🚀 {package_info['total_languages']} languages supported

Additional individual packages: {len(package_info['individual_packages'])}
"""
    elif package_info['individual_packages']:
        return f"""
⚠️  Limited tree-sitter support detected
📦 Individual packages: {len(package_info['individual_packages'])} languages
💡 For better support, install a comprehensive package:

Recommended:
    pip install tree-sitter-language-pack

Alternative:
    pip install tree-sitter-languages

This will provide support for many more languages including LaTeX.
"""
    else:
        return """
❌ No tree-sitter packages found

Install comprehensive support:
    pip install tree-sitter-language-pack

Or install specific languages:
    pip install tree-sitter-python tree-sitter-javascript
    pip install tree-sitter-html tree-sitter-css tree-sitter-json

Note: LaTeX requires a comprehensive package.
"""

# Keep existing compatibility functions
def create_python_parser(config):
    """Create Python parser using available packages"""
    return _create_tree_sitter_parser('python', {
        'extensions': {'.py', '.pyx', '.pyi'},
        'class_name': 'PythonParser'
    }, config, check_parser_availability())

def create_javascript_parser(config):
    """Create JavaScript parser using available packages"""
    return _create_tree_sitter_parser('javascript', {
        'extensions': {'.js', '.jsx', '.mjs', '.cjs'},
        'class_name': 'JavaScriptParser'
    }, config, check_parser_availability())

def create_html_parser(config):
    """Create HTML parser using available packages"""
    return _create_tree_sitter_parser('html', {
        'extensions': {'.html', '.htm'},
        'class_name': 'HTMLParser'
    }, config, check_parser_availability())

def create_json_parser(config):
    """Create JSON parser using available packages"""
    return _create_tree_sitter_parser('json', {
        'extensions': {'.json'},
        'class_name': 'JSONParser'
    }, config, check_parser_availability())

def create_markdown_parser(config):
    """Create Markdown parser using available packages"""
    return _create_tree_sitter_parser('markdown', {
        'extensions': {'.md', '.markdown', '.mdown', '.mkd'},
        'class_name': 'MarkdownParser'
    }, config, check_parser_availability())

def create_latex_parser(config):
    """Create LaTeX parser (heuristic-based)"""
    try:
        from .latex import HeuristicLaTeXParser
        return HeuristicLaTeXParser(config)
    except ImportError:
        return None