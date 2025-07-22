# src/chuk_code_raptor/chunking/parsers/available_parsers.py
"""
Available Tree-sitter Parsers - DISCOVERY SYSTEM ONLY
=====================================================

Clean parser discovery system that imports parsers from separate files.
Uses individual tree-sitter packages since tree-sitter-languages has API issues.

This file ONLY handles:
- Package availability checking
- Parser discovery logic
- Base class for working with individual packages

Individual parsers are defined in separate files:
- python.py -> PythonParser
- javascript.py -> JavaScriptParser
- etc.
"""

from typing import Dict, List, Optional, Any, Set
import logging
from abc import abstractmethod

from ..tree_sitter_base import TreeSitterParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

def get_installation_recommendations() -> str:
    """Get recommendations for installing tree-sitter packages"""
    return """
🌲 Tree-sitter Parser Installation Recommendations:

CURRENT STATUS: Individual packages work best due to API compatibility.

For reliable parsing, install individual packages:
   pip install tree-sitter-python
   pip install tree-sitter-javascript  
   pip install tree-sitter-html
   pip install tree-sitter-css
   pip install tree-sitter-json

Note: tree-sitter-languages currently has API compatibility issues.
Individual packages are more reliable and work perfectly.

Supported languages with individual packages:
- Python: ✅ tree-sitter-python
- JavaScript: ✅ tree-sitter-javascript  
- HTML: ✅ tree-sitter-html
- CSS: ✅ tree-sitter-css
- JSON: ✅ tree-sitter-json

For maximum reliability, use individual packages.
"""

def check_parser_availability() -> Dict[str, Any]:
    """Check what tree-sitter packages are available - focus on individual packages"""
    comprehensive_packages = []
    
    # Check tree-sitter-languages (but we know it has issues)
    try:
        import tree_sitter_languages
        comprehensive_packages.append("tree-sitter-languages")
        logger.debug("✅ tree-sitter-languages found (but has API issues)")
    except ImportError:
        logger.debug("❌ tree-sitter-languages not available")
    
    # Check individual packages (these work!)
    individual_packages = []
    working_packages = []
    
    packages_to_check = [
        'python', 'javascript', 'typescript', 'html', 'css', 'json', 
        'markdown', 'rust', 'go', 'java', 'cpp', 'c'
    ]
    
    for lang in packages_to_check:
        try:
            package_name = f"tree_sitter_{lang}"
            module = __import__(package_name, fromlist=['language'])
            
            # Check if it has a language function
            if hasattr(module, 'language') and callable(getattr(module, 'language')):
                individual_packages.append(f'tree-sitter-{lang}')
                working_packages.append(lang)
                logger.debug(f"✅ tree-sitter-{lang} available and working")
            else:
                logger.debug(f"⚠️  tree-sitter-{lang} found but no language function")
                
        except ImportError:
            logger.debug(f"❌ tree-sitter-{lang} not available")
        except Exception as e:
            logger.debug(f"❌ tree-sitter-{lang} error: {e}")
    
    return {
        'comprehensive_packages': comprehensive_packages,
        'individual_packages': individual_packages,
        'working_packages': working_packages,
        'total_packages': len(comprehensive_packages) + len(individual_packages),
        'has_comprehensive': len(comprehensive_packages) > 0,
        'has_working_individual': len(working_packages) > 0,
        'recommendations': get_installation_recommendations() if not working_packages else None
    }

class WorkingTreeSitterParser(TreeSitterParser):
    """
    Base class for tree-sitter parsers that use individual packages.
    
    This class handles the working pattern for individual tree-sitter packages.
    Subclasses should be defined in separate files (python.py, javascript.py, etc.)
    """
    
    def __init__(self, config):
        # Store config first
        self.config = config
        self.supported_languages: Set[str] = set()
        self.supported_extensions: Set[str] = set()
        self.parser_type: str = "tree_sitter_individual"
        self.name = self.__class__.__name__
        self._language = None
        self._package_used = None
        
        # Language-specific settings - must be set by subclasses BEFORE calling parent init
        if not hasattr(self, 'language_name') or not self.language_name:
            raise ValueError(f"{self.__class__.__name__} must set language_name before calling super().__init__()")
        
        # Try to initialize with individual packages
        if self._try_initialize_individual():
            # Only call parent __init__ if we successfully got a language
            try:
                super().__init__(config)
                logger.debug(f"✅ {self.name} fully initialized using individual package")
            except Exception as e:
                logger.error(f"❌ {self.name} failed TreeSitterParser init: {e}")
                self._language = None
                self._package_used = None
        else:
            logger.debug(f"❌ {self.name} failed to initialize - no individual package available")
    
    def _try_initialize_individual(self) -> bool:
        """Try to initialize using individual tree-sitter-{language} package"""
        try:
            import tree_sitter
            package_name = f"tree_sitter_{self.language_name}"
            
            try:
                # Import the specific package
                module = __import__(package_name, fromlist=['language'])
                logger.debug(f"✅ Imported {package_name}")
                
                # Get the language function
                if hasattr(module, 'language') and callable(getattr(module, 'language')):
                    language_func = getattr(module, 'language')
                    
                    # Call the function to get the language object
                    language_obj = language_func()
                    self._language = tree_sitter.Language(language_obj)
                    self._package_used = package_name
                    
                    logger.debug(f"✅ Got {self.language_name} from {package_name}")
                    return True
                else:
                    logger.debug(f"❌ No language function in {package_name}")
                    return False
                    
            except ImportError as e:
                logger.debug(f"❌ Individual package {package_name} not available: {e}")
                return False
                
        except ImportError:
            logger.debug("tree-sitter base package not available")
            return False
        except Exception as e:
            logger.debug(f"❌ Individual package failed for {self.language_name}: {e}")
            return False
    
    def _get_tree_sitter_language(self):
        """Get the initialized language"""
        if self._language is not None:
            return self._language
        else:
            raise ImportError(f"No tree-sitter language available for {self.language_name}")
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        """Check if we can parse this language"""
        return (self._language is not None and 
                (language in self.supported_languages or 
                 file_extension in self.supported_extensions))

def _load_parser_classes():
    """Load parser classes from individual files"""
    parser_classes = {}
    
    # Try to import specific parser classes from separate files
    parser_modules = [
        ('python', 'python', 'PythonParser'),
        ('javascript', 'javascript', 'JavaScriptParser'),
        ('typescript', 'typescript', 'TypeScriptParser'),
        ('markdown', 'markdown', 'MarkdownParser'),
        ('latex', 'latex', 'LaTeXParser'),
        ('tex', 'latex', 'LaTeXParser'),  # Alias for LaTeX
        ('html', 'html', 'HTMLParser'),
        ('css', 'css', 'CSSParser'),
        ('json', 'json', 'JSONParser'),
    ]
    
    for language, module_name, class_name in parser_modules:
        try:
            # Import from separate file
            module = __import__(f'chuk_code_raptor.chunking.parsers.{module_name}', fromlist=[class_name])
            parser_class = getattr(module, class_name)
            parser_classes[language] = parser_class
            logger.debug(f"✅ Loaded {class_name} from {module_name}.py")
        except ImportError as e:
            logger.debug(f"❌ Could not import {class_name} from {module_name}.py: {e}")
        except AttributeError as e:
            logger.debug(f"❌ {class_name} not found in {module_name}.py: {e}")
        except Exception as e:
            logger.debug(f"❌ Error loading {class_name}: {e}")
    
    return parser_classes

def discover_available_parsers(config=None) -> Dict[str, TreeSitterParser]:
    """
    Discover which parsers are actually available and return working instances.
    
    This imports parsers from separate files and returns working instances.
    """
    from ..config import ChunkingConfig
    config = config or ChunkingConfig()
    
    # Load parser classes from separate files
    parser_classes = _load_parser_classes()
    
    if not parser_classes:
        logger.warning("⚠️  No parser classes could be loaded from separate files")
        return {}
    
    available_parsers = {}
    successful_parsers = []
    failed_parsers = []
    
    logger.info(f"🔍 Discovering parsers from {len(parser_classes)} loaded classes...")
    
    for language, parser_class in parser_classes.items():
        try:
            # Try to create an instance of the parser
            parser_instance = parser_class(config)
            
            # Check if it successfully initialized (has a working language)
            if hasattr(parser_instance, '_language') and parser_instance._language is not None:
                # Register this parser for all its supported languages
                for supported_lang in parser_instance.supported_languages:
                    available_parsers[supported_lang] = parser_instance
                
                successful_parsers.append({
                    'name': parser_instance.name,
                    'languages': list(parser_instance.supported_languages),
                    'package': getattr(parser_instance, '_package_used', 'unknown')
                })
                
                logger.debug(f"✅ {parser_instance.name} available for {list(parser_instance.supported_languages)}")
            else:
                failed_parsers.append({
                    'name': parser_class.__name__,
                    'reason': 'No tree-sitter package available'
                })
                logger.debug(f"❌ {parser_class.__name__} failed - no language available")
                
        except Exception as e:
            failed_parsers.append({
                'name': parser_class.__name__,
                'reason': str(e)
            })
            logger.debug(f"❌ {parser_class.__name__} failed with error: {e}")
    
    # Log summary
    if successful_parsers:
        logger.info(f"✅ Successfully initialized {len(successful_parsers)} tree-sitter parsers:")
        for parser_info in successful_parsers:
            logger.info(f"   {parser_info['name']}: {', '.join(parser_info['languages'])} (via {parser_info['package']})")
    
    if failed_parsers:
        logger.info(f"❌ Failed to initialize {len(failed_parsers)} parsers:")
        for parser_info in failed_parsers:
            logger.debug(f"   {parser_info['name']}: {parser_info['reason']}")
    
    if not available_parsers:
        logger.warning("⚠️  No tree-sitter parsers could be initialized")
        logger.info("💡 Install individual packages: pip install tree-sitter-python tree-sitter-javascript")
    
    return available_parsers

def get_installation_help() -> str:
    """Get help for installing tree-sitter packages"""
    availability = check_parser_availability()
    
    if availability['has_working_individual']:
        working_count = len(availability['working_packages'])
        return f"""
✅ {working_count} individual tree-sitter packages working: {', '.join(availability['working_packages'])}

Your setup is working! Individual packages are more reliable than tree-sitter-languages.

To add more languages, install individual packages:
   pip install tree-sitter-rust
   pip install tree-sitter-go
   pip install tree-sitter-java

Note: tree-sitter-languages has API compatibility issues, so individual packages are preferred.
"""
    else:
        return get_installation_recommendations()

# Legacy compatibility - keep the old class name as an alias
AvailableTreeSitterParser = WorkingTreeSitterParser

if __name__ == "__main__":
    # Test the working discovery system
    print("🔍 Testing Working Tree-sitter Parser Discovery")
    print("=" * 50)
    
    # Check package availability
    availability = check_parser_availability()
    print(f"Package availability:")
    print(f"  Individual packages: {availability['individual_packages']}")
    print(f"  Working packages: {availability['working_packages']}")
    print(f"  Total packages: {availability['total_packages']}")
    
    # Test loading parser classes
    print(f"\nLoading parser classes from separate files...")
    parser_classes = _load_parser_classes()
    print(f"Loaded classes: {list(parser_classes.keys())}")
    
    # Test parser discovery
    print(f"\nDiscovering parsers...")
    available_parsers = discover_available_parsers()
    
    print(f"\nResults:")
    print(f"  Available parsers: {len(available_parsers)}")
    for lang, parser in available_parsers.items():
        package = getattr(parser, '_package_used', 'unknown')
        print(f"    {lang}: {parser.name} (via {package})")
    
    if not available_parsers:
        print("\n" + get_installation_help())