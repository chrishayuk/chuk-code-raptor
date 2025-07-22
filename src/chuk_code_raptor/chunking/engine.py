# src/chuk_code_raptor/chunking/engine.py - Updated Version
"""
Chunking Engine - YAML Configuration Edition
=============================================

Updated engine that uses the YAML-based parser registry for dynamic
parser discovery and management. No more hard-coding!
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import time

from chuk_code_raptor.core.models import FileInfo
from .config import ChunkingConfig
from .semantic_chunk import SemanticChunk, ContentType
from .base import BaseParser, ParseContext, UnsupportedLanguageError
from .parsers.registry import get_registry

logger = logging.getLogger(__name__)

class ChunkingEngine:
    """Main engine for chunking operations using YAML-based parser registry"""
    
    def __init__(self, config: ChunkingConfig = None):
        """
        Initialize chunking engine with YAML-based parser discovery.
        
        Args:
            config: ChunkingConfig to use (creates default if None)
        """
        self.config = config or ChunkingConfig()
        self.registry = get_registry()
        self.parsers: Dict[str, BaseParser] = {}
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_chunked': 0,
            'chunks_created': 0,
            'total_processing_time': 0.0,
            'chunker_usage': {},
            'errors': []
        }
        
        # Discover and register parsers using YAML config
        self._discover_and_register_parsers()
        
        total_parsers = len(self.parsers)
        logger.info(f"ChunkingEngine initialized with {total_parsers} parsers (YAML-configured)")
    
    def _discover_and_register_parsers(self):
        """Discover and register available parsers using YAML registry"""
        try:
            # Get available parsers from registry
            available_parsers = self.registry.discover_available_parsers(self.config)
            
            if available_parsers:
                self.parsers = available_parsers.copy()
                logger.info(f"✅ Registered {len(available_parsers)} parsers from YAML config")
                
                # Show registry statistics
                stats = self.registry.get_parser_stats()
                logger.info(f"   📊 Total configured: {stats['total_parsers']}")
                logger.info(f"   ✅ Available: {stats['available_parsers']}")
                logger.info(f"   ❌ Unavailable: {stats['unavailable_parsers']}")
                logger.info(f"   🌍 Languages: {stats['supported_languages']}")
                logger.info(f"   📄 Extensions: {stats['supported_extensions']}")
                
                # Show package status
                if stats['comprehensive_packages'] > 0:
                    logger.info(f"   📦 Using comprehensive tree-sitter packages")
                elif stats['individual_packages'] > 0:
                    logger.info(f"   📦 Using {stats['individual_packages']} individual packages")
                else:
                    logger.warning("   ⚠️  No tree-sitter packages found")
                
            else:
                logger.warning("⚠️  No parsers could be registered from YAML config")
                self._show_installation_help()
                
        except Exception as e:
            logger.error(f"❌ Failed to load parsers from YAML config: {e}")
            self._show_installation_help()
    
    def _show_installation_help(self):
        """Show installation help when no parsers are available"""
        try:
            help_text = self.registry.get_installation_help()
            logger.info("💡 Installation help:")
            for line in help_text.split('\n'):
                if line.strip():
                    logger.info(f"   {line}")
        except Exception:
            logger.info("💡 Install tree-sitter packages: pip install tree-sitter-languages")
    
    def chunk_file(self, file_path: str, language: str = None, content: str = None) -> List[SemanticChunk]:
        """
        Chunk a file into semantic chunks.
        
        Args:
            file_path: Path to file to chunk
            language: Programming language (auto-detected if None)
            content: File content (read from file if None)
            
        Returns:
            List of SemanticChunk objects
        """
        file_path = Path(file_path)
        
        # Read content if not provided
        if content is None:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Detect language if not provided
        if language is None:
            language = self._detect_language(file_path)
        
        return self.chunk_content(content, language, str(file_path))
    
    def chunk_content(self, content: str, language: str, file_path: str = "unknown") -> List[SemanticChunk]:
        """Chunk content string into semantic chunks"""
        if not content.strip():
            return []
        
        parser = self._get_parser(language, file_path)
        if not parser:
            # Provide helpful error message
            error_msg = self._build_helpful_error_message(language)
            self.stats['errors'].append(error_msg)
            logger.warning(f"⚠️  {error_msg}")
            raise UnsupportedLanguageError(error_msg)
        
        context = ParseContext(
            file_path=file_path,
            language=language,
            content_type=self._detect_content_type(file_path, language),
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size
        )
        
        try:
            start_time = time.time()
            chunks = parser.parse(content, context)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            self.stats['files_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            self._update_stats(parser, chunks, processing_time)
            
            logger.debug(f"✅ Chunked {file_path} into {len(chunks)} chunks using {parser.name}")
            return chunks
            
        except Exception as e:
            error = f"Error parsing {file_path} with {parser.name}: {e}"
            self.stats['errors'].append(error)
            logger.error(f"❌ {error}")
            raise
    
    def _get_parser(self, language: str, file_path: str = None) -> Optional[BaseParser]:
        """Get parser for language using YAML registry"""
        
        # Try direct language lookup
        parser = self.parsers.get(language)
        if parser:
            return parser
        
        # Try language alias lookup
        parser_name = self.registry.get_parser_for_language(language)
        if parser_name and parser_name in self.parsers:
            # Map the language to the actual parser for future lookups
            actual_parser = None
            for lang, p in self.parsers.items():
                if hasattr(p, '_config_name') and p._config_name == parser_name:
                    actual_parser = p
                    break
            if actual_parser:
                self.parsers[language] = actual_parser
                return actual_parser
        
        # Try extension-based lookup
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            parser_name = self.registry.get_parser_for_extension(file_ext)
            if parser_name:
                # Find parser instance by config name
                for lang, parser in self.parsers.items():
                    if hasattr(parser, '_config_name') and parser._config_name == parser_name:
                        # Cache this mapping
                        self.parsers[language] = parser
                        return parser
        
        return None
    
    def _detect_language(self, file_path: Path) -> str:
        """Pure YAML-based language detection"""
        extension = file_path.suffix.lower()
        
        # Use registry for extension mapping
        parser_name = self.registry.get_parser_for_extension(extension)
        if parser_name:
            parser_config = self.registry.get_parser_config(parser_name)
            if parser_config and parser_config.languages:
                detected = parser_config.languages[0]  # Use primary language
                logger.debug(f"🔍 Detected language '{detected}' for extension '{extension}' via YAML config")
                return detected
        
        # No hard-coded fallbacks - use 'unknown' and let registry handle it
        logger.debug(f"🔍 No YAML mapping for extension '{extension}', using 'unknown'")
        return 'unknown'
    
    def _detect_content_type(self, file_path: str, language: str) -> ContentType:
        """Pure YAML-based content type detection"""
        # Get parser config for this language
        parser_name = self.registry.get_parser_for_language(language)
        if parser_name:
            parser_config = self.registry.get_parser_config(parser_name)
            if parser_config:
                # Use parser-specific config or infer from language name
                primary_lang = parser_config.languages[0] if parser_config.languages else language
                
                # YAML-configurable content type mapping
                content_type_map = {
                    'markdown': ContentType.MARKDOWN,
                    'html': ContentType.HTML,
                    'xml': ContentType.XML,
                    'json': ContentType.JSON,
                    'yaml': ContentType.YAML,
                    'css': ContentType.CSS,
                    'python': ContentType.CODE,
                    'javascript': ContentType.CODE,
                    'typescript': ContentType.CODE,
                    'rust': ContentType.CODE,
                    'go': ContentType.CODE,
                    'java': ContentType.CODE,
                    'cpp': ContentType.CODE,
                    'c': ContentType.CODE,
                }
                
                return content_type_map.get(primary_lang, ContentType.PLAINTEXT)
        
        # Pure fallback to plaintext - no hard-coded mappings
        return ContentType.PLAINTEXT
    
    def _build_helpful_error_message(self, language: str) -> str:
        """Build a helpful error message for unsupported languages"""
        available_langs = sorted(self.get_supported_languages())
        error_msg = f"No parser available for language: {language}"
        
        if not available_langs:
            error_msg += "\n💡 No parsers available. Install tree-sitter packages:"
            error_msg += "\n   pip install tree-sitter-languages"
        else:
            # Show similar languages
            similar = [lang for lang in available_langs 
                      if language.lower() in lang.lower() or lang.lower() in language.lower()]
            if similar:
                error_msg += f"\n💡 Did you mean: {', '.join(similar)}"
            
            error_msg += f"\n💡 Available languages: {', '.join(available_langs[:10])}"
            if len(available_langs) > 10:
                error_msg += f" (and {len(available_langs) - 10} more)"
        
        return error_msg
    
    def _update_stats(self, parser: BaseParser, chunks: List[SemanticChunk], processing_time: float):
        """Update internal statistics"""
        self.stats['files_chunked'] += 1
        self.stats['total_processing_time'] += processing_time
        
        # Track parser usage
        parser_name = getattr(parser, 'name', parser.__class__.__name__)
        parser_type = getattr(parser, 'parser_type', 'unknown')
        config_name = getattr(parser, '_config_name', parser_name)
        
        parser_key = f"{parser_name} ({parser_type})"
        
        if parser_key not in self.stats['chunker_usage']:
            self.stats['chunker_usage'][parser_key] = {
                'files_processed': 0,
                'chunks_created': 0,
                'total_time': 0.0,
                'parser_type': parser_type,
                'config_name': config_name
            }
        
        self.stats['chunker_usage'][parser_key]['files_processed'] += 1
        self.stats['chunker_usage'][parser_key]['chunks_created'] += len(chunks)
        self.stats['chunker_usage'][parser_key]['total_time'] += processing_time
    
    def chunk_file_info(self, file_info: FileInfo, content: str = None) -> List[SemanticChunk]:
        """
        Chunk a file using FileInfo metadata.
        
        Args:
            file_info: FileInfo object with file metadata
            content: File content (read from file if None)
            
        Returns:
            List of SemanticChunk objects
        """
        return self.chunk_file(
            file_path=file_info.path,
            language=file_info.language,
            content=content
        )
    
    def get_supported_languages(self) -> List[str]:
        """Get list of all supported languages from YAML config"""
        return self.registry.get_supported_languages()
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions from YAML config"""
        return self.registry.get_supported_extensions()
    
    def can_chunk_language(self, language: str) -> bool:
        """Check if the engine can chunk the given language"""
        return language in self.parsers or self.registry.get_parser_for_language(language) is not None
    
    def can_chunk_file(self, file_path: str) -> bool:
        """Check if the engine can chunk the given file"""
        file_path = Path(file_path)
        language = self._detect_language(file_path)
        return self.can_chunk_language(language)
    
    def get_chunker_for_language(self, language: str, file_extension: str = None):
        """Get the parser that would be used for a language/extension"""
        return self._get_parser(language, f"test{file_extension}" if file_extension else None)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        stats = self.stats.copy()
        
        # Add registry information
        registry_stats = self.registry.get_parser_stats()
        stats.update({
            'available_chunkers': len(self.parsers),
            'configured_parsers': registry_stats['total_parsers'],
            'supported_languages': len(self.get_supported_languages()),
            'supported_extensions': len(self.get_supported_extensions()),
            'parser_types': registry_stats['parser_types'],
            'package_availability': registry_stats['package_availability']
        })
        
        # Calculate averages
        if stats['files_chunked'] > 0:
            stats['avg_chunks_per_file'] = stats['chunks_created'] / stats['files_chunked']
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['files_chunked']
        else:
            stats['avg_chunks_per_file'] = 0
            stats['avg_processing_time'] = 0
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = {
            'files_processed': 0,
            'files_chunked': 0,
            'chunks_created': 0,
            'total_processing_time': 0.0,
            'chunker_usage': {},
            'errors': []
        }
    
    def reload_parsers(self):
        """Reload parsers from YAML configuration"""
        logger.info("Reloading parsers from YAML configuration...")
        self.parsers.clear()
        self.registry.reload_config()
        self._discover_and_register_parsers()
        logger.info("Parser reload complete")
    
    def print_status(self):
        """Print engine status information"""
        stats = self.get_statistics()
        
        print("🚀 ChunkingEngine Status (YAML-Configured)")
        print("=" * 50)
        print(f"Configured parsers: {stats['configured_parsers']}")
        print(f"Available parsers: {stats['available_chunkers']}")
        print(f"Supported languages: {stats['supported_languages']}")
        print(f"Supported extensions: {stats['supported_extensions']}")
        
        if stats['parser_types']:
            print(f"\nParser types:")
            for parser_type, count in stats['parser_types'].items():
                print(f"  {parser_type}: {count}")
        
        package_info = stats['package_availability']
        if package_info.get('comprehensive'):
            print(f"\nComprehensive packages: {len(package_info['comprehensive'])}")
            for pkg in package_info['comprehensive']:
                print(f"  ✓ {pkg}")
        
        if package_info.get('individual'):
            print(f"\nIndividual packages: {len(package_info['individual'])}")
        
        if stats['files_chunked'] > 0:
            print(f"\nProcessing stats:")
            print(f"  Files processed: {stats['files_chunked']}")
            print(f"  Chunks created: {stats['chunks_created']}")
            print(f"  Avg chunks/file: {stats['avg_chunks_per_file']:.1f}")
            print(f"  Avg time/file: {stats['avg_processing_time']*1000:.1f}ms")
        
        print(f"\n💡 Configuration file: {self.registry.config_path}")
    
    def add_custom_parser(self, name: str, parser_class, languages: List[str], 
                         extensions: List[str], description: str = "Custom parser"):
        """
        Add a custom parser at runtime
        
        Args:
            name: Parser name
            parser_class: Parser class
            languages: Supported languages
            extensions: Supported file extensions
            description: Parser description
        """
        try:
            # Create parser instance
            parser = parser_class(self.config)
            
            # Set metadata
            parser.supported_languages = set(languages)
            parser.supported_extensions = set(extensions)
            parser.name = name
            parser._config_name = name
            
            # Register with engine
            for language in languages:
                self.parsers[language] = parser
            
            # Register with registry for future lookups
            self.registry.add_parser_runtime(name, parser, languages, extensions)
            
            logger.info(f"✅ Added custom parser '{name}' for languages: {', '.join(languages)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to add custom parser '{name}': {e}")
            raise

# Convenience function for backward compatibility
def create_engine(config: ChunkingConfig = None) -> ChunkingEngine:
    """Create a ChunkingEngine with YAML-based configuration"""
    return ChunkingEngine(config)