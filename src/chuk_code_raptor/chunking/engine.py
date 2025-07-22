# src/chuk_code_raptor/chunking/engine.py
"""
Chunking Engine - Practical Edition
===================================

Main engine that uses the practical parser discovery system.
Automatically adapts to available tree-sitter packages.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import time

from chuk_code_raptor.core.models import FileInfo
from .config import ChunkingConfig
from .semantic_chunk import SemanticChunk, ContentType
from .base import BaseParser, ParseContext, UnsupportedLanguageError

logger = logging.getLogger(__name__)

class ChunkingEngine:
    """Main engine for chunking operations using practical parser discovery"""
    
    def __init__(self, config: ChunkingConfig = None):
        """
        Initialize chunking engine with automatic parser discovery.
        
        Args:
            config: ChunkingConfig to use (creates default if None)
        """
        self.config = config or ChunkingConfig()
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
        
        # Discover and register parsers
        self._discover_and_register_parsers()
        
        total_parsers = len(self.parsers)
        logger.info(f"ChunkingEngine initialized with {total_parsers} parsers")
    
    def _discover_and_register_parsers(self):
        """Discover and register available parsers using the working discovery system"""
        try:
            from .parsers.available_parsers import discover_available_parsers
            
            # Get available parsers from the working discovery system
            available_parsers = discover_available_parsers(self.config)
            
            if available_parsers:
                # Register working parsers directly - they're already mapped by language
                self.parsers = available_parsers.copy()
                
                logger.info(f"✅ Registered {len(available_parsers)} working parsers")
                
                # Show what packages are being used
                package_usage = {}
                seen_parsers = set()
                
                for lang, parser in available_parsers.items():
                    parser_id = id(parser)
                    if parser_id not in seen_parsers:
                        # Check both TreeSitterParser and AvailableTreeSitterParser patterns
                        if hasattr(parser, '_package_used'):
                            package = parser._package_used
                        else:
                            # Infer from language name for TreeSitterParser
                            package = f"tree_sitter_{lang}"
                        
                        if package not in package_usage:
                            package_usage[package] = []
                        
                        # Get all languages this parser supports
                        parser_languages = list(getattr(parser, 'supported_languages', [lang]))
                        package_usage[package].extend(parser_languages)
                        seen_parsers.add(parser_id)
                
                for package, languages in package_usage.items():
                    unique_langs = list(set(languages))
                    logger.info(f"   📦 {package}: {', '.join(unique_langs[:5])}")
            else:
                logger.warning("⚠️  No parsers could be registered")
                self._show_installation_help()
                
        except ImportError as e:
            logger.error(f"❌ Parser discovery not available: {e}")
            self._show_installation_help()
        except Exception as e:
            logger.error(f"❌ Parser discovery failed: {e}")
            self._show_installation_help()
    
    def _show_installation_help(self):
        """Show installation help when no parsers are available"""
        try:
            from .parsers.available_parsers import get_installation_help
            help_text = get_installation_help()
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
        
        parser = self._get_parser(language)
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
    
    def _build_helpful_error_message(self, language: str) -> str:
        """Build a helpful error message for unsupported languages"""
        available_langs = sorted(self.get_supported_languages())
        error_msg = f"No parser available for language: {language}"
        
        if not available_langs:
            error_msg += "\n💡 No parsers available. Install tree-sitter packages:"
            error_msg += "\n   pip install tree-sitter-python tree-sitter-javascript"
            error_msg += "\n   pip install tree-sitter-html tree-sitter-css tree-sitter-json"
        else:
            # Show similar languages
            similar = [lang for lang in available_langs if language.lower() in lang.lower() or lang.lower() in language.lower()]
            if similar:
                error_msg += f"\n💡 Did you mean: {', '.join(similar)}"
            
            error_msg += f"\n💡 Available languages: {', '.join(available_langs[:10])}"
            if len(available_langs) > 10:
                error_msg += f" (and {len(available_langs) - 10} more)"
            
            # Check if this might be a missing package issue
            if language in ['latex', 'tex'] and language not in available_langs:
                error_msg += f"\n💡 LaTeX requires: pip install tree-sitter-latex"
            elif language in ['typescript'] and language not in available_langs:
                error_msg += f"\n💡 TypeScript requires: pip install tree-sitter-typescript"
        
        return error_msg
    
    def _get_parser(self, language: str) -> Optional[BaseParser]:
        """Get parser for language with enhanced discovery"""
        parser = self.parsers.get(language)
        if parser:
            return parser
        
        # Try alternative language names
        language_aliases = {
            'tex': 'latex',
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'md': 'markdown',
            'yml': 'yaml',
        }
        
        alias = language_aliases.get(language)
        if alias and alias in self.parsers:
            return self.parsers[alias]
        
        return None
    
    def _detect_language(self, file_path: Path) -> str:
        """Enhanced language detection"""
        extension = file_path.suffix.lower()
        
        # Enhanced extension mapping
        extension_map = {
            '.py': 'python', '.pyx': 'python', '.pyi': 'python',
            '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript',
            '.md': 'markdown', '.markdown': 'markdown', '.mdown': 'markdown', '.mkd': 'markdown',
            '.html': 'html', '.htm': 'html',
            '.css': 'css', '.scss': 'css', '.sass': 'css', '.less': 'css',
            '.json': 'json',
            '.yaml': 'yaml', '.yml': 'yaml',
            '.tex': 'latex', '.latex': 'latex', '.ltx': 'latex',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp', '.cxx': 'cpp', '.cc': 'cpp',
            '.c': 'c',
            '.h': 'c', '.hpp': 'cpp',
            '.php': 'php',
            '.rb': 'ruby',
            '.sh': 'bash', '.bash': 'bash',
            '.lua': 'lua',
            '.vim': 'vim',
            '.sql': 'sql',
            '.xml': 'xml',
            '.toml': 'toml',
        }
        
        detected = extension_map.get(extension, 'text')
        logger.debug(f"🔍 Detected language '{detected}' for extension '{extension}'")
        return detected
    
    def _detect_content_type(self, file_path: str, language: str) -> ContentType:
        """Enhanced content type detection"""
        extension = Path(file_path).suffix.lower()
        
        if extension in {'.md', '.markdown', '.mdown', '.mkd'}:
            return ContentType.MARKDOWN
        elif extension in {'.html', '.htm'}:
            return ContentType.HTML
        elif extension == '.json':
            return ContentType.JSON
        elif extension in {'.yaml', '.yml'}:
            return ContentType.YAML
        elif extension in {'.xml'}:
            return ContentType.XML
        elif language in {'python', 'javascript', 'typescript', 'rust', 'go', 'java', 'cpp', 'c'}:
            return ContentType.CODE
        elif extension in {'.css', '.scss', '.sass', '.less'}:
            return ContentType.CSS
        else:
            return ContentType.PLAINTEXT
    
    def _update_stats(self, parser: BaseParser, chunks: List[SemanticChunk], processing_time: float):
        """Update internal statistics"""
        self.stats['files_chunked'] += 1
        self.stats['total_processing_time'] += processing_time
        
        # Track parser usage
        parser_name = getattr(parser, 'name', parser.__class__.__name__)
        parser_type = getattr(parser, 'parser_type', 'tree_sitter')
        
        # Get package info for both parser types
        if hasattr(parser, '_package_used'):
            package_used = parser._package_used
        else:
            # Infer from supported languages for TreeSitterParser
            languages = getattr(parser, 'supported_languages', set())
            if languages:
                first_lang = next(iter(languages))
                package_used = f"tree_sitter_{first_lang}"
            else:
                package_used = 'unknown'
        
        parser_key = f"{parser_name} ({parser_type})"
        
        if parser_key not in self.stats['chunker_usage']:
            self.stats['chunker_usage'][parser_key] = {
                'files_processed': 0,
                'chunks_created': 0,
                'total_time': 0.0,
                'parser_type': parser_type,
                'package_used': package_used
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
        """Get list of all supported languages"""
        return sorted(list(self.parsers.keys()))
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions"""
        extensions = set()
        seen_parsers = set()
        for parser in self.parsers.values():
            parser_id = id(parser)
            if parser_id not in seen_parsers:
                extensions.update(getattr(parser, 'supported_extensions', set()))
                seen_parsers.add(parser_id)
        return sorted(list(extensions))
    
    def can_chunk_language(self, language: str) -> bool:
        """Check if the engine can chunk the given language"""
        return language in self.parsers
    
    def can_chunk_file(self, file_path: str) -> bool:
        """Check if the engine can chunk the given file"""
        file_path = Path(file_path)
        language = self._detect_language(file_path)
        return self.can_chunk_language(language)
    
    def get_chunker_for_language(self, language: str, file_extension: str = None):
        """Get the parser that would be used for a language/extension"""
        return self._get_parser(language)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        stats = self.stats.copy()
        
        # Add parser information
        unique_parsers = set(self.parsers.values())
        stats['available_chunkers'] = len(unique_parsers)
        stats['supported_languages'] = len(self.get_supported_languages())
        stats['supported_extensions'] = len(self.get_supported_extensions())
        
        # Parser type breakdown
        parser_types = {}
        package_usage = {}
        seen_parsers = set()
        
        for parser in unique_parsers:
            parser_id = id(parser)
            if parser_id not in seen_parsers:
                parser_type = getattr(parser, 'parser_type', 'tree_sitter')
                
                # Get package info for both parser types
                if hasattr(parser, '_package_used'):
                    package_used = parser._package_used
                else:
                    # Infer from supported languages for TreeSitterParser
                    languages = getattr(parser, 'supported_languages', set())
                    if languages:
                        first_lang = next(iter(languages))
                        package_used = f"tree_sitter_{first_lang}"
                    else:
                        package_used = 'unknown'
                
                parser_types[parser_type] = parser_types.get(parser_type, 0) + 1
                package_usage[package_used] = package_usage.get(package_used, 0) + 1
                seen_parsers.add(parser_id)
        
        stats['parser_types'] = parser_types
        stats['package_usage'] = package_usage
        
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
    
    def print_status(self):
        """Print engine status information"""
        stats = self.get_statistics()
        
        print("🚀 ChunkingEngine Status")
        print("=" * 40)
        print(f"Supported languages: {stats['supported_languages']}")
        print(f"Supported extensions: {stats['supported_extensions']}")
        print(f"Available parsers: {stats['available_chunkers']}")
        
        if stats['parser_types']:
            print(f"\nParser types:")
            for parser_type, count in stats['parser_types'].items():
                print(f"  {parser_type}: {count}")
        
        if stats['package_usage']:
            print(f"\nPackages used:")
            for package, count in stats['package_usage'].items():
                print(f"  {package}: {count} parsers")
        
        if stats['files_chunked'] > 0:
            print(f"\nProcessing stats:")
            print(f"  Files processed: {stats['files_chunked']}")
            print(f"  Chunks created: {stats['chunks_created']}")
            print(f"  Avg chunks/file: {stats['avg_chunks_per_file']:.1f}")
            print(f"  Avg time/file: {stats['avg_processing_time']*1000:.1f}ms")