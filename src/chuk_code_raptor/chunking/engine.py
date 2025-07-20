# src/chuk_code_raptor/chunking/engine.py
"""
Chunking Engine
===============

Main engine that coordinates chunking operations.
Clean, unified implementation supporting both tree-sitter and heuristic parsers.
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
    """Main engine for chunking operations using clean parser architecture"""
    
    def __init__(self, config: ChunkingConfig = None):
        """
        Initialize chunking engine.
        
        Args:
            config: ChunkingConfig to use (creates default if None)
        """
        self.config = config or ChunkingConfig()
        self.parsers: Dict[str, BaseParser] = {}
        
        # Statistics (preserving original structure)
        self.stats = {
            'files_processed': 0,
            'files_chunked': 0,
            'chunks_created': 0,
            'total_processing_time': 0.0,
            'chunker_usage': {},
            'errors': []
        }
        
        # Auto-discover and register parsers
        self._register_parsers()
        
        logger.info(f"Initialized ChunkingEngine with {len(self.parsers)} parser instances")
    
    def _register_parsers(self):
        """Auto-discover and register available parsers"""
        parser_configs = [
            # Tree-sitter parsers (preferred when available)
            ('python', 'chuk_code_raptor.chunking.parsers.python', 'PythonParser'),
            ('markdown', 'chuk_code_raptor.chunking.parsers.markdown', 'MarkdownParser'),
            ('javascript', 'chuk_code_raptor.chunking.parsers.javascript', 'JavaScriptParser'),
            ('json', 'chuk_code_raptor.chunking.parsers.json', 'JSONParser'),
            ('html', 'chuk_code_raptor.chunking.parsers.html', 'HTMLParser'),
            ('css', 'chuk_code_raptor.chunking.parsers.css', 'CSSParser'),
            # Skip rust for now due to tree-sitter version issues
            # ('rust', 'chuk_code_raptor.chunking.parsers.rust', 'RustParser'),
        ]
        
        successful_parsers = []
        failed_parsers = []
        
        for language, module_path, class_name in parser_configs:
            try:
                # Import the parser module
                module = __import__(module_path, fromlist=[class_name])
                parser_class = getattr(module, class_name)
                
                # Create parser instance
                parser = parser_class(self.config)
                
                # Verify the parser is functional
                if hasattr(parser, 'can_parse') and callable(parser.can_parse):
                    # Register for all supported languages
                    for lang in parser.supported_languages:
                        self.parsers[lang] = parser
                    
                    successful_parsers.append((class_name, list(parser.supported_languages)))
                    logger.debug(f"✅ Registered {class_name} for {list(parser.supported_languages)}")
                else:
                    failed_parsers.append((class_name, "Missing can_parse method"))
                    logger.warning(f"❌ {class_name} missing can_parse method")
                
            except ImportError as e:
                failed_parsers.append((class_name, f"Import error: {e}"))
                logger.debug(f"❌ {class_name} not available: {e}")
            except Exception as e:
                failed_parsers.append((class_name, f"Initialization error: {e}"))
                logger.warning(f"❌ Failed to register {class_name}: {e}")
        
        # Log summary
        if successful_parsers:
            total_languages = len(self.parsers)
            logger.info(f"✅ Successfully registered {len(successful_parsers)} parsers for {total_languages} languages:")
            for parser_name, languages in successful_parsers:
                logger.info(f"   {parser_name}: {', '.join(languages)}")
        
        if failed_parsers:
            logger.info(f"❌ Failed to register {len(failed_parsers)} parsers:")
            for parser_name, error in failed_parsers:
                logger.info(f"   {parser_name}: {error}")
        
        if not self.parsers:
            logger.error("⚠️  No parsers were successfully registered!")
    
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
            error = f"No parser available for language: {language}"
            self.stats['errors'].append(error)
            logger.warning(f"⚠️  {error}")
            raise UnsupportedLanguageError(error)
        
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
    
    def _get_parser(self, language: str) -> Optional[BaseParser]:
        """Get parser for language"""
        parser = self.parsers.get(language)
        if parser:
            logger.debug(f"🔍 Using {parser.name} for language '{language}'")
        else:
            logger.debug(f"❌ No parser found for language '{language}'. Available: {list(self.parsers.keys())}")
        return parser
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect language from file extension"""
        extension = file_path.suffix.lower()
        
        extension_map = {
            '.py': 'python',
            '.pyx': 'python',
            '.pyi': 'python',
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.rs': 'rust',
            '.json': 'json',
            '.html': 'html',
            '.htm': 'html',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.css': 'css',
            '.scss': 'css', 
            '.sass': 'css',
            '.less': 'css',
        }
        
        detected = extension_map.get(extension, 'text')
        logger.debug(f"🔍 Detected language '{detected}' for extension '{extension}'")
        return detected
    
    def _detect_content_type(self, file_path: str, language: str) -> ContentType:
        """Detect content type"""
        extension = Path(file_path).suffix.lower()
        
        if extension in {'.md', '.markdown'}:
            return ContentType.MARKDOWN
        elif extension in {'.html', '.htm'}:
            return ContentType.HTML
        elif extension == '.json':
            return ContentType.JSON
        elif extension in {'.yaml', '.yml'}:
            return ContentType.YAML
        elif language in {'python', 'javascript', 'typescript', 'rust'}:
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
        if parser_name not in self.stats['chunker_usage']:
            self.stats['chunker_usage'][parser_name] = {
                'files_processed': 0,
                'chunks_created': 0,
                'total_time': 0.0
            }
        
        self.stats['chunker_usage'][parser_name]['files_processed'] += 1
        self.stats['chunker_usage'][parser_name]['chunks_created'] += len(chunks)
        self.stats['chunker_usage'][parser_name]['total_time'] += processing_time
    
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
        return list(self.parsers.keys())
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions"""
        extensions = set()
        seen_parsers = set()
        for parser in self.parsers.values():
            parser_id = id(parser)
            if parser_id not in seen_parsers:
                extensions.update(parser.supported_extensions)
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