# src/chuk_code_raptor/chunking/parsers/feed.py
"""
Feed Parser Coordinator - FIXED
===============================

Orchestrates the dedicated RSS/Atom parsers and provides intelligent feed type detection
and routing to the appropriate specialized parser.

This coordinator replaces the monolithic RSSAtomParser with a clean architecture that:
- Detects feed type accurately 
- Routes to specialized parsers (RSS2Parser, RSS1RDFParser, AtomParser)
- Provides fallback handling
- Maintains backward compatibility
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any, Type
import logging

from ..base import BaseParser, ParseContext  # Changed: inherit from BaseParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from chuk_code_raptor.core.models import ChunkType

# Import the specialized parsers
from .rss import RSSParser
from .rdf import RDFParser  
from .atom import AtomParser

logger = logging.getLogger(__name__)


class FeedParserCoordinator(BaseParser):  # Changed: inherit from BaseParser instead of TreeSitterParser
    """
    Coordinates multiple specialized feed parsers.
    
    This coordinator:
    1. Analyzes feed content to determine format
    2. Routes to appropriate specialized parser
    3. Handles edge cases and fallbacks
    4. Provides unified interface for all feed types
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.supported_languages = {'rss', 'atom', 'feed', 'syndication', 'rss_atom'}
        self.supported_extensions = {'.rss', '.atom', '.xml', '.feed', '.rdf'}
        self.name = "FeedParserCoordinator"
        
        # Configuration
        self.enable_format_detection = getattr(config, 'feed_enable_format_detection', True)
        self.strict_format_validation = getattr(config, 'feed_strict_format_validation', False)
        self.fallback_to_generic = getattr(config, 'feed_fallback_to_generic', True)
        
        # Initialize specialized parsers
        self._parsers = {}
        try:
            self._parsers['rss2'] = RSSParser(config)
            logger.debug("RSS 2.0 parser initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize RSS2Parser: {e}")
        
        try:
            self._parsers['rss1'] = RDFParser(config)
            logger.debug("RSS 1.0 RDF parser initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize RSS1RDFParser: {e}")
        
        try:
            self._parsers['atom'] = AtomParser(config)
            logger.debug("Atom parser initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize AtomParser: {e}")
        
        if not self._parsers:
            logger.error("No specialized feed parsers could be initialized")
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        """Check if coordinator can handle the language/extension"""
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions or
                any(parser.can_parse(language, file_extension) 
                    for parser in self._parsers.values()))
    
    def parse(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """
        Parse feed content by detecting format and routing to appropriate parser.
        """
        if not content.strip():
            return []
        
        # Detect feed format
        feed_format = self._detect_feed_format(content)
        logger.info(f"Detected feed format: {feed_format} for {context.file_path}")
        
        # Route to appropriate parser
        if feed_format in self._parsers:
            try:
                parser = self._parsers[feed_format]
                chunks = parser.parse(content, context)
                
                if chunks:
                    logger.info(f"Successfully parsed {len(chunks)} chunks using {parser.name}")
                    return self._post_process_chunks(chunks, feed_format)
                else:
                    logger.warning(f"{parser.name} returned no chunks")
            
            except Exception as e:
                logger.error(f"Error in {feed_format} parser: {e}")
        
        # Fallback strategies
        if self.fallback_to_generic:
            return self._fallback_parse(content, context, feed_format)
        
        return []
    
    def _detect_feed_format(self, content: str) -> str:
        """
        Detect feed format from content analysis.
        
        Returns: 'rss2', 'rss1', 'atom', or 'unknown'
        """
        if not self.enable_format_detection:
            return 'unknown'
        
        try:
            # Quick text-based detection first (faster)
            text_format = self._detect_format_by_text(content)
            if text_format != 'unknown':
                logger.debug(f"Text-based detection: {text_format}")
            
            # XML-based detection for accuracy
            xml_format = self._detect_format_by_xml(content)
            if xml_format != 'unknown':
                logger.debug(f"XML-based detection: {xml_format}")
                return xml_format
            
            # Fall back to text detection
            return text_format
            
        except Exception as e:
            logger.warning(f"Feed format detection failed: {e}")
            return 'unknown'
    
    def _detect_format_by_text(self, content: str) -> str:
        """Quick text-based format detection"""
        content_lower = content.lower()
        
        # Atom 1.0 indicators (most specific first)
        atom_patterns = [
            r'<feed\s+xmlns="http://www\.w3\.org/2005/atom"',
            r'xmlns="http://www\.w3\.org/2005/atom"',
            r'2005/atom',
            r'<feed\s+.*atom'
        ]
        
        for pattern in atom_patterns:
            if re.search(pattern, content_lower):
                return 'atom'
        
        # RSS 1.0 RDF indicators
        rss1_patterns = [
            r'<rdf:rdf.*xmlns:rss="http://purl\.org/rss/1\.0/"',
            r'rss/1\.0/',
            r'<rdf:rdf.*rdf-syntax-ns',
            r'xmlns:rdf="http://www\.w3\.org/1999/02/22-rdf-syntax-ns#".*rss'
        ]
        
        for pattern in rss1_patterns:
            if re.search(pattern, content_lower):
                return 'rss1'
        
        # RSS 2.0 indicators
        rss2_patterns = [
            r'<rss\s+version="2\.[0-9]"',
            r'<rss\s+version="2\.0"',
            r'<rss.*version.*2\.'
        ]
        
        for pattern in rss2_patterns:
            if re.search(pattern, content_lower):
                return 'rss2'
        
        # Generic RSS (assume 2.0)
        if '<rss' in content_lower and '<channel' in content_lower:
            return 'rss2'
        
        return 'unknown'
    
    def _detect_format_by_xml(self, content: str) -> str:
        """XML structure-based format detection"""
        try:
            # Clean content for parsing
            cleaned_content = content.strip()
            if cleaned_content.startswith('\ufeff'):
                cleaned_content = cleaned_content[1:]  # Remove BOM
            
            root = ET.fromstring(cleaned_content)
            
            # Check root element and attributes
            root_tag = root.tag.lower()
            
            # Atom feed
            if (root.tag == '{http://www.w3.org/2005/Atom}feed' or
                root_tag == 'feed' and 
                root.get('xmlns') == 'http://www.w3.org/2005/Atom'):
                return 'atom'
            
            # RSS 1.0 RDF
            if (root.tag == '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF' or
                root_tag == 'rdf:rdf' or 
                (root_tag == 'rdf' and 'rss/1.0' in str(root.attrib))):
                
                # Verify it contains RSS 1.0 elements
                rss_ns = 'http://purl.org/rss/1.0/'
                if root.find(f'{{{rss_ns}}}channel') is not None:
                    return 'rss1'
                
                # Check for RSS elements without namespace
                for child in root:
                    child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if child_tag.lower() == 'channel':
                        return 'rss1'
            
            # RSS 2.0
            if root_tag == 'rss':
                version = root.get('version', '2.0')
                if version.startswith('2.'):
                    return 'rss2'
                elif version.startswith('0.9') or version.startswith('1.'):
                    return 'rss2'  # Treat older RSS as RSS 2.0 compatible
                else:
                    return 'rss2'  # Default assumption
            
            return 'unknown'
            
        except ET.ParseError as e:
            logger.debug(f"XML parsing failed during format detection: {e}")
            return 'unknown'
        except Exception as e:
            logger.warning(f"XML format detection error: {e}")
            return 'unknown'
    
    def _fallback_parse(self, content: str, context: ParseContext, detected_format: str) -> List[SemanticChunk]:
        """
        Fallback parsing when specialized parsers fail.
        """
        logger.info(f"Attempting fallback parsing for {detected_format} format")
        
        # Try other parsers if the detected format failed
        fallback_order = ['rss2', 'atom', 'rss1']
        
        # Remove the failed format and try others
        if detected_format in fallback_order:
            fallback_order.remove(detected_format)
        
        for format_name in fallback_order:
            if format_name in self._parsers:
                try:
                    parser = self._parsers[format_name]
                    chunks = parser.parse(content, context)
                    
                    if chunks:
                        logger.info(f"Fallback successful with {parser.name}")
                        return self._post_process_chunks(chunks, format_name)
                        
                except Exception as e:
                    logger.debug(f"Fallback parser {format_name} failed: {e}")
        
        # Final fallback: create basic chunk
        return self._create_generic_fallback_chunk(content, context, detected_format)
    
    def _create_generic_fallback_chunk(self, content: str, context: ParseContext, 
                                     detected_format: str) -> List[SemanticChunk]:
        """Create a basic fallback chunk when all parsing fails"""
        
        # Extract basic information
        title = self._extract_basic_title(content)
        description = self._extract_basic_description(content)
        item_count = self._estimate_item_count(content)
        
        # Create summary
        summary_parts = [f"Feed Content ({detected_format.upper() if detected_format != 'unknown' else 'Unknown Format'})"]
        
        if title:
            summary_parts.append(f"Title: {title}")
        
        if description:
            summary_parts.append(f"Description: {description}")
        
        summary_parts.append(f"Estimated entries: {item_count}")
        summary_parts.append(f"Content size: {len(content):,} characters")
        
        content_summary = '\n'.join(summary_parts)
        
        chunk_id = create_chunk_id(
            context.file_path,
            1,
            ChunkType.TEXT_BLOCK,
            "feed_fallback"
        )
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=content_summary,
            start_line=1,
            end_line=len(summary_parts),
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=0.4,
            metadata={
                'parser': self.name,
                'feed_type': detected_format,
                'semantic_type': 'Feed Fallback',
                'extraction_method': 'coordinator_fallback',
                'detected_format': detected_format,
                'estimated_items': item_count,
                'parsing_status': 'fallback'
            }
        )
        
        # Add basic tags
        if hasattr(chunk, 'add_tag'):
            chunk.add_tag('feed_fallback', source='feed_coordinator')
            chunk.add_tag('syndication', source='feed_coordinator')
            if detected_format != 'unknown':
                chunk.add_tag(f'{detected_format}_fallback', source='feed_coordinator')
        
        return [chunk]
    
    def _post_process_chunks(self, chunks: List[SemanticChunk], feed_format: str) -> List[SemanticChunk]:
        """Post-process chunks from specialized parsers"""
        
        # Add coordinator metadata
        for chunk in chunks:
            if chunk.metadata:
                chunk.metadata['coordinator'] = self.name
                chunk.metadata['routing_format'] = feed_format
            
            # Add coordinator tag
            if hasattr(chunk, 'add_tag'):
                chunk.add_tag('feed_coordinated', source='feed_coordinator')
        
        return chunks
    
    def _extract_basic_title(self, content: str) -> str:
        """Extract basic title from any feed format"""
        # Try various title patterns
        patterns = [
            r'<title[^>]*>([^<]+)</title>',
            r'<atom:title[^>]*>([^<]+)</atom:title>',
            r'<rss:title[^>]*>([^<]+)</rss:title>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                title = match.group(1).strip()
                # Clean up CDATA
                title = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', title)
                if title:
                    return title[:100]  # Limit length
        
        return "Unknown Feed"
    
    def _extract_basic_description(self, content: str) -> str:
        """Extract basic description from any feed format"""
        patterns = [
            r'<description[^>]*>([^<]+)</description>',
            r'<subtitle[^>]*>([^<]+)</subtitle>',
            r'<summary[^>]*>([^<]+)</summary>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                desc = match.group(1).strip()
                # Clean up CDATA and HTML
                desc = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', desc)
                desc = re.sub(r'<[^>]+>', '', desc)
                if desc:
                    return desc[:200] + ("..." if len(desc) > 200 else "")
        
        return ""
    
    def _estimate_item_count(self, content: str) -> int:
        """Estimate number of items/entries in feed"""
        # Count different item patterns
        item_patterns = [
            r'<item\b',
            r'<entry\b',
            r'<rss:item\b',
            r'<atom:entry\b'
        ]
        
        max_count = 0
        for pattern in item_patterns:
            count = len(re.findall(pattern, content, re.IGNORECASE))
            max_count = max(max_count, count)
        
        return max_count
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported feed formats"""
        return list(self._parsers.keys())
    
    def get_parser_for_format(self, format_name: str) -> Optional[BaseParser]:
        """Get specific parser for a format"""
        return self._parsers.get(format_name)
    
    def is_format_supported(self, format_name: str) -> bool:
        """Check if a format is supported"""
        return format_name in self._parsers
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about available parsers"""
        info = {
            'coordinator': self.name,
            'available_parsers': {},
            'total_formats': len(self._parsers),
            'format_detection': self.enable_format_detection,
            'strict_validation': self.strict_format_validation
        }
        
        for format_name, parser in self._parsers.items():
            info['available_parsers'][format_name] = {
                'name': parser.name,
                'class': parser.__class__.__name__,
                'supported_languages': getattr(parser, 'supported_languages', set()),
                'supported_extensions': getattr(parser, 'supported_extensions', set())
            }
        
        return info