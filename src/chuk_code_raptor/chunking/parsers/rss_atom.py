# src/chuk_code_raptor/chunking/parsers/rss_atom.py
"""
RSS/Atom Parser - Integrated with YAML Registry System
======================================================

Comprehensive parser for RSS 2.0, RSS 1.0, and Atom 1.0 feeds.
Follows your existing architecture patterns and integrates with the registry system.
Fixed RSS 1.0 parsing issues.
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
import logging

from ..tree_sitter_base import TreeSitterParser, get_tree_sitter_language_robust
from ..heuristic_base import HeuristicParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from ..base import ParseContext
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)


class RSSAtomParser(TreeSitterParser):
    """
    RSS/Atom feed parser that follows your architecture patterns.
    
    Uses tree-sitter-xml as primary parser with comprehensive heuristic fallback
    for feed-specific semantic understanding.
    """
    
    def __init__(self, config):
        # Initialize parent with config first
        super().__init__(config)
        
        # Set our specific language support
        self.supported_languages = {'rss', 'atom', 'feed', 'syndication'}
        self.supported_extensions = {'.rss', '.atom', '.xml', '.feed'}
        self.name = "RSSAtomParser"
        
        # Feed-specific configuration from YAML
        self.extract_full_content = getattr(config, 'rss_atom_extract_full_content', True)
        self.prefer_content_encoded = getattr(config, 'rss_atom_prefer_content_encoded', True)
        self.clean_html_content = getattr(config, 'rss_atom_clean_html_content', True)
        self.extract_media_metadata = getattr(config, 'rss_atom_extract_media_metadata', True)
        self.max_entries_per_feed = getattr(config, 'rss_atom_max_entries_per_feed', 100)
        self.min_entry_content_length = getattr(config, 'rss_atom_min_entry_content_length', 50)
        self.chunk_by_entry = getattr(config, 'rss_atom_chunk_by_entry', True)
        self.include_feed_metadata = getattr(config, 'rss_atom_include_feed_metadata', True)
        
        # XML namespaces commonly used in feeds
        self.namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rss': 'http://purl.org/rss/1.0/',
            'content': 'http://purl.org/rss/1.0/modules/content/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'sy': 'http://purl.org/rss/1.0/modules/syndication/',
            'media': 'http://search.yahoo.com/mrss/',
            'itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd'
        }
    
    def _get_tree_sitter_language(self):
        """Get tree-sitter XML language with robust package handling"""
        language, package_used = get_tree_sitter_language_robust(
            'xml', 
            ['tree_sitter_xml', 'tree_sitter_languages', 'tree_sitter_language_pack']
        )
        
        if language is None:
            raise ImportError("No tree-sitter XML package found. Install with: pip install tree-sitter-xml")
        
        self._package_used = package_used
        return language
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """XML AST node types to chunk types mapping for feeds"""
        return {
            'element': ChunkType.TEXT_BLOCK,
            'start_tag': ChunkType.TEXT_BLOCK,
            'comment': ChunkType.COMMENT,
            'processing_instruction': ChunkType.TEXT_BLOCK,
            'cdata_section': ChunkType.TEXT_BLOCK,
            'text': ChunkType.TEXT_BLOCK,
        }
    
    def parse(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """
        Enhanced parse method that tries tree-sitter first, then falls back to heuristic parsing.
        Follows your existing pattern for robust parsing.
        """
        if not content.strip():
            return []
        
        # First attempt: tree-sitter parsing
        tree_sitter_chunks = []
        try:
            if self.parser is not None and self.language is not None:
                tree_sitter_chunks = self._parse_with_tree_sitter(content, context)
                logger.debug(f"Tree-sitter RSS/Atom parsing produced {len(tree_sitter_chunks)} chunks")
        except Exception as e:
            logger.warning(f"Tree-sitter RSS/Atom parsing failed: {e}")
        
        # Check if tree-sitter parsing was sufficient
        if self._is_feed_parsing_successful(tree_sitter_chunks, content):
            logger.info(f"Using tree-sitter results for RSS/Atom parsing: {len(tree_sitter_chunks)} chunks")
            return self._post_process(tree_sitter_chunks)
        
        # Fallback: heuristic feed parsing
        logger.info("Tree-sitter insufficient for RSS/Atom, using heuristic feed analysis")
        try:
            heuristic_chunks = self._parse_feed_heuristically(content, context)
            
            if heuristic_chunks and len(heuristic_chunks) > len(tree_sitter_chunks):
                logger.info(f"Heuristic RSS/Atom analysis produced {len(heuristic_chunks)} chunks")
                return self._post_process(heuristic_chunks)
        except Exception as e:
            logger.error(f"Heuristic RSS/Atom parsing failed: {e}")
            
            # Create a basic fallback chunk for the feed
            if content.strip():
                chunk_id = create_chunk_id(context.file_path, 1, ChunkType.TEXT_BLOCK, "feed_fallback")
                
                # Detect basic feed info for fallback
                feed_type = self._detect_feed_type_from_content(content)
                title = self._extract_basic_title(content)
                
                fallback_content = f"Feed: {title}\nType: {feed_type}\nSize: {len(content)} characters"
                
                fallback_chunk = SemanticChunk(
                    id=chunk_id,
                    file_path=context.file_path,
                    content=fallback_content,
                    start_line=1,
                    end_line=3,
                    content_type=context.content_type,
                    chunk_type=ChunkType.TEXT_BLOCK,
                    language=context.language,
                    importance_score=0.5,
                    metadata={
                        'parser': self.name,
                        'parser_type': 'rss_atom_fallback',
                        'extraction_method': 'fallback',
                        'semantic_type': 'Feed Fallback',
                        'error': str(e)
                    }
                )
                
                # Add basic tags using add_tag method
                if hasattr(fallback_chunk, 'add_tag'):
                    fallback_chunk.add_tag('feed_fallback', source='error_recovery')
                    fallback_chunk.add_tag('parsing_error', source='error_recovery')
                
                return [fallback_chunk]
        
        # Final fallback: return tree-sitter result even if poor
        logger.warning("Heuristic parsing didn't improve results, using tree-sitter fallback")
        return self._post_process(tree_sitter_chunks) if tree_sitter_chunks else []
    
    def _parse_with_tree_sitter(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse using tree-sitter with feed-aware processing"""
        tree = self.parser.parse(bytes(content, 'utf8'))
        root_node = tree.root_node
        
        chunks = []
        
        # Detect feed type from tree structure
        feed_type = self._detect_feed_type_from_tree(root_node, content)
        
        # Extract using tree-sitter but with feed-specific logic
        if feed_type == 'atom':
            chunks = self._extract_atom_feed_tree_sitter(root_node, content, context)
        elif feed_type in ['rss2', 'rss1']:
            chunks = self._extract_rss_feed_tree_sitter(root_node, content, context, feed_type)
        else:
            # Generic XML parsing
            self._traverse_and_extract(root_node, content, context, chunks)
        
        return chunks
    
    def _is_feed_parsing_successful(self, chunks: List[SemanticChunk], content: str) -> bool:
        """Determine if feed parsing was successful"""
        if not chunks:
            return False
        
        # For feeds, we expect at least a metadata chunk and some entries
        has_feed_metadata = any('feed_metadata' in chunk.metadata.get('feed_type', '') for chunk in chunks)
        has_entries = any('entry' in chunk.metadata.get('semantic_type', '').lower() for chunk in chunks)
        
        # If we have both metadata and entries, consider it successful
        if has_feed_metadata and has_entries:
            return True
        
        # If we only have one chunk that's the entire document, likely failed
        if len(chunks) == 1 and len(chunks[0].content) / len(content) > 0.90:
            return False
        
        return len(chunks) >= 2  # At least some structure detected
    
    def _parse_feed_heuristically(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """
        Heuristic feed parsing when tree-sitter fails.
        This is the comprehensive feed-specific logic.
        """
        try:
            # Clean content before parsing
            cleaned_content = content.strip()
            if not cleaned_content:
                return []
            
            # Parse XML using ElementTree with better error handling
            try:
                root = ET.fromstring(cleaned_content)
            except ET.ParseError as e:
                logger.warning(f"XML parsing error in {context.file_path}: {e}")
                # Try to fix common XML issues
                try:
                    # Remove BOM and clean up
                    if cleaned_content.startswith('\ufeff'):
                        cleaned_content = cleaned_content[1:]
                    
                    # Try parsing again
                    root = ET.fromstring(cleaned_content)
                except ET.ParseError:
                    logger.error(f"Could not parse XML even after cleanup in {context.file_path}")
                    return []
            
            # Detect feed type
            feed_type = self._detect_feed_type_from_element(root)
            
            if feed_type == 'atom':
                return self._parse_atom_feed_heuristic(root, cleaned_content, context)
            elif feed_type in ['rss2', 'rss1']:
                return self._parse_rss_feed_heuristic(root, cleaned_content, context, feed_type)
            else:
                return self._parse_generic_xml_feed(root, cleaned_content, context)
                
        except Exception as e:
            logger.error(f"RSS/Atom parsing error in {context.file_path}: {e}")
            return []
    
    def _detect_feed_type_from_element(self, root) -> str:
        """Detect feed type from ElementTree root"""
        root_tag = root.tag.lower()
        
        # Check for Atom feed
        if root.tag == '{http://www.w3.org/2005/Atom}feed':
            return 'atom'
        elif 'feed' in root_tag:
            return 'atom'
        
        # Check for RSS 1.0 (RDF)
        elif root.tag == '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF':
            return 'rss1'
        elif 'rdf' in root_tag:
            return 'rss1'
        
        # Check for RSS 2.0
        elif root.tag == 'rss':
            version = root.get('version', '2.0')
            return f'rss{version.split(".")[0]}'
        elif 'rss' in root_tag:
            return 'rss2'
        
        else:
            return 'unknown'
    
    def _detect_feed_type_from_tree(self, root_node, content: str) -> str:
        """Detect feed type from tree-sitter AST"""
        # Look for root element name in the tree
        for child in root_node.children:
            if child.type == 'element':
                # Extract element name
                element_name = self._extract_element_name_from_node(child, content)
                if element_name:
                    if element_name.lower() == 'feed':
                        return 'atom'
                    elif element_name.lower() == 'rss':
                        return 'rss2'
                    elif element_name.lower() == 'rdf':
                        return 'rss1'
        
        return 'unknown'
    
    def _extract_element_name_from_node(self, node, content: str) -> Optional[str]:
        """Extract element name from tree-sitter node"""
        for child in node.children:
            if child.type == 'start_tag':
                for grandchild in child.children:
                    if grandchild.type == 'tag_name':
                        return content[grandchild.start_byte:grandchild.end_byte]
        return None
    
    def _parse_atom_feed_heuristic(self, root, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse Atom 1.0 feed using heuristic analysis"""
        chunks = []
        
        # Parse feed metadata
        if self.include_feed_metadata:
            feed_metadata = self._extract_atom_feed_metadata(root)
            feed_chunk = self._create_feed_metadata_chunk(
                feed_metadata, content, context, 'atom'
            )
            chunks.append(feed_chunk)
        
        # Parse individual entries
        entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
        entries = entries[:self.max_entries_per_feed]  # Limit entries
        
        for i, entry in enumerate(entries):
            entry_chunk = self._parse_atom_entry_heuristic(entry, context, i + 1)
            if entry_chunk:
                chunks.append(entry_chunk)
        
        return chunks
    
    def _parse_rss_feed_heuristic(self, root, content: str, context: ParseContext, 
                                feed_type: str) -> List[SemanticChunk]:
        """Parse RSS feed using heuristic analysis"""
        chunks = []
        
        if feed_type == 'rss1':
            # RSS 1.0 (RDF) has a different structure
            return self._parse_rss1_rdf_feed(root, content, context)
        else:
            # RSS 2.0 structure
            # Find channel element
            channel = root.find('.//channel')
            
            if channel is None:
                logger.warning(f"No channel found in RSS feed {context.file_path}")
                return []
            
            # Parse feed metadata
            if self.include_feed_metadata:
                feed_metadata = self._extract_rss_feed_metadata(channel, feed_type)
                feed_chunk = self._create_feed_metadata_chunk(
                    feed_metadata, content, context, feed_type
                )
                chunks.append(feed_chunk)
            
            # Parse individual items
            items = channel.findall('.//item')
            items = items[:self.max_entries_per_feed]  # Limit items
            
            for i, item in enumerate(items):
                item_chunk = self._parse_rss_item_heuristic(item, context, i + 1, feed_type)
                if item_chunk:
                    chunks.append(item_chunk)
        
        return chunks
    
    def _create_feed_metadata_chunk(self, metadata: Dict[str, Any], content: str, 
                                  context: ParseContext, feed_type: str) -> SemanticChunk:
        """Create chunk for feed-level metadata"""
        # Create readable summary of feed metadata
        summary_parts = []
        if metadata.get('title'):
            summary_parts.append(f"Feed: {metadata['title']}")
        if metadata.get('description') or metadata.get('subtitle'):
            desc = metadata.get('description') or metadata.get('subtitle')
            summary_parts.append(f"Description: {desc}")
        if metadata.get('language'):
            summary_parts.append(f"Language: {metadata['language']}")
        if metadata.get('lastBuildDate') or metadata.get('updated'):
            date = metadata.get('lastBuildDate') or metadata.get('updated')
            summary_parts.append(f"Updated: {date}")
        
        summary = '\n'.join(summary_parts) if summary_parts else f"Feed metadata for {feed_type}"
        
        chunk_id = create_chunk_id(
            context.file_path, 
            1, 
            ChunkType.METADATA, 
            f"feed_metadata_{feed_type}"
        )
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=summary,
            start_line=1,
            end_line=len(summary.split('\n')),
            content_type=context.content_type,
            chunk_type=ChunkType.METADATA,
            language=context.language,
            importance_score=0.9,  # High importance for feed metadata
            metadata={
                'parser': self.name,
                'parser_type': 'rss_atom',
                'feed_type': feed_type,
                'feed_metadata': metadata,
                'extraction_method': 'heuristic',
                'semantic_type': 'RSS/Atom Feed Metadata'
            }
        )
        
        # Add semantic tags using add_tag method
        if hasattr(chunk, 'add_tag'):
            chunk.add_tag('feed_metadata', source='rss_atom_parser')
            chunk.add_tag(feed_type, source='rss_atom_parser')
            chunk.add_tag('syndication', source='rss_atom_parser')
        
        # Add category tags from metadata
        if metadata.get('categories'):
            for cat in metadata['categories']:
                if isinstance(cat, dict):
                    term = cat.get('term') or cat.get('text')
                else:
                    term = str(cat)
                if term and hasattr(chunk, 'add_tag'):
                    chunk.add_tag(f"category:{term}", source='rss_atom_parser')
        
        return chunk
    
    def _parse_atom_entry_heuristic(self, entry, context: ParseContext, entry_num: int) -> Optional[SemanticChunk]:
        """Parse individual Atom entry"""
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        # Extract basic entry data
        entry_data = self._extract_atom_entry_data(entry, ns)
        
        # Create readable content for the entry
        content_parts = self._build_entry_content_parts(entry_data)
        content = '\n\n'.join(content_parts)
        
        # Skip if content is too short
        if len(content) < self.min_entry_content_length:
            return None
        
        chunk_id = create_chunk_id(
            context.file_path,
            entry_num * 10,  # Approximate line numbers
            ChunkType.TEXT_BLOCK,
            f"atom_entry_{entry_num}"
        )
        
        # Calculate importance based on content richness
        importance = self._calculate_entry_importance(entry_data, content)
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=content,
            start_line=entry_num * 10,
            end_line=entry_num * 10 + len(content.split('\n')),
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=importance,
            metadata={
                'parser': self.name,
                'parser_type': 'rss_atom',
                'feed_type': 'atom',
                'entry_number': entry_num,
                'entry_data': entry_data,
                'extraction_method': 'heuristic',
                'semantic_type': 'Atom Feed Entry'
            }
        )
        
        # Add semantic tags
        self._add_entry_semantic_tags(chunk, entry_data)
        
        return chunk
    
    def _parse_rss_item_heuristic(self, item, context: ParseContext, item_num: int, 
                                feed_type: str) -> Optional[SemanticChunk]:
        """Parse individual RSS item"""
        # Extract basic item data
        item_data = self._extract_rss_item_data(item)
        
        # Create readable content for the item
        content_parts = self._build_entry_content_parts(item_data)
        content = '\n\n'.join(content_parts)
        
        # Skip if content is too short
        if len(content) < self.min_entry_content_length:
            return None
        
        chunk_id = create_chunk_id(
            context.file_path,
            item_num * 10,
            ChunkType.TEXT_BLOCK,
            f"rss_item_{item_num}"
        )
        
        # Calculate importance
        importance = self._calculate_entry_importance(item_data, content)
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=content,
            start_line=item_num * 10,
            end_line=item_num * 10 + len(content.split('\n')),
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=importance,
            metadata={
                'parser': self.name,
                'parser_type': 'rss_atom',
                'feed_type': feed_type,
                'entry_number': item_num,
                'entry_data': item_data,
                'extraction_method': 'heuristic',
                'semantic_type': 'RSS Feed Entry'
            }
        )
        
        # Add semantic tags
        self._add_entry_semantic_tags(chunk, item_data)
        
        return chunk
    
    def _extract_atom_feed_metadata(self, root) -> Dict[str, Any]:
        """Extract metadata from Atom feed"""
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        metadata = {
            'format': 'atom',
            'title': self._get_text(root, 'atom:title', ns),
            'subtitle': self._get_text(root, 'atom:subtitle', ns),
            'id': self._get_text(root, 'atom:id', ns),
            'updated': self._get_text(root, 'atom:updated', ns),
            'rights': self._get_text(root, 'atom:rights', ns),
            'generator': self._get_text(root, 'atom:generator', ns),
            'language': root.get('{http://www.w3.org/XML/1998/namespace}lang'),
        }
        
        # Extract links, authors, categories
        metadata.update(self._extract_atom_links_authors_categories(root, ns))
        
        return metadata
    
    def _extract_rss_feed_metadata(self, channel, feed_type: str) -> Dict[str, Any]:
        """Extract metadata from RSS feed channel"""
        metadata = {
            'format': feed_type,
            'title': self._get_element_text(channel, 'title'),
            'description': self._get_element_text(channel, 'description'),
            'link': self._get_element_text(channel, 'link'),
            'language': self._get_element_text(channel, 'language'),
            'copyright': self._get_element_text(channel, 'copyright'),
            'managingEditor': self._get_element_text(channel, 'managingEditor'),
            'webMaster': self._get_element_text(channel, 'webMaster'),
            'pubDate': self._get_element_text(channel, 'pubDate'),
            'lastBuildDate': self._get_element_text(channel, 'lastBuildDate'),
            'generator': self._get_element_text(channel, 'generator'),
            'ttl': self._get_element_text(channel, 'ttl'),
        }
        
        # Extract categories and image
        metadata.update(self._extract_rss_categories_image(channel))
        
        return metadata
    
    def _extract_atom_entry_data(self, entry, ns) -> Dict[str, Any]:
        """Extract data from Atom entry"""
        entry_data = {
            'title': self._get_text(entry, 'atom:title', ns),
            'id': self._get_text(entry, 'atom:id', ns),
            'updated': self._get_text(entry, 'atom:updated', ns),
            'published': self._get_text(entry, 'atom:published', ns),
            'summary': self._get_text(entry, 'atom:summary', ns),
            'rights': self._get_text(entry, 'atom:rights', ns),
        }
        
        # Extract content
        content_elem = entry.find('atom:content', ns)
        if content_elem is not None:
            content_type = content_elem.get('type', 'text')
            if content_type in ['html', 'xhtml']:
                entry_data['content'] = self._extract_html_content(content_elem)
            else:
                entry_data['content'] = content_elem.text or ''
            entry_data['content_type'] = content_type
        
        # Extract links, authors, categories
        entry_data.update(self._extract_atom_links_authors_categories(entry, ns))
        
        return entry_data
    
    def _extract_rss_item_data(self, item) -> Dict[str, Any]:
        """Extract data from RSS item"""
        item_data = {
            'title': self._get_element_text(item, 'title'),
            'description': self._get_element_text(item, 'description'),
            'link': self._get_element_text(item, 'link'),
            'guid': self._get_element_text(item, 'guid'),
            'pubDate': self._get_element_text(item, 'pubDate'),
            'author': self._get_element_text(item, 'author'),
            'comments': self._get_element_text(item, 'comments'),
        }
        
        # Extract content:encoded if available
        content_elem = item.find('.//{http://purl.org/rss/1.0/modules/content/}encoded')
        if content_elem is not None and self.prefer_content_encoded:
            item_data['content'] = content_elem.text or ''
            item_data['content_type'] = 'html'
        
        # Extract Dublin Core metadata, categories, enclosures
        item_data.update(self._extract_rss_dublin_core(item))
        item_data.update(self._extract_rss_categories_enclosures(item))
        
        return item_data
    
    def _build_entry_content_parts(self, entry_data: Dict[str, Any]) -> List[str]:
        """Build readable content parts from entry data"""
        content_parts = []
        
        if entry_data.get('title'):
            content_parts.append(f"Title: {entry_data['title']}")
        
        # Author information
        if entry_data.get('authors'):
            authors = [author.get('name', str(author)) for author in entry_data['authors']]
            content_parts.append(f"Author(s): {', '.join(filter(None, authors))}")
        elif entry_data.get('author'):
            content_parts.append(f"Author: {entry_data['author']}")
        elif entry_data.get('dc_creator'):
            content_parts.append(f"Author: {entry_data['dc_creator']}")
        
        # Publication date
        if entry_data.get('published') or entry_data.get('pubDate'):
            date = entry_data.get('published') or entry_data.get('pubDate')
            content_parts.append(f"Published: {date}")
        
        # Summary or description
        if entry_data.get('summary'):
            content_parts.append(f"Summary: {entry_data['summary']}")
        elif entry_data.get('description'):
            content_parts.append(f"Description: {entry_data['description']}")
        
        # Full content if available
        if entry_data.get('content') and self.extract_full_content:
            if entry_data.get('content_type') in ['html', 'xhtml'] and self.clean_html_content:
                clean_content = self._clean_html_content(entry_data['content'])
                content_parts.append(f"Content: {clean_content}")
            else:
                content_parts.append(f"Content: {entry_data['content']}")
        
        # Link
        if entry_data.get('link'):
            content_parts.append(f"Link: {entry_data['link']}")
        elif entry_data.get('links'):
            for link in entry_data['links']:
                if link.get('rel') == 'alternate' and link.get('href'):
                    content_parts.append(f"Link: {link['href']}")
                    break
        
        return content_parts
    
    def _calculate_entry_importance(self, entry_data: Dict[str, Any], content: str) -> float:
        """Calculate importance score for feed entry"""
        importance = 0.7  # Base score for feed entries
        
        # Boost for full content
        if entry_data.get('content') and len(entry_data['content']) > 200:
            importance += 0.1
        
        # Boost for categories/tags
        if entry_data.get('categories'):
            importance += 0.1
        
        # Boost for media content
        if entry_data.get('enclosures'):
            importance += 0.1
        
        # Boost for longer content
        if len(content) > 500:
            importance += 0.05
        
        return min(importance, 1.0)
    
    def _add_entry_semantic_tags(self, chunk: SemanticChunk, entry_data: Dict[str, Any]):
        """Add semantic tags to entry chunks"""
        if hasattr(chunk, 'add_tag'):
            chunk.add_tag('rss_atom_entry', source='rss_atom_parser')
            chunk.add_tag('feed_entry', source='rss_atom_parser')
            chunk.add_tag('article', source='rss_atom_parser')
            
            # Category/topic tags
            if entry_data.get('categories'):
                for cat in entry_data['categories']:
                    if isinstance(cat, dict):
                        term = cat.get('term') or cat.get('text')
                    else:
                        term = str(cat)
                    if term:
                        chunk.add_tag(f"topic:{term}", source='rss_atom_parser')
            
            # Media tags
            if entry_data.get('enclosures'):
                chunk.add_tag('multimedia', source='rss_atom_parser')
                for enc in entry_data['enclosures']:
                    if enc.get('type', '').startswith('audio'):
                        chunk.add_tag('podcast', source='rss_atom_parser')
                    elif enc.get('type', '').startswith('video'):
                        chunk.add_tag('video', source='rss_atom_parser')
    
    # Utility methods
    
    def _get_text(self, element, xpath: str, namespaces: dict) -> Optional[str]:
        """Get text content from XML element using XPath with namespaces"""
        try:
            elem = element.find(xpath, namespaces)
            return elem.text if elem is not None else None
        except Exception:
            return None
    
    def _get_element_text(self, parent, tag: str) -> Optional[str]:
        """Get text content from XML element"""
        elem = parent.find(tag)
        return elem.text if elem is not None else None
    
    def _extract_html_content(self, element) -> str:
        """Extract content from HTML-containing XML element"""
        if element.text:
            return element.text
        try:
            return ''.join(ET.tostring(child, encoding='unicode') for child in element)
        except Exception:
            return element.text or ''
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content for display in chunk"""
        # Simple HTML tag removal
        clean_text = re.sub(r'<[^>]+>', '', html_content)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        # Limit length for chunk display
        if len(clean_text) > 500:
            clean_text = clean_text[:497] + "..."
        return clean_text.strip()
    
    def _extract_atom_links_authors_categories(self, element, ns) -> Dict[str, Any]:
        """Extract links, authors, and categories from Atom element"""
        data = {}
        
        # Links
        links = []
        for link in element.findall('atom:link', ns):
            link_data = {
                'href': link.get('href'),
                'rel': link.get('rel', 'alternate'),
                'type': link.get('type'),
                'title': link.get('title')
            }
            links.append(link_data)
        data['links'] = links
        
        # Authors
        authors = []
        for author in element.findall('atom:author', ns):
            author_data = {
                'name': self._get_text(author, 'atom:name', ns),
                'email': self._get_text(author, 'atom:email', ns),
                'uri': self._get_text(author, 'atom:uri', ns)
            }
            authors.append(author_data)
        data['authors'] = authors
        
        # Categories
        categories = []
        for category in element.findall('atom:category', ns):
            categories.append({
                'term': category.get('term'),
                'scheme': category.get('scheme'),
                'label': category.get('label')
            })
        data['categories'] = categories
        
        return data
    
    def _extract_rss_categories_image(self, channel) -> Dict[str, Any]:
        """Extract categories and image from RSS channel"""
        data = {}
        
        # Categories
        categories = []
        for category in channel.findall('.//category'):
            categories.append({
                'text': category.text,
                'domain': category.get('domain')
            })
        data['categories'] = categories
        
        # Image
        image = channel.find('.//image')
        if image is not None:
            data['image'] = {
                'url': self._get_element_text(image, 'url'),
                'title': self._get_element_text(image, 'title'),
                'link': self._get_element_text(image, 'link'),
                'width': self._get_element_text(image, 'width'),
                'height': self._get_element_text(image, 'height')
            }
        
        return data
    
    def _extract_rss_dublin_core(self, item) -> Dict[str, Any]:
        """Extract Dublin Core metadata from RSS item"""
        return {
            'dc_creator': self._get_element_text(item, './/{http://purl.org/dc/elements/1.1/}creator'),
            'dc_date': self._get_element_text(item, './/{http://purl.org/dc/elements/1.1/}date'),
            'dc_subject': self._get_element_text(item, './/{http://purl.org/dc/elements/1.1/}subject'),
        }
    
    def _extract_rss_categories_enclosures(self, item) -> Dict[str, Any]:
        """Extract categories and enclosures from RSS item"""
        data = {}
        
        # Categories
        categories = []
        for category in item.findall('.//category'):
            categories.append({
                'text': category.text,
                'domain': category.get('domain')
            })
        data['categories'] = categories
        
        # Enclosures
        enclosures = []
        for enclosure in item.findall('.//enclosure'):
            enclosures.append({
                'url': enclosure.get('url'),
                'type': enclosure.get('type'),
                'length': enclosure.get('length')
            })
        data['enclosures'] = enclosures
        
        return data
    
    def _parse_generic_xml_feed(self, root, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Fallback parser for unknown XML feed formats"""
        chunks = []
        
        # Try to extract basic information
        title_elem = root.find('.//title') or root.find('.//name')
        title = title_elem.text if title_elem is not None else "Unknown Feed"
        
        chunk_id = create_chunk_id(context.file_path, 1, ChunkType.TEXT_BLOCK, "generic_feed")
        
        summary = f"Generic XML Feed: {title}\nFormat: Unknown\nElements: {len(list(root.iter()))}"
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=summary,
            start_line=1,
            end_line=3,
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=0.5,
            metadata={
                'parser': self.name,
                'parser_type': 'rss_atom',
                'feed_type': 'generic_xml',
                'extraction_method': 'fallback',
                'semantic_type': 'Generic Feed'
            }
        )
        
        if hasattr(chunk, 'add_tag'):
            chunk.add_tag('xml_feed', source='rss_atom_parser')
            chunk.add_tag('unknown_format', source='rss_atom_parser')
        
        chunks.append(chunk)
        return chunks
    
    # Helper methods for fallback
    def _detect_feed_type_from_content(self, content: str) -> str:
        """Detect feed type from raw content"""
        content_lower = content.lower()
        if '<feed xmlns="http://www.w3.org/2005/Atom"' in content:
            return 'atom'
        elif '<rss version="2.0"' in content:
            return 'rss2'
        elif '<rdf:rdf' in content_lower:
            return 'rss1'
        elif '<rss' in content_lower:
            return 'rss'
        elif '<feed' in content_lower:
            return 'atom'
        else:
            return 'unknown_feed'

    def _extract_basic_title(self, content: str) -> str:
        """Extract basic title from feed content"""
        # Simple regex to find title
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        
        # Try to find any text that might be a title
        name_match = re.search(r'<name[^>]*>([^<]+)</name>', content, re.IGNORECASE)
        if name_match:
            return name_match.group(1).strip()
        
        return "Unknown Feed"
    
    def _parse_rss1_rdf_feed(self, root, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse RSS 1.0 (RDF) feed with improved structure handling - FIXED VERSION"""
        chunks = []
        
        try:
            # Register all namespaces for ElementTree
            for prefix, uri in self.namespaces.items():
                try:
                    ET.register_namespace(prefix, uri)
                except:
                    pass  # Ignore registration errors
            
            # Parse feed metadata from channel element with improved detection
            if self.include_feed_metadata:
                channel = self._find_rss1_channel(root)
                
                if channel is not None:
                    feed_metadata = self._extract_rss1_feed_metadata_enhanced(channel)
                    feed_chunk = self._create_feed_metadata_chunk(
                        feed_metadata, content, context, 'rss1'
                    )
                    chunks.append(feed_chunk)
                    logger.debug("RSS 1.0 feed metadata extracted successfully")
            
            # Parse items - RSS 1.0 items can be in multiple places
            items = self._find_rss1_items(root)
            
            logger.info(f"Found {len(items)} RSS 1.0 items")
            
            items = items[:self.max_entries_per_feed]  # Limit items
            
            for i, item in enumerate(items):
                item_chunk = self._parse_rss1_item_heuristic(item, context, i + 1)
                if item_chunk:
                    chunks.append(item_chunk)
            
            logger.info(f"Created {len(chunks)} chunks from RSS 1.0 feed")
            
        except Exception as e:
            logger.error(f"Error parsing RSS 1.0 feed: {e}")
            
            # Create a basic fallback chunk
            if not chunks:
                fallback_chunk = self._create_rss1_fallback_chunk(root, content, context)
                if fallback_chunk:
                    chunks.append(fallback_chunk)
        
        return chunks
    
    def _find_rss1_channel(self, root):
        """Find RSS 1.0 channel element using multiple strategies"""
        # Strategy 1: Direct namespace query
        try:
            channel = root.find('.//{http://purl.org/rss/1.0/}channel')
            if channel is not None:
                return channel
        except:
            pass
        
        # Strategy 2: Search by tag name (without namespace)
        for elem in root.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_name.lower() == 'channel':
                return elem
        
        # Strategy 3: Search by rdf:about attribute containing "channel"
        for elem in root.iter():
            about = elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if about and 'channel' in about.lower():
                return elem
        
        return None
    
    def _find_rss1_items(self, root):
        """Find RSS 1.0 item elements using multiple strategies"""
        items = []
        
        # Strategy 1: Direct namespace query
        try:
            items.extend(root.findall('.//{http://purl.org/rss/1.0/}item'))
        except:
            pass
        
        # Strategy 2: Search by tag name (without namespace)
        for elem in root.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_name.lower() == 'item':
                items.append(elem)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in items:
            # Use element memory address as unique identifier
            item_id = id(item)
            if item_id not in seen:
                seen.add(item_id)
                unique_items.append(item)
        
        return unique_items
    
    def _extract_rss1_feed_metadata_enhanced(self, channel) -> Dict[str, Any]:
        """Enhanced RSS 1.0 metadata extraction with multiple fallback strategies"""
        metadata = {
            'format': 'rss1',
        }
        
        # Extract basic elements using multiple strategies
        basic_elements = ['title', 'description', 'link']
        for elem_name in basic_elements:
            value = self._get_rss1_element_text_enhanced(channel, elem_name)
            metadata[elem_name] = value
        
        # Extract rdf:about
        about = channel.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
        if not about:
            about = channel.get('about')  # Try without namespace
        metadata['about'] = about
        
        # Extract Dublin Core metadata with enhanced search
        dc_elements = {
            'dc_creator': 'creator',
            'dc_date': 'date', 
            'dc_language': 'language',
            'dc_publisher': 'publisher',
            'dc_rights': 'rights'
        }
        
        for key, dc_elem in dc_elements.items():
            value = self._get_dublin_core_element(channel, dc_elem)
            if value:
                metadata[key] = value
        
        return metadata
    
    def _get_rss1_element_text_enhanced(self, parent, tag_name: str) -> Optional[str]:
        """Enhanced element text extraction for RSS 1.0 with multiple fallback strategies"""
        # Strategy 1: Try with RSS 1.0 namespace
        try:
            elem = parent.find(f'{{http://purl.org/rss/1.0/}}{tag_name}')
            if elem is not None and elem.text:
                return elem.text.strip()
        except:
            pass
        
        # Strategy 2: Try without namespace
        try:
            elem = parent.find(tag_name)
            if elem is not None and elem.text:
                return elem.text.strip()
        except:
            pass
        
        # Strategy 3: Search all children by tag name
        for child in parent:
            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if child_tag.lower() == tag_name.lower() and child.text:
                return child.text.strip()
        
        return None
    
    def _get_dublin_core_element(self, parent, dc_element: str) -> Optional[str]:
        """Extract Dublin Core element with namespace handling"""
        # Strategy 1: Full Dublin Core namespace
        try:
            elem = parent.find(f'{{http://purl.org/dc/elements/1.1/}}{dc_element}')
            if elem is not None and elem.text:
                return elem.text.strip()
        except:
            pass
        
        # Strategy 2: dc: prefix
        try:
            elem = parent.find(f'dc:{dc_element}')
            if elem is not None and elem.text:
                return elem.text.strip()
        except:
            pass
        
        # Strategy 3: Search all children
        for child in parent:
            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if child_tag.lower() == dc_element.lower() and child.text:
                return child.text.strip()
        
        return None
    
    def _parse_rss1_item_heuristic(self, item, context: ParseContext, item_num: int) -> Optional[SemanticChunk]:
        """Parse individual RSS 1.0 item with enhanced extraction"""
        # Extract basic item data using enhanced methods
        item_data = self._extract_rss1_item_data_enhanced(item)
        
        # Create readable content for the item
        content_parts = self._build_entry_content_parts(item_data)
        content = '\n\n'.join(content_parts)
        
        # Skip if content is too short
        if len(content) < self.min_entry_content_length:
            logger.debug(f"Skipping RSS 1.0 item {item_num}: content too short ({len(content)} chars)")
            return None
        
        chunk_id = create_chunk_id(
            context.file_path,
            item_num * 10,
            ChunkType.TEXT_BLOCK,
            f"rss1_item_{item_num}"
        )
        
        # Calculate importance
        importance = self._calculate_entry_importance(item_data, content)
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=content,
            start_line=item_num * 10,
            end_line=item_num * 10 + len(content.split('\n')),
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=importance,
            metadata={
                'parser': self.name,
                'parser_type': 'rss_atom',
                'feed_type': 'rss1',
                'entry_number': item_num,
                'entry_data': item_data,
                'extraction_method': 'heuristic',
                'semantic_type': 'RSS 1.0 Item'
            }
        )
        
        # Add semantic tags
        self._add_entry_semantic_tags(chunk, item_data)
        
        return chunk
    
    def _extract_rss1_item_data_enhanced(self, item) -> Dict[str, Any]:
        """Enhanced RSS 1.0 item data extraction"""
        item_data = {}
        
        # Extract basic RSS 1.0 elements
        basic_elements = ['title', 'description', 'link']
        for elem_name in basic_elements:
            value = self._get_rss1_element_text_enhanced(item, elem_name)
            item_data[elem_name] = value
        
        # Extract rdf:about
        about = item.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
        if not about:
            about = item.get('about')  # Try without namespace
        item_data['about'] = about
        
        # Extract Dublin Core metadata with enhanced methods
        dc_elements = {
            'dc_creator': 'creator',
            'dc_date': 'date',
            'dc_subject': 'subject',
            'dc_identifier': 'identifier'
        }
        
        for key, dc_elem in dc_elements.items():
            value = self._get_dublin_core_element(item, dc_elem)
            if value:
                item_data[key] = value
        
        # Map Dublin Core to standard RSS fields for compatibility
        if item_data.get('dc_creator'):
            item_data['author'] = item_data['dc_creator']
        
        if item_data.get('dc_date'):
            item_data['pubDate'] = item_data['dc_date']
        
        return item_data
    
    def _create_rss1_fallback_chunk(self, root, content: str, context: ParseContext) -> Optional[SemanticChunk]:
        """Create a fallback chunk for RSS 1.0 when parsing fails"""
        try:
            # Extract basic info using fallback methods
            items = self._find_rss1_items(root)
            channel = self._find_rss1_channel(root)
            
            title = "RSS 1.0 Feed"
            if channel is not None:
                title_text = self._get_rss1_element_text_enhanced(channel, 'title')
                if title_text:
                    title = title_text
            
            fallback_content = f"RSS 1.0 (RDF) Feed: {title}\nItems found: {len(items)}\nTotal size: {len(content)} characters"
            
            chunk_id = create_chunk_id(context.file_path, 1, ChunkType.TEXT_BLOCK, "rss1_fallback")
            
            chunk = SemanticChunk(
                id=chunk_id,
                file_path=context.file_path,
                content=fallback_content,
                start_line=1,
                end_line=3,
                content_type=context.content_type,
                chunk_type=ChunkType.TEXT_BLOCK,
                language=context.language,
                importance_score=0.6,
                metadata={
                    'parser': self.name,
                    'parser_type': 'rss_atom',
                    'feed_type': 'rss1',
                    'extraction_method': 'fallback',
                    'semantic_type': 'RSS 1.0 Fallback',
                    'items_found': len(items)
                }
            )
            
            if hasattr(chunk, 'add_tag'):
                chunk.add_tag('rss1', source='rss_atom_parser')
                chunk.add_tag('rdf', source='rss_atom_parser')
                chunk.add_tag('feed_fallback', source='rss_atom_parser')
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to create RSS 1.0 fallback chunk: {e}")
            return None
    
    # Tree-sitter specific methods (placeholders for now)
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from AST node for feeds"""
        if node.type == 'element':
            element_name = self._extract_element_name_from_node(node, content)
            if element_name:
                return element_name.lower()
        
        return node.type
    
    def _extract_atom_feed_tree_sitter(self, root_node, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Extract Atom feed using tree-sitter (placeholder for future implementation)"""
        # For now, fall back to heuristic parsing
        return self._parse_feed_heuristically(content, context)
    
    def _extract_rss_feed_tree_sitter(self, root_node, content: str, context: ParseContext, feed_type: str) -> List[SemanticChunk]:
        """Extract RSS feed using tree-sitter (placeholder for future implementation)"""
        # For now, fall back to heuristic parsing
        return self._parse_feed_heuristically(content, context)