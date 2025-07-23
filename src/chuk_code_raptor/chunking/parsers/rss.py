# src/chuk_code_raptor/chunking/parsers/rss.py
"""
RSS 2.0 Parser - Dedicated Implementation
==================================================

Focused parser for RSS 2.0 feeds with comprehensive support for:
- RSS 2.0 specification compliance
- Content:encoded extensions
- Dublin Core metadata
- Media enclosures (podcasts, videos)
- iTunes podcast extensions
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional, Dict, Any
from email.utils import parsedate_to_datetime
import logging

from ..tree_sitter_base import TreeSitterParser, get_tree_sitter_language_robust
from ..semantic_chunk import SemanticChunk, create_chunk_id
from ..base import ParseContext
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)


class RSSParser(TreeSitterParser):
    """
    Dedicated RSS 2.0 parser with comprehensive feature support.
    
    Supports:
    - RSS 2.0 specification
    - Content:encoded for full HTML content
    - Dublin Core metadata extensions
    - Media enclosures (RSS media extensions)
    - iTunes podcast extensions
    - Category and tag extraction
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.supported_languages = {'rss', 'rss2'}
        self.supported_extensions = {'.rss', '.xml'}
        self.name = "RSSParser"
        
        # RSS 2.0 specific configuration
        self.extract_full_content = getattr(config, 'rss2_extract_full_content', True)
        self.prefer_content_encoded = getattr(config, 'rss2_prefer_content_encoded', True)
        self.clean_html_content = getattr(config, 'rss2_clean_html_content', True)
        self.extract_media_metadata = getattr(config, 'rss2_extract_media_metadata', True)
        self.extract_itunes_metadata = getattr(config, 'rss2_extract_itunes_metadata', True)
        self.max_items_per_feed = getattr(config, 'rss2_max_items_per_feed', 200)
        self.min_item_content_length = getattr(config, 'rss2_min_item_content_length', 30)
        self.include_feed_metadata = getattr(config, 'rss2_include_feed_metadata', True)
        
        # RSS 2.0 namespaces
        self.namespaces = {
            'content': 'http://purl.org/rss/1.0/modules/content/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'media': 'http://search.yahoo.com/mrss/',
            'itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd',
            'atom': 'http://www.w3.org/2005/Atom'
        }
    
    def _get_tree_sitter_language(self):
        """Get tree-sitter XML language"""
        language, package_used = get_tree_sitter_language_robust(
            'xml', 
            ['tree_sitter_xml', 'tree_sitter_languages']
        )
        
        if language is None:
            raise ImportError("tree-sitter XML package required for RSS parsing")
        
        self._package_used = package_used
        return language
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """XML node types for RSS 2.0"""
        return {
            'element': ChunkType.TEXT_BLOCK,
            'start_tag': ChunkType.TEXT_BLOCK,
            'comment': ChunkType.COMMENT,
            'text': ChunkType.TEXT_BLOCK,
            'cdata_section': ChunkType.TEXT_BLOCK
        }
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from RSS node - not applicable for RSS XML"""
        return None
    
    def parse(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse RSS 2.0 feed content"""
        if not content.strip():
            return []
        
        # Validate RSS 2.0 format
        if not self._is_rss2_feed(content):
            logger.warning(f"Content does not appear to be RSS 2.0: {context.file_path}")
            return []
        
        try:
            # Parse with ElementTree for RSS 2.0 structure
            root = ET.fromstring(content.strip())
            
            # Verify RSS 2.0 structure
            if root.tag != 'rss':
                logger.warning(f"Root element is not <rss>: {root.tag}")
                return []
            
            version = root.get('version', '2.0')
            if not version.startswith('2.'):
                logger.warning(f"RSS version {version} may not be fully supported")
            
            # Find channel element
            channel = root.find('channel')
            if channel is None:
                logger.error("No <channel> element found in RSS feed")
                return []
            
            chunks = []
            
            # Extract feed metadata
            if self.include_feed_metadata:
                metadata_chunk = self._create_feed_metadata_chunk(channel, context)
                if metadata_chunk:
                    chunks.append(metadata_chunk)
            
            # Extract items
            items = channel.findall('item')
            logger.info(f"Found {len(items)} RSS 2.0 items")
            
            # Limit items if configured
            if len(items) > self.max_items_per_feed:
                logger.info(f"Limiting to {self.max_items_per_feed} items")
                items = items[:self.max_items_per_feed]
            
            # Process each item
            for i, item in enumerate(items, 1):
                item_chunk = self._parse_rss_item(item, context, i)
                if item_chunk:
                    chunks.append(item_chunk)
            
            logger.info(f"Created {len(chunks)} chunks from RSS 2.0 feed")
            return chunks
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error in RSS 2.0 feed {context.file_path}: {e}")
            return self._create_fallback_chunk(content, context, f"XML Parse Error: {e}")
        
        except Exception as e:
            logger.error(f"Error parsing RSS 2.0 feed {context.file_path}: {e}")
            return self._create_fallback_chunk(content, context, f"Parse Error: {e}")
    
    def _is_rss2_feed(self, content: str) -> bool:
        """Check if content is RSS 2.0 format"""
        content_lower = content.lower()
        
        # Check for RSS 2.0 indicators
        rss_indicators = [
            '<rss version="2.0"',
            '<rss version="2.1"',  # Some feeds use 2.1
            '<rss xmlns='
        ]
        
        return any(indicator in content_lower for indicator in rss_indicators)
    
    def _create_feed_metadata_chunk(self, channel, context: ParseContext) -> Optional[SemanticChunk]:
        """Create feed metadata chunk from RSS 2.0 channel"""
        try:
            # Extract channel metadata
            metadata = self._extract_channel_metadata(channel)
            
            # Create readable summary
            summary_parts = ["=== RSS 2.0 FEED METADATA ==="]
            
            # Basic information
            if metadata.get('title'):
                summary_parts.append(f"Title: {metadata['title']}")
            
            if metadata.get('description'):
                desc = metadata['description']
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                summary_parts.append(f"Description: {desc}")
            
            if metadata.get('link'):
                summary_parts.append(f"Website: {metadata['link']}")
            
            # Extended metadata
            if metadata.get('language'):
                summary_parts.append(f"Language: {metadata['language']}")
            
            if metadata.get('copyright'):
                summary_parts.append(f"Copyright: {metadata['copyright']}")
            
            if metadata.get('managingEditor'):
                summary_parts.append(f"Editor: {metadata['managingEditor']}")
            
            if metadata.get('lastBuildDate'):
                summary_parts.append(f"Last Updated: {metadata['lastBuildDate']}")
            
            # Categories
            categories = metadata.get('categories', [])
            if categories:
                cat_list = [cat.get('text', str(cat)) for cat in categories[:5]]
                summary_parts.append(f"Categories: {', '.join(cat_list)}")
            
            # iTunes podcast metadata
            itunes_meta = metadata.get('itunes', {})
            if itunes_meta:
                if itunes_meta.get('author'):
                    summary_parts.append(f"Podcast Author: {itunes_meta['author']}")
                if itunes_meta.get('category'):
                    summary_parts.append(f"Podcast Category: {itunes_meta['category']}")
                if itunes_meta.get('explicit'):
                    summary_parts.append(f"Explicit Content: {itunes_meta['explicit']}")
            
            summary = '\n'.join(summary_parts)
            
            chunk_id = create_chunk_id(
                context.file_path, 
                1, 
                ChunkType.METADATA, 
                "rss2_feed_metadata"
            )
            
            chunk = SemanticChunk(
                id=chunk_id,
                file_path=context.file_path,
                content=summary,
                start_line=1,
                end_line=len(summary_parts),
                content_type=context.content_type,
                chunk_type=ChunkType.METADATA,
                language=context.language,
                importance_score=0.95,
                metadata={
                    'parser': self.name,
                    'feed_type': 'rss2',
                    'semantic_type': 'RSS 2.0 Feed Metadata',
                    'feed_metadata': metadata,
                    'extraction_method': 'rss2_parser'
                }
            )
            
            # Add semantic tags
            if hasattr(chunk, 'add_tag'):
                chunk.add_tag('rss2_feed', source='rss2_parser')
                chunk.add_tag('feed_metadata', source='rss2_parser')
                chunk.add_tag('syndication', source='rss2_parser')
                
                # Add category tags
                for category in categories[:10]:  # Limit tags
                    cat_text = category.get('text', str(category))
                    if cat_text:
                        chunk.add_tag(f"category:{cat_text.lower()}", source='rss2_parser')
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating RSS 2.0 metadata chunk: {e}")
            return None
    
    def _extract_channel_metadata(self, channel) -> Dict[str, Any]:
        """Extract metadata from RSS 2.0 channel element"""
        metadata = {'format': 'rss2'}
        
        # Basic RSS 2.0 elements
        basic_elements = {
            'title': 'title',
            'description': 'description', 
            'link': 'link',
            'language': 'language',
            'copyright': 'copyright',
            'managingEditor': 'managingEditor',
            'webMaster': 'webMaster',
            'pubDate': 'pubDate',
            'lastBuildDate': 'lastBuildDate',
            'generator': 'generator',
            'docs': 'docs',
            'ttl': 'ttl',
            'rating': 'rating'
        }
        
        for key, element_name in basic_elements.items():
            elem = channel.find(element_name)
            if elem is not None and elem.text:
                metadata[key] = elem.text.strip()
        
        # Categories
        categories = []
        for category in channel.findall('category'):
            cat_data = {
                'text': category.text.strip() if category.text else '',
                'domain': category.get('domain')
            }
            categories.append(cat_data)
        metadata['categories'] = categories
        
        # Image
        image = channel.find('image')
        if image is not None:
            metadata['image'] = {
                'url': self._get_element_text(image, 'url'),
                'title': self._get_element_text(image, 'title'),
                'link': self._get_element_text(image, 'link'),
                'width': self._get_element_text(image, 'width'),
                'height': self._get_element_text(image, 'height'),
                'description': self._get_element_text(image, 'description')
            }
        
        # iTunes podcast metadata
        if self.extract_itunes_metadata:
            metadata['itunes'] = self._extract_itunes_metadata(channel)
        
        # Atom extensions (some RSS 2.0 feeds include Atom elements)
        atom_link = channel.find('{http://www.w3.org/2005/Atom}link')
        if atom_link is not None:
            metadata['atom_link'] = {
                'href': atom_link.get('href'),
                'rel': atom_link.get('rel'),
                'type': atom_link.get('type')
            }
        
        return metadata
    
    def _parse_rss_item(self, item, context: ParseContext, item_num: int) -> Optional[SemanticChunk]:
        """Parse individual RSS 2.0 item"""
        try:
            # Extract item data
            item_data = self._extract_item_data(item)
            
            # Create content summary
            content_parts = self._build_item_content(item_data)
            content = '\n\n'.join(content_parts)
            
            # Skip if content too short
            if len(content) < self.min_item_content_length:
                logger.debug(f"Skipping RSS item {item_num}: content too short")
                return None
            
            # Calculate line numbers (approximate)
            start_line = item_num * 20  # Rough estimate
            end_line = start_line + len(content.split('\n'))
            
            chunk_id = create_chunk_id(
                context.file_path,
                start_line,
                ChunkType.TEXT_BLOCK,
                f"rss2_item_{item_num}"
            )
            
            # Calculate importance score
            importance = self._calculate_item_importance(item_data, content)
            
            chunk = SemanticChunk(
                id=chunk_id,
                file_path=context.file_path,
                content=content,
                start_line=start_line,
                end_line=end_line,
                content_type=context.content_type,
                chunk_type=ChunkType.TEXT_BLOCK,
                language=context.language,
                importance_score=importance,
                metadata={
                    'parser': self.name,
                    'feed_type': 'rss2',
                    'semantic_type': 'RSS 2.0 Item',
                    'item_number': item_num,
                    'item_data': item_data,
                    'extraction_method': 'rss2_parser'
                }
            )
            
            # Add semantic tags
            self._add_item_semantic_tags(chunk, item_data)
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error parsing RSS 2.0 item {item_num}: {e}")
            return None
    
    def _extract_item_data(self, item) -> Dict[str, Any]:
        """Extract data from RSS 2.0 item element"""
        item_data = {}
        
        # Basic RSS 2.0 item elements
        basic_elements = {
            'title': 'title',
            'description': 'description',
            'link': 'link',
            'author': 'author',
            'category': 'category',
            'comments': 'comments',
            'pubDate': 'pubDate',
            'source': 'source'
        }
        
        for key, element_name in basic_elements.items():
            elem = item.find(element_name)
            if elem is not None and elem.text:
                item_data[key] = elem.text.strip()
        
        # GUID
        guid = item.find('guid')
        if guid is not None:
            item_data['guid'] = {
                'value': guid.text.strip() if guid.text else '',
                'isPermaLink': guid.get('isPermaLink', 'true')
            }
        
        # Content:encoded (full HTML content)
        content_elem = item.find('{http://purl.org/rss/1.0/modules/content/}encoded')
        if content_elem is not None and self.prefer_content_encoded:
            item_data['content_encoded'] = content_elem.text or ''
            item_data['has_full_content'] = True
        
        # Categories (can be multiple)
        categories = []
        for category in item.findall('category'):
            cat_data = {
                'text': category.text.strip() if category.text else '',
                'domain': category.get('domain')
            }
            categories.append(cat_data)
        item_data['categories'] = categories
        
        # Enclosures (media files)
        enclosures = []
        for enclosure in item.findall('enclosure'):
            enc_data = {
                'url': enclosure.get('url'),
                'type': enclosure.get('type'),
                'length': enclosure.get('length')
            }
            enclosures.append(enc_data)
        item_data['enclosures'] = enclosures
        
        # Dublin Core metadata
        item_data.update(self._extract_dublin_core_metadata(item))
        
        # iTunes podcast metadata
        if self.extract_itunes_metadata:
            item_data['itunes'] = self._extract_itunes_metadata(item)
        
        # Media RSS metadata
        if self.extract_media_metadata:
            item_data['media'] = self._extract_media_metadata(item)
        
        return item_data
    
    def _build_item_content(self, item_data: Dict[str, Any]) -> List[str]:
        """Build readable content from item data"""
        content_parts = []
        
        # Title
        if item_data.get('title'):
            content_parts.append(f"Title: {item_data['title']}")
        
        # Author
        if item_data.get('author'):
            content_parts.append(f"Author: {item_data['author']}")
        elif item_data.get('dc_creator'):
            content_parts.append(f"Author: {item_data['dc_creator']}")
        
        # Publication date
        if item_data.get('pubDate'):
            content_parts.append(f"Published: {item_data['pubDate']}")
        
        # Categories
        categories = item_data.get('categories', [])
        if categories:
            cat_texts = [cat.get('text', '') for cat in categories if cat.get('text')]
            if cat_texts:
                content_parts.append(f"Categories: {', '.join(cat_texts)}")
        
        # Description
        if item_data.get('description'):
            desc = item_data['description']
            content_parts.append(f"Description: {desc}")
        
        # Full content (content:encoded)
        if item_data.get('content_encoded') and self.extract_full_content:
            full_content = item_data['content_encoded']
            if self.clean_html_content:
                full_content = self._clean_html_content(full_content)
            content_parts.append(f"Full Content: {full_content}")
        
        # Link
        if item_data.get('link'):
            content_parts.append(f"Link: {item_data['link']}")
        
        # Media information
        enclosures = item_data.get('enclosures', [])
        if enclosures:
            media_info = []
            for enc in enclosures:
                media_type = enc.get('type', 'unknown')
                url = enc.get('url', '')
                length = enc.get('length', '')
                
                info = f"{media_type}"
                if length:
                    try:
                        # Convert bytes to MB for display
                        length_mb = int(length) / (1024 * 1024)
                        info += f" ({length_mb:.1f} MB)"
                    except:
                        info += f" ({length} bytes)"
                
                if url:
                    filename = url.split('/')[-1]
                    info += f" - {filename}"
                
                media_info.append(info)
            
            content_parts.append(f"Media: {'; '.join(media_info)}")
        
        return content_parts
    
    def _calculate_item_importance(self, item_data: Dict[str, Any], content: str) -> float:
        """Calculate importance score for RSS item"""
        score = 0.7  # Base score
        
        # Content richness
        if item_data.get('content_encoded'):
            score += 0.15
        
        if len(content) > 1000:
            score += 0.1
        elif len(content) > 500:
            score += 0.05
        
        # Metadata completeness
        if item_data.get('author') or item_data.get('dc_creator'):
            score += 0.05
        
        if item_data.get('categories'):
            score += 0.05
        
        # Media content
        if item_data.get('enclosures'):
            score += 0.1
        
        # Recent content (if parseable date)
        if item_data.get('pubDate'):
            try:
                pub_date = parsedate_to_datetime(item_data['pubDate'])
                days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
                if days_old < 7:
                    score += 0.05
                elif days_old < 30:
                    score += 0.02
            except:
                pass
        
        return min(score, 1.0)
    
    def _add_item_semantic_tags(self, chunk: SemanticChunk, item_data: Dict[str, Any]):
        """Add semantic tags to RSS item chunks"""
        if not hasattr(chunk, 'add_tag'):
            return
        
        # Base tags
        chunk.add_tag('rss2_item', source='rss2_parser')
        chunk.add_tag('rss_item', source='rss2_parser')
        chunk.add_tag('article', source='rss2_parser')
        
        # Category tags
        categories = item_data.get('categories', [])
        for category in categories:
            cat_text = category.get('text', '')
            if cat_text:
                chunk.add_tag(f"topic:{cat_text.lower()}", source='rss2_parser')
        
        # Media tags
        enclosures = item_data.get('enclosures', [])
        if enclosures:
            chunk.add_tag('multimedia', source='rss2_parser')
            
            for enc in enclosures:
                media_type = enc.get('type', '').lower()
                if 'audio' in media_type:
                    chunk.add_tag('podcast', source='rss2_parser')
                    chunk.add_tag('audio', source='rss2_parser')
                elif 'video' in media_type:
                    chunk.add_tag('video', source='rss2_parser')
                elif 'image' in media_type:
                    chunk.add_tag('image', source='rss2_parser')
        
        # Content type tags
        if item_data.get('content_encoded'):
            chunk.add_tag('full_content', source='rss2_parser')
        
        if item_data.get('author') or item_data.get('dc_creator'):
            chunk.add_tag('authored', source='rss2_parser')
    
    # Utility methods
    
    def _get_element_text(self, parent, tag_name: str) -> Optional[str]:
        """Get text from child element"""
        elem = parent.find(tag_name)
        return elem.text.strip() if elem is not None and elem.text else None
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content for readable display"""
        if not html_content:
            return ""
        
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', html_content)
        
        # Clean up whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = clean_text.strip()
        
        # Limit length for chunk content
        if len(clean_text) > 2000:
            clean_text = clean_text[:1997] + "..."
        
        return clean_text
    
    def _extract_dublin_core_metadata(self, element) -> Dict[str, Any]:
        """Extract Dublin Core metadata elements"""
        dc_data = {}
        
        dc_elements = {
            'dc_creator': '{http://purl.org/dc/elements/1.1/}creator',
            'dc_date': '{http://purl.org/dc/elements/1.1/}date',
            'dc_subject': '{http://purl.org/dc/elements/1.1/}subject',
            'dc_publisher': '{http://purl.org/dc/elements/1.1/}publisher',
            'dc_rights': '{http://purl.org/dc/elements/1.1/}rights'
        }
        
        for key, xpath in dc_elements.items():
            elem = element.find(xpath)
            if elem is not None and elem.text:
                dc_data[key] = elem.text.strip()
        
        return dc_data
    
    def _extract_itunes_metadata(self, element) -> Dict[str, Any]:
        """Extract iTunes podcast metadata"""
        itunes_data = {}
        
        itunes_elements = {
            'author': '{http://www.itunes.com/dtds/podcast-1.0.dtd}author',
            'subtitle': '{http://www.itunes.com/dtds/podcast-1.0.dtd}subtitle',
            'summary': '{http://www.itunes.com/dtds/podcast-1.0.dtd}summary',
            'duration': '{http://www.itunes.com/dtds/podcast-1.0.dtd}duration',
            'explicit': '{http://www.itunes.com/dtds/podcast-1.0.dtd}explicit',
            'episode': '{http://www.itunes.com/dtds/podcast-1.0.dtd}episode',
            'season': '{http://www.itunes.com/dtds/podcast-1.0.dtd}season',
            'episodeType': '{http://www.itunes.com/dtds/podcast-1.0.dtd}episodeType'
        }
        
        for key, xpath in itunes_elements.items():
            elem = element.find(xpath)
            if elem is not None and elem.text:
                itunes_data[key] = elem.text.strip()
        
        # iTunes category (special handling)
        category = element.find('{http://www.itunes.com/dtds/podcast-1.0.dtd}category')
        if category is not None:
            itunes_data['category'] = category.get('text', '')
        
        # iTunes image
        image = element.find('{http://www.itunes.com/dtds/podcast-1.0.dtd}image')
        if image is not None:
            itunes_data['image'] = image.get('href', '')
        
        return itunes_data
    
    def _extract_media_metadata(self, element) -> Dict[str, Any]:
        """Extract Media RSS metadata"""
        media_data = {}
        
        # Media content
        media_content = element.find('{http://search.yahoo.com/mrss/}content')
        if media_content is not None:
            media_data['content'] = {
                'url': media_content.get('url'),
                'type': media_content.get('type'),
                'medium': media_content.get('medium'),
                'duration': media_content.get('duration'),
                'width': media_content.get('width'),
                'height': media_content.get('height')
            }
        
        # Media thumbnail
        media_thumbnail = element.find('{http://search.yahoo.com/mrss/}thumbnail')
        if media_thumbnail is not None:
            media_data['thumbnail'] = {
                'url': media_thumbnail.get('url'),
                'width': media_thumbnail.get('width'),
                'height': media_thumbnail.get('height')
            }
        
        return media_data
    
    def _create_fallback_chunk(self, content: str, context: ParseContext, error_msg: str) -> List[SemanticChunk]:
        """Create fallback chunk when parsing fails"""
        chunk_id = create_chunk_id(
            context.file_path, 
            1, 
            ChunkType.TEXT_BLOCK, 
            "rss2_fallback"
        )
        
        # Basic content analysis
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "RSS 2.0 Feed"
        
        fallback_content = f"RSS 2.0 Feed: {title}\nSize: {len(content)} characters\nError: {error_msg}"
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=fallback_content,
            start_line=1,
            end_line=3,
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=0.3,
            metadata={
                'parser': self.name,
                'feed_type': 'rss2',
                'semantic_type': 'RSS 2.0 Fallback',
                'extraction_method': 'fallback',
                'error': error_msg
            }
        )
        
        if hasattr(chunk, 'add_tag'):
            chunk.add_tag('rss2_fallback', source='rss2_parser')
            chunk.add_tag('parsing_error', source='rss2_parser')
        
        return [chunk]