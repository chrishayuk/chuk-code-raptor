# src/chuk_code_raptor/chunking/parsers/rdf.py
"""
RSS 1.0 (RDF) Parser - Dedicated Implementation
=======================================================

Specialized parser for RSS 1.0 (RDF/Site Summary) feeds with proper handling of:
- RDF structure where items are siblings of channel, not children
- Dublin Core metadata extensions
- RSS 1.0 modules (syndication, content, etc.)
- Proper namespace handling for academic and scientific feeds
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from ..tree_sitter_base import TreeSitterParser, get_tree_sitter_language_robust
from ..semantic_chunk import SemanticChunk, create_chunk_id
from ..base import ParseContext
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)


class RDFParser(TreeSitterParser):
    """
    Dedicated RSS 1.0 (RDF) parser with proper structural understanding.
    
    Key Features:
    - Correct RSS 1.0 structure handling (items as siblings of channel)
    - Comprehensive RDF namespace support
    - Dublin Core metadata extraction
    - RSS 1.0 modules support (syndication, content, etc.)
    - Academic and scientific feed optimization
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.supported_languages = {'rss1', 'rdf'}
        self.supported_extensions = {'.rdf', '.xml', '.rss'}
        self.name = "RSS1RDFParser"
        
        # RSS 1.0 specific configuration
        self.extract_dublin_core = getattr(config, 'rss1_extract_dublin_core', True)
        self.extract_syndication_info = getattr(config, 'rss1_extract_syndication_info', True)
        self.max_items_per_feed = getattr(config, 'rss1_max_items_per_feed', 100)
        self.min_item_content_length = getattr(config, 'rss1_min_item_content_length', 20)
        self.include_feed_metadata = getattr(config, 'rss1_include_feed_metadata', True)
        self.preserve_rdf_structure = getattr(config, 'rss1_preserve_rdf_structure', True)
        
        # RSS 1.0 RDF namespaces
        self.namespaces = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rss': 'http://purl.org/rss/1.0/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'sy': 'http://purl.org/rss/1.0/modules/syndication/',
            'content': 'http://purl.org/rss/1.0/modules/content/',
            'admin': 'http://webns.net/mvcb/',
            'cc': 'http://web.resource.org/cc/',
            'foaf': 'http://xmlns.com/foaf/0.1/'
        }
        
        # Register namespaces with ElementTree
        for prefix, uri in self.namespaces.items():
            try:
                ET.register_namespace(prefix, uri)
            except:
                pass  # Ignore registration errors
    
    def _get_tree_sitter_language(self):
        """Get tree-sitter XML language"""
        language, package_used = get_tree_sitter_language_robust(
            'xml', 
            ['tree_sitter_xml', 'tree_sitter_languages']
        )
        
        if language is None:
            raise ImportError("tree-sitter XML package required for RSS 1.0 parsing")
        
        self._package_used = package_used
        return language
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """XML node types for RSS 1.0 RDF"""
        return {
            'element': ChunkType.TEXT_BLOCK,
            'start_tag': ChunkType.TEXT_BLOCK,
            'comment': ChunkType.COMMENT,
            'text': ChunkType.TEXT_BLOCK,
            'cdata_section': ChunkType.TEXT_BLOCK
        }
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from RDF node - not applicable for RDF XML"""
        return None
    
    def parse(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse RSS 1.0 (RDF) feed content"""
        if not content.strip():
            return []
        
        # Validate RSS 1.0 RDF format
        if not self._is_rss1_rdf_feed(content):
            logger.warning(f"Content does not appear to be RSS 1.0 RDF: {context.file_path}")
            return []
        
        try:
            # Parse with ElementTree
            root = ET.fromstring(content.strip())
            
            # Verify RDF structure
            if not self._is_rdf_root(root):
                logger.warning(f"Root element is not RDF: {root.tag}")
                return []
            
            logger.info(f"Processing RSS 1.0 RDF feed with root: {root.tag}")
            
            chunks = []
            
            # Find channel element (should be a direct child of RDF)
            channel = self._find_rss1_channel(root)
            
            # Extract feed metadata
            if self.include_feed_metadata and channel is not None:
                metadata_chunk = self._create_feed_metadata_chunk(channel, root, context)
                if metadata_chunk:
                    chunks.append(metadata_chunk)
            
            # Find items (CRITICAL: items are siblings of channel in RSS 1.0)
            items = self._find_rss1_items(root)
            logger.info(f"Found {len(items)} RSS 1.0 items")
            
            # Debug: log item structure
            for i, item in enumerate(items[:3]):  # Log first 3 items
                title = self._get_rss1_element_text(item, 'title')
                about = item.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', 'no-about')
                logger.debug(f"RSS 1.0 Item {i+1}: title='{title}', about='{about}'")
            
            # Limit items if configured
            if len(items) > self.max_items_per_feed:
                logger.info(f"Limiting to {self.max_items_per_feed} items")
                items = items[:self.max_items_per_feed]
            
            # Process each item
            for i, item in enumerate(items, 1):
                item_chunk = self._parse_rss1_item(item, context, i)
                if item_chunk:
                    chunks.append(item_chunk)
                else:
                    logger.debug(f"No chunk created for RSS 1.0 item {i}")
            
            logger.info(f"Created {len(chunks)} chunks from RSS 1.0 RDF feed")
            return chunks
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error in RSS 1.0 feed {context.file_path}: {e}")
            return self._create_fallback_chunk(content, context, f"XML Parse Error: {e}")
        
        except Exception as e:
            logger.error(f"Error parsing RSS 1.0 feed {context.file_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_fallback_chunk(content, context, f"Parse Error: {e}")
    
    def _is_rss1_rdf_feed(self, content: str) -> bool:
        """Check if content is RSS 1.0 RDF format"""
        content_lower = content.lower()
        
        # RSS 1.0 indicators
        rss1_indicators = [
            '<rdf:rdf',
            'xmlns:rss="http://purl.org/rss/1.0/"',
            'rss/1.0/',
            'rdf-syntax-ns'
        ]
        
        return any(indicator in content_lower for indicator in rss1_indicators)
    
    def _is_rdf_root(self, root) -> bool:
        """Check if root element is RDF"""
        # Handle both namespaced and non-namespaced RDF
        tag_name = root.tag.split('}')[-1] if '}' in root.tag else root.tag
        return tag_name.lower() == 'rdf' or root.tag == '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF'
    
    def _find_rss1_channel(self, root) -> Optional[ET.Element]:
        """Find RSS 1.0 channel element (direct child of RDF)"""
        # Strategy 1: Direct namespace search
        channel = root.find('{http://purl.org/rss/1.0/}channel')
        if channel is not None:
            logger.debug("Found channel via RSS 1.0 namespace")
            return channel
        
        # Strategy 2: Search direct children by tag name
        for child in root:
            tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag_name.lower() == 'channel':
                logger.debug("Found channel via tag name search")
                return child
        
        # Strategy 3: Search by rdf:about attribute pattern
        for child in root:
            about = child.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '')
            if about and ('channel' in about.lower() or about.endswith('/')):
                # Channel typically has rdf:about pointing to the site URL
                tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if tag_name.lower() == 'channel':
                    logger.debug("Found channel via rdf:about pattern")
                    return child
        
        logger.warning("No RSS 1.0 channel element found")
        return None
    
    def _find_rss1_items(self, root) -> List[ET.Element]:
        """Find RSS 1.0 item elements (direct children of RDF root)"""
        items = []
        
        # RSS 1.0 CRITICAL DIFFERENCE: Items are direct children of RDF, not inside channel
        
        # Strategy 1: Direct namespace search for items at root level
        for child in root:
            if child.tag == '{http://purl.org/rss/1.0/}item':
                items.append(child)
        
        # Strategy 2: Search direct children by tag name
        if not items:
            for child in root:
                tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if tag_name.lower() == 'item':
                    items.append(child)
        
        # Strategy 3: Verify items by rdf:about attribute presence
        verified_items = []
        for item in items:
            about = item.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if about:  # RSS 1.0 items should have rdf:about
                verified_items.append(item)
            else:
                # Check for about without namespace
                about_alt = item.get('about')
                if about_alt:
                    verified_items.append(item)
        
        if verified_items:
            logger.debug(f"Verified {len(verified_items)} RSS 1.0 items with rdf:about attributes")
            return verified_items
        
        logger.debug(f"Found {len(items)} RSS 1.0 items (unverified)")
        return items
    
    def _create_feed_metadata_chunk(self, channel, root, context: ParseContext) -> Optional[SemanticChunk]:
        """Create feed metadata chunk from RSS 1.0 channel and RDF root"""
        try:
            # Extract channel and RDF metadata
            metadata = self._extract_rss1_metadata(channel, root)
            
            # Create readable summary
            summary_parts = ["=== RSS 1.0 (RDF) FEED METADATA ==="]
            
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
            
            # RDF specific metadata
            if metadata.get('about'):
                summary_parts.append(f"RDF About: {metadata['about']}")
            
            # Dublin Core metadata
            dc_elements = [
                ('dc_creator', 'Creator'),
                ('dc_publisher', 'Publisher'),
                ('dc_date', 'Date'),
                ('dc_language', 'Language'),
                ('dc_rights', 'Rights'),
                ('dc_subject', 'Subject')
            ]
            
            for dc_key, dc_label in dc_elements:
                if metadata.get(dc_key):
                    summary_parts.append(f"{dc_label}: {metadata[dc_key]}")
            
            # Syndication information
            sy_info = metadata.get('syndication', {})
            if sy_info:
                if sy_info.get('updatePeriod'):
                    update_freq = f"{sy_info.get('updateFrequency', '1')} times per {sy_info['updatePeriod']}"
                    summary_parts.append(f"Update Frequency: {update_freq}")
            
            # Items count from RDF structure
            items_seq = metadata.get('items_sequence', [])
            if items_seq:
                summary_parts.append(f"Declared Items: {len(items_seq)}")
            
            summary = '\n'.join(summary_parts)
            
            chunk_id = create_chunk_id(
                context.file_path, 
                1, 
                ChunkType.METADATA, 
                "rss1_feed_metadata"
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
                    'feed_type': 'rss1',
                    'semantic_type': 'RSS 1.0 RDF Feed Metadata',
                    'feed_metadata': metadata,
                    'extraction_method': 'rss1_rdf_parser'
                }
            )
            
            # Add semantic tags
            if hasattr(chunk, 'add_tag'):
                chunk.add_tag('rss1_feed', source='rss1_rdf_parser')
                chunk.add_tag('rdf_feed', source='rss1_rdf_parser')
                chunk.add_tag('feed_metadata', source='rss1_rdf_parser')
                chunk.add_tag('syndication', source='rss1_rdf_parser')
                chunk.add_tag('dublin_core', source='rss1_rdf_parser')
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating RSS 1.0 metadata chunk: {e}")
            return None
    
    def _extract_rss1_metadata(self, channel, root) -> Dict[str, Any]:
        """Extract metadata from RSS 1.0 channel and RDF root"""
        metadata = {'format': 'rss1'}
        
        if channel is not None:
            # Basic RSS 1.0 channel elements
            basic_elements = ['title', 'description', 'link']
            for element_name in basic_elements:
                value = self._get_rss1_element_text(channel, element_name)
                if value:
                    metadata[element_name] = value
            
            # RDF about attribute
            about = channel.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if not about:
                about = channel.get('about')
            metadata['about'] = about
            
            # Dublin Core metadata from channel
            metadata.update(self._extract_dublin_core_metadata(channel))
            
            # Syndication metadata from channel
            if self.extract_syndication_info:
                metadata['syndication'] = self._extract_syndication_metadata(channel)
        
        # Extract items sequence from RDF structure
        items_elem = root.find('{http://purl.org/rss/1.0/}items')
        if items_elem is not None:
            seq_elem = items_elem.find('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Seq')
            if seq_elem is not None:
                items_sequence = []
                for li in seq_elem.findall('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li'):
                    resource = li.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                    if resource:
                        items_sequence.append(resource)
                metadata['items_sequence'] = items_sequence
        
        return metadata
    
    def _parse_rss1_item(self, item, context: ParseContext, item_num: int) -> Optional[SemanticChunk]:
        """Parse individual RSS 1.0 item"""
        try:
            # Extract item data
            item_data = self._extract_rss1_item_data(item)
            
            # Debug log
            logger.debug(f"RSS 1.0 Item {item_num} data: {list(item_data.keys())}")
            
            # Create content summary
            content_parts = self._build_rss1_item_content(item_data)
            content = '\n\n'.join(content_parts)
            
            # Skip if content too short (be more lenient for academic content)
            min_length = max(self.min_item_content_length, 15)
            if len(content) < min_length:
                logger.debug(f"Skipping RSS 1.0 item {item_num}: content too short ({len(content)} < {min_length})")
                return None
            
            # Calculate line numbers (approximate)
            start_line = item_num * 15  # RSS 1.0 items tend to be shorter
            end_line = start_line + len(content.split('\n'))
            
            chunk_id = create_chunk_id(
                context.file_path,
                start_line,
                ChunkType.TEXT_BLOCK,
                f"rss1_item_{item_num}"
            )
            
            # Calculate importance score
            importance = self._calculate_rss1_item_importance(item_data, content)
            
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
                    'feed_type': 'rss1',
                    'semantic_type': 'RSS 1.0 RDF Item',
                    'item_number': item_num,
                    'item_data': item_data,
                    'extraction_method': 'rss1_rdf_parser'
                }
            )
            
            # Add semantic tags
            self._add_rss1_item_semantic_tags(chunk, item_data)
            
            logger.debug(f"Created RSS 1.0 item chunk {item_num} with {len(content)} characters")
            return chunk
            
        except Exception as e:
            logger.error(f"Error parsing RSS 1.0 item {item_num}: {e}")
            return None
    
    def _extract_rss1_item_data(self, item) -> Dict[str, Any]:
        """Extract data from RSS 1.0 item element"""
        item_data = {}
        
        # Basic RSS 1.0 item elements
        basic_elements = ['title', 'description', 'link']
        for element_name in basic_elements:
            value = self._get_rss1_element_text(item, element_name)
            if value:
                item_data[element_name] = value
        
        # RDF about attribute (unique identifier for RSS 1.0 items)
        about = item.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
        if not about:
            about = item.get('about')
        item_data['about'] = about
        
        # Dublin Core metadata (very common in RSS 1.0)
        if self.extract_dublin_core:
            dc_data = self._extract_dublin_core_metadata(item)
            item_data.update(dc_data)
            
            # Map Dublin Core to standard RSS fields for compatibility
            if dc_data.get('dc_creator'):
                item_data['author'] = dc_data['dc_creator']
            if dc_data.get('dc_date'):
                item_data['pubDate'] = dc_data['dc_date']
        
        # Content module (less common in RSS 1.0 but possible)
        content_elem = item.find('{http://purl.org/rss/1.0/modules/content/}encoded')
        if content_elem is not None and content_elem.text:
            item_data['content_encoded'] = content_elem.text.strip()
        
        return item_data
    
    def _build_rss1_item_content(self, item_data: Dict[str, Any]) -> List[str]:
        """Build readable content from RSS 1.0 item data"""
        content_parts = []
        
        # Title
        if item_data.get('title'):
            content_parts.append(f"Title: {item_data['title']}")
        
        # Author (from Dublin Core creator)
        if item_data.get('dc_creator'):
            content_parts.append(f"Creator: {item_data['dc_creator']}")
        elif item_data.get('author'):
            content_parts.append(f"Author: {item_data['author']}")
        
        # Publication date (from Dublin Core)
        if item_data.get('dc_date'):
            content_parts.append(f"Date: {item_data['dc_date']}")
        elif item_data.get('pubDate'):
            content_parts.append(f"Published: {item_data['pubDate']}")
        
        # Subject/Topic (from Dublin Core)
        if item_data.get('dc_subject'):
            content_parts.append(f"Subject: {item_data['dc_subject']}")
        
        # Description
        if item_data.get('description'):
            content_parts.append(f"Description: {item_data['description']}")
        
        # Full content if available
        if item_data.get('content_encoded'):
            content_parts.append(f"Full Content: {item_data['content_encoded']}")
        
        # Link
        if item_data.get('link'):
            content_parts.append(f"Link: {item_data['link']}")
        
        # RDF identifier
        if item_data.get('about'):
            content_parts.append(f"RDF Resource: {item_data['about']}")
        
        # Additional Dublin Core metadata
        dc_extras = [
            ('dc_publisher', 'Publisher'),
            ('dc_rights', 'Rights'),
            ('dc_identifier', 'Identifier'),
            ('dc_source', 'Source')
        ]
        
        for dc_key, dc_label in dc_extras:
            if item_data.get(dc_key):
                content_parts.append(f"{dc_label}: {item_data[dc_key]}")
        
        return content_parts
    
    def _calculate_rss1_item_importance(self, item_data: Dict[str, Any], content: str) -> float:
        """Calculate importance score for RSS 1.0 item"""
        score = 0.75  # Base score (slightly higher for academic content)
        
        # Dublin Core metadata adds credibility
        dc_fields = ['dc_creator', 'dc_date', 'dc_subject', 'dc_publisher']
        dc_count = sum(1 for field in dc_fields if item_data.get(field))
        score += dc_count * 0.05  # Up to 0.2 for complete DC metadata
        
        # Content length
        if len(content) > 500:
            score += 0.1
        elif len(content) > 200:
            score += 0.05
        
        # Full content availability
        if item_data.get('content_encoded'):
            score += 0.1
        
        # RDF structure (proper about attribute)
        if item_data.get('about'):
            score += 0.05
        
        return min(score, 1.0)
    
    def _add_rss1_item_semantic_tags(self, chunk: SemanticChunk, item_data: Dict[str, Any]):
        """Add semantic tags to RSS 1.0 item chunks"""
        if not hasattr(chunk, 'add_tag'):
            return
        
        # Base tags
        chunk.add_tag('rss1_item', source='rss1_rdf_parser')
        chunk.add_tag('rdf_item', source='rss1_rdf_parser')
        chunk.add_tag('article', source='rss1_rdf_parser')
        chunk.add_tag('academic', source='rss1_rdf_parser')  # RSS 1.0 often used for academic content
        
        # Dublin Core tags
        if item_data.get('dc_creator'):
            chunk.add_tag('authored', source='rss1_rdf_parser')
            chunk.add_tag('dublin_core', source='rss1_rdf_parser')
        
        if item_data.get('dc_subject'):
            # Use DC subject as topic tag
            subject = item_data['dc_subject'].lower()
            chunk.add_tag(f"topic:{subject}", source='rss1_rdf_parser')
        
        # Academic/research tags based on content
        if item_data.get('dc_publisher'):
            chunk.add_tag('published', source='rss1_rdf_parser')
        
        if item_data.get('about') and 'doi' in item_data['about'].lower():
            chunk.add_tag('doi', source='rss1_rdf_parser')
            chunk.add_tag('research_paper', source='rss1_rdf_parser')
    
    # Utility methods
    
    def _get_rss1_element_text(self, parent, tag_name: str) -> Optional[str]:
        """Get text from RSS 1.0 element with namespace handling"""
        # Strategy 1: RSS 1.0 namespace
        elem = parent.find(f'{{http://purl.org/rss/1.0/}}{tag_name}')
        if elem is not None and elem.text:
            return elem.text.strip()
        
        # Strategy 2: No namespace
        elem = parent.find(tag_name)
        if elem is not None and elem.text:
            return elem.text.strip()
        
        # Strategy 3: Search all children by tag name
        for child in parent:
            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if child_tag.lower() == tag_name.lower() and child.text:
                return child.text.strip()
        
        return None
    
    def _extract_dublin_core_metadata(self, element) -> Dict[str, Any]:
        """Extract Dublin Core metadata from element"""
        dc_data = {}
        
        dc_elements = {
            'dc_creator': 'creator',
            'dc_date': 'date',
            'dc_subject': 'subject',
            'dc_publisher': 'publisher',
            'dc_rights': 'rights',
            'dc_identifier': 'identifier',
            'dc_source': 'source',
            'dc_language': 'language',
            'dc_type': 'type'
        }
        
        for key, dc_element in dc_elements.items():
            # Try full Dublin Core namespace
            elem = element.find(f'{{http://purl.org/dc/elements/1.1/}}{dc_element}')
            if elem is not None and elem.text:
                dc_data[key] = elem.text.strip()
                continue
            
            # Try dc: prefix
            elem = element.find(f'dc:{dc_element}')
            if elem is not None and elem.text:
                dc_data[key] = elem.text.strip()
                continue
            
            # Search by tag name
            for child in element:
                child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if child_tag.lower() == dc_element.lower() and child.text:
                    dc_data[key] = child.text.strip()
                    break
        
        return dc_data
    
    def _extract_syndication_metadata(self, element) -> Dict[str, Any]:
        """Extract RSS 1.0 syndication module metadata"""
        sy_data = {}
        
        sy_elements = {
            'updatePeriod': 'updatePeriod',
            'updateFrequency': 'updateFrequency',
            'updateBase': 'updateBase'
        }
        
        for key, sy_element in sy_elements.items():
            # Try syndication namespace
            elem = element.find(f'{{http://purl.org/rss/1.0/modules/syndication/}}{sy_element}')
            if elem is not None and elem.text:
                sy_data[key] = elem.text.strip()
                continue
            
            # Try sy: prefix
            elem = element.find(f'sy:{sy_element}')
            if elem is not None and elem.text:
                sy_data[key] = elem.text.strip()
        
        return sy_data
    
    def _create_fallback_chunk(self, content: str, context: ParseContext, error_msg: str) -> List[SemanticChunk]:
        """Create fallback chunk when RSS 1.0 parsing fails"""
        chunk_id = create_chunk_id(
            context.file_path, 
            1, 
            ChunkType.TEXT_BLOCK, 
            "rss1_fallback"
        )
        
        # Basic content analysis
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "RSS 1.0 RDF Feed"
        
        fallback_content = f"RSS 1.0 (RDF) Feed: {title}\nSize: {len(content)} characters\nError: {error_msg}"
        
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
                'feed_type': 'rss1',
                'semantic_type': 'RSS 1.0 RDF Fallback',
                'extraction_method': 'fallback',
                'error': error_msg
            }
        )
        
        if hasattr(chunk, 'add_tag'):
            chunk.add_tag('rss1_fallback', source='rss1_rdf_parser')
            chunk.add_tag('rdf_fallback', source='rss1_rdf_parser')
            chunk.add_tag('parsing_error', source='rss1_rdf_parser')
        
        return [chunk]