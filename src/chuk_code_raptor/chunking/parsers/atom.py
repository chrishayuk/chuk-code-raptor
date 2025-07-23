# src/chuk_code_raptor/chunking/parsers/atom.py
"""
Atom 1.0 Parser - Dedicated Implementation - FIXED
==================================================

Specialized parser for Atom 1.0 syndication feeds with comprehensive support for:
- Atom 1.0 specification compliance
- Rich content handling (text, html, xhtml)
- Multiple link relationships
- Person constructs (authors, contributors)
- Category taxonomies
- Content negotiation
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


def parse_feed_date(date_string: str) -> Optional[datetime]:
    """Parse feed dates without external dependencies"""
    if not date_string:
        return None
    
    try:
        # Handle ISO format (Atom): "2025-01-15T09:15:00Z" or "2025-01-15T09:15:00+00:00"
        if 'T' in date_string:
            # Remove timezone info for simple parsing
            clean_date = date_string.replace('Z', '').split('+')[0].split('-', 3)
            if len(clean_date) >= 3:
                date_part = clean_date[0] + '-' + clean_date[1] + '-' + clean_date[2]
                if 'T' in date_part:
                    return datetime.fromisoformat(date_part.replace('T', ' '))
        
        # Handle RFC 2822 format (RSS): "Mon, 15 Jan 2025 10:00:00 GMT"
        if ',' in date_string and any(month in date_string for month in 
                                     ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
            # Simple RFC 2822 parsing - just extract year, month, day
            parts = date_string.split()
            if len(parts) >= 4:
                try:
                    day = int(parts[1])
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    month = month_names.index(parts[2]) + 1
                    year = int(parts[3])
                    return datetime(year, month, day)
                except (ValueError, IndexError):
                    pass
        
        return None
    except:
        return None


class AtomParser(TreeSitterParser):
    """
    Dedicated Atom 1.0 parser with full specification support.
    
    Features:
    - Atom 1.0 specification compliance
    - Rich content type handling (text, html, xhtml)
    - Multiple link relationships (alternate, related, self, etc.)
    - Person constructs with names, emails, URIs
    - Category schemes and taxonomies
    - Proper date/time parsing (dependency-free)
    - Content negotiation support
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.supported_languages = {'atom'}
        self.supported_extensions = {'.atom', '.xml'}
        self.name = "AtomParser"
        
        # Atom specific configuration
        self.extract_full_content = getattr(config, 'atom_extract_full_content', True)
        self.clean_html_content = getattr(config, 'atom_clean_html_content', True)
        self.extract_all_links = getattr(config, 'atom_extract_all_links', True)
        self.extract_categories = getattr(config, 'atom_extract_categories', True)
        self.max_entries_per_feed = getattr(config, 'atom_max_entries_per_feed', 200)
        self.min_entry_content_length = getattr(config, 'atom_min_entry_content_length', 25)
        self.include_feed_metadata = getattr(config, 'atom_include_feed_metadata', True)
        self.preserve_xhtml_content = getattr(config, 'atom_preserve_xhtml_content', False)
        
        # Atom namespace
        self.atom_ns = 'http://www.w3.org/2005/Atom'
        self.namespaces = {
            'atom': self.atom_ns
        }
        
        # Register namespace
        ET.register_namespace('atom', self.atom_ns)
    
    def _get_tree_sitter_language(self):
        """Get tree-sitter XML language"""
        language, package_used = get_tree_sitter_language_robust(
            'xml', 
            ['tree_sitter_xml', 'tree_sitter_languages']
        )
        
        if language is None:
            raise ImportError("tree-sitter XML package required for Atom parsing")
        
        self._package_used = package_used
        return language
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """XML node types for Atom"""
        return {
            'element': ChunkType.TEXT_BLOCK,
            'start_tag': ChunkType.TEXT_BLOCK,
            'comment': ChunkType.COMMENT,
            'text': ChunkType.TEXT_BLOCK,
            'cdata_section': ChunkType.TEXT_BLOCK
        }
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from Atom node - not applicable for Atom XML"""
        return None
    
    def parse(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse Atom 1.0 feed content"""
        if not content.strip():
            return []
        
        # Validate Atom format
        if not self._is_atom_feed(content):
            logger.warning(f"Content does not appear to be Atom 1.0: {context.file_path}")
            return []
        
        try:
            # Parse with ElementTree
            root = ET.fromstring(content.strip())
            
            # Verify Atom structure
            if not self._is_atom_root(root):
                logger.warning(f"Root element is not Atom feed: {root.tag}")
                return []
            
            logger.info(f"Processing Atom 1.0 feed with root: {root.tag}")
            
            chunks = []
            
            # Extract feed metadata
            if self.include_feed_metadata:
                metadata_chunk = self._create_feed_metadata_chunk(root, context)
                if metadata_chunk:
                    chunks.append(metadata_chunk)
            
            # Find entries
            entries = root.findall(f'{{{self.atom_ns}}}entry')
            logger.info(f"Found {len(entries)} Atom entries")
            
            # Limit entries if configured
            if len(entries) > self.max_entries_per_feed:
                logger.info(f"Limiting to {self.max_entries_per_feed} entries")
                entries = entries[:self.max_entries_per_feed]
            
            # Process each entry
            for i, entry in enumerate(entries, 1):
                entry_chunk = self._parse_atom_entry(entry, context, i)
                if entry_chunk:
                    chunks.append(entry_chunk)
            
            logger.info(f"Created {len(chunks)} chunks from Atom feed")
            return chunks
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error in Atom feed {context.file_path}: {e}")
            return self._create_fallback_chunk(content, context, f"XML Parse Error: {e}")
        
        except Exception as e:
            logger.error(f"Error parsing Atom feed {context.file_path}: {e}")
            return self._create_fallback_chunk(content, context, f"Parse Error: {e}")
    
    def _is_atom_feed(self, content: str) -> bool:
        """Check if content is Atom 1.0 format"""
        content_lower = content.lower()
        
        # Atom 1.0 indicators
        atom_indicators = [
            'xmlns="http://www.w3.org/2005/atom"',
            '<feed xmlns="http://www.w3.org/2005/atom"',
            'atom/1.0',
            '2005/atom'
        ]
        
        return any(indicator in content_lower for indicator in atom_indicators)
    
    def _is_atom_root(self, root) -> bool:
        """Check if root element is Atom feed"""
        return root.tag == f'{{{self.atom_ns}}}feed' or root.tag == 'feed'
    
    def _create_feed_metadata_chunk(self, feed_root, context: ParseContext) -> Optional[SemanticChunk]:
        """Create feed metadata chunk from Atom feed element"""
        try:
            # Extract feed metadata
            metadata = self._extract_atom_feed_metadata(feed_root)
            
            # Create readable summary
            summary_parts = ["=== ATOM 1.0 FEED METADATA ==="]
            
            # Basic information
            if metadata.get('title'):
                title_text = metadata['title'].get('text', '') if isinstance(metadata['title'], dict) else metadata['title']
                summary_parts.append(f"Title: {title_text}")
            
            if metadata.get('subtitle'):
                subtitle_text = metadata['subtitle'].get('text', '') if isinstance(metadata['subtitle'], dict) else metadata['subtitle']
                summary_parts.append(f"Subtitle: {subtitle_text}")
            
            # Authors
            authors = metadata.get('authors', [])
            if authors:
                author_names = []
                for author in authors[:3]:  # Limit to first 3
                    name = author.get('name', 'Unknown')
                    if author.get('email'):
                        name += f" ({author['email']})"
                    author_names.append(name)
                summary_parts.append(f"Authors: {', '.join(author_names)}")
            
            # Links
            links = metadata.get('links', [])
            for link in links:
                if link.get('rel') == 'alternate' and link.get('type') == 'text/html':
                    summary_parts.append(f"Website: {link.get('href', '')}")
                    break
            
            # Dates
            if metadata.get('updated'):
                summary_parts.append(f"Last Updated: {metadata['updated']}")
            
            # ID
            if metadata.get('id'):
                summary_parts.append(f"Feed ID: {metadata['id']}")
            
            # Language
            if metadata.get('lang'):
                summary_parts.append(f"Language: {metadata['lang']}")
            
            # Rights
            if metadata.get('rights'):
                rights_text = metadata['rights'].get('text', '') if isinstance(metadata['rights'], dict) else metadata['rights']
                summary_parts.append(f"Rights: {rights_text}")
            
            # Categories
            categories = metadata.get('categories', [])
            if categories:
                cat_terms = [cat.get('term', '') for cat in categories[:5]]
                cat_terms = [term for term in cat_terms if term]
                if cat_terms:
                    summary_parts.append(f"Categories: {', '.join(cat_terms)}")
            
            # Generator
            if metadata.get('generator'):
                gen = metadata['generator']
                gen_text = gen.get('text', '') if isinstance(gen, dict) else gen
                summary_parts.append(f"Generator: {gen_text}")
            
            summary = '\n'.join(summary_parts)
            
            chunk_id = create_chunk_id(
                context.file_path, 
                1, 
                ChunkType.METADATA, 
                "atom_feed_metadata"
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
                    'feed_type': 'atom',
                    'semantic_type': 'Atom 1.0 Feed Metadata',
                    'feed_metadata': metadata,
                    'extraction_method': 'atom_parser'
                }
            )
            
            # Add semantic tags
            if hasattr(chunk, 'add_tag'):
                chunk.add_tag('atom_feed', source='atom_parser')
                chunk.add_tag('feed_metadata', source='atom_parser')
                chunk.add_tag('syndication', source='atom_parser')
                
                # Add category tags
                for category in categories[:10]:  # Limit tags
                    term = category.get('term', '')
                    if term:
                        chunk.add_tag(f"category:{term.lower()}", source='atom_parser')
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating Atom metadata chunk: {e}")
            return None
    
    def _extract_atom_feed_metadata(self, feed_root) -> Dict[str, Any]:
        """Extract metadata from Atom feed element"""
        metadata = {'format': 'atom'}
        
        # Basic Atom feed elements
        text_constructs = ['title', 'subtitle', 'rights', 'summary']
        for element_name in text_constructs:
            text_construct = self._extract_text_construct(feed_root, element_name)
            if text_construct:
                metadata[element_name] = text_construct
        
        # Simple text elements
        simple_elements = ['id', 'updated', 'published']
        for element_name in simple_elements:
            value = self._get_atom_element_text(feed_root, element_name)
            if value:
                metadata[element_name] = value
        
        # Language
        lang = feed_root.get('{http://www.w3.org/XML/1998/namespace}lang')
        if lang:
            metadata['lang'] = lang
        
        # Links
        if self.extract_all_links:
            metadata['links'] = self._extract_atom_links(feed_root)
        
        # Authors and contributors
        metadata['authors'] = self._extract_atom_persons(feed_root, 'author')
        metadata['contributors'] = self._extract_atom_persons(feed_root, 'contributor')
        
        # Categories
        if self.extract_categories:
            metadata['categories'] = self._extract_atom_categories(feed_root)
        
        # Generator
        generator = self._extract_generator(feed_root)
        if generator:
            metadata['generator'] = generator
        
        # Icon and logo
        icon = self._get_atom_element_text(feed_root, 'icon')
        if icon:
            metadata['icon'] = icon
        
        logo = self._get_atom_element_text(feed_root, 'logo')
        if logo:
            metadata['logo'] = logo
        
        return metadata
    
    def _parse_atom_entry(self, entry, context: ParseContext, entry_num: int) -> Optional[SemanticChunk]:
        """Parse individual Atom entry"""
        try:
            # Extract entry data
            entry_data = self._extract_atom_entry_data(entry)
            
            # Create content summary
            content_parts = self._build_atom_entry_content(entry_data)
            content = '\n\n'.join(content_parts)
            
            # Skip if content too short
            if len(content) < self.min_entry_content_length:
                logger.debug(f"Skipping Atom entry {entry_num}: content too short")
                return None
            
            # Calculate line numbers (approximate)
            start_line = entry_num * 25  # Atom entries tend to be longer
            end_line = start_line + len(content.split('\n'))
            
            chunk_id = create_chunk_id(
                context.file_path,
                start_line,
                ChunkType.TEXT_BLOCK,
                f"atom_entry_{entry_num}"
            )
            
            # Calculate importance score
            importance = self._calculate_atom_entry_importance(entry_data, content)
            
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
                    'feed_type': 'atom',
                    'semantic_type': 'Atom 1.0 Entry',
                    'entry_number': entry_num,
                    'entry_data': entry_data,
                    'extraction_method': 'atom_parser'
                }
            )
            
            # Add semantic tags
            self._add_atom_entry_semantic_tags(chunk, entry_data)
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error parsing Atom entry {entry_num}: {e}")
            return None
    
    def _extract_atom_entry_data(self, entry) -> Dict[str, Any]:
        """Extract data from Atom entry element"""
        entry_data = {}
        
        # Text constructs
        text_constructs = ['title', 'summary', 'rights']
        for element_name in text_constructs:
            text_construct = self._extract_text_construct(entry, element_name)
            if text_construct:
                entry_data[element_name] = text_construct
        
        # Content (special handling for different types)
        content = self._extract_atom_content(entry)
        if content:
            entry_data['content'] = content
        
        # Simple text elements
        simple_elements = ['id', 'updated', 'published']
        for element_name in simple_elements:
            value = self._get_atom_element_text(entry, element_name)
            if value:
                entry_data[element_name] = value
        
        # Links
        if self.extract_all_links:
            entry_data['links'] = self._extract_atom_links(entry)
        
        # Authors and contributors
        entry_data['authors'] = self._extract_atom_persons(entry, 'author')
        entry_data['contributors'] = self._extract_atom_persons(entry, 'contributor')
        
        # Categories
        if self.extract_categories:
            entry_data['categories'] = self._extract_atom_categories(entry)
        
        # Source (if entry is from another feed)
        source = self._extract_source(entry)
        if source:
            entry_data['source'] = source
        
        return entry_data
    
    def _build_atom_entry_content(self, entry_data: Dict[str, Any]) -> List[str]:
        """Build readable content from Atom entry data"""
        content_parts = []
        
        # Title
        title = entry_data.get('title', {})
        if title:
            title_text = title.get('text', '') if isinstance(title, dict) else title
            if title_text:
                content_parts.append(f"Title: {title_text}")
        
        # Authors
        authors = entry_data.get('authors', [])
        if authors:
            author_names = []
            for author in authors:
                name = author.get('name', 'Unknown')
                if author.get('email'):
                    name += f" <{author['email']}>"
                author_names.append(name)
            content_parts.append(f"Authors: {', '.join(author_names)}")
        
        # Publication dates
        if entry_data.get('published'):
            content_parts.append(f"Published: {entry_data['published']}")
        
        if entry_data.get('updated'):
            content_parts.append(f"Updated: {entry_data['updated']}")
        
        # Categories
        categories = entry_data.get('categories', [])
        if categories:
            cat_terms = []
            for cat in categories:
                term = cat.get('term', '')
                label = cat.get('label', '')
                if label and label != term:
                    cat_terms.append(f"{term} ({label})")
                elif term:
                    cat_terms.append(term)
            if cat_terms:
                content_parts.append(f"Categories: {', '.join(cat_terms)}")
        
        # Summary
        summary = entry_data.get('summary', {})
        if summary:
            summary_text = summary.get('text', '') if isinstance(summary, dict) else summary
            if summary_text:
                content_parts.append(f"Summary: {summary_text}")
        
        # Full content
        content = entry_data.get('content', {})
        if content and self.extract_full_content:
            content_text = content.get('text', '')
            content_type = content.get('type', 'text')
            
            if content_text:
                if content_type in ['html', 'xhtml'] and self.clean_html_content:
                    content_text = self._clean_html_content(content_text)
                
                content_parts.append(f"Content: {content_text}")
        
        # Primary link
        links = entry_data.get('links', [])
        for link in links:
            if link.get('rel') == 'alternate' and link.get('type') == 'text/html':
                content_parts.append(f"Link: {link.get('href', '')}")
                break
        
        # Entry ID
        if entry_data.get('id'):
            content_parts.append(f"Entry ID: {entry_data['id']}")
        
        # Source information
        source = entry_data.get('source', {})
        if source:
            source_title = source.get('title', '')
            source_uri = source.get('uri', '')
            if source_title:
                source_info = source_title
                if source_uri:
                    source_info += f" ({source_uri})"
                content_parts.append(f"Source: {source_info}")
        
        return content_parts
    
    def _calculate_atom_entry_importance(self, entry_data: Dict[str, Any], content: str) -> float:
        """Calculate importance score for Atom entry"""
        score = 0.75  # Base score
        
        # Content richness
        if entry_data.get('content'):
            content_obj = entry_data['content']
            if isinstance(content_obj, dict):
                content_text = content_obj.get('text', '')
                if len(content_text) > 500:
                    score += 0.15
                elif len(content_text) > 200:
                    score += 0.1
        
        # Overall content length
        if len(content) > 1000:
            score += 0.05
        
        # Metadata completeness
        if entry_data.get('authors'):
            score += 0.05
        
        if entry_data.get('categories'):
            score += 0.05
        
        # Recent content - using dependency-free date parsing
        if entry_data.get('published'):
            try:
                pub_date = parse_feed_date(entry_data['published'])
                if pub_date:
                    days_old = (datetime.now() - pub_date).days
                    if days_old < 7:
                        score += 0.05
                    elif days_old < 30:
                        score += 0.02
            except:
                pass
        
        # Multiple links (rich content)
        links = entry_data.get('links', [])
        if len(links) > 2:
            score += 0.03
        
        return min(score, 1.0)
    
    def _add_atom_entry_semantic_tags(self, chunk: SemanticChunk, entry_data: Dict[str, Any]):
        """Add semantic tags to Atom entry chunks"""
        if not hasattr(chunk, 'add_tag'):
            return
        
        # Base tags
        chunk.add_tag('atom_entry', source='atom_parser')
        chunk.add_tag('entry', source='atom_parser')
        chunk.add_tag('article', source='atom_parser')
        
        # Category tags
        categories = entry_data.get('categories', [])
        for category in categories:
            term = category.get('term', '')
            if term:
                chunk.add_tag(f"topic:{term.lower()}", source='atom_parser')
        
        # Content type tags
        content = entry_data.get('content', {})
        if content:
            content_type = content.get('type', 'text')
            chunk.add_tag(f"content_type:{content_type}", source='atom_parser')
            
            if content_type in ['html', 'xhtml']:
                chunk.add_tag('rich_content', source='atom_parser')
        
        # Author tags
        authors = entry_data.get('authors', [])
        if authors:
            chunk.add_tag('authored', source='atom_parser')
            for author in authors[:3]:  # Limit author tags
                name = author.get('name', '')
                if name:
                    # Clean name for tag
                    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', name).strip().lower()
                    if clean_name:
                        chunk.add_tag(f"author:{clean_name.replace(' ', '_')}", source='atom_parser')
        
        # Link relationship tags
        links = entry_data.get('links', [])
        for link in links:
            rel = link.get('rel', 'alternate')
            if rel in ['related', 'via', 'replies']:
                chunk.add_tag(f"link:{rel}", source='atom_parser')
    
    # Utility methods for Atom-specific constructs
    
    def _get_atom_element_text(self, parent, element_name: str) -> Optional[str]:
        """Get text from Atom element"""
        elem = parent.find(f'{{{self.atom_ns}}}{element_name}')
        if elem is not None and elem.text:
            return elem.text.strip()
        return None
    
    def _extract_text_construct(self, parent, element_name: str) -> Optional[Dict[str, Any]]:
        """Extract Atom text construct (text, html, or xhtml)"""
        elem = parent.find(f'{{{self.atom_ns}}}{element_name}')
        if elem is None:
            return None
        
        text_construct = {
            'type': elem.get('type', 'text'),
            'text': ''
        }
        
        if text_construct['type'] == 'xhtml':
            # XHTML content - preserve structure if configured
            if self.preserve_xhtml_content:
                try:
                    text_construct['text'] = ET.tostring(elem, encoding='unicode', method='html')
                except:
                    text_construct['text'] = elem.text or ''
            else:
                # Extract text content only
                text_construct['text'] = ''.join(elem.itertext())
        else:
            # Text or HTML content
            text_construct['text'] = elem.text or ''
        
        text_construct['text'] = text_construct['text'].strip()
        return text_construct if text_construct['text'] else None
    
    def _extract_atom_content(self, entry) -> Optional[Dict[str, Any]]:
        """Extract Atom content element"""
        content_elem = entry.find(f'{{{self.atom_ns}}}content')
        if content_elem is None:
            return None
        
        content = {
            'type': content_elem.get('type', 'text'),
            'text': ''
        }
        
        # Handle different content types
        if content['type'] == 'xhtml':
            if self.preserve_xhtml_content:
                try:
                    content['text'] = ET.tostring(content_elem, encoding='unicode', method='html')
                except:
                    content['text'] = ''.join(content_elem.itertext())
            else:
                content['text'] = ''.join(content_elem.itertext())
        elif content['type'] in ['text', 'html']:
            content['text'] = content_elem.text or ''
        else:
            # Other content types (e.g., application/xml)
            content['src'] = content_elem.get('src')
            if content['src']:
                content['text'] = f"External content: {content['src']}"
            else:
                content['text'] = content_elem.text or ''
        
        content['text'] = content['text'].strip()
        return content if content['text'] else None
    
    def _extract_atom_links(self, element) -> List[Dict[str, Any]]:
        """Extract Atom link elements"""
        links = []
        
        for link_elem in element.findall(f'{{{self.atom_ns}}}link'):
            link = {
                'href': link_elem.get('href'),
                'rel': link_elem.get('rel', 'alternate'),
                'type': link_elem.get('type'),
                'hreflang': link_elem.get('hreflang'),
                'title': link_elem.get('title'),
                'length': link_elem.get('length')
            }
            
            # Only include links with href
            if link['href']:
                links.append(link)
        
        return links
    
    def _extract_atom_persons(self, element, person_type: str) -> List[Dict[str, Any]]:
        """Extract Atom person constructs (author, contributor)"""
        persons = []
        
        for person_elem in element.findall(f'{{{self.atom_ns}}}{person_type}'):
            person = {
                'name': self._get_atom_element_text(person_elem, 'name'),
                'email': self._get_atom_element_text(person_elem, 'email'),
                'uri': self._get_atom_element_text(person_elem, 'uri')
            }
            
            # Only include persons with at least a name
            if person['name']:
                persons.append(person)
        
        return persons
    
    def _extract_atom_categories(self, element) -> List[Dict[str, Any]]:
        """Extract Atom category elements"""
        categories = []
        
        for cat_elem in element.findall(f'{{{self.atom_ns}}}category'):
            category = {
                'term': cat_elem.get('term'),
                'scheme': cat_elem.get('scheme'),
                'label': cat_elem.get('label')
            }
            
            # Only include categories with term
            if category['term']:
                categories.append(category)
        
        return categories
    
    def _extract_generator(self, feed_root) -> Optional[Dict[str, Any]]:
        """Extract Atom generator element"""
        gen_elem = feed_root.find(f'{{{self.atom_ns}}}generator')
        if gen_elem is None:
            return None
        
        generator = {
            'text': gen_elem.text or '',
            'uri': gen_elem.get('uri'),
            'version': gen_elem.get('version')
        }
        
        return generator if generator['text'] else None
    
    def _extract_source(self, entry) -> Optional[Dict[str, Any]]:
        """Extract Atom source element"""
        source_elem = entry.find(f'{{{self.atom_ns}}}source')
        if source_elem is None:
            return None
        
        source = {
            'uri': source_elem.get('uri'),
            'title': source_elem.text or ''
        }
        
        return source if source['uri'] or source['title'] else None
    
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
        if len(clean_text) > 2500:
            clean_text = clean_text[:2497] + "..."
        
        return clean_text
    
    def _create_fallback_chunk(self, content: str, context: ParseContext, error_msg: str) -> List[SemanticChunk]:
        """Create fallback chunk when Atom parsing fails"""
        chunk_id = create_chunk_id(
            context.file_path, 
            1, 
            ChunkType.TEXT_BLOCK, 
            "atom_fallback"
        )
        
        # Basic content analysis
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "Atom 1.0 Feed"
        
        fallback_content = f"Atom 1.0 Feed: {title}\nSize: {len(content)} characters\nError: {error_msg}"
        
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
                'feed_type': 'atom',
                'semantic_type': 'Atom 1.0 Fallback',
                'extraction_method': 'fallback',
                'error': error_msg
            }
        )
        
        if hasattr(chunk, 'add_tag'):
            chunk.add_tag('atom_fallback', source='atom_parser')
            chunk.add_tag('parsing_error', source='atom_parser')
        
        return [chunk]