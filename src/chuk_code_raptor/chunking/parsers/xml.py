# src/chuk_code_raptor/chunking/parsers/xml.py
"""
XML Parser
==========

XML parser using tree-sitter-xml with heuristic fallback for semantic chunking.
Follows the common TreeSitterParser architecture with enhanced XML-specific analysis.
"""

import logging
import re
from typing import List, Dict, Optional, Set, Tuple
from xml.sax.saxutils import escape, unescape
from collections import defaultdict

from ..tree_sitter_base import TreeSitterParser
from ..heuristic_base import HeuristicParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from ..base import ParseContext
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class XMLParser(TreeSitterParser):
    """XML parser using tree-sitter-xml with comprehensive semantic analysis"""
    
    def __init__(self, config):
        # Call parent first
        super().__init__(config)
        
        # Set our specific supported languages and extensions
        self.supported_languages = {'xml'}
        self.supported_extensions = {'.xml', '.xhtml', '.svg', '.rss', '.atom', '.sitemap', '.xsd', '.wsdl', '.pom'}
        
        # XML-specific configuration
        self.preserve_atomic_elements = getattr(config, 'xml_preserve_atomic_elements', True)
        self.group_similar_elements = getattr(config, 'xml_group_similar_elements', True)
        self.extract_cdata = getattr(config, 'xml_extract_cdata', True)
        self.namespace_aware = getattr(config, 'xml_namespace_aware', True)
        
        # XML semantic patterns for heuristic fallback
        self.xml_patterns = {
            'element_start': r'<([^/\s>]+)(?:\s[^>]*)?>',
            'element_end': r'</([^>]+)>',
            'self_closing': r'<([^/\s>]+)(?:\s[^>]*)?/>',
            'cdata': r'<!\[CDATA\[(.*?)\]\]>',
            'comment': r'<!--(.*?)-->',
            'processing_instruction': r'<\?([^>]*)\?>',
            'doctype': r'<!DOCTYPE[^>]*>',
            'namespace': r'xmlns(?::([^=]+))?=["\']([^"\']*)["\']',
            'attribute': r'(\w+)=["\']([^"\']*)["\']',
        }
        
        # Semantic element categories
        self.semantic_categories = {
            'structural': {
                'html', 'head', 'body', 'header', 'footer', 'nav', 'main', 'article', 'section', 'aside',
                'div', 'span', 'container', 'wrapper', 'content', 'layout'
            },
            'content': {
                'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'pre', 'code',
                'text', 'content', 'description', 'summary', 'abstract'
            },
            'data': {
                'table', 'tr', 'td', 'th', 'thead', 'tbody', 'tfoot',
                'ul', 'ol', 'li', 'dl', 'dt', 'dd',
                'data', 'item', 'entry', 'record', 'row', 'field'
            },
            'metadata': {
                'meta', 'title', 'link', 'style', 'script',
                'property', 'attribute', 'config', 'setting'
            },
            'media': {
                'img', 'video', 'audio', 'source', 'track',
                'figure', 'figcaption', 'picture', 'canvas', 'svg'
            },
            'form': {
                'form', 'input', 'textarea', 'select', 'option', 'button',
                'fieldset', 'legend', 'label', 'output'
            },
            'custom': set()  # Will be populated dynamically
        }
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _get_tree_sitter_language(self):
        """Get XML tree-sitter language with robust package handling"""
        from ..tree_sitter_base import get_tree_sitter_language_robust
        
        # Try different package sources for XML with improved compatibility
        language, package_used = get_tree_sitter_language_robust(
            'xml', 
            ['tree_sitter_languages', 'tree_sitter_language_pack', 'tree_sitter_xml']
        )
        
        if language is None:
            # Try direct import with different API patterns
            try:
                import tree_sitter_xml
                
                # Try different ways to get the language
                if hasattr(tree_sitter_xml, 'language'):
                    # New API style
                    language = tree_sitter_xml.language()
                    package_used = 'tree_sitter_xml'
                elif hasattr(tree_sitter_xml, 'LANGUAGE'):
                    # Some packages use LANGUAGE constant
                    import tree_sitter
                    language = tree_sitter.Language(tree_sitter_xml.LANGUAGE)
                    package_used = 'tree_sitter_xml'
                else:
                    # Try to find language function/constant
                    for attr_name in dir(tree_sitter_xml):
                        if 'language' in attr_name.lower() and callable(getattr(tree_sitter_xml, attr_name)):
                            language = getattr(tree_sitter_xml, attr_name)()
                            package_used = 'tree_sitter_xml'
                            break
            except ImportError:
                pass
        
        if language is None:
            raise ImportError(
                "No compatible tree-sitter XML package found. "
                "Try: pip install tree-sitter-languages or pip install tree-sitter-xml"
            )
        
        self._package_used = package_used
        return language
    
    def _get_chunk_node_types(self) -> Dict[str, ChunkType]:
        """XML AST node types to chunk types mapping"""
        return {
            'element': ChunkType.TEXT_BLOCK,
            'start_tag': ChunkType.TEXT_BLOCK,
            'end_tag': ChunkType.TEXT_BLOCK,
            'self_closing_tag': ChunkType.TEXT_BLOCK,
            'comment': ChunkType.COMMENT,
            'processing_instruction': ChunkType.TEXT_BLOCK,
            'cdata_section': ChunkType.TEXT_BLOCK,
            'text': ChunkType.TEXT_BLOCK,
        }
    
    def parse(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Enhanced parse method with automatic fallback to heuristic parsing"""
        if not content.strip():
            return []
        
        # First attempt: tree-sitter parsing with enhanced error handling
        tree_sitter_chunks = []
        tree_sitter_error = None
        
        try:
            if self.parser is not None and self.language is not None:
                tree_sitter_chunks = self._parse_with_tree_sitter(content, context)
                logger.debug(f"Tree-sitter XML produced {len(tree_sitter_chunks)} chunks")
        except Exception as e:
            tree_sitter_error = str(e)
            logger.warning(f"Tree-sitter XML parsing failed: {e}")
        
        # Check if tree-sitter parsing was sufficient
        if self._is_parsing_successful(tree_sitter_chunks, content):
            logger.info(f"Using tree-sitter results for XML parsing: {len(tree_sitter_chunks)} chunks")
            try:
                return self._post_process(tree_sitter_chunks)
            except Exception as e:
                logger.warning(f"Tree-sitter post-processing failed: {e}, falling back to heuristic")
        
        # Fallback: heuristic XML parsing
        logger.info("Tree-sitter insufficient for XML, using heuristic XML analysis")
        try:
            heuristic_chunks = self._parse_xml_heuristically(content, context)
            
            if heuristic_chunks and len(heuristic_chunks) > len(tree_sitter_chunks):
                logger.info(f"Heuristic XML analysis produced {len(heuristic_chunks)} chunks")
                return self._post_process(heuristic_chunks)
        except Exception as e:
            logger.error(f"Heuristic XML parsing also failed: {e}")
            
            # Create a basic fallback chunk
            if content.strip():
                chunk_id = create_chunk_id(context.file_path, 1, ChunkType.TEXT_BLOCK, "xml_fallback")
                
                fallback_chunk = SemanticChunk(
                    id=chunk_id,
                    file_path=context.file_path,
                    content=content,
                    start_line=1,
                    end_line=content.count('\n') + 1,
                    content_type=context.content_type,
                    chunk_type=ChunkType.TEXT_BLOCK,
                    language=context.language,
                    importance_score=0.5,
                    metadata={
                        'parser': self.name,
                        'parser_type': 'xml_fallback',
                        'extraction_method': 'fallback',
                        'tree_sitter_error': tree_sitter_error,
                        'heuristic_error': str(e)
                    }
                )
                
                fallback_chunk.add_tag('xml_fallback', source='error_recovery')
                fallback_chunk.add_tag('parsing_error', source='error_recovery')
                
                return [fallback_chunk]
        
        # Final fallback: return tree-sitter result even if poor
        logger.warning("All parsing methods had issues, using best available result")
        return self._post_process(tree_sitter_chunks) if tree_sitter_chunks else []
    
    def _is_parsing_successful(self, chunks: List[SemanticChunk], content: str) -> bool:
        """Determine if tree-sitter parsing was successful"""
        if not chunks:
            logger.debug("No chunks created - parsing failed")
            return False
        
        # If we only got one chunk that's nearly the entire document, it's likely failed
        if len(chunks) == 1:
            chunk = chunks[0]
            content_ratio = len(chunk.content) / len(content)
            if content_ratio > 0.90:
                logger.debug(f"Single chunk contains {content_ratio:.1%} of content - likely failed parsing")
                return False
        
        # Check for reasonable chunk distribution
        if chunks:
            avg_chunk_size = sum(len(chunk.content) for chunk in chunks) / len(chunks)
            if avg_chunk_size > self.config.target_chunk_size * 2:
                logger.debug(f"Average chunk size {avg_chunk_size} is too large")
                return False
        
        # If we have multiple reasonably-sized chunks, consider it successful
        if len(chunks) >= 3:
            return True
        
        return False
    
    def _parse_xml_heuristically(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse XML content using heuristic methods when tree-sitter fails"""
        chunks = []
        
        # Clean and normalize content
        normalized_content = self._normalize_xml_content(content)
        
        # Extract document structure
        doc_info = self._analyze_document_structure(normalized_content)
        
        # Extract chunks based on semantic structure
        if doc_info['has_root_element']:
            chunks = self._extract_semantic_elements(normalized_content, context, doc_info)
        else:
            # Fragment parsing
            chunks = self._extract_xml_fragments(normalized_content, context)
        
        return chunks
    
    def _normalize_xml_content(self, content: str) -> str:
        """Normalize XML content for better parsing"""
        # Remove excessive whitespace while preserving structure
        normalized = re.sub(r'\n\s*\n', '\n', content)
        
        # Normalize indentation
        lines = normalized.split('\n')
        normalized_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                normalized_lines.append(stripped)
            elif normalized_lines:  # Preserve one blank line
                normalized_lines.append('')
        
        return '\n'.join(normalized_lines)
    
    def _analyze_document_structure(self, content: str) -> Dict:
        """Analyze the overall structure of the XML document"""
        info = {
            'has_root_element': False,
            'root_element': None,
            'namespace_prefixes': set(),
            'element_counts': {},
            'max_depth': 0,
            'document_type': 'unknown'
        }
        
        # Find root element
        root_match = re.search(r'<([^/\s>?!]+)', content)
        if root_match:
            info['has_root_element'] = True
            info['root_element'] = root_match.group(1)
            
            # Determine document type from root element
            root_name = info['root_element'].lower()
            if root_name in ['html', 'xhtml']:
                info['document_type'] = 'html'
            elif root_name in ['svg']:
                info['document_type'] = 'svg'
            elif root_name in ['rss', 'feed']:
                info['document_type'] = 'feed'
            elif root_name in ['configuration', 'config']:
                info['document_type'] = 'config'
            elif root_name in ['project', 'pom']:
                info['document_type'] = 'build'
            else:
                info['document_type'] = 'data'
        
        # Extract namespaces
        namespaces = re.findall(self.xml_patterns['namespace'], content)
        for prefix, uri in namespaces:
            if prefix:
                info['namespace_prefixes'].add(prefix)
        
        # Count elements
        elements = re.findall(r'<([^/\s>?!]+)', content)
        for element in elements:
            # Remove namespace prefix for counting
            local_name = element.split(':')[-1]
            info['element_counts'][local_name] = info['element_counts'].get(local_name, 0) + 1
        
        return info
    
    def _extract_semantic_elements(self, content: str, context: ParseContext, doc_info: Dict) -> List[SemanticChunk]:
        """Extract elements based on semantic significance"""
        chunks = []
        processed_ranges = set()
        
        # Strategy 1: Extract significant top-level elements
        top_level_chunks = self._extract_top_level_elements(content, context, doc_info, processed_ranges)
        chunks.extend(top_level_chunks)
        
        # Strategy 2: Extract repeated/collection elements
        collection_chunks = self._extract_collection_elements(content, context, doc_info, processed_ranges)
        chunks.extend(collection_chunks)
        
        # Strategy 3: Extract content-heavy elements
        content_chunks = self._extract_content_elements(content, context, processed_ranges)
        chunks.extend(content_chunks)
        
        # Strategy 4: Extract remaining unprocessed content
        remaining_chunks = self._extract_remaining_xml_content(content, context, processed_ranges)
        chunks.extend(remaining_chunks)
        
        return chunks
    
    def _extract_top_level_elements(self, content: str, context: ParseContext, 
                                   doc_info: Dict, processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """Extract semantically significant top-level elements"""
        chunks = []
        
        # Find major structural elements
        significant_elements = self._find_significant_elements(content, doc_info)
        
        for element_info in significant_elements:
            start_line = element_info['start_line']
            end_line = element_info['end_line']
            
            # Skip if overlaps with processed content
            if self._overlaps_with_processed(start_line, end_line, processed_ranges):
                continue
            
            element_content = element_info['content']
            element_name = element_info['name']
            
            chunk = self._create_xml_chunk(
                content=element_content,
                start_line=start_line,
                end_line=end_line,
                context=context,
                element_name=element_name,
                xml_type='element',
                semantic_category=self._get_element_category(element_name)
            )
            
            if chunk:
                chunks.append(chunk)
                processed_ranges.add((start_line, end_line))
        
        return chunks
    
    def _find_significant_elements(self, content: str, doc_info: Dict) -> List[Dict]:
        """Find semantically significant elements in the document"""
        significant = []
        lines = content.split('\n')
        
        # Look for elements that are likely to be semantically meaningful
        element_stack = []
        current_element = None
        
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()
            
            # Find opening tags
            start_match = re.search(r'<([^/\s>?!]+)(?:\s[^>]*)?>(?!.*</\1>)', stripped)
            if start_match:
                element_name = start_match.group(1)
                
                # Start tracking if this is a significant element
                if self._is_significant_element(element_name, doc_info):
                    current_element = {
                        'name': element_name,
                        'start_line': line_num,
                        'start_index': i,
                        'content_lines': [line]
                    }
                    element_stack.append(current_element)
            
            # Find closing tags
            end_match = re.search(r'</([^>]+)>', stripped)
            if end_match and element_stack:
                closing_name = end_match.group(1)
                
                # Find matching opening tag
                for j in range(len(element_stack) - 1, -1, -1):
                    if element_stack[j]['name'] == closing_name:
                        element = element_stack.pop(j)
                        element['end_line'] = line_num
                        element['end_index'] = i
                        element['content_lines'].append(line)
                        element['content'] = '\n'.join(element['content_lines'])
                        
                        # Add to significant elements if it meets criteria
                        if self._meets_chunk_criteria(element):
                            significant.append(element)
                        break
            
            # Add line to all open elements
            for element in element_stack:
                element['content_lines'].append(line)
        
        return significant
    
    def _is_significant_element(self, element_name: str, doc_info: Dict) -> bool:
        """Determine if an element is semantically significant"""
        # Remove namespace prefix
        local_name = element_name.split(':')[-1].lower()
        
        # Always significant structural elements
        structural_elements = {
            'head', 'body', 'header', 'footer', 'nav', 'main', 'article', 'section',
            'chapter', 'part', 'div', 'container', 'content'
        }
        
        if local_name in structural_elements:
            return True
        
        # Content elements that might be significant
        content_elements = {
            'p', 'blockquote', 'pre', 'table', 'ul', 'ol', 'dl',
            'item', 'entry', 'record', 'data', 'text', 'description'
        }
        
        if local_name in content_elements:
            return True
        
        # Document-type specific significance
        doc_type = doc_info.get('document_type', 'unknown')
        
        if doc_type == 'feed' and local_name in {'item', 'entry', 'channel'}:
            return True
        elif doc_type == 'config' and local_name in {'property', 'setting', 'configuration'}:
            return True
        elif doc_type == 'build' and local_name in {'dependency', 'plugin', 'module'}:
            return True
        elif doc_type == 'svg' and local_name in {'g', 'path', 'rect', 'circle', 'text'}:
            return True
        
        # Check if it's a repeated element (likely part of a collection)
        element_count = doc_info.get('element_counts', {}).get(local_name, 0)
        if element_count > 1:
            return True
        
        return False
    
    def _meets_chunk_criteria(self, element: Dict) -> bool:
        """Check if an element meets the criteria to become a chunk"""
        content = element['content']
        
        # Size criteria
        if len(content) < self.config.min_chunk_size:
            # Allow smaller chunks for certain element types
            element_name = element['name'].split(':')[-1].lower()
            important_small_elements = {'title', 'meta', 'link', 'item', 'entry'}
            if element_name not in important_small_elements:
                return False
        
        if len(content) > self.config.max_chunk_size:
            # Allow large chunks for atomic elements if configured
            if not self.preserve_atomic_elements:
                return False
        
        # Content criteria - must have some meaningful content
        text_content = re.sub(r'<[^>]*>', '', content).strip()
        if len(text_content) < 10:  # Very minimal text content
            return False
        
        return True
    
    def _extract_collection_elements(self, content: str, context: ParseContext, 
                                   doc_info: Dict, processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """Extract repeated elements that form collections"""
        chunks = []
        
        # Find repeated elements
        element_counts = doc_info.get('element_counts', {})
        repeated_elements = {name: count for name, count in element_counts.items() if count > 2}
        
        for element_name, count in repeated_elements.items():
            # Extract instances of this repeated element
            element_chunks = self._extract_repeated_element_instances(
                content, element_name, context, processed_ranges
            )
            
            # Group similar elements if configured
            if self.group_similar_elements and len(element_chunks) > 3:
                grouped_chunks = self._group_similar_elements_into_chunks(element_chunks, context)
                chunks.extend(grouped_chunks)
            else:
                chunks.extend(element_chunks)
        
        return chunks
    
    def _extract_repeated_element_instances(self, content: str, element_name: str, 
                                          context: ParseContext, processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """Extract individual instances of a repeated element"""
        chunks = []
        lines = content.split('\n')
        
        # Find all instances of this element
        pattern = rf'<{re.escape(element_name)}(?:\s[^>]*)?>.*?</{re.escape(element_name)}>'
        matches = list(re.finditer(pattern, content, re.DOTALL))
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = match.end()
            
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            # Skip if overlaps with processed content
            if self._overlaps_with_processed(start_line, end_line, processed_ranges):
                continue
            
            element_content = match.group(0)
            
            # Only create chunk if it meets criteria
            if len(element_content.strip()) >= self.config.min_chunk_size // 2:  # More lenient for repeated elements
                chunk = self._create_xml_chunk(
                    content=element_content,
                    start_line=start_line,
                    end_line=end_line,
                    context=context,
                    element_name=element_name,
                    xml_type='repeated_element',
                    semantic_category=self._get_element_category(element_name),
                    instance_number=i + 1
                )
                
                if chunk:
                    chunks.append(chunk)
                    processed_ranges.add((start_line, end_line))
        
        return chunks
    
    def _group_similar_elements_into_chunks(self, element_chunks: List[SemanticChunk], 
                                          context: ParseContext) -> List[SemanticChunk]:
        """Group similar elements into consolidated chunks"""
        if not element_chunks:
            return []
        
        # Group chunks by element name
        grouped_by_element = defaultdict(list)
        for chunk in element_chunks:
            element_name = chunk.metadata.get('element_name', 'unknown')
            grouped_by_element[element_name].append(chunk)
        
        consolidated_chunks = []
        
        for element_name, chunks in grouped_by_element.items():
            if len(chunks) <= 2:
                # Don't group if there are only 1-2 instances
                consolidated_chunks.extend(chunks)
                continue
            
            # Calculate group size based on target chunk size
            group_size = max(2, min(5, self.config.target_chunk_size // 200))
            
            # Group chunks into batches
            for i in range(0, len(chunks), group_size):
                batch = chunks[i:i + group_size]
                
                if len(batch) == 1:
                    consolidated_chunks.append(batch[0])
                    continue
                
                # Merge batch into a single chunk
                merged_content = []
                merged_metadata = {
                    'parser': self.name,
                    'parser_type': 'xml_heuristic',
                    'element_name': element_name,
                    'xml_type': 'grouped_elements',
                    'semantic_category': batch[0].metadata.get('semantic_category', 'custom'),
                    'extraction_method': 'heuristic',
                    'grouped_count': len(batch),
                    'grouped_elements': [c.metadata.get('element_name') for c in batch]
                }
                
                total_importance = 0
                start_line = min(c.start_line for c in batch)
                end_line = max(c.end_line for c in batch)
                
                for chunk in batch:
                    merged_content.append(chunk.content)
                    total_importance += chunk.importance_score
                
                # Create consolidated chunk
                merged_chunk_content = '\n\n'.join(merged_content)
                
                chunk_id = create_chunk_id(
                    context.file_path,
                    start_line,
                    ChunkType.TEXT_BLOCK,
                    f"grouped_{element_name}_{start_line}"
                )
                
                merged_chunk = SemanticChunk(
                    id=chunk_id,
                    file_path=context.file_path,
                    content=merged_chunk_content,
                    start_line=start_line,
                    end_line=end_line,
                    content_type=context.content_type,
                    chunk_type=ChunkType.TEXT_BLOCK,
                    language=context.language,
                    importance_score=total_importance / len(batch),
                    metadata=merged_metadata
                )
                
                # Merge semantic tags from all chunks
                all_tags = set()
                for chunk in batch:
                    if hasattr(chunk, 'semantic_tags') and chunk.semantic_tags:
                        for tag in chunk.semantic_tags:
                            all_tags.add(tag.name)
                
                # Add consolidated tags
                for tag_name in all_tags:
                    merged_chunk.add_tag(tag_name, source='heuristic_group')
                
                merged_chunk.add_tag('grouped_elements', source='heuristic')
                merged_chunk.add_tag(f'grouped_{element_name}', source='heuristic')
                
                consolidated_chunks.append(merged_chunk)
        
        return consolidated_chunks
    
    def _extract_content_elements(self, content: str, context: ParseContext, 
                                processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """Extract elements with significant text content"""
        chunks = []
        
        # Look for elements with substantial text content
        content_pattern = r'<([^/\s>?!]+)(?:\s[^>]*)?>([^<]{20,})</\1>'
        matches = re.finditer(content_pattern, content, re.DOTALL)
        
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            # Skip if overlaps with processed content
            if self._overlaps_with_processed(start_line, end_line, processed_ranges):
                continue
            
            element_name = match.group(1)
            text_content = match.group(2).strip()
            full_content = match.group(0)
            
            # Only create chunk if text content is substantial
            if len(text_content) >= self.config.min_chunk_size:
                chunk = self._create_xml_chunk(
                    content=full_content,
                    start_line=start_line,
                    end_line=end_line,
                    context=context,
                    element_name=element_name,
                    xml_type='content_element',
                    semantic_category=self._get_element_category(element_name),
                    text_content_length=len(text_content)
                )
                
                if chunk:
                    chunks.append(chunk)
                    processed_ranges.add((start_line, end_line))
        
        return chunks
    
    def _extract_remaining_xml_content(self, content: str, context: ParseContext, 
                                     processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """Extract any remaining unprocessed XML content"""
        chunks = []
        lines = content.split('\n')
        
        current_start = 1
        for start_line, end_line in sorted(processed_ranges):
            # Check for gap before this processed range
            if current_start < start_line:
                gap_content = '\n'.join(lines[current_start-1:start_line-1]).strip()
                if len(gap_content) >= self.config.min_chunk_size:
                    chunk = self._create_xml_chunk(
                        content=gap_content,
                        start_line=current_start,
                        end_line=start_line - 1,
                        context=context,
                        element_name='mixed_content',
                        xml_type='remaining_content',
                        semantic_category='mixed'
                    )
                    if chunk:
                        chunks.append(chunk)
            
            current_start = end_line + 1
        
        # Handle content after last processed range
        if current_start <= len(lines):
            gap_content = '\n'.join(lines[current_start-1:]).strip()
            if len(gap_content) >= self.config.min_chunk_size:
                chunk = self._create_xml_chunk(
                    content=gap_content,
                    start_line=current_start,
                    end_line=len(lines),
                    context=context,
                    element_name='mixed_content',
                    xml_type='remaining_content',
                    semantic_category='mixed'
                )
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def _create_xml_chunk(self, content: str, start_line: int, end_line: int,
                         context: ParseContext, element_name: str, xml_type: str,
                         semantic_category: str, **additional_metadata) -> Optional[SemanticChunk]:
        """Create an XML-specific semantic chunk"""
        
        chunk_id = create_chunk_id(
            context.file_path,
            start_line,
            ChunkType.TEXT_BLOCK,
            f"{xml_type}_{element_name}_{start_line}"
        )
        
        chunk = SemanticChunk(
            id=chunk_id,
            file_path=context.file_path,
            content=content,
            start_line=start_line,
            end_line=end_line,
            content_type=context.content_type,
            chunk_type=ChunkType.TEXT_BLOCK,
            language=context.language,
            importance_score=self._calculate_xml_importance(element_name, xml_type, content),
            metadata={
                'parser': self.name,
                'parser_type': 'xml_heuristic',
                'element_name': element_name,
                'xml_type': xml_type,
                'semantic_category': semantic_category,
                'extraction_method': 'heuristic',
                **additional_metadata
            }
        )
        
        # Add XML-specific semantic tags
        self._add_xml_semantic_tags(chunk, element_name, xml_type, semantic_category)
        
        # Analyze XML content
        self._analyze_xml_content(chunk)
        
        return chunk
    
    def _calculate_xml_importance(self, element_name: str, xml_type: str, content: str) -> float:
        """Calculate importance score for XML chunks"""
        base_score = 0.5
        local_name = element_name.split(':')[-1].lower()
        
        # Structural elements are more important
        if xml_type in ['element', 'content_element']:
            base_score += 0.2
        
        # Important elements get higher scores
        important_elements = {
            'title', 'head', 'body', 'article', 'section', 'chapter',
            'item', 'entry', 'record', 'data', 'content'
        }
        if local_name in important_elements:
            base_score += 0.2
        
        # Content with more text is more important
        text_content = re.sub(r'<[^>]*>', '', content).strip()
        if len(text_content) > 200:
            base_score += 0.1
        
        return min(1.0, max(0.1, base_score))
    
    def _add_xml_semantic_tags(self, chunk: SemanticChunk, element_name: str, 
                              xml_type: str, semantic_category: str):
        """Add XML-specific semantic tags"""
        local_name = element_name.split(':')[-1].lower()
        
        # Basic tags
        chunk.add_tag('xml', source='heuristic')
        chunk.add_tag(f'xml_{xml_type}', source='heuristic')
        chunk.add_tag(f'element_{local_name}', source='heuristic')
        chunk.add_tag(f'category_{semantic_category}', source='heuristic')
        
        # Semantic category tags
        if semantic_category in self.semantic_categories:
            for category_element in self.semantic_categories[semantic_category]:
                if local_name == category_element:
                    chunk.add_tag(f'semantic_{semantic_category}', source='heuristic')
                    break
    
    def _analyze_xml_content(self, chunk: SemanticChunk):
        """Analyze XML chunk content for additional features"""
        content = chunk.content
        
        # Extract attributes
        attributes = re.findall(self.xml_patterns['attribute'], content)
        if attributes:
            chunk.metadata['attributes'] = dict(attributes)
            chunk.metadata['attribute_count'] = len(attributes)
            chunk.add_tag('has_attributes', source='heuristic')
        
        # Check for CDATA
        cdata_matches = re.findall(self.xml_patterns['cdata'], content, re.DOTALL)
        if cdata_matches:
            chunk.add_tag('has_cdata', source='heuristic')
            chunk.metadata['cdata_sections'] = len(cdata_matches)
        
        # Check for namespaces
        namespace_matches = re.findall(self.xml_patterns['namespace'], content)
        if namespace_matches:
            chunk.add_tag('has_namespaces', source='heuristic')
            namespaces = {}
            for prefix, uri in namespace_matches:
                namespaces[prefix or 'default'] = uri
            chunk.metadata['namespaces'] = namespaces
        
        # Analyze text content ratio
        text_content = re.sub(r'<[^>]*>', '', content).strip()
        if content:
            text_ratio = len(text_content) / len(content)
            chunk.metadata['text_content_ratio'] = text_ratio
            
            if text_ratio > 0.7:
                chunk.add_tag('text_heavy', source='heuristic')
            elif text_ratio < 0.3:
                chunk.add_tag('markup_heavy', source='heuristic')
        
        # Count nested elements
        nested_elements = re.findall(r'<([^/\s>?!]+)', content)
        if nested_elements:
            chunk.metadata['nested_element_count'] = len(nested_elements)
            unique_nested = set(elem.split(':')[-1].lower() for elem in nested_elements)
            chunk.metadata['unique_nested_elements'] = len(unique_nested)
    
    def _get_element_category(self, element_name: str) -> str:
        """Get semantic category for an element"""
        local_name = element_name.split(':')[-1].lower()
        
        for category, elements in self.semantic_categories.items():
            if local_name in elements:
                return category
        
        return 'custom'
    
    def _overlaps_with_processed(self, start_line: int, end_line: int, 
                                processed_ranges: Set[Tuple[int, int]]) -> bool:
        """Check if a range overlaps with already processed ranges"""
        for proc_start, proc_end in processed_ranges:
            if not (end_line < proc_start or start_line > proc_end):
                return True
        return False
    
    def _extract_xml_fragments(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Parse XML fragments when no clear document structure exists"""
        chunks = []
        
        # Simple element extraction for fragments
        element_pattern = r'<[^/][^>]*>(?:[^<]*(?:<[^>]*>[^<]*</[^>]*>)*[^<]*)</[^>]*>'
        matches = re.finditer(element_pattern, content, re.DOTALL)
        
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            element_content = match.group(0)
            
            if len(element_content.strip()) >= self.config.min_chunk_size:
                # Extract element name
                name_match = re.search(r'<([^/\s>]+)', element_content)
                element_name = name_match.group(1) if name_match else 'fragment'
                
                chunk = self._create_xml_chunk(
                    content=element_content,
                    start_line=start_line,
                    end_line=end_line,
                    context=context,
                    element_name=element_name,
                    xml_type='fragment',
                    semantic_category=self._get_element_category(element_name)
                )
                
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    # Tree-sitter specific methods (when tree-sitter is available)
    
    def _extract_identifier(self, node, content: str) -> Optional[str]:
        """Extract identifier from XML AST node"""
        if node.type == 'element':
            # Try to find the element name from start_tag child
            for child in node.children:
                if child.type == 'start_tag':
                    for grandchild in child.children:
                        if grandchild.type == 'tag_name':
                            tag_name = content[grandchild.start_byte:grandchild.end_byte]
                            return tag_name
        elif node.type in ['start_tag', 'end_tag', 'self_closing_tag']:
            # Extract tag name directly
            for child in node.children:
                if child.type == 'tag_name':
                    tag_name = content[child.start_byte:child.end_byte]
                    return tag_name
        elif node.type == 'comment':
            return 'comment'
        elif node.type == 'processing_instruction':
            return 'processing_instruction'
        elif node.type == 'cdata_section':
            return 'cdata'
        
        return node.type
    
    def _add_semantic_tags(self, chunk: SemanticChunk, node, content: str):
        """Add XML-specific semantic tags for tree-sitter parsing"""
        super()._add_semantic_tags(chunk, node, content)
        
        chunk.add_tag('xml', source='tree_sitter')
        
        if node.type == 'element':
            chunk.add_tag('xml_element', source='tree_sitter')
            
            # Extract element name and add specific tags
            element_name = self._extract_element_name_from_node(node, content)
            if element_name:
                local_name = element_name.split(':')[-1].lower()
                chunk.add_tag(f'element_{local_name}', source='tree_sitter')
                
                # Add category-based tags
                category = self._get_element_category(element_name)
                chunk.add_tag(f'category_{category}', source='tree_sitter')
                chunk.metadata['element_name'] = element_name
                chunk.metadata['semantic_category'] = category
        
        elif node.type == 'comment':
            chunk.add_tag('xml_comment', source='tree_sitter')
        
        elif node.type == 'processing_instruction':
            chunk.add_tag('xml_processing_instruction', source='tree_sitter')
        
        elif node.type == 'cdata_section':
            chunk.add_tag('xml_cdata', source='tree_sitter')
    
    def _extract_element_name_from_node(self, node, content: str) -> Optional[str]:
        """Extract element name from element node"""
        for child in node.children:
            if child.type == 'start_tag':
                for grandchild in child.children:
                    if grandchild.type == 'tag_name':
                        return content[grandchild.start_byte:grandchild.end_byte]
        return None
    
    def _extract_dependencies(self, chunk: SemanticChunk, node, content: str):
        """Extract XML dependencies (references, imports, etc.)"""
        chunk_content = chunk.content
        
        # Look for ID references
        id_refs = re.findall(r'(?:href|src|ref|idref)=["\']#?([^"\']*)["\']', chunk_content)
        for ref in id_refs:
            if ref:
                chunk.dependencies.append(f"references:{ref}")
        
        # Look for namespace declarations and usage
        namespaces = re.findall(self.xml_patterns['namespace'], chunk_content)
        for prefix, uri in namespaces:
            if prefix:
                chunk.dependencies.append(f"namespace:{prefix}")
        
        # Look for imports or includes
        import_patterns = [
            r'<import[^>]*href=["\']([^"\']*)["\']',
            r'<include[^>]*href=["\']([^"\']*)["\']',
            r'<xi:include[^>]*href=["\']([^"\']*)["\']'
        ]
        
        for pattern in import_patterns:
            imports = re.findall(pattern, chunk_content)
            for imp in imports:
                chunk.dependencies.append(f"imports:{imp}")


# Additional utility functions for XML parsing

def create_xml_parser_with_config(config, parsing_strategy: str = 'balanced'):
    """Create XML parser with specific configuration"""
    
    # XML-specific configuration
    if parsing_strategy == 'fine':
        config.xml_preserve_atomic_elements = True
        config.xml_group_similar_elements = False
        config.xml_extract_cdata = True
        config.min_chunk_size = 50
    elif parsing_strategy == 'coarse':
        config.xml_preserve_atomic_elements = True
        config.xml_group_similar_elements = True
        config.xml_extract_cdata = False
        config.min_chunk_size = 200
    else:  # balanced
        config.xml_preserve_atomic_elements = True
        config.xml_group_similar_elements = True
        config.xml_extract_cdata = True
        config.min_chunk_size = 100
    
    config.xml_namespace_aware = True
    
    return XMLParser(config)


def analyze_xml_document_type(content: str) -> str:
    """Analyze XML content to determine document type"""
    content_lower = content.lower()
    
    if '<html' in content_lower or '<xhtml' in content_lower:
        return 'html'
    elif '<svg' in content_lower:
        return 'svg'
    elif '<rss' in content_lower or '<feed' in content_lower:
        return 'feed'
    elif '<configuration' in content_lower or '<config' in content_lower:
        return 'config'
    elif '<project' in content_lower or '<pom' in content_lower:
        return 'build'
    elif '<schema' in content_lower or '<xsd:' in content_lower:
        return 'schema'
    elif '<wsdl' in content_lower or '<soap' in content_lower:
        return 'webservice'
    else:
        return 'data'


# Example usage and testing
if __name__ == "__main__":
    from ..config import ChunkingConfig
    
    # Create XML parser
    config = ChunkingConfig()
    xml_parser = XMLParser(config)
    
    # Test with sample XML
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <catalog>
        <book id="1">
            <title>Introduction to XML</title>
            <author>John Doe</author>
            <description>A comprehensive guide to XML processing and parsing.</description>
        </book>
        <book id="2">
            <title>Advanced XML Techniques</title>
            <author>Jane Smith</author>
            <description>Advanced topics in XML including XSLT and XPath.</description>
        </book>
    </catalog>"""
    
    print("XML Parser Implementation Complete!")
    print("Features:")
    print("- Tree-sitter based parsing with heuristic fallback")
    print("- Semantic element categorization")
    print("- Namespace awareness")
    print("- Content-based chunk extraction")
    print("- Configurable parsing strategies")
    print("- Support for various XML document types")