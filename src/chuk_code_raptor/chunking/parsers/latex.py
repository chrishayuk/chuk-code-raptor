# src/chuk_code_raptor/chunking/parsers/latex.py
"""
Enhanced Heuristic LaTeX Parser - Complete Implementation
========================================================

Advanced heuristic-based LaTeX parser that addresses the key issues identified:
1. Environment detection: 25 environments → comprehensive extraction
2. Mathematical content: 53 elements → aggressive detection  
3. Section granularity: 29 sections → configurable strategies

This implementation provides a complete replacement for the existing LaTeX parser
with significant improvements in accuracy and comprehensiveness.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import logging
import re
from pathlib import Path
from collections import defaultdict, Counter

from ..heuristic_base import HeuristicParser
from ..semantic_chunk import SemanticChunk, create_chunk_id
from ..base import ParseContext
from chuk_code_raptor.core.models import ChunkType

logger = logging.getLogger(__name__)

class HeuristicLaTeXParser(HeuristicParser):
    """
    Enhanced heuristic-based LaTeX parser with comprehensive semantic understanding.
    
    Key improvements:
    - Comprehensive environment detection (targets 25+ environments vs previous 1)
    - Aggressive mathematical content analysis (targets 53+ elements vs previous 43)
    - Configurable parsing strategies for different use cases
    - Better academic document structure recognition
    - Enhanced metadata extraction and semantic tagging
    """
    
    def __init__(self, config):
        # Initialize parent class first
        super().__init__(config)
        
        # Set our specific language support
        self.name = "HeuristicLaTeXParser"
        self.supported_languages = {'latex', 'tex'}
        self.supported_extensions = {'.tex', '.latex', '.ltx', '.sty', '.cls', '.bib'}
        
        # Enhanced configuration options
        self.parsing_strategy = getattr(config, 'latex_parsing_strategy', 'balanced')
        self.environment_extraction_mode = getattr(config, 'latex_env_mode', 'comprehensive')
        self.math_detection_level = getattr(config, 'latex_math_level', 'aggressive')
        self.preserve_all_environments = getattr(config, 'latex_preserve_all_environments', True)
        self.subsection_granularity = getattr(config, 'latex_subsection_granularity', True)
        self.enhanced_math_detection = getattr(config, 'latex_enhanced_math_detection', True)
        
        # Comprehensive LaTeX patterns for enhanced parsing
        self.patterns = {
            # Document structure patterns
            'documentclass': r'\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}',
            'sections': r'\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^}]+)\}',
            'title_commands': r'\\(title|author|date|maketitle)\*?\{([^}]*)\}',
            
            # Comprehensive environment patterns
            'begin_env': r'\\begin\{([^}]+)\}',
            'end_env': r'\\end\{([^}]+)\}',
            'environments': r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}',
            
            # Standard document environments
            'standard_envs': r'\\begin\{(abstract|quote|quotation|verse|flushleft|flushright|center)\}.*?\\end\{\1\}',
            
            # List environments  
            'list_envs': r'\\begin\{(itemize|enumerate|description|trivlist)\}.*?\\end\{\1\}',
            
            # Mathematical environments (comprehensive)
            'math_envs': r'\\begin\{(equation|align|gather|multline|split|eqnarray|alignat|flalign|displaymath|math|cases|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix)\*?\}.*?\\end\{\1\*?\}',
            
            # Float environments
            'float_envs': r'\\begin\{(figure|table|wrapfigure|wraptable|sidewaysfigure|sidewaystable)\*?\}.*?\\end\{\1\*?\}',
            
            # Tabular environments
            'tabular_envs': r'\\begin\{(tabular|array|tabularx|longtable|supertabular|xtabular)\}.*?\\end\{\1\}',
            
            # Code environments
            'code_envs': r'\\begin\{(verbatim|lstlisting|minted|alltt|Verbatim)\*?\}.*?\\end\{\1\*?\}',
            
            # Theorem-like environments
            'theorem_envs': r'\\begin\{(theorem|lemma|proposition|corollary|definition|proof|example|remark|note|claim|fact|observation)\*?\}.*?\\end\{\1\*?\}',
            
            # Algorithm environments
            'algorithm_envs': r'\\begin\{(algorithm|algorithmic|pseudocode)\*?\}.*?\\end\{\1\*?\}',
            
            # Special environments
            'special_envs': r'\\begin\{(minipage|multicols|tikzpicture|pgfpicture|subfigure|subequations)\}.*?\\end\{\1\}',
            
            # Enhanced mathematical patterns for aggressive detection
            'inline_dollar': r'\$[^$\n]+\$',
            'inline_paren': r'\\\([^)]+\\\)',
            'inline_ensuremath': r'\\ensuremath\{[^}]+\}',
            'display_dollar': r'\$\$[^$]+\$\$',
            'display_bracket': r'\\\[[^\]]+\\\]',
            'math_commands': r'\\(?:frac|sqrt|sum|int|prod|lim|sin|cos|tan|log|ln|exp|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|sigma|omega)\b',
            'math_symbols': r'[∑∏∫√±×÷≤≥≠≈∞∂∇∆⊂⊃∈∉∀∃]',
            'math_operators': r'\\(?:cdot|times|div|pm|mp|oplus|ominus|otimes|cup|cap|subset|supset|in|notin|forall|exists)',
            'matrix_like': r'\\begin\{[pbvBV]?matrix\}.*?\\end\{[pbvBV]?matrix\}',
            'cases_like': r'\\begin\{cases\}.*?\\end\{cases\}',
            'aligned_like': r'\\begin\{aligned\}.*?\\end\{aligned\}',
            
            # Enhanced citation and reference patterns
            'citations': r'\\(?:cite|citep|citet|nocite)(?:\[[^\]]*\])*\{([^}]+)\}',
            'references': r'\\(?:ref|eqref|pageref|nameref)\{([^}]+)\}',
            'labels': r'\\label\{([^}]+)\}',
            
            # Graphics and includes
            'graphics': r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',
            'includes': r'\\(?:input|include|subfile|subimport)\{([^}]+)\}',
            
            # Bibliography and packages
            'bibliography': r'\\(?:bibliography|addbibresource)\{([^}]+)\}',
            'bibitem': r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}',
            'usepackage': r'\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}',
            'newcommand': r'\\(?:newcommand|renewcommand|providecommand)\{([^}]+)\}',
            
            # Additional patterns
            'comments': r'%.*$',
            'footnotes': r'\\footnote\{([^}]+)\}',
            'marginpar': r'\\marginpar\{([^}]+)\}',
        }
        
        # Section hierarchy levels
        self.section_hierarchy = {
            'part': 0,
            'chapter': 1,
            'section': 2,
            'subsection': 3,
            'subsubsection': 4,
            'paragraph': 5,
            'subparagraph': 6
        }
        
        # Environment categories for systematic processing
        self.environment_categories = {
            'atomic_environments': {
                'equation', 'align', 'gather', 'multline', 'split', 'eqnarray',
                'alignat', 'flalign', 'figure', 'table', 'tabular', 'array',
                'tikzpicture', 'pgfpicture', 'lstlisting', 'verbatim', 'minted',
                'algorithm', 'algorithmic', 'proof', 'theorem', 'lemma',
                'proposition', 'corollary', 'definition', 'example', 'remark'
            },
            'math_environments': {
                'equation', 'align', 'gather', 'multline', 'split', 'eqnarray',
                'alignat', 'flalign', 'displaymath', 'math', 'cases', 'matrix',
                'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix', 'smallmatrix'
            },
            'float_environments': {
                'figure', 'table', 'wrapfigure', 'wraptable', 'sidewaysfigure', 'sidewaystable'
            },
            'list_environments': {
                'itemize', 'enumerate', 'description', 'trivlist'
            },
            'code_environments': {
                'verbatim', 'lstlisting', 'minted', 'alltt', 'Verbatim'
            },
            'theorem_environments': {
                'theorem', 'lemma', 'proposition', 'corollary', 'definition',
                'proof', 'example', 'remark', 'note', 'claim', 'fact', 'observation'
            }
        }
        
        # Mathematical pattern categories for comprehensive detection
        self.math_pattern_categories = {
            'inline_patterns': [
                ('inline_dollar', r'\$[^$\n]+\$'),
                ('inline_paren', r'\\\([^)]+\\\)'),
                ('inline_ensuremath', r'\\ensuremath\{[^}]+\}'),
            ],
            'display_patterns': [
                ('display_dollar', r'\$\$[^$]+\$\$'),
                ('display_bracket', r'\\\[[^\]]+\\\]'),
            ],
            'command_patterns': [
                ('math_commands', r'\\(?:frac|sqrt|sum|int|prod|lim|sin|cos|tan|log|ln|exp)\b'),
                ('greek_letters', r'\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|sigma|omega)\b'),
                ('math_operators', r'\\(?:cdot|times|div|pm|mp|oplus|ominus|otimes)\b'),
                ('relations', r'\\(?:leq|geq|neq|approx|subset|supset|in|notin|forall|exists)\b'),
            ],
            'symbol_patterns': [
                ('unicode_math', r'[∑∏∫√±×÷≤≥≠≈∞∂∇∆⊂⊃∈∉∀∃]'),
                ('superscript', r'\^[^{]|\^\{[^}]*\}'),
                ('subscript', r'_[^{]|_\{[^}]*\}'),
            ],
            'structure_patterns': [
                ('matrix_like', r'\\begin\{[pbvBV]?matrix\}.*?\\end\{[pbvBV]?matrix\}'),
                ('cases_like', r'\\begin\{cases\}.*?\\end\{cases\}'),
                ('aligned_like', r'\\begin\{aligned\}.*?\\end\{aligned\}'),
            ]
        }
    
    def can_parse(self, language: str, file_extension: str) -> bool:
        """Check if this parser can handle the given language/extension"""
        return (language in self.supported_languages or 
                file_extension in self.supported_extensions)
    
    def _extract_chunks_heuristically(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """
        Main entry point for enhanced LaTeX chunk extraction.
        
        This method orchestrates the comprehensive parsing strategy to address
        the issues identified in the demo output.
        """
        logger.info(f"Starting enhanced LaTeX parsing with strategy: {self.parsing_strategy}")
        
        chunks = []
        lines = content.split('\n')
        processed_ranges = set()  # Track processed content to avoid overlaps
        
        # Phase 1: Extract document preamble
        preamble_chunks = self._extract_preamble_comprehensive(content, context)
        chunks.extend(preamble_chunks)
        for chunk in preamble_chunks:
            processed_ranges.add((chunk.start_line, chunk.end_line))
        
        # Phase 2: Extract all environments comprehensively (addresses 25→1 issue)
        if self.environment_extraction_mode == 'comprehensive':
            env_chunks = self._extract_environments_comprehensive(content, context, processed_ranges)
            chunks.extend(env_chunks)
            for chunk in env_chunks:
                processed_ranges.add((chunk.start_line, chunk.end_line))
        
        # Phase 3: Extract mathematical content aggressively (addresses 53→43 issue) 
        if self.enhanced_math_detection and self.math_detection_level == 'aggressive':
            math_chunks = self._extract_mathematical_content_aggressive(content, context, processed_ranges)
            chunks.extend(math_chunks)
            for chunk in math_chunks:
                processed_ranges.add((chunk.start_line, chunk.end_line))
        
        # Phase 4: Extract sections with configurable granularity (addresses 29→11 issue)
        section_chunks = self._extract_sections_enhanced(content, context, processed_ranges)
        chunks.extend(section_chunks)
        for chunk in section_chunks:
            processed_ranges.add((chunk.start_line, chunk.end_line))
        
        # Phase 5: Extract remaining content (fallback)
        remaining_chunks = self._extract_remaining_content(content, context, processed_ranges)
        chunks.extend(remaining_chunks)
        
        # Post-processing: enhance metadata and validate chunks
        enhanced_chunks = self._post_process_chunks(chunks, content)
        
        logger.info(f"Enhanced LaTeX parsing completed: {len(enhanced_chunks)} chunks created")
        return enhanced_chunks
    
    def _extract_preamble_comprehensive(self, content: str, context: ParseContext) -> List[SemanticChunk]:
        """Extract document preamble with enhanced analysis"""
        chunks = []
        lines = content.split('\n')
        
        # Find document begin
        preamble_end = self._find_document_begin(lines)
        if preamble_end <= 0:
            return chunks
        
        preamble_content = '\n'.join(lines[0:preamble_end]).strip()
        if len(preamble_content) < 20:  # Very minimal preamble requirement
            return chunks
        
        chunk = self._create_enhanced_chunk(
            content=preamble_content,
            start_line=1,
            end_line=preamble_end,
            chunk_type=ChunkType.TEXT_BLOCK,
            context=context,
            identifier="preamble",
            latex_type="preamble",
            semantic_type="Document Preamble"
        )
        
        # Enhanced preamble analysis
        self._analyze_preamble_enhanced(chunk)
        chunks.append(chunk)
        
        return chunks
    
    def _extract_environments_comprehensive(self, content: str, context: ParseContext, 
                                         processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """
        Comprehensive environment extraction to address the 25→1 environment issue.
        
        This method systematically extracts all environment types to achieve
        better coverage than the previous single environment chunk.
        """
        chunks = []
        
        # Process each environment category systematically
        for category, env_pattern in self._get_environment_patterns_by_category().items():
            category_chunks = self._extract_environment_category(
                content, context, category, env_pattern, processed_ranges
            )
            chunks.extend(category_chunks)
        
        # Also extract any remaining environments not caught by categories
        remaining_env_chunks = self._extract_remaining_environments(content, context, processed_ranges)
        chunks.extend(remaining_env_chunks)
        
        logger.info(f"Environment extraction: {len(chunks)} environment chunks created")
        return chunks
    
    def _get_environment_patterns_by_category(self) -> Dict[str, str]:
        """Get environment patterns organized by category"""
        return {
            'standard_envs': r'\\begin\{(abstract|quote|quotation|verse|flushleft|flushright|center)\}.*?\\end\{\1\}',
            'list_envs': r'\\begin\{(itemize|enumerate|description|trivlist)\}.*?\\end\{\1\}',
            'math_envs': r'\\begin\{(equation|align|gather|multline|split|eqnarray|alignat|flalign|displaymath|math|cases|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix)\*?\}.*?\\end\{\1\*?\}',
            'float_envs': r'\\begin\{(figure|table|wrapfigure|wraptable|sidewaysfigure|sidewaystable)\*?\}.*?\\end\{\1\*?\}',
            'tabular_envs': r'\\begin\{(tabular|array|tabularx|longtable|supertabular|xtabular)\}.*?\\end\{\1\}',
            'code_envs': r'\\begin\{(verbatim|lstlisting|minted|alltt|Verbatim)\*?\}.*?\\end\{\1\*?\}',
            'theorem_envs': r'\\begin\{(theorem|lemma|proposition|corollary|definition|proof|example|remark|note|claim|fact|observation)\*?\}.*?\\end\{\1\*?\}',
            'algorithm_envs': r'\\begin\{(algorithm|algorithmic|pseudocode)\*?\}.*?\\end\{\1\*?\}',
            'special_envs': r'\\begin\{(minipage|multicols|tikzpicture|pgfpicture|subfigure|subequations)\}.*?\\end\{\1\}',
        }
    
    def _extract_environment_category(self, content: str, context: ParseContext, 
                                    category: str, pattern: str, 
                                    processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """Extract environments from a specific category"""
        chunks = []
        matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            
            # Calculate line numbers
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            # Skip if overlaps with already processed content
            if self._overlaps_with_processed(start_line, end_line, processed_ranges):
                continue
            
            env_content = match.group(0)
            env_name = self._extract_environment_name(env_content)
            
            # Apply parsing strategy to determine if this should be a chunk
            if self._should_create_environment_chunk(env_name, env_content, category):
                chunk = self._create_enhanced_environment_chunk(
                    env_content, start_line, end_line, env_name, category, context
                )
                if chunk:
                    chunks.append(chunk)
                    processed_ranges.add((start_line, end_line))
        
        return chunks
    
    def _extract_mathematical_content_aggressive(self, content: str, context: ParseContext,
                                               processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """
        Aggressive mathematical content detection to improve 43→53+ coverage.
        
        This method uses comprehensive pattern matching to detect all forms
        of mathematical content that might have been missed.
        """
        chunks = []
        
        # Process each category of mathematical patterns
        for category, patterns in self.math_pattern_categories.items():
            for pattern_name, pattern in patterns:
                category_chunks = self._extract_math_pattern_chunks(
                    content, context, pattern_name, pattern, processed_ranges
                )
                chunks.extend(category_chunks)
        
        logger.info(f"Mathematical content extraction: {len(chunks)} math chunks created")
        return chunks
    
    def _extract_math_pattern_chunks(self, content: str, context: ParseContext,
                                   pattern_name: str, pattern: str,
                                   processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """Extract chunks for a specific mathematical pattern"""
        chunks = []
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            # Skip if overlaps with processed content
            if self._overlaps_with_processed(start_line, end_line, processed_ranges):
                continue
            
            math_content = match.group(0)
            
            # Apply significance filter based on detection level
            if self._is_significant_math_content(math_content, pattern_name):
                chunk = self._create_enhanced_math_chunk(
                    math_content, start_line, end_line, pattern_name, context
                )
                if chunk:
                    chunks.append(chunk)
                    processed_ranges.add((start_line, end_line))
        
        return chunks
    
    def _extract_sections_enhanced(self, content: str, context: ParseContext,
                                 processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """
        Enhanced section extraction to better handle the 29→11 section issue.
        
        This method provides configurable granularity for section handling
        based on the parsing strategy.
        """
        chunks = []
        lines = content.split('\n')
        section_positions = []
        
        # Find all section positions
        for i, line in enumerate(lines):
            section_match = re.search(self.patterns['sections'], line)
            if section_match:
                section_type = section_match.group(1)
                section_title = section_match.group(2)
                section_level = self.section_hierarchy.get(section_type, 10)
                
                section_positions.append({
                    'line': i + 1,
                    'type': section_type,
                    'title': section_title,
                    'level': section_level,
                    'index': i
                })
        
        # Process sections based on strategy
        for i, section_info in enumerate(section_positions):
            # Determine section end
            if i + 1 < len(section_positions):
                next_section = section_positions[i + 1]
                section_end_line = next_section['line'] - 1
                section_end_index = next_section['index'] - 1
            else:
                section_end_line = len(lines)
                section_end_index = len(lines) - 1
            
            # Apply granularity rules based on parsing strategy
            if self._should_create_section_chunk(section_info, section_positions, i):
                start_line = section_info['line']
                end_line = section_end_line
                
                # Skip if overlaps with processed content
                if self._overlaps_with_processed(start_line, end_line, processed_ranges):
                    continue
                
                section_content = '\n'.join(lines[section_info['index']:section_end_index + 1])
                
                chunk = self._create_enhanced_section_chunk(
                    section_content, start_line, end_line, section_info, context
                )
                if chunk:
                    chunks.append(chunk)
                    processed_ranges.add((start_line, end_line))
        
        logger.info(f"Section extraction: {len(chunks)} section chunks created")
        return chunks
    
    def _extract_remaining_environments(self, content: str, context: ParseContext,
                                      processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """Extract any remaining environments not caught by category-specific extraction"""
        chunks = []
        
        # Generic environment pattern
        env_pattern = r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}'
        matches = re.finditer(env_pattern, content, re.DOTALL)
        
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            # Skip if already processed
            if self._overlaps_with_processed(start_line, end_line, processed_ranges):
                continue
            
            env_content = match.group(0)
            env_name = match.group(1)
            
            # Only create chunk if significant and not already handled
            if len(env_content.strip()) >= 20:  # Minimal content requirement
                chunk = self._create_enhanced_environment_chunk(
                    env_content, start_line, end_line, env_name, 'generic', context
                )
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def _extract_remaining_content(self, content: str, context: ParseContext,
                                 processed_ranges: Set[Tuple[int, int]]) -> List[SemanticChunk]:
        """Extract any remaining unprocessed content as fallback chunks"""
        chunks = []
        lines = content.split('\n')
        
        current_start = 1
        for start_line, end_line in sorted(processed_ranges):
            # Check for gap before this processed range
            if current_start < start_line:
                gap_content = '\n'.join(lines[current_start-1:start_line-1]).strip()
                if len(gap_content) >= self.config.min_chunk_size:
                    chunk = self._create_enhanced_chunk(
                        content=gap_content,
                        start_line=current_start,
                        end_line=start_line - 1,
                        chunk_type=ChunkType.TEXT_BLOCK,
                        context=context,
                        identifier=f"content_{current_start}",
                        latex_type="content",
                        semantic_type="General Content"
                    )
                    if chunk:
                        self._analyze_general_content(chunk)
                        chunks.append(chunk)
            
            current_start = end_line + 1
        
        # Handle content after last processed range
        if current_start <= len(lines):
            gap_content = '\n'.join(lines[current_start-1:]).strip()
            if len(gap_content) >= self.config.min_chunk_size:
                chunk = self._create_enhanced_chunk(
                    content=gap_content,
                    start_line=current_start,
                    end_line=len(lines),
                    chunk_type=ChunkType.TEXT_BLOCK,
                    context=context,
                    identifier=f"content_{current_start}",
                    latex_type="content",
                    semantic_type="General Content"
                )
                if chunk:
                    self._analyze_general_content(chunk)
                    chunks.append(chunk)
        
        return chunks
    
    # Helper methods for chunk creation and analysis
    
    def _create_enhanced_chunk(self, content: str, start_line: int, end_line: int,
                             chunk_type: ChunkType, context: ParseContext,
                             identifier: str, latex_type: str, semantic_type: str,
                             **additional_metadata) -> SemanticChunk:
        """Create an enhanced chunk with comprehensive metadata"""
        
        chunk = self._create_heuristic_chunk(
            content=content,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            context=context,
            identifier=identifier,
            latex_type=latex_type,
            semantic_type=semantic_type,
            **additional_metadata
        )
        
        # Add enhanced metadata
        chunk.metadata.update({
            'parser_version': 'enhanced_heuristic_v2',
            'extraction_strategy': self.parsing_strategy,
            'content_length': len(content),
            'line_count': end_line - start_line + 1,
        })
        
        return chunk
    
    def _create_enhanced_environment_chunk(self, content: str, start_line: int, end_line: int,
                                         env_name: str, category: str, context: ParseContext) -> Optional[SemanticChunk]:
        """Create an enhanced environment chunk with detailed analysis"""
        
        chunk = self._create_enhanced_chunk(
            content=content,
            start_line=start_line,
            end_line=end_line,
            chunk_type=ChunkType.TEXT_BLOCK,
            context=context,
            identifier=f"env_{env_name}_{start_line}",
            latex_type="environment",
            semantic_type=self._get_environment_semantic_type(env_name),
            environment_name=env_name,
            environment_category=category
        )
        
        # Enhanced environment tagging
        chunk.add_tag('environment', source='heuristic')
        chunk.add_tag(f'env_{env_name}', source='heuristic')
        chunk.add_tag(f'env_category_{category}', source='heuristic')
        
        # Category-specific tagging
        self._add_environment_category_tags(chunk, env_name, category)
        
        # Comprehensive content analysis
        self._analyze_environment_content_enhanced(chunk, env_name)
        
        return chunk
    
    def _create_enhanced_math_chunk(self, content: str, start_line: int, end_line: int,
                                  pattern_name: str, context: ParseContext) -> Optional[SemanticChunk]:
        """Create an enhanced mathematical content chunk"""
        
        chunk = self._create_enhanced_chunk(
            content=content,
            start_line=start_line,
            end_line=end_line,
            chunk_type=ChunkType.TEXT_BLOCK,
            context=context,
            identifier=f"math_{pattern_name}_{start_line}",
            latex_type="mathematics",
            semantic_type="Mathematical Content",
            math_pattern_type=pattern_name
        )
        
        # Mathematical content tagging
        chunk.add_tag('mathematics', source='heuristic')
        chunk.add_tag(f'math_{pattern_name}', source='heuristic')
        
        # Analyze mathematical complexity
        self._analyze_mathematical_complexity(chunk)
        
        return chunk
    
    def _create_enhanced_section_chunk(self, content: str, start_line: int, end_line: int,
                                     section_info: Dict, context: ParseContext) -> Optional[SemanticChunk]:
        """Create an enhanced section chunk with hierarchical analysis"""
        
        section_type = section_info['type']
        section_title = section_info['title']
        section_level = section_info['level']
        
        chunk = self._create_enhanced_chunk(
            content=content,
            start_line=start_line,
            end_line=end_line,
            chunk_type=ChunkType.TEXT_BLOCK,
            context=context,
            identifier=f"section_{self._clean_identifier(section_title)}_{start_line}",
            latex_type="section",
            semantic_type=self._get_section_semantic_type(section_title),
            section_type=section_type,
            section_level=section_level,
            section_title=section_title
        )
        
        # Section tagging
        chunk.add_tag('section', source='heuristic')
        chunk.add_tag(f'section_{section_type}', source='heuristic')
        chunk.add_tag(f'section_level_{section_level}', source='heuristic')
        
        # Analyze section content
        self._analyze_section_content_enhanced(chunk, section_title, section_type)
        
        return chunk
    
    # Analysis methods
    
    def _analyze_preamble_enhanced(self, chunk: SemanticChunk):
        """Enhanced preamble analysis"""
        content = chunk.content
        
        # Document class analysis
        doc_class_match = re.search(self.patterns['documentclass'], content)
        if doc_class_match:
            doc_class = doc_class_match.group(1)
            chunk.metadata['document_class'] = doc_class
            chunk.add_tag(f'docclass_{doc_class}', source='heuristic')
            
            # Document type inference
            if doc_class in ['article', 'paper']:
                chunk.add_tag('academic_paper', source='heuristic')
            elif doc_class in ['book', 'report']:
                chunk.add_tag('long_document', source='heuristic')
            elif doc_class in ['beamer']:
                chunk.add_tag('presentation', source='heuristic')
        
        # Package analysis
        packages = re.findall(self.patterns['usepackage'], content)
        if packages:
            chunk.metadata['packages'] = packages
            chunk.metadata['package_count'] = len(packages)
            chunk.add_tag('has_packages', source='heuristic')
            
            # Specialized package detection
            math_packages = {'amsmath', 'amsfonts', 'amssymb', 'mathtools'}
            graphics_packages = {'graphicx', 'tikz', 'pgfplots', 'pgf'}
            table_packages = {'booktabs', 'longtable', 'tabularx'}
            
            if any(pkg in math_packages for pkg in packages):
                chunk.add_tag('math_heavy', source='heuristic')
            if any(pkg in graphics_packages for pkg in packages):
                chunk.add_tag('graphics_heavy', source='heuristic')
            if any(pkg in table_packages for pkg in packages):
                chunk.add_tag('table_heavy', source='heuristic')
        
        # Custom command analysis
        custom_commands = re.findall(self.patterns['newcommand'], content)
        if custom_commands:
            chunk.metadata['custom_commands'] = custom_commands
            chunk.add_tag('defines_commands', source='heuristic')
    
    def _analyze_environment_content_enhanced(self, chunk: SemanticChunk, env_name: str):
        """Enhanced environment content analysis"""
        content = chunk.content
        
        # Basic environment properties
        chunk.metadata['environment_length'] = len(content)
        chunk.metadata['is_atomic'] = env_name in self.environment_categories['atomic_environments']
        
        # Content analysis by environment type
        if env_name in self.environment_categories['math_environments']:
            self._analyze_mathematical_environment(chunk, env_name)
        elif env_name in self.environment_categories['float_environments']:
            self._analyze_float_environment(chunk, env_name)
        elif env_name in self.environment_categories['theorem_environments']:
            self._analyze_theorem_environment(chunk, env_name)
        elif env_name in self.environment_categories['code_environments']:
            self._analyze_code_environment(chunk, env_name)
        
        # General content features
        self._analyze_latex_content_comprehensive(chunk)
    
    def _analyze_mathematical_complexity(self, chunk: SemanticChunk):
        """Analyze mathematical content complexity with enhanced metrics"""
        content = chunk.content
        
        complexity_metrics = {
            'fractions': len(re.findall(r'\\frac\{', content)),
            'integrals': len(re.findall(r'\\int', content)),
            'summations': len(re.findall(r'\\sum', content)),
            'products': len(re.findall(r'\\prod', content)),
            'limits': len(re.findall(r'\\lim', content)),
            'roots': len(re.findall(r'\\sqrt', content)),
            'matrices': len(re.findall(r'\\begin\{[pbvBV]?matrix\}', content)),
            'greek_letters': len(re.findall(r'\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|sigma|omega)', content)),
            'superscripts': len(re.findall(r'\^[^{]|\^\{[^}]*\}', content)),
            'subscripts': len(re.findall(r'_[^{]|_\{[^}]*\}', content)),
        }
        
        total_complexity = sum(complexity_metrics.values())
        chunk.metadata['math_complexity_score'] = total_complexity
        chunk.metadata['math_elements'] = complexity_metrics
        
        # Complexity classification
        if total_complexity > 20:
            complexity_level = 'very_complex'
            chunk.add_tag('math_very_complex', source='heuristic')
        elif total_complexity > 10:
            complexity_level = 'complex'
            chunk.add_tag('math_complex', source='heuristic')
        elif total_complexity > 5:
            complexity_level = 'moderate'
            chunk.add_tag('math_moderate', source='heuristic')
        else:
            complexity_level = 'simple'
            chunk.add_tag('math_simple', source='heuristic')
        
        chunk.metadata['math_complexity_level'] = complexity_level
    
    def _analyze_section_content_enhanced(self, chunk: SemanticChunk, title: str, section_type: str):
        """Enhanced section content analysis"""
        title_lower = title.lower()
        content_lower = chunk.content.lower()
        
        # Academic section classification
        academic_keywords = {
            'introduction': ['introduction', 'intro', 'background', 'motivation'],
            'methodology': ['method', 'approach', 'algorithm', 'technique', 'procedure'],
            'results': ['result', 'experiment', 'evaluation', 'performance', 'finding'],
            'discussion': ['discussion', 'analysis', 'interpretation', 'implication'],
            'conclusion': ['conclusion', 'summary', 'future work', 'closing'],
            'literature': ['related work', 'literature', 'prior work', 'survey'],
            'theory': ['theory', 'theoretical', 'mathematical foundation', 'formal'],
        }
        
        for category, keywords in academic_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                chunk.add_tag(f'academic_{category}', source='heuristic')
                chunk.metadata['academic_section_type'] = category
                break
        
        # Content density analysis
        self._analyze_latex_content_comprehensive(chunk)
        
        # Section-specific analysis
        if section_type in ['section', 'subsection']:
            chunk.add_tag('major_section', source='heuristic')
        elif section_type in ['subsubsection', 'paragraph']:
            chunk.add_tag('minor_section', source='heuristic')
    
    def _analyze_latex_content_comprehensive(self, chunk: SemanticChunk):
        """Comprehensive LaTeX content analysis with enhanced pattern detection"""
        content = chunk.content
        
        # Enhanced mathematical content counting
        math_counts = {
            'inline_math': len(re.findall(r'\$[^$\n]+\$', content)),
            'display_math_dollar': len(re.findall(r'\$\$[^$]+\$\$', content)),
            'display_math_bracket': len(re.findall(r'\\\[[^\]]+\\\]', content)),
            'equation_environments': len(re.findall(r'\\begin\{(equation|align|gather|multline|split|eqnarray|alignat|flalign)\*?\}', content)),
            'math_commands': len(re.findall(self.patterns['math_commands'], content)),
            'greek_letters': len(re.findall(r'\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|sigma|omega)', content)),
        }
        
        total_math = sum(math_counts.values())
        if total_math > 0:
            chunk.add_tag('has_mathematics', source='heuristic')
            chunk.metadata['math_element_counts'] = math_counts
            chunk.metadata['total_math_elements'] = total_math
            
            # Math intensity classification
            if total_math > 20:
                chunk.add_tag('math_intensive', source='heuristic')
            elif total_math > 10:
                chunk.add_tag('math_heavy', source='heuristic')
            elif total_math > 5:
                chunk.add_tag('math_moderate', source='heuristic')
            else:
                chunk.add_tag('math_light', source='heuristic')
        
        # Enhanced citation and reference analysis
        citations = re.findall(self.patterns['citations'], content)
        references = re.findall(self.patterns['references'], content)
        labels = re.findall(self.patterns['labels'], content)
        
        if citations:
            chunk.add_tag('has_citations', source='heuristic')
            chunk.metadata['citations'] = citations
            chunk.metadata['citation_count'] = len(citations)
        
        if references:
            chunk.add_tag('has_references', source='heuristic')
            chunk.metadata['references'] = references
            chunk.metadata['reference_count'] = len(references)
        
        if labels:
            chunk.add_tag('has_labels', source='heuristic')
            chunk.metadata['labels'] = labels
            chunk.metadata['label_count'] = len(labels)
        
        # Graphics and media analysis
        graphics = re.findall(self.patterns['graphics'], content)
        if graphics:
            chunk.add_tag('has_graphics', source='heuristic')
            chunk.metadata['graphics'] = graphics
            chunk.metadata['graphics_count'] = len(graphics)
        
        # Environment analysis within content
        nested_envs = re.findall(r'\\begin\{([^}]+)\}', content)
        if nested_envs:
            chunk.metadata['nested_environments'] = list(set(nested_envs))
            chunk.metadata['nested_env_count'] = len(nested_envs)
        
        # Command density analysis
        commands = re.findall(r'\\([a-zA-Z]+)', content)
        if commands:
            command_counter = Counter(commands)
            chunk.metadata['command_frequency'] = dict(command_counter.most_common(10))
            chunk.metadata['unique_commands'] = len(set(commands))
            chunk.metadata['total_commands'] = len(commands)
    
    # Utility and decision methods
    
    def _find_document_begin(self, lines: List[str]) -> int:
        """Find where the document content begins"""
        for i, line in enumerate(lines):
            if re.search(r'\\begin\{document\}', line.strip()):
                return i
        return 0
    
    def _extract_environment_name(self, env_content: str) -> str:
        """Extract environment name from environment content"""
        match = re.search(r'\\begin\{([^}]+)\}', env_content)
        return match.group(1) if match else 'unknown'
    
    def _should_create_environment_chunk(self, env_name: str, content: str, category: str) -> bool:
        """Determine if an environment should become its own chunk"""
        
        # Always create chunks for atomic environments
        if env_name in self.environment_categories['atomic_environments']:
            return True
        
        # Strategy-based decisions
        if self.parsing_strategy == 'fine':
            return len(content.strip()) >= 10  # Very lenient for fine-grained
        elif self.parsing_strategy == 'balanced':
            # Balanced approach - consider significance
            significant_envs = {
                'abstract', 'figure', 'table', 'algorithm', 'theorem', 'proof',
                'equation', 'align', 'gather', 'lstlisting', 'verbatim', 'itemize', 'enumerate'
            }
            return (env_name in significant_envs or 
                    len(content.strip()) >= self.config.min_chunk_size // 2)
        else:  # coarse
            # Only major structural environments
            major_envs = {'abstract', 'figure', 'table', 'algorithm', 'theorem'}
            return env_name in major_envs and len(content.strip()) >= self.config.min_chunk_size
    
    def _should_create_section_chunk(self, section_info: Dict, all_sections: List[Dict], index: int) -> bool:
        """Determine if a section should become its own chunk"""
        section_level = section_info['level']
        
        if self.parsing_strategy == 'fine':
            # Include all sections and subsections
            return section_level <= 4
        elif self.parsing_strategy == 'balanced':
            # Include major sections and important subsections
            return section_level <= 3 or (section_level == 4 and self.subsection_granularity)
        else:  # coarse
            # Only major sections
            return section_level <= 2
    
    def _is_significant_math_content(self, content: str, pattern_name: str) -> bool:
        """Determine if mathematical content is significant enough for its own chunk"""
        
        if self.math_detection_level == 'basic':
            # Only display math and equation environments
            return pattern_name in ['display_dollar', 'display_bracket', 'math_envs']
        elif self.math_detection_level == 'standard':
            # Include most math but filter trivial cases
            trivial_patterns = {'unicode_math', 'superscript', 'subscript'}
            return pattern_name not in trivial_patterns and len(content.strip()) > 3
        else:  # aggressive
            # Include almost all mathematical content
            return len(content.strip()) > 1
    
    def _overlaps_with_processed(self, start_line: int, end_line: int, 
                                processed_ranges: Set[Tuple[int, int]]) -> bool:
        """Check if a range overlaps with already processed ranges"""
        for proc_start, proc_end in processed_ranges:
            if not (end_line < proc_start or start_line > proc_end):
                return True
        return False
    
    def _get_environment_semantic_type(self, env_name: str) -> str:
        """Get semantic type for an environment"""
        type_mapping = {
            'abstract': 'Document Abstract',
            'equation': 'Mathematical Equation',
            'align': 'Mathematical Equations',
            'figure': 'Figure Display',
            'table': 'Data Table',
            'algorithm': 'Algorithm Description',
            'theorem': 'Mathematical Theorem',
            'proof': 'Mathematical Proof',
            'lstlisting': 'Code Listing',
            'verbatim': 'Verbatim Text',
            'itemize': 'Unordered List',
            'enumerate': 'Ordered List',
        }
        return type_mapping.get(env_name, f'{env_name.title()} Environment')
    
    def _get_section_semantic_type(self, title: str) -> str:
        """Determine semantic type of a section based on its title"""
        title_lower = title.lower()
        
        if any(keyword in title_lower for keyword in ['introduction', 'intro']):
            return 'Introduction Section'
        elif any(keyword in title_lower for keyword in ['related work', 'literature', 'background']):
            return 'Literature Review'
        elif any(keyword in title_lower for keyword in ['method', 'approach', 'algorithm']):
            return 'Methodology Section'
        elif any(keyword in title_lower for keyword in ['result', 'experiment', 'evaluation']):
            return 'Results Section'
        elif any(keyword in title_lower for keyword in ['discussion', 'analysis']):
            return 'Discussion Section'
        elif any(keyword in title_lower for keyword in ['conclusion', 'summary']):
            return 'Conclusion Section'
        else:
            return 'Section Content'
    
    def _add_environment_category_tags(self, chunk: SemanticChunk, env_name: str, category: str):
        """Add category-specific tags to environment chunks"""
        category_tags = {
            'math_envs': ['mathematical', 'equation', 'formal_math'],
            'float_envs': ['visual', 'presentation', 'display'],
            'theorem_envs': ['theoretical', 'academic', 'formal_statement'],
            'code_envs': ['technical', 'implementation', 'source_code'],
            'algorithm_envs': ['algorithmic', 'procedural', 'computational'],
            'list_envs': ['structured', 'enumeration', 'organized'],
            'tabular_envs': ['data', 'tabular', 'structured_data'],
            'standard_envs': ['textual', 'formatted', 'presentation'],
        }
        
        if category in category_tags:
            for tag in category_tags[category]:
                chunk.add_tag(tag, source='heuristic')
    
    def _analyze_mathematical_environment(self, chunk: SemanticChunk, env_name: str):
        """Analyze mathematical environment content"""
        chunk.add_tag('mathematical_environment', source='heuristic')
        if env_name in ['equation', 'align', 'gather']:
            chunk.add_tag('numbered_equation', source='heuristic')
        
        # Analyze mathematical complexity within the environment
        self._analyze_mathematical_complexity(chunk)
    
    def _analyze_float_environment(self, chunk: SemanticChunk, env_name: str):
        """Analyze float environment (figure/table) content"""
        content = chunk.content
        
        if 'figure' in env_name:
            chunk.add_tag('figure_content', source='heuristic')
            if re.search(r'\\includegraphics', content):
                chunk.add_tag('contains_image', source='heuristic')
            if re.search(r'\\caption', content):
                chunk.add_tag('has_caption', source='heuristic')
        
        elif 'table' in env_name:
            chunk.add_tag('table_content', source='heuristic')
            if re.search(r'\\begin\{tabular\}', content):
                chunk.add_tag('contains_tabular', source='heuristic')
    
    def _analyze_theorem_environment(self, chunk: SemanticChunk, env_name: str):
        """Analyze theorem-like environment content"""
        chunk.add_tag('formal_statement', source='heuristic')
        
        if env_name == 'proof':
            chunk.add_tag('mathematical_proof', source='heuristic')
        elif env_name in ['theorem', 'lemma', 'proposition']:
            chunk.add_tag('mathematical_theorem', source='heuristic')
        elif env_name == 'definition':
            chunk.add_tag('formal_definition', source='heuristic')
    
    def _analyze_code_environment(self, chunk: SemanticChunk, env_name: str):
        """Analyze code environment content"""
        chunk.add_tag('source_code', source='heuristic')
        
        # Try to detect programming language
        content = chunk.content
        if re.search(r'language\s*=\s*([^,\]]+)', content):
            lang_match = re.search(r'language\s*=\s*([^,\]]+)', content)
            if lang_match:
                language = lang_match.group(1).strip()
                chunk.add_tag(f'code_{language.lower()}', source='heuristic')
                chunk.metadata['programming_language'] = language
    
    def _analyze_general_content(self, chunk: SemanticChunk):
        """Analyze general content chunks"""
        chunk.add_tag('general_content', source='heuristic')
        self._analyze_latex_content_comprehensive(chunk)
    
    def _clean_identifier(self, text: str) -> str:
        """Clean text for use as identifier"""
        clean_text = re.sub(r'\\[a-zA-Z]+\{?', '', text)
        clean_text = re.sub(r'[^\w\s-]', '', clean_text)
        clean_text = re.sub(r'\s+', '_', clean_text)
        return clean_text.lower()[:50]
    
    def _post_process_chunks(self, chunks: List[SemanticChunk], content: str) -> List[SemanticChunk]:
        """Post-process chunks to enhance metadata and validate quality"""
        enhanced_chunks = []
        
        for chunk in chunks:
            # Validate chunk quality
            if self._is_valid_chunk(chunk):
                # Add final metadata
                chunk.metadata['total_content_length'] = len(content)
                chunk.metadata['chunk_position'] = len(enhanced_chunks) + 1
                
                # Calculate relative importance based on content
                chunk.importance_score = self._calculate_enhanced_importance(chunk)
                
                enhanced_chunks.append(chunk)
        
        # Add inter-chunk relationships
        self._add_chunk_relationships(enhanced_chunks)
        
        return enhanced_chunks
    
    def _is_valid_chunk(self, chunk: SemanticChunk) -> bool:
        """Validate if a chunk meets quality criteria"""
        # Always include certain types regardless of size
        latex_type = chunk.metadata.get('latex_type', '')
        if latex_type in ['preamble', 'environment']:
            return True
        
        # For other chunks, apply size and content filters
        if len(chunk.content.strip()) < 10:  # Minimum content requirement
            return False
        
        # Check for substantial content (not just whitespace and commands)
        content_without_commands = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})?', '', chunk.content)
        meaningful_content = re.sub(r'\s+', ' ', content_without_commands).strip()
        
        return len(meaningful_content) >= 5
    
    def _calculate_enhanced_importance(self, chunk: SemanticChunk) -> float:
        """Calculate enhanced importance score based on content analysis"""
        base_score = 0.5
        
        # Boost for academic content
        if any(tag.name.startswith('academic_') for tag in chunk.semantic_tags):
            base_score += 0.2
        
        # Boost for mathematical content
        if chunk.metadata.get('total_math_elements', 0) > 0:
            base_score += 0.1
        
        # Boost for structural elements
        latex_type = chunk.metadata.get('latex_type', '')
        if latex_type in ['preamble', 'section']:
            base_score += 0.2
        elif latex_type == 'environment':
            env_name = chunk.metadata.get('environment_name', '')
            if env_name in self.environment_categories['atomic_environments']:
                base_score += 0.15
        
        # Boost for citations and references
        if chunk.metadata.get('citation_count', 0) > 0:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _add_chunk_relationships(self, chunks: List[SemanticChunk]):
        """Add relationships between related chunks"""
        for i, chunk in enumerate(chunks):
            # Add sequential relationships
            if i > 0:
                chunk.dependencies.append(chunks[i-1].id)
            
            # Add section-subsection relationships
            if chunk.metadata.get('latex_type') == 'section':
                section_level = chunk.metadata.get('section_level', 0)
                # Find parent section
                for j in range(i-1, -1, -1):
                    other_chunk = chunks[j]
                    if (other_chunk.metadata.get('latex_type') == 'section' and
                        other_chunk.metadata.get('section_level', 0) < section_level):
                        chunk.dependencies.append(other_chunk.id)
                        break


# Configuration helper for easy setup
class LaTeXConfig:
    """Configuration helper for the enhanced LaTeX parser"""
    
    @staticmethod
    def apply_enhanced_config(config):
        """Apply enhanced LaTeX parsing configuration"""
        # Enhanced parsing settings
        config.latex_parsing_strategy = 'balanced'  # 'fine', 'balanced', 'coarse'
        config.latex_env_mode = 'comprehensive'     # 'minimal', 'comprehensive'
        config.latex_math_level = 'aggressive'      # 'basic', 'standard', 'aggressive'
        config.latex_preserve_all_environments = True
        config.latex_subsection_granularity = True
        config.latex_enhanced_math_detection = True
        
        return config
    
    @staticmethod
    def for_academic_papers(config):
        """Configuration optimized for academic papers"""
        config = LaTeXConfig.apply_enhanced_config(config)
        config.latex_parsing_strategy = 'balanced'
        config.latex_math_level = 'aggressive'
        config.latex_subsection_granularity = True
        config.min_chunk_size = 100
        return config
    
    @staticmethod
    def for_comprehensive_analysis(config):
        """Configuration for maximum detail extraction"""
        config = LaTeXConfig.apply_enhanced_config(config)
        config.latex_parsing_strategy = 'fine'
        config.latex_env_mode = 'comprehensive'
        config.latex_math_level = 'aggressive'
        config.min_chunk_size = 50
        return config


# Usage demonstration
def demonstrate_enhanced_parser():
    """
    Demonstration of the enhanced LaTeX parser improvements.
    
    This addresses the key issues from the demo:
    1. Environment detection: 25+ environments vs previous 1
    2. Mathematical content: 53+ elements vs previous 43
    3. Section handling: Configurable strategies for 29+ sections vs previous 11
    """
    
    print("Enhanced LaTeX Parser - Key Improvements")
    print("="*50)
    print("🔧 Environment Detection:")
    print("   • Comprehensive category-based extraction")
    print("   • Targets 25+ environments (vs previous 1)")
    print("   • Configurable granularity control")
    print()
    print("🧮 Mathematical Content:")
    print("   • Aggressive pattern matching")
    print("   • Targets 53+ math elements (vs previous 43)")
    print("   • Multi-level detection strategies")
    print()
    print("📚 Section Processing:")
    print("   • Enhanced granularity control")
    print("   • Better handling of 29+ sections (vs previous 11)")
    print("   • Configurable parsing strategies")
    print()
    print("⚙️  Configuration Options:")
    print("   • Fine-grained: Maximum detail extraction")
    print("   • Balanced: Good detail/performance balance")
    print("   • Coarse: Larger chunks for summarization")
    
    return "Enhanced parser ready for deployment!"


# Register the enhanced parser (replace the existing one)
# This would typically be done in the parser registry
# The existing HeuristicLaTeXParser is now enhanced with these improvements