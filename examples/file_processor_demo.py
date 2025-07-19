#!/usr/bin/env python3
# examples/file_processor_demo.py
"""
Enhanced FileProcessor Example Script
====================================

Demonstrates how to use the modular FileProcessor system with YAML configuration,
specifically designed to work with the CodeRaptor samples directory structure.

This script shows:
1. Loading configuration from YAML
2. Setting up the FileProcessor with samples directory
3. Scanning directories for changes with smart filtering
4. Processing individual files with detailed analysis
5. Analyzing results and statistics with rich output
6. Demonstrating chunking and language detection
7. Performance metrics and timing

Usage:
    python file_processor_demo.py [directory_to_scan] [config_file]
    
Examples:
    # Process the entire samples directory
    python file_processor_demo.py ./samples
    
    # Process just the test project
    python file_processor_demo.py ./samples/test_project
    
    # Use specific config for Python-only processing
    python file_processor_demo.py ./samples ./configs/python-only.yaml
    
    # Use web development config
    python file_processor_demo.py ./samples/test_project/src/frontend ./configs/web-dev.yaml
    
    # Verbose mode with detailed output
    python file_processor_demo.py ./samples --verbose
"""

import sys
import logging
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
import re
from collections import defaultdict, Counter

# Import our modules (assuming they're in the same directory for this example)
from chuk_code_raptor.file_processor.config import ConfigLoader, ProcessorConfig

# Enhanced FileProcessor with chunking capabilities
class EnhancedFileProcessor:
    """Enhanced FileProcessor with chunking and detailed analysis"""
    
    def __init__(self, index_dir: Path, config: ProcessorConfig):
        self.index_dir = Path(index_dir)
        self.config = config
        self.stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'total_size': 0,
            'total_lines': 0,
            'languages': {},
            'extensions': {},
            'chunks_created': 0,
            'processing_time': 0,
            'errors': [],
            'largest_files': [],
            'most_complex_files': []
        }
        self.processed_files = []
        
        print(f"🚀 Initialized Enhanced FileProcessor")
        print(f"   Index directory: {self.index_dir}")
        print(f"   Supported languages: {len(config.languages)}")
        print(f"   Max file size: {config.processing['max_file_size_mb']} MB")
        print(f"   Ignore patterns: {len(config.ignore_patterns['files']) + len(config.ignore_patterns['directories'])}")
    
    def detect_changes(self, directory: Path) -> Tuple[List[Path], List[Path], List[Path]]:
        """Enhanced change detection with filtering and categorization"""
        print(f"\n🔍 Scanning directory: {directory}")
        start_time = time.time()
        
        all_files = []
        ignored_files = []
        
        # Walk through directory tree
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                if self._should_ignore(file_path):
                    ignored_files.append(file_path)
                else:
                    all_files.append(file_path)
        
        # Categorize by language for better overview
        by_language = defaultdict(list)
        for file_path in all_files:
            extension = file_path.suffix.lower()
            language = self.config.languages.get(extension, 'unknown')
            by_language[language].append(file_path)
        
        scan_time = time.time() - start_time
        
        print(f"   📊 Scan Results ({scan_time:.2f}s):")
        print(f"      Total files found: {len(all_files) + len(ignored_files)}")
        print(f"      Files to process: {len(all_files)}")
        print(f"      Files ignored: {len(ignored_files)}")
        
        print(f"   🗣️  Languages detected:")
        for language, files in sorted(by_language.items(), key=lambda x: len(x[1]), reverse=True):
            if language != 'unknown' or len(files) <= 5:  # Show unknown only if few files
                print(f"      {language}: {len(files)} files")
        
        # For demo purposes, we'll treat all files as "added"
        # In real implementation, this would check modification times, hashes, etc.
        added = all_files
        modified = []
        deleted = []
        
        return added, modified, deleted
    
    def _should_ignore(self, file_path: Path) -> bool:
        """Enhanced ignore logic with better pattern matching"""
        path_str = str(file_path)
        relative_path = file_path.name
        
        # Check if file is too large
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.processing['max_file_size_mb']:
                return True
        except:
            return True  # Can't read file stats
        
        # Check directory patterns
        for part in file_path.parts:
            if part.lower() in [p.lower() for p in self.config.ignore_patterns['directories']]:
                return True
        
        # Check file patterns
        for pattern in self.config.ignore_patterns['files']:
            if pattern.startswith('*.'):
                # Extension pattern
                if path_str.lower().endswith(pattern[1:].lower()):
                    return True
            elif pattern.startswith('*') and pattern.endswith('*'):
                # Contains pattern
                if pattern[1:-1].lower() in path_str.lower():
                    return True
            elif pattern.startswith('*'):
                # Ends with pattern
                if path_str.lower().endswith(pattern[1:].lower()):
                    return True
            elif pattern.endswith('*'):
                # Starts with pattern
                if path_str.lower().startswith(pattern[:-1].lower()):
                    return True
            else:
                # Exact match
                if pattern.lower() in path_str.lower():
                    return True
        
        return False
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Enhanced encoding detection"""
        # Try encodings in priority order
        for encoding in self.config.encoding['priority']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Try to read first KB
                return encoding
            except UnicodeDecodeError:
                continue
            except Exception:
                break
        
        return 'binary'
    
    def _analyze_content(self, content: str, language: str, file_path: Path) -> Dict[str, Any]:
        """Analyze file content and extract metadata"""
        analysis = {
            'line_count': content.count('\n') + 1,
            'char_count': len(content),
            'blank_lines': len([line for line in content.split('\n') if not line.strip()]),
            'comment_lines': 0,
            'functions': [],
            'classes': [],
            'imports': [],
            'complexity_score': 0
        }
        
        lines = content.split('\n')
        
        if language == 'python':
            analysis.update(self._analyze_python_content(lines))
        elif language in ['javascript', 'typescript']:
            analysis.update(self._analyze_js_content(lines))
        elif language == 'markdown':
            analysis.update(self._analyze_markdown_content(lines))
        elif language in ['yaml', 'yml']:
            analysis.update(self._analyze_yaml_content(lines))
        
        # Calculate complexity score
        analysis['complexity_score'] = (
            len(analysis['functions']) * 2 +
            len(analysis['classes']) * 3 +
            analysis['line_count'] * 0.1 +
            len(analysis['imports']) * 0.5
        )
        
        return analysis
    
    def _analyze_python_content(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze Python-specific content"""
        functions = []
        classes = []
        imports = []
        comment_lines = 0
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Count comment lines
            if stripped.startswith('#'):
                comment_lines += 1
            
            # Find functions
            if stripped.startswith('def '):
                match = re.match(r'def\s+(\w+)\s*\(', stripped)
                if match:
                    functions.append({
                        'name': match.group(1),
                        'line': i,
                        'type': 'function'
                    })
            
            # Find classes
            if stripped.startswith('class '):
                match = re.match(r'class\s+(\w+)', stripped)
                if match:
                    classes.append({
                        'name': match.group(1),
                        'line': i,
                        'type': 'class'
                    })
            
            # Find imports
            if stripped.startswith(('import ', 'from ')):
                imports.append({
                    'statement': stripped,
                    'line': i
                })
        
        return {
            'comment_lines': comment_lines,
            'functions': functions,
            'classes': classes,
            'imports': imports
        }
    
    def _analyze_js_content(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript content"""
        functions = []
        classes = []
        imports = []
        comment_lines = 0
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Count comment lines
            if stripped.startswith('//') or stripped.startswith('/*'):
                comment_lines += 1
            
            # Find functions (various patterns)
            function_patterns = [
                r'function\s+(\w+)\s*\(',
                r'const\s+(\w+)\s*=\s*\(',
                r'(\w+)\s*:\s*function\s*\(',
                r'(\w+)\s*=>\s*',
            ]
            
            for pattern in function_patterns:
                match = re.search(pattern, stripped)
                if match:
                    functions.append({
                        'name': match.group(1),
                        'line': i,
                        'type': 'function'
                    })
                    break
            
            # Find classes
            if 'class ' in stripped:
                match = re.search(r'class\s+(\w+)', stripped)
                if match:
                    classes.append({
                        'name': match.group(1),
                        'line': i,
                        'type': 'class'
                    })
            
            # Find imports
            if stripped.startswith(('import ', 'export ', 'require(')):
                imports.append({
                    'statement': stripped,
                    'line': i
                })
        
        return {
            'comment_lines': comment_lines,
            'functions': functions,
            'classes': classes,
            'imports': imports
        }
    
    def _analyze_markdown_content(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze Markdown content"""
        headings = []
        code_blocks = []
        links = []
        
        in_code_block = False
        current_code_block = None
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Find headings
            if stripped.startswith('#'):
                level = len(re.match(r'^#+', stripped).group())
                title = stripped[level:].strip()
                headings.append({
                    'title': title,
                    'level': level,
                    'line': i
                })
            
            # Find code blocks
            if stripped.startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    language = stripped[3:].strip()
                    current_code_block = {
                        'language': language,
                        'start_line': i,
                        'lines': []
                    }
                else:
                    in_code_block = False
                    if current_code_block:
                        current_code_block['end_line'] = i
                        current_code_block['line_count'] = len(current_code_block['lines'])
                        code_blocks.append(current_code_block)
                        current_code_block = None
            elif in_code_block and current_code_block:
                current_code_block['lines'].append(line)
            
            # Find links
            link_matches = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line)
            for text, url in link_matches:
                links.append({
                    'text': text,
                    'url': url,
                    'line': i
                })
        
        return {
            'headings': headings,
            'code_blocks': code_blocks,
            'links': links,
            'functions': headings,  # Treat headings as functions for consistency
            'classes': [],
            'imports': links  # Treat links as imports for consistency
        }
    
    def _analyze_yaml_content(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze YAML content"""
        top_level_keys = []
        comment_lines = 0
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if stripped.startswith('#'):
                comment_lines += 1
            elif ':' in stripped and not stripped.startswith(' '):
                key = stripped.split(':')[0].strip()
                top_level_keys.append({
                    'name': key,
                    'line': i,
                    'type': 'key'
                })
        
        return {
            'comment_lines': comment_lines,
            'top_level_keys': top_level_keys,
            'functions': top_level_keys,  # Treat keys as functions for consistency
            'classes': [],
            'imports': []
        }
    
    def _create_chunks(self, content: str, language: str, analysis: Dict[str, Any], file_path: Path) -> List[Dict[str, Any]]:
        """Create semantic chunks from file content"""
        chunks = []
        
        if language == 'python':
            chunks.extend(self._create_python_chunks(content, analysis, file_path))
        elif language in ['javascript', 'typescript']:
            chunks.extend(self._create_js_chunks(content, analysis, file_path))
        elif language == 'markdown':
            chunks.extend(self._create_markdown_chunks(content, analysis, file_path))
        else:
            # Generic chunking for other file types
            chunks.extend(self._create_generic_chunks(content, analysis, file_path))
        
        return chunks
    
    def _create_python_chunks(self, content: str, analysis: Dict[str, Any], file_path: Path) -> List[Dict[str, Any]]:
        """Create chunks for Python files"""
        chunks = []
        lines = content.split('\n')
        
        # Create chunks for each function and class
        for item in analysis['functions'] + analysis['classes']:
            start_line = item['line'] - 1
            
            # Find the end of the function/class
            end_line = len(lines)
            indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
            
            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if line.strip() and (len(line) - len(line.lstrip())) <= indent_level:
                    if not line.strip().startswith(('#', '"""', "'''")):
                        end_line = i
                        break
            
            chunk_content = '\n'.join(lines[start_line:end_line])
            chunks.append({
                'id': f"{file_path.name}:{item['type']}:{item['name']}:{start_line + 1}",
                'type': item['type'],
                'name': item['name'],
                'content': chunk_content,
                'start_line': start_line + 1,
                'end_line': end_line,
                'size': len(chunk_content),
                'language': 'python'
            })
        
        return chunks
    
    def _create_js_chunks(self, content: str, analysis: Dict[str, Any], file_path: Path) -> List[Dict[str, Any]]:
        """Create chunks for JavaScript/TypeScript files"""
        chunks = []
        lines = content.split('\n')
        
        # Similar to Python but with different syntax patterns
        for item in analysis['functions'] + analysis['classes']:
            start_line = item['line'] - 1
            end_line = len(lines)
            
            # Find matching braces for JS/TS
            brace_count = 0
            for i in range(start_line, len(lines)):
                line = lines[i]
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0 and i > start_line:
                    end_line = i + 1
                    break
            
            chunk_content = '\n'.join(lines[start_line:end_line])
            chunks.append({
                'id': f"{file_path.name}:{item['type']}:{item['name']}:{start_line + 1}",
                'type': item['type'],
                'name': item['name'],
                'content': chunk_content,
                'start_line': start_line + 1,
                'end_line': end_line,
                'size': len(chunk_content),
                'language': 'javascript'
            })
        
        return chunks
    
    def _create_markdown_chunks(self, content: str, analysis: Dict[str, Any], file_path: Path) -> List[Dict[str, Any]]:
        """Create chunks for Markdown files"""
        chunks = []
        lines = content.split('\n')
        
        # Create chunks for each heading section
        headings = analysis.get('headings', [])
        for i, heading in enumerate(headings):
            start_line = heading['line'] - 1
            
            # Find the next heading of same or higher level
            end_line = len(lines)
            for j in range(i + 1, len(headings)):
                next_heading = headings[j]
                if next_heading['level'] <= heading['level']:
                    end_line = next_heading['line'] - 1
                    break
            
            chunk_content = '\n'.join(lines[start_line:end_line])
            chunks.append({
                'id': f"{file_path.name}:section:{heading['title']}:{start_line + 1}",
                'type': 'section',
                'name': heading['title'],
                'content': chunk_content,
                'start_line': start_line + 1,
                'end_line': end_line,
                'size': len(chunk_content),
                'language': 'markdown',
                'level': heading['level']
            })
        
        return chunks
    
    def _create_generic_chunks(self, content: str, analysis: Dict[str, Any], file_path: Path) -> List[Dict[str, Any]]:
        """Create generic chunks for other file types"""
        chunks = []
        
        # Simple chunking by line count
        lines = content.split('\n')
        chunk_size = 50  # lines per chunk
        
        for i in range(0, len(lines), chunk_size):
            end_i = min(i + chunk_size, len(lines))
            chunk_content = '\n'.join(lines[i:end_i])
            
            chunks.append({
                'id': f"{file_path.name}:chunk:{i + 1}",
                'type': 'chunk',
                'name': f"Lines {i + 1}-{end_i}",
                'content': chunk_content,
                'start_line': i + 1,
                'end_line': end_i,
                'size': len(chunk_content),
                'language': 'unknown'
            })
        
        return chunks
    
    def process_file(self, file_path: Path, base_directory: Path) -> Optional[Dict[str, Any]]:
        """Enhanced file processing with detailed analysis"""
        start_time = time.time()
        
        try:
            # Get file info
            stats = file_path.stat()
            
            # Detect language
            extension = file_path.suffix.lower()
            language = self.config.languages.get(extension, 'unknown')
            
            # Detect encoding and read content
            encoding = self._detect_encoding(file_path)
            content = None
            content_hash = None
            
            if encoding != 'binary':
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                except Exception as e:
                    self.stats['errors'].append(f"Error reading {file_path}: {e}")
                    return None
            
            # Analyze content
            analysis = {}
            chunks = []
            
            if content:
                analysis = self._analyze_content(content, language, file_path)
                chunks = self._create_chunks(content, language, analysis, file_path)
            
            # Update statistics
            self.stats['files_processed'] += 1
            self.stats['total_size'] += stats.st_size
            self.stats['total_lines'] += analysis.get('line_count', 0)
            self.stats['languages'][language] = self.stats['languages'].get(language, 0) + 1
            self.stats['extensions'][extension] = self.stats['extensions'].get(extension, 0) + 1
            self.stats['chunks_created'] += len(chunks)
            
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            
            # Create file info object
            relative_path = str(file_path.relative_to(base_directory))
            
            file_info = {
                'path': str(file_path),
                'relative_path': relative_path,
                'language': language,
                'extension': extension,
                'file_size': stats.st_size,
                'encoding': encoding,
                'line_count': analysis.get('line_count', 0),
                'char_count': analysis.get('char_count', 0),
                'blank_lines': analysis.get('blank_lines', 0),
                'comment_lines': analysis.get('comment_lines', 0),
                'functions': analysis.get('functions', []),
                'classes': analysis.get('classes', []),
                'imports': analysis.get('imports', []),
                'complexity_score': analysis.get('complexity_score', 0),
                'chunks': chunks,
                'chunk_count': len(chunks),
                'content_hash': content_hash,
                'last_modified': datetime.fromtimestamp(stats.st_mtime),
                'processing_time': processing_time,
                'content_preview': content[:300] if content else None
            }
            
            # Track largest and most complex files
            self.stats['largest_files'].append((stats.st_size, relative_path))
            self.stats['largest_files'] = sorted(self.stats['largest_files'], reverse=True)[:10]
            
            complexity = analysis.get('complexity_score', 0)
            self.stats['most_complex_files'].append((complexity, relative_path))
            self.stats['most_complex_files'] = sorted(self.stats['most_complex_files'], reverse=True)[:10]
            
            self.processed_files.append(file_info)
            return file_info
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            self.stats['errors'].append(error_msg)
            print(f"   ❌ {error_msg}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return self.stats
    
    def get_processed_files(self) -> List[Dict[str, Any]]:
        """Get all processed files"""
        return self.processed_files

def setup_logging(verbose: bool = False):
    """Configure logging for the demo"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def print_file_info(file_info: Dict[str, Any], verbose: bool = False):
    """Pretty print file information with enhanced details"""
    if not file_info:
        return
    
    print(f"   📄 {file_info['relative_path']}")
    print(f"      Language: {file_info['language']} ({file_info['extension']})")
    print(f"      Size: {file_info['file_size']:,} bytes ({file_info['file_size']/(1024):.1f} KB)")
    print(f"      Lines: {file_info['line_count']:,} ({file_info['blank_lines']} blank, {file_info['comment_lines']} comments)")
    print(f"      Encoding: {file_info['encoding']}")
    print(f"      Complexity: {file_info['complexity_score']:.1f}")
    print(f"      Chunks: {file_info['chunk_count']}")
    print(f"      Processing time: {file_info['processing_time']:.3f}s")
    
    # Show functions/classes if any
    if file_info['functions']:
        func_names = [f['name'] for f in file_info['functions'][:3]]
        more = f" (+{len(file_info['functions']) - 3})" if len(file_info['functions']) > 3 else ""
        print(f"      Functions: {', '.join(func_names)}{more}")
    
    if file_info['classes']:
        class_names = [c['name'] for c in file_info['classes'][:3]]
        more = f" (+{len(file_info['classes']) - 3})" if len(file_info['classes']) > 3 else ""
        print(f"      Classes: {', '.join(class_names)}{more}")
    
    if verbose:
        print(f"      Modified: {file_info['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"      Hash: {file_info['content_hash'][:16]}..." if file_info['content_hash'] else "      Hash: None")
        
        if file_info['content_preview']:
            preview = file_info['content_preview'].replace('\n', '\\n')
            if len(preview) > 100:
                preview = preview[:100] + "..."
            print(f"      Preview: {preview}")
        
        if file_info['chunks']:
            print(f"      Chunk details:")
            for chunk in file_info['chunks'][:2]:  # Show first 2 chunks
                print(f"        - {chunk['type']}: {chunk['name']} (lines {chunk['start_line']}-{chunk['end_line']})")

def print_comprehensive_statistics(stats: Dict[str, Any], processed_files: List[Dict[str, Any]]):
    """Print comprehensive processing statistics"""
    print(f"\n📊 Comprehensive Processing Statistics")
    print("=" * 50)
    
    # Basic stats
    print(f"📈 Processing Summary:")
    print(f"   Files processed: {stats['files_processed']:,}")
    print(f"   Files skipped: {stats['files_skipped']:,}")
    print(f"   Total size: {stats['total_size'] / (1024*1024):.2f} MB")
    print(f"   Total lines: {stats['total_lines']:,}")
    print(f"   Total chunks: {stats['chunks_created']:,}")
    print(f"   Processing time: {stats['processing_time']:.2f}s")
    print(f"   Average time per file: {stats['processing_time']/max(stats['files_processed'], 1):.3f}s")
    print(f"   Errors: {len(stats['errors'])}")
    
    # Language distribution
    if stats['languages']:
        print(f"\n🗣️  Language Distribution:")
        sorted_languages = sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True)
        total_files = sum(stats['languages'].values())
        
        for language, count in sorted_languages:
            percentage = (count / total_files) * 100
            bar_length = int(percentage / 2)  # Scale bar to max 50 chars
            bar = "█" * bar_length + "░" * (25 - bar_length)
            print(f"   {language:12} │{bar}│ {count:3} files ({percentage:5.1f}%)")
    
    # Extension distribution
    if stats['extensions']:
        print(f"\n📁 File Extension Distribution:")
        sorted_extensions = sorted(stats['extensions'].items(), key=lambda x: x[1], reverse=True)
        
        for i, (extension, count) in enumerate(sorted_extensions[:10]):
            ext_display = extension if extension else '(no ext)'
            print(f"   {ext_display:8} : {count:3} files")
        
        if len(sorted_extensions) > 10:
            others = sum(count for _, count in sorted_extensions[10:])
            print(f"   Others   : {others:3} files")
    
    # Complexity analysis
    if processed_files:
        complexities = [f['complexity_score'] for f in processed_files if f['complexity_score'] > 0]
        if complexities:
            print(f"\n🧠 Complexity Analysis:")
            print(f"   Average complexity: {sum(complexities)/len(complexities):.1f}")
            print(f"   Max complexity: {max(complexities):.1f}")
            print(f"   Files with functions/classes: {len(complexities)}")
    
    # Largest files
    if stats['largest_files']:
        print(f"\n📏 Largest Files:")
        for size, path in stats['largest_files'][:5]:
            size_mb = size / (1024 * 1024)
            if size_mb >= 1:
                print(f"   {size_mb:6.2f} MB - {path}")
            else:
                print(f"   {size/1024:6.1f} KB - {path}")
    
    # Most complex files
    if stats['most_complex_files']:
        complex_files = [f for f in stats['most_complex_files'] if f[0] > 0]
        if complex_files:
            print(f"\n🔥 Most Complex Files:")
            for complexity, path in complex_files[:5]:
                print(f"   {complexity:6.1f} - {path}")
    
    # Chunk statistics
    if processed_files:
        chunk_types = Counter()
        chunk_sizes = []
        
        for file_info in processed_files:
            for chunk in file_info['chunks']:
                chunk_types[chunk['type']] += 1
                chunk_sizes.append(chunk['size'])
        
        if chunk_types:
            print(f"\n🧩 Chunk Type Distribution:")
            for chunk_type, count in chunk_types.most_common():
                print(f"   {chunk_type:10} : {count:3} chunks")
        
        if chunk_sizes:
            print(f"\n📏 Chunk Size Statistics:")
            print(f"   Average size: {sum(chunk_sizes)/len(chunk_sizes):.0f} chars")
            print(f"   Largest chunk: {max(chunk_sizes):,} chars")
            print(f"   Smallest chunk: {min(chunk_sizes):,} chars")
    
    # Error summary
    if stats['errors']:
        print(f"\n❌ Errors Encountered:")
        for i, error in enumerate(stats['errors'][:5]):
            print(f"   {i+1}. {error}")
        if len(stats['errors']) > 5:
            print(f"   ... and {len(stats['errors']) - 5} more errors")

def print_samples_analysis(processed_files: List[Dict[str, Any]]):
    """Print specific analysis for the samples directory structure"""
    print(f"\n🧪 Samples Directory Analysis")
    print("=" * 50)
    
    # Categorize by directory structure
    by_directory = defaultdict(list)
    by_sample_type = defaultdict(list)
    
    for file_info in processed_files:
        path_parts = Path(file_info['relative_path']).parts
        
        # Group by immediate parent directory
        if len(path_parts) > 1:
            parent = path_parts[0] if path_parts[0] != 'samples' else path_parts[1] if len(path_parts) > 1 else 'root'
            by_directory[parent].append(file_info)
        
        # Group by sample type (src/python, src/frontend, docs, etc.)
        if 'src/python' in file_info['relative_path']:
            by_sample_type['Python Backend'].append(file_info)
        elif 'src/frontend' in file_info['relative_path']:
            by_sample_type['Frontend (React/TS)'].append(file_info)
        elif 'docs/' in file_info['relative_path']:
            by_sample_type['Documentation'].append(file_info)
        elif 'config/' in file_info['relative_path']:
            by_sample_type['Configuration'].append(file_info)
        elif 'tests/' in file_info['relative_path']:
            by_sample_type['Tests'].append(file_info)
        elif 'scripts/' in file_info['relative_path']:
            by_sample_type['Scripts'].append(file_info)
        else:
            by_sample_type['Other'].append(file_info)
    
    # Directory breakdown
    print(f"📁 Directory Structure Analysis:")
    for directory, files in sorted(by_directory.items()):
        total_size = sum(f['file_size'] for f in files)
        total_lines = sum(f['line_count'] for f in files)
        total_chunks = sum(f['chunk_count'] for f in files)
        
        print(f"   {directory:15} │ {len(files):3} files │ {total_size/1024:7.1f} KB │ {total_lines:5} lines │ {total_chunks:3} chunks")
    
    # Sample type analysis
    print(f"\n🎯 Sample Type Analysis:")
    for sample_type, files in sorted(by_sample_type.items()):
        if not files:
            continue
            
        languages = Counter(f['language'] for f in files)
        total_functions = sum(len(f['functions']) for f in files)
        total_classes = sum(len(f['classes']) for f in files)
        avg_complexity = sum(f['complexity_score'] for f in files) / len(files) if files else 0
        
        print(f"\n   📂 {sample_type}:")
        print(f"      Files: {len(files)}")
        print(f"      Languages: {', '.join(f'{lang}({count})' for lang, count in languages.most_common(3))}")
        print(f"      Functions: {total_functions}, Classes: {total_classes}")
        print(f"      Avg Complexity: {avg_complexity:.1f}")
        
        # Show most interesting files
        interesting_files = sorted(files, key=lambda f: f['complexity_score'], reverse=True)[:2]
        for file_info in interesting_files:
            if file_info['complexity_score'] > 0:
                print(f"         - {Path(file_info['relative_path']).name} (complexity: {file_info['complexity_score']:.1f})")

def create_default_config() -> str:
    """Create a default configuration file for the demo"""
    config_content = '''# CodeRaptor FileProcessor Configuration
# Demo configuration for samples directory

# Language mappings
languages:
  # Python
  .py: python
  .pyx: python
  .pyi: python
  
  # JavaScript/TypeScript
  .js: javascript
  .jsx: javascript
  .ts: typescript
  .tsx: typescript
  .mjs: javascript
  .cjs: javascript
  
  # Web
  .html: html
  .htm: html
  .css: css
  .scss: css
  .sass: css
  .less: css
  
  # Documentation
  .md: markdown
  .rst: rst
  .txt: text
  
  # Configuration
  .yaml: yaml
  .yml: yaml
  .json: json
  .toml: toml
  .ini: ini
  .cfg: config
  
  # Shell/Scripts
  .sh: shell
  .bash: shell
  .zsh: shell
  .fish: shell
  .bat: batch
  .cmd: batch
  .ps1: powershell
  
  # Other
  .xml: xml
  .csv: csv
  .sql: sql
  .dockerfile: dockerfile
  .gitignore: gitignore
  .env: env

# Processing configuration
processing:
  max_file_size_mb: 10
  enable_content_detection: true
  chunk_size_lines: 50
  overlap_lines: 5

# Encoding detection
encoding:
  priority:
    - utf-8
    - utf-16
    - latin1
    - ascii

# Files and directories to ignore
ignore_patterns:
  directories:
    - __pycache__
    - .git
    - .svn
    - .hg
    - node_modules
    - .venv
    - venv
    - env
    - .pytest_cache
    - .mypy_cache
    - .tox
    - build
    - dist
    - .egg-info
    - .idea
    - .vscode
    - .DS_Store
    - target
    - bin
    - obj
  
  files:
    - "*.pyc"
    - "*.pyo"
    - "*.pyd"
    - "*.so"
    - "*.dylib"
    - "*.dll"
    - "*.exe"
    - "*.o"
    - "*.a"
    - "*.lib"
    - "*.log"
    - "*.tmp"
    - "*.temp"
    - "*.swp"
    - "*.swo"
    - "*~"
    - ".gitignore"
    - ".gitattributes"
    - "*.min.js"
    - "*.min.css"
    - "package-lock.json"
    - "yarn.lock"
    - "Pipfile.lock"
'''
    
    config_path = Path("demo_config.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return str(config_path)

def main():
    """Enhanced main demo function"""
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    setup_logging(verbose)
    
    print("=" * 70)
    print("🦖 Enhanced CodeRaptor FileProcessor Demo")
    print("   Designed for CodeRaptor Samples Directory")
    print("=" * 70)
    
    # Parse command line arguments
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    
    if len(args) > 0:
        directory_to_scan = Path(args[0])
    else:
        # Look for samples directory
        possible_dirs = [
            Path("./samples"),
            Path("../samples"),
            Path("./samples/test_project"),
            Path(".")
        ]
        directory_to_scan = None
        for dir_path in possible_dirs:
            if dir_path.exists() and dir_path.is_dir():
                directory_to_scan = dir_path
                break
        
        if not directory_to_scan:
            print("❌ No samples directory found!")
            print("   Please run the create_test_directory.py script first")
            print("   Or specify a directory to scan")
            print("\nUsage:")
            print("   python file_processor_demo.py [directory] [config_file] [--verbose]")
            return 1
    
    if len(args) > 1:
        config_file = Path(args[1])
    else:
        # Look for config file or create default
        possible_configs = [
            Path("demo_config.yaml"),
            Path("file_processor_config.yaml"),
            Path("configs/default.yaml"),
            Path("../configs/default.yaml")
        ]
        
        config_file = None
        for config_path in possible_configs:
            if config_path.exists():
                config_file = config_path
                break
        
        if not config_file:
            print("🔧 No configuration file found, creating default...")
            config_file = Path(create_default_config())
            print(f"   Created: {config_file}")
    
    # Validate inputs
    if not directory_to_scan.exists():
        print(f"❌ Directory not found: {directory_to_scan}")
        return 1
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return 1
    
    print(f"📁 Scanning directory: {directory_to_scan.absolute()}")
    print(f"⚙️  Using config: {config_file.absolute()}")
    print(f"🔍 Verbose mode: {'ON' if verbose else 'OFF'}")
    
    try:
        # Load configuration
        print(f"\n🔧 Loading configuration...")
        config = ConfigLoader.load_config(config_file)
        print(f"   ✅ Loaded {len(config.languages)} language mappings")
        print(f"   ✅ Configured {len(config.ignore_patterns['directories'])} directory ignore patterns")
        print(f"   ✅ Configured {len(config.ignore_patterns['files'])} file ignore patterns")
        
        # Initialize FileProcessor
        print(f"\n🚀 Initializing Enhanced FileProcessor...")
        index_dir = Path("./demo_index")
        processor = EnhancedFileProcessor(index_dir, config)
        
        # Detect changes
        total_start_time = time.time()
        added, modified, deleted = processor.detect_changes(directory_to_scan)
        
        if not added and not modified:
            print("\n✅ No files to process!")
            return 0
        
        # Process files
        print(f"\n⚡ Processing files...")
        files_to_process = added + modified
        
        # Limit files for demo if too many
        max_files = 100 if not verbose else 50
        if len(files_to_process) > max_files:
            print(f"   📊 Found {len(files_to_process)} files, limiting to {max_files} for demo")
            files_to_process = files_to_process[:max_files]
            processor.stats['files_skipped'] = len(added + modified) - max_files
        
        processed_count = 0
        for i, file_path in enumerate(files_to_process):
            if i % 10 == 0 or verbose:
                print(f"   Processing {i+1}/{len(files_to_process)}: {file_path.name}")
            elif i % 50 == 0:
                print(f"   ... processed {i} files")
            
            file_info = processor.process_file(file_path, directory_to_scan)
            if file_info:
                processed_count += 1
        
        total_time = time.time() - total_start_time
        
        # Get results
        processed_files = processor.get_processed_files()
        stats = processor.get_stats()
        
        print(f"\n✅ Processing completed in {total_time:.2f}s!")
        print(f"   Successfully processed: {processed_count}/{len(files_to_process)} files")
        
        # Show detailed results
        if verbose and processed_files:
            print(f"\n📋 Detailed File Analysis (first 10):")
            for file_info in processed_files[:10]:
                print_file_info(file_info, verbose=True)
                print()
        
        # Show comprehensive statistics
        print_comprehensive_statistics(stats, processed_files)
        
        # Show samples-specific analysis
        if 'samples' in str(directory_to_scan) or 'test_project' in str(directory_to_scan):
            print_samples_analysis(processed_files)
        
        # Show configuration summary
        print(f"\n⚙️  Configuration Summary:")
        print(f"   Max file size: {config.processing['max_file_size_mb']} MB")
        print(f"   Encoding priority: {' -> '.join(config.encoding['priority'][:3])}...")
        print(f"   Content detection: {config.processing.get('enable_content_detection', False)}")
        print(f"   Total language mappings: {len(config.languages)}")
        
        # Suggest next steps
        print(f"\n🎯 Next Steps:")
        print(f"   1. Try processing with different configs:")
        print(f"      python {sys.argv[0]} {directory_to_scan} --verbose")
        print(f"   2. Create custom config for specific languages")
        print(f"   3. Process different directories:")
        print(f"      python {sys.argv[0]} ./samples/test_project/src/python")
        print(f"   4. Analyze the generated chunks and statistics")
        
        print(f"\n🦖 Demo completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)