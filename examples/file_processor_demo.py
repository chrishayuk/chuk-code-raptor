#!/usr/bin/env python3
# examples/file_processor_demo.py
"""
FileProcessor Example Script
============================

Demonstrates how to use the modular FileProcessor system with YAML configuration.

This script shows:
1. Loading configuration from YAML
2. Setting up the FileProcessor
3. Scanning directories for changes
4. Processing individual files
5. Analyzing results and statistics

Usage:
    python file_processor_demo.py [directory_to_scan] [config_file]
    
Examples:
    python file_processor_demo.py ./my_project
    python file_processor_demo.py ./src ./configs/python-only.yaml
    python file_processor_demo.py . ./configs/web-dev.yaml
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Import our modules (assuming they're in the same directory for this example)
from chuk_code_raptor.file_processor.config import ConfigLoader, ProcessorConfig

# Mock FileProcessor class since we don't have all modules yet
class FileProcessor:
    """Mock FileProcessor for demonstration"""
    
    def __init__(self, index_dir: Path, config: ProcessorConfig):
        self.index_dir = Path(index_dir)
        self.config = config
        self.stats = {
            'files_processed': 0,
            'total_size': 0,
            'languages': {},
            'errors': []
        }
        
        print(f"🚀 Initialized FileProcessor")
        print(f"   Index directory: {self.index_dir}")
        print(f"   Supported languages: {len(config.languages)}")
        print(f"   Max file size: {config.processing['max_file_size_mb']} MB")
    
    def detect_changes(self, directory: Path):
        """Mock change detection"""
        print(f"\n🔍 Scanning directory: {directory}")
        
        files = list(directory.rglob('*'))
        files = [f for f in files if f.is_file() and not self._should_ignore(f)]
        
        # Mock: assume all files are "added" for first run
        added = files[:10]  # Limit for demo
        modified = []
        deleted = []
        
        print(f"   Found {len(files)} total files")
        print(f"   Changes: {len(added)} added, {len(modified)} modified, {len(deleted)} deleted")
        
        return added, modified, deleted
    
    def _should_ignore(self, file_path: Path):
        """Mock ignore logic"""
        path_str = str(file_path).lower()
        
        # Check directories
        for part in file_path.parts:
            if part.lower() in self.config.ignore_patterns['directories']:
                return True
        
        # Check file patterns
        for pattern in self.config.ignore_patterns['files']:
            if pattern.startswith('*.'):
                if path_str.endswith(pattern[1:]):
                    return True
            elif pattern in path_str:
                return True
        
        return False
    
    def process_file(self, file_path: Path, base_directory: Path):
        """Mock file processing"""
        try:
            # Get file info
            stats = file_path.stat()
            
            # Detect language
            extension = file_path.suffix.lower()
            language = self.config.languages.get(extension, 'unknown')
            
            # Mock encoding detection
            encoding = 'utf-8'
            line_count = 0
            
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    line_count = content.count('\n') + 1
            except:
                encoding = 'binary'
                content = None
            
            # Update stats
            self.stats['files_processed'] += 1
            self.stats['total_size'] += stats.st_size
            self.stats['languages'][language] = self.stats['languages'].get(language, 0) + 1
            
            # Create mock FileInfo
            relative_path = str(file_path.relative_to(base_directory))
            
            return {
                'path': str(file_path),
                'relative_path': relative_path,
                'language': language,
                'file_size': stats.st_size,
                'encoding': encoding,
                'line_count': line_count,
                'last_modified': datetime.fromtimestamp(stats.st_mtime),
                'content_preview': content[:200] if content else None
            }
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            self.stats['errors'].append(error_msg)
            print(f"   ❌ {error_msg}")
            return None
    
    def get_stats(self):
        """Get processing statistics"""
        return self.stats

def setup_logging():
    """Configure logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def print_file_info(file_info, verbose=False):
    """Pretty print file information"""
    if not file_info:
        return
    
    print(f"   📄 {file_info['relative_path']}")
    print(f"      Language: {file_info['language']}")
    print(f"      Size: {file_info['file_size']:,} bytes")
    print(f"      Lines: {file_info['line_count']:,}")
    print(f"      Encoding: {file_info['encoding']}")
    print(f"      Modified: {file_info['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    if verbose and file_info['content_preview']:
        preview = file_info['content_preview'].replace('\n', '\\n')
        print(f"      Preview: {preview}...")

def print_statistics(stats):
    """Pretty print processing statistics"""
    print(f"\n📊 Processing Statistics")
    print(f"   Files processed: {stats['files_processed']:,}")
    print(f"   Total size: {stats['total_size'] / (1024*1024):.2f} MB")
    print(f"   Errors: {len(stats['errors'])}")
    
    if stats['languages']:
        print(f"\n   Languages found:")
        sorted_languages = sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True)
        for language, count in sorted_languages:
            percentage = (count / stats['files_processed']) * 100
            print(f"      {language}: {count} files ({percentage:.1f}%)")
    
    if stats['errors']:
        print(f"\n   Errors encountered:")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"      ❌ {error}")
        if len(stats['errors']) > 5:
            print(f"      ... and {len(stats['errors']) - 5} more errors")

def main():
    """Main demo function"""
    setup_logging()
    
    print("=" * 60)
    print("🦖 CodeRaptor FileProcessor Demo")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        directory_to_scan = Path(sys.argv[1])
    else:
        directory_to_scan = Path(".")
    
    if len(sys.argv) > 2:
        config_file = Path(sys.argv[2])
    else:
        # Look for config file in common locations
        possible_configs = [
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
            print("❌ No configuration file found!")
            print("   Please provide a config file or create file_processor_config.yaml")
            print("\nUsage:")
            print("   python example_file_processor.py [directory] [config_file]")
            return 1
    
    # Validate inputs
    if not directory_to_scan.exists():
        print(f"❌ Directory not found: {directory_to_scan}")
        return 1
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return 1
    
    print(f"📁 Scanning directory: {directory_to_scan.absolute()}")
    print(f"⚙️  Using config: {config_file.absolute()}")
    
    try:
        # Load configuration
        print(f"\n🔧 Loading configuration...")
        config = ConfigLoader.load_config(config_file)
        print(f"   ✅ Loaded {len(config.languages)} language mappings")
        print(f"   ✅ Configured {len(config.ignore_patterns['directories'])} ignore patterns")
        
        # Initialize FileProcessor
        print(f"\n🚀 Initializing FileProcessor...")
        index_dir = Path("./demo_index")
        processor = FileProcessor(index_dir, config)
        
        # Detect changes
        added, modified, deleted = processor.detect_changes(directory_to_scan)
        
        if not added and not modified:
            print("\n✅ No files to process!")
            return 0
        
        # Process files
        print(f"\n⚡ Processing files...")
        processed_files = []
        
        for i, file_path in enumerate(added + modified):
            if i >= 20:  # Limit for demo
                print(f"   ... limiting to first 20 files for demo")
                break
            
            print(f"   Processing {i+1}/{min(20, len(added + modified))}: {file_path.name}")
            file_info = processor.process_file(file_path, directory_to_scan)
            
            if file_info:
                processed_files.append(file_info)
        
        # Show results
        print(f"\n📋 Processed Files Summary:")
        for file_info in processed_files[:5]:  # Show first 5 in detail
            print_file_info(file_info, verbose=True)
            print()
        
        if len(processed_files) > 5:
            print(f"   ... and {len(processed_files) - 5} more files")
        
        # Show statistics
        stats = processor.get_stats()
        print_statistics(stats)
        
        # Show configuration summary
        print(f"\n⚙️  Configuration Summary:")
        print(f"   Max file size: {config.processing['max_file_size_mb']} MB")
        print(f"   Encoding priority: {config.encoding['priority']}")
        print(f"   Content detection: {config.processing.get('enable_content_detection', False)}")
        
        # Show top language mappings
        print(f"\n🗣️  Top Language Mappings:")
        lang_items = list(config.languages.items())
        for ext, lang in lang_items[:10]:
            print(f"   {ext} -> {lang}")
        if len(lang_items) > 10:
            print(f"   ... and {len(lang_items) - 10} more mappings")
        
        # Show ignore patterns
        print(f"\n🚫 Ignore Patterns:")
        print(f"   Directories: {config.ignore_patterns['directories'][:5]}")
        print(f"   Files: {config.ignore_patterns['files'][:5]}")
        
        print(f"\n✅ Demo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)