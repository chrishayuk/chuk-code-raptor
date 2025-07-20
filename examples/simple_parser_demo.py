#!/usr/bin/env python3
# examples/simple_parser_demo.py
"""
Clean Chunking System Demo
==========================

Demo of the clean chunking system with tree-sitter and heuristic parsers.
Tests the new unified architecture.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Reduce logging noise
logging.basicConfig(level=logging.WARNING)

def print_banner():
    """Print demo banner"""
    print("=" * 70)
    print("🌟 CLEAN CHUNKING SYSTEM DEMO")
    print("   Modern Tree-sitter + Heuristic Parsing")
    print("=" * 70)
    print()

def get_sample_files():
    """Get paths to existing sample files"""
    script_dir = Path(__file__).parent
    samples_dir = script_dir / "samples"
    
    print(f"📁 Looking for samples in: {samples_dir}")
    
    sample_files = {
        'python': samples_dir / "sample.py",
        'markdown': samples_dir / "sample.md",
        'javascript': samples_dir / "sample.js", 
        'rust': samples_dir / "sample.rs",
        'json': samples_dir / "sample.json",
        'html': samples_dir / "sample.html"
    }
    
    existing_files = {}
    for language, file_path in sample_files.items():
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    existing_files[language] = file_path
                    print(f"✅ Found {language}: {file_path.name} ({len(content):,} chars)")
                else:
                    print(f"⚠️  Empty: {file_path.name}")
            except Exception as e:
                print(f"❌ Error reading {file_path.name}: {e}")
        else:
            print(f"❌ Missing: {file_path.name}")
    
    return existing_files

def test_parser(language: str, file_path: Path) -> Dict[str, Any]:
    """Test a specific parser with its sample file"""
    try:
        # Import the clean chunking system
        from chuk_code_raptor.chunking import ChunkingEngine, ChunkingConfig
        
        # Create engine and test if language is supported
        config = ChunkingConfig()
        engine = ChunkingEngine(config)
        
        if not engine.can_chunk_language(language):
            return {'error': f'No parser available for {language}', 'language': language}
        
        # Read content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get parser info
        parser = engine._get_parser(language)
        parser_type = getattr(parser, 'parser_type', 'unknown')
        
        # Parse content
        start_time = time.time()
        chunks = engine.chunk_content(content, language, str(file_path))
        end_time = time.time()
        
        # Analyze results
        processing_time = (end_time - start_time) * 1000
        
        chunk_types = {}
        total_deps = 0
        total_tags = 0
        semantic_insights = []
        
        for chunk in chunks:
            # Count chunk types
            chunk_type = chunk.chunk_type.value
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # Count dependencies
            if hasattr(chunk, 'dependencies') and chunk.dependencies:
                total_deps += len(chunk.dependencies)
            
            # Count semantic tags
            if hasattr(chunk, 'semantic_tags') and chunk.semantic_tags:
                tags = [tag.name for tag in chunk.semantic_tags]
                total_tags += len(tags)
                semantic_insights.extend(tags)
        
        return {
            'language': language,
            'parser_type': parser_type,
            'file_size': len(content),
            'line_count': len(content.split('\n')),
            'chunk_count': len(chunks),
            'processing_time': round(processing_time, 2),
            'chunk_types': chunk_types,
            'total_dependencies': total_deps,
            'total_semantic_tags': total_tags,
            'semantic_insights': list(set(semantic_insights))[:8],  # Unique insights
            'sample_chunks': [
                {
                    'id': chunk.id,
                    'type': chunk.chunk_type.value,
                    'lines': f"{chunk.start_line}-{chunk.end_line}",
                    'size': len(chunk.content),
                    'tags': [tag.name for tag in chunk.semantic_tags][:3] if hasattr(chunk, 'semantic_tags') else [],
                    'importance': getattr(chunk, 'importance_score', 0.0)
                }
                for chunk in chunks[:3]  # First 3 chunks
            ]
        }
        
    except Exception as e:
        return {'error': str(e), 'language': language}

def print_results(result: Dict[str, Any], verbose: bool = False):
    """Print test results for a language"""
    if 'error' in result:
        print(f"❌ {result['language'].upper()}: {result['error']}")
        return
    
    lang = result['language'].upper()
    parser_type = result.get('parser_type', 'unknown')
    
    print(f"✅ {lang} ({parser_type} parser):")
    print(f"   📄 File: {result['file_size']:,} chars, {result['line_count']} lines")
    print(f"   ⚡ Processing: {result['chunk_count']} chunks in {result['processing_time']}ms")
    print(f"   🔗 Dependencies: {result['total_dependencies']}")
    print(f"   🏷️  Semantic tags: {result['total_semantic_tags']}")
    
    # Show chunk types
    if result['chunk_types']:
        types_str = ", ".join([f"{count} {type_}" for type_, count in result['chunk_types'].items()])
        print(f"   📊 Types: {types_str}")
    
    # Show semantic insights
    if result['semantic_insights']:
        insights_str = ", ".join(result['semantic_insights'])
        print(f"   🧠 Insights: {insights_str}")
    
    # Show sample chunks if verbose
    if verbose and result.get('sample_chunks'):
        print(f"   📋 Sample chunks:")
        for chunk in result['sample_chunks']:
            importance = chunk.get('importance', 0.0)
            print(f"      • {chunk['type']}: {chunk['size']} chars (importance: {importance:.2f})")
            if chunk['tags']:
                print(f"        Tags: {', '.join(chunk['tags'])}")
    
    print()

def show_system_info():
    """Show information about the chunking system"""
    try:
        from chuk_code_raptor.chunking import get_parser_info, get_supported_languages
        from chuk_code_raptor.chunking.parsers import discover_available_parsers
        
        print("🔍 SYSTEM INFORMATION")
        print("-" * 30)
        
        # Show available parsers
        available_parsers = discover_available_parsers()
        print(f"Available parsers: {len(available_parsers)}")
        for language, info in available_parsers.items():
            print(f"  • {language}: {info['type']} parser")
        
        # Show supported languages
        supported = get_supported_languages()
        print(f"Supported languages: {', '.join(supported)}")
        
        print()
        
    except Exception as e:
        print(f"⚠️  Could not get system info: {e}")
        print()

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Clean chunking system demo")
    parser.add_argument('--language', '-l', help="Test specific language only")
    parser.add_argument('--verbose', '-v', action='store_true', help="Show detailed output")
    parser.add_argument('--info', '-i', action='store_true', help="Show system information")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.info:
        show_system_info()
    
    # Get sample files
    sample_files = get_sample_files()
    
    if not sample_files:
        print("❌ No sample files found!")
        print("   Make sure sample files exist in examples/samples/")
        return 1
    
    print(f"\n🧪 Found {len(sample_files)} sample files to test")
    print()
    
    # Test parsers
    languages_to_test = [args.language] if args.language else sample_files.keys()
    results = []
    
    for language in languages_to_test:
        if language not in sample_files:
            print(f"⚠️  No sample file for {language}")
            continue
        
        print(f"🔬 Testing {language.upper()} parser...")
        file_path = sample_files[language]
        result = test_parser(language, file_path)
        
        if result:
            results.append(result)
            print_results(result, args.verbose)
        else:
            print(f"❌ {language.upper()}: Test failed")
    
    # Summary
    if results:
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        print("=" * 50)
        print("📊 SUMMARY")
        print("=" * 50)
        print(f"✅ Successful: {len(successful)} parsers")
        print(f"❌ Failed: {len(failed)} parsers")
        
        if successful:
            total_chunks = sum(r['chunk_count'] for r in successful)
            total_size = sum(r['file_size'] for r in successful)
            total_time = sum(r['processing_time'] for r in successful)
            
            print(f"📈 Overall stats:")
            print(f"   Total chunks: {total_chunks:,}")
            print(f"   Total content: {total_size:,} characters")
            print(f"   Total time: {total_time:.1f}ms")
            if total_time > 0:
                print(f"   Speed: {total_size / (total_time / 1000):,.0f} chars/sec")
            
            # Show parser types used
            parser_types = {}
            for result in successful:
                ptype = result.get('parser_type', 'unknown')
                parser_types[ptype] = parser_types.get(ptype, 0) + 1
            
            if parser_types:
                types_str = ", ".join([f"{count} {ptype}" for ptype, count in parser_types.items()])
                print(f"   Parser types: {types_str}")
        
        if failed:
            print(f"\n❌ Failed parsers:")
            for result in failed:
                print(f"   {result['language']}: {result['error']}")
        
        print(f"\n🎉 Clean chunking system demo completed!")
        
        if len(successful) >= 2:
            print("🌟 Multiple parsers working - system is functional!")
        elif len(successful) >= 1:
            print("👍 At least one parser working - good foundation!")
        
        # Provide helpful tips
        if any('tree-sitter' in r.get('error', '') for r in failed):
            print("\n💡 Tip: Some parsers need tree-sitter libraries:")
            print("   pip install tree-sitter tree-sitter-python tree-sitter-markdown")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)