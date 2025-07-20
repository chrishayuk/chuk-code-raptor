#!/usr/bin/env python3
"""
Enhanced Debug Test
==================

Test the engine with more substantial content to ensure chunking works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_python_chunking():
    """Test Python chunking with real content"""
    print("🔬 Testing Python chunking with substantial content...")
    
    try:
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        config = ChunkingConfig()
        engine = ChunkingEngine(config)
        
        # More substantial Python test content
        test_content = '''
import os
import sys
from typing import List, Dict

class DataProcessor:
    """A class for processing data"""
    
    def __init__(self, name: str):
        self.name = name
        self.data = []
    
    def add_data(self, item):
        """Add an item to the data list"""
        self.data.append(item)
    
    def process(self) -> List[str]:
        """Process all data items"""
        results = []
        for item in self.data:
            if isinstance(item, str):
                results.append(item.upper())
        return results

def main():
    """Main function"""
    processor = DataProcessor("test")
    processor.add_data("hello")
    processor.add_data("world")
    
    results = processor.process()
    print(f"Processed: {results}")

if __name__ == "__main__":
    main()
'''
        
        print(f"📄 Test content: {len(test_content)} characters")
        print(f"📋 Supported languages: {engine.get_supported_languages()}")
        
        # Test chunking
        chunks = engine.chunk_content(test_content, 'python', 'test.py')
        print(f"✅ Chunking successful: {len(chunks)} chunks created")
        
        if chunks:
            print(f"\n📊 CHUNK DETAILS:")
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i+1}:")
                print(f"    Type: {chunk.chunk_type.value}")
                print(f"    Lines: {chunk.start_line}-{chunk.end_line}")
                print(f"    Size: {len(chunk.content)} chars")
                print(f"    Preview: {chunk.content[:100]}...")
                if hasattr(chunk, 'semantic_tags') and chunk.semantic_tags:
                    tags = [tag.name for tag in chunk.semantic_tags]
                    print(f"    Tags: {tags}")
                print()
        else:
            print("❌ No chunks created - this suggests an issue with the parsing logic")
            
            # Let's debug what's happening
            print("\n🔍 DEBUGGING:")
            parser = engine._get_parser('python')
            if parser:
                print(f"✅ Parser found: {parser.name}")
                print(f"📋 Parser can parse Python: {parser.can_parse('python', '.py')}")
                
                # Test parsing directly
                from chuk_code_raptor.chunking.base import ParseContext
                from chuk_code_raptor.chunking.semantic_chunk import ContentType
                
                context = ParseContext(
                    file_path='test.py',
                    language='python',
                    content_type=ContentType.CODE,
                    max_chunk_size=2000,
                    min_chunk_size=50
                )
                
                try:
                    direct_chunks = parser.parse(test_content, context)
                    print(f"✅ Direct parsing: {len(direct_chunks)} chunks")
                    
                    if direct_chunks:
                        print("✅ Parsing is working! Issue might be in engine layer")
                    else:
                        print("❌ Direct parsing also returns 0 chunks - issue in parser logic")
                        
                except Exception as e:
                    print(f"❌ Direct parsing failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("❌ No parser found for Python")
        
        return len(chunks) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_markdown_chunking():
    """Test Markdown chunking"""
    print("\n🔬 Testing Markdown chunking...")
    
    try:
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        config = ChunkingConfig()
        engine = ChunkingEngine(config)
        
        # Markdown test content
        test_content = '''
# Main Heading

This is a paragraph with some content.

## Sub Heading

Another paragraph here.

```python
def example():
    return "Hello World"
```

### Another Section

- Item 1
- Item 2
- Item 3

That's the end.
'''
        
        chunks = engine.chunk_content(test_content, 'markdown', 'test.md')
        print(f"✅ Markdown chunking: {len(chunks)} chunks created")
        
        if chunks:
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i+1}: {chunk.chunk_type.value} - {chunk.content[:50]}...")
        
        return len(chunks) > 0
        
    except Exception as e:
        print(f"❌ Markdown test failed: {e}")
        return False

def main():
    print("🚀 Enhanced Chunking Test")
    print("=" * 50)
    
    python_success = test_python_chunking()
    markdown_success = test_markdown_chunking()
    
    print("\n" + "=" * 50)
    print("📊 FINAL SUMMARY")
    print("=" * 50)
    
    if python_success:
        print("✅ Python chunking: WORKING")
    else:
        print("❌ Python chunking: NEEDS DEBUGGING")
    
    if markdown_success:
        print("✅ Markdown chunking: WORKING")
    else:
        print("❌ Markdown chunking: NEEDS DEBUGGING")
    
    if python_success or markdown_success:
        print("\n🎉 At least one parser is working! The architecture is sound.")
        print("Now let's run the main demo to see the full results...")
    else:
        print("\n💥 Need to debug the parsing logic.")

if __name__ == "__main__":
    main()