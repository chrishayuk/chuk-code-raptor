#!/usr/bin/env python3
# examples/semantic_tree_sitter_demo.py
"""
Semantic Tree-sitter Chunking Demo
==================================

Demonstrates the semantic tree-sitter chunking system with full semantic understanding.
Shows how SemanticChunk objects provide rich context and relationships.

Requirements:
    pip install tree-sitter tree-sitter-python pyyaml

Usage:
    python semantic_tree_sitter_demo.py
"""

import sys
from pathlib import Path
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import our chunking system - using available imports
from chuk_code_raptor.chunking import (
    ChunkingEngine,
    ChunkingConfig,
    ChunkingStrategy,
    get_supported_languages,
    get_supported_extensions,
    PRECISE_CONFIG,
    DEFAULT_CONFIG
)

# Try to import tree-sitter chunker - handle if not available
try:
    from chuk_code_raptor.chunking.tree_sitter_chunker import (
        PythonSemanticChunker,
        TreeSitterConfigLoader,
        load_language_config
    )
    TREE_SITTER_AVAILABLE = True
except ImportError as e:
    print(f"Tree-sitter chunker not available: {e}")
    TREE_SITTER_AVAILABLE = False

# Import semantic chunks
from chuk_code_raptor.chunking.semantic_chunk import SemanticChunk, SemanticCodeChunk, ContentType
from chuk_code_raptor.core.models import ChunkType

def print_banner():
    """Print demo banner"""
    print("=" * 80)
    print("🧠 SEMANTIC TREE-SITTER CHUNKING DEMO")
    print("   Intelligent AST-based Code Understanding with Full Semantic Context")
    print("=" * 80)
    print()

def demonstrate_config_loading():
    """Demonstrate YAML configuration loading"""
    if not TREE_SITTER_AVAILABLE:
        print("⏭️  Skipping config demo - tree-sitter not available")
        return
        
    print("📋 CONFIGURATION LOADING DEMO")
    print("-" * 50)
    
    try:
        # Load Python configuration
        config_loader = TreeSitterConfigLoader()
        python_config = load_language_config('python')
        
        print(f"✅ Loaded Python semantic configuration:")
        print(f"   Language: {python_config.language_name}")
        print(f"   Extensions: {python_config.file_extensions}")
        print(f"   Chunk types: {len(python_config.chunk_node_types)}")
        print(f"   Atomic types: {len(python_config.atomic_node_types)}")
        print(f"   Importance weights: {len(python_config.importance_weights)}")
        
        # Show some specific mappings
        print(f"\n   Key semantic chunk mappings:")
        for node_type, chunk_type in list(python_config.chunk_node_types.items())[:5]:
            print(f"     {node_type} -> {chunk_type.value}")
        
        # Validate configuration
        issues = config_loader.validate_config('python')
        if issues:
            print(f"\n   ⚠️  Configuration issues:")
            for issue in issues:
                print(f"     - {issue}")
        else:
            print(f"\n   ✅ Semantic configuration is valid!")
            
    except Exception as e:
        print(f"❌ Error loading config: {e}")
    
    print()

def create_test_python_code() -> str:
    """Create comprehensive Python test code with semantic richness"""
    return '''
import os
import sys
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class Config:
    """Configuration class with dataclass decorator."""
    name: str
    debug: bool = False
    options: Dict[str, any] = field(default_factory=dict)

class BaseProcessor(ABC):
    """Abstract base class demonstrating inheritance."""
    
    def __init__(self, config: Config):
        self.config = config
        self._initialized = True
    
    @abstractmethod
    async def process(self, data: List[Dict]) -> Optional[Dict]:
        """Abstract method that must be implemented."""
        pass
    
    @property
    def is_ready(self) -> bool:
        """Property with type hint."""
        return self._initialized

class DataProcessor(BaseProcessor):
    """Concrete implementation of BaseProcessor."""
    
    def __init__(self, config: Config, batch_size: int = 100):
        super().__init__(config)
        self.batch_size = batch_size
        self.processed_count = 0
    
    async def process(self, data: List[Dict]) -> Optional[Dict]:
        """Process data in batches with async support."""
        if not data:
            return None
        
        results = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_result = await self._process_batch(batch)
            results.extend(batch_result)
            self.processed_count += len(batch)
        
        return {
            'processed_items': len(results),
            'total_processed': self.processed_count,
            'results': results
        }
    
    async def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Private method for batch processing."""
        processed = []
        for item in batch:
            processed_item = {
                **item,
                'processed': True,
                'timestamp': 12345
            }
            processed.append(processed_item)
        
        return processed

def create_processor(config_path: str, **kwargs):
    """Factory function for creating processors."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return DataProcessor(Config(name="test"), **kwargs)

if __name__ == "__main__":
    processor = create_processor('config.json', batch_size=25)
    print("Demo completed")
'''

def demonstrate_semantic_chunking_engine():
    """Demonstrate the semantic ChunkingEngine interface"""
    print("🚀 SEMANTIC CHUNKING ENGINE DEMO")
    print("-" * 50)
    
    # Show supported languages and extensions
    languages = get_supported_languages()
    extensions = get_supported_extensions()
    
    print(f"✅ Semantic chunking system initialized")
    print(f"   Supported languages: {languages}")
    print(f"   Supported extensions: {extensions}")
    
    # Create chunking engine with semantic-aware configs
    print(f"\n📋 Testing semantic configurations:")
    
    configs = [
        ("Default Semantic", DEFAULT_CONFIG),
        ("Precise Semantic", PRECISE_CONFIG),
    ]
    
    python_code = create_test_python_code()
    
    for config_name, config in configs:
        print(f"\n   {config_name} Configuration:")
        print(f"     Target size: {config.target_chunk_size}")
        print(f"     Min size: {config.min_chunk_size}")
        print(f"     Max size: {config.max_chunk_size}")
        print(f"     Strategy: {config.primary_strategy.value}")
        print(f"     Dependency tracking: {config.enable_dependency_tracking}")
        print(f"     Semantic analysis: {config.enable_similarity_analysis}")
        
        # Create engine and chunk content
        engine = ChunkingEngine(config)
        chunks = engine.chunk_content(python_code, 'python', 'test_module.py')
        
        # Analyze semantic chunks
        semantic_chunks = [c for c in chunks if isinstance(c, SemanticChunk)]
        code_chunks = [c for c in chunks if isinstance(c, SemanticCodeChunk)]
        
        # Show results
        avg_size = sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0
        
        print(f"     Results: {len(chunks)} chunks ({len(semantic_chunks)} semantic, {len(code_chunks)} code)")
        print(f"     Average size: {avg_size:.0f} chars")
        print(f"     Semantic features: dependencies, tags, relationships")
    
    print()

def demonstrate_semantic_python_chunking():
    """Demonstrate Python semantic tree-sitter chunking"""
    if not TREE_SITTER_AVAILABLE:
        print("⏭️  Skipping Python semantic chunking demo - tree-sitter not available")
        return
        
    print("🐍 PYTHON SEMANTIC CHUNKING DEMO")
    print("-" * 50)
    
    try:
        # Create semantic chunking configuration
        config = ChunkingConfig(
            target_chunk_size=800,
            min_chunk_size=50,
            max_chunk_size=2000,
            preserve_atomic_nodes=True,
            enable_dependency_tracking=True,
            enable_similarity_analysis=True,
            enable_contextual_linking=True
        )
        
        # Initialize Python semantic chunker
        chunker = PythonSemanticChunker(config)
        print(f"✅ Initialized Python semantic chunker")
        print(f"   Supported languages: {chunker.supported_languages}")
        print(f"   Supported extensions: {chunker.supported_extensions}")
        print(f"   Parser ready: {chunker.parser is not None}")
        print(f"   Language loaded: {chunker.language is not None}")
        print(f"   Semantic analysis: Enabled")
        
        # Test Python code
        python_code = create_test_python_code()
        
        print(f"\n📝 Test code stats:")
        print(f"   Lines: {len(python_code.split())}")
        print(f"   Characters: {len(python_code):,}")
        
        # Perform semantic chunking
        print(f"\n🔄 Semantic chunking Python code...")
        chunks = chunker.chunk_content(python_code, 'python', 'test_module.py')
        
        print(f"✅ Semantic chunking completed!")
        print(f"   Created {len(chunks)} semantic chunks")
        
        # Analyze semantic chunks
        chunk_types = {}
        total_size = 0
        total_dependencies = 0
        total_tags = 0
        total_relationships = 0
        
        for chunk in chunks:
            chunk_type = chunk.chunk_type.value
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            total_size += len(chunk.content)
            total_dependencies += len(chunk.dependencies)
            total_tags += len(chunk.semantic_tags)
            total_relationships += len(chunk.relationships)
        
        print(f"\n📊 Semantic Analysis:")
        print(f"   Total content covered: {total_size:,} characters")
        print(f"   Coverage: {(total_size/len(python_code)*100):.1f}%")
        print(f"   Total dependencies: {total_dependencies}")
        print(f"   Total semantic tags: {total_tags}")
        print(f"   Total relationships: {total_relationships}")
        
        print(f"\n   Chunk types distribution:")
        for chunk_type, count in sorted(chunk_types.items()):
            print(f"     {chunk_type}: {count}")
        
        # Show detailed semantic chunk information
        print(f"\n📋 Detailed Semantic Chunk Information:")
        print("-" * 70)
        
        for i, chunk in enumerate(chunks[:5], 1):  # Show first 5 chunks
            print(f"{i:2d}. {chunk.chunk_type.value.upper()}: {chunk.id}")
            print(f"    Type: {type(chunk).__name__}")
            print(f"    Lines: {chunk.start_line}-{chunk.end_line} ({chunk.line_count} lines)")
            print(f"    Size: {len(chunk.content):,} chars ({chunk.word_count} words)")
            print(f"    Content Type: {chunk.content_type.value}")
            
            # Show semantic tags
            if chunk.semantic_tags:
                tag_names = [tag.name for tag in chunk.semantic_tags]
                print(f"    Semantic Tags: {', '.join(tag_names)}")
            
            # Show dependencies
            if chunk.dependencies:
                print(f"    Dependencies: {len(chunk.dependencies)}")
            
            # Show relationships
            if chunk.relationships:
                print(f"    Relationships: {len(chunk.relationships)}")
            
            # Show semantic code chunk specific features
            if isinstance(chunk, SemanticCodeChunk):
                print(f"    Accessibility: {chunk.accessibility}")
                if chunk.imports:
                    print(f"    Imports: {', '.join(chunk.imports[:3])}")
                if chunk.function_calls:
                    print(f"    Function Calls: {', '.join(chunk.function_calls[:3])}")
            
            print()

        print(f"\n🎯 Semantic Features Demonstrated:")
        print(f"   ✅ Full semantic understanding with SemanticChunk objects")
        print(f"   ✅ Dependency tracking and relationship mapping")
        print(f"   ✅ Semantic tag extraction and categorization")
        print(f"   ✅ Python-specific pattern recognition")
        print(f"   ✅ AST-based parsing with semantic enhancement")
        
    except Exception as e:
        print(f"❌ Error in semantic chunking: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_basic_semantic_chunks():
    """Demonstrate basic semantic chunk creation and features"""
    print("🧩 BASIC SEMANTIC CHUNKS DEMO")
    print("-" * 50)
    
    # Create a basic semantic code chunk
    code_chunk = SemanticCodeChunk(
        id="demo_function",
        file_path="demo.py",
        content="def hello_world():\n    print('Hello, World!')\n    return True",
        start_line=1,
        end_line=3,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.FUNCTION
    )
    
    # Add semantic information
    code_chunk.add_semantic_tag("function", confidence=1.0, source="manual")
    code_chunk.add_semantic_tag("hello_world", confidence=0.9, source="pattern")
    code_chunk.function_calls = ["print"]
    code_chunk.variables_used = []
    code_chunk.code_patterns = ["function_definition"]
    
    print(f"✅ Created semantic code chunk:")
    print(f"   ID: {code_chunk.id}")
    print(f"   Type: {type(code_chunk).__name__}")
    print(f"   Content Type: {code_chunk.content_type.value}")
    print(f"   Chunk Type: {code_chunk.chunk_type.value}")
    print(f"   Lines: {code_chunk.start_line}-{code_chunk.end_line}")
    print(f"   Size: {len(code_chunk.content)} chars")
    print(f"   Semantic Tags: {[tag.name for tag in code_chunk.semantic_tags]}")
    print(f"   Function Calls: {code_chunk.function_calls}")
    print(f"   Code Patterns: {code_chunk.code_patterns}")
    
    # Show quality metrics
    from chuk_code_raptor.chunking.semantic_chunk import calculate_code_quality_metrics
    quality_metrics = calculate_code_quality_metrics(code_chunk)
    print(f"   Quality Metrics: {quality_metrics}")
    
    print()

def main():
    """Main demo function"""
    print_banner()
    
    # Show system info
    try:
        from chuk_code_raptor.chunking import __version__
        print(f"📦 CodeRaptor version: {__version__}")
    except:
        print(f"📦 CodeRaptor version: development")
    
    print(f"🧠 Semantic chunking: Enabled")
    print(f"🌲 Tree-sitter: {'Enabled' if TREE_SITTER_AVAILABLE else 'Not Available'}")
    print()
    
    # Demo basic semantic chunks
    demonstrate_basic_semantic_chunks()
    
    # Demo configuration loading
    demonstrate_config_loading()
    
    # Demo semantic engine interface
    demonstrate_semantic_chunking_engine()
    
    # Demo Python semantic chunking
    demonstrate_semantic_python_chunking()
    
    print("🎉 Semantic tree-sitter chunking demo completed!")
    print("\nKey Semantic Features:")
    print("✅ Full semantic understanding with SemanticChunk objects")
    print("✅ Dependency tracking and relationship mapping")
    print("✅ Semantic tag extraction and categorization")
    print("✅ Language-specific pattern recognition")
    print("✅ Quality metrics and complexity scoring")
    
    if not TREE_SITTER_AVAILABLE:
        print("\n⚠️  To enable full tree-sitter features:")
        print("- Install tree-sitter: pip install tree-sitter tree-sitter-python")
        print("- Implement the semantic tree-sitter chunkers")

if __name__ == "__main__":
    main()