#!/usr/bin/env python3
"""
Comprehensive Python Parsing Demo
=================================

Demonstrates the full capabilities of the semantic chunking system with Python code.
Shows semantic analysis, dependency tracking, quality metrics, and relationships.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def load_sample_file():
    """Load the enhanced sample Python file from examples/samples/sample.py"""
    sample_file = Path(__file__).parent.parent.parent / "examples" / "samples" / "sample.py"
    
    if not sample_file.exists():
        print(f"❌ Sample file not found: {sample_file}")
        print("   Please ensure examples/samples/sample.py exists")
        return None
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ Loaded sample file: {sample_file} ({len(content)} characters)")
        return content
    except Exception as e:
        print(f"❌ Error reading sample file: {e}")
        return None

def analyze_chunk_quality(chunk) -> Dict[str, Any]:
    """Analyze the quality and characteristics of a chunk"""
    analysis = {
        'id': chunk.id,
        'type': chunk.chunk_type.value,
        'size_chars': len(chunk.content),
        'size_lines': chunk.line_count,
        'importance': chunk.importance_score,
        'tags': [tag.name for tag in chunk.semantic_tags] if hasattr(chunk, 'semantic_tags') and chunk.semantic_tags else [],
        'dependencies': chunk.dependencies[:5] if chunk.dependencies else [],  # First 5
        'metadata_keys': list(chunk.metadata.keys()) if chunk.metadata else [],
        'preview': chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
    }
    
    # Add semantic analysis
    content_lower = chunk.content.lower()
    analysis['features'] = {
        'has_docstring': '"""' in chunk.content or "'''" in chunk.content,
        'has_type_hints': any(hint in content_lower for hint in [': str', ': int', ': bool', ': list', ': dict', '-> ']),
        'has_async': 'async ' in content_lower,
        'has_decorators': '@' in chunk.content,
        'has_inheritance': 'class ' in content_lower and '(' in chunk.content,
        'has_error_handling': any(keyword in content_lower for keyword in ['try:', 'except', 'raise']),
        'complexity_indicators': {
            'loops': content_lower.count('for ') + content_lower.count('while '),
            'conditions': content_lower.count('if ') + content_lower.count('elif '),
            'function_calls': len(chunk.dependencies) if chunk.dependencies else 0
        }
    }
    
    return analysis

def create_chunk_relationships_map(chunks) -> Dict[str, List[str]]:
    """Create a map of chunk relationships"""
    relationships = defaultdict(list)
    
    for chunk in chunks:
        chunk_id = chunk.id
        
        # Dependencies
        for dep in chunk.dependencies:
            if ':' in dep:
                dep_type, dep_name = dep.split(':', 1)
                relationships[chunk_id].append(f"{dep_type} -> {dep_name}")
        
        # Look for related chunks by content
        for other_chunk in chunks:
            if other_chunk.id != chunk_id:
                # Check if this chunk calls functions defined in other chunks
                if hasattr(other_chunk, 'metadata') and other_chunk.metadata:
                    identifier = other_chunk.metadata.get('identifier')
                    if identifier and identifier in chunk.content:
                        relationships[chunk_id].append(f"uses -> {other_chunk.id}")
    
    return dict(relationships)

def generate_semantic_summary(chunks) -> Dict[str, Any]:
    """Generate a semantic summary of all chunks"""
    summary = {
        'total_chunks': len(chunks),
        'chunk_types': defaultdict(int),
        'size_distribution': {
            'small': 0,   # < 100 chars
            'medium': 0,  # 100-500 chars  
            'large': 0,   # 500+ chars
        },
        'features_found': defaultdict(int),
        'complexity_analysis': {
            'total_functions': 0,
            'total_classes': 0,
            'total_imports': 0,
            'async_functions': 0,
            'decorated_functions': 0,
            'classes_with_inheritance': 0
        },
        'quality_metrics': {
            'high_importance': 0,
            'with_docstrings': 0,
            'with_type_hints': 0,
            'with_error_handling': 0
        }
    }
    
    for chunk in chunks:
        # Type distribution
        summary['chunk_types'][chunk.chunk_type.value] += 1
        
        # Size distribution
        size = len(chunk.content)
        if size < 100:
            summary['size_distribution']['small'] += 1
        elif size < 500:
            summary['size_distribution']['medium'] += 1
        else:
            summary['size_distribution']['large'] += 1
        
        # Analyze features
        analysis = analyze_chunk_quality(chunk)
        features = analysis.get('features', {})
        
        for feature, present in features.items():
            if isinstance(present, bool) and present:
                summary['features_found'][feature] += 1
        
        # Complexity analysis
        if chunk.chunk_type.value == 'function':
            summary['complexity_analysis']['total_functions'] += 1
            if features.get('has_async'):
                summary['complexity_analysis']['async_functions'] += 1
            if features.get('has_decorators'):
                summary['complexity_analysis']['decorated_functions'] += 1
        elif chunk.chunk_type.value == 'class':
            summary['complexity_analysis']['total_classes'] += 1
            if features.get('has_inheritance'):
                summary['complexity_analysis']['classes_with_inheritance'] += 1
        elif chunk.chunk_type.value == 'import':
            summary['complexity_analysis']['total_imports'] += 1
        
        # Quality metrics
        if chunk.importance_score > 0.8:
            summary['quality_metrics']['high_importance'] += 1
        if features.get('has_docstring'):
            summary['quality_metrics']['with_docstrings'] += 1
        if features.get('has_type_hints'):
            summary['quality_metrics']['with_type_hints'] += 1
        if features.get('has_error_handling'):
            summary['quality_metrics']['with_error_handling'] += 1
    
    return summary

def print_detailed_analysis(chunks, relationships, summary):
    """Print comprehensive analysis of the chunking results"""
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE PYTHON PARSING ANALYSIS")
    print("="*80)
    
    # Overall summary
    print(f"\n📊 OVERALL SUMMARY")
    print(f"   Total chunks created: {summary['total_chunks']}")
    print(f"   Chunk types: {dict(summary['chunk_types'])}")
    print(f"   Size distribution: {dict(summary['size_distribution'])}")
    
    # Quality analysis
    print(f"\n⭐ QUALITY METRICS")
    quality = summary['quality_metrics']
    print(f"   High importance chunks: {quality['high_importance']}")
    print(f"   With docstrings: {quality['with_docstrings']}")
    print(f"   With type hints: {quality['with_type_hints']}")
    print(f"   With error handling: {quality['with_error_handling']}")
    
    # Complexity analysis
    print(f"\n🧮 COMPLEXITY ANALYSIS")
    complexity = summary['complexity_analysis']
    print(f"   Functions: {complexity['total_functions']} (async: {complexity['async_functions']}, decorated: {complexity['decorated_functions']})")
    print(f"   Classes: {complexity['total_classes']} (with inheritance: {complexity['classes_with_inheritance']})")
    print(f"   Imports: {complexity['total_imports']}")
    
    # Features found
    print(f"\n🔧 FEATURES DETECTED")
    for feature, count in summary['features_found'].items():
        if count > 0:
            print(f"   {feature.replace('_', ' ').title()}: {count} chunks")
    
    # Detailed chunk analysis
    print(f"\n📋 DETAILED CHUNK ANALYSIS")
    print("-" * 80)
    
    for i, chunk in enumerate(chunks, 1):
        analysis = analyze_chunk_quality(chunk)
        
        print(f"\n🔸 Chunk {i}: {analysis['id']}")
        print(f"   Type: {analysis['type']}")
        print(f"   Size: {analysis['size_chars']} chars, {analysis['size_lines']} lines")
        print(f"   Importance: {analysis['importance']:.2f}")
        
        if analysis['tags']:
            print(f"   Tags: {', '.join(analysis['tags'])}")
        
        if analysis['dependencies']:
            print(f"   Dependencies: {', '.join(analysis['dependencies'])}")
        
        # Feature highlights
        features = analysis['features']
        highlights = []
        if features['has_docstring']:
            highlights.append("📝 Documented")
        if features['has_type_hints']:
            highlights.append("🏷️ Type Hints")
        if features['has_async']:
            highlights.append("⚡ Async")
        if features['has_decorators']:
            highlights.append("🎭 Decorated")
        if features['has_inheritance']:
            highlights.append("🧬 Inheritance")
        if features['has_error_handling']:
            highlights.append("🛡️ Error Handling")
        
        if highlights:
            print(f"   Features: {' '.join(highlights)}")
        
        # Complexity indicators
        complexity_ind = features['complexity_indicators']
        if any(complexity_ind.values()):
            complexity_parts = []
            if complexity_ind['loops']:
                complexity_parts.append(f"{complexity_ind['loops']} loops")
            if complexity_ind['conditions']:
                complexity_parts.append(f"{complexity_ind['conditions']} conditions")
            if complexity_ind['function_calls']:
                complexity_parts.append(f"{complexity_ind['function_calls']} calls")
            print(f"   Complexity: {', '.join(complexity_parts)}")
        
        print(f"   Preview: {analysis['preview']}")
    
    # Relationships
    if relationships:
        print(f"\n🔗 CHUNK RELATIONSHIPS")
        print("-" * 80)
        for chunk_id, relations in relationships.items():
            if relations:
                print(f"\n{chunk_id}:")
                for relation in relations[:5]:  # Limit to first 5
                    print(f"   → {relation}")

def main():
    """Main demo function"""
    print("🐍 COMPREHENSIVE PYTHON PARSING DEMO")
    print("="*60)
    
    try:
        # Import chunking system
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig, PRECISE_CONFIG
        
        print("✅ Chunking system imported successfully")
        
        # Load enhanced sample file
        test_content = load_sample_file()
        if not test_content:
            print("❌ Cannot proceed without sample file")
            return
        
        line_count = test_content.count('\n') + 1
        print(f"📄 Enhanced sample loaded: {len(test_content)} characters, {line_count} lines")
        
        # Configure engine for comprehensive analysis
        config = ChunkingConfig(
            target_chunk_size=800,
            min_chunk_size=30,
            preserve_atomic_nodes=True,
            include_decorators=True,
            include_docstrings=True,
            group_imports=True,
            enable_dependency_tracking=True,
            enable_similarity_analysis=True,
            enable_contextual_linking=True
        )
        
        print(f"⚙️  Using enhanced configuration:")
        print(f"   Target chunk size: {config.target_chunk_size}")
        print(f"   Preserve atomic nodes: {config.preserve_atomic_nodes}")
        print(f"   Dependency tracking: {config.enable_dependency_tracking}")
        
        # Create engine and chunk content
        engine = ChunkingEngine(config)
        print(f"🔧 Engine initialized with {len(engine.get_supported_languages())} language parsers")
        
        # Perform chunking
        print("\n🚀 Starting semantic chunking of enhanced sample...")
        chunks = engine.chunk_content(test_content, 'python', 'examples/samples/sample.py')
        print(f"✅ Chunking completed: {len(chunks)} chunks created")
        
        if not chunks:
            print("❌ No chunks were created - there may be an issue with the parser")
            return
        
        # Analyze results
        print("\n🔍 Analyzing chunk quality and relationships...")
        relationships = create_chunk_relationships_map(chunks)
        summary = generate_semantic_summary(chunks)
        
        # Print comprehensive analysis
        print_detailed_analysis(chunks, relationships, summary)
        
        # Print statistics
        stats = engine.get_statistics()
        print(f"\n📈 ENGINE STATISTICS")
        print("-" * 80)
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Chunks created: {stats['chunks_created']}")
        print(f"   Average chunks per file: {stats.get('avg_chunks_per_file', 0):.1f}")
        print(f"   Processing time: {stats.get('total_processing_time', 0):.3f}s")
        
        if stats.get('chunker_usage'):
            print(f"   Parser usage:")
            for parser, usage in stats['chunker_usage'].items():
                print(f"     {parser}: {usage['chunks_created']} chunks in {usage['total_time']:.3f}s")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"💡 The chunking system successfully analyzed the enhanced sample file with {len(chunks)} semantic chunks")
        print(f"   This demonstrates comprehensive parsing of production-quality Python code")
        print(f"   with advanced patterns: async/await, protocols, generics, context managers,")
        print(f"   factory patterns, comprehensive error handling, and sophisticated inheritance.")
        
        # Suggest next steps
        print(f"\n🔍 Analysis shows:")
        complexity = summary['complexity_analysis']
        quality = summary['quality_metrics']
        
        print(f"   • Code complexity: {complexity['total_functions']} functions, {complexity['total_classes']} classes")
        print(f"   • Modern patterns: {complexity['async_functions']} async functions, {complexity['decorated_functions']} decorated")
        print(f"   • Code quality: {quality['with_docstrings']} documented, {quality['with_type_hints']} with type hints")
        print(f"   • Error handling: {quality['with_error_handling']} chunks with proper error handling")
        
        if quality['high_importance'] > 0:
            print(f"   • {quality['high_importance']} high-importance chunks identified for prioritized processing")
        
        print(f"\n🎯 This level of semantic analysis makes the chunking system ideal for:")
        print(f"   • RAG applications requiring precise code context")
        print(f"   • Documentation generation with semantic understanding") 
        print(f"   • Code review automation with quality assessment")
        print(f"   • Refactoring tools that understand code relationships")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package is in your Python path")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()