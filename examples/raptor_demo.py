#!/usr/bin/env python3
"""
RAPTOR Integration Demo
======================

Comprehensive demonstration of RAPTOR (Recursive Abstractive Processing 
for Tree-Organized Retrieval) integrated with SemanticChunks and CPG.

Shows hierarchical abstractions, intelligent query routing, and scalable search.
"""

import json
from pathlib import Path
from typing import List

# Import RAPTOR modules
from chuk_code_raptor.raptor.builder import RaptorBuilder
from chuk_code_raptor.raptor.models import HierarchyLevel, classify_query_type

# Import existing infrastructure
from chuk_code_raptor.graph.builder import CPGBuilder
from chuk_code_raptor.chunking.semantic_chunk import (
    SemanticChunk, create_chunk_id, QualityMetric, ContentType
)
from chuk_code_raptor.core.models import ChunkType

def create_chunks_from_sample_file() -> List[SemanticChunk]:
    """Create SemanticChunks from the external sample Python file"""
    chunks = []
    
    sample_file = Path("examples/samples/sample.py")
    
    if not sample_file.exists():
        print(f"Warning: Sample file not found at {sample_file}")
        print("Creating a minimal sample for demo purposes...")
        return create_minimal_sample()
    
    # Read the sample file
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading sample file: {e}")
        return create_minimal_sample()
    
    # Parse into semantic chunks (simplified - in reality use tree-sitter)
    chunks_data = [
        # Configuration classes
        ("ProcessingConfig", "class", 100, 180, "Advanced configuration for data processing operations"),
        ("ProcessingMetrics", "class", 185, 220, "Metrics tracking for processing operations"),
        
        # Protocol definitions  
        ("ProcessorProtocol", "class", 225, 240, "Protocol defining the interface for data processors"),
        ("CacheProtocol", "class", 245, 260, "Protocol for caching implementations"),
        
        # Cache implementation
        ("MemoryCache", "class", 265, 330, "Simple in-memory cache implementation with TTL"),
        
        # Base processor
        ("BaseProcessor", "class", 335, 450, "Abstract base class for all data processors"),
        
        # Advanced async processor
        ("AsyncDataProcessor", "class", 455, 650, "Advanced asynchronous data processor"),
        
        # File processor
        ("FileProcessor", "class", 655, 750, "Specialized processor for file operations"),
        
        # Factory and utilities
        ("ProcessorFactory", "class", 755, 790, "Factory for creating different types of processors"),
        ("get_default_config", "function", 795, 810, "Get default configuration for processor type"),
        ("comprehensive_demo", "function", 850, 950, "Main demonstration function")
    ]
    
    for name, chunk_type, start_line, end_line, description in chunks_data:
        # Extract content for this chunk
        lines = content.split('\n')
        if start_line <= len(lines):
            chunk_content = '\n'.join(lines[max(0, start_line-1):min(len(lines), end_line)])
        else:
            # Fallback content if line numbers don't match
            chunk_content = f"# {name}\n# {description}\nclass {name}:\n    pass"
        
        # Create chunk based on type
        ct = ChunkType.CLASS if chunk_type == "class" else ChunkType.FUNCTION
        
        chunk = SemanticChunk(
            id=create_chunk_id(str(sample_file), start_line, ct, name),
            file_path=str(sample_file),
            content=chunk_content,
            start_line=start_line,
            end_line=end_line,
            content_type=ContentType.CODE,
            language="python",
            chunk_type=ct,
            summary=description
        )
        
        # Add semantic tags based on the chunk name and type
        if "Config" in name:
            chunk.add_semantic_tag("configuration", confidence=0.95, source="analysis")
            chunk.add_semantic_tag("dataclass", confidence=0.9, source="ast")
        elif "Protocol" in name:
            chunk.add_semantic_tag("protocol", confidence=0.95, source="ast")
            chunk.add_semantic_tag("interface", confidence=0.9, source="analysis")
        elif "Cache" in name:
            chunk.add_semantic_tag("caching", confidence=0.95, source="analysis")
            chunk.add_semantic_tag("memory-management", confidence=0.8, source="analysis")
        elif "Processor" in name:
            chunk.add_semantic_tag("processing", confidence=0.95, source="analysis")
            chunk.add_semantic_tag("async", confidence=0.9, source="ast")
            chunk.add_semantic_tag("business-logic", confidence=0.85, source="analysis")
        elif "Factory" in name:
            chunk.add_semantic_tag("factory-pattern", confidence=0.9, source="pattern")
            chunk.add_semantic_tag("creation", confidence=0.8, source="analysis")
        
        # Add common tags
        chunk.add_semantic_tag("python", confidence=1.0, source="ast")
        chunk.add_semantic_tag("async-programming", confidence=0.8, source="analysis")
        chunk.add_semantic_tag("enterprise", confidence=0.7, source="analysis")
        
        # Set quality scores based on chunk characteristics
        if "demo" in name.lower() or "example" in name.lower():
            # Demo code typically has lower production quality
            chunk.set_quality_score(QualityMetric.MAINTAINABILITY, 0.75)
            chunk.set_quality_score(QualityMetric.READABILITY, 0.85)
            chunk.set_quality_score(QualityMetric.SEMANTIC_COHERENCE, 0.80)
        elif "Protocol" in name or "ABC" in chunk_content:
            # Interfaces/protocols typically high quality
            chunk.set_quality_score(QualityMetric.MAINTAINABILITY, 0.95)
            chunk.set_quality_score(QualityMetric.READABILITY, 0.90)
            chunk.set_quality_score(QualityMetric.SEMANTIC_COHERENCE, 0.95)
        else:
            # Regular implementation code
            chunk.set_quality_score(QualityMetric.MAINTAINABILITY, 0.85)
            chunk.set_quality_score(QualityMetric.READABILITY, 0.80)
            chunk.set_quality_score(QualityMetric.SEMANTIC_COHERENCE, 0.85)
        
        chunks.append(chunk)
    
    # Add relationships between chunks
    if len(chunks) >= 6:  # Ensure we have enough chunks
        # BaseProcessor is used by AsyncDataProcessor
        chunks[5].add_relationship(chunks[4].id, "depends_on", strength=0.9, 
                                  context="inherits from BaseProcessor")
        
        # AsyncDataProcessor uses MemoryCache
        chunks[6].add_relationship(chunks[4].id, "uses", strength=0.7,
                                  context="caching implementation")
        
        # ProcessorFactory creates processors
        if len(chunks) >= 9:
            chunks[8].add_relationship(chunks[5].id, "creates", strength=0.8,
                                      context="factory pattern")
            chunks[8].add_relationship(chunks[6].id, "creates", strength=0.8,
                                      context="factory pattern")
    
    # Add mock embeddings for semantic similarity
    base_embeddings = {
        'config': [0.2, 0.1, 0.8, 0.3] * 75,     # 300-dim
        'protocol': [0.8, 0.2, 0.1, 0.4] * 75,
        'cache': [0.3, 0.8, 0.2, 0.5] * 75,
        'processor': [0.5, 0.3, 0.8, 0.2] * 75,
        'factory': [0.7, 0.5, 0.3, 0.8] * 75,
        'demo': [0.4, 0.6, 0.5, 0.7] * 75
    }
    
    for chunk in chunks:
        if "Config" in chunk.id or "Metrics" in chunk.id:
            embedding_type = 'config'
        elif "Protocol" in chunk.id:
            embedding_type = 'protocol'
        elif "Cache" in chunk.id:
            embedding_type = 'cache'
        elif "Processor" in chunk.id:
            embedding_type = 'processor'
        elif "Factory" in chunk.id:
            embedding_type = 'factory'
        else:
            embedding_type = 'demo'
        
        base = base_embeddings[embedding_type]
        # Add slight variations for uniqueness
        embedding = [x + (hash(chunk.id) % 20 - 10) / 1000 for x in base]
        chunk.set_embedding(embedding, "text-embedding-ada-002", 1)
    
    return chunks

def create_minimal_sample() -> List[SemanticChunk]:
    """Create a minimal sample when the external file is not available"""
    chunks = []
    
    minimal_chunks_data = [
        ("SimpleConfig", "class", "Configuration class for basic settings"),
        ("DataProcessor", "class", "Main data processing class"),
        ("FileHandler", "class", "File operations handler"),
        ("CacheManager", "class", "Cache management functionality"),
        ("ProcessorFactory", "class", "Factory for creating processors"),
        ("process_data", "function", "Main processing function"),
        ("validate_input", "function", "Input validation function"),
        ("setup_logging", "function", "Logging configuration function")
    ]
    
    for i, (name, chunk_type, description) in enumerate(minimal_chunks_data):
        ct = ChunkType.CLASS if chunk_type == "class" else ChunkType.FUNCTION
        
        # Create simple content based on type
        if chunk_type == "class":
            content = f"""class {name}:
    \"\"\"
    {description}
    \"\"\"
    def __init__(self):
        self.initialized = True
    
    def process(self):
        return "processed"
"""
        else:
            content = f"""def {name}(data):
    \"\"\"
    {description}
    \"\"\"
    if not data:
        return None
    return data
"""
        
        chunk = SemanticChunk(
            id=create_chunk_id("examples/minimal.py", i*10 + 1, ct, name),
            file_path="examples/minimal.py",
            content=content,
            start_line=i*10 + 1,
            end_line=i*10 + 10,
            content_type=ContentType.CODE,
            language="python",
            chunk_type=ct,
            summary=description
        )
        
        # Add basic tags and quality
        chunk.add_semantic_tag("python", confidence=1.0, source="ast")
        chunk.add_semantic_tag("minimal-example", confidence=0.9, source="manual")
        
        if "Config" in name:
            chunk.add_semantic_tag("configuration", confidence=0.8, source="analysis")
        elif "Cache" in name:
            chunk.add_semantic_tag("caching", confidence=0.8, source="analysis")
        elif "Factory" in name:
            chunk.add_semantic_tag("factory-pattern", confidence=0.8, source="pattern")
        elif "Processor" in name:
            chunk.add_semantic_tag("processing", confidence=0.8, source="analysis")
        
        chunk.set_quality_score(QualityMetric.READABILITY, 0.85)
        chunk.set_quality_score(QualityMetric.MAINTAINABILITY, 0.80)
        
        # Add basic embedding
        embedding = [0.5 + (i * 0.1) % 1.0] * 300
        chunk.set_embedding(embedding, "text-embedding-ada-002", 1)
        
        chunks.append(chunk)
    
    return chunks

def demo_raptor_construction():
    """Demo RAPTOR hierarchy construction"""
    print("="*70)
    print(" RAPTOR HIERARCHY CONSTRUCTION")
    print("="*70)
    
    chunks = create_chunks_from_sample_file()
    print(f"Created codebase from sample file with {len(chunks)} chunks")
    
    # Build CPG first
    cpg_builder = CPGBuilder()
    cpg = cpg_builder.build_from_chunks(chunks)
    print(f"Built CPG: {len(cpg.nodes)} nodes, {len(cpg.edges)} edges")
    
    # Build RAPTOR with CPG integration
    raptor_builder = RaptorBuilder(cpg)
    build_summary = raptor_builder.build_from_chunks(chunks)
    
    print(f"Built RAPTOR hierarchy:")
    hierarchy_stats = build_summary['hierarchy_stats']
    print(f"  Total nodes: {hierarchy_stats['total_nodes']}")
    print(f"  Build time: {build_summary['total_build_time']:.2f}s")
    print(f"  Compression ratio: {hierarchy_stats['compression_ratio']:.3f}")
    
    print("\n--- Hierarchy Levels ---")
    for level, count in hierarchy_stats['nodes_by_level'].items():
        level_name = HierarchyLevel(level).name
        print(f"  Level {level} ({level_name}): {count} nodes")
    
    return raptor_builder, chunks

def demo_intelligent_query_routing(raptor_builder):
    """Demo intelligent query routing based on query type"""
    print("\n" + "="*70)
    print(" INTELLIGENT QUERY ROUTING")
    print("="*70)
    
    queries = [
        ("How does the processing system work?", "architectural"),
        ("Show me the cache implementation", "implementation"),
        ("What calls the ProcessorFactory?", "relationship"),
        ("What's the API for configuration?", "api"),
        ("Where are the quality issues?", "quality")
    ]
    
    for query, expected_type in queries:
        print(f"\n--- Query: '{query}' ---")
        
        detected_type = classify_query_type(query)
        print(f"Detected type: {detected_type} (expected: {expected_type})")
        
        # Perform search
        results = raptor_builder.intelligent_search(query, max_results=3)
        
        print(f"Search method: {results['search_method']}")
        print(f"Results found: {results['total_found']}")
        
        for i, result in enumerate(results['results'][:2]):
            level_name = HierarchyLevel(result['level']).name
            print(f"  {i+1}. [{level_name}] {result.get('file_path', 'N/A')}")
            print(f"     Score: {result['score']:.3f}")
            summary = result['summary'][:100] + "..." if len(result['summary']) > 100 else result['summary']
            print(f"     Summary: {summary}")

def demo_hierarchical_context(raptor_builder, chunks):
    """Demo hierarchical context retrieval"""
    print("\n" + "="*70)
    print(" HIERARCHICAL CONTEXT RETRIEVAL")
    print("="*70)
    
    # Get context for the first processor chunk
    target_chunk = None
    for chunk in chunks:
        if "Processor" in chunk.id or "Config" in chunk.id:
            target_chunk = chunk
            break
    
    if not target_chunk:
        target_chunk = chunks[0] if chunks else None
    
    if not target_chunk:
        print("No chunks available for context demo")
        return
    
    print(f"Getting hierarchical context for: {target_chunk.id}")
    
    context = raptor_builder.get_hierarchical_context(target_chunk.id)
    
    print("\n--- Hierarchy Path ---")
    for level_info in context['hierarchy_path']:
        level_name = HierarchyLevel(level_info['level']).name
        print(f"  Level {level_info['level']} ({level_name}): {level_info['title']}")
        print(f"    Keywords: {', '.join(level_info['keywords'])}")
        if level_info['summary']:
            summary = level_info['summary'][:150] + "..." if len(level_info['summary']) > 150 else level_info['summary']
            print(f"    Summary: {summary}")
    
    print(f"\n--- Related Components ---")
    print(f"Related chunks: {len(context['related_chunks'])}")
    for related_id in context['related_chunks'][:5]:
        print(f"  - {related_id}")
    
    print(f"\n--- Architectural Context ---")
    arch_context = context['architectural_context']
    print(f"  Impact scope: {arch_context['impact_scope']} components")
    print(f"  Direct dependencies: {arch_context['direct_dependencies']}")
    print(f"  Importance level: {arch_context['importance_level']}")

def demo_token_efficient_search(raptor_builder):
    """Demo token-efficient search with budget management"""
    print("\n" + "="*70)
    print(" TOKEN-EFFICIENT SEARCH")
    print("="*70)
    
    queries_and_budgets = [
        ("How does configuration work?", 2000),
        ("Explain the processing architecture", 4000),
        ("Show me all caching-related code", 8000)
    ]
    
    for query, token_budget in queries_and_budgets:
        print(f"\n--- Query: '{query}' (Budget: {token_budget} tokens) ---")
        
        results = raptor_builder.intelligent_search(
            query, max_results=10, max_tokens=token_budget
        )
        
        tokens_used = results.get('tokens_used', 0)
        efficiency = (tokens_used / token_budget) * 100 if token_budget > 0 else 0
        
        print(f"Tokens used: {tokens_used}/{token_budget} ({efficiency:.1f}%)")
        print(f"Results returned: {len(results['results'])}")
        print(f"Search method: {results['search_method']}")
        
        # Show level distribution of results
        level_counts = {}
        for result in results['results']:
            level_name = HierarchyLevel(result['level']).name
            level_counts[level_name] = level_counts.get(level_name, 0) + 1
        
        print(f"Level distribution: {level_counts}")

def demo_incremental_updates(raptor_builder, chunks):
    """Demo incremental RAPTOR updates"""
    print("\n" + "="*70)
    print(" INCREMENTAL RAPTOR UPDATES")
    print("="*70)
    
    # Find a processor chunk to modify
    processor_chunk = None
    for chunk in chunks:
        if "Processor" in chunk.id:
            processor_chunk = chunk
            break
    
    if not processor_chunk:
        print("No processor chunk found for update demo")
        return
    
    print(f"Original chunk version: {processor_chunk.version}")
    print(f"Original fingerprint: {processor_chunk.combined_fingerprint[:16]}...")
    
    # Modify the chunk
    processor_chunk.content += "\n\n    def new_feature(self):\n        return 'enhanced functionality'"
    processor_chunk.update_fingerprints()
    
    print(f"Modified version: {processor_chunk.version}")
    print(f"Modified fingerprint: {processor_chunk.combined_fingerprint[:16]}...")
    
    # Update RAPTOR
    update_summary = raptor_builder.update_from_changes([processor_chunk])
    
    print(f"\nUpdate Results:")
    print(f"  Affected RAPTOR nodes: {update_summary['affected_raptor_nodes']}")
    print(f"  Affected CPG nodes: {update_summary['affected_cpg_nodes']}")
    print(f"  Update time: {update_summary['total_update_time']:.3f}s")
    print(f"  Efficiency ratio: {update_summary['efficiency_ratio']:.2f}")

def demo_scalability_analysis(raptor_builder):
    """Demo scalability metrics and performance analysis"""
    print("\n" + "="*70)
    print(" SCALABILITY ANALYSIS")
    print("="*70)
    
    hierarchy_summary = raptor_builder.export_hierarchy_summary()
    
    print("--- Hierarchy Overview ---")
    overview = hierarchy_summary['hierarchy_overview']
    print(f"  Total nodes: {overview['total_nodes']}")
    print(f"  Total chunks: {overview['total_chunks']}")
    print(f"  Build time: {overview['build_time']:.2f}s")
    print(f"  Compression ratio: {overview['compression_ratio']:.3f}")
    
    print("\n--- Performance Metrics ---")
    perf = hierarchy_summary['performance_metrics']
    print(f"  Hierarchy depth: {perf['hierarchy_depth']}")
    print(f"  Scalability score: {perf['scalability_score']:.3f}")
    print(f"  Average quality: {perf['average_quality']:.3f}")
    
    print("\n--- Level Samples ---")
    for level, sample in hierarchy_summary['level_samples'].items():
        level_name = HierarchyLevel(level).name
        print(f"  Level {level} ({level_name}):")
        print(f"    Title: {sample['title']}")
        print(f"    Keywords: {', '.join(sample['keywords'])}")

def main():
    """Run the complete RAPTOR demo"""
    print("RAPTOR (RECURSIVE ABSTRACTIVE PROCESSING) INTEGRATION DEMO")
    print("Showcasing hierarchical abstractions with SemanticChunk + CPG integration")
    
    # Build hierarchy
    raptor_builder, chunks = demo_raptor_construction()
    
    # Demo intelligent features
    demo_intelligent_query_routing(raptor_builder)
    demo_hierarchical_context(raptor_builder, chunks)
    demo_token_efficient_search(raptor_builder)
    demo_incremental_updates(raptor_builder, chunks)
    demo_scalability_analysis(raptor_builder)
    
    print("\n" + "="*70)
    print(" RAPTOR DEMO COMPLETE")
    print("="*70)
    print("RAPTOR capabilities demonstrated:")
    print("✅ Hierarchical abstraction (chunks → files → modules → repository)")
    print("✅ Intelligent query routing based on query type")
    print("✅ Token-efficient search with budget management")
    print("✅ Incremental updates with change propagation")
    print("✅ Integration with SemanticChunk fingerprints")
    print("✅ CPG relationship enhancement")
    print("✅ Scalable architecture (O(log n) memory usage)")
    print("✅ Multi-level context retrieval")
    
    print("\n🚀 RAPTOR enables:")
    print("  • Netflix-scale codebase analysis")
    print("  • Smart query routing (architectural vs implementation)")
    print("  • Token-efficient LLM context")
    print("  • Incremental knowledge updates")
    print("  • Hierarchical code understanding")

if __name__ == "__main__":
    main()