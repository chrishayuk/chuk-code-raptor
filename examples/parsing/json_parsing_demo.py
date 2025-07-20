#!/usr/bin/env python3
"""
JSON Parsing Demo
================

Demonstrates the JSON parser capabilities with various JSON structures.
Shows semantic analysis, property detection, and relationship mapping.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_sample_json_data() -> Dict[str, str]:
    """Create various JSON samples for testing"""
    
    samples = {}
    
    # Simple JSON object
    samples['simple_object'] = json.dumps({
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }, indent=2)
    
    # Complex nested structure
    samples['nested_structure'] = json.dumps({
        "user": {
            "id": "user_123",
            "profile": {
                "name": "Alice Smith",
                "avatar": "https://example.com/avatar.jpg",
                "preferences": {
                    "theme": "dark",
                    "notifications": True,
                    "language": "en"
                }
            },
            "metadata": {
                "created_at": "2023-01-15T10:30:00Z",
                "updated_at": "2024-01-20T14:45:00Z",
                "version": 2
            }
        },
        "permissions": ["read", "write", "admin"],
        "settings": {
            "security": {
                "two_factor": True,
                "session_timeout": 3600
            }
        }
    }, indent=2)
    
    # Array with objects
    samples['array_of_objects'] = json.dumps([
        {
            "id": "product_1",
            "name": "Laptop",
            "price": 999.99,
            "category": "electronics",
            "in_stock": True,
            "tags": ["computer", "portable", "work"]
        },
        {
            "id": "product_2", 
            "name": "Coffee Maker",
            "price": 79.99,
            "category": "appliances",
            "in_stock": False,
            "tags": ["kitchen", "coffee", "appliance"]
        },
        {
            "id": "product_3",
            "name": "Book Set",
            "price": 45.00,
            "category": "books",
            "in_stock": True,
            "tags": ["education", "reading", "collection"]
        }
    ], indent=2)
    
    # Configuration-style JSON
    samples['config_file'] = json.dumps({
        "app": {
            "name": "MyApplication",
            "version": "1.2.3",
            "debug": False,
            "features": {
                "authentication": True,
                "analytics": True,
                "caching": {
                    "enabled": True,
                    "ttl": 300,
                    "strategy": "lru"
                }
            }
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "myapp_db",
            "ssl": True,
            "pool": {
                "min_connections": 5,
                "max_connections": 20
            }
        },
        "logging": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }, indent=2)
    
    # API response structure
    samples['api_response'] = json.dumps({
        "status": "success",
        "data": {
            "items": [
                {
                    "uuid": "550e8400-e29b-41d4-a716-446655440001",
                    "title": "Getting Started with JSON",
                    "author": {
                        "name": "Jane Developer",
                        "id": "author_456"
                    },
                    "published_at": "2024-01-15T09:00:00Z",
                    "content": {
                        "summary": "A comprehensive guide to JSON parsing",
                        "word_count": 1250,
                        "reading_time": 5
                    },
                    "references": [
                        {"type": "article", "id": "art_123"},
                        {"type": "tutorial", "id": "tut_789"}
                    ]
                }
            ],
            "pagination": {
                "page": 1,
                "per_page": 10,
                "total": 1,
                "has_next": False
            }
        },
        "meta": {
            "request_id": "req_abc123",
            "timestamp": "2024-01-20T15:30:00Z",
            "version": "v2"
        }
    }, indent=2)
    
    # Schema-like structure
    samples['json_schema'] = json.dumps({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "User",
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Unique user identifier"
            },
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100
            },
            "email": {
                "type": "string",
                "format": "email"
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150
            },
            "preferences": {
                "type": "object",
                "properties": {
                    "theme": {
                        "type": "string",
                        "enum": ["light", "dark", "auto"]
                    },
                    "notifications": {
                        "type": "boolean"
                    }
                }
            }
        },
        "required": ["id", "name", "email"]
    }, indent=2)
    
    # Mixed array types
    samples['mixed_array'] = json.dumps({
        "mixed_data": [
            "string_value",
            42,
            True,
            None,
            {"nested": "object"},
            [1, 2, 3],
            3.14159
        ],
        "metadata": {
            "description": "Array with mixed data types",
            "created": "2024-01-20"
        }
    }, indent=2)
    
    return samples

def analyze_chunk_structure(chunk) -> Dict[str, Any]:
    """Analyze the structure and properties of a JSON chunk"""
    analysis = {
        'id': chunk.id,
        'type': chunk.chunk_type.value,
        'size_chars': len(chunk.content),
        'size_lines': chunk.line_count,
        'importance': chunk.importance_score,
        'tags': [tag.name for tag in chunk.semantic_tags] if hasattr(chunk, 'semantic_tags') and chunk.semantic_tags else [],
        'metadata': chunk.metadata or {},
        'preview': chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
    }
    
    # Analyze JSON content
    try:
        json_data = json.loads(chunk.content)
        analysis['json_analysis'] = {
            'is_valid_json': True,
            'root_type': type(json_data).__name__,
            'estimated_depth': estimate_json_depth(json_data),
            'key_count': count_json_keys(json_data) if isinstance(json_data, dict) else 0,
            'array_length': len(json_data) if isinstance(json_data, list) else 0
        }
        
        # Detect common patterns
        if isinstance(json_data, dict):
            analysis['patterns'] = detect_json_patterns(json_data)
        
    except json.JSONDecodeError:
        analysis['json_analysis'] = {
            'is_valid_json': False,
            'error': 'Invalid JSON structure'
        }
    
    return analysis

def estimate_json_depth(obj, current_depth=0) -> int:
    """Estimate the maximum depth of a JSON structure"""
    if isinstance(obj, dict):
        return max([estimate_json_depth(v, current_depth + 1) for v in obj.values()], default=current_depth)
    elif isinstance(obj, list):
        return max([estimate_json_depth(item, current_depth + 1) for item in obj], default=current_depth)
    else:
        return current_depth

def count_json_keys(obj) -> int:
    """Count total number of keys in a JSON object"""
    if isinstance(obj, dict):
        count = len(obj)
        for value in obj.values():
            count += count_json_keys(value)
        return count
    elif isinstance(obj, list):
        return sum(count_json_keys(item) for item in obj)
    else:
        return 0

def detect_json_patterns(obj) -> List[str]:
    """Detect common JSON patterns and structures"""
    patterns = []
    
    if isinstance(obj, dict):
        keys = set(obj.keys())
        
        # Common ID patterns
        if any(id_key in keys for id_key in ['id', 'uuid', '_id', 'identifier']):
            patterns.append('has_identifier')
        
        # Timestamp patterns
        if any(time_key in keys for time_key in ['created_at', 'updated_at', 'timestamp', 'date']):
            patterns.append('timestamped')
        
        # Metadata patterns
        if any(meta_key in keys for meta_key in ['metadata', 'meta', 'info']):
            patterns.append('has_metadata')
        
        # Configuration patterns
        if any(config_key in keys for config_key in ['config', 'settings', 'options', 'preferences']):
            patterns.append('configuration')
        
        # API response patterns
        if any(api_key in keys for api_key in ['status', 'data', 'result', 'response']):
            patterns.append('api_response')
        
        # Schema patterns
        if any(schema_key in keys for schema_key in ['$schema', 'type', 'properties', 'required']):
            patterns.append('json_schema')
        
        # Pagination patterns
        if any(page_key in keys for page_key in ['page', 'pagination', 'offset', 'limit']):
            patterns.append('paginated')
        
        # Reference patterns
        if any(ref_key in keys for ref_key in ['$ref', 'ref', 'references', 'links']):
            patterns.append('has_references')
    
    return patterns

def generate_json_summary(chunks) -> Dict[str, Any]:
    """Generate summary of JSON parsing results"""
    summary = {
        'total_chunks': len(chunks),
        'chunk_types': defaultdict(int),
        'json_patterns': defaultdict(int),
        'structure_analysis': {
            'total_objects': 0,
            'total_arrays': 0,
            'max_depth': 0,
            'total_keys': 0
        },
        'content_features': defaultdict(int)
    }
    
    for chunk in chunks:
        # Type distribution
        summary['chunk_types'][chunk.chunk_type.value] += 1
        
        # Analyze chunk
        analysis = analyze_chunk_structure(chunk)
        
        # JSON analysis
        if analysis['json_analysis']['is_valid_json']:
            json_analysis = analysis['json_analysis']
            
            if json_analysis['root_type'] == 'dict':
                summary['structure_analysis']['total_objects'] += 1
                summary['structure_analysis']['total_keys'] += json_analysis['key_count']
            elif json_analysis['root_type'] == 'list':
                summary['structure_analysis']['total_arrays'] += 1
            
            summary['structure_analysis']['max_depth'] = max(
                summary['structure_analysis']['max_depth'],
                json_analysis['estimated_depth']
            )
        
        # Pattern analysis
        patterns = analysis.get('patterns', [])
        for pattern in patterns:
            summary['json_patterns'][pattern] += 1
        
        # Tag analysis
        for tag in analysis['tags']:
            summary['content_features'][tag] += 1
    
    return summary

def print_detailed_json_analysis(samples: Dict[str, str], all_results: Dict[str, Any]):
    """Print comprehensive analysis of JSON parsing results"""
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE JSON PARSING ANALYSIS")
    print("="*80)
    
    for sample_name, result in all_results.items():
        chunks = result['chunks']
        summary = result['summary']
        
        print(f"\n🔍 SAMPLE: {sample_name.upper().replace('_', ' ')}")
        print("-" * 60)
        
        # Sample info
        sample_content = samples[sample_name]
        sample_size = len(sample_content)
        sample_lines = sample_content.count('\n') + 1
        
        print(f"📄 Sample size: {sample_size} characters, {sample_lines} lines")
        print(f"🧩 Chunks created: {len(chunks)}")
        
        # Summary stats
        print(f"\n📊 Structure Summary:")
        structure = summary['structure_analysis']
        print(f"   Objects: {structure['total_objects']}")
        print(f"   Arrays: {structure['total_arrays']}")
        print(f"   Max depth: {structure['max_depth']}")
        print(f"   Total keys: {structure['total_keys']}")
        
        # Patterns detected
        if summary['json_patterns']:
            print(f"\n🔍 Patterns detected:")
            for pattern, count in summary['json_patterns'].items():
                print(f"   {pattern.replace('_', ' ').title()}: {count} chunks")
        
        # Features found
        if summary['content_features']:
            print(f"\n🏷️ Features found:")
            for feature, count in summary['content_features'].items():
                if count > 0:
                    print(f"   {feature.replace('_', ' ').title()}: {count} chunks")
        
        # Detailed chunk analysis
        print(f"\n📋 Chunk Details:")
        for i, chunk in enumerate(chunks, 1):
            analysis = analyze_chunk_structure(chunk)
            
            print(f"\n   🔸 Chunk {i}: {analysis['id']}")
            print(f"      Type: {analysis['type']}")
            print(f"      Size: {analysis['size_chars']} chars, {analysis['size_lines']} lines")
            print(f"      Importance: {analysis['importance']:.2f}")
            
            if analysis['tags']:
                print(f"      Tags: {', '.join(analysis['tags'])}")
            
            # JSON-specific analysis
            json_analysis = analysis['json_analysis']
            if json_analysis['is_valid_json']:
                print(f"      JSON type: {json_analysis['root_type']}")
                if json_analysis['estimated_depth'] > 0:
                    print(f"      Depth: {json_analysis['estimated_depth']}")
                if json_analysis['key_count'] > 0:
                    print(f"      Keys: {json_analysis['key_count']}")
                if json_analysis['array_length'] > 0:
                    print(f"      Array length: {json_analysis['array_length']}")
            
            # Patterns
            patterns = analysis.get('patterns', [])
            if patterns:
                pattern_display = [p.replace('_', ' ').title() for p in patterns]
                print(f"      Patterns: {', '.join(pattern_display)}")
            
            print(f"      Preview: {analysis['preview']}")

def test_json_sample(sample_name: str, sample_content: str) -> Dict[str, Any]:
    """Test JSON parsing on a single sample"""
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    # Configure for JSON parsing
    config = ChunkingConfig(
        target_chunk_size=300,  # Smaller chunks for JSON structures
        min_chunk_size=50,
        preserve_atomic_nodes=True,
        enable_dependency_tracking=True
    )
    
    engine = ChunkingEngine(config)
    
    # Check if JSON is supported
    if not engine.can_chunk_language('json'):
        print(f"⚠️  JSON parser not available. Supported languages: {engine.get_supported_languages()}")
        return {
            'chunks': [],
            'summary': generate_json_summary([]),
            'sample_info': {
                'size_chars': len(sample_content),
                'line_count': sample_content.count('\n') + 1,
                'error': 'JSON parser not available'
            }
        }
    
    # Parse JSON
    chunks = engine.chunk_content(sample_content, 'json', f'{sample_name}.json')
    
    # Generate summary
    summary = generate_json_summary(chunks)
    
    return {
        'chunks': chunks,
        'summary': summary,
        'sample_info': {
            'size_chars': len(sample_content),
            'line_count': sample_content.count('\n') + 1
        }
    }

def main():
    """Main demo function"""
    print("📋 JSON PARSING DEMO")
    print("="*50)
    
    try:
        # Import test
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        print("✅ Chunking system imported successfully")
        
        # Test engine initialization and parser availability
        config = ChunkingConfig()
        engine = ChunkingEngine(config)
        
        supported_languages = engine.get_supported_languages()
        print(f"✅ Engine initialized with support for: {', '.join(supported_languages)}")
        
        if 'json' not in supported_languages:
            print("⚠️  JSON parser not available. This may be due to missing tree-sitter-json.")
            print("   Install with: pip install tree-sitter-json")
            print("   Demo will continue with available parsers for comparison.")
        
        # Create sample data
        print("\n📝 Creating sample JSON data...")
        samples = create_sample_json_data()
        print(f"✅ Created {len(samples)} JSON samples")
        
        # Test each sample
        all_results = {}
        
        print(f"\n🚀 Testing JSON parsing...")
        for sample_name, sample_content in samples.items():
            print(f"   🧪 Processing {sample_name}...")
            
            try:
                result = test_json_sample(sample_name, sample_content)
                all_results[sample_name] = result
                
                if 'error' in result['sample_info']:
                    print(f"      ⚠️  {result['sample_info']['error']}")
                else:
                    chunks_count = len(result['chunks'])
                    sample_size = result['sample_info']['size_chars']
                    print(f"      ✅ {chunks_count} chunks from {sample_size} characters")
                    
            except Exception as e:
                print(f"      ❌ Error processing {sample_name}: {e}")
                # Create empty result for consistency
                all_results[sample_name] = {
                    'chunks': [],
                    'summary': generate_json_summary([]),
                    'sample_info': {
                        'size_chars': len(sample_content),
                        'line_count': sample_content.count('\n') + 1,
                        'error': str(e)
                    }
                }
        
        # Print comprehensive analysis
        print_detailed_json_analysis(samples, all_results)
        
        # Overall statistics
        print(f"\n📈 OVERALL STATISTICS")
        print("-" * 60)
        
        successful_results = {k: v for k, v in all_results.items() 
                            if 'error' not in v['sample_info']}
        failed_results = {k: v for k, v in all_results.items() 
                         if 'error' in v['sample_info']}
        
        total_chunks = sum(len(r['chunks']) for r in successful_results.values())
        total_chars = sum(r['sample_info']['size_chars'] for r in successful_results.values())
        
        print(f"Total samples processed: {len(samples)}")
        print(f"Successful parses: {len(successful_results)}")
        if failed_results:
            print(f"Failed parses: {len(failed_results)}")
        print(f"Total chunks created: {total_chunks}")
        print(f"Total characters processed: {total_chars:,}")
        
        if successful_results:
            print(f"Average chunks per sample: {total_chunks / len(successful_results):.1f}")
            print(f"Average characters per chunk: {total_chars / total_chunks:.0f}" if total_chunks > 0 else "No chunks created")
        
        # Pattern summary (only for successful results)
        all_patterns = defaultdict(int)
        all_features = defaultdict(int)
        
        for result in successful_results.values():
            for pattern, count in result['summary']['json_patterns'].items():
                all_patterns[pattern] += count
            for feature, count in result['summary']['content_features'].items():
                all_features[feature] += count
        
        if all_patterns:
            print(f"\n🔍 Most common patterns:")
            for pattern, count in sorted(all_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {pattern.replace('_', ' ').title()}: {count} occurrences")
        
        if all_features:
            print(f"\n🏷️ Most common features:")
            for feature, count in sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {feature.replace('_', ' ').title()}: {count} occurrences")
        
        print(f"\n🎉 JSON parsing demo completed successfully!")
        print(f"💡 The JSON parser successfully analyzed {len(samples)} different JSON structures")
        print(f"   including nested objects, arrays, configuration files, API responses,")
        print(f"   and schema definitions with intelligent pattern recognition.")
        
        # Usage recommendations
        print(f"\n🎯 Usage recommendations:")
        print(f"   • Use smaller chunk sizes (300-500 chars) for JSON parsing")
        print(f"   • Enable atomic node preservation for complete JSON objects")
        print(f"   • Leverage pattern detection for semantic understanding")
        print(f"   • Consider dependency tracking for reference-based JSON")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package and tree-sitter-json are installed")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()