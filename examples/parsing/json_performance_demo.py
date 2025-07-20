#!/usr/bin/env python3
"""
JSON Parser Performance Testing
===============================

Comprehensive performance testing suite for the JSON parser.
Tests parsing speed, memory usage, and scalability across different JSON scenarios.
"""

import sys
import time
import psutil
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import tempfile
import gc
import random
import string

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class PerformanceProfiler:
    """Performance profiling utility"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        """Start profiling"""
        gc.collect()  # Clean up before measurement
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def stop(self) -> Dict[str, float]:
        """Stop profiling and return metrics"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'duration_ms': (end_time - self.start_time) * 1000,
            'memory_start_mb': self.start_memory,
            'memory_end_mb': end_memory,
            'memory_delta_mb': end_memory - self.start_memory
        }

def generate_test_json(structure_type: str, target_size: int) -> str:
    """Generate test JSON data of specified structure type and approximate size"""
    
    def random_string(length: int = 10) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def random_value():
        return random.choice([
            random_string(random.randint(5, 20)),
            random.randint(1, 1000),
            random.choice([True, False]),
            None,
            round(random.uniform(0, 1000), 2)
        ])
    
    if structure_type == "simple_objects":
        # Generate array of simple objects
        objects = []
        while len(json.dumps(objects, indent=2)) < target_size:
            obj = {
                "id": f"obj_{len(objects)}",
                "name": random_string(15),
                "value": random.randint(1, 100),
                "active": random.choice([True, False]),
                "created_at": "2024-01-01T00:00:00Z"
            }
            objects.append(obj)
        return json.dumps(objects, indent=2)
    
    elif structure_type == "nested_objects":
        # Generate deeply nested object structure
        def create_nested_object(depth: int, current_depth: int = 0) -> dict:
            if current_depth >= depth:
                return {"value": random_value()}
            
            obj = {
                "id": random_string(8),
                "metadata": {
                    "level": current_depth,
                    "nested_data": create_nested_object(depth, current_depth + 1)
                },
                "properties": {
                    f"prop_{i}": random_value() for i in range(3)
                }
            }
            return obj
        
        # Create multiple nested objects until we reach target size
        objects = []
        while len(json.dumps(objects, indent=2)) < target_size:
            objects.append(create_nested_object(4))
        
        return json.dumps(objects, indent=2)
    
    elif structure_type == "wide_objects":
        # Generate objects with many properties
        def create_wide_object() -> dict:
            obj = {"id": random_string(10)}
            # Add many properties
            for i in range(50):
                obj[f"field_{i}"] = random_value()
            return obj
        
        objects = []
        while len(json.dumps(objects, indent=2)) < target_size:
            objects.append(create_wide_object())
        
        return json.dumps(objects, indent=2)
    
    elif structure_type == "mixed_arrays":
        # Generate arrays with mixed data types
        def create_mixed_array() -> list:
            array = []
            for _ in range(20):
                if random.random() < 0.3:
                    # Nested object
                    array.append({
                        "type": "object",
                        "data": {f"key_{i}": random_value() for i in range(5)}
                    })
                elif random.random() < 0.6:
                    # Nested array
                    array.append([random_value() for _ in range(5)])
                else:
                    # Simple value
                    array.append(random_value())
            return array
        
        data = {"mixed_arrays": []}
        while len(json.dumps(data, indent=2)) < target_size:
            data["mixed_arrays"].append(create_mixed_array())
        
        return json.dumps(data, indent=2)
    
    elif structure_type == "config_style":
        # Generate configuration-style JSON
        config = {
            "application": {
                "name": "TestApp",
                "version": "1.0.0",
                "debug": False,
                "features": {}
            },
            "database": {
                "connections": {},
                "pools": {}
            },
            "services": {},
            "logging": {
                "level": "INFO",
                "handlers": []
            }
        }
        
        # Add features
        for i in range(20):
            config["application"]["features"][f"feature_{i}"] = {
                "enabled": random.choice([True, False]),
                "config": {f"param_{j}": random_value() for j in range(5)}
            }
        
        # Add database connections
        for i in range(10):
            config["database"]["connections"][f"db_{i}"] = {
                "host": f"host{i}.example.com",
                "port": 5432 + i,
                "database": f"app_db_{i}",
                "ssl": random.choice([True, False]),
                "pool_settings": {
                    "min": random.randint(1, 5),
                    "max": random.randint(10, 50)
                }
            }
        
        # Add services
        for i in range(15):
            config["services"][f"service_{i}"] = {
                "enabled": random.choice([True, False]),
                "endpoint": f"https://service{i}.example.com",
                "timeout": random.randint(30, 300),
                "retries": random.randint(1, 5),
                "config": {f"setting_{j}": random_value() for j in range(8)}
            }
        
        # Add logging handlers
        for i in range(5):
            config["logging"]["handlers"].append({
                "type": random.choice(["console", "file", "syslog"]),
                "level": random.choice(["DEBUG", "INFO", "WARN", "ERROR"]),
                "config": {f"option_{j}": random_value() for j in range(3)}
            })
        
        result = json.dumps(config, indent=2)
        
        # If too small, duplicate and modify
        while len(result) < target_size:
            new_config = json.loads(result)
            suffix = len(new_config.get("applications", {}))
            new_config[f"application_instance_{suffix}"] = config["application"]
            result = json.dumps(new_config, indent=2)
        
        return result
    
    elif structure_type == "api_responses":
        # Generate API response-style JSON
        def create_api_response():
            return {
                "status": "success",
                "timestamp": "2024-01-01T00:00:00Z",
                "data": {
                    "items": [
                        {
                            "id": random_string(16),
                            "uuid": f"{random_string(8)}-{random_string(4)}-{random_string(4)}-{random_string(4)}-{random_string(12)}",
                            "name": random_string(20),
                            "description": random_string(100),
                            "metadata": {
                                "created_at": "2024-01-01T00:00:00Z",
                                "updated_at": "2024-01-01T00:00:00Z",
                                "version": random.randint(1, 10),
                                "tags": [random_string(8) for _ in range(5)]
                            },
                            "attributes": {f"attr_{j}": random_value() for j in range(10)},
                            "relationships": {
                                "parent": {"id": random_string(16), "type": "parent"},
                                "children": [{"id": random_string(16), "type": "child"} for _ in range(3)]
                            }
                        } for _ in range(10)
                    ],
                    "pagination": {
                        "page": 1,
                        "per_page": 10,
                        "total": 100,
                        "pages": 10,
                        "has_next": True,
                        "has_prev": False
                    }
                },
                "meta": {
                    "request_id": random_string(32),
                    "processing_time": round(random.uniform(0.1, 2.0), 3),
                    "api_version": "v2",
                    "rate_limit": {
                        "remaining": random.randint(0, 1000),
                        "reset_at": "2024-01-01T01:00:00Z"
                    }
                }
            }
        
        responses = []
        while len(json.dumps(responses, indent=2)) < target_size:
            responses.append(create_api_response())
        
        return json.dumps(responses, indent=2)
    
    else:  # default to simple
        return json.dumps({"message": "Simple JSON", "size": target_size}, indent=2)

def run_parsing_benchmark(content: str, iterations: int = 5) -> Dict[str, Any]:
    """Run parsing benchmark with multiple iterations"""
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    config = ChunkingConfig(
        target_chunk_size=400,  # Smaller for JSON structures
        min_chunk_size=50,
        preserve_atomic_nodes=True,
        enable_dependency_tracking=True
    )
    
    engine = ChunkingEngine(config)
    profiler = PerformanceProfiler()
    
    results = []
    chunks_counts = []
    
    for i in range(iterations):
        profiler.start()
        chunks = engine.chunk_content(content, 'json', f'test_file_{i}.json')
        metrics = profiler.stop()
        
        results.append(metrics)
        chunks_counts.append(len(chunks))
        
        # Clean up between iterations
        gc.collect()
    
    # Calculate statistics
    durations = [r['duration_ms'] for r in results]
    memory_deltas = [r['memory_delta_mb'] for r in results]
    
    return {
        'iterations': iterations,
        'avg_duration_ms': statistics.mean(durations),
        'min_duration_ms': min(durations),
        'max_duration_ms': max(durations),
        'std_duration_ms': statistics.stdev(durations) if len(durations) > 1 else 0,
        'avg_memory_delta_mb': statistics.mean(memory_deltas),
        'avg_chunks_created': statistics.mean(chunks_counts),
        'chunks_per_second': statistics.mean(chunks_counts) / (statistics.mean(durations) / 1000),
        'chars_per_second': len(content) / (statistics.mean(durations) / 1000),
        'all_results': results
    }

def run_scalability_test() -> List[Dict[str, Any]]:
    """Test JSON parser scalability across different structures and sizes"""
    
    test_scenarios = [
        # Small JSON files
        {'structure': 'simple_objects', 'size': 1000, 'label': 'Small Simple Objects'},
        {'structure': 'nested_objects', 'size': 1000, 'label': 'Small Nested Objects'},
        {'structure': 'wide_objects', 'size': 1000, 'label': 'Small Wide Objects'},
        {'structure': 'mixed_arrays', 'size': 1000, 'label': 'Small Mixed Arrays'},
        
        # Medium JSON files
        {'structure': 'simple_objects', 'size': 5000, 'label': 'Medium Simple Objects'},
        {'structure': 'nested_objects', 'size': 5000, 'label': 'Medium Nested Objects'},
        {'structure': 'config_style', 'size': 5000, 'label': 'Medium Config Style'},
        {'structure': 'api_responses', 'size': 5000, 'label': 'Medium API Responses'},
        
        # Large JSON files
        {'structure': 'simple_objects', 'size': 20000, 'label': 'Large Simple Objects'},
        {'structure': 'nested_objects', 'size': 20000, 'label': 'Large Nested Objects'},
        {'structure': 'config_style', 'size': 20000, 'label': 'Large Config Style'},
        {'structure': 'api_responses', 'size': 20000, 'label': 'Large API Responses'},
        
        # Very large JSON files
        {'structure': 'config_style', 'size': 50000, 'label': 'XLarge Config Style'},
        {'structure': 'api_responses', 'size': 50000, 'label': 'XLarge API Responses'},
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"🧪 Testing {scenario['label']} (~{scenario['size']} chars, {scenario['structure']})...")
        
        # Generate test content
        content = generate_test_json(scenario['structure'], scenario['size'])
        
        # Validate JSON
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            print(f"   ❌ Generated invalid JSON: {e}")
            continue
        
        # Run benchmark
        benchmark_result = run_parsing_benchmark(content, iterations=3)
        
        # Analyze JSON structure
        json_data = json.loads(content)
        structure_info = analyze_json_structure(json_data)
        
        # Combine scenario info with results
        result = {
            **scenario,
            'actual_size': len(content),
            'line_count': content.count('\n') + 1,
            'json_structure': structure_info,
            **benchmark_result
        }
        
        results.append(result)
        
        print(f"   ✅ {benchmark_result['avg_duration_ms']:.1f}ms avg, "
              f"{benchmark_result['avg_chunks_created']:.0f} chunks, "
              f"{benchmark_result['chars_per_second']:.0f} chars/sec")
    
    return results

def analyze_json_structure(data) -> Dict[str, Any]:
    """Analyze JSON structure characteristics"""
    def count_elements(obj, depth=0):
        if isinstance(obj, dict):
            return {
                'objects': 1 + sum(count_elements(v, depth+1)['objects'] for v in obj.values()),
                'arrays': sum(count_elements(v, depth+1)['arrays'] for v in obj.values()),
                'max_depth': max([depth] + [count_elements(v, depth+1)['max_depth'] for v in obj.values()]),
                'total_keys': len(obj) + sum(count_elements(v, depth+1)['total_keys'] for v in obj.values())
            }
        elif isinstance(obj, list):
            return {
                'objects': sum(count_elements(item, depth+1)['objects'] for item in obj),
                'arrays': 1 + sum(count_elements(item, depth+1)['arrays'] for item in obj),
                'max_depth': max([depth] + [count_elements(item, depth+1)['max_depth'] for item in obj]),
                'total_keys': sum(count_elements(item, depth+1)['total_keys'] for item in obj)
            }
        else:
            return {'objects': 0, 'arrays': 0, 'max_depth': depth, 'total_keys': 0}
    
    return count_elements(data)

def run_memory_stress_test() -> Dict[str, Any]:
    """Test memory usage under stress conditions"""
    print("🔥 Running JSON memory stress test...")
    
    profiler = PerformanceProfiler()
    
    # Generate large, complex JSON content
    large_content = generate_test_json('api_responses', 100000)
    
    # Test repeated parsing
    profiler.start()
    
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    config = ChunkingConfig(enable_dependency_tracking=True)
    engine = ChunkingEngine(config)
    
    total_chunks = 0
    iterations = 10
    
    for i in range(iterations):
        chunks = engine.chunk_content(large_content, 'json', f'stress_test_{i}.json')
        total_chunks += len(chunks)
        
        if i % 2 == 0:
            gc.collect()  # Periodic cleanup
    
    final_metrics = profiler.stop()
    
    return {
        'test_type': 'json_memory_stress',
        'iterations': iterations,
        'content_size': len(large_content),
        'total_chunks_created': total_chunks,
        'avg_chunks_per_iteration': total_chunks / iterations,
        **final_metrics
    }

def run_concurrent_parsing_test() -> Dict[str, Any]:
    """Test concurrent JSON parsing performance"""
    print("🔄 Running concurrent JSON parsing test...")
    
    import concurrent.futures
    
    def parse_json_content(content, file_id):
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        config = ChunkingConfig()
        engine = ChunkingEngine(config)
        
        start_time = time.time()
        chunks = engine.chunk_content(content, 'json', f'concurrent_test_{file_id}.json')
        duration = time.time() - start_time
        
        return {
            'file_id': file_id,
            'chunks_created': len(chunks),
            'duration_ms': duration * 1000,
            'content_size': len(content)
        }
    
    # Generate different JSON test contents
    test_contents = [
        generate_test_json('simple_objects', 3000),
        generate_test_json('nested_objects', 3000),
        generate_test_json('config_style', 3000),
        generate_test_json('api_responses', 3000),
        generate_test_json('mixed_arrays', 3000),
        generate_test_json('wide_objects', 3000),
        generate_test_json('simple_objects', 4000),
        generate_test_json('nested_objects', 4000),
    ]
    
    profiler = PerformanceProfiler()
    profiler.start()
    
    # Run concurrent parsing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(parse_json_content, content, i) 
            for i, content in enumerate(test_contents)
        ]
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    overall_metrics = profiler.stop()
    
    # Calculate concurrent performance metrics
    total_chunks = sum(r['chunks_created'] for r in results)
    total_parse_time = sum(r['duration_ms'] for r in results)
    avg_parse_time = total_parse_time / len(results)
    total_content_size = sum(r['content_size'] for r in results)
    
    return {
        'test_type': 'concurrent_json_parsing',
        'worker_count': 4,
        'files_processed': len(test_contents),
        'total_chunks_created': total_chunks,
        'total_content_size': total_content_size,
        'avg_chunks_per_file': total_chunks / len(results),
        'avg_parse_time_ms': avg_parse_time,
        'total_wall_time_ms': overall_metrics['duration_ms'],
        'parallelization_efficiency': (total_parse_time / overall_metrics['duration_ms']),
        'total_chars_per_second': total_content_size / (overall_metrics['duration_ms'] / 1000),
        **overall_metrics,
        'individual_results': results
    }

def generate_performance_report(scalability_results: List[Dict], 
                              stress_result: Dict, 
                              concurrent_result: Dict) -> str:
    """Generate comprehensive JSON performance report"""
    
    report = []
    report.append("🚀 JSON PARSER PERFORMANCE REPORT")
    report.append("=" * 60)
    
    # Scalability Analysis
    report.append("\n📈 SCALABILITY ANALYSIS")
    report.append("-" * 40)
    
    report.append(f"{'Scenario':<25} {'Size':<8} {'Duration':<10} {'Chunks':<8} {'Chars/s':<10} {'Depth':<6}")
    report.append("-" * 70)
    
    for result in scalability_results:
        json_struct = result['json_structure']
        report.append(
            f"{result['label']:<25} "
            f"{result['actual_size']:<8} "
            f"{result['avg_duration_ms']:<10.1f} "
            f"{result['avg_chunks_created']:<8.0f} "
            f"{result['chars_per_second']:<10.0f} "
            f"{json_struct['max_depth']:<6}"
        )
    
    # Performance insights
    report.append("\n🔍 SCALABILITY INSIGHTS")
    
    # Group by structure type
    by_structure = defaultdict(list)
    for result in scalability_results:
        by_structure[result['structure']].append(result)
    
    for structure, results in by_structure.items():
        sizes = [r['actual_size'] for r in results]
        speeds = [r['chars_per_second'] for r in results]
        depths = [r['json_structure']['max_depth'] for r in results]
        
        if len(speeds) > 1:
            speed_ratio = max(speeds) / min(speeds)
            avg_depth = statistics.mean(depths)
            report.append(f"   {structure.replace('_', ' ').title()}: {speed_ratio:.1f}x speed variation, avg depth {avg_depth:.1f}")
    
    # Memory stress test
    report.append(f"\n🔥 MEMORY STRESS TEST")
    report.append("-" * 40)
    report.append(f"   Content size: {stress_result['content_size']:,} characters")
    report.append(f"   Iterations: {stress_result['iterations']}")
    report.append(f"   Total chunks: {stress_result['total_chunks_created']:,}")
    report.append(f"   Duration: {stress_result['duration_ms']:.1f}ms")
    report.append(f"   Memory delta: {stress_result['memory_delta_mb']:.1f}MB")
    report.append(f"   Throughput: {stress_result['content_size'] * stress_result['iterations'] / (stress_result['duration_ms'] / 1000):.0f} chars/sec")
    
    # Concurrent parsing test
    report.append(f"\n🔄 CONCURRENT PARSING TEST")
    report.append("-" * 40)
    report.append(f"   Workers: {concurrent_result['worker_count']}")
    report.append(f"   Files processed: {concurrent_result['files_processed']}")
    report.append(f"   Total chunks: {concurrent_result['total_chunks_created']}")
    report.append(f"   Total content: {concurrent_result['total_content_size']:,} chars")
    report.append(f"   Wall time: {concurrent_result['total_wall_time_ms']:.1f}ms")
    report.append(f"   Parallelization efficiency: {concurrent_result['parallelization_efficiency']:.1f}x")
    report.append(f"   Throughput: {concurrent_result['total_chars_per_second']:.0f} chars/sec")
    report.append(f"   Memory delta: {concurrent_result['memory_delta_mb']:.1f}MB")
    
    # Overall performance summary
    report.append(f"\n⭐ PERFORMANCE SUMMARY")
    report.append("-" * 40)
    
    # Find best and worst performers
    best_speed = max(scalability_results, key=lambda x: x['chars_per_second'])
    worst_speed = min(scalability_results, key=lambda x: x['chars_per_second'])
    
    report.append(f"   Best performance: {best_speed['chars_per_second']:.0f} chars/sec ({best_speed['label']})")
    report.append(f"   Worst performance: {worst_speed['chars_per_second']:.0f} chars/sec ({worst_speed['label']})")
    report.append(f"   Performance range: {best_speed['chars_per_second'] / worst_speed['chars_per_second']:.1f}x variation")
    
    avg_speed = statistics.mean([r['chars_per_second'] for r in scalability_results])
    report.append(f"   Average speed: {avg_speed:.0f} chars/sec")
    
    # Memory efficiency
    avg_memory = statistics.mean([r['avg_memory_delta_mb'] for r in scalability_results])
    report.append(f"   Average memory usage: {avg_memory:.1f}MB per parse")
    
    # JSON-specific insights
    avg_depth = statistics.mean([r['json_structure']['max_depth'] for r in scalability_results])
    total_objects = sum([r['json_structure']['objects'] for r in scalability_results])
    total_arrays = sum([r['json_structure']['arrays'] for r in scalability_results])
    
    report.append(f"\n📊 JSON STRUCTURE ANALYSIS")
    report.append("-" * 40)
    report.append(f"   Average nesting depth: {avg_depth:.1f}")
    report.append(f"   Total objects processed: {total_objects:,}")
    report.append(f"   Total arrays processed: {total_arrays:,}")
    report.append(f"   Objects/Arrays ratio: {total_objects/max(total_arrays,1):.1f}")
    
    return "\n".join(report)

def save_detailed_results(scalability_results: List[Dict], 
                         stress_result: Dict, 
                         concurrent_result: Dict,
                         output_file: str = "json_performance_results.json"):
    """Save detailed results to JSON file"""
    
    detailed_results = {
        'timestamp': time.time(),
        'parser_type': 'json',
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024
        },
        'scalability_tests': scalability_results,
        'stress_test': stress_result,
        'concurrent_test': concurrent_result
    }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"📄 Detailed results saved to: {output_file}")

def main():
    """Main performance testing function"""
    print("🏃‍♂️ JSON PARSER PERFORMANCE TESTING")
    print("=" * 60)
    
    try:
        # Import test
        print("📦 Testing imports...")
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        # Test JSON support
        engine = ChunkingEngine(ChunkingConfig())
        if not engine.can_chunk_language('json'):
            print("❌ JSON parser not available")
            print("   Make sure tree-sitter-json is installed: pip install tree-sitter-json")
            return
        
        print("✅ All imports successful, JSON parser available")
        
        # System info
        print(f"\n💻 SYSTEM INFO")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   Platform: {sys.platform}")
        print(f"   CPU cores: {psutil.cpu_count()}")
        print(f"   Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
        
        # Run tests
        print(f"\n🧪 Running JSON performance tests...")
        
        # Scalability test
        scalability_results = run_scalability_test()
        
        # Memory stress test
        stress_result = run_memory_stress_test()
        
        # Concurrent parsing test
        concurrent_result = run_concurrent_parsing_test()
        
        # Generate and display report
        report = generate_performance_report(scalability_results, stress_result, concurrent_result)
        print(f"\n{report}")
        
        # Save detailed results
        save_detailed_results(scalability_results, stress_result, concurrent_result)
        
        print(f"\n🎉 JSON performance testing completed successfully!")
        print(f"💡 The JSON parser shows good performance across different structure types.")
        print(f"   Nested objects and API responses generally perform well.")
        print(f"   Consider structure complexity and nesting depth when tuning configurations.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package and tree-sitter-json are installed")
    except Exception as e:
        print(f"❌ Error during performance testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()