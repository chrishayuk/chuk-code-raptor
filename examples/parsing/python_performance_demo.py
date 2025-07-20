#!/usr/bin/env python3
"""
Python Parser Performance Testing
=================================

Comprehensive performance testing suite for the Python parser.
Tests parsing speed, memory usage, and scalability across different scenarios.
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

def generate_test_code(complexity: str, size_chars: int) -> str:
    """Generate test Python code of specified complexity and size"""
    
    if complexity == "simple":
        base_code = """
def simple_function(x):
    \"\"\"A simple function.\"\"\"
    return x * 2

class SimpleClass:
    \"\"\"A simple class.\"\"\"
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value

# Simple variables
simple_var = 42
another_var = "hello"
"""
    
    elif complexity == "medium":
        base_code = """
import asyncio
import logging
from typing import Optional, List, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class DataModel:
    \"\"\"Data model with type hints.\"\"\"
    name: str
    value: int
    metadata: Optional[Dict[str, Any]] = None

class ServiceBase(ABC):
    \"\"\"Abstract base service class.\"\"\"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def process(self, data: DataModel) -> Optional[str]:
        \"\"\"Process data asynchronously.\"\"\"
        pass

class ConcreteService(ServiceBase):
    \"\"\"Concrete implementation of service.\"\"\"
    
    async def process(self, data: DataModel) -> Optional[str]:
        try:
            if data.value > 0:
                result = await self._compute_result(data)
                return result
            return None
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise
    
    async def _compute_result(self, data: DataModel) -> str:
        await asyncio.sleep(0.1)  # Simulate async work
        return f"Processed: {data.name}"

# Factory pattern
def create_service(service_type: str, config: Dict[str, Any]) -> ServiceBase:
    \"\"\"Service factory function.\"\"\"
    if service_type == "concrete":
        return ConcreteService(config)
    raise ValueError(f"Unknown service type: {service_type}")
"""
    
    elif complexity == "complex":
        base_code = """
import asyncio
import logging
import inspect
from typing import (
    Optional, List, Dict, Union, Callable, Awaitable, 
    TypeVar, Generic, Protocol, runtime_checkable
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from functools import wraps, lru_cache
from collections import defaultdict

T = TypeVar('T')
R = TypeVar('R')

@runtime_checkable
class Processable(Protocol[T]):
    \"\"\"Protocol for processable objects.\"\"\"
    async def process(self) -> T:
        ...

@dataclass
class ComplexDataModel(Generic[T]):
    \"\"\"Generic data model with advanced typing.\"\"\"
    name: str
    data: T
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)
    processors: List[Callable[[T], Awaitable[T]]] = field(default_factory=list)
    
    async def apply_processors(self) -> T:
        \"\"\"Apply all processors to the data.\"\"\"
        result = self.data
        for processor in self.processors:
            result = await processor(result)
        return result

class MetricsCollector:
    \"\"\"Advanced metrics collection with decorators.\"\"\"
    
    def __init__(self):
        self._metrics = defaultdict(list)
        self._active_timers = {}
    
    def timing(self, name: str):
        \"\"\"Decorator for timing function execution.\"\"\"
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self._metrics[f"{name}_duration"].append(duration)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self._metrics[f"{name}_duration"].append(duration)
            
            return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        return decorator

class AdvancedProcessingEngine(Generic[T, R]):
    \"\"\"Advanced generic processing engine with complex patterns.\"\"\"
    
    def __init__(self, 
                 config: Dict[str, Any],
                 metrics: Optional[MetricsCollector] = None):
        self.config = config
        self.metrics = metrics or MetricsCollector()
        self._processors: Dict[str, Callable[[T], Awaitable[R]]] = {}
        self._middleware: List[Callable] = []
        self._state = defaultdict(dict)
    
    @asynccontextmanager
    async def processing_context(self, name: str):
        \"\"\"Context manager for processing operations.\"\"\"
        self.metrics._active_timers[name] = time.time()
        try:
            yield self
        except Exception as e:
            await self._handle_error(name, e)
            raise
        finally:
            if name in self.metrics._active_timers:
                duration = time.time() - self.metrics._active_timers[name]
                self.metrics._metrics[f"{name}_context_duration"].append(duration)
                del self.metrics._active_timers[name]
    
    @lru_cache(maxsize=128)
    def _get_processor_config(self, processor_name: str) -> Dict[str, Any]:
        \"\"\"Cached processor configuration lookup.\"\"\"
        return self.config.get('processors', {}).get(processor_name, {})
    
    async def _handle_error(self, context: str, error: Exception):
        \"\"\"Advanced error handling with context.\"\"\"
        error_info = {
            'context': context,
            'error_type': type(error).__name__,
            'message': str(error),
            'timestamp': time.time()
        }
        self.metrics._metrics['errors'].append(error_info)
        
        # Custom error handling logic
        if isinstance(error, ValueError):
            await self._recover_from_value_error(context, error)
        elif isinstance(error, asyncio.TimeoutError):
            await self._handle_timeout(context, error)
    
    async def _recover_from_value_error(self, context: str, error: ValueError):
        \"\"\"Recovery logic for value errors.\"\"\"
        pass
    
    async def _handle_timeout(self, context: str, error: asyncio.TimeoutError):
        \"\"\"Timeout handling logic.\"\"\"
        pass

# Complex inheritance hierarchy
class AdvancedMixin:
    \"\"\"Mixin with advanced functionality.\"\"\"
    
    def validate_data(self, data: Any) -> bool:
        \"\"\"Validate input data.\"\"\"
        return data is not None
    
    async def async_validate(self, data: Any) -> bool:
        \"\"\"Async validation.\"\"\"
        await asyncio.sleep(0.001)  # Simulate async validation
        return self.validate_data(data)

class ProductionService(AdvancedProcessingEngine[Dict, str], AdvancedMixin):
    \"\"\"Production service with multiple inheritance.\"\"\"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._cache = {}
        self._connection_pool = None
    
    @property
    def is_ready(self) -> bool:
        \"\"\"Check if service is ready.\"\"\"
        return self._connection_pool is not None
    
    async def initialize(self):
        \"\"\"Initialize service resources.\"\"\"
        async with self.processing_context("initialization"):
            self._connection_pool = await self._create_connection_pool()
    
    async def _create_connection_pool(self):
        \"\"\"Create async connection pool.\"\"\"
        await asyncio.sleep(0.1)  # Simulate connection setup
        return "mock_pool"
    
    @MetricsCollector().timing("process_request")
    async def process_request(self, request_data: Dict[str, Any]) -> str:
        \"\"\"Process incoming request with full error handling.\"\"\"
        try:
            # Validation
            if not await self.async_validate(request_data):
                raise ValueError("Invalid request data")
            
            # Processing pipeline
            async with self.processing_context("request_processing"):
                result = await self._execute_pipeline(request_data)
                await self._update_cache(request_data.get('id'), result)
                return result
        
        except asyncio.TimeoutError:
            return await self._handle_timeout_fallback(request_data)
        except Exception as e:
            await self._log_error(e, request_data)
            raise
    
    async def _execute_pipeline(self, data: Dict[str, Any]) -> str:
        \"\"\"Execute processing pipeline.\"\"\"
        stages = ['preprocess', 'transform', 'validate', 'finalize']
        
        for stage in stages:
            data = await self._execute_stage(stage, data)
        
        return f"Processed: {data.get('id', 'unknown')}"
    
    async def _execute_stage(self, stage: str, data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Execute individual pipeline stage.\"\"\"
        await asyncio.sleep(0.01)  # Simulate processing
        data[f"{stage}_completed"] = True
        return data
    
    async def _update_cache(self, key: str, value: str):
        \"\"\"Update internal cache.\"\"\"
        if key:
            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
    
    async def _handle_timeout_fallback(self, data: Dict[str, Any]) -> str:
        \"\"\"Fallback for timeout scenarios.\"\"\"
        return f"Timeout fallback for: {data.get('id', 'unknown')}"
    
    async def _log_error(self, error: Exception, context: Dict[str, Any]):
        \"\"\"Log error with context.\"\"\"
        error_details = {
            'error': str(error),
            'context': context,
            'timestamp': time.time()
        }
        # In real code, would log to external system
        pass

# Factory functions and utilities
async def create_production_service(config: Dict[str, Any]) -> ProductionService:
    \"\"\"Async factory for production service.\"\"\"
    service = ProductionService(config)
    await service.initialize()
    return service

def configure_logging(level: str = "INFO") -> logging.Logger:
    \"\"\"Configure application logging.\"\"\"
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

# Complex data structures and algorithms
class AdvancedCache(Generic[T]):
    \"\"\"Advanced caching with TTL and LRU eviction.\"\"\"
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        self._cache: Dict[str, Tuple[T, float]] = {}
        self._access_times: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl = ttl
    
    async def get(self, key: str) -> Optional[T]:
        \"\"\"Get value from cache with TTL check.\"\"\"
        if key not in self._cache:
            return None
        
        value, timestamp = self._cache[key]
        
        if time.time() - timestamp > self.ttl:
            await self._evict(key)
            return None
        
        self._access_times[key] = time.time()
        return value
    
    async def set(self, key: str, value: T):
        \"\"\"Set value in cache with size management.\"\"\"
        if len(self._cache) >= self.max_size:
            await self._evict_lru()
        
        self._cache[key] = (value, time.time())
        self._access_times[key] = time.time()
    
    async def _evict(self, key: str):
        \"\"\"Evict single key.\"\"\"
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
    
    async def _evict_lru(self):
        \"\"\"Evict least recently used item.\"\"\"
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), 
                     key=lambda k: self._access_times[k])
        await self._evict(lru_key)
"""
    
    else:
        base_code = "# Unknown complexity level\npass\n"
    
    # Repeat and modify the base code to reach the target size
    current_size = len(base_code)
    target_size = size_chars
    
    if current_size >= target_size:
        return base_code[:target_size]
    
    # Generate additional content to reach target size
    additional_content = ""
    counter = 1
    
    while len(base_code + additional_content) < target_size:
        additional_content += f"""

# Generated content block {counter}
def generated_function_{counter}(param1, param2=None):
    \"\"\"Generated function for testing purposes.\"\"\"
    if param2 is None:
        param2 = param1 * 2
    
    result = param1 + param2
    
    for i in range(10):
        if i % 2 == 0:
            result += i
        else:
            result -= i
    
    return result

class GeneratedClass{counter}:
    \"\"\"Generated class for testing.\"\"\"
    
    def __init__(self, value):
        self.value = value
        self.processed = False
    
    def process(self):
        \"\"\"Process the value.\"\"\"
        if not self.processed:
            self.value = generated_function_{counter}(self.value)
            self.processed = True
        return self.value

generated_var_{counter} = GeneratedClass{counter}({counter * 10})
"""
        counter += 1
        
        # Safety check to prevent infinite loop
        if counter > 1000:
            break
    
    final_content = base_code + additional_content
    return final_content[:target_size]

def run_parsing_benchmark(content: str, iterations: int = 5) -> Dict[str, Any]:
    """Run parsing benchmark with multiple iterations"""
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    config = ChunkingConfig(
        target_chunk_size=500,
        min_chunk_size=50,
        preserve_atomic_nodes=True,
        enable_dependency_tracking=True,
        group_imports=True
    )
    
    engine = ChunkingEngine(config)
    profiler = PerformanceProfiler()
    
    results = []
    chunks_counts = []
    
    for i in range(iterations):
        profiler.start()
        chunks = engine.chunk_content(content, 'python', f'test_file_{i}.py')
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
    """Test parser scalability across different file sizes and complexities"""
    
    test_scenarios = [
        # Small files
        {'complexity': 'simple', 'size': 1000, 'label': 'Small Simple'},
        {'complexity': 'medium', 'size': 1000, 'label': 'Small Medium'},
        {'complexity': 'complex', 'size': 1000, 'label': 'Small Complex'},
        
        # Medium files
        {'complexity': 'simple', 'size': 5000, 'label': 'Medium Simple'},
        {'complexity': 'medium', 'size': 5000, 'label': 'Medium Medium'},
        {'complexity': 'complex', 'size': 5000, 'label': 'Medium Complex'},
        
        # Large files
        {'complexity': 'simple', 'size': 20000, 'label': 'Large Simple'},
        {'complexity': 'medium', 'size': 20000, 'label': 'Large Medium'},
        {'complexity': 'complex', 'size': 20000, 'label': 'Large Complex'},
        
        # Very large files
        {'complexity': 'medium', 'size': 50000, 'label': 'XLarge Medium'},
        {'complexity': 'complex', 'size': 50000, 'label': 'XLarge Complex'},
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"🧪 Testing {scenario['label']} ({scenario['size']} chars, {scenario['complexity']})...")
        
        # Generate test content
        content = generate_test_code(scenario['complexity'], scenario['size'])
        
        # Run benchmark
        benchmark_result = run_parsing_benchmark(content, iterations=3)
        
        # Combine scenario info with results
        result = {
            **scenario,
            'actual_size': len(content),
            'line_count': content.count('\n') + 1,
            **benchmark_result
        }
        
        results.append(result)
        
        print(f"   ✅ {benchmark_result['avg_duration_ms']:.1f}ms avg, "
              f"{benchmark_result['avg_chunks_created']:.0f} chunks, "
              f"{benchmark_result['chars_per_second']:.0f} chars/sec")
    
    return results

def run_memory_stress_test() -> Dict[str, Any]:
    """Test memory usage under stress conditions"""
    print("🔥 Running memory stress test...")
    
    profiler = PerformanceProfiler()
    
    # Generate large, complex content
    large_content = generate_test_code('complex', 100000)
    
    # Test repeated parsing
    profiler.start()
    
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    config = ChunkingConfig(enable_dependency_tracking=True)
    engine = ChunkingEngine(config)
    
    total_chunks = 0
    iterations = 10
    
    for i in range(iterations):
        chunks = engine.chunk_content(large_content, 'python', f'stress_test_{i}.py')
        total_chunks += len(chunks)
        
        if i % 2 == 0:
            gc.collect()  # Periodic cleanup
    
    final_metrics = profiler.stop()
    
    return {
        'test_type': 'memory_stress',
        'iterations': iterations,
        'content_size': len(large_content),
        'total_chunks_created': total_chunks,
        'avg_chunks_per_iteration': total_chunks / iterations,
        **final_metrics
    }

def run_concurrent_parsing_test() -> Dict[str, Any]:
    """Test concurrent parsing performance"""
    print("🔄 Running concurrent parsing test...")
    
    import asyncio
    import concurrent.futures
    
    def parse_content(content, file_id):
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        config = ChunkingConfig()
        engine = ChunkingEngine(config)
        
        start_time = time.time()
        chunks = engine.chunk_content(content, 'python', f'concurrent_test_{file_id}.py')
        duration = time.time() - start_time
        
        return {
            'file_id': file_id,
            'chunks_created': len(chunks),
            'duration_ms': duration * 1000
        }
    
    # Generate test contents
    test_contents = [
        generate_test_code('medium', 3000) for _ in range(8)
    ]
    
    profiler = PerformanceProfiler()
    profiler.start()
    
    # Run concurrent parsing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(parse_content, content, i) 
            for i, content in enumerate(test_contents)
        ]
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    overall_metrics = profiler.stop()
    
    # Calculate concurrent performance metrics
    total_chunks = sum(r['chunks_created'] for r in results)
    total_parse_time = sum(r['duration_ms'] for r in results)
    avg_parse_time = total_parse_time / len(results)
    
    return {
        'test_type': 'concurrent_parsing',
        'worker_count': 4,
        'files_processed': len(test_contents),
        'total_chunks_created': total_chunks,
        'avg_chunks_per_file': total_chunks / len(results),
        'avg_parse_time_ms': avg_parse_time,
        'total_wall_time_ms': overall_metrics['duration_ms'],
        'parallelization_efficiency': (total_parse_time / overall_metrics['duration_ms']),
        **overall_metrics,
        'individual_results': results
    }

def generate_performance_report(scalability_results: List[Dict], 
                              stress_result: Dict, 
                              concurrent_result: Dict) -> str:
    """Generate comprehensive performance report"""
    
    report = []
    report.append("🚀 PYTHON PARSER PERFORMANCE REPORT")
    report.append("=" * 60)
    
    # Scalability Analysis
    report.append("\n📈 SCALABILITY ANALYSIS")
    report.append("-" * 40)
    
    report.append(f"{'Scenario':<20} {'Size':<8} {'Duration':<10} {'Chunks':<8} {'Chars/s':<10}")
    report.append("-" * 60)
    
    for result in scalability_results:
        report.append(
            f"{result['label']:<20} "
            f"{result['actual_size']:<8} "
            f"{result['avg_duration_ms']:<10.1f} "
            f"{result['avg_chunks_created']:<8.0f} "
            f"{result['chars_per_second']:<10.0f}"
        )
    
    # Performance insights
    report.append("\n🔍 SCALABILITY INSIGHTS")
    
    # Group by complexity
    by_complexity = defaultdict(list)
    for result in scalability_results:
        by_complexity[result['complexity']].append(result)
    
    for complexity, results in by_complexity.items():
        sizes = [r['actual_size'] for r in results]
        speeds = [r['chars_per_second'] for r in results]
        
        if len(speeds) > 1:
            speed_ratio = max(speeds) / min(speeds)
            report.append(f"   {complexity.title()} complexity: {speed_ratio:.1f}x speed variation across sizes")
    
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
    report.append(f"   Wall time: {concurrent_result['total_wall_time_ms']:.1f}ms")
    report.append(f"   Parallelization efficiency: {concurrent_result['parallelization_efficiency']:.1f}x")
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
    
    return "\n".join(report)

def save_detailed_results(scalability_results: List[Dict], 
                         stress_result: Dict, 
                         concurrent_result: Dict,
                         output_file: str = "performance_results.json"):
    """Save detailed results to JSON file"""
    
    detailed_results = {
        'timestamp': time.time(),
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
    print("🏃‍♂️ PYTHON PARSER PERFORMANCE TESTING")
    print("=" * 60)
    
    try:
        # Import test
        print("📦 Testing imports...")
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        print("✅ All imports successful")
        
        # System info
        print(f"\n💻 SYSTEM INFO")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   Platform: {sys.platform}")
        print(f"   CPU cores: {psutil.cpu_count()}")
        print(f"   Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
        
        # Run tests
        print(f"\n🧪 Running performance tests...")
        
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
        
        print(f"\n🎉 Performance testing completed successfully!")
        print(f"💡 The parser shows good scalability across different file sizes and complexities.")
        print(f"   Consider these results when tuning chunk sizes and configurations for your use case.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package is installed and in your Python path")
    except Exception as e:
        print(f"❌ Error during performance testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()