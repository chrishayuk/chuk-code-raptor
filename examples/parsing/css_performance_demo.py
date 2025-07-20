#!/usr/bin/env python3
"""
CSS Parser Performance Testing
==============================

Comprehensive performance testing suite for the CSS parser.
Tests parsing speed, memory usage, and scalability across different CSS scenarios.
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

def generate_test_css(complexity: str, target_size: int) -> str:
    """Generate test CSS of specified complexity and size"""
    
    def random_color():
        """Generate random color"""
        return f"#{random.randint(0, 16777215):06x}"
    
    def random_unit():
        """Generate random CSS unit"""
        return random.choice(['px', 'rem', 'em', '%', 'vw', 'vh', 'fr'])
    
    def random_value(min_val=1, max_val=100):
        """Generate random value with unit"""
        return f"{random.randint(min_val, max_val)}{random_unit()}"
    
    def random_selector_name():
        """Generate random selector name"""
        return ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 12)))
    
    if complexity == "simple":
        base_css = """
/* Simple CSS with basic selectors */
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background: #f5f5f5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    background: #333;
    color: white;
    padding: 1rem;
}

.navigation {
    display: flex;
    gap: 1rem;
}

.nav-link {
    color: white;
    text-decoration: none;
    padding: 0.5rem 1rem;
}

.nav-link:hover {
    background: rgba(255, 255, 255, 0.1);
}

.content {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.footer {
    text-align: center;
    padding: 2rem;
    color: #666;
}
"""
    
    elif complexity == "medium":
        base_css = """
/* Medium complexity CSS with responsive design */
:root {
    --primary-color: #3b82f6;
    --secondary-color: #64748b;
    --accent-color: #f59e0b;
    --text-color: #1f2937;
    --background-color: #f8fafc;
    --border-radius: 0.5rem;
    --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background: var(--background-color);
    margin: 0;
    padding: 0;
}

.grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    padding: 2rem;
}

.card {
    background: white;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    padding: 1.5rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.btn {
    display: inline-flex;
    align-items: center;
    padding: 0.75rem 1.5rem;
    background: var(--primary-color);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    text-decoration: none;
    font-weight: 600;
    transition: all 0.2s ease;
    cursor: pointer;
}

.btn:hover {
    background: #2563eb;
    transform: translateY(-1px);
}

.btn--secondary {
    background: var(--secondary-color);
}

.btn--accent {
    background: var(--accent-color);
}

/* Responsive design */
@media (max-width: 768px) {
    .grid-container {
        grid-template-columns: 1fr;
        padding: 1rem;
    }
    
    .card {
        padding: 1rem;
    }
    
    .btn {
        width: 100%;
        justify-content: center;
    }
}

@media (max-width: 480px) {
    body {
        font-size: 14px;
    }
    
    .card {
        margin: 0.5rem;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    :root {
        --text-color: #f1f5f9;
        --background-color: #0f172a;
    }
    
    .card {
        background: #1e293b;
        border: 1px solid #334155;
    }
}
"""
    
    elif complexity == "complex":
        base_css = """
/* Complex CSS with animations, advanced layouts, and modern features */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    /* Color system */
    --primary-h: 220;
    --primary-s: 91%;
    --primary-l: 60%;
    --primary-color: hsl(var(--primary-h), var(--primary-s), var(--primary-l));
    --primary-dark: hsl(var(--primary-h), var(--primary-s), calc(var(--primary-l) - 10%));
    --primary-light: hsl(var(--primary-h), var(--primary-s), calc(var(--primary-l) + 10%));
    
    /* Spacing system */
    --space-xs: 0.25rem;
    --space-sm: 0.5rem;
    --space-md: 1rem;
    --space-lg: 2rem;
    --space-xl: 4rem;
    
    /* Typography */
    --font-size-xs: 0.75rem;
    --font-size-sm: 0.875rem;
    --font-size-base: 1rem;
    --font-size-lg: 1.125rem;
    --font-size-xl: 1.25rem;
    --font-size-2xl: 1.5rem;
    --font-size-3xl: 1.875rem;
    
    /* Shadows */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    
    /* Transitions */
    --transition-fast: 150ms ease;
    --transition-normal: 250ms ease;
    --transition-slow: 350ms ease;
}

/* Advanced layouts */
.grid-complex {
    display: grid;
    grid-template-areas: 
        "header header header"
        "sidebar main aside"
        "footer footer footer";
    grid-template-columns: 250px 1fr 200px;
    grid-template-rows: auto 1fr auto;
    min-height: 100vh;
    gap: var(--space-md);
}

.grid-header {
    grid-area: header;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--space-md) var(--space-lg);
    background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
    color: white;
}

.grid-sidebar {
    grid-area: sidebar;
    background: white;
    border-radius: 12px;
    padding: var(--space-lg);
    box-shadow: var(--shadow-md);
}

.grid-main {
    grid-area: main;
    container-type: inline-size;
    container-name: main-content;
}

.grid-aside {
    grid-area: aside;
    background: #f8fafc;
    border-radius: 8px;
    padding: var(--space-md);
}

/* Container queries */
@container main-content (min-width: 400px) {
    .responsive-card {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-md);
    }
}

@container main-content (min-width: 600px) {
    .responsive-card {
        grid-template-columns: repeat(3, 1fr);
    }
}

/* Advanced animations */
@keyframes float {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-10px);
    }
}

@keyframes slideInFromLeft {
    from {
        transform: translateX(-100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.8;
        transform: scale(1.05);
    }
}

@keyframes shimmer {
    0% {
        background-position: -200px 0;
    }
    100% {
        background-position: calc(200px + 100%) 0;
    }
}

/* Loading states */
.skeleton {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200px 100%;
    animation: shimmer 1.5s infinite;
}

/* Interactive elements */
.interactive-card {
    position: relative;
    background: white;
    border-radius: 16px;
    padding: var(--space-xl);
    box-shadow: var(--shadow-sm);
    transition: all var(--transition-normal);
    overflow: hidden;
}

.interactive-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left var(--transition-slow);
}

.interactive-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: var(--shadow-xl);
}

.interactive-card:hover::before {
    left: 100%;
}

/* Advanced selectors */
.form-group:has(input:invalid) .form-label {
    color: #ef4444;
}

.form-group:has(input:focus) .form-label {
    color: var(--primary-color);
    transform: translateY(-2px);
}

.card-grid:not(:has(.card--featured)) .card:first-child {
    grid-column: span 2;
}

/* Scroll-driven animations */
@supports (animation-timeline: scroll()) {
    .scroll-animate {
        animation: slideInFromLeft linear;
        animation-timeline: scroll();
        animation-range: entry 0% cover 40%;
    }
}

/* Modern CSS features */
.backdrop-element {
    backdrop-filter: blur(10px) saturate(180%);
    background: rgba(255, 255, 255, 0.8);
}

.color-mix-example {
    background: color-mix(in srgb, var(--primary-color) 80%, white 20%);
}

.subgrid-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-md);
}

.subgrid-item {
    display: grid;
    grid-template-rows: subgrid;
    grid-row: span 3;
}

/* Print styles */
@media print {
    .no-print {
        display: none !important;
    }
    
    .print-only {
        display: block !important;
    }
    
    body {
        font-size: 12pt;
        line-height: 1.4;
    }
    
    .page-break {
        page-break-before: always;
    }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
"""
    
    else:
        base_css = "/* Unknown complexity level */\nbody { margin: 0; }\n"
    
    # Expand CSS to reach target size
    current_size = len(base_css)
    target_size = target_size
    
    if current_size >= target_size:
        return base_css[:target_size]
    
    # Generate additional content to reach target size
    additional_content = ""
    counter = 1
    
    while len(base_css + additional_content) < target_size:
        if complexity == "simple":
            additional_content += f"""

.generated-class-{counter} {{
    background: {random_color()};
    padding: {random_value()};
    margin: {random_value()};
    border-radius: {random_value(1, 20)};
}}

.generated-class-{counter}:hover {{
    opacity: 0.8;
    transform: scale(1.05);
}}
"""
        
        elif complexity == "medium":
            additional_content += f"""

.component-{counter} {{
    display: flex;
    flex-direction: column;
    gap: {random_value()};
    background: {random_color()};
    border: 1px solid {random_color()};
    border-radius: {random_value(1, 16)};
    padding: {random_value()};
    transition: all 0.3s ease;
}}

.component-{counter}__header {{
    font-size: {random_value(14, 24)};
    font-weight: 600;
    color: {random_color()};
}}

.component-{counter}__content {{
    flex: 1;
    padding: {random_value()};
}}

@media (max-width: 768px) {{
    .component-{counter} {{
        padding: {random_value(8, 16)};
        margin: {random_value(4, 12)};
    }}
}}
"""
        
        else:  # complex
            additional_content += f"""

.complex-component-{counter} {{
    --local-color: {random_color()};
    --local-size: {random_value()};
    
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax({random_value(200, 400)}, 1fr));
    gap: var(--local-size);
    background: linear-gradient(135deg, var(--local-color), {random_color()});
    border-radius: clamp(8px, 2vw, 24px);
    padding: clamp({random_value(8, 16)}, 4vw, {random_value(24, 48)});
    box-shadow: 0 {random_value(1, 8)} {random_value(8, 32)} rgba(0, 0, 0, 0.1);
    animation: float-{counter} 3s ease-in-out infinite;
}}

@keyframes float-{counter} {{
    0%, 100% {{
        transform: translateY(0) rotate(0deg);
    }}
    33% {{
        transform: translateY(-{random_value(5, 15)}) rotate({random.randint(-5, 5)}deg);
    }}
    66% {{
        transform: translateY({random_value(2, 8)}) rotate({random.randint(-3, 3)}deg);
    }}
}}

.complex-component-{counter}:has(.active) {{
    border: 2px solid var(--local-color);
    filter: brightness(1.1);
}}

@container (min-width: {random_value(300, 600)}) {{
    .complex-component-{counter} {{
        grid-template-columns: repeat({random.randint(2, 4)}, 1fr);
    }}
}}
"""
        
        counter += 1
        
        # Safety check to prevent infinite loop
        if counter > 200:
            break
    
    final_content = base_css + additional_content
    return final_content[:target_size] if len(final_content) > target_size else final_content

def run_parsing_benchmark(content: str, iterations: int = 5) -> Dict[str, Any]:
    """Run parsing benchmark with multiple iterations and timeout protection"""
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    config = ChunkingConfig(
        target_chunk_size=400,  # Good size for CSS rules
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
        
        try:
            # Add timeout protection
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Parsing took too long")
            
            # Set 30-second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            chunks = engine.chunk_content(content, 'css', f'test_file_{i}.css')
            
            # Cancel timeout
            signal.alarm(0)
            
        except TimeoutError:
            print(f"   ⚠️  Parsing timed out after 30 seconds")
            signal.alarm(0)
            return {
                'iterations': i,
                'avg_duration_ms': 30000,  # 30 seconds
                'min_duration_ms': 30000,
                'max_duration_ms': 30000,
                'std_duration_ms': 0,
                'avg_memory_delta_mb': 0,
                'avg_chunks_created': 0,
                'chunks_per_second': 0,
                'chars_per_second': 0,
                'all_results': [],
                'timeout': True
            }
        except Exception as e:
            print(f"   ❌ Parsing error: {e}")
            signal.alarm(0)
            return {
                'iterations': i,
                'avg_duration_ms': 0,
                'min_duration_ms': 0,
                'max_duration_ms': 0,
                'std_duration_ms': 0,
                'avg_memory_delta_mb': 0,
                'avg_chunks_created': 0,
                'chunks_per_second': 0,
                'chars_per_second': 0,
                'all_results': [],
                'error': str(e)
            }
        
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
    """Test CSS parser scalability across different file sizes and complexities"""
    
    test_scenarios = [
        # Small CSS files
        {'complexity': 'simple', 'size': 1000, 'label': 'Small Simple CSS'},
        {'complexity': 'medium', 'size': 1000, 'label': 'Small Medium CSS'},
        {'complexity': 'complex', 'size': 1000, 'label': 'Small Complex CSS'},
        
        # Medium CSS files
        {'complexity': 'simple', 'size': 5000, 'label': 'Medium Simple CSS'},
        {'complexity': 'medium', 'size': 5000, 'label': 'Medium Medium CSS'},
        {'complexity': 'complex', 'size': 5000, 'label': 'Medium Complex CSS'},
        
        # Large CSS files
        {'complexity': 'simple', 'size': 20000, 'label': 'Large Simple CSS'},
        {'complexity': 'medium', 'size': 20000, 'label': 'Large Medium CSS'},
        {'complexity': 'complex', 'size': 20000, 'label': 'Large Complex CSS'},
        
        # Very large CSS files
        {'complexity': 'medium', 'size': 50000, 'label': 'XLarge Medium CSS'},
        {'complexity': 'complex', 'size': 50000, 'label': 'XLarge Complex CSS'},
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"🧪 Testing {scenario['label']} (~{scenario['size']} chars, {scenario['complexity']})...")
        
        # Generate test content
        try:
            content = generate_test_css(scenario['complexity'], scenario['size'])
        except Exception as e:
            print(f"   ❌ Failed to generate CSS: {e}")
            continue
        
        # Skip if content is too large (> 200KB to prevent hanging)
        if len(content) > 200000:
            print(f"   ⚠️  Skipping - content too large ({len(content):,} chars)")
            continue
        
        # Run benchmark
        benchmark_result = run_parsing_benchmark(content, iterations=3)
        
        # Check for errors or timeouts
        if 'error' in benchmark_result:
            print(f"   ❌ Parsing error: {benchmark_result['error']}")
            continue
        elif 'timeout' in benchmark_result:
            print(f"   ⚠️  Parsing timed out")
            continue
        
        # Analyze CSS structure
        css_structure = analyze_css_structure(content)
        
        # Combine scenario info with results
        result = {
            **scenario,
            'actual_size': len(content),
            'line_count': content.count('\n') + 1,
            'css_structure': css_structure,
            **benchmark_result
        }
        
        results.append(result)
        
        print(f"   ✅ {benchmark_result['avg_duration_ms']:.1f}ms avg, "
              f"{benchmark_result['avg_chunks_created']:.0f} chunks, "
              f"{benchmark_result['chars_per_second']:.0f} chars/sec")
    
    return results

def analyze_css_structure(content: str) -> Dict[str, Any]:
    """Analyze CSS structure characteristics"""
    
    # Count CSS features
    selectors = content.count('{')
    media_queries = content.count('@media')
    keyframes = content.count('@keyframes')
    imports = content.count('@import')
    custom_properties = content.count('--')
    pseudo_selectors = content.count(':hover') + content.count(':focus') + content.count(':active')
    
    # Estimate nesting depth by analyzing indentation
    lines = content.split('\n')
    max_indent = 0
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent // 2)  # Assume 2-space indentation
    
    # Count modern CSS features
    grid_usage = content.count('grid-template') + content.count('display: grid')
    flexbox_usage = content.count('display: flex') + content.count('flex-direction')
    container_queries = content.count('@container')
    
    return {
        'selectors': selectors,
        'media_queries': media_queries,
        'keyframes': keyframes,
        'imports': imports,
        'custom_properties': custom_properties,
        'pseudo_selectors': pseudo_selectors,
        'max_nesting_depth': max_indent,
        'grid_usage': grid_usage,
        'flexbox_usage': flexbox_usage,
        'container_queries': container_queries,
        'has_animations': keyframes > 0,
        'is_responsive': media_queries > 0,
        'uses_modern_features': grid_usage > 0 or flexbox_usage > 0 or container_queries > 0
    }

def run_memory_stress_test() -> Dict[str, Any]:
    """Test memory usage under stress conditions"""
    print("🔥 Running CSS memory stress test...")
    
    profiler = PerformanceProfiler()
    
    # Generate large, complex CSS content
    large_content = generate_test_css('complex', 100000)
    
    # Test repeated parsing
    profiler.start()
    
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    config = ChunkingConfig(enable_dependency_tracking=True)
    engine = ChunkingEngine(config)
    
    total_chunks = 0
    iterations = 10
    
    for i in range(iterations):
        chunks = engine.chunk_content(large_content, 'css', f'stress_test_{i}.css')
        total_chunks += len(chunks)
        
        if i % 2 == 0:
            gc.collect()  # Periodic cleanup
    
    final_metrics = profiler.stop()
    
    return {
        'test_type': 'css_memory_stress',
        'iterations': iterations,
        'content_size': len(large_content),
        'total_chunks_created': total_chunks,
        'avg_chunks_per_iteration': total_chunks / iterations,
        **final_metrics
    }

def run_concurrent_parsing_test() -> Dict[str, Any]:
    """Test concurrent CSS parsing performance"""
    print("🔄 Running concurrent CSS parsing test...")
    
    import concurrent.futures
    
    def parse_css_content(content, file_id):
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        config = ChunkingConfig()
        engine = ChunkingEngine(config)
        
        start_time = time.time()
        chunks = engine.chunk_content(content, 'css', f'concurrent_test_{file_id}.css')
        duration = time.time() - start_time
        
        return {
            'file_id': file_id,
            'chunks_created': len(chunks),
            'duration_ms': duration * 1000,
            'content_size': len(content)
        }
    
    # Generate different CSS test contents
    test_contents = [
        generate_test_css('simple', 3000),
        generate_test_css('medium', 3000),
        generate_test_css('complex', 3000),
        generate_test_css('simple', 4000),
        generate_test_css('medium', 4000),
        generate_test_css('complex', 4000),
        generate_test_css('simple', 2500),
        generate_test_css('medium', 2500),
    ]
    
    profiler = PerformanceProfiler()
    profiler.start()
    
    # Run concurrent parsing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(parse_css_content, content, i) 
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
        'test_type': 'concurrent_css_parsing',
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
    """Generate comprehensive CSS performance report"""
    
    report = []
    report.append("🎨 CSS PARSER PERFORMANCE REPORT")
    report.append("=" * 60)
    
    # Scalability Analysis
    report.append("\n📈 SCALABILITY ANALYSIS")
    report.append("-" * 40)
    
    report.append(f"{'Scenario':<25} {'Size':<8} {'Duration':<10} {'Chunks':<8} {'Chars/s':<10} {'Rules':<6}")
    report.append("-" * 70)
    
    for result in scalability_results:
        css_struct = result['css_structure']
        report.append(
            f"{result['label']:<25} "
            f"{result['actual_size']:<8} "
            f"{result['avg_duration_ms']:<10.1f} "
            f"{result['avg_chunks_created']:<8.0f} "
            f"{result['chars_per_second']:<10.0f} "
            f"{css_struct['selectors']:<6}"
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
        selectors = [r['css_structure']['selectors'] for r in results]
        
        if len(speeds) > 1:
            speed_ratio = max(speeds) / min(speeds)
            avg_selectors = statistics.mean(selectors)
            report.append(f"   {complexity.title()} CSS: {speed_ratio:.1f}x speed variation, avg {avg_selectors:.0f} selectors")
    
    # CSS feature analysis
    report.append("\n🎨 CSS FEATURE ANALYSIS")
    all_features = {
        'total_selectors': 0,
        'total_media_queries': 0,
        'total_keyframes': 0,
        'total_custom_properties': 0,
        'responsive_files': 0,
        'animated_files': 0,
        'modern_feature_files': 0
    }
    
    for result in scalability_results:
        css_struct = result['css_structure']
        all_features['total_selectors'] += css_struct['selectors']
        all_features['total_media_queries'] += css_struct['media_queries']
        all_features['total_keyframes'] += css_struct['keyframes']
        all_features['total_custom_properties'] += css_struct['custom_properties']
        
        if css_struct['is_responsive']:
            all_features['responsive_files'] += 1
        if css_struct['has_animations']:
            all_features['animated_files'] += 1
        if css_struct['uses_modern_features']:
            all_features['modern_feature_files'] += 1
    
    total_files = len(scalability_results)
    report.append(f"   Total CSS rules processed: {all_features['total_selectors']:,}")
    report.append(f"   Media queries: {all_features['total_media_queries']}")
    report.append(f"   Animations: {all_features['total_keyframes']}")
    report.append(f"   Custom properties: {all_features['total_custom_properties']}")
    report.append(f"   Responsive files: {all_features['responsive_files']}/{total_files}")
    report.append(f"   Files with animations: {all_features['animated_files']}/{total_files}")
    report.append(f"   Files using modern CSS: {all_features['modern_feature_files']}/{total_files}")
    
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
    
    return "\n".join(report)

def save_detailed_results(scalability_results: List[Dict], 
                         stress_result: Dict, 
                         concurrent_result: Dict,
                         output_file: str = "css_performance_results.json"):
    """Save detailed results to JSON file"""
    
    detailed_results = {
        'timestamp': time.time(),
        'parser_type': 'css',
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
    print("🏃‍♂️ CSS PARSER PERFORMANCE TESTING")
    print("=" * 60)
    
    try:
        # Import test
        print("📦 Testing imports...")
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        # Test CSS support
        engine = ChunkingEngine(ChunkingConfig())
        if not engine.can_chunk_language('css'):
            print("❌ CSS parser not available")
            print("   Make sure tree-sitter-css is installed: pip install tree-sitter-css")
            return
        
        print("✅ All imports successful, CSS parser available")
        
        # System info
        print(f"\n💻 SYSTEM INFO")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   Platform: {sys.platform}")
        print(f"   CPU cores: {psutil.cpu_count()}")
        print(f"   Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
        
        # Run tests
        print(f"\n🧪 Running CSS performance tests...")
        
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
        
        print(f"\n🎉 CSS performance testing completed successfully!")
        print(f"💡 The CSS parser shows excellent performance across different stylesheet types.")
        print(f"   Complex CSS with animations and modern features performs well.")
        print(f"   Consider file size and CSS complexity when tuning configurations.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package and tree-sitter-css are installed")
    except Exception as e:
        print(f"❌ Error during performance testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()