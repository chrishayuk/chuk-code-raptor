# Sample Documentation

A comprehensive example demonstrating various Markdown features for semantic chunking.

## Table of Contents

- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Getting Started

This section covers the basic setup and installation process.

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.11 or higher
- Node.js 18+ (for JavaScript examples)
- Rust 1.70+ (for Rust examples)

### Installation

Install the package using pip:

```bash
pip install chuk-code-raptor
```

For development installation:

```bash
git clone https://github.com/your-org/chuk-code-raptor.git
cd chuk-code-raptor
pip install -e ".[dev]"
```

### Quick Start

Here's a simple example to get you started:

```python
from chuk_code_raptor.chunking import ChunkingEngine

# Create engine with default settings
engine = ChunkingEngine()

# Chunk a Python file
chunks = engine.chunk_file("example.py")

# Examine the results
for chunk in chunks:
    print(f"{chunk.chunk_type.value}: {chunk.content_preview}")
```

## API Reference

The main API consists of several key components.

### ChunkingEngine

The primary interface for chunking operations.

#### Constructor

```python
ChunkingEngine(config: ChunkingConfig = None)
```

**Parameters:**
- `config` (optional): Configuration object for chunking behavior

#### Methods

##### chunk_file()

Chunk a file into semantic chunks.

```python
def chunk_file(self, file_path: str, language: str = None) -> List[SemanticChunk]:
```

**Parameters:**
- `file_path`: Path to the file to chunk
- `language` (optional): Programming language, auto-detected if not provided

**Returns:**
- List of `SemanticChunk` objects

**Example:**
```python
chunks = engine.chunk_file("src/main.py", "python")
```

##### chunk_content()

Chunk content string directly.

```python
def chunk_content(self, content: str, language: str, file_path: str) -> List[SemanticChunk]:
```

**Parameters:**
- `content`: String content to chunk
- `language`: Programming language
- `file_path`: Source file path (for metadata)

### SemanticChunk

Represents a semantic chunk of content with rich metadata.

#### Properties

- `id`: Unique identifier
- `content`: The actual content
- `start_line`, `end_line`: Line boundaries
- `chunk_type`: Type of chunk (function, class, etc.)
- `semantic_tags`: List of semantic tags
- `dependencies`: Dependencies on other chunks
- `metadata`: Additional metadata

#### Methods

##### add_semantic_tag()

Add a semantic tag to the chunk.

```python
def add_semantic_tag(self, name: str, confidence: float = 1.0, source: str = "manual"):
```

## Configuration

Configure the chunking behavior using `ChunkingConfig`.

### Basic Configuration

```python
from chuk_code_raptor.chunking import ChunkingConfig, ChunkingStrategy

config = ChunkingConfig(
    target_chunk_size=800,
    min_chunk_size=50,
    max_chunk_size=2000,
    primary_strategy=ChunkingStrategy.STRUCTURAL
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `target_chunk_size` | int | 800 | Preferred chunk size in characters |
| `min_chunk_size` | int | 50 | Minimum chunk size |
| `max_chunk_size` | int | 2000 | Maximum chunk size |
| `preserve_atomic_nodes` | bool | True | Don't split functions/classes |
| `enable_dependency_tracking` | bool | True | Track dependencies |
| `primary_strategy` | ChunkingStrategy | STRUCTURAL | Chunking strategy |

### Predefined Configurations

Several predefined configurations are available:

- `DEFAULT_CONFIG`: Balanced settings for general use
- `FAST_CONFIG`: Optimized for speed
- `PRECISE_CONFIG`: Optimized for quality
- `LARGE_FILES_CONFIG`: For processing large codebases

```python
from chuk_code_raptor.chunking import PRECISE_CONFIG

engine = ChunkingEngine(PRECISE_CONFIG)
```

## Examples

### Processing Multiple Files

```python
import os
from pathlib import Path

def process_project(project_path):
    engine = ChunkingEngine()
    all_chunks = []
    
    # Find all Python files
    for file_path in Path(project_path).rglob("*.py"):
        if file_path.is_file():
            chunks = engine.chunk_file(str(file_path))
            all_chunks.extend(chunks)
    
    return all_chunks

# Process entire project
chunks = process_project("./src")
print(f"Found {len(chunks)} chunks across the project")
```

### Analyzing Chunk Dependencies

```python
def analyze_dependencies(chunks):
    dependency_graph = {}
    
    for chunk in chunks:
        if chunk.dependencies:
            dependency_graph[chunk.id] = chunk.dependencies
    
    return dependency_graph

# Build dependency graph
deps = analyze_dependencies(chunks)
for chunk_id, dependencies in deps.items():
    print(f"{chunk_id} depends on: {dependencies}")
```

### Custom Semantic Analysis

```python
def add_custom_tags(chunks):
    for chunk in chunks:
        # Add performance-critical tag
        if any(keyword in chunk.content.lower() 
               for keyword in ['performance', 'optimize', 'fast']):
            chunk.add_semantic_tag('performance_critical', confidence=0.8)
        
        # Add API endpoint tag
        if '@app.route' in chunk.content or 'def api_' in chunk.content:
            chunk.add_semantic_tag('api_endpoint', confidence=0.9)

add_custom_tags(chunks)
```

## Troubleshooting

### Common Issues

#### ImportError: tree-sitter not found

**Problem:** Tree-sitter dependencies are missing.

**Solution:** Install the required language parsers:

```bash
pip install tree-sitter-python tree-sitter-javascript
```

#### No chunks created

**Problem:** The file language is not detected correctly.

**Solution:** Specify the language explicitly:

```python
chunks = engine.chunk_file("myfile.txt", language="python")
```

#### Large chunks not split

**Problem:** Atomic nodes are preserved even when large.

**Solution:** Adjust configuration:

```python
config = ChunkingConfig(
    preserve_atomic_nodes=False,
    max_chunk_size=1000
)
```

### Debug Mode

Enable debug logging to see what's happening:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

engine = ChunkingEngine()
chunks = engine.chunk_file("debug_me.py")
```

### Performance Issues

For large files, use the optimized configuration:

```python
from chuk_code_raptor.chunking import LARGE_FILES_CONFIG

engine = ChunkingEngine(LARGE_FILES_CONFIG)
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/your-org/chuk-code-raptor/issues)
2. Review the [API documentation](https://chuk-code-raptor.readthedocs.io/)
3. Join our [Discord community](https://discord.gg/coderagtor)

---

## Advanced Topics

### Custom Parsers

You can create custom parsers for new languages:

```python
from chuk_code_raptor.chunking.parsers import register_parser
from chuk_code_raptor.chunking.tree_sitter_chunker import BaseSemanticTreeSitterChunker

class MyLanguageParser(BaseSemanticTreeSitterChunker):
    def _get_language_module_name(self):
        return "tree_sitter_mylang"
    
    # Implement other required methods...

register_parser(MyLanguageParser, ['mylang'])
```

### Extending Semantic Analysis

Add custom semantic analysis:

```python
def analyze_complexity(chunk):
    # Simple complexity metric based on nesting
    nesting_level = chunk.content.count('    ')  # 4-space indents
    complexity = min(nesting_level / 10.0, 1.0)
    
    chunk.set_quality_score(QualityMetric.COMPLEXITY, complexity)

for chunk in chunks:
    analyze_complexity(chunk)
```

### Integration with RAG Systems

Prepare chunks for RAG applications:

```python
def prepare_for_rag(chunks):
    rag_docs = []
    
    for chunk in chunks:
        if isinstance(chunk, SemanticCodeChunk):
            doc = {
                'content': chunk.content,
                'metadata': {
                    'file_path': chunk.file_path,
                    'chunk_type': chunk.chunk_type.value,
                    'semantic_tags': [tag.name for tag in chunk.semantic_tags],
                    'dependencies': chunk.dependencies,
                    'function_calls': chunk.function_calls,
                    'imports': chunk.imports
                }
            }
            rag_docs.append(doc)
    
    return rag_docs

rag_documents = prepare_for_rag(chunks)
```

This comprehensive documentation demonstrates the system's capabilities while providing practical examples for users.