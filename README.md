# Supported Languages and Parsers

ChukCodeRaptor includes specialized parsers for different programming languages and file formats, each optimized for semantic chunking and code analysis.

## Parser Types

### 🔧 Tree-sitter Based Parsers
High-quality AST-based parsing with semantic understanding:

| Language | Extensions | Parser Class | Features |
|----------|------------|--------------|----------|
| **Python** | `.py`, `.pyx`, `.pyi`, `.pyw` | `PythonParser` | Code quality assessment, import analysis, architectural role detection |
| **JavaScript/TypeScript** | `.js`, `.jsx`, `.ts`, `.tsx`, `.mjs`, `.cjs` | `JavaScriptParser` | Framework detection (React hooks, etc.), async patterns |
| **HTML** | `.html`, `.htm`, `.xhtml` | `HTMLParser` | Semantic element processing, CSS relationship analysis |
| **CSS** | `.css`, `.scss`, `.sass`, `.less` | `CSSParser` | Selector analysis, specificity calculation, responsive design detection |
| **Markdown** | `.md`, `.markdown`, `.mdown`, `.mkd` | `MarkdownParser` | Section-based chunking, automatic fallback to manual parsing |
| **JSON** | `.json`, `.jsonl`, `.ndjson` | `JSONParser` | Schema analysis, pattern detection, hierarchy preservation |
| **XML** | `.xml`, `.xhtml`, `.svg`, `.sitemap`, `.xsd`, `.wsdl`, `.pom` | `XMLParser` | Namespace-aware, semantic categorization, heuristic fallback |
| **Rust** | `.rs` | `RustParser` | Trait/impl detection, ownership analysis, visibility modifiers |

### 🌐 Feed Parsers (Specialized)
Dedicated parsers for web syndication formats:

| Format | Extensions | Parser Class | Features |
|--------|------------|--------------|----------|
| **RSS/Atom Coordinator** | `.rss`, `.atom`, `.xml`, `.feed`, `.rdf` | `FeedParserCoordinator` | Auto-detection, intelligent routing to specialized parsers |
| **RSS 2.0** | `.rss`, `.xml` | `RSSParser` | iTunes podcast support, media enclosures, Dublin Core metadata |
| **RSS 1.0 (RDF)** | `.rdf`, `.xml`, `.rss` | `RDFParser` | RDF structure handling, academic feeds, Dublin Core extraction |
| **Atom 1.0** | `.atom`, `.xml` | `AtomParser` | Full Atom spec compliance, content negotiation, rich metadata |

### ⚡ Simple Tree-sitter Parsers
Basic AST parsing for additional languages:

| Language | Extensions | Description |
|----------|------------|-------------|
| **Go** | `.go` | Function and type declarations, package imports |
| **Java** | `.java` | Classes, methods, interfaces, package structure |
| **C++** | `.cpp`, `.cxx`, `.cc`, `.hpp`, `.hxx`, `.h++` | Functions, classes, namespaces, preprocessor directives |
| **C** | `.c`, `.h` | Function definitions, struct declarations, includes |
| **YAML** | `.yaml`, `.yml` | Block and flow constructs, document structure |
| **TOML** | `.toml` | Tables, key-value pairs, configuration sections |

### 📝 Heuristic Parsers
Pattern-based parsing for complex formats:

| Format | Extensions | Parser Class | Features |
|--------|------------|--------------|----------|
| **LaTeX** | `.tex`, `.latex`, `.ltx`, `.sty`, `.cls`, `.bib` | `HeuristicLaTeXParser` | Environment detection, mathematical content, academic structure |
| **Plain Text** | `.txt`, `.log`, `.cfg`, `.ini`, `.conf`, `.properties`, `.env` | `TextParser` | Log parsing, configuration files, structured text |

## Parser Features

### 🎯 Semantic Analysis
- **Code Quality Assessment**: Docstring analysis, type hint coverage, complexity metrics
- **Architectural Roles**: Detection of data access, business logic, presentation patterns
- **Dependency Tracking**: Import analysis, function calls, cross-references
- **Content Classification**: Technical vs. narrative content, difficulty levels

### 🔍 Enhanced Chunking Strategies
- **Semantic Boundaries**: Respect language constructs (functions, classes, sections)
- **Content-Aware Sizing**: Adaptive chunk sizes based on content complexity
- **Hierarchical Structure**: Maintain document organization and relationships
- **Fallback Mechanisms**: Graceful degradation when specialized parsing fails

### 📊 Metadata Extraction
- **Structural Information**: Element types, nesting levels, relationships
- **Quality Metrics**: Code complexity, documentation coverage, best practices
- **Semantic Tags**: Automated categorization and feature detection
- **Cross-References**: Links, imports, dependencies between chunks

## Usage Examples

### Basic Language Detection
```python
from chuk_code_raptor.chunking import ChunkingPipeline

# Auto-detect parser based on file extension
pipeline = ChunkingPipeline()
chunks = pipeline.chunk_file("example.py")  # Uses PythonParser
chunks = pipeline.chunk_file("data.json")   # Uses JSONParser
chunks = pipeline.chunk_file("feed.rss")    # Uses FeedParserCoordinator → RSSParser
```

### Parser-Specific Configuration
```python
# Configure feed parser for podcast processing
config = ChunkingConfig()
config.rss_extract_itunes_metadata = True
config.rss_extract_media_metadata = True
config.rss_max_items_per_feed = 50

# Configure LaTeX parser for academic papers
config.latex_parsing_strategy = 'fine'
config.latex_math_detection_level = 'aggressive'
config.latex_preserve_all_environments = True

pipeline = ChunkingPipeline(config)
```

### Manual Parser Selection
```python
from chuk_code_raptor.chunking.parsers import PythonParser, MarkdownParser

# Use specific parser regardless of file extension
python_parser = PythonParser(config)
chunks = python_parser.parse(python_code, context)

# Chain parsers for mixed content
markdown_parser = MarkdownParser(config)
chunks = markdown_parser.parse(documentation, context)
```

## Installation Requirements

### Core Dependencies
```bash
# Basic tree-sitter support
pip install tree-sitter

# Comprehensive language support (recommended)
pip install tree-sitter-languages
```

### Individual Language Packages
```bash
# For specific languages only
pip install tree-sitter-python tree-sitter-javascript
pip install tree-sitter-html tree-sitter-css
pip install tree-sitter-json tree-sitter-xml
pip install tree-sitter-rust tree-sitter-go
```

### Feed Processing Dependencies
```bash
# XML processing for RSS/Atom feeds
pip install tree-sitter-xml

# Optional: Enhanced date parsing
pip install python-dateutil
```

## Parser Fallback Chain

Each parser has a built-in fallback strategy:

1. **Primary Parser**: Language-specific tree-sitter or heuristic parsing
2. **Secondary Fallback**: Generic parser for the file type category
3. **Final Fallback**: Plain text parser with basic structure detection

Example fallback chain for RSS feeds:
```
RSS file → FeedParserCoordinator → RSSParser → XMLParser → TextParser
```

## Performance Characteristics

| Parser Type | Speed | Accuracy | Memory Usage |
|-------------|-------|----------|--------------|
| Tree-sitter | Fast | High | Low |
| Heuristic | Medium | Medium-High | Low |
| Simple | Fast | Medium | Low |
| Specialized | Medium | Very High | Medium |

## Contributing New Parsers

To add support for a new language:

1. **Simple Parser**: Extend `SimpleTreeSitterParser` in `parsers/simple.py`
2. **Dedicated Parser**: Create new file following existing patterns
3. **Heuristic Parser**: Extend `HeuristicParser` for non-AST languages
4. **Update Configuration**: Add parser entry to `config.yaml`

See the contributing guide for detailed parser development instructions.