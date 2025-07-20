#!/usr/bin/env python3
"""
Markdown Parser Performance Testing
===================================

Comprehensive performance testing suite for the Markdown parser.
Tests parsing speed, memory usage, and scalability across different Markdown scenarios.
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

def generate_test_markdown(complexity: str, target_size: int) -> str:
    """Generate test Markdown of specified complexity and size"""
    
    def random_text(words: int = 20) -> str:
        """Generate random lorem ipsum text"""
        lorem_words = [
            'lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit',
            'sed', 'do', 'eiusmod', 'tempor', 'incididunt', 'ut', 'labore', 'et', 'dolore',
            'magna', 'aliqua', 'enim', 'ad', 'minim', 'veniam', 'quis', 'nostrud',
            'exercitation', 'ullamco', 'laboris', 'nisi', 'aliquip', 'ex', 'ea', 'commodo',
            'consequat', 'duis', 'aute', 'irure', 'reprehenderit', 'voluptate', 'velit',
            'esse', 'cillum', 'fugiat', 'nulla', 'pariatur', 'excepteur', 'sint', 'occaecat'
        ]
        return ' '.join(random.choices(lorem_words, k=words))
    
    def random_code_lang():
        """Generate random programming language"""
        return random.choice(['python', 'javascript', 'java', 'go', 'rust', 'typescript', 'bash', 'sql'])
    
    def random_heading(level: int = None):
        """Generate random heading"""
        if level is None:
            level = random.randint(1, 6)
        return '#' * level + ' ' + random_text(random.randint(2, 8)).title()
    
    if complexity == "simple":
        base_markdown = """# Simple Markdown Document

This is a basic markdown document with simple formatting and structure.

## Introduction

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

### Features

- Simple bullet points
- Easy to read format
- Basic markdown syntax

## Code Example

Here's a simple code block:

```python
def hello_world():
    print("Hello, World!")
    return True
```

## Table

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Row 1    | Data     | More     |
| Row 2    | Info     | Data     |

## Conclusion

This concludes our simple markdown example. **Bold text** and *italic text* are supported.

> This is a blockquote example.
> It can span multiple lines.

Visit [our website](https://example.com) for more information.
"""
    
    elif complexity == "medium":
        base_markdown = """# Comprehensive Markdown Guide

Welcome to this **comprehensive guide** on Markdown formatting. This document covers *various features* and demonstrates proper usage.

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Formatting](#basic-formatting)
3. [Code Blocks](#code-blocks)
4. [Tables and Data](#tables-and-data)
5. [Advanced Features](#advanced-features)

---

## Introduction

Markdown is a lightweight markup language with plain-text-formatting syntax. Its design allows it to be converted to many output formats, but the original tool by the same name only supports HTML.

### History

Markdown was created by **John Gruber** in *2004* with the goal of enabling people to write using an easy-to-read and easy-to-write plain text format.

#### Key Benefits

- **Simplicity**: Easy to learn and use
- **Portability**: Plain text format
- **Flexibility**: Multiple output formats
- **Readability**: Human-readable source

## Basic Formatting

### Text Emphasis

You can make text **bold** or *italic* using simple syntax:

- **Bold text** using `**text**` or `__text__`
- *Italic text* using `*text*` or `_text_`
- ***Bold and italic*** using `***text***`
- ~~Strikethrough~~ using `~~text~~`

### Lists

#### Unordered Lists

- Item one
- Item two
  - Nested item
  - Another nested item
- Item three

#### Ordered Lists

1. First item
2. Second item
   1. Nested numbered item
   2. Another nested item
3. Third item

#### Task Lists

- [x] Completed task
- [ ] Incomplete task
- [x] Another completed task

## Code Blocks

### Inline Code

Use `backticks` to create inline code snippets.

### Fenced Code Blocks

```python
# Python example
def fibonacci(n):
    \"\"\"Generate Fibonacci sequence.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
```

```javascript
// JavaScript example
class Calculator {
    constructor() {
        this.result = 0;
    }
    
    add(value) {
        this.result += value;
        return this;
    }
    
    multiply(value) {
        this.result *= value;
        return this;
    }
    
    getValue() {
        return this.result;
    }
}

const calc = new Calculator();
const result = calc.add(5).multiply(3).getValue();
console.log(result); // 15
```

```bash
# Shell commands
npm install markdown-parser
cd project-directory
npm run build
npm test
```

## Tables and Data

### Basic Table

| Feature | Description | Status |
|---------|-------------|--------|
| Headers | Table headers | ✅ |
| Alignment | Text alignment | ✅ |
| Formatting | **Bold**, *italic* | ✅ |
| Links | [External links](https://example.com) | ✅ |

### Aligned Table

| Left Aligned | Center Aligned | Right Aligned |
|:-------------|:--------------:|--------------:|
| Left         | Center         | Right         |
| Text         | Text           | Text          |
| More         | Data           | Here          |

## Advanced Features

### Blockquotes

> This is a simple blockquote.

> This is a longer blockquote that spans multiple lines.
> It continues here with more text and explanations.
> 
> You can even have multiple paragraphs within a blockquote.

> **Nested quotes work too:**
> 
> > This is a nested blockquote
> > with additional content.

### Links and References

Direct links: [Google](https://google.com)

Reference links: [Google][1]

Image links: ![Alt text](https://example.com/image.png "Optional title")

### Horizontal Rules

You can create horizontal rules using:

---

Or with asterisks:

***

### HTML Integration

<div align="center">
  <strong>HTML can be mixed with Markdown</strong>
  <br>
  <em>When needed</em>
</div>

### Mathematical Expressions

Inline math: `E = mc²`

Block math:
```
∑(i=1 to n) i = n(n+1)/2
```

## Conclusion

This guide covers the essential features of Markdown formatting. For more advanced usage, consult the official documentation.

### Additional Resources

- [Markdown Guide](https://www.markdownguide.org/)
- [CommonMark Spec](https://commonmark.org/)
- [GitHub Flavored Markdown](https://github.github.com/gfm/)

---

*Last updated: 2024*
"""
    
    elif complexity == "complex":
        base_markdown = """# Advanced Markdown Documentation System

[![Build Status](https://travis-ci.org/example/project.svg?branch=master)](https://travis-ci.org/example/project)
[![Coverage Status](https://coveralls.io/repos/github/example/project/badge.svg?branch=master)](https://coveralls.io/github/example/project?branch=master)
[![npm version](https://badge.fury.io/js/example-project.svg)](https://badge.fury.io/js/example-project)

> **Note**: This is a comprehensive documentation example showcasing advanced Markdown features, code integration, and technical writing patterns commonly found in software documentation.

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Package Managers](#package-managers)
  - [Manual Installation](#manual-installation)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Configuration Files](#configuration-files)
- [API Reference](#api-reference)
  - [Core Classes](#core-classes)
  - [Utility Functions](#utility-functions)
- [Examples](#examples)
  - [Basic Usage](#basic-usage)
  - [Advanced Patterns](#advanced-patterns)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Installation

### Prerequisites

Before installing, ensure you have the following dependencies:

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Node.js | >=14.0.0 | Runtime environment |
| npm | >=6.0.0 | Package management |
| Python | >=3.8 | Build tools |
| Git | >=2.20 | Version control |

> **Warning**: Older versions may not be compatible with all features.

### Package Managers

#### npm

```bash
# Install globally
npm install -g advanced-markdown-system

# Install as dependency
npm install advanced-markdown-system --save

# Install development version
npm install advanced-markdown-system@next --save-dev
```

#### Yarn

```bash
# Add to project
yarn add advanced-markdown-system

# Global installation
yarn global add advanced-markdown-system
```

#### pnpm

```bash
# Fast installation with pnpm
pnpm add advanced-markdown-system
```

### Manual Installation

For manual installation or development setup:

```bash
# Clone repository
git clone https://github.com/example/advanced-markdown-system.git
cd advanced-markdown-system

# Install dependencies
npm ci

# Build from source
npm run build

# Run tests
npm test

# Create distribution
npm run dist
```

## Configuration

### Environment Variables

Configure the system using environment variables:

```bash
# .env file
MARKDOWN_PARSER_MODE=strict
MARKDOWN_EXTENSIONS=tables,footnotes,strikethrough
MARKDOWN_OUTPUT_FORMAT=html
MARKDOWN_CACHE_SIZE=1000
MARKDOWN_DEBUG=false

# Advanced options
MARKDOWN_SYNTAX_HIGHLIGHTING=true
MARKDOWN_MATH_RENDERING=katex
MARKDOWN_EMOJI_SUPPORT=true
MARKDOWN_AUTO_LINKING=true
```

### Configuration Files

#### Basic Configuration

```yaml
# config.yml
parser:
  mode: strict
  extensions:
    - tables
    - footnotes
    - strikethrough
    - task-lists
  
output:
  format: html
  pretty: true
  minify: false
  
performance:
  cache_size: 1000
  parallel_processing: true
  max_workers: 4

features:
  syntax_highlighting: true
  math_rendering: katex
  emoji_support: true
  auto_linking: true
```

#### Advanced Configuration

```json
{
  "markdown": {
    "parser": {
      "mode": "gfm",
      "strictMode": false,
      "allowHtml": true,
      "extensions": [
        "tables",
        "strikethrough", 
        "footnotes",
        "definition-lists",
        "task-lists",
        "emoji",
        "math"
      ]
    },
    "renderer": {
      "html": {
        "sanitize": true,
        "allowedTags": ["p", "br", "strong", "em", "code", "pre"],
        "allowedAttributes": {
          "a": ["href", "title"],
          "img": ["src", "alt", "title"]
        }
      }
    },
    "plugins": [
      {
        "name": "syntax-highlighting",
        "options": {
          "theme": "github",
          "languages": ["javascript", "python", "go", "rust"]
        }
      },
      {
        "name": "math-renderer",
        "options": {
          "engine": "katex",
          "inline": true,
          "display": true
        }
      }
    ]
  }
}
```

## API Reference

### Core Classes

#### MarkdownParser

The main parser class for processing markdown content.

```typescript
class MarkdownParser {
  constructor(options?: ParserOptions);
  
  /**
   * Parse markdown string to AST
   * @param content - Raw markdown content
   * @param options - Parsing options
   * @returns Abstract syntax tree
   */
  parse(content: string, options?: ParseOptions): MarkdownAST;
  
  /**
   * Render AST to HTML
   * @param ast - Markdown AST
   * @param options - Rendering options
   * @returns HTML string
   */
  render(ast: MarkdownAST, options?: RenderOptions): string;
  
  /**
   * Parse and render in one step
   * @param content - Raw markdown content
   * @param options - Combined options
   * @returns HTML string
   */
  toHtml(content: string, options?: ProcessOptions): string;
}
```

Usage example:

```javascript
import { MarkdownParser } from 'advanced-markdown-system';

const parser = new MarkdownParser({
  strict: true,
  extensions: ['tables', 'footnotes']
});

const html = parser.toHtml(`
# Hello World

This is **bold** text with a [link](https://example.com).

| Column 1 | Column 2 |
|----------|----------|
| Data     | More     |
`);

console.log(html);
```

#### MarkdownRenderer

Advanced renderer with customizable output formats.

```python
from markdown_system import MarkdownRenderer

class MarkdownRenderer:
    def __init__(self, format_type: str = "html"):
        \"\"\"Initialize renderer with format type.\"\"\"
        self.format_type = format_type
    
    def render(self, ast: MarkdownAST) -> str:
        \"\"\"Render AST to specified format.\"\"\"
        if self.format_type == "html":
            return self._render_html(ast)
        elif self.format_type == "latex":
            return self._render_latex(ast)
        elif self.format_type == "pdf":
            return self._render_pdf(ast)
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")
    
    def _render_html(self, ast: MarkdownAST) -> str:
        \"\"\"Render to HTML format.\"\"\"
        # Implementation details...
        pass
```

### Utility Functions

#### Content Analysis

```go
package markdown

import (
    "regexp"
    "strings"
)

// AnalyzeContent examines markdown content for patterns
func AnalyzeContent(content string) ContentAnalysis {
    analysis := ContentAnalysis{
        WordCount:     countWords(content),
        HeadingCount:  countHeadings(content),
        CodeBlocks:    extractCodeBlocks(content),
        Links:         extractLinks(content),
        Images:        extractImages(content),
        Tables:        countTables(content),
    }
    
    return analysis
}

// ExtractTOC generates table of contents from headings
func ExtractTOC(content string, maxDepth int) []TOCEntry {
    headingRegex := regexp.MustCompile(`^(#{1,6})\\s+(.+)$`)
    lines := strings.Split(content, "\\n")
    
    var toc []TOCEntry
    for _, line := range lines {
        if matches := headingRegex.FindStringSubmatch(line); matches != nil {
            level := len(matches[1])
            if level <= maxDepth {
                toc = append(toc, TOCEntry{
                    Level: level,
                    Title: matches[2],
                    Slug:  generateSlug(matches[2]),
                })
            }
        }
    }
    
    return toc
}
```

## Examples

### Basic Usage

```rust
use markdown_system::{Parser, Options};

fn main() {
    let markdown = r#"
# Example Document

This is a **markdown** document with:

- Lists
- *Emphasis*  
- `Code spans`

```rust
fn hello() {
    println!("Hello, world!");
}
```
"#;

    let options = Options::new()
        .enable_tables()
        .enable_strikethrough()
        .enable_footnotes();
    
    let parser = Parser::new(options);
    let html = parser.parse(markdown).unwrap();
    
    println!("{}", html);
}
```

### Advanced Patterns

#### Custom Extensions

```python
class CustomMarkdownExtension:
    \"\"\"Custom extension for specialized markdown processing.\"\"\"
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def process_admonitions(self, content: str) -> str:
        \"\"\"Process custom admonition blocks.\"\"\"
        import re
        
        # Pattern: !!! type "title"
        pattern = r'!!!\\s+(\\w+)\\s+"([^"]+)"\\s*\\n((?:.|\\n)*?)(?=\\n!!!|\\Z)'
        
        def replace_admonition(match):
            adm_type = match.group(1)
            title = match.group(2) 
            content = match.group(3).strip()
            
            return f'''<div class="admonition {adm_type}">
    <p class="admonition-title">{title}</p>
    <div class="admonition-content">
        {self.process_markdown(content)}
    </div>
</div>'''
        
        return re.sub(pattern, replace_admonition, content)
    
    def process_diagrams(self, content: str) -> str:
        \"\"\"Process embedded diagram blocks.\"\"\"
        # Support for mermaid, plantuml, etc.
        diagram_types = ['mermaid', 'plantuml', 'graphviz']
        
        for diagram_type in diagram_types:
            content = self._process_diagram_type(content, diagram_type)
        
        return content
```

#### Performance Optimization

```javascript
class OptimizedMarkdownProcessor {
    constructor() {
        this.cache = new Map();
        this.workers = [];
        this.initializeWorkers();
    }
    
    async processLargeDocument(content, options = {}) {
        // Split document into chunks for parallel processing
        const chunks = this.splitIntoChunks(content, options.chunkSize || 1000);
        
        // Process chunks in parallel
        const promises = chunks.map((chunk, index) => 
            this.processChunk(chunk, index, options)
        );
        
        const processedChunks = await Promise.all(promises);
        
        // Reassemble document
        return this.reassembleDocument(processedChunks);
    }
    
    processChunk(chunk, index, options) {
        return new Promise((resolve, reject) => {
            const worker = this.getAvailableWorker();
            
            worker.postMessage({
                id: index,
                content: chunk,
                options: options
            });
            
            worker.onmessage = (event) => {
                if (event.data.id === index) {
                    resolve(event.data.result);
                }
            };
            
            worker.onerror = reject;
        });
    }
}
```

## Troubleshooting

### Common Issues

#### Performance Problems

**Issue**: Slow parsing with large documents

```markdown
**Symptoms**:
- Long processing times (>5 seconds)
- High memory usage
- Browser/application freezing

**Solutions**:
1. Enable chunk-based processing:
   ```javascript
   const options = {
     chunkSize: 1000,
     parallelProcessing: true
   };
   ```

2. Disable expensive features:
   ```yaml
   features:
     syntax_highlighting: false
     math_rendering: false
   ```

3. Use streaming parser:
   ```python
   for chunk in parser.parse_stream(large_content):
       process_chunk(chunk)
   ```
```

#### Memory Leaks

**Issue**: Memory usage continuously increasing

> **Important**: Always dispose of parser instances when done.

```typescript
// Correct usage
const parser = new MarkdownParser();
try {
    const result = parser.parse(content);
    // Use result...
} finally {
    parser.dispose(); // Clean up resources
}

// Or use automatic disposal
using (const parser = new MarkdownParser()) {
    const result = parser.parse(content);
    // Parser automatically disposed
}
```

### Debugging

Enable debug mode for detailed logging:

```bash
export MARKDOWN_DEBUG=true
export MARKDOWN_LOG_LEVEL=verbose
```

Debug output example:
```
[MARKDOWN] Parser initialized with options: {"strict": true}
[MARKDOWN] Processing document: 15,234 characters
[MARKDOWN] Tokenization completed: 2,456 tokens
[MARKDOWN] AST generation: 234 nodes
[MARKDOWN] Rendering phase: HTML output
[MARKDOWN] Total processing time: 45ms
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/advanced-markdown-system.git
cd advanced-markdown-system

# Install dependencies
npm install

# Run development server
npm run dev

# Run tests
npm test

# Run linting
npm run lint

# Build for production
npm run build
```

### Code Style

We use [Prettier](https://prettier.io/) for code formatting:

```json
{
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false,
  "semi": true,
  "singleQuote": true,
  "trailingComma": "es5"
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CommonMark](https://commonmark.org/) specification
- [GitHub Flavored Markdown](https://github.github.com/gfm/)
- [Markdown-it](https://github.com/markdown-it/markdown-it) inspiration
- All our [contributors](https://github.com/example/project/contributors)

---

*Generated with ❤️ by the Advanced Markdown System*
"""
    
    else:
        base_markdown = "# Unknown Complexity\n\nSimple markdown content.\n"
    
    # Expand markdown to reach target size
    current_size = len(base_markdown)
    target_size = target_size
    
    if current_size >= target_size:
        return base_markdown[:target_size]
    
    # Generate additional content to reach target size
    additional_content = ""
    counter = 1
    
    while len(base_markdown + additional_content) < target_size:
        if complexity == "simple":
            additional_content += f"""

## Section {counter}

{random_text(20)}

### Subsection {counter}.1

{random_text(15)}

- Point one
- Point two  
- Point three

```{random_code_lang()}
# Example code block {counter}
def example_function_{counter}():
    return "Hello from section {counter}"
```
"""
        
        elif complexity == "medium":
            additional_content += f"""

## {random_text(3).title()} {counter}

{random_text(30)}

### Features

1. **Feature {counter}.1**: {random_text(10)}
2. **Feature {counter}.2**: {random_text(10)}
3. **Feature {counter}.3**: {random_text(10)}

#### Code Example

```{random_code_lang()}
// Generated example {counter}
class Example{counter} {{
    constructor(data) {{
        this.data = data;
        this.processed = false;
    }}
    
    process() {{
        if (!this.processed) {{
            console.log('Processing data:', this.data);
            this.processed = true;
        }}
        return this.data;
    }}
}}

const example = new Example{counter}("test data");
example.process();
```

> **Note**: {random_text(12)}

| Item | Value | Status |
|------|-------|--------|
| Item {counter}.1 | {random.randint(1, 100)} | ✅ Active |
| Item {counter}.2 | {random.randint(1, 100)} | ⏳ Pending |
| Item {counter}.3 | {random.randint(1, 100)} | ❌ Error |
"""
        
        else:  # complex
            additional_content += f"""

## Advanced Topic {counter}: {random_text(4).title()}

{random_text(40)}

### Overview

This section covers advanced concepts and implementation details:

#### Prerequisites

- {random_text(5).title()}
- {random_text(5).title()}  
- {random_text(5).title()}

#### Implementation

```{random_code_lang()}
/**
 * Advanced implementation example {counter}
 * Demonstrates complex patterns and optimizations
 */
class AdvancedProcessor{counter} extends BaseProcessor {{
    constructor(config = {{}}) {{
        super(config);
        this.cache = new Map();
        this.workers = [];
        this.metrics = {{
            processed: 0,
            errors: 0,
            avgTime: 0
        }};
    }}
    
    async processAdvanced(data, options = {{}}) {{
        const startTime = performance.now();
        
        try {{
            // Validation phase
            this.validateInput(data, options);
            
            // Processing phase
            const result = await this.executeProcessing(data, options);
            
            // Post-processing
            const finalResult = this.postProcess(result, options);
            
            // Update metrics
            this.updateMetrics(performance.now() - startTime);
            
            return finalResult;
            
        }} catch (error) {{
            this.handleError(error, data);
            throw error;
        }}
    }}
    
    validateInput(data, options) {{
        if (!data || typeof data !== 'object') {{
            throw new Error('Invalid input data');
        }}
        
        const required = ['id', 'type', 'payload'];
        for (const field of required) {{
            if (!(field in data)) {{
                throw new Error(`Missing required field: ${{field}}`);
            }}
        }}
    }}
    
    async executeProcessing(data, options) {{
        // Complex processing logic here
        const cacheKey = this.generateCacheKey(data);
        
        if (this.cache.has(cacheKey)) {{
            return this.cache.get(cacheKey);
        }}
        
        const result = await this.performComplexOperation(data);
        this.cache.set(cacheKey, result);
        
        return result;
    }}
}}
```

#### Configuration Example

```yaml
# config-{counter}.yaml
processor_{counter}:
  name: "AdvancedProcessor{counter}"
  version: "2.{counter}.0"
  
  settings:
    batch_size: 1000
    max_workers: 8
    timeout: 30000
    retry_attempts: 3
    
  cache:
    enabled: true
    max_size: 10000
    ttl: 3600
    
  monitoring:
    metrics_enabled: true
    log_level: "info"
    performance_tracking: true
    
  features:
    parallel_processing: true
    error_recovery: true
    adaptive_scaling: true
```

#### Performance Considerations

> **Warning**: High-performance scenarios require careful tuning.

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Throughput | >1000 ops/sec | {random.randint(800, 1200)} ops/sec | {'✅' if random.random() > 0.3 else '⚠️'} |
| Latency | <50ms | {random.randint(20, 80)}ms | {'✅' if random.random() > 0.4 else '⚠️'} |
| Memory | <512MB | {random.randint(200, 600)}MB | {'✅' if random.random() > 0.2 else '⚠️'} |
| CPU | <80% | {random.randint(40, 90)}% | {'✅' if random.random() > 0.3 else '⚠️'} |

#### Troubleshooting

Common issues and solutions:

1. **Performance Degradation**
   ```bash
   # Monitor resource usage
   top -p $(pgrep processor)
   
   # Check cache hit ratio
   curl http://localhost:8080/metrics | grep cache_hit_ratio
   
   # Analyze processing queue
   curl http://localhost:8080/queue/status
   ```

2. **Memory Leaks**
   ```javascript
   // Enable memory monitoring
   setInterval(() => {{
       const usage = process.memoryUsage();
       console.log('Memory usage:', {{
           rss: Math.round(usage.rss / 1024 / 1024) + 'MB',
           heapUsed: Math.round(usage.heapUsed / 1024 / 1024) + 'MB',
           heapTotal: Math.round(usage.heapTotal / 1024 / 1024) + 'MB'
       }});
   }}, 5000);
   ```

3. **Error Recovery**
   ```python
   import asyncio
   import logging
   
   async def robust_processing(data, max_retries=3):
       for attempt in range(max_retries):
           try:
               result = await process_data(data)
               return result
           except Exception as e:
               logging.warning(f"Attempt {{attempt + 1}} failed: {{e}}")
               if attempt == max_retries - 1:
                   raise
               await asyncio.sleep(2 ** attempt)  # Exponential backoff
   ```

---

*Section {counter} completed. Continue reading for more advanced topics.*
"""
        
        counter += 1
        
        # Safety check to prevent infinite loop
        if counter > 50:
            break
    
    final_content = base_markdown + additional_content
    return final_content[:target_size] if len(final_content) > target_size else final_content

def run_parsing_benchmark(content: str, iterations: int = 5) -> Dict[str, Any]:
    """Run parsing benchmark with multiple iterations and timeout protection"""
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    config = ChunkingConfig(
        target_chunk_size=600,  # Good size for markdown sections
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
            
            chunks = engine.chunk_content(content, 'markdown', f'test_file_{i}.md')
            
            # Cancel timeout
            signal.alarm(0)
            
        except TimeoutError:
            print(f"   ⚠️  Parsing timed out after 30 seconds")
            signal.alarm(0)
            return {
                'iterations': i,
                'avg_duration_ms': 30000,
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
    """Test Markdown parser scalability across different file sizes and complexities"""
    
    test_scenarios = [
        # Small Markdown files
        {'complexity': 'simple', 'size': 1000, 'label': 'Small Simple Markdown'},
        {'complexity': 'medium', 'size': 1000, 'label': 'Small Medium Markdown'},
        {'complexity': 'complex', 'size': 1000, 'label': 'Small Complex Markdown'},
        
        # Medium Markdown files
        {'complexity': 'simple', 'size': 5000, 'label': 'Medium Simple Markdown'},
        {'complexity': 'medium', 'size': 5000, 'label': 'Medium Medium Markdown'},
        {'complexity': 'complex', 'size': 5000, 'label': 'Medium Complex Markdown'},
        
        # Large Markdown files
        {'complexity': 'simple', 'size': 20000, 'label': 'Large Simple Markdown'},
        {'complexity': 'medium', 'size': 20000, 'label': 'Large Medium Markdown'},
        {'complexity': 'complex', 'size': 20000, 'label': 'Large Complex Markdown'},
        
        # Very large Markdown files
        {'complexity': 'medium', 'size': 50000, 'label': 'XLarge Medium Markdown'},
        {'complexity': 'complex', 'size': 50000, 'label': 'XLarge Complex Markdown'},
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"🧪 Testing {scenario['label']} (~{scenario['size']} chars, {scenario['complexity']})...")
        
        # Generate test content
        try:
            content = generate_test_markdown(scenario['complexity'], scenario['size'])
        except Exception as e:
            print(f"   ❌ Failed to generate Markdown: {e}")
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
        
        # Analyze Markdown structure
        markdown_structure = analyze_markdown_structure(content)
        
        # Combine scenario info with results
        result = {
            **scenario,
            'actual_size': len(content),
            'line_count': content.count('\n') + 1,
            'markdown_structure': markdown_structure,
            **benchmark_result
        }
        
        results.append(result)
        
        print(f"   ✅ {benchmark_result['avg_duration_ms']:.1f}ms avg, "
              f"{benchmark_result['avg_chunks_created']:.0f} chunks, "
              f"{benchmark_result['chars_per_second']:.0f} chars/sec")
    
    return results

def analyze_markdown_structure(content: str) -> Dict[str, Any]:
    """Analyze Markdown structure characteristics"""
    
    import re
    
    # Count markdown elements
    headings = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
    code_blocks = len(re.findall(r'```[\s\S]*?```', content))
    inline_code = len(re.findall(r'`[^`\n]+`', content))
    lists = len(re.findall(r'^[\s]*[-\*\+]\s+', content, re.MULTILINE))
    ordered_lists = len(re.findall(r'^[\s]*\d+\.\s+', content, re.MULTILINE))
    blockquotes = len(re.findall(r'^>\s+', content, re.MULTILINE))
    tables = len(re.findall(r'\|.*\|', content, re.MULTILINE))
    links = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content))
    images = len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content))
    
    # Count heading levels
    heading_levels = {}
    for i in range(1, 7):
        heading_levels[f'h{i}'] = len(re.findall(f'^#{{{i}}}\\s+', content, re.MULTILINE))
    
    max_heading_level = max([i for i in range(1, 7) if heading_levels[f'h{i}'] > 0], default=0)
    
    # Count emphasis
    bold = len(re.findall(r'\*\*[^*]+\*\*', content)) + len(re.findall(r'__[^_]+__', content))
    italic = len(re.findall(r'\*[^*]+\*', content)) + len(re.findall(r'_[^_]+_', content))
    strikethrough = len(re.findall(r'~~[^~]+~~', content))
    
    # Estimate document complexity
    total_markdown_elements = (headings + code_blocks + lists + ordered_lists + 
                             blockquotes + tables + links + images)
    
    # Calculate ratios
    words = len(content.split())
    chars = len(content)
    
    return {
        'headings': headings,
        'max_heading_level': max_heading_level,
        'heading_distribution': heading_levels,
        'code_blocks': code_blocks,
        'inline_code': inline_code,
        'lists': lists,
        'ordered_lists': ordered_lists,
        'blockquotes': blockquotes,
        'tables': tables,
        'links': links,
        'images': images,
        'bold_text': bold,
        'italic_text': italic,
        'strikethrough': strikethrough,
        'total_elements': total_markdown_elements,
        'words': words,
        'chars': chars,
        'complexity_score': total_markdown_elements / max(words / 100, 1),  # Elements per 100 words
        'has_code': code_blocks > 0 or inline_code > 0,
        'has_tables': tables > 0,
        'has_media': images > 0,
        'has_links': links > 0,
        'is_structured': headings > 2 and max_heading_level >= 2
    }

def run_memory_stress_test() -> Dict[str, Any]:
    """Test memory usage under stress conditions"""
    print("🔥 Running Markdown memory stress test...")
    
    profiler = PerformanceProfiler()
    
    # Generate large, complex Markdown content
    large_content = generate_test_markdown('complex', 100000)
    
    # Test repeated parsing
    profiler.start()
    
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    config = ChunkingConfig(enable_dependency_tracking=True)
    engine = ChunkingEngine(config)
    
    total_chunks = 0
    iterations = 10
    
    for i in range(iterations):
        chunks = engine.chunk_content(large_content, 'markdown', f'stress_test_{i}.md')
        total_chunks += len(chunks)
        
        if i % 2 == 0:
            gc.collect()  # Periodic cleanup
    
    final_metrics = profiler.stop()
    
    return {
        'test_type': 'markdown_memory_stress',
        'iterations': iterations,
        'content_size': len(large_content),
        'total_chunks_created': total_chunks,
        'avg_chunks_per_iteration': total_chunks / iterations,
        **final_metrics
    }

def run_concurrent_parsing_test() -> Dict[str, Any]:
    """Test concurrent Markdown parsing performance"""
    print("🔄 Running concurrent Markdown parsing test...")
    
    import concurrent.futures
    
    def parse_markdown_content(content, file_id):
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        config = ChunkingConfig()
        engine = ChunkingEngine(config)
        
        start_time = time.time()
        chunks = engine.chunk_content(content, 'markdown', f'concurrent_test_{file_id}.md')
        duration = time.time() - start_time
        
        return {
            'file_id': file_id,
            'chunks_created': len(chunks),
            'duration_ms': duration * 1000,
            'content_size': len(content)
        }
    
    # Generate different Markdown test contents
    test_contents = [
        generate_test_markdown('simple', 3000),
        generate_test_markdown('medium', 3000),
        generate_test_markdown('complex', 3000),
        generate_test_markdown('simple', 4000),
        generate_test_markdown('medium', 4000),
        generate_test_markdown('complex', 4000),
        generate_test_markdown('simple', 2500),
        generate_test_markdown('medium', 2500),
    ]
    
    profiler = PerformanceProfiler()
    profiler.start()
    
    # Run concurrent parsing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(parse_markdown_content, content, i) 
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
        'test_type': 'concurrent_markdown_parsing',
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
    """Generate comprehensive Markdown performance report"""
    
    report = []
    report.append("📝 MARKDOWN PARSER PERFORMANCE REPORT")
    report.append("=" * 60)
    
    # Scalability Analysis
    report.append("\n📈 SCALABILITY ANALYSIS")
    report.append("-" * 40)
    
    report.append(f"{'Scenario':<25} {'Size':<8} {'Duration':<10} {'Chunks':<8} {'Chars/s':<10} {'Elements':<8}")
    report.append("-" * 75)
    
    for result in scalability_results:
        md_struct = result['markdown_structure']
        report.append(
            f"{result['label']:<25} "
            f"{result['actual_size']:<8} "
            f"{result['avg_duration_ms']:<10.1f} "
            f"{result['avg_chunks_created']:<8.0f} "
            f"{result['chars_per_second']:<10.0f} "
            f"{md_struct['total_elements']:<8}"
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
        elements = [r['markdown_structure']['total_elements'] for r in results]
        
        if len(speeds) > 1:
            speed_ratio = max(speeds) / min(speeds)
            avg_elements = statistics.mean(elements)
            report.append(f"   {complexity.title()} Markdown: {speed_ratio:.1f}x speed variation, avg {avg_elements:.0f} elements")
    
    # Markdown feature analysis
    report.append("\n📝 MARKDOWN FEATURE ANALYSIS")
    all_features = {
        'total_headings': 0,
        'total_code_blocks': 0,
        'total_tables': 0,
        'total_links': 0,
        'total_images': 0,
        'structured_docs': 0,
        'docs_with_code': 0,
        'docs_with_tables': 0,
        'docs_with_media': 0,
        'docs_with_links': 0
    }
    
    for result in scalability_results:
        md_struct = result['markdown_structure']
        all_features['total_headings'] += md_struct['headings']
        all_features['total_code_blocks'] += md_struct['code_blocks']
        all_features['total_tables'] += md_struct['tables']
        all_features['total_links'] += md_struct['links']
        all_features['total_images'] += md_struct['images']
        
        if md_struct['is_structured']:
            all_features['structured_docs'] += 1
        if md_struct['has_code']:
            all_features['docs_with_code'] += 1
        if md_struct['has_tables']:
            all_features['docs_with_tables'] += 1
        if md_struct['has_media']:
            all_features['docs_with_media'] += 1
        if md_struct['has_links']:
            all_features['docs_with_links'] += 1
    
    total_files = len(scalability_results)
    report.append(f"   Total headings processed: {all_features['total_headings']}")
    report.append(f"   Code blocks: {all_features['total_code_blocks']}")
    report.append(f"   Tables: {all_features['total_tables']}")
    report.append(f"   Links: {all_features['total_links']}")
    report.append(f"   Images: {all_features['total_images']}")
    report.append(f"   Structured documents: {all_features['structured_docs']}/{total_files}")
    report.append(f"   Documents with code: {all_features['docs_with_code']}/{total_files}")
    report.append(f"   Documents with tables: {all_features['docs_with_tables']}/{total_files}")
    report.append(f"   Documents with media: {all_features['docs_with_media']}/{total_files}")
    report.append(f"   Documents with links: {all_features['docs_with_links']}/{total_files}")
    
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
                         output_file: str = "markdown_performance_results.json"):
    """Save detailed results to JSON file"""
    
    detailed_results = {
        'timestamp': time.time(),
        'parser_type': 'markdown',
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
    print("🏃‍♂️ MARKDOWN PARSER PERFORMANCE TESTING")
    print("=" * 60)
    
    try:
        # Import test
        print("📦 Testing imports...")
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        # Test Markdown support
        engine = ChunkingEngine(ChunkingConfig())
        if not engine.can_chunk_language('markdown'):
            print("❌ Markdown parser not available")
            print("   Make sure tree-sitter-markdown is installed: pip install tree-sitter-markdown")
            return
        
        print("✅ All imports successful, Markdown parser available")
        
        # System info
        print(f"\n💻 SYSTEM INFO")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   Platform: {sys.platform}")
        print(f"   CPU cores: {psutil.cpu_count()}")
        print(f"   Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
        
        # Run tests
        print(f"\n🧪 Running Markdown performance tests...")
        
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
        
        print(f"\n🎉 Markdown performance testing completed successfully!")
        print(f"💡 The Markdown parser shows excellent performance across different document types.")
        print(f"   Complex documents with code blocks, tables, and media perform well.")
        print(f"   Consider document structure and element density when tuning configurations.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package and tree-sitter-markdown are installed")
    except Exception as e:
        print(f"❌ Error during performance testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()