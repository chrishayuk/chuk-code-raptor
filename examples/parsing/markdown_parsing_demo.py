#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Markdown Parsing Demo
====================

Demonstrates the Markdown parser capabilities using tree-sitter analysis.
Shows semantic analysis, element detection, and content structure understanding.
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def load_sample_markdown():
    """Load or create a sample Markdown file"""
    # First try to find an existing sample
    sample_paths = [
        Path(__file__).parent.parent.parent / "examples" / "samples" / "sample.md",
        Path(__file__).parent.parent.parent / "examples" / "samples" / "README.md",
        Path(__file__).parent / "sample_markdown.md"
    ]
    
    for sample_file in sample_paths:
        if sample_file.exists():
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"✅ Loaded sample file: {sample_file} ({len(content)} characters)")
                return content
            except Exception as e:
                print(f"⚠️  Error reading {sample_file}: {e}")
                continue
    
    # If no sample found, create a comprehensive sample
    print("📝 Creating sample Markdown content...")
    return create_sample_markdown()

def create_sample_markdown() -> str:
    """Create a comprehensive sample Markdown document"""
    return """# Complete Markdown Feature Demo

This document demonstrates various Markdown features and parsing capabilities of the semantic chunking system.

## Table of Contents

1. [Basic Formatting](#basic-formatting)
2. [Code Examples](#code-examples)
3. [Lists and Tables](#lists-and-tables)
4. [Links and Media](#links-and-media)
5. [Advanced Features](#advanced-features)

---

## Basic Formatting

### Text Emphasis

Markdown supports various text formatting options:

- **Bold text** using `**bold**` or `__bold__`
- *Italic text* using `*italic*` or `_italic_`
- ***Bold and italic*** using `***text***`
- ~~Strikethrough~~ using `~~strikethrough~~`
- `Inline code` using backticks

### Headings

This document uses multiple heading levels to demonstrate hierarchical structure:

#### Level 4 Heading
##### Level 5 Heading
###### Level 6 Heading

> **Note**: Proper heading hierarchy is important for document structure and accessibility.

## Code Examples

### Inline Code

Use `backticks` to create inline code like `console.log()` or `import sys`.

### Code Blocks

#### Python Example

```python
def fibonacci(n):
    '''Generate Fibonacci sequence up to n terms.'''
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    
    return sequence

# Usage example
fib_numbers = fibonacci(10)
print(f"First 10 Fibonacci numbers: {fib_numbers}")
```

#### JavaScript Example

```javascript
class DataProcessor {
    constructor(options = {}) {
        this.options = {
            batchSize: 100,
            timeout: 5000,
            ...options
        };
        this.queue = [];
    }
    
    async process(data) {
        try {
            const result = await this.performOperation(data);
            return this.formatResult(result);
        } catch (error) {
            console.error('Processing failed:', error);
            throw error;
        }
    }
    
    performOperation(data) {
        return new Promise((resolve) => {
            setTimeout(() => resolve(data.toUpperCase()), 100);
        });
    }
}

// Usage
const processor = new DataProcessor({ batchSize: 50 });
processor.process("hello world").then(console.log);
```

#### Shell Commands

```bash
# Setup development environment
git clone https://github.com/example/project.git
cd project

# Install dependencies
npm install
pip install -r requirements.txt

# Run tests
npm test
python -m pytest tests/

# Build and deploy
npm run build
docker build -t myapp .
docker run -p 3000:3000 myapp
```

## Lists and Tables

### Unordered Lists

- **Project Management**
  - Planning and roadmaps
  - Task tracking
  - Team coordination
  - Resource allocation

- **Development Tools**
  - Version control (Git)
  - Code editors (VS Code, Vim)
  - Testing frameworks
  - CI/CD pipelines

- **Documentation**
  - API documentation
  - User guides
  - Technical specifications
  - Code comments

### Ordered Lists

1. **Planning Phase**
   1. Requirements gathering
   2. Architecture design
   3. Technology selection
   4. Timeline estimation

2. **Development Phase**
   1. Environment setup
   2. Core implementation
   3. Feature development
   4. Testing and debugging

3. **Deployment Phase**
   1. Production setup
   2. Performance optimization
   3. Security hardening
   4. Monitoring configuration

### Task Lists

- [x] Create project structure
- [x] Set up development environment
- [x] Implement core features
- [ ] Add comprehensive tests
- [ ] Write documentation
- [ ] Performance optimization
- [ ] Security audit

### Tables

#### Feature Comparison

| Feature | Basic Plan | Pro Plan | Enterprise |
|---------|------------|----------|------------|
| Users | 5 | 25 | Unlimited |
| Storage | 10GB | 100GB | 1TB |
| API Calls | 1,000/month | 10,000/month | Unlimited |
| Support | Email | Email + Chat | 24/7 Phone |
| Price | $10/month | $50/month | Contact Us |

#### Performance Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Response Time | <100ms | 85ms | ✅ Good |
| Throughput | >1000 req/s | 1200 req/s | ✅ Excellent |
| Uptime | 99.9% | 99.95% | ✅ Excellent |
| Error Rate | <0.1% | 0.05% | ✅ Good |

## Links and Media

### External Links

Visit our [documentation](https://docs.example.com) for detailed guides.
Check out the [GitHub repository](https://github.com/example/project) for source code.
Join our [community forum](https://forum.example.com) for discussions.

### Reference Links

This is a reference to [Google][google-link] and [GitHub][github-link].

[google-link]: https://google.com "Google Search"
[github-link]: https://github.com "GitHub Platform"

### Images

![Architecture Diagram](https://example.com/images/architecture.png "System Architecture")

![API Flow](https://example.com/images/api-flow.svg "API Request Flow")

### Email and Automatic Links

Contact us at support@example.com or visit https://example.com/contact.

## Advanced Features

### Blockquotes

> **Einstein's Wisdom**: "Try not to become a person of success, but rather try to become a person of value."

> This is a longer blockquote that demonstrates how the parser handles multi-line quoted content.
> 
> It can contain multiple paragraphs and even **formatting** within the quote.
> 
> > Nested quotes are also supported for when you need to quote within a quote.

### Mathematical Expressions

The quadratic formula is: `x = (-b ± √(b² - 4ac)) / 2a`

For more complex math:
```
∑(i=1 to n) i = n(n+1)/2
```

### HTML Integration

<div align="center">
  <strong>HTML can be embedded in Markdown</strong><br>
  <em>when needed for specific formatting</em>
</div>

### Horizontal Rules

Above this line is content.

---

Below this line is more content.

***

And this is yet another section.

## Conclusion

This document demonstrates the comprehensive parsing capabilities of the Markdown chunking system. The parser can handle:

- ✅ **Structural elements**: headings, lists, tables
- ✅ **Code content**: inline code, fenced blocks, syntax highlighting
- ✅ **Rich formatting**: emphasis, links, images, blockquotes
- ✅ **Advanced features**: footnotes, abbreviations, HTML integration
- ✅ **Technical content**: API docs, configuration examples, error handling

The semantic analysis provides intelligent chunking that respects document structure while maintaining context and relationships between elements.

---

*Document created for Markdown parsing demonstration*
*Last updated: 2024*
"""

def analyze_chunk_structure(chunk) -> Dict[str, Any]:
    """Analyze the structure and properties of a Markdown chunk"""
    analysis = {
        'id': chunk.id,
        'type': chunk.chunk_type.value,
        'size_chars': len(chunk.content),
        'size_lines': chunk.line_count,
        'importance': chunk.importance_score,
        'tags': [tag.name for tag in chunk.semantic_tags] if hasattr(chunk, 'semantic_tags') and chunk.semantic_tags else [],
        'metadata': chunk.metadata or {},
        'dependencies': chunk.dependencies[:5] if chunk.dependencies else [],  # First 5
        'preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
    }
    
    # Extract Markdown-specific information
    section_type = chunk.metadata.get('section_type') if chunk.metadata else None
    heading_level = chunk.metadata.get('heading_level') if chunk.metadata else None
    code_language = chunk.metadata.get('code_language') if chunk.metadata else None
    
    analysis['markdown_analysis'] = {
        'section_type': section_type,
        'heading_level': heading_level,
        'code_language': code_language,
        'semantic_type': detect_markdown_semantic_type(analysis['tags'], chunk.content),
        'content_features': detect_content_features(chunk.content),
        'complexity_indicators': detect_complexity_indicators(chunk.content)
    }
    
    return analysis

def detect_markdown_semantic_type(tags: List[str], content: str) -> str:
    """Detect the semantic type of a Markdown chunk"""
    content_lower = content.lower()
    
    if 'heading' in tags:
        return 'Heading'
    elif 'code_block' in tags or 'code' in tags:
        return 'Code Block'
    elif 'table' in tags or 'tabular_data' in tags:
        return 'Table'
    elif 'list' in tags:
        return 'List Content'
    elif 'blockquote' in tags or 'quote' in tags:
        return 'Blockquote'
    elif 'paragraph' in tags:
        return 'Paragraph'
    elif any(keyword in content_lower for keyword in ['api', 'endpoint', 'method']):
        return 'API Documentation'
    elif any(keyword in content_lower for keyword in ['config', 'settings', 'yaml', 'json']):
        return 'Configuration'
    elif any(keyword in content_lower for keyword in ['install', 'setup', 'usage']):
        return 'Instructions'
    elif any(keyword in content_lower for keyword in ['example', 'demo', 'tutorial']):
        return 'Example'
    elif any(keyword in content_lower for keyword in ['error', 'troubleshoot', 'debug']):
        return 'Troubleshooting'
    else:
        return 'Generic Content'

def detect_content_features(content: str) -> List[str]:
    """Detect specific content features in Markdown"""
    features = []
    
    # Code-related features
    if '```' in content:
        features.append('Fenced code blocks')
    if '`' in content and content.count('`') >= 2:
        features.append('Inline code')
    
    # Links and media
    if re.search(r'\[([^\]]+)\]\(([^)]+)\)', content):
        features.append('Links')
    if re.search(r'!\[([^\]]*)\]\(([^)]+)\)', content):
        features.append('Images')
    
    # Lists
    if re.search(r'^[\s]*[-\*\+]\s+', content, re.MULTILINE):
        features.append('Unordered lists')
    if re.search(r'^[\s]*\d+\.\s+', content, re.MULTILINE):
        features.append('Ordered lists')
    if re.search(r'- \[[ x]\]', content):
        features.append('Task lists')
    
    # Tables
    if '|' in content and re.search(r'\|.*\|', content):
        features.append('Tables')
    
    # Emphasis
    if '**' in content or '__' in content:
        features.append('Bold text')
    if ('*' in content and not content.count('*') == content.count('**') * 2) or '_' in content:
        features.append('Italic text')
    if '~~' in content:
        features.append('Strikethrough')
    
    # Other features
    if '>' in content and re.search(r'^>\s+', content, re.MULTILINE):
        features.append('Blockquotes')
    if re.search(r'^#{1,6}\s+', content, re.MULTILINE):
        features.append('Headings')
    if '---' in content or '***' in content:
        features.append('Horizontal rules')
    if '<' in content and '>' in content:
        features.append('HTML elements')
    
    return features

def detect_complexity_indicators(content: str) -> Dict[str, Any]:
    """Detect complexity indicators in Markdown content"""
    
    # Count various elements
    headings = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
    code_blocks = len(re.findall(r'```[\s\S]*?```', content))
    inline_code = len(re.findall(r'`[^`\n]+`', content))
    tables = len(re.findall(r'\|.*\|', content, re.MULTILINE))
    links = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content))
    images = len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content))
    
    # Calculate complexity score
    complexity_score = (
        headings * 2 +
        code_blocks * 5 +
        inline_code * 1 +
        tables * 3 +
        links * 1 +
        images * 2
    )
    
    # Determine complexity level
    if complexity_score <= 5:
        complexity_level = 'Simple'
    elif complexity_score <= 15:
        complexity_level = 'Medium'
    elif complexity_score <= 30:
        complexity_level = 'Complex'
    else:
        complexity_level = 'Very Complex'
    
    return {
        'score': complexity_score,
        'level': complexity_level,
        'elements': {
            'headings': headings,
            'code_blocks': code_blocks,
            'inline_code': inline_code,
            'tables': tables,
            'links': links,
            'images': images
        }
    }

def generate_markdown_summary(chunks) -> Dict[str, Any]:
    """Generate summary of Markdown parsing results"""
    summary = {
        'total_chunks': len(chunks),
        'chunk_types': defaultdict(int),
        'semantic_types': defaultdict(int),
        'content_features': defaultdict(int),
        'structure_analysis': {
            'total_headings': 0,
            'max_heading_level': 0,
            'code_blocks': 0,
            'tables': 0,
            'lists': 0,
            'blockquotes': 0
        },
        'complexity_distribution': defaultdict(int),
        'language_distribution': defaultdict(int)
    }
    
    for chunk in chunks:
        # Type distribution
        summary['chunk_types'][chunk.chunk_type.value] += 1
        
        # Analyze chunk
        analysis = analyze_chunk_structure(chunk)
        md_analysis = analysis['markdown_analysis']
        
        # Semantic type distribution
        semantic_type = md_analysis['semantic_type']
        summary['semantic_types'][semantic_type] += 1
        
        # Content features
        for feature in md_analysis['content_features']:
            summary['content_features'][feature] += 1
        
        # Structure analysis
        heading_level = md_analysis['heading_level']
        if heading_level:
            summary['structure_analysis']['total_headings'] += 1
            summary['structure_analysis']['max_heading_level'] = max(
                summary['structure_analysis']['max_heading_level'], heading_level
            )
        
        if 'code_block' in analysis['tags']:
            summary['structure_analysis']['code_blocks'] += 1
            
            # Track programming languages
            code_language = md_analysis['code_language']
            if code_language:
                summary['language_distribution'][code_language] += 1
        
        if 'table' in analysis['tags']:
            summary['structure_analysis']['tables'] += 1
        
        if 'list' in analysis['tags']:
            summary['structure_analysis']['lists'] += 1
        
        if 'blockquote' in analysis['tags']:
            summary['structure_analysis']['blockquotes'] += 1
        
        # Complexity distribution
        complexity_level = md_analysis['complexity_indicators']['level']
        summary['complexity_distribution'][complexity_level] += 1
    
    return summary

def extract_document_outline(chunks) -> Dict[str, Any]:
    """Extract the hierarchical outline of the Markdown document"""
    outline = {
        'headings': [],
        'sections': [],
        'code_blocks': [],
        'tables': []
    }
    
    for chunk in chunks:
        analysis = analyze_chunk_structure(chunk)
        md_analysis = analysis['markdown_analysis']
        
        # Extract headings with hierarchy
        if md_analysis['heading_level']:
            heading_text = extract_heading_text(chunk.content)
            outline['headings'].append({
                'level': md_analysis['heading_level'],
                'text': heading_text,
                'chunk_id': chunk.id,
                'line': chunk.start_line
            })
        
        # Extract code blocks with language
        if md_analysis['code_language']:
            outline['code_blocks'].append({
                'language': md_analysis['code_language'],
                'chunk_id': chunk.id,
                'line': chunk.start_line,
                'size': len(chunk.content)
            })
        
        # Extract tables
        if 'table' in analysis['tags']:
            table_info = analyze_table_structure(chunk.content)
            outline['tables'].append({
                'rows': table_info['rows'],
                'columns': table_info['columns'],
                'chunk_id': chunk.id,
                'line': chunk.start_line
            })
        
        # Track major sections
        if md_analysis['semantic_type'] in ['API Documentation', 'Configuration', 'Instructions']:
            outline['sections'].append({
                'type': md_analysis['semantic_type'],
                'chunk_id': chunk.id,
                'line': chunk.start_line,
                'features': md_analysis['content_features']
            })
    
    return outline

def extract_heading_text(content: str) -> str:
    """Extract clean heading text from Markdown content"""
    # Find heading line
    for line in content.split('\n'):
        if line.strip().startswith('#'):
            # Remove hash marks and clean up
            text = re.sub(r'^#+\s*', '', line.strip())
            return text[:80] + "..." if len(text) > 80 else text
    return "Heading"

def analyze_table_structure(content: str) -> Dict[str, int]:
    """Analyze table structure in Markdown content"""
    lines = content.split('\n')
    table_lines = [line for line in lines if '|' in line and line.strip()]
    
    if not table_lines:
        return {'rows': 0, 'columns': 0}
    
    # Count columns from first row
    first_row = table_lines[0]
    columns = len([cell.strip() for cell in first_row.split('|') if cell.strip()])
    
    # Count actual data rows (exclude separator rows)
    data_rows = 0
    for line in table_lines:
        if not re.match(r'^[\s\|:-]+$', line.strip()):
            data_rows += 1
    
    return {'rows': data_rows, 'columns': columns}

def analyze_markdown_manually(content: str) -> Dict[str, Any]:
    """Manual analysis of Markdown structure when tree-sitter parsing fails"""
    lines = content.split('\n')
    
    # Find headings
    headings = []
    for i, line in enumerate(lines):
        if re.match(r'^#{1,6}\s+', line):
            level = len(line) - len(line.lstrip('#'))
            text = line.lstrip('#').strip()
            headings.append({
                'level': level,
                'text': text,
                'line': i + 1
            })
    
    # Find code blocks
    code_blocks = []
    in_code_block = False
    code_start = 0
    code_lang = None
    
    for i, line in enumerate(lines):
        if line.strip().startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_start = i + 1
                # Extract language
                lang_match = re.match(r'^```(\w+)', line.strip())
                code_lang = lang_match.group(1) if lang_match else 'text'
            else:
                in_code_block = False
                code_blocks.append({
                    'language': code_lang,
                    'start_line': code_start,
                    'end_line': i + 1,
                    'size': i - code_start + 1
                })
    
    # Find sections (content between headings)
    sections = []
    current_section_start = 1
    
    for heading in headings:
        if heading['line'] > current_section_start:
            sections.append({
                'start_line': current_section_start,
                'end_line': heading['line'] - 1,
                'heading_before': headings[headings.index(heading) - 1] if headings.index(heading) > 0 else None
            })
        current_section_start = heading['line']
    
    # Add final section
    if current_section_start < len(lines):
        sections.append({
            'start_line': current_section_start,
            'end_line': len(lines),
            'heading_before': headings[-1] if headings else None
        })
    
    return {
        'headings': len(headings),
        'heading_levels': [h['level'] for h in headings],
        'code_blocks': len(code_blocks),
        'languages': list(set(cb['language'] for cb in code_blocks)),
        'sections': len(sections),
        'structure': {
            'headings': headings,
            'code_blocks': code_blocks,
            'sections': sections
        }
    }

def print_manual_analysis(manual_analysis: Dict[str, Any]):
    """Print manual analysis results when tree-sitter parsing fails"""
    print(f"\n🔍 MANUAL STRUCTURE ANALYSIS")
    print("-" * 60)
    
    structure = manual_analysis['structure']
    
    print(f"Document contains:")
    print(f"   • {manual_analysis['headings']} headings")
    print(f"   • {manual_analysis['code_blocks']} code blocks")
    print(f"   • {manual_analysis['sections']} content sections")
    
    if manual_analysis['languages']:
        print(f"   • Programming languages: {', '.join(manual_analysis['languages'])}")
    
    # Show heading structure
    if structure['headings']:
        print(f"\n📖 HEADING STRUCTURE")
        print("-" * 40)
        for heading in structure['headings'][:10]:  # First 10
            indent = "  " * (heading['level'] - 1)
            print(f"   {indent}H{heading['level']}: {heading['text'][:60]}{'...' if len(heading['text']) > 60 else ''}")
        
        if len(structure['headings']) > 10:
            print(f"   ... and {len(structure['headings']) - 10} more headings")
    
    # Show code blocks
    if structure['code_blocks']:
        print(f"\n💻 CODE BLOCKS")
        print("-" * 40)
        lang_counts = {}
        for cb in structure['code_blocks']:
            lang_counts[cb['language']] = lang_counts.get(cb['language'], 0) + 1
        
        for lang, count in lang_counts.items():
            print(f"   {lang}: {count} blocks")
    
    print(f"\n💡 ANALYSIS NOTES:")
    print(f"   This manual analysis suggests the document has clear structure that")
    print(f"   should be parseable by tree-sitter. The single-chunk result may indicate:")
    print(f"   • Tree-sitter-markdown parser configuration issues")
    print(f"   • Document complexity exceeding parser capabilities") 
    print(f"   • Parser version incompatibility")
    print(f"   • Need for different chunking strategy")

def print_detailed_markdown_analysis(content: str, chunks: List, summary: Dict[str, Any], outline: Dict[str, Any], manual_analysis: Dict[str, Any] = None):
    """Print comprehensive analysis of Markdown parsing results"""
    print("\n" + "="*80)
    print("📝 COMPREHENSIVE MARKDOWN PARSING ANALYSIS")
    print("="*80)
    
    # Document info
    content_size = len(content)
    content_lines = content.count('\n') + 1
    words = len(content.split())
    
    print(f"\n📄 MARKDOWN DOCUMENT ANALYSIS")
    print("-" * 60)
    print(f"Document size: {content_size:,} characters, {content_lines} lines, {words:,} words")
    print(f"Chunks created: {len(chunks)}")
    print(f"Average chunk size: {content_size // len(chunks) if chunks else 0} characters")
    
    # Structure summary
    print(f"\n🏗️  DOCUMENT STRUCTURE")
    print("-" * 60)
    structure = summary['structure_analysis']
    print(f"Headings found: {structure['total_headings']} (max level: h{structure['max_heading_level']})")
    print(f"Code blocks: {structure['code_blocks']}")
    print(f"Tables: {structure['tables']}")
    print(f"Lists: {structure['lists']}")
    print(f"Blockquotes: {structure['blockquotes']}")
    
    # Content features
    if summary['content_features']:
        print(f"\n🎯 CONTENT FEATURES")
        print("-" * 60)
        sorted_features = sorted(summary['content_features'].items(), key=lambda x: x[1], reverse=True)
        for feature, count in sorted_features:
            print(f"   {feature}: {count} occurrences")
    
    # Semantic types
    if summary['semantic_types']:
        print(f"\n📋 SEMANTIC TYPES")
        print("-" * 60)
        sorted_types = sorted(summary['semantic_types'].items(), key=lambda x: x[1], reverse=True)
        for semantic_type, count in sorted_types:
            print(f"   {semantic_type}: {count} chunks")
    
    # Programming languages
    if summary['language_distribution']:
        print(f"\n💻 PROGRAMMING LANGUAGES")
        print("-" * 60)
        sorted_langs = sorted(summary['language_distribution'].items(), key=lambda x: x[1], reverse=True)
        for language, count in sorted_langs:
            print(f"   {language}: {count} code blocks")
    
    # Document outline
    if outline['headings']:
        print(f"\n📖 DOCUMENT OUTLINE")
        print("-" * 60)
        for heading in outline['headings']:
            level = heading['level']
            indent = "  " * (level - 1) if level else ""
            print(f"   {indent}H{level}: {heading['text']}")
    
    # Code blocks summary
    if outline['code_blocks']:
        print(f"\n💾 CODE BLOCKS")
        print("-" * 60)
        lang_summary = defaultdict(list)
        for code_block in outline['code_blocks']:
            lang_summary[code_block['language']].append(code_block)
        
        for language, blocks in lang_summary.items():
            total_size = sum(block['size'] for block in blocks)
            print(f"   {language}: {len(blocks)} blocks, {total_size:,} characters total")
    
    # Tables summary
    if outline['tables']:
        print(f"\n📊 TABLES")
        print("-" * 60)
        for i, table in enumerate(outline['tables'], 1):
            print(f"   Table {i}: {table['rows']} rows × {table['columns']} columns")
    
    # Complexity analysis
    if summary['complexity_distribution']:
        print(f"\n📊 COMPLEXITY DISTRIBUTION")
        print("-" * 60)
        for complexity, count in summary['complexity_distribution'].items():
            percentage = (count / len(chunks)) * 100
            print(f"   {complexity}: {count} chunks ({percentage:.1f}%)")
    
    # If we only got one chunk, show manual analysis
    if len(chunks) == 1 and manual_analysis:
        print_manual_analysis(manual_analysis)
    
    # Detailed chunk analysis
    print(f"\n📋 DETAILED CHUNK ANALYSIS")
    print("-" * 80)
    
    # Group chunks by semantic type for better organization
    chunks_by_type = defaultdict(list)
    for chunk in chunks:
        analysis = analyze_chunk_structure(chunk)
        semantic_type = analysis['markdown_analysis']['semantic_type']
        chunks_by_type[semantic_type].append((chunk, analysis))
    
    for semantic_type, chunk_list in chunks_by_type.items():
        print(f"\n🔸 {semantic_type.upper()} CHUNKS ({len(chunk_list)})")
        
        for chunk, analysis in chunk_list[:3]:  # Show first 3 of each type
            md_analysis = analysis['markdown_analysis']
            
            print(f"\n   📌 {analysis['id']}")
            print(f"      Type: {semantic_type}")
            print(f"      Size: {analysis['size_chars']} chars, {analysis['size_lines']} lines")
            print(f"      Importance: {analysis['importance']:.2f}")
            
            if md_analysis['heading_level']:
                print(f"      Heading Level: H{md_analysis['heading_level']}")
            
            if md_analysis['code_language']:
                print(f"      Language: {md_analysis['code_language']}")
            
            if analysis['tags']:
                print(f"      Semantic tags: {', '.join(analysis['tags'])}")
            
            if md_analysis['content_features']:
                print(f"      Features: {', '.join(md_analysis['content_features'])}")
            
            if md_analysis['complexity_indicators']:
                complexity = md_analysis['complexity_indicators']
                print(f"      Complexity: {complexity['level']} (score: {complexity['score']})")
            
            if analysis['dependencies']:
                print(f"      Dependencies: {', '.join(analysis['dependencies'])}")
            
            print(f"      Preview: {analysis['preview']}")
        
        if len(chunk_list) > 3:
            print(f"   ... and {len(chunk_list) - 3} more {semantic_type.lower()} chunks")

def test_markdown_parsing(content: str) -> Dict[str, Any]:
    """Test Markdown parsing on the sample content with enhanced parser"""
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    # Check if Markdown is supported
    config = ChunkingConfig()
    engine = ChunkingEngine(config)
    
    if not engine.can_chunk_language('markdown'):
        print(f"⚠️  Markdown parser not available. Supported languages: {engine.get_supported_languages()}")
        return {
            'chunks': [],
            'summary': generate_markdown_summary([]),
            'outline': extract_document_outline([]),
            'sample_info': {
                'size_chars': len(content),
                'line_count': content.count('\n') + 1,
                'word_count': len(content.split()),
                'error': 'Markdown parser not available'
            }
        }
    
    # Try the enhanced parsing approach
    parsing_results = []
    
    # Strategy 1: Enhanced parser with semantic fallback
    print("   🔄 Trying enhanced parser with semantic fallback...")
    try:
        config1 = ChunkingConfig(
            target_chunk_size=500,
            min_chunk_size=30,
            preserve_atomic_nodes=True,
            enable_dependency_tracking=True
        )
        engine1 = ChunkingEngine(config1)
        chunks1 = engine1.chunk_content(content, 'markdown', 'sample.md')
        parsing_results.append(('enhanced_semantic', chunks1, config1))
        print(f"      ✅ Enhanced semantic: {len(chunks1)} chunks")
        
        # Check if parser used manual extraction
        if chunks1:
            manual_chunks = sum(1 for chunk in chunks1 if chunk.metadata.get('manual_extraction'))
            if manual_chunks > 0:
                print(f"      📋 Manual semantic extraction used for {manual_chunks}/{len(chunks1)} chunks")
        
    except Exception as e:
        print(f"      ❌ Enhanced semantic failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Strategy 2: Smaller chunks for better segmentation
    print("   🔄 Trying smaller chunks...")
    try:
        config2 = ChunkingConfig(
            target_chunk_size=300,
            min_chunk_size=20,
            preserve_atomic_nodes=False,
            enable_dependency_tracking=True
        )
        engine2 = ChunkingEngine(config2)
        chunks2 = engine2.chunk_content(content, 'markdown', 'sample.md')
        parsing_results.append(('small_semantic', chunks2, config2))
        print(f"      ✅ Small semantic: {len(chunks2)} chunks")
    except Exception as e:
        print(f"      ❌ Small semantic failed: {e}")
    
    # Strategy 3: Very fine-grained chunking
    print("   🔄 Trying fine-grained chunking...")
    try:
        config3 = ChunkingConfig(
            target_chunk_size=200,
            min_chunk_size=15,
            preserve_atomic_nodes=False,
            enable_dependency_tracking=False
        )
        engine3 = ChunkingEngine(config3)
        chunks3 = engine3.chunk_content(content, 'markdown', 'sample.md')
        parsing_results.append(('fine_grained', chunks3, config3))
        print(f"      ✅ Fine-grained: {len(chunks3)} chunks")
    except Exception as e:
        print(f"      ❌ Fine-grained failed: {e}")
    
    # Choose the best result (prefer more chunks with reasonable distribution)
    best_result = None
    
    # Prefer results with good chunk distribution
    for strategy, chunks, config in parsing_results:
        if len(chunks) > 5:  # Good number of chunks
            best_result = (strategy, chunks, config)
            break
        elif len(chunks) > 1 and not best_result:  # Backup choice
            best_result = (strategy, chunks, config)
    
    # Fallback to any available result
    if not best_result and parsing_results:
        best_result = parsing_results[0]
        print(f"   ⚠️  Using fallback result with {len(best_result[1])} chunks")
        
        # Still do manual analysis for comparison
        manual_analysis = analyze_markdown_manually(content)
        print(f"       📊 Manual analysis found:")
        print(f"          - {manual_analysis['headings']} headings")
        print(f"          - {manual_analysis['code_blocks']} code blocks") 
        print(f"          - {manual_analysis['sections']} potential sections")
    
    if best_result:
        strategy, chunks, config = best_result
        print(f"   ✅ Using '{strategy}' strategy with {len(chunks)} chunks")
        
        # Analyze parsing method distribution
        if chunks:
            tree_sitter_chunks = sum(1 for chunk in chunks if not chunk.metadata.get('manual_extraction'))
            manual_chunks = sum(1 for chunk in chunks if chunk.metadata.get('manual_extraction'))
            print(f"       📈 Parsing breakdown: {tree_sitter_chunks} tree-sitter, {manual_chunks} manual extraction")
        
        # Generate analysis
        summary = generate_markdown_summary(chunks)
        outline = extract_document_outline(chunks)
        
        return {
            'chunks': chunks,
            'summary': summary,
            'outline': outline,
            'strategy_used': strategy,
            'parsing_method_breakdown': {
                'tree_sitter_chunks': tree_sitter_chunks if chunks else 0,
                'manual_chunks': manual_chunks if chunks else 0,
                'total_chunks': len(chunks)
            },
            'sample_info': {
                'size_chars': len(content),
                'line_count': content.count('\n') + 1,
                'word_count': len(content.split())
            }
        }
    else:
        return {
            'chunks': [],
            'summary': generate_markdown_summary([]),
            'outline': extract_document_outline([]),
            'sample_info': {
                'size_chars': len(content),
                'line_count': content.count('\n') + 1,
                'word_count': len(content.split()),
                'error': 'All parsing strategies failed'
            }
        }

def main():
    """Main demo function"""
    print("📝 MARKDOWN PARSING DEMO")
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
        
        if 'markdown' not in supported_languages:
            print("⚠️  Markdown parser not available. This may be due to missing tree-sitter-markdown.")
            print("   Install with: pip install tree-sitter-markdown")
            print("   Demo will show what the analysis would look like.")
        
        # Load sample Markdown
        print(f"\n📄 Loading sample Markdown content...")
        markdown_content = load_sample_markdown()
        
        if not markdown_content:
            print("❌ Cannot proceed without Markdown content")
            return
        
        line_count = markdown_content.count('\n') + 1
        word_count = len(markdown_content.split())
        print(f"📄 Sample Markdown loaded: {len(markdown_content)} characters, {line_count} lines, {word_count:,} words")
        
        # Test Markdown parsing
        print(f"\n🚀 Testing Markdown parsing...")
        try:
            result = test_markdown_parsing(markdown_content)
            
            if 'error' in result['sample_info']:
                print(f"⚠️  {result['sample_info']['error']}")
            else:
                chunks_count = len(result['chunks'])
                sample_size = result['sample_info']['size_chars']
                strategy = result.get('strategy_used', 'unknown')
                print(f"✅ Successfully parsed Markdown: {chunks_count} chunks from {sample_size} characters using '{strategy}' strategy")
                
                # If we got only one chunk, do manual analysis
                manual_analysis = None
                if chunks_count == 1:
                    print(f"   🔍 Single chunk detected - performing manual structure analysis...")
                    manual_analysis = analyze_markdown_manually(markdown_content)
                
                # Print comprehensive analysis
                print_detailed_markdown_analysis(
                    markdown_content, 
                    result['chunks'], 
                    result['summary'], 
                    result['outline'],
                    manual_analysis
                )
                
                # Overall statistics
                print(f"\n📈 OVERALL STATISTICS")
                print("-" * 60)
                
                summary = result['summary']
                print(f"Total chunks created: {summary['total_chunks']}")
                print(f"Semantic types found: {len(summary['semantic_types'])}")
                print(f"Content features detected: {len(summary['content_features'])}")
                print(f"Programming languages: {len(summary['language_distribution'])}")
                
                # Show parsing method breakdown
                if 'parsing_method_breakdown' in result:
                    breakdown = result['parsing_method_breakdown']
                    print(f"Parsing method breakdown:")
                    print(f"   • Tree-sitter chunks: {breakdown['tree_sitter_chunks']}")
                    print(f"   • Manual extraction chunks: {breakdown['manual_chunks']}")
                    print(f"   • Total chunks: {breakdown['total_chunks']}")
                
                if summary['total_chunks'] > 0:
                    avg_chunk_size = len(markdown_content) / summary['total_chunks']
                    print(f"Average chunk size: {avg_chunk_size:.0f} characters")
                
                # Top semantic types
                if summary['semantic_types']:
                    print(f"\nMost common content types:")
                    sorted_types = sorted(summary['semantic_types'].items(), key=lambda x: x[1], reverse=True)
                    for semantic_type, count in sorted_types[:5]:
                        print(f"   {semantic_type}: {count} chunks")
                
                print(f"\n🎉 Enhanced Markdown parsing demo completed successfully!")
                
                # Enhanced success message based on results
                if 'parsing_method_breakdown' in result:
                    breakdown = result['parsing_method_breakdown']
                    if breakdown['manual_chunks'] > 0:
                        print(f"💡 The enhanced Markdown parser successfully used semantic fallback!")
                        print(f"   Manual extraction handled {breakdown['manual_chunks']} chunks where tree-sitter")
                        print(f"   failed to properly segment the document structure.")
                    else:
                        print(f"💡 Tree-sitter parsing worked well for this document!")
                        print(f"   All {breakdown['tree_sitter_chunks']} chunks were created via tree-sitter analysis.")
                else:
                    print(f"💡 The Markdown parser successfully analyzed the document with intelligent")
                    print(f"   semantic understanding, proper structure detection, and comprehensive")
                    print(f"   content feature recognition.")
                
                # Usage recommendations
                print(f"\n🎯 Enhanced Markdown parsing insights:")
                structure = summary['structure_analysis']
                print(f"   • Document structure: {structure['total_headings']} headings, {structure['code_blocks']} code blocks")
                print(f"   • Content richness: {structure['tables']} tables, {structure['lists']} lists")
                print(f"   • Technical content: {len(summary['language_distribution'])} programming languages")
                print(f"   • Complexity: {len(summary['complexity_distribution'])} different complexity levels")
                
                # Performance insights
                if summary['total_chunks'] > 1:
                    print(f"   • Parsing success: Enhanced parser created {summary['total_chunks']} well-structured chunks")
                    if 'parsing_method_breakdown' in result and result['parsing_method_breakdown']['manual_chunks'] > 0:
                        print(f"   • Hybrid approach: Successfully combined tree-sitter + manual extraction")
                else:
                    print(f"   • Parsing limitation: Only {summary['total_chunks']} chunk created - document may need alternative approach")
                
        except Exception as e:
            print(f"❌ Error during Markdown parsing: {e}")
            import traceback
            traceback.print_exc()
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package and tree-sitter-markdown are installed")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()