#!/usr/bin/env python3
"""
HTML Parsing Demo
================

Demonstrates the HTML parser capabilities using the existing sample.html file.
Shows semantic analysis, element detection, and content structure understanding.
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def load_sample_html():
    """Load the sample HTML file"""
    sample_file = Path(__file__).parent.parent.parent / "examples" / "samples" / "sample.html"
    
    if not sample_file.exists():
        print(f"❌ Sample file not found: {sample_file}")
        print("   Please ensure examples/samples/sample.html exists")
        return None
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ Loaded sample file: {sample_file} ({len(content)} characters)")
        return content
    except Exception as e:
        print(f"❌ Error reading sample file: {e}")
        return None

def analyze_chunk_structure(chunk) -> Dict[str, Any]:
    """Analyze the structure and properties of an HTML chunk"""
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
    
    # Extract HTML-specific information
    tag_name = chunk.metadata.get('tag_name') if chunk.metadata else None
    element_id = chunk.metadata.get('element_id') if chunk.metadata else None
    css_classes = chunk.metadata.get('css_classes', []) if chunk.metadata else []
    attributes = chunk.metadata.get('attributes', {}) if chunk.metadata else {}
    
    analysis['html_analysis'] = {
        'tag_name': tag_name,
        'element_id': element_id,
        'css_classes': css_classes,
        'attributes': attributes,
        'has_semantic_tags': any(tag in analysis['tags'] for tag in [
            'content_section', 'page_structure', 'navigation', 'heading', 
            'list', 'tabular_data', 'interactive', 'code'
        ]),
        'semantic_type': detect_semantic_type(analysis['tags'], tag_name),
        'accessibility_features': detect_accessibility_features(chunk.content, attributes)
    }
    
    return analysis

def detect_semantic_type(tags: List[str], tag_name: str) -> str:
    """Detect the semantic type of an HTML element"""
    if 'navigation' in tags:
        return 'Navigation'
    elif any(tag.startswith('heading') for tag in tags):
        return 'Heading'
    elif 'content_section' in tags:
        return 'Content Section'
    elif 'page_structure' in tags:
        return 'Page Structure'
    elif 'interactive' in tags:
        return 'Interactive Element'
    elif 'tabular_data' in tags:
        return 'Data Table'
    elif 'list' in tags:
        return 'List Content'
    elif 'code' in tags:
        return 'Code Block'
    elif tag_name in ['div', 'span']:
        return 'Container'
    elif tag_name in ['p', 'text']:
        return 'Text Content'
    elif tag_name in ['img', 'video', 'audio']:
        return 'Media'
    else:
        return 'Generic Element'

def detect_accessibility_features(content: str, attributes: Dict[str, str]) -> List[str]:
    """Detect accessibility features in HTML content"""
    features = []
    
    # ARIA attributes
    if any(attr.startswith('aria-') for attr in attributes.keys()):
        features.append('ARIA attributes')
    
    # Role attribute
    if 'role' in attributes:
        features.append(f'Role: {attributes["role"]}')
    
    # Alt text for images
    if 'alt' in attributes:
        features.append('Alt text')
    
    # Labels
    if 'aria-label' in attributes:
        features.append('ARIA label')
    
    # Semantic HTML5 elements
    semantic_tags = ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer']
    if any(tag in content.lower() for tag in semantic_tags):
        features.append('Semantic HTML5')
    
    return features

def generate_html_summary(chunks) -> Dict[str, Any]:
    """Generate summary of HTML parsing results"""
    summary = {
        'total_chunks': len(chunks),
        'chunk_types': defaultdict(int),
        'semantic_types': defaultdict(int),
        'html_elements': defaultdict(int),
        'structure_analysis': {
            'total_headings': 0,
            'navigation_elements': 0,
            'interactive_elements': 0,
            'content_sections': 0,
            'max_heading_level': 0
        },
        'accessibility_features': defaultdict(int),
        'content_features': defaultdict(int)
    }
    
    for chunk in chunks:
        # Type distribution
        summary['chunk_types'][chunk.chunk_type.value] += 1
        
        # Analyze chunk
        analysis = analyze_chunk_structure(chunk)
        html_analysis = analysis['html_analysis']
        
        # Semantic type distribution
        semantic_type = html_analysis['semantic_type']
        summary['semantic_types'][semantic_type] += 1
        
        # HTML element distribution
        tag_name = html_analysis['tag_name']
        if tag_name:
            summary['html_elements'][tag_name] += 1
        
        # Structure analysis
        if 'heading' in analysis['tags']:
            summary['structure_analysis']['total_headings'] += 1
            # Extract heading level
            for tag in analysis['tags']:
                if tag.startswith('heading_level_'):
                    level = int(tag.split('_')[-1])
                    summary['structure_analysis']['max_heading_level'] = max(
                        summary['structure_analysis']['max_heading_level'], level
                    )
        
        if 'navigation' in analysis['tags']:
            summary['structure_analysis']['navigation_elements'] += 1
        
        if 'interactive' in analysis['tags']:
            summary['structure_analysis']['interactive_elements'] += 1
        
        if 'content_section' in analysis['tags']:
            summary['structure_analysis']['content_sections'] += 1
        
        # Accessibility features
        accessibility_features = html_analysis['accessibility_features']
        for feature in accessibility_features:
            summary['accessibility_features'][feature] += 1
        
        # Content features (tags)
        for tag in analysis['tags']:
            summary['content_features'][tag] += 1
    
    return summary

def extract_page_hierarchy(chunks) -> Dict[str, Any]:
    """Extract the hierarchical structure of the HTML page"""
    hierarchy = {
        'headings': [],
        'sections': [],
        'navigation': [],
        'structure': {}
    }
    
    for chunk in chunks:
        analysis = analyze_chunk_structure(chunk)
        html_analysis = analysis['html_analysis']
        tag_name = html_analysis['tag_name']
        
        # Extract headings with hierarchy
        if 'heading' in analysis['tags']:
            heading_level = None
            for tag in analysis['tags']:
                if tag.startswith('heading_level_'):
                    heading_level = int(tag.split('_')[-1])
                    break
            
            # Extract heading text
            heading_text = extract_text_from_content(chunk.content)
            
            hierarchy['headings'].append({
                'level': heading_level,
                'text': heading_text,
                'tag': tag_name,
                'id': html_analysis['element_id'],
                'chunk_id': chunk.id
            })
        
        # Extract sections
        if tag_name in ['section', 'article', 'div'] and html_analysis['element_id']:
            hierarchy['sections'].append({
                'tag': tag_name,
                'id': html_analysis['element_id'],
                'classes': html_analysis['css_classes'],
                'chunk_id': chunk.id
            })
        
        # Extract navigation
        if 'navigation' in analysis['tags']:
            nav_links = extract_links_from_content(chunk.content)
            hierarchy['navigation'].append({
                'tag': tag_name,
                'id': html_analysis['element_id'],
                'links': nav_links,
                'chunk_id': chunk.id
            })
    
    return hierarchy

def extract_text_from_content(content: str) -> str:
    """Extract plain text from HTML content"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', content)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:100] + "..." if len(text) > 100 else text

def extract_links_from_content(content: str) -> List[Dict[str, str]]:
    """Extract links from HTML content"""
    links = []
    # Find href attributes
    href_pattern = r'href=["\']([^"\']+)["\']'
    hrefs = re.findall(href_pattern, content)
    
    # Find link text (simplified)
    link_text_pattern = r'<a[^>]*>([^<]+)</a>'
    texts = re.findall(link_text_pattern, content)
    
    for i, href in enumerate(hrefs):
        text = texts[i] if i < len(texts) else "Link"
        links.append({
            'href': href,
            'text': text.strip()
        })
    
    return links

def print_detailed_html_analysis(content: str, chunks: List, summary: Dict[str, Any], hierarchy: Dict[str, Any]):
    """Print comprehensive analysis of HTML parsing results"""
    print("\n" + "="*80)
    print("🌐 COMPREHENSIVE HTML PARSING ANALYSIS")
    print("="*80)
    
    # File info
    content_size = len(content)
    content_lines = content.count('\n') + 1
    
    print(f"\n📄 HTML DOCUMENT ANALYSIS")
    print("-" * 60)
    print(f"Document size: {content_size:,} characters, {content_lines} lines")
    print(f"Chunks created: {len(chunks)}")
    print(f"Average chunk size: {content_size // len(chunks) if chunks else 0} characters")
    
    # Structure summary
    print(f"\n🏗️  DOCUMENT STRUCTURE")
    print("-" * 60)
    structure = summary['structure_analysis']
    print(f"Headings found: {structure['total_headings']} (max level: h{structure['max_heading_level']})")
    print(f"Navigation elements: {structure['navigation_elements']}")
    print(f"Interactive elements: {structure['interactive_elements']}")
    print(f"Content sections: {structure['content_sections']}")
    
    # Element distribution
    if summary['html_elements']:
        print(f"\n🏷️  HTML ELEMENTS")
        print("-" * 60)
        sorted_elements = sorted(summary['html_elements'].items(), key=lambda x: x[1], reverse=True)
        for element, count in sorted_elements[:10]:  # Top 10
            print(f"   {element}: {count} chunks")
    
    # Semantic types
    if summary['semantic_types']:
        print(f"\n🎯 SEMANTIC TYPES")
        print("-" * 60)
        sorted_types = sorted(summary['semantic_types'].items(), key=lambda x: x[1], reverse=True)
        for semantic_type, count in sorted_types:
            print(f"   {semantic_type}: {count} chunks")
    
    # Accessibility features
    if summary['accessibility_features']:
        print(f"\n♿ ACCESSIBILITY FEATURES")
        print("-" * 60)
        for feature, count in summary['accessibility_features'].items():
            print(f"   {feature}: {count} occurrences")
    
    # Page hierarchy
    if hierarchy['headings']:
        print(f"\n📋 PAGE HIERARCHY")
        print("-" * 60)
        current_level = 0
        for heading in hierarchy['headings']:
            level = heading['level']
            indent = "  " * (level - 1) if level else ""
            heading_id = f" (#{heading['id']})" if heading['id'] else ""
            print(f"   {indent}H{level}: {heading['text']}{heading_id}")
    
    # Navigation structure
    if hierarchy['navigation']:
        print(f"\n🧭 NAVIGATION STRUCTURE")
        print("-" * 60)
        for i, nav in enumerate(hierarchy['navigation'], 1):
            nav_id = f" (#{nav['id']})" if nav['id'] else ""
            print(f"   Navigation {i}{nav_id}:")
            for link in nav['links'][:5]:  # First 5 links
                print(f"     → {link['text']} ({link['href']})")
            if len(nav['links']) > 5:
                print(f"     ... and {len(nav['links']) - 5} more links")
    
    # Detailed chunk analysis
    print(f"\n📋 DETAILED CHUNK ANALYSIS")
    print("-" * 80)
    
    # Group chunks by semantic type for better organization
    chunks_by_type = defaultdict(list)
    for chunk in chunks:
        analysis = analyze_chunk_structure(chunk)
        semantic_type = analysis['html_analysis']['semantic_type']
        chunks_by_type[semantic_type].append((chunk, analysis))
    
    for semantic_type, chunk_list in chunks_by_type.items():
        print(f"\n🔸 {semantic_type.upper()} CHUNKS ({len(chunk_list)})")
        
        for chunk, analysis in chunk_list[:3]:  # Show first 3 of each type
            html_analysis = analysis['html_analysis']
            
            print(f"\n   📌 {analysis['id']}")
            print(f"      Tag: <{html_analysis['tag_name']}>" if html_analysis['tag_name'] else "      Tag: Unknown")
            print(f"      Size: {analysis['size_chars']} chars, {analysis['size_lines']} lines")
            print(f"      Importance: {analysis['importance']:.2f}")
            
            if html_analysis['element_id']:
                print(f"      ID: #{html_analysis['element_id']}")
            
            if html_analysis['css_classes']:
                print(f"      Classes: {', '.join(html_analysis['css_classes'])}")
            
            if analysis['tags']:
                semantic_tags = [tag for tag in analysis['tags'] if not tag.startswith(('class_', 'id_'))]
                if semantic_tags:
                    print(f"      Semantic tags: {', '.join(semantic_tags)}")
            
            if html_analysis['accessibility_features']:
                print(f"      Accessibility: {', '.join(html_analysis['accessibility_features'])}")
            
            if analysis['dependencies']:
                print(f"      Dependencies: {', '.join(analysis['dependencies'])}")
            
            print(f"      Preview: {analysis['preview']}")
        
        if len(chunk_list) > 3:
            print(f"   ... and {len(chunk_list) - 3} more {semantic_type.lower()} chunks")

def test_html_parsing(content: str) -> Dict[str, Any]:
    """Test HTML parsing on the sample content"""
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    # Configure for HTML parsing
    config = ChunkingConfig(
        target_chunk_size=500,  # Reasonable size for HTML elements
        min_chunk_size=50,
        preserve_atomic_nodes=True,
        enable_dependency_tracking=True
    )
    
    engine = ChunkingEngine(config)
    
    # Check if HTML is supported
    if not engine.can_chunk_language('html'):
        print(f"⚠️  HTML parser not available. Supported languages: {engine.get_supported_languages()}")
        return {
            'chunks': [],
            'summary': generate_html_summary([]),
            'hierarchy': extract_page_hierarchy([]),
            'sample_info': {
                'size_chars': len(content),
                'line_count': content.count('\n') + 1,
                'error': 'HTML parser not available'
            }
        }
    
    # Parse HTML
    chunks = engine.chunk_content(content, 'html', 'sample.html')
    
    # Generate analysis
    summary = generate_html_summary(chunks)
    hierarchy = extract_page_hierarchy(chunks)
    
    return {
        'chunks': chunks,
        'summary': summary,
        'hierarchy': hierarchy,
        'sample_info': {
            'size_chars': len(content),
            'line_count': content.count('\n') + 1
        }
    }

def main():
    """Main demo function"""
    print("🌐 HTML PARSING DEMO")
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
        
        if 'html' not in supported_languages:
            print("⚠️  HTML parser not available. This may be due to missing tree-sitter-html.")
            print("   Install with: pip install tree-sitter-html")
            print("   Demo will show what the analysis would look like.")
        
        # Load sample HTML
        print(f"\n📄 Loading sample HTML file...")
        html_content = load_sample_html()
        
        if not html_content:
            print("❌ Cannot proceed without sample HTML file")
            return
        
        line_count = html_content.count('\n') + 1
        print(f"📄 Sample HTML loaded: {len(html_content)} characters, {line_count} lines")
        
        # Test HTML parsing
        print(f"\n🚀 Testing HTML parsing...")
        try:
            result = test_html_parsing(html_content)
            
            if 'error' in result['sample_info']:
                print(f"⚠️  {result['sample_info']['error']}")
            else:
                chunks_count = len(result['chunks'])
                sample_size = result['sample_info']['size_chars']
                print(f"✅ Successfully parsed HTML: {chunks_count} chunks from {sample_size} characters")
                
                # Print comprehensive analysis
                print_detailed_html_analysis(
                    html_content, 
                    result['chunks'], 
                    result['summary'], 
                    result['hierarchy']
                )
                
                # Overall statistics
                print(f"\n📈 OVERALL STATISTICS")
                print("-" * 60)
                
                summary = result['summary']
                print(f"Total chunks created: {summary['total_chunks']}")
                print(f"Unique HTML elements: {len(summary['html_elements'])}")
                print(f"Semantic types found: {len(summary['semantic_types'])}")
                print(f"Accessibility features: {len(summary['accessibility_features'])}")
                
                if summary['total_chunks'] > 0:
                    avg_chunk_size = len(html_content) / summary['total_chunks']
                    print(f"Average chunk size: {avg_chunk_size:.0f} characters")
                
                # Top semantic types
                if summary['semantic_types']:
                    print(f"\nMost common semantic types:")
                    sorted_types = sorted(summary['semantic_types'].items(), key=lambda x: x[1], reverse=True)
                    for semantic_type, count in sorted_types[:5]:
                        print(f"   {semantic_type}: {count} chunks")
                
                print(f"\n🎉 HTML parsing demo completed successfully!")
                print(f"💡 The HTML parser successfully analyzed the sample HTML document")
                print(f"   with intelligent semantic understanding, accessibility awareness,")
                print(f"   and comprehensive structure detection.")
                
                # Usage recommendations
                print(f"\n🎯 HTML parsing insights:")
                structure = summary['structure_analysis']
                print(f"   • Document structure: {structure['total_headings']} headings, {structure['content_sections']} sections")
                print(f"   • Navigation: {structure['navigation_elements']} nav elements found")
                print(f"   • Interactivity: {structure['interactive_elements']} interactive elements")
                print(f"   • Accessibility: {len(summary['accessibility_features'])} accessibility features detected")
                
        except Exception as e:
            print(f"❌ Error during HTML parsing: {e}")
            import traceback
            traceback.print_exc()
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package and tree-sitter-html are installed")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()