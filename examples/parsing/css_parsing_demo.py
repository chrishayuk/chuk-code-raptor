#!/usr/bin/env python3
"""
CSS Parser Demo with HTML-CSS Dependency Tracking
=================================================

Demonstrates the CSS parser capabilities and HTML-CSS relationship analysis.
Shows semantic CSS analysis, selector parsing, and cross-file dependencies.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_sample_css_data() -> Dict[str, str]:
    """Create various CSS samples for testing"""
    
    samples = {}
    
    # Basic CSS with selectors
    samples['basic_styles'] = """/* Basic site styles */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0;
    padding: 0;
    line-height: 1.6;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Header styles */
.site-header {
    background: #fff;
    border-bottom: 1px solid #e5e7eb;
    position: sticky;
    top: 0;
    z-index: 100;
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
}

/* Navigation */
.main-navigation ul {
    display: flex;
    list-style: none;
    margin: 0;
    padding: 0;
    gap: 2rem;
}

.nav-link {
    text-decoration: none;
    color: #6b7280;
    font-weight: 500;
    transition: color 0.2s ease;
}

.nav-link:hover {
    color: #3b82f6;
}

/* Button styles */
.btn {
    display: inline-flex;
    align-items: center;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.2s ease;
    border: none;
    cursor: pointer;
}

.btn-primary {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    color: white;
    box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
}

.btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 15px rgba(59, 130, 246, 0.3);
}

/* Hero section */
#hero {
    padding: 4rem 0;
    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 700;
    margin: 0 0 1.5rem;
    background: linear-gradient(135deg, #1e293b, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}"""

    # Responsive CSS with media queries
    samples['responsive_styles'] = """/* Responsive design styles */

/* Desktop first approach */
.grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2rem;
}

.sidebar {
    width: 300px;
    position: fixed;
    height: 100vh;
    overflow-y: auto;
}

/* Tablet styles */
@media (max-width: 1024px) {
    .grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
    }
    
    .sidebar {
        width: 250px;
    }
    
    .hero-title {
        font-size: 2.5rem;
    }
}

/* Mobile styles */
@media (max-width: 768px) {
    .container {
        padding: 0 16px;
    }
    
    .grid {
        grid-template-columns: 1fr;
        gap: 1rem;
    }
    
    .sidebar {
        position: static;
        width: 100%;
        height: auto;
    }
    
    .header-content {
        flex-direction: column;
        gap: 1rem;
    }
    
    .main-navigation ul {
        flex-direction: column;
        gap: 1rem;
    }
    
    .hero-title {
        font-size: 2rem;
    }
}

/* Small mobile */
@media (max-width: 480px) {
    .btn {
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
    }
    
    .hero-title {
        font-size: 1.75rem;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    body {
        background: #0f172a;
        color: #e2e8f0;
    }
    
    .site-header {
        background: #1e293b;
        border-bottom-color: #334155;
    }
    
    .nav-link {
        color: #94a3b8;
    }
    
    .nav-link:hover {
        color: #60a5fa;
    }
}

/* Reduced motion preference */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}"""

    # CSS animations and keyframes
    samples['animations'] = """/* Animation styles */

/* Keyframe animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInLeft {
    from {
        transform: translateX(-100%);
    }
    to {
        transform: translateX(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
}

@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

/* Animation classes */
.animate-fade-in {
    animation: fadeInUp 0.6s ease-out;
}

.animate-slide-in {
    animation: slideInLeft 0.8s ease-out;
}

.animate-pulse {
    animation: pulse 2s infinite;
}

.loading-spinner {
    animation: spin 1s linear infinite;
}

/* Hover animations */
.hover-lift {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.hover-lift:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
}

/* Complex animations */
.bounce-in {
    animation: bounceIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

@keyframes bounceIn {
    0% {
        opacity: 0;
        transform: scale(0.3);
    }
    50% {
        opacity: 1;
        transform: scale(1.05);
    }
    70% {
        transform: scale(0.9);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

/* Staggered animations */
.stagger-animation {
    animation-delay: calc(var(--animation-order, 0) * 0.1s);
}"""

    # CSS imports and external dependencies
    samples['imports_and_fonts'] = """/* External imports and fonts */

/* Google Fonts import */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Local font imports */
@import url('./fonts/custom-icons.css');

/* Component imports */
@import './components/buttons.css';
@import './components/forms.css';
@import './components/cards.css';

/* Custom font face definitions */
@font-face {
    font-family: 'CustomFont';
    src: url('../fonts/custom-font.woff2') format('woff2'),
         url('../fonts/custom-font.woff') format('woff');
    font-weight: 400;
    font-style: normal;
    font-display: swap;
}

/* CSS custom properties (variables) */
:root {
    --primary-color: #3b82f6;
    --secondary-color: #1e293b;
    --accent-color: #f59e0b;
    --text-color: #374151;
    --background-color: #ffffff;
    --border-color: #e5e7eb;
    
    --font-family-sans: 'Inter', system-ui, sans-serif;
    --font-family-mono: 'Fira Code', monospace;
    
    --border-radius: 0.5rem;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

/* Dark theme variables */
[data-theme="dark"] {
    --primary-color: #60a5fa;
    --secondary-color: #f8fafc;
    --text-color: #e5e7eb;
    --background-color: #0f172a;
    --border-color: #374151;
}"""

    return samples

def analyze_css_chunk(chunk) -> Dict[str, Any]:
    """Analyze the structure and properties of a CSS chunk"""
    analysis = {
        'id': chunk.id,
        'type': chunk.chunk_type.value,
        'size_chars': len(chunk.content),
        'size_lines': chunk.line_count,
        'importance': chunk.importance_score,
        'tags': [tag.name for tag in chunk.semantic_tags] if hasattr(chunk, 'semantic_tags') and chunk.semantic_tags else [],
        'metadata': chunk.metadata or {},
        'dependencies': chunk.dependencies[:5] if chunk.dependencies else [],
        'preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
    }
    
    # CSS-specific analysis
    css_selector = chunk.metadata.get('css_selector') if chunk.metadata else None
    css_properties = chunk.metadata.get('css_properties', []) if chunk.metadata else []
    selector_specificity = chunk.metadata.get('selector_specificity', 0) if chunk.metadata else 0
    html_targets = chunk.metadata.get('html_targets', []) if chunk.metadata else []
    
    analysis['css_analysis'] = {
        'selector': css_selector,
        'properties': css_properties,
        'specificity': selector_specificity,
        'html_targets': html_targets,
        'css_category': detect_css_category(analysis['tags']),
        'responsive_features': detect_responsive_features(chunk.content, analysis['tags'])
    }
    
    return analysis

def detect_css_category(tags: List[str]) -> str:
    """Detect the category of CSS rule"""
    if 'css_media_query' in tags:
        return 'Media Query'
    elif 'css_animation' in tags or 'keyframes' in tags:
        return 'Animation'
    elif 'css_import' in tags:
        return 'Import'
    elif 'css_at_rule' in tags:
        return 'At-Rule'
    elif 'id_selector' in tags:
        return 'ID Selector'
    elif 'class_selector' in tags:
        return 'Class Selector'
    elif 'element_selector' in tags:
        return 'Element Selector'
    elif 'pseudo_class' in tags or 'pseudo_element' in tags:
        return 'Pseudo Selector'
    else:
        return 'CSS Rule'

def detect_responsive_features(content: str, tags: List[str]) -> List[str]:
    """Detect responsive design features in CSS"""
    features = []
    content_lower = content.lower()
    
    if 'breakpoint_rule' in tags or 'min-width' in content_lower or 'max-width' in content_lower:
        features.append('Responsive breakpoints')
    
    if 'color_scheme_preference' in tags:
        features.append('Dark mode support')
    
    if 'motion_preference' in tags:
        features.append('Reduced motion support')
    
    if 'css_layout' in tags or 'grid' in content_lower or 'flexbox' in content_lower:
        features.append('Modern layout')
    
    if '@container' in content_lower:
        features.append('Container queries')
    
    return features

def generate_css_summary(chunks) -> Dict[str, Any]:
    """Generate summary of CSS parsing results"""
    summary = {
        'total_chunks': len(chunks),
        'chunk_types': defaultdict(int),
        'css_categories': defaultdict(int),
        'css_features': defaultdict(int),
        'selector_analysis': {
            'total_selectors': 0,
            'id_selectors': 0,
            'class_selectors': 0,
            'element_selectors': 0,
            'pseudo_selectors': 0,
            'avg_specificity': 0
        },
        'responsive_features': defaultdict(int),
        'imports_and_dependencies': defaultdict(int)
    }
    
    total_specificity = 0
    
    for chunk in chunks:
        # Type distribution
        summary['chunk_types'][chunk.chunk_type.value] += 1
        
        # Analyze chunk
        analysis = analyze_css_chunk(chunk)
        css_analysis = analysis['css_analysis']
        
        # CSS category distribution
        css_category = css_analysis['css_category']
        summary['css_categories'][css_category] += 1
        
        # Selector analysis
        if css_analysis['selector']:
            summary['selector_analysis']['total_selectors'] += 1
            total_specificity += css_analysis['specificity']
            
            # Count selector types
            if 'id_selector' in analysis['tags']:
                summary['selector_analysis']['id_selectors'] += 1
            if 'class_selector' in analysis['tags']:
                summary['selector_analysis']['class_selectors'] += 1
            if 'element_selector' in analysis['tags']:
                summary['selector_analysis']['element_selectors'] += 1
            if 'pseudo_class' in analysis['tags'] or 'pseudo_element' in analysis['tags']:
                summary['selector_analysis']['pseudo_selectors'] += 1
        
        # Responsive features
        responsive_features = css_analysis['responsive_features']
        for feature in responsive_features:
            summary['responsive_features'][feature] += 1
        
        # CSS features from tags
        for tag in analysis['tags']:
            if tag.startswith('css_'):
                summary['css_features'][tag] += 1
        
        # Dependencies
        for dep in analysis['dependencies']:
            if ':' in dep:
                dep_type = dep.split(':', 1)[0]
                summary['imports_and_dependencies'][dep_type] += 1
    
    # Calculate average specificity
    if summary['selector_analysis']['total_selectors'] > 0:
        summary['selector_analysis']['avg_specificity'] = total_specificity / summary['selector_analysis']['total_selectors']
    
    return summary

def test_css_parsing(content: str, sample_name: str) -> Dict[str, Any]:
    """Test CSS parsing on sample content"""
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    # Configure for CSS parsing
    config = ChunkingConfig(
        target_chunk_size=300,  # Good size for CSS rules
        min_chunk_size=30,
        preserve_atomic_nodes=True,
        enable_dependency_tracking=True
    )
    
    engine = ChunkingEngine(config)
    
    # Check if CSS is supported
    if not engine.can_chunk_language('css'):
        print(f"⚠️  CSS parser not available. Supported languages: {engine.get_supported_languages()}")
        return {
            'chunks': [],
            'summary': generate_css_summary([]),
            'sample_info': {
                'size_chars': len(content),
                'line_count': content.count('\n') + 1,
                'error': 'CSS parser not available'
            }
        }
    
    # Parse CSS
    chunks = engine.chunk_content(content, 'css', f'{sample_name}.css')
    
    # Generate analysis
    summary = generate_css_summary(chunks)
    
    return {
        'chunks': chunks,
        'summary': summary,
        'sample_info': {
            'size_chars': len(content),
            'line_count': content.count('\n') + 1
        }
    }

def demonstrate_html_css_relationships():
    """Demonstrate HTML-CSS dependency tracking"""
    print(f"\n🔗 HTML-CSS DEPENDENCY TRACKING DEMO")
    print("=" * 60)
    
    try:
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        from chuk_code_raptor.chunking.parsers.css import CSSParser
        
        # Sample HTML
        html_content = """
        <div class="container">
            <header id="main-header" class="site-header">
                <h1 class="site-title">My Site</h1>
                <nav class="main-navigation">
                    <ul class="nav-list">
                        <li class="nav-item"><a href="#" class="nav-link">Home</a></li>
                    </ul>
                </nav>
            </header>
            <main class="content">
                <section id="hero" class="hero-section">
                    <h2 class="hero-title">Welcome</h2>
                </section>
            </main>
        </div>
        """
        
        # Sample CSS
        css_content = """
        .container { max-width: 1200px; margin: 0 auto; }
        #main-header { background: white; padding: 1rem; }
        .site-header { border-bottom: 1px solid #ccc; }
        .site-title { font-size: 2rem; color: #333; }
        .main-navigation ul { display: flex; list-style: none; }
        .nav-link { text-decoration: none; color: #666; }
        .nav-link:hover { color: #0066cc; }
        #hero { padding: 4rem 0; background: #f5f5f5; }
        .hero-title { font-size: 3rem; text-align: center; }
        """
        
        config = ChunkingConfig(target_chunk_size=200, min_chunk_size=20)
        engine = ChunkingEngine(config)
        
        # Parse both HTML and CSS
        print("Parsing HTML and CSS content...")
        html_chunks = engine.chunk_content(html_content, 'html', 'sample.html')
        css_chunks = engine.chunk_content(css_content, 'css', 'sample.css')
        
        print(f"✅ HTML chunks: {len(html_chunks)}")
        print(f"✅ CSS chunks: {len(css_chunks)}")
        
        # Analyze relationships
        if css_chunks and html_chunks:
            css_parser = CSSParser(config)
            relationships = css_parser.get_html_css_relationships(css_chunks, html_chunks)
            
            print(f"\n🔗 HTML-CSS Relationships Found:")
            for css_id, html_ids in relationships.items():
                css_chunk = next((c for c in css_chunks if c.id == css_id), None)
                if css_chunk:
                    selector = css_chunk.metadata.get('css_selector', 'Unknown')
                    print(f"   CSS: {selector}")
                    print(f"   → Targets {len(html_ids)} HTML element(s)")
                    for html_id in html_ids[:3]:  # Show first 3
                        html_chunk = next((c for c in html_chunks if c.id == html_id), None)
                        if html_chunk:
                            tag = html_chunk.metadata.get('tag_name', 'unknown')
                            element_id = html_chunk.metadata.get('element_id', '')
                            classes = html_chunk.metadata.get('css_classes', [])
                            print(f"     • <{tag}> {f'#{element_id}' if element_id else ''} {'.'.join(classes)}")
                    print()
        
    except Exception as e:
        print(f"❌ Error in dependency tracking demo: {e}")

def print_detailed_css_analysis(samples: Dict[str, str], all_results: Dict[str, Any]):
    """Print comprehensive analysis of CSS parsing results"""
    print("\n" + "="*80)
    print("🎨 COMPREHENSIVE CSS PARSING ANALYSIS")
    print("="*80)
    
    for sample_name, result in all_results.items():
        if 'error' in result['sample_info']:
            continue
            
        chunks = result['chunks']
        summary = result['summary']
        
        print(f"\n🎨 SAMPLE: {sample_name.upper().replace('_', ' ')}")
        print("-" * 60)
        
        # Sample info
        sample_size = result['sample_info']['size_chars']
        sample_lines = result['sample_info']['line_count']
        
        print(f"📄 Sample size: {sample_size} characters, {sample_lines} lines")
        print(f"🧩 Chunks created: {len(chunks)}")
        
        # CSS summary
        print(f"\n🎯 CSS Structure Summary:")
        selector_analysis = summary['selector_analysis']
        print(f"   Total CSS rules: {selector_analysis['total_selectors']}")
        print(f"   Average specificity: {selector_analysis['avg_specificity']:.1f}")
        print(f"   ID selectors: {selector_analysis['id_selectors']}")
        print(f"   Class selectors: {selector_analysis['class_selectors']}")
        print(f"   Element selectors: {selector_analysis['element_selectors']}")
        
        # CSS categories
        if summary['css_categories']:
            print(f"\n📊 CSS Categories:")
            for category, count in summary['css_categories'].items():
                print(f"   {category}: {count} chunks")
        
        # Responsive features
        if summary['responsive_features']:
            print(f"\n📱 Responsive Features:")
            for feature, count in summary['responsive_features'].items():
                print(f"   {feature}: {count} occurrences")
        
        # Dependencies
        if summary['imports_and_dependencies']:
            print(f"\n🔗 Dependencies:")
            for dep_type, count in summary['imports_and_dependencies'].items():
                print(f"   {dep_type}: {count} references")

def main():
    """Main demo function"""
    print("🎨 CSS PARSING DEMO")
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
        
        if 'css' not in supported_languages:
            print("⚠️  CSS parser not available. This may be due to missing tree-sitter-css.")
            print("   Install with: pip install tree-sitter-css")
            print("   Demo will show what the analysis would look like.")
        
        # Create sample CSS data
        print(f"\n📝 Creating sample CSS data...")
        samples = create_sample_css_data()
        print(f"✅ Created {len(samples)} CSS samples")
        
        # Test each sample
        all_results = {}
        
        print(f"\n🚀 Testing CSS parsing...")
        for sample_name, sample_content in samples.items():
            print(f"   🧪 Processing {sample_name}...")
            
            try:
                result = test_css_parsing(sample_content, sample_name)
                all_results[sample_name] = result
                
                if 'error' in result['sample_info']:
                    print(f"      ⚠️  {result['sample_info']['error']}")
                else:
                    chunks_count = len(result['chunks'])
                    sample_size = result['sample_info']['size_chars']
                    print(f"      ✅ {chunks_count} chunks from {sample_size} characters")
                    
            except Exception as e:
                print(f"      ❌ Error processing {sample_name}: {e}")
        
        # Print comprehensive analysis
        print_detailed_css_analysis(samples, all_results)
        
        # Demonstrate HTML-CSS relationships
        demonstrate_html_css_relationships()
        
        print(f"\n🎉 CSS parsing demo completed successfully!")
        print(f"💡 The CSS parser provides sophisticated analysis of stylesheets")
        print(f"   with semantic understanding, dependency tracking, and HTML relationships.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package and tree-sitter-css are installed")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()