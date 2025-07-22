#!/usr/bin/env python3
"""
Clean Parser Demo - Pure Registry Edition
=========================================

Demonstrates the YAML-based parser registry system.
No hardcoding - everything is configured through YAML.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from chuk_code_raptor.chunking import (
    ChunkingEngine,
    ChunkingConfig,
    get_registry,
    get_parser_info,
    get_installation_help,
    get_supported_languages,
    get_supported_extensions
)

def print_separator(title: str):
    """Print a nice separator"""
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)

def demo_registry_system():
    """Demonstrate the YAML-based registry system"""
    print_separator("YAML Parser Registry System")
    
    # Get registry instance
    registry = get_registry()
    print(f"✅ Registry loaded from: {registry.config_path}")
    
    # Show parser statistics
    stats = registry.get_parser_stats()
    print(f"\n📊 Parser Statistics:")
    print(f"   Total configured parsers: {stats['total_parsers']}")
    print(f"   Available parsers: {stats['available_parsers']}")
    print(f"   Unavailable parsers: {stats['unavailable_parsers']}")
    print(f"   Supported languages: {stats['supported_languages']}")
    print(f"   Supported extensions: {stats['supported_extensions']}")
    
    # Show parser types
    if stats['parser_types']:
        print(f"\n🔧 Parser Types:")
        for parser_type, count in stats['parser_types'].items():
            print(f"   {parser_type}: {count}")
    
    # Show package availability
    package_info = stats['package_availability']
    print(f"\n📦 Package Availability:")
    
    if package_info.get('comprehensive'):
        print(f"   Comprehensive packages: {len(package_info['comprehensive'])}")
        for pkg in package_info['comprehensive']:
            print(f"     ✓ {pkg}")
    else:
        print(f"   Comprehensive packages: None")
    
    if package_info.get('individual'):
        print(f"   Individual packages: {len(package_info['individual'])}")
        # Show just a few examples
        individual_items = list(package_info['individual'].items())[:5]
        for lang, pkg in individual_items:
            print(f"     ✓ {lang}: {pkg}")
        if len(package_info['individual']) > 5:
            print(f"     ... and {len(package_info['individual']) - 5} more")
    else:
        print(f"   Individual packages: None")
    
    return True

def demo_parser_discovery():
    """Demonstrate parser discovery capabilities"""
    print_separator("Parser Discovery")
    
    # Get supported languages and extensions
    languages = get_supported_languages()
    extensions = get_supported_extensions()
    
    print(f"🌍 Supported Languages ({len(languages)}):")
    if languages:
        # Group languages for better display
        for i in range(0, len(languages), 6):
            group = languages[i:i+6]
            print(f"   {', '.join(group)}")
    else:
        print("   None available")
    
    print(f"\n📄 Supported Extensions ({len(extensions)}):")
    if extensions:
        # Group extensions for better display
        for i in range(0, len(extensions), 8):
            group = extensions[i:i+8]
            print(f"   {', '.join(group)}")
    else:
        print("   None available")
    
    return len(languages) > 0

def demo_engine_creation():
    """Demonstrate chunking engine creation"""
    print_separator("Chunking Engine Creation")
    
    # Create engine with default config
    config = ChunkingConfig(target_chunk_size=800)
    engine = ChunkingEngine(config)
    
    print(f"✅ ChunkingEngine created successfully")
    
    # Get engine statistics
    stats = engine.get_statistics()
    print(f"\n📊 Engine Statistics:")
    print(f"   Available chunkers: {stats.get('available_chunkers', 0)}")
    print(f"   Configured parsers: {stats.get('configured_parsers', 0)}")
    print(f"   Supported languages: {stats.get('supported_languages', 0)}")
    print(f"   Supported extensions: {stats.get('supported_extensions', 0)}")
    
    if stats.get('parser_types'):
        print(f"\n🔧 Parser Types in Engine:")
        for parser_type, count in stats['parser_types'].items():
            print(f"   {parser_type}: {count}")
    
    return True

def demo_sample_chunking():
    """Demonstrate chunking with sample content"""
    print_separator("Sample Content Chunking")
    
    # Create engine
    engine = ChunkingEngine()
    
    # Sample Python content
    python_content = '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n
    
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib[n]

class MathUtils:
    """Utility class for mathematical operations."""
    
    @staticmethod
    def factorial(n):
        """Calculate factorial of n."""
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n - 1)
    
    @staticmethod
    def is_prime(n):
        """Check if n is a prime number."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

# Example usage
if __name__ == "__main__":
    print(f"Fibonacci(10): {calculate_fibonacci(10)}")
    print(f"Factorial(5): {MathUtils.factorial(5)}")
    print(f"Is 17 prime? {MathUtils.is_prime(17)}")
'''
    
    try:
        # Try to chunk the content
        chunks = engine.chunk_content(python_content, "python", "demo.py")
        
        print(f"✅ Successfully chunked Python content")
        print(f"   Generated {len(chunks)} chunks")
        
        if chunks:
            print(f"\n📄 Chunk Details:")
            for i, chunk in enumerate(chunks, 1):
                chunk_type = chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type)
                preview = chunk.content[:50].replace('\n', ' ') + "..." if len(chunk.content) > 50 else chunk.content.replace('\n', ' ')
                print(f"   {i}. {chunk_type}: {preview}")
                
                # Show semantic tags if available
                if hasattr(chunk, 'semantic_tags') and chunk.semantic_tags:
                    tag_names = [tag.name for tag in chunk.semantic_tags][:3]  # First 3 tags
                    print(f"      Tags: {', '.join(tag_names)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_installation_help():
    """Show installation help"""
    print_separator("Installation Help")
    
    help_text = get_installation_help()
    print(help_text)
    return True

def main():
    """Main demonstration"""
    print("🚀 Clean Parser Demo - Pure Registry Edition")
    print("🌲 Everything configured through YAML - no hardcoding!")
    
    # Run all demonstrations
    success_count = 0
    total_demos = 5
    
    # Demo 1: Registry System
    try:
        if demo_registry_system():
            success_count += 1
    except Exception as e:
        print(f"❌ Registry demo failed: {e}")
    
    # Demo 2: Parser Discovery
    try:
        if demo_parser_discovery():
            success_count += 1
    except Exception as e:
        print(f"❌ Discovery demo failed: {e}")
    
    # Demo 3: Engine Creation
    try:
        if demo_engine_creation():
            success_count += 1
    except Exception as e:
        print(f"❌ Engine demo failed: {e}")
    
    # Demo 4: Sample Chunking (most important test)
    try:
        if demo_sample_chunking():
            success_count += 1
    except Exception as e:
        print(f"❌ Chunking demo failed: {e}")
    
    # Demo 5: Installation Help
    try:
        if demo_installation_help():
            success_count += 1
    except Exception as e:
        print(f"❌ Help demo failed: {e}")
    
    # Summary
    print_separator("Demo Summary")
    print(f"✅ Successful demos: {success_count}/{total_demos}")
    
    if success_count == total_demos:
        print("🎉 All demos completed successfully!")
        print("🚀 Your parser system is fully functional!")
    elif success_count > 0:
        print("⚠️  Some demos succeeded, but there may be missing dependencies")
        print("💡 Try installing more tree-sitter packages for full functionality")
    else:
        print("❌ No demos succeeded - check your installation")
        print("💡 Try: pip install tree-sitter tree-sitter-python")
    
    print("\n🔧 For more information:")
    print("   - Check parser status: engine.print_status()")
    print("   - View configuration: get_registry().config_path")
    print("   - Install packages: pip install tree-sitter tree-sitter-python")
    
    return success_count > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)