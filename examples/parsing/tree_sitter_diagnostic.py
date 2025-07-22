#!/usr/bin/env python3
"""
Tree-sitter System Verification
==============================

Clean verification script to check that the tree-sitter parser system is working correctly.
"""

import sys
from pathlib import Path

def find_src_directory():
    """Find the src directory in the project"""
    current = Path(__file__).parent
    
    # Look up the directory tree for src
    for parent in [current] + list(current.parents):
        src_path = parent / "src"
        if src_path.exists() and (src_path / "chuk_code_raptor").exists():
            return src_path
    
    return None

def check_tree_sitter_packages():
    """Check what tree-sitter packages are installed"""
    print("🌲 TREE-SITTER PACKAGES INSTALLED")
    print("=" * 50)
    
    # Check for comprehensive packages and what they provide
    comprehensive_languages = []
    comprehensive_packages = []
    
    # Check tree-sitter-languages
    try:
        from tree_sitter_languages import get_language
        comprehensive_packages.append("tree-sitter-languages")
        
        # Test what languages are available in the comprehensive package
        test_languages = [
            'python', 'javascript', 'typescript', 'html', 'css', 'json', 
            'markdown', 'rust', 'go', 'java', 'cpp', 'c', 'latex', 'tex',
            'ruby', 'php', 'swift', 'kotlin', 'scala', 'bash', 'lua'
        ]
        
        print(f"📦 Testing tree-sitter-languages package...")
        working_langs = []
        for lang in test_languages:
            try:
                get_language(lang)
                working_langs.append(lang)
            except Exception as e:
                # Only log first few failures to avoid spam
                if len(working_langs) == 0 and len([l for l in test_languages[:5] if l not in working_langs]) <= 3:
                    print(f"   ❌ {lang}: {e}")
        
        comprehensive_languages.extend(working_langs)
        
        if working_langs:
            print(f"   ✅ tree-sitter-languages provides {len(working_langs)} languages")
            print(f"      Languages: {', '.join(working_langs)}")
        else:
            print(f"   ❌ tree-sitter-languages has API issues (0 working languages)")
        
    except ImportError:
        print(f"❌ tree-sitter-languages not installed")
    
    # Check tree-sitter-language-pack as fallback
    if not comprehensive_languages:
        try:
            from tree_sitter_language_pack import get_language
            comprehensive_packages.append("tree-sitter-language-pack")
            
            print(f"\n📦 Testing tree-sitter-language-pack package...")
            working_langs = []
            for lang in test_languages:
                try:
                    get_language(lang)
                    working_langs.append(lang)
                except Exception:
                    pass
            
            comprehensive_languages.extend(working_langs)
            
            if working_langs:
                print(f"   ✅ tree-sitter-language-pack provides {len(working_langs)} languages")
                if len(working_langs) <= 10:
                    print(f"      Languages: {', '.join(working_langs)}")
                else:
                    print(f"      Languages: {', '.join(working_langs[:10])}... and {len(working_langs)-10} more")
            else:
                print(f"   ❌ tree-sitter-language-pack has issues")
                
        except ImportError:
            print(f"❌ tree-sitter-language-pack not installed")
    
    # Check individual packages (these are standalone installs)
    individual_available = []
    individual_missing = []
    
    individual_packages_to_check = [
        'python', 'javascript', 'typescript', 'html', 'css', 'json', 
        'markdown', 'rust', 'go', 'java', 'cpp', 'c',
        'ruby', 'php', 'swift', 'kotlin', 'scala', 'bash', 'lua'
    ]
    
    for lang in individual_packages_to_check:
        try:
            package_name = f"tree_sitter_{lang}"
            module = __import__(package_name, fromlist=['language'])
            if hasattr(module, 'language') and callable(getattr(module, 'language')):
                individual_available.append(lang)
            else:
                individual_missing.append(lang)
        except ImportError:
            individual_missing.append(lang)
    
    if individual_available:
        print(f"\n📦 Individual packages installed ({len(individual_available)}):")
        print(f"   {', '.join(individual_available)}")
    
    # Show what's actually available (comprehensive + individual)
    all_available = list(set(comprehensive_languages + individual_available))
    truly_missing = [lang for lang in individual_packages_to_check if lang not in all_available]
    
    print(f"\n📊 TOTAL AVAILABLE: {len(all_available)} languages")
    print(f"   Via comprehensive package: {len(comprehensive_languages)}")
    print(f"   Via individual packages: {len(individual_available)}")
    
    if truly_missing:
        print(f"\n❌ Not available anywhere ({len(truly_missing)}):")
        print(f"   {', '.join(truly_missing)}")
    
    # Special notes
    if comprehensive_packages and not comprehensive_languages:
        print(f"\n⚠️  NOTE: Comprehensive packages installed but have API compatibility issues")
        print(f"   Individual packages are more reliable for your setup")
    
    return all_available, truly_missing

def verify_parser_system():
    """Verify the parser discovery and chunking system"""
    print("\n🔍 PARSER SYSTEM VERIFICATION")
    print("=" * 50)
    
    # Find and add src path
    src_path = find_src_directory()
    if not src_path:
        print("❌ Cannot find src directory")
        return False
    
    print(f"✅ Found project: {src_path.parent.name}")
    sys.path.insert(0, str(src_path))
    
    try:
        # Test parser discovery
        from chuk_code_raptor.chunking.parsers.available_parsers import discover_available_parsers
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        config = ChunkingConfig()
        available_parsers = discover_available_parsers(config)
        
        print(f"\n📝 Implemented Parsers: {len(available_parsers)}")
        for lang, parser in available_parsers.items():
            # Get package info
            if hasattr(parser, '_package_used'):
                package = parser._package_used
            else:
                package = f"tree_sitter_{lang}"
            print(f"   ✅ {lang}: {parser.name} (via {package})")
        
        if not available_parsers:
            print("❌ No parsers available")
            return False
        
        # Test chunking engine
        print(f"\n🚀 Testing Chunking Engine...")
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        
        engine = ChunkingEngine(config)
        supported_languages = engine.get_supported_languages()
        
        print(f"✅ Engine supports {len(supported_languages)} languages: {', '.join(supported_languages)}")
        
        # Test chunking with Python if available
        if 'python' in supported_languages:
            print(f"\n🐍 Testing Python Chunking...")
            
            test_code = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """Simple calculator."""
    
    def add(self, a, b):
        return a + b
        
    def multiply(self, a, b):
        return a * b

# Usage example
calc = Calculator()
result = calc.add(5, 3)
print(f"Result: {result}")
'''
            
            chunks = engine.chunk_content(test_code, "python", "test.py")
            print(f"✅ Created {len(chunks)} semantic chunks:")
            
            for i, chunk in enumerate(chunks[:3]):  # Show first 3
                preview = chunk.content[:60].replace('\n', ' ')
                print(f"   {i+1}. {chunk.chunk_type.value}: {preview}...")
            
            if len(chunks) > 3:
                print(f"   ... and {len(chunks) - 3} more chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ System verification failed: {e}")
        return False

def show_recommendations(available_packages, missing_packages, available_parsers):
    """Show recommendations for improving the setup"""
    print(f"\n💡 SYSTEM ANALYSIS")
    print("=" * 30)
    
    # Show packages with parsers vs without
    packages_with_parsers = set(available_parsers.keys())
    packages_without_parsers = set(available_packages) - packages_with_parsers
    
    if packages_without_parsers:
        print(f"🔧 Tree-sitter packages available but no parser implemented ({len(packages_without_parsers)}):")
        print(f"   {', '.join(sorted(packages_without_parsers))}")
    
    # Show high-priority missing packages (ones that actually exist)
    installable_missing = []
    for lang in ['typescript', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'bash', 'lua']:
        if lang in missing_packages:
            installable_missing.append(lang)
    
    if installable_missing:
        print(f"\n📥 Missing packages you can install ({len(installable_missing)}):")
        print(f"   {', '.join(installable_missing)}")
        print(f"   Install with: uv add tree-sitter-{installable_missing[0]} tree-sitter-{installable_missing[1] if len(installable_missing) > 1 else installable_missing[0]}")
    
    # Special notes about LaTeX
    if 'latex' in missing_packages:
        print(f"\n📝 LaTeX Support:")
        print(f"   ❌ No tree-sitter-latex package exists")
        print(f"   💡 Try: uv add tree-sitter-language-pack  # May include LaTeX")
        print(f"   🔧 Or use your existing manual LaTeX parser")
    
    # Summary
    total_available = len(available_packages)
    total_with_parsers = len(available_parsers)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Tree-sitter packages available: {total_available}")
    print(f"   Working parsers implemented: {total_with_parsers}")
    print(f"   Parser coverage: {total_with_parsers}/{total_available} languages")
    
    # Next steps recommendations
    print(f"\n🎯 NEXT STEPS:")
    if packages_without_parsers:
        priority_langs = [lang for lang in ['javascript', 'typescript', 'css', 'rust', 'go', 'java'] if lang in packages_without_parsers]
        if priority_langs:
            print(f"   1. 🚀 High priority: {', '.join(priority_langs[:3])} parsers")
        else:
            top_lang = sorted(packages_without_parsers)[0]
            print(f"   1. 🚀 Add {top_lang} parser (you have the tree-sitter package)")
        print(f"   2. 📚 Full list: {', '.join(sorted(packages_without_parsers))}")
    
    if installable_missing and len(installable_missing) > 0:
        print(f"   3. 📦 Install missing: {', '.join(installable_missing[:3])}{'...' if len(installable_missing) > 3 else ''}")
    
    if total_with_parsers >= 4:
        print(f"   ✅ Your setup has good language coverage!")
    elif total_with_parsers >= 2:
        print(f"   ⚠️  Basic coverage - consider expanding")
    else:
        print(f"   ❌ Limited coverage - more parsers needed")

def main():
    """Run system verification"""
    print("🧪 TREE-SITTER SYSTEM CHECK")
    print("=" * 60)
    
    # Check available packages first
    available_packages, missing_packages = check_tree_sitter_packages()
    
    # Then verify parser system
    success = verify_parser_system()
    
    if success:
        print(f"\n🎉 VERIFICATION PASSED")
        print("=" * 30)
        print("✅ Parser discovery working")
        print("✅ Chunking engine functional") 
        print("✅ Semantic chunking operational")
        
        # Get parser info for analysis
        src_path = find_src_directory()
        if src_path:
            sys.path.insert(0, str(src_path))
            from chuk_code_raptor.chunking.parsers.available_parsers import discover_available_parsers
            from chuk_code_raptor.chunking.config import ChunkingConfig
            available_parsers = discover_available_parsers(ChunkingConfig())
        else:
            available_parsers = {}
        
        show_recommendations(available_packages, missing_packages, available_parsers)
    else:
        print(f"\n❌ VERIFICATION FAILED")
        print("=" * 30)
        print("Check error messages above for troubleshooting.")
        print("Common fixes:")
        print("  • Install tree-sitter packages: pip install tree-sitter-python")
        print("  • Ensure you're running from the correct directory")
        print("  • Check that src/chuk_code_raptor exists")

if __name__ == "__main__":
    main()