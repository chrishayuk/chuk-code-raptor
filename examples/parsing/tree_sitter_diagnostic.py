#!/usr/bin/env python3
"""
Tree-sitter Installation Diagnostic
===================================

Check what tree-sitter packages are available and provide installation guidance.
"""

import sys
import subprocess

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def check_tree_sitter_base():
    """Check tree-sitter base installation"""
    print("🔍 Checking tree-sitter base installation...")
    
    try:
        import tree_sitter
        print("✅ tree-sitter base package is installed")
        
        # Try to create a basic parser
        parser = tree_sitter.Parser()
        print("✅ tree-sitter Parser can be created")
        
        return True
    except ImportError:
        print("❌ tree-sitter base package is NOT installed")
        print("💡 Install with: pip install tree-sitter")
        return False
    except Exception as e:
        print(f"❌ tree-sitter base package error: {e}")
        return False

def check_individual_packages():
    """Check individual tree-sitter language packages"""
    print("\n🔍 Checking individual tree-sitter language packages...")
    
    packages = [
        ('tree_sitter_python', 'Python'),
        ('tree_sitter_javascript', 'JavaScript'),
        ('tree_sitter_html', 'HTML'),
        ('tree_sitter_css', 'CSS'),
        ('tree_sitter_json', 'JSON'),
        ('tree_sitter_markdown', 'Markdown'),
        ('tree_sitter_xml', 'XML'),
        ('tree_sitter_rust', 'Rust'),
        ('tree_sitter_go', 'Go'),
        ('tree_sitter_java', 'Java'),
        ('tree_sitter_cpp', 'C++'),
        ('tree_sitter_c', 'C'),
        ('tree_sitter_yaml', 'YAML'),
    ]
    
    available = []
    missing = []
    
    for package, lang in packages:
        if check_package(package):
            available.append((package, lang))
            print(f"✅ {lang}: {package}")
        else:
            missing.append((package, lang))
            print(f"❌ {lang}: {package}")
    
    return available, missing

def check_comprehensive_packages():
    """Check comprehensive tree-sitter packages"""
    print("\n🔍 Checking comprehensive tree-sitter packages...")
    
    packages = [
        ('tree_sitter_languages', 'tree-sitter-languages'),
        ('tree_sitter_language_pack', 'tree-sitter-language-pack'),
    ]
    
    available = []
    
    for package, display_name in packages:
        if check_package(package):
            available.append((package, display_name))
            print(f"✅ {display_name}: {package}")
        else:
            print(f"❌ {display_name}: {package}")
    
    return available

def test_language_creation():
    """Test creating language objects"""
    print("\n🔍 Testing language object creation...")
    
    try:
        import tree_sitter
        
        # Test Python if available
        try:
            import tree_sitter_python
            py_lang = tree_sitter.Language(tree_sitter_python.language())
            print("✅ Python language object created successfully")
        except ImportError:
            print("⚠️  tree_sitter_python not available for testing")
        except Exception as e:
            print(f"❌ Failed to create Python language object: {e}")
        
        # Test JavaScript if available
        try:
            import tree_sitter_javascript
            js_lang = tree_sitter.Language(tree_sitter_javascript.language())
            print("✅ JavaScript language object created successfully")
        except ImportError:
            print("⚠️  tree_sitter_javascript not available for testing")
        except Exception as e:
            print(f"❌ Failed to create JavaScript language object: {e}")
            
    except ImportError:
        print("❌ Cannot test language creation - tree-sitter base not available")

def provide_installation_guidance(available_individual, missing_individual, available_comprehensive):
    """Provide installation guidance"""
    print("\n💡 Installation Guidance")
    print("=" * 40)
    
    if not check_package('tree_sitter'):
        print("🚨 PRIORITY: Install tree-sitter base package:")
        print("   pip install tree-sitter")
        print()
    
    if available_comprehensive:
        print("✅ You have comprehensive packages installed:")
        for package, name in available_comprehensive:
            print(f"   {name}")
        print("   This should provide good language coverage!")
    else:
        print("📦 No comprehensive packages found.")
        print("   For maximum convenience, install:")
        print("   pip install tree-sitter-languages")
        print("   OR")
        print("   pip install tree-sitter-language-pack")
        print()
    
    if available_individual:
        print(f"✅ You have {len(available_individual)} individual packages:")
        for package, lang in available_individual[:5]:  # Show first 5
            print(f"   {lang}")
        if len(available_individual) > 5:
            print(f"   ... and {len(available_individual) - 5} more")
        print()
    
    if missing_individual:
        print("📋 To install missing individual packages:")
        for package, lang in missing_individual[:5]:  # Show first 5
            pip_name = package.replace('_', '-')
            print(f"   pip install {pip_name}  # for {lang}")
        if len(missing_individual) > 5:
            print(f"   ... and {len(missing_individual) - 5} more")
        print()
    
    # Provide specific recommendations
    if len(available_individual) == 0 and len(available_comprehensive) == 0:
        print("🚨 RECOMMENDED ACTION:")
        print("   pip install tree-sitter tree-sitter-python tree-sitter-javascript")
        print("   This will get you started with the most common languages.")
    elif len(available_individual) < 3 and len(available_comprehensive) == 0:
        print("💡 RECOMMENDED ACTION:")
        print("   pip install tree-sitter-languages")
        print("   This will give you comprehensive language support.")

def main():
    """Main diagnostic function"""
    print("🌲 Tree-sitter Installation Diagnostic")
    print("=" * 50)
    
    # Check base installation
    has_base = check_tree_sitter_base()
    
    # Check packages
    available_individual, missing_individual = check_individual_packages()
    available_comprehensive = check_comprehensive_packages()
    
    # Test functionality
    if has_base:
        test_language_creation()
    
    # Provide guidance
    provide_installation_guidance(available_individual, missing_individual, available_comprehensive)
    
    # Summary
    print("\n📊 Summary")
    print("=" * 20)
    print(f"tree-sitter base: {'✅' if has_base else '❌'}")
    print(f"Individual packages: {len(available_individual)}/{len(available_individual) + len(missing_individual)}")
    print(f"Comprehensive packages: {len(available_comprehensive)}/2")
    
    if has_base and (available_individual or available_comprehensive):
        print("\n🎉 You're ready to use tree-sitter parsers!")
        return True
    else:
        print("\n⚠️  Some packages need to be installed for full functionality.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)