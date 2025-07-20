#!/usr/bin/env python3
"""
Debug Parser Loading
===================

Quick script to debug why the Python parser isn't loading.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def debug_parser_loading():
    print("🔍 DEBUGGING PARSER LOADING")
    print("=" * 50)
    
    try:
        print("1. Testing ChunkingEngine import...")
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        print("✅ ChunkingEngine imported successfully")
        
        print("\n2. Testing Python parser import directly...")
        try:
            from chuk_code_raptor.chunking.parsers.python import PythonParser
            print("✅ PythonParser imported successfully")
            
            print("\n3. Testing Python parser initialization...")
            from chuk_code_raptor.chunking.config import ChunkingConfig
            config = ChunkingConfig()
            parser = PythonParser(config)
            print(f"✅ PythonParser initialized: {parser.name}")
            print(f"   Supported languages: {parser.supported_languages}")
            print(f"   Supported extensions: {parser.supported_extensions}")
            print(f"   Can parse Python: {parser.can_parse('python', '.py')}")
            
        except Exception as e:
            print(f"❌ Python parser error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n4. Testing engine initialization...")
        config = ChunkingConfig()
        engine = ChunkingEngine(config)
        print(f"✅ Engine initialized")
        print(f"   Parsers count: {len(engine.parsers)}")
        print(f"   Supported languages: {engine.get_supported_languages()}")
        print(f"   Available parsers: {list(engine.parsers.keys())}")
        
        print("\n5. Testing direct parser lookup...")
        python_parser = engine._get_parser('python')
        if python_parser:
            print(f"✅ Found Python parser: {python_parser.name}")
        else:
            print("❌ No Python parser found in engine")
        
        print("\n6. Testing parser registry process...")
        # Let's see what the engine's _register_parsers method found
        if hasattr(engine, 'stats'):
            print(f"   Engine stats: {engine.stats}")
        
        # Check the actual parser registration
        print("\n7. Manual parser registration test...")
        try:
            parser_configs = [
                ('python', 'chuk_code_raptor.chunking.parsers.python', 'PythonParser'),
            ]
            
            for language, module_path, class_name in parser_configs:
                try:
                    print(f"   Testing {language}...")
                    module = __import__(module_path, fromlist=[class_name])
                    print(f"   ✅ Module imported: {module}")
                    parser_class = getattr(module, class_name)
                    print(f"   ✅ Class found: {parser_class}")
                    
                    parser = parser_class(config)
                    print(f"   ✅ Parser created: {parser}")
                    print(f"   ✅ Can parse check: {parser.can_parse('python', '.py')}")
                    
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        except Exception as e:
            print(f"❌ Manual registration test failed: {e}")
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_parser_loading()