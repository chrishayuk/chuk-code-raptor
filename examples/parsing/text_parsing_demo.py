#!/usr/bin/env python3
"""
Clean Text Parser Demo
=====================

A concise demonstration of TextParser capabilities with focused output.

Usage:
    python clean_text_parser_demo.py
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from chuk_code_raptor.chunking.parsers.text import TextParser
    from chuk_code_raptor.chunking.base import ParseContext
    from chuk_code_raptor.chunking.semantic_chunk import ContentType
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class CleanTextParserDemo:
    """Clean, focused demo of TextParser capabilities"""
    
    def __init__(self):
        self.config = self._create_config()
        self.parser = TextParser(self.config)
        
    def _create_config(self):
        class Config:
            min_chunk_size = 30
            max_chunk_size = 1500
            target_chunk_size = 500
        return Config()
    
    def _create_context(self, filename="demo.txt"):
        return ParseContext(
            file_path=filename,
            language="text", 
            content_type=ContentType.PLAINTEXT,
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            enable_semantic_analysis=True,
            enable_dependency_tracking=True,
            metadata={}
        )
    
    def _get_tags(self, chunk):
        """Extract tags from chunk safely"""
        if hasattr(chunk, 'semantic_tags') and chunk.semantic_tags:
            return [tag.name for tag in chunk.semantic_tags]
        elif hasattr(chunk, 'tag_names'):
            return chunk.tag_names
        return []
    
    def _analyze_and_display(self, content, content_type, description):
        """Analyze content and show clean results"""
        print(f"\n🔍 {description}")
        print(f"   Content: {len(content)} chars")
        
        # Detect text type
        if hasattr(self.parser, '_analyze_text_type'):
            text_type = self.parser._analyze_text_type(content)
            print(f"   Type: {text_type}")
        
        # Parse content
        context = self._create_context(f"{content_type.lower()}.txt")
        try:
            if hasattr(self.parser, '_extract_chunks_heuristically'):
                chunks = self.parser._extract_chunks_heuristically(content, context)
            else:
                chunks = []
            
            if chunks:
                print(f"   Chunks: {len(chunks)}")
                
                # Show chunk types
                chunk_types = {}
                interesting_tags = set()
                
                for chunk in chunks:
                    chunk_type = chunk.metadata.get('text_type', 'unknown')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    
                    # Collect interesting tags
                    tags = self._get_tags(chunk)
                    for tag in tags:
                        if any(keyword in tag for keyword in ['contains_', 'technical', 'complex', 'simple']):
                            interesting_tags.add(tag)
                
                if len(chunk_types) > 1:
                    types_str = ", ".join(f"{k}({v})" for k, v in chunk_types.items())
                    print(f"   Types: {types_str}")
                
                if interesting_tags:
                    tags_str = ", ".join(sorted(interesting_tags))
                    print(f"   Features: {tags_str}")
            else:
                print("   ⚠️  No chunks generated")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    def demo_content_types(self):
        """Demo different content types with clean output"""
        print("📚 TextParser Content Analysis Demo")
        print("=" * 45)
        
        # Plain text article
        article = """Getting Started with Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.

Key concepts include supervised learning, unsupervised learning, and reinforcement learning. Each approach serves different purposes and problem types.

Modern applications include recommendation systems, image recognition, natural language processing, and autonomous vehicles."""
        
        self._analyze_and_display(article, "article", "📄 Technical Article")
        
        # Application logs
        logs = """2024-07-25 09:00:01 INFO Application startup initiated
2024-07-25 09:00:02 INFO Database connection established
2024-07-25 09:00:03 INFO Cache system initialized  
2024-07-25 09:15:23 WARNING High memory usage: 87%
2024-07-25 09:15:24 INFO Memory cleanup completed
2024-07-25 09:30:45 ERROR Database timeout - retrying
2024-07-25 09:30:46 INFO Connection restored successfully
2024-07-25 10:00:00 INFO Hourly health check passed"""
        
        self._analyze_and_display(logs, "logs", "📋 Application Logs")
        
        # Configuration file
        config = """# Database Settings
database.host=prod-db-01.company.com
database.port=5432
database.name=production
database.username=app_user
database.timeout=30000

# Cache Configuration  
redis.host=cache-cluster.company.com
redis.port=6379
redis.max_connections=100

# Application Settings
app.debug=false
app.max_upload_size=10485760
app.session_timeout=3600"""
        
        self._analyze_and_display(config, "config", "⚙️  Configuration File")
        
        # Structured documentation
        docs = """API DOCUMENTATION

OVERVIEW
This API provides access to user management and data processing services.

AUTHENTICATION
All requests require a valid API key in the Authorization header.

ENDPOINTS

1. User Management
   GET /api/users - List all users
   POST /api/users - Create new user
   PUT /api/users/{id} - Update user

2. Data Processing
   POST /api/process - Submit processing job
   GET /api/jobs/{id} - Check job status

ERROR HANDLING
The API returns standard HTTP status codes and JSON error responses."""
        
        self._analyze_and_display(docs, "docs", "📖 API Documentation")
        
        # Mixed content with patterns
        mixed = """System Status Report

Contact: admin@company.com
Website: https://status.company.com
Config: /etc/app/production.conf

Current Status:
INFO: All systems operational
WARNING: High CPU usage (89%)  
ERROR: Failed backup to /backup/daily/
DEBUG: Memory usage: 6.2GB/8GB

Settings:
monitoring_enabled=true
alert_threshold=85
backup_retention=30"""
        
        self._analyze_and_display(mixed, "mixed", "🔀 Mixed Content")
    
    def demo_pattern_detection(self):
        """Demo pattern detection with clean output"""
        print(f"\n🎯 Pattern Detection Examples")
        print("-" * 30)
        
        patterns_text = """Contact Information:
Email: support@example.com, admin@company.org
URLs: https://www.example.com, http://api.service.com/v1
Files: /home/user/config.json, C:\\Program Files\\app\\settings.ini
Logs: 2024-07-25 10:00:00 INFO System started
Config: timeout=5000, debug_enabled=true"""
        
        context = self._create_context("patterns.txt")
        if hasattr(self.parser, '_extract_chunks_heuristically'):
            chunks = self.parser._extract_chunks_heuristically(patterns_text, context)
            
            detected_patterns = set()
            for chunk in chunks:
                tags = self._get_tags(chunk)
                for tag in tags:
                    if 'contains_' in tag or tag == 'technical_content':
                        detected_patterns.add(tag)
            
            if detected_patterns:
                print("Detected patterns:")
                for pattern in sorted(detected_patterns):
                    icon = {"contains_email": "📧", "contains_url": "🔗", 
                           "contains_filepath": "📁", "technical_content": "⚙️"}.get(pattern, "🔍")
                    print(f"  {icon} {pattern.replace('contains_', '').replace('_', ' ').title()}")
    
    def demo_text_complexity(self):
        """Demo text complexity analysis"""
        print(f"\n📊 Text Complexity Analysis")
        print("-" * 30)
        
        examples = [
            ("Simple text. Short sentences. Easy reading.", "Simple"),
            ("This extraordinarily complex sentence demonstrates sophisticated linguistic structures with multiple subordinate clauses, advanced vocabulary, and intricate grammatical relationships that challenge comprehension.", "Complex"),
            ("ERROR: Database connection failed. WARNING: High memory usage detected.", "Technical")
        ]
        
        for text, label in examples:
            try:
                from chuk_code_raptor.chunking.semantic_chunk import create_chunk_id, SemanticChunk
                from chuk_code_raptor.core.models import ChunkType
                
                chunk_id = create_chunk_id("test.txt", 1, ChunkType.TEXT_BLOCK, label.lower())
                chunk = SemanticChunk(
                    id=chunk_id,
                    file_path="test.txt",
                    content=text,
                    start_line=1,
                    end_line=1,
                    chunk_type=ChunkType.TEXT_BLOCK,
                    content_type=ContentType.PLAINTEXT,
                    metadata={}
                )
                
                if hasattr(self.parser, '_analyze_text_content'):
                    self.parser._analyze_text_content(chunk)
                    
                    # Get statistics
                    word_count = chunk.metadata.get('word_count', 0)
                    avg_word_len = chunk.metadata.get('avg_word_length', 0)
                    avg_sent_len = chunk.metadata.get('avg_sentence_length', 0)
                    
                    # Get classification
                    tags = self._get_tags(chunk)
                    classification = next((tag for tag in tags if tag in ['simple_text', 'complex_text', 'technical_content']), 'unknown')
                    
                    print(f"{label:>9}: {word_count} words, avg {avg_word_len:.1f} chars/word, {avg_sent_len:.1f} words/sentence → {classification}")
                    
            except Exception as e:
                print(f"{label:>9}: Analysis error - {e}")
    
    def run_demo(self):
        """Run the complete clean demo"""
        try:
            self.demo_content_types()
            self.demo_pattern_detection()
            self.demo_text_complexity()
            
            print(f"\n✅ Demo completed!")
            print("💡 TextParser successfully analyzed multiple content types with intelligent chunking")
            
        except KeyboardInterrupt:
            print("\n❌ Demo interrupted")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")


def main():
    """Run the clean demo"""
    demo = CleanTextParserDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()