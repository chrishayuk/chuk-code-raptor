#!/usr/bin/env python3
"""
XML Parser Demo Script
======================

Comprehensive demonstration of the XML parser capabilities including:
- Various XML document types (HTML, SVG, Config, Build files)
- Semantic element categorization
- Namespace handling
- Tree-sitter vs heuristic parsing comparison
- Performance analysis across different XML structures
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_sample_xml_data() -> Dict[str, str]:
    """Create various XML samples for testing different document types"""
    
    samples = {}
    
    # HTML Document
    samples['html_document'] = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample HTML Document</title>
    <link rel="stylesheet" href="styles.css">
    <script src="app.js"></script>
</head>
<body>
    <header id="main-header" class="site-header">
        <nav class="main-navigation">
            <ul class="nav-list">
                <li class="nav-item"><a href="#home" class="nav-link">Home</a></li>
                <li class="nav-item"><a href="#about" class="nav-link">About</a></li>
                <li class="nav-item"><a href="#contact" class="nav-link">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <main class="content">
        <section id="hero" class="hero-section">
            <h1 class="hero-title">Welcome to Our Site</h1>
            <p class="hero-description">This is a comprehensive demonstration of XML parsing capabilities.</p>
            <button class="btn btn-primary">Get Started</button>
        </section>
        
        <section id="features" class="features-section">
            <div class="container">
                <h2>Features</h2>
                <div class="feature-grid">
                    <article class="feature-card">
                        <h3>Fast Processing</h3>
                        <p>Lightning-fast XML parsing with tree-sitter technology.</p>
                    </article>
                    <article class="feature-card">
                        <h3>Semantic Analysis</h3>
                        <p>Intelligent categorization of XML elements and structures.</p>
                    </article>
                    <article class="feature-card">
                        <h3>Flexible Output</h3>
                        <p>Multiple output formats and customizable chunk sizes.</p>
                    </article>
                </div>
            </div>
        </section>
    </main>
    
    <footer class="site-footer">
        <p>&copy; 2025 XML Parser Demo. All rights reserved.</p>
    </footer>
</body>
</html>"""

    # SVG Document
    samples['svg_graphics'] = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="400" height="300" viewBox="0 0 400 300">
    
    <defs>
        <linearGradient id="backgroundGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#3b82f6;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#1d4ed8;stop-opacity:1" />
        </linearGradient>
        
        <filter id="dropShadow">
            <feGaussianBlur in="SourceAlpha" stdDeviation="3"/>
            <feOffset dx="2" dy="2" result="offset"/>
            <feComponentTransfer>
                <feFuncA type="linear" slope="0.5"/>
            </feComponentTransfer>
            <feMerge>
                <feMergeNode/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
    </defs>
    
    <!-- Background -->
    <rect width="100%" height="100%" fill="url(#backgroundGradient)"/>
    
    <!-- Main content group -->
    <g id="main-content" transform="translate(50, 50)">
        <circle cx="100" cy="75" r="50" fill="#ffffff" 
                filter="url(#dropShadow)" opacity="0.9"/>
        
        <rect x="25" y="25" width="150" height="100" 
              fill="#f59e0b" rx="10" ry="10" 
              filter="url(#dropShadow)"/>
        
        <text x="100" y="80" text-anchor="middle" 
              font-family="Arial, sans-serif" font-size="16" 
              fill="#1f2937" font-weight="bold">
            XML Parser
        </text>
        
        <text x="100" y="100" text-anchor="middle" 
              font-family="Arial, sans-serif" font-size="12" 
              fill="#374151">
            Semantic Analysis
        </text>
    </g>
    
    <!-- Decorative elements -->
    <g id="decorations">
        <path d="M 50 200 Q 200 150 350 200" 
              stroke="#ffffff" stroke-width="3" 
              fill="none" opacity="0.7"/>
        
        <circle cx="80" cy="220" r="8" fill="#ffffff" opacity="0.6"/>
        <circle cx="150" cy="180" r="6" fill="#ffffff" opacity="0.8"/>
        <circle cx="320" cy="220" r="10" fill="#ffffff" opacity="0.5"/>
    </g>
    
    <!-- Animation example -->
    <g id="animated-elements">
        <circle cx="350" cy="50" r="15" fill="#ef4444">
            <animate attributeName="r" values="15;25;15" dur="2s" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="1;0.5;1" dur="2s" repeatCount="indefinite"/>
        </circle>
    </g>
</svg>"""

    # Configuration XML
    samples['config_file'] = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns="http://example.com/config/v1" 
               xmlns:security="http://example.com/security/v1">
    
    <metadata>
        <version>2.1.0</version>
        <description>Application configuration for XML Parser Demo</description>
        <lastModified>2025-01-15T10:30:00Z</lastModified>
        <author>System Administrator</author>
    </metadata>
    
    <database>
        <connection name="primary">
            <host>localhost</host>
            <port>5432</port>
            <database>xml_parser_db</database>
            <username>app_user</username>
            <security:password encrypted="true">$2b$12$encrypted_password_hash</security:password>
            <connectionTimeout>30000</connectionTimeout>
            <maxConnections>50</maxConnections>
        </connection>
        
        <connection name="cache">
            <host>redis.internal</host>
            <port>6379</port>
            <database>0</database>
            <timeout>5000</timeout>
        </connection>
    </database>
    
    <application>
        <settings>
            <property name="debug" type="boolean">false</property>
            <property name="logLevel" type="string">INFO</property>
            <property name="maxFileSize" type="integer">10485760</property>
            <property name="enableCaching" type="boolean">true</property>
            <property name="cacheExpiry" type="duration">3600</property>
        </settings>
        
        <features>
            <feature name="xmlParsing" enabled="true">
                <setting name="maxDepth">100</setting>
                <setting name="chunkSize">2000</setting>
                <setting name="preserveWhitespace">false</setting>
            </feature>
            
            <feature name="analytics" enabled="false">
                <setting name="trackingId">GA-XXXXXXXXX</setting>
                <setting name="anonymizeIp">true</setting>
            </feature>
        </features>
    </application>
    
    <security:permissions>
        <security:role name="admin">
            <security:permission>read</security:permission>
            <security:permission>write</security:permission>
            <security:permission>delete</security:permission>
            <security:permission>configure</security:permission>
        </security:role>
        
        <security:role name="user">
            <security:permission>read</security:permission>
            <security:permission>write</security:permission>
        </security:role>
        
        <security:role name="guest">
            <security:permission>read</security:permission>
        </security:role>
    </security:permissions>
    
    <logging>
        <appender name="file" type="rolling">
            <file>logs/application.log</file>
            <maxFileSize>100MB</maxFileSize>
            <maxHistory>30</maxHistory>
            <pattern>%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </appender>
        
        <appender name="console" type="console">
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </appender>
        
        <logger name="com.example.xmlparser" level="DEBUG"/>
        <logger name="org.springframework" level="WARN"/>
        <root level="INFO">
            <appender-ref ref="file"/>
            <appender-ref ref="console"/>
        </root>
    </logging>
</configuration>"""

    # Build Configuration (Maven POM)
    samples['build_config'] = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
                             http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>xml-parser-demo</artifactId>
    <version>1.0.0-SNAPSHOT</version>
    <packaging>jar</packaging>
    
    <name>XML Parser Demo</name>
    <description>Demonstration project for XML parsing capabilities</description>
    <url>https://github.com/example/xml-parser-demo</url>
    
    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <junit.version>5.9.2</junit.version>
        <logback.version>1.4.5</logback.version>
        <jackson.version>2.15.2</jackson.version>
    </properties>
    
    <dependencies>
        <!-- Core dependencies -->
        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-api</artifactId>
            <version>2.0.6</version>
        </dependency>
        
        <dependency>
            <groupId>ch.qos.logback</groupId>
            <artifactId>logback-classic</artifactId>
            <version>${logback.version}</version>
        </dependency>
        
        <!-- JSON processing -->
        <dependency>
            <groupId>com.fasterxml.jackson.core</groupId>
            <artifactId>jackson-core</artifactId>
            <version>${jackson.version}</version>
        </dependency>
        
        <dependency>
            <groupId>com.fasterxml.jackson.core</groupId>
            <artifactId>jackson-databind</artifactId>
            <version>${jackson.version}</version>
        </dependency>
        
        <!-- XML processing -->
        <dependency>
            <groupId>org.dom4j</groupId>
            <artifactId>dom4j</artifactId>
            <version>2.1.4</version>
        </dependency>
        
        <!-- Test dependencies -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter-engine</artifactId>
            <version>${junit.version}</version>
            <scope>test</scope>
        </dependency>
        
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter-params</artifactId>
            <version>${junit.version}</version>
            <scope>test</scope>
        </dependency>
        
        <dependency>
            <groupId>org.mockito</groupId>
            <artifactId>mockito-core</artifactId>
            <version>5.1.1</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>17</source>
                    <target>17</target>
                </configuration>
            </plugin>
            
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.0.0-M9</version>
                <configuration>
                    <includes>
                        <include>**/*Test.java</include>
                        <include>**/*Tests.java</include>
                    </includes>
                </configuration>
            </plugin>
            
            <plugin>
                <groupId>org.jacoco</groupId>
                <artifactId>jacoco-maven-plugin</artifactId>
                <version>0.8.8</version>
                <executions>
                    <execution>
                        <goals>
                            <goal>prepare-agent</goal>
                        </goals>
                    </execution>
                    <execution>
                        <id>report</id>
                        <phase>test</phase>
                        <goals>
                            <goal>report</goal>
                        </goals>
                    </execution>
                </executions>
            </plugin>
            
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
                <version>3.0.4</version>
                <configuration>
                    <mainClass>com.example.xmlparser.Application</mainClass>
                </configuration>
            </plugin>
        </plugins>
    </build>
    
    <profiles>
        <profile>
            <id>development</id>
            <activation>
                <activeByDefault>true</activeByDefault>
            </activation>
            <properties>
                <spring.profiles.active>dev</spring.profiles.active>
            </properties>
        </profile>
        
        <profile>
            <id>production</id>
            <properties>
                <spring.profiles.active>prod</spring.profiles.active>
            </properties>
            <build>
                <plugins>
                    <plugin>
                        <groupId>org.apache.maven.plugins</groupId>
                        <artifactId>maven-war-plugin</artifactId>
                        <version>3.2.3</version>
                        <configuration>
                            <webXml>false</webXml>
                        </configuration>
                    </plugin>
                </plugins>
            </build>
        </profile>
    </profiles>
</project>"""

    # Data XML with collections
    samples['data_collection'] = """<?xml version="1.0" encoding="UTF-8"?>
<library xmlns="http://example.com/library/v1">
    <metadata>
        <name>Central Library System</name>
        <location>Downtown Branch</location>
        <established>1995-03-15</established>
        <totalBooks>15420</totalBooks>
        <lastUpdated>2025-01-15T14:30:00Z</lastUpdated>
    </metadata>
    
    <categories>
        <category id="fiction">
            <name>Fiction</name>
            <description>Novels, short stories, and fictional works</description>
            <color>#3b82f6</color>
        </category>
        <category id="nonfiction">
            <name>Non-Fiction</name>
            <description>Biographies, history, science, and factual works</description>
            <color>#ef4444</color>
        </category>
        <category id="reference">
            <name>Reference</name>
            <description>Dictionaries, encyclopedias, and reference materials</description>
            <color>#10b981</color>
        </category>
    </categories>
    
    <books>
        <book id="B001" category="fiction" available="true">
            <title>The Digital Frontier</title>
            <author>
                <firstName>Sarah</firstName>
                <lastName>Mitchell</lastName>
                <nationality>American</nationality>
            </author>
            <publication>
                <publisher>TechPress Publishing</publisher>
                <year>2023</year>
                <isbn>978-0-123456-78-9</isbn>
                <pages>342</pages>
            </publication>
            <description>
                <![CDATA[
                A thrilling exploration of artificial intelligence and its impact on society.
                Set in the near future, this novel examines the relationship between humans
                and advanced AI systems in a world where technology has transformed every
                aspect of daily life.
                ]]>
            </description>
            <tags>
                <tag>science fiction</tag>
                <tag>artificial intelligence</tag>
                <tag>technology</tag>
                <tag>future society</tag>
            </tags>
            <rating>4.7</rating>
            <reviews>156</reviews>
        </book>
        
        <book id="B002" category="nonfiction" available="true">
            <title>Climate Change: A Comprehensive Guide</title>
            <author>
                <firstName>Dr. Maria</firstName>
                <lastName>Rodriguez</lastName>
                <nationality>Spanish</nationality>
                <credentials>PhD Environmental Science</credentials>
            </author>
            <publication>
                <publisher>Environmental Studies Press</publisher>
                <year>2024</year>
                <isbn>978-0-987654-32-1</isbn>
                <pages>456</pages>
                <edition>3rd</edition>
            </publication>
            <description>
                <![CDATA[
                An authoritative examination of climate change science, impacts, and solutions.
                This updated edition includes the latest research on global warming trends,
                renewable energy technologies, and international climate policies.
                ]]>
            </description>
            <tags>
                <tag>climate change</tag>
                <tag>environmental science</tag>
                <tag>sustainability</tag>
                <tag>global warming</tag>
            </tags>
            <rating>4.9</rating>
            <reviews>89</reviews>
        </book>
        
        <book id="B003" category="reference" available="false">
            <title>Oxford Dictionary of Computer Science</title>
            <author>
                <organization>Oxford University Press</organization>
                <editors>
                    <editor>Prof. John Smith</editor>
                    <editor>Dr. Emily Johnson</editor>
                </editors>
            </author>
            <publication>
                <publisher>Oxford University Press</publisher>
                <year>2022</year>
                <isbn>978-0-199876-54-3</isbn>
                <pages>892</pages>
                <edition>8th</edition>
            </publication>
            <description>
                <![CDATA[
                Comprehensive reference work covering all aspects of computer science,
                from algorithms and data structures to artificial intelligence and
                quantum computing. Updated with the latest terminology and concepts.
                ]]>
            </description>
            <tags>
                <tag>computer science</tag>
                <tag>dictionary</tag>
                <tag>reference</tag>
                <tag>technology</tag>
            </tags>
            <checkoutInfo>
                <borrower>John Doe</borrower>
                <dueDate>2025-02-01</dueDate>
                <renewable>true</renewable>
            </checkoutInfo>
        </book>
    </books>
    
    <statistics>
        <borrowingStats>
            <totalCheckouts>1247</totalCheckouts>
            <activeLoans>89</activeLoans>
            <overdueItems>12</overdueItems>
            <reservations>34</reservations>
        </borrowingStats>
        
        <categoryStats>
            <stat category="fiction" count="8934" percentage="57.9"/>
            <stat category="nonfiction" count="4521" percentage="29.3"/>
            <stat category="reference" count="1965" percentage="12.8"/>
        </categoryStats>
    </statistics>
</library>"""

    return samples

def analyze_xml_chunk(chunk) -> Dict[str, Any]:
    """Analyze the structure and properties of an XML chunk"""
    analysis = {
        'id': chunk.id,
        'type': chunk.chunk_type.value,
        'size_chars': len(chunk.content),
        'size_lines': chunk.line_count,
        'importance': chunk.importance_score,
        'tags': [tag.name for tag in chunk.semantic_tags] if hasattr(chunk, 'semantic_tags') and chunk.semantic_tags else [],
        'metadata': chunk.metadata or {},
        'dependencies': chunk.dependencies[:3] if chunk.dependencies else [],
        'preview': chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
    }
    
    # XML-specific analysis
    element_name = chunk.metadata.get('element_name') if chunk.metadata else None
    semantic_category = chunk.metadata.get('semantic_category') if chunk.metadata else None
    xml_type = chunk.metadata.get('xml_type') if chunk.metadata else None
    namespaces = chunk.metadata.get('namespaces', {}) if chunk.metadata else {}
    
    analysis['xml_analysis'] = {
        'element_name': element_name,
        'semantic_category': semantic_category,
        'xml_type': xml_type,
        'has_namespaces': bool(namespaces),
        'namespace_count': len(namespaces),
        'document_type': detect_xml_document_type(analysis['tags'], chunk.content),
        'structural_role': detect_structural_role(analysis['tags'], element_name)
    }
    
    return analysis

def detect_xml_document_type(tags: List[str], content: str) -> str:
    """Detect the type of XML document from tags and content"""
    content_lower = content.lower()
    
    if any('html' in tag for tag in tags) or '<html' in content_lower:
        return 'HTML Document'
    elif any('svg' in tag for tag in tags) or '<svg' in content_lower:
        return 'SVG Graphics'
    elif 'config' in content_lower or 'configuration' in content_lower:
        return 'Configuration File'
    elif 'project' in content_lower or 'pom' in content_lower:
        return 'Build Configuration'
    elif any('data' in tag for tag in tags):
        return 'Data Document'
    else:
        return 'Generic XML'

def detect_structural_role(tags: List[str], element_name: str) -> str:
    """Detect the structural role of an XML element"""
    if not element_name:
        return 'Unknown'
    
    element_lower = element_name.lower()
    
    if any('structural' in tag for tag in tags):
        return 'Structural Element'
    elif any('content' in tag for tag in tags):
        return 'Content Element'
    elif any('metadata' in tag for tag in tags):
        return 'Metadata Element'
    elif element_lower in ['head', 'header', 'footer', 'nav', 'main']:
        return 'Layout Element'
    elif element_lower in ['div', 'section', 'article', 'aside']:
        return 'Container Element'
    elif element_lower in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'text']:
        return 'Text Element'
    elif element_lower in ['title', 'meta', 'link', 'script']:
        return 'Document Metadata'
    else:
        return 'Content Element'

def generate_xml_summary(chunks) -> Dict[str, Any]:
    """Generate summary of XML parsing results"""
    summary = {
        'total_chunks': len(chunks),
        'chunk_types': defaultdict(int),
        'xml_document_types': defaultdict(int),
        'semantic_categories': defaultdict(int),
        'structural_roles': defaultdict(int),
        'namespace_analysis': {
            'chunks_with_namespaces': 0,
            'total_namespaces': 0,
            'unique_namespaces': set()
        },
        'element_analysis': {
            'total_elements': 0,
            'unique_elements': set(),
            'most_common_elements': defaultdict(int)
        },
        'content_analysis': {
            'avg_chunk_size': 0,
            'total_content_size': 0,
            'size_distribution': defaultdict(int)
        }
    }
    
    total_content_size = 0
    
    for chunk in chunks:
        # Type distribution
        summary['chunk_types'][chunk.chunk_type.value] += 1
        
        # Analyze chunk
        analysis = analyze_xml_chunk(chunk)
        xml_analysis = analysis['xml_analysis']
        
        # Document type distribution
        doc_type = xml_analysis['document_type']
        summary['xml_document_types'][doc_type] += 1
        
        # Semantic category distribution
        semantic_category = xml_analysis['semantic_category']
        if semantic_category:
            summary['semantic_categories'][semantic_category] += 1
        
        # Structural role distribution
        structural_role = xml_analysis['structural_role']
        summary['structural_roles'][structural_role] += 1
        
        # Namespace analysis
        if xml_analysis['has_namespaces']:
            summary['namespace_analysis']['chunks_with_namespaces'] += 1
            summary['namespace_analysis']['total_namespaces'] += xml_analysis['namespace_count']
            
            # Extract namespace URIs from metadata
            namespaces = chunk.metadata.get('namespaces', {}) if chunk.metadata else {}
            for uri in namespaces.values():
                summary['namespace_analysis']['unique_namespaces'].add(uri)
        
        # Element analysis
        element_name = xml_analysis['element_name']
        if element_name:
            summary['element_analysis']['total_elements'] += 1
            summary['element_analysis']['unique_elements'].add(element_name)
            summary['element_analysis']['most_common_elements'][element_name] += 1
        
        # Content size analysis
        chunk_size = analysis['size_chars']
        total_content_size += chunk_size
        
        if chunk_size < 100:
            summary['content_analysis']['size_distribution']['small'] += 1
        elif chunk_size < 500:
            summary['content_analysis']['size_distribution']['medium'] += 1
        elif chunk_size < 2000:
            summary['content_analysis']['size_distribution']['large'] += 1
        else:
            summary['content_analysis']['size_distribution']['xlarge'] += 1
    
    # Calculate averages
    if chunks:
        summary['content_analysis']['avg_chunk_size'] = total_content_size / len(chunks)
        summary['content_analysis']['total_content_size'] = total_content_size
    
    # Convert sets to lists for JSON serialization
    summary['namespace_analysis']['unique_namespaces'] = list(summary['namespace_analysis']['unique_namespaces'])
    summary['element_analysis']['unique_elements'] = list(summary['element_analysis']['unique_elements'])
    
    return summary

def test_xml_parsing(content: str, sample_name: str) -> Dict[str, Any]:
    """Test XML parsing on sample content"""
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    # Configure for XML parsing
    config = ChunkingConfig(
        target_chunk_size=800,  # Good size for XML elements
        min_chunk_size=50,
        preserve_atomic_nodes=True,
        enable_dependency_tracking=True
    )
    
    engine = ChunkingEngine(config)
    
    # Check if XML is supported
    if not engine.can_chunk_language('xml'):
        print(f"⚠️  XML parser not available. Supported languages: {engine.get_supported_languages()}")
        return {
            'chunks': [],
            'summary': generate_xml_summary([]),
            'sample_info': {
                'size_chars': len(content),
                'line_count': content.count('\n') + 1,
                'error': 'XML parser not available'
            }
        }
    
    # Parse XML
    start_time = time.time()
    chunks = engine.chunk_content(content, 'xml', f'{sample_name}.xml')
    parse_time = time.time() - start_time
    
    # Generate analysis
    summary = generate_xml_summary(chunks)
    
    return {
        'chunks': chunks,
        'summary': summary,
        'sample_info': {
            'size_chars': len(content),
            'line_count': content.count('\n') + 1,
            'parse_time_ms': parse_time * 1000,
            'elements_detected': len(summary['element_analysis']['unique_elements'])
        }
    }

def demonstrate_parser_comparison():
    """Demonstrate tree-sitter vs heuristic parsing comparison"""
    print(f"\n🔍 PARSER COMPARISON DEMO")
    print("=" * 60)
    
    # Create a complex XML sample that might challenge tree-sitter
    complex_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <!-- Complex XML with various challenging features -->
    <root xmlns:ns1="http://example.com/ns1" xmlns:ns2="http://example.com/ns2">
        <ns1:metadata>
            <title><![CDATA[Complex & "Challenging" XML <Document>]]></title>
            <description>This XML contains various challenging elements</description>
        </ns1:metadata>
        
        <data-section>
            <item id="1" type="primary">
                <content>
                    <p>Some content with <em>emphasis</em> and special chars: &amp; &lt; &gt;</p>
                </content>
            </item>
            <!-- Self-closing and mixed content -->
            <item id="2" type="secondary"/>
            <item id="3">Mixed content <span>with inline elements</span> and text</item>
        </data-section>
        
        <ns2:processing-instructions>
            <?xml-stylesheet type="text/xsl" href="transform.xsl"?>
            <ns2:custom-element attr1="value1" attr2='value2'>
                <!-- Nested comment -->
                <![CDATA[Raw data with <tags> and & symbols]]>
            </ns2:custom-element>
        </ns2:processing-instructions>
    </root>"""
    
    try:
        result = test_xml_parsing(complex_xml, "complex_comparison")
        
        if 'error' not in result['sample_info']:
            chunks = result['chunks']
            summary = result['summary']
            
            print(f"✅ Successfully parsed complex XML:")
            print(f"   📊 Total chunks: {summary['total_chunks']}")
            print(f"   🏷️  Unique elements: {len(summary['element_analysis']['unique_elements'])}")
            print(f"   🔤 Namespaces found: {len(summary['namespace_analysis']['unique_namespaces'])}")
            print(f"   ⏱️  Parse time: {result['sample_info']['parse_time_ms']:.1f}ms")
            
            # Show parsing method used
            parsing_methods = set()
            for chunk in chunks:
                if chunk.metadata and 'extraction_method' in chunk.metadata:
                    parsing_methods.add(chunk.metadata['extraction_method'])
            
            if parsing_methods:
                print(f"   🔧 Parsing methods used: {', '.join(parsing_methods)}")
            
            # Show some example chunks
            print(f"\n📝 Example chunks:")
            for i, chunk in enumerate(chunks[:3]):
                analysis = analyze_xml_chunk(chunk)
                element = analysis['xml_analysis']['element_name'] or 'unknown'
                role = analysis['xml_analysis']['structural_role']
                print(f"   {i+1}. Element '{element}' ({role}) - {analysis['size_chars']} chars")
        
        else:
            print(f"❌ Parsing failed: {result['sample_info']['error']}")
    
    except Exception as e:
        print(f"❌ Error in comparison demo: {e}")

def demonstrate_namespace_handling():
    """Demonstrate XML namespace handling capabilities"""
    print(f"\n🏷️ NAMESPACE HANDLING DEMO")
    print("=" * 60)
    
    namespace_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"
                   xmlns:web="http://www.example.com/webservice/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                   xmlns:xsd="http://www.w3.org/2001/XMLSchema">
        
        <soap:Header>
            <web:Authentication>
                <web:Username>demo_user</web:Username>
                <web:Password>demo_pass</web:Password>
                <web:Token xsi:type="xsd:string">abc123def456</web:Token>
            </web:Authentication>
        </soap:Header>
        
        <soap:Body>
            <web:GetUserInfo>
                <web:UserId xsi:type="xsd:int">12345</web:UserId>
                <web:IncludeDetails xsi:type="xsd:boolean">true</web:IncludeDetails>
            </web:GetUserInfo>
        </soap:Body>
    </soap:Envelope>"""
    
    try:
        result = test_xml_parsing(namespace_xml, "namespace_demo")
        
        if 'error' not in result['sample_info']:
            summary = result['summary']
            namespace_info = summary['namespace_analysis']
            
            print(f"✅ Namespace analysis complete:")
            print(f"   📦 Chunks with namespaces: {namespace_info['chunks_with_namespaces']}")
            print(f"   🔢 Total namespace declarations: {namespace_info['total_namespaces']}")
            print(f"   🌐 Unique namespace URIs:")
            
            for i, uri in enumerate(namespace_info['unique_namespaces'], 1):
                print(f"      {i}. {uri}")
            
            # Show namespace usage in chunks
            chunks_with_ns = [chunk for chunk in result['chunks'] 
                             if chunk.metadata and chunk.metadata.get('namespaces')]
            
            if chunks_with_ns:
                print(f"\n🔍 Namespace usage examples:")
                for chunk in chunks_with_ns[:2]:
                    element = chunk.metadata.get('element_name', 'unknown')
                    namespaces = chunk.metadata.get('namespaces', {})
                    print(f"   Element '{element}' uses namespaces:")
                    for prefix, uri in namespaces.items():
                        print(f"     {prefix or 'default'}: {uri}")
        
        else:
            print(f"❌ Namespace demo failed: {result['sample_info']['error']}")
    
    except Exception as e:
        print(f"❌ Error in namespace demo: {e}")

def print_detailed_xml_analysis(samples: Dict[str, str], all_results: Dict[str, Any]):
    """Print comprehensive analysis of XML parsing results"""
    print("\n" + "="*80)
    print("📄 COMPREHENSIVE XML PARSING ANALYSIS")
    print("="*80)
    
    for sample_name, result in all_results.items():
        if 'error' in result['sample_info']:
            continue
            
        chunks = result['chunks']
        summary = result['summary']
        
        print(f"\n📄 SAMPLE: {sample_name.upper().replace('_', ' ')}")
        print("-" * 60)
        
        # Sample info
        sample_info = result['sample_info']
        print(f"📊 Document size: {sample_info['size_chars']} characters, {sample_info['line_count']} lines")
        print(f"🧩 Chunks created: {len(chunks)}")
        print(f"⏱️ Parse time: {sample_info['parse_time_ms']:.1f}ms")
        print(f"🏷️ Elements detected: {sample_info['elements_detected']}")
        
        # Document type analysis
        print(f"\n📋 Document Type Analysis:")
        for doc_type, count in summary['xml_document_types'].items():
            print(f"   {doc_type}: {count} chunks")
        
        # Semantic structure
        print(f"\n🏗️ Semantic Structure:")
        for category, count in summary['semantic_categories'].items():
            print(f"   {category}: {count} chunks")
        
        # Element analysis
        element_analysis = summary['element_analysis']
        print(f"\n🏷️ Element Analysis:")
        print(f"   Total elements: {element_analysis['total_elements']}")
        print(f"   Unique elements: {len(element_analysis['unique_elements'])}")
        
        # Show most common elements
        most_common = sorted(element_analysis['most_common_elements'].items(), 
                           key=lambda x: x[1], reverse=True)
        if most_common:
            print(f"   Most common elements:")
            for element, count in most_common[:5]:
                print(f"     {element}: {count}")
        
        # Namespace analysis
        namespace_info = summary['namespace_analysis']
        if namespace_info['chunks_with_namespaces'] > 0:
            print(f"\n🌐 Namespace Usage:")
            print(f"   Chunks with namespaces: {namespace_info['chunks_with_namespaces']}")
            print(f"   Total namespace declarations: {namespace_info['total_namespaces']}")
            print(f"   Unique namespaces: {len(namespace_info['unique_namespaces'])}")
        
        # Content distribution
        content_analysis = summary['content_analysis']
        print(f"\n📏 Content Distribution:")
        print(f"   Average chunk size: {content_analysis['avg_chunk_size']:.0f} characters")
        size_dist = content_analysis['size_distribution']
        for size_category, count in size_dist.items():
            print(f"   {size_category.title()} chunks: {count}")

def main():
    """Main demo function"""
    print("📄 XML PARSING DEMO")
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
        
        if 'xml' not in supported_languages:
            print("⚠️  XML parser not available. This may be due to missing tree-sitter-xml.")
            print("   Install with: pip install tree-sitter-xml")
            print("   Demo will show what the analysis would look like.")
        
        # Create sample XML data
        print(f"\n📝 Creating sample XML data...")
        samples = create_sample_xml_data()
        print(f"✅ Created {len(samples)} XML samples:")
        for name, content in samples.items():
            lines = content.count('\n') + 1
            chars = len(content)
            print(f"   • {name}: {chars:,} chars, {lines} lines")
        
        # Test each sample
        all_results = {}
        
        print(f"\n🚀 Testing XML parsing...")
        for sample_name, sample_content in samples.items():
            print(f"   🧪 Processing {sample_name}...")
            
            try:
                result = test_xml_parsing(sample_content, sample_name)
                all_results[sample_name] = result
                
                if 'error' in result['sample_info']:
                    print(f"      ⚠️  {result['sample_info']['error']}")
                else:
                    chunks_count = len(result['chunks'])
                    parse_time = result['sample_info']['parse_time_ms']
                    elements = result['sample_info']['elements_detected']
                    print(f"      ✅ {chunks_count} chunks, {elements} elements, {parse_time:.1f}ms")
                    
            except Exception as e:
                print(f"      ❌ Error processing {sample_name}: {e}")
        
        # Print comprehensive analysis
        print_detailed_xml_analysis(samples, all_results)
        
        # Demonstrate special features
        demonstrate_parser_comparison()
        demonstrate_namespace_handling()
        
        print(f"\n🎉 XML parsing demo completed successfully!")
        print(f"💡 The XML parser provides sophisticated analysis of various XML document types")
        print(f"   with semantic understanding, namespace support, and robust error handling.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package and tree-sitter-xml are installed")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()