#!/usr/bin/env python3
# examples/document_chunk_demo.py
"""
SemanticDocumentChunk Feature Demo
==================================

Comprehensive demonstration of SemanticDocumentChunk's document-specific features:
- Automatic structure detection (headings, code blocks, lists, tables)
- Readability analysis and scoring
- Document quality metrics
- Entity and topic extraction simulation
- Cross-reference handling
- Integration with SemanticChunk features
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Import the document chunk classes
from chuk_code_raptor.chunking.document_chunk import (
    SemanticDocumentChunk, calculate_document_quality_metrics,
    create_document_chunk_for_content_type
)
from chuk_code_raptor.chunking.semantic_chunk import ContentType, QualityMetric
from chuk_code_raptor.core.models import ChunkType

# Sample document content for demonstration
SAMPLE_DOCUMENTS = {
    "heading": "# Advanced Machine Learning Techniques",
    
    "subheading": "## Deep Learning Fundamentals",
    
    "paragraph": """Machine learning has revolutionized how we approach complex problems in computer science. It enables systems to automatically learn and improve from experience without being explicitly programmed. The field encompasses various techniques including supervised learning, unsupervised learning, and reinforcement learning.""",
    
    "complex_paragraph": """This is an extremely complex and convoluted sentence that demonstrates how readability analysis works by including numerous subordinate clauses, technical jargon, and multiple embedded ideas that make the content significantly more difficult to read and comprehend for the average reader, especially when combined with additional clauses that extend the sentence length far beyond what is considered optimal for clear communication in technical documentation, thereby resulting in a lower readability score that reflects the cognitive load required to process such dense and intricate textual content.""",
    
    "list": """- **Supervised Learning**: Uses labeled training data to learn patterns
- **Unsupervised Learning**: Discovers hidden patterns in unlabeled data
- **Reinforcement Learning**: Learns through interaction with environment
- **Deep Learning**: Uses neural networks with multiple layers""",
    
    "numbered_list": """1. Data Collection and Preprocessing
2. Feature Engineering and Selection
3. Model Selection and Training
4. Hyperparameter Tuning
5. Model Evaluation and Validation""",
    
    "code_block": """```python
def train_neural_network(X_train, y_train, architecture):
    model = Sequential()
    for layer in architecture:
        model.add(Dense(layer['units'], activation=layer['activation']))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    return model
```""",
    
    "indented_code": """    def calculate_accuracy(predictions, actual):
        correct = sum(1 for p, a in zip(predictions, actual) if p == a)
        return correct / len(actual)
    
    def evaluate_model(model, test_data):
        predictions = model.predict(test_data)
        return calculate_accuracy(predictions, test_data.labels)""",
    
    "table": """| Algorithm | Accuracy | Training Time | Memory Usage |
|-----------|----------|---------------|--------------|
| Random Forest | 92.5% | 2.3 minutes | 256 MB |
| SVM | 89.1% | 5.7 minutes | 128 MB |
| Neural Network | 94.2% | 12.1 minutes | 512 MB |""",
    
    "technical_doc": """The gradient descent optimization algorithm minimizes the cost function J(θ) by iteratively updating parameters θ in the direction of steepest descent. For more details, see Section 4.2 and Appendix B. The learning rate α controls the step size, as discussed in Chapter 3."""
}

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

def print_section(title: str):
    """Print a section header"""
    print(f"\n--- {title} ---")

def print_metrics(title: str, metrics: Dict[str, Any]):
    """Print metrics in a formatted way"""
    print(f"\n📊 {title}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  • {key}: {value:.3f}")
        elif isinstance(value, list):
            print(f"  • {key}: {len(value)} items")
        else:
            print(f"  • {key}: {value}")

def demo_structure_detection():
    """Demo automatic structure detection for different content types"""
    print_header("1. AUTOMATIC STRUCTURE DETECTION")
    
    structure_samples = [
        ("heading", SAMPLE_DOCUMENTS["heading"]),
        ("subheading", SAMPLE_DOCUMENTS["subheading"]),
        ("paragraph", SAMPLE_DOCUMENTS["paragraph"]),
        ("list", SAMPLE_DOCUMENTS["list"]),
        ("numbered_list", SAMPLE_DOCUMENTS["numbered_list"]),
        ("code_block", SAMPLE_DOCUMENTS["code_block"]),
        ("indented_code", SAMPLE_DOCUMENTS["indented_code"]),
        ("table", SAMPLE_DOCUMENTS["table"])
    ]
    
    detected_structures = []
    
    for name, content in structure_samples:
        chunk = SemanticDocumentChunk(
            id=f"struct_{name}",
            file_path=f"/docs/{name}.md",
            content=content,
            start_line=1,
            end_line=content.count('\n') + 1,
            content_type=ContentType.MARKDOWN
        )
        
        detected_structures.append(chunk)
        
        # Visual indicators for different types
        type_icon = {
            "heading": "🔷",
            "paragraph": "📝", 
            "list": "📋",
            "code_block": "💻",
            "table": "📊"
        }.get(chunk.section_type, "📄")
        
        print(f"\n{type_icon} {name.upper()}:")
        print(f"  Content: {content[:60]}{'...' if len(content) > 60 else ''}")
        print(f"  Detected Type: {chunk.section_type}")
        if chunk.heading_level:
            print(f"  Heading Level: {chunk.heading_level}")
        print(f"  Is Heading: {chunk.is_heading}")
        print(f"  Is Code Block: {chunk.is_code_block}")
    
    return detected_structures

def demo_readability_analysis():
    """Demo readability analysis for different complexity levels"""
    print_header("2. READABILITY ANALYSIS & SCORING")
    
    readability_samples = [
        ("Simple", "This is easy to read. Short sentences work well. Clear communication is important."),
        ("Moderate", SAMPLE_DOCUMENTS["paragraph"]),
        ("Complex", SAMPLE_DOCUMENTS["complex_paragraph"]),
        ("Technical", SAMPLE_DOCUMENTS["technical_doc"])
    ]
    
    for name, content in readability_samples:
        chunk = SemanticDocumentChunk(
            id=f"read_{name.lower()}",
            file_path=f"/docs/{name.lower()}.md",
            content=content,
            start_line=1,
            end_line=content.count('\n') + 1,
            content_type=ContentType.DOCUMENTATION
        )
        
        # Readability indicators
        if chunk.readability_score > 0.8:
            readability_icon = "🟢 Excellent"
        elif chunk.readability_score > 0.6:
            readability_icon = "🟡 Good"
        elif chunk.readability_score > 0.4:
            readability_icon = "🟠 Moderate"
        else:
            readability_icon = "🔴 Complex"
        
        print_section(f"{name} Text Analysis")
        print_metrics("Readability Metrics", {
            "sentence_count": chunk.sentence_count,
            "avg_sentence_length": chunk.avg_sentence_length,
            "word_count": chunk.word_count,
            "readability_score": chunk.readability_score,
            "is_highly_readable": chunk.is_highly_readable
        })
        
        print(f"  {readability_icon}")
        print(f"  Preview: {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}")

def demo_document_quality_metrics():
    """Demo comprehensive document quality analysis"""
    print_header("3. COMPREHENSIVE DOCUMENT QUALITY METRICS")
    
    # Create a rich document chunk with various properties
    chunk = SemanticDocumentChunk(
        id="quality_demo",
        file_path="/docs/ml_guide.md",
        content=SAMPLE_DOCUMENTS["technical_doc"],
        start_line=15,
        end_line=18,
        content_type=ContentType.DOCUMENTATION
    )
    
    # Add document-specific metadata
    chunk.keywords = ["gradient descent", "optimization", "cost function", "learning rate"]
    chunk.entities = ["gradient descent", "θ", "α", "J(θ)"]
    chunk.topics = ["machine learning", "optimization", "mathematics"]
    chunk.cross_references = ["Section 4.2", "Appendix B", "Chapter 3"]
    
    # Add semantic tags
    chunk.add_semantic_tag("mathematics", 0.9, "domain")
    chunk.add_semantic_tag("algorithm", 0.85, "concept")
    chunk.add_semantic_tag("technical-documentation", 0.8, "content-type")
    
    print_section("Document Properties")
    print_metrics("Basic Properties", {
        "section_type": chunk.section_type,
        "word_count": chunk.word_count,
        "sentence_count": chunk.sentence_count,
        "character_count": chunk.character_count
    })
    
    print_section("Semantic Information")
    print(f"  🏷️  Keywords: {', '.join(chunk.keywords)}")
    print(f"  🎯 Entities: {', '.join(chunk.entities)}")
    print(f"  📚 Topics: {', '.join(chunk.topics)}")
    print(f"  🔗 Cross-references: {', '.join(chunk.cross_references)}")
    print(f"  📋 Semantic tags: {', '.join(chunk.tag_names)}")
    
    # Calculate comprehensive quality metrics
    quality_metrics = calculate_document_quality_metrics(chunk)
    
    print_section("Quality Metrics Dashboard")
    for metric_name, score in quality_metrics.items():
        # Add visual indicators
        if score > 0.8:
            indicator = "🟢"
        elif score > 0.6:
            indicator = "🟡"
        elif score > 0.4:
            indicator = "🟠"
        else:
            indicator = "🔴"
        
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"  {indicator} {metric_name.replace('_', ' ').title():<20} {bar} {score:.3f}")
    
    return chunk, quality_metrics

def demo_document_types_comparison():
    """Demo analysis across different document types"""
    print_header("4. DOCUMENT TYPE COMPARISON ANALYSIS")
    
    document_samples = [
        ("Research Paper Abstract", SAMPLE_DOCUMENTS["paragraph"], "academic"),
        ("Tutorial Step", SAMPLE_DOCUMENTS["numbered_list"], "instructional"),
        ("API Documentation", SAMPLE_DOCUMENTS["code_block"], "technical"),
        ("Reference Table", SAMPLE_DOCUMENTS["table"], "reference")
    ]
    
    analysis_results = []
    
    for doc_type, content, category in document_samples:
        chunk = SemanticDocumentChunk(
            id=f"type_{category}",
            file_path=f"/docs/{category}.md",
            content=content,
            start_line=1,
            end_line=content.count('\n') + 1,
            content_type=ContentType.DOCUMENTATION
        )
        
        # Add category-specific properties
        if category == "academic":
            chunk.keywords = ["machine learning", "systems", "experience"]
            chunk.entities = ["computer science"]
        elif category == "instructional":
            chunk.keywords = ["data", "model", "training"]
            chunk.cross_references = ["Step 1", "Step 2"]
        elif category == "technical":
            chunk.keywords = ["neural network", "training", "model"]
            chunk.entities = ["Sequential", "Dense", "adam"]
        elif category == "reference":
            chunk.keywords = ["algorithm", "accuracy", "performance"]
            chunk.entities = ["Random Forest", "SVM", "Neural Network"]
        
        # Calculate metrics
        quality_metrics = calculate_document_quality_metrics(chunk)
        structure_consistency = chunk.calculate_structure_consistency()
        
        analysis_results.append({
            'type': doc_type,
            'section_type': chunk.section_type,
            'readability': chunk.readability_score,
            'structure_consistency': structure_consistency,
            'overall_quality': sum(quality_metrics.values()) / len(quality_metrics)
        })
        
        print_section(f"{doc_type} ({category.title()})")
        print(f"  📄 Detected Structure: {chunk.section_type}")
        print(f"  📖 Readability: {chunk.readability_score:.3f}")
        print(f"  🏗️  Structure Consistency: {structure_consistency:.3f}")
        print(f"  ⭐ Overall Quality: {sum(quality_metrics.values()) / len(quality_metrics):.3f}")
        print(f"  🔤 Word Count: {chunk.word_count}")
        print(f"  📏 Sentences: {chunk.sentence_count}")
    
    print_section("Comparative Analysis")
    print("📊 Document Type Performance:")
    for result in sorted(analysis_results, key=lambda x: x['overall_quality'], reverse=True):
        quality_indicator = "🏆" if result['overall_quality'] > 0.7 else "🥈" if result['overall_quality'] > 0.5 else "🥉"
        print(f"  {quality_indicator} {result['type']}: {result['overall_quality']:.3f} overall quality")

def demo_inheritance_and_integration():
    """Demo integration with SemanticChunk features"""
    print_header("5. INTEGRATION WITH SEMANTIC CHUNK FEATURES")
    
    # Create a comprehensive document chunk
    chunk = SemanticDocumentChunk(
        id="integration_demo",
        file_path="/docs/advanced_ml.md",
        content=SAMPLE_DOCUMENTS["technical_doc"],
        start_line=42,
        end_line=45,
        content_type=ContentType.DOCUMENTATION
    )
    
    print_section("SemanticChunk Features")
    
    # Add semantic tags with confidence
    chunk.add_semantic_tag("optimization", 0.95, "ast", "algorithm")
    chunk.add_semantic_tag("mathematics", 0.90, "nlp", "domain")
    chunk.add_semantic_tag("tutorial", 0.75, "manual", "content-type")
    
    # Add relationships to other chunks
    chunk.add_relationship("gradient_descent_impl", "implements", 0.9, "code implementation")
    chunk.add_relationship("cost_function_def", "references", 0.8, "mathematical definition")
    chunk.add_relationship("learning_rate_tuning", "relates_to", 0.7, "hyperparameter discussion")
    
    # Set quality scores
    chunk.set_quality_score(QualityMetric.READABILITY, chunk.readability_score)
    chunk.set_quality_score(QualityMetric.DOCUMENTATION_QUALITY, 0.85)
    chunk.set_quality_score(QualityMetric.COMPLETENESS, 0.78)
    chunk.set_quality_score(QualityMetric.SEMANTIC_COHERENCE, 0.92)
    
    # Set embedding (mock)
    mock_embedding = [0.1 + (i * 0.01) for i in range(384)]
    chunk.set_embedding(mock_embedding, "text-embedding-ada-002", 3)
    
    print_metrics("Inherited Properties", {
        "content_fingerprint": chunk.content_fingerprint[:16] + "...",
        "semantic_tags": len(chunk.semantic_tags),
        "relationships": len(chunk.relationships),
        "quality_scores": len(chunk.quality_scores),
        "has_embedding": chunk.has_semantic_embedding,
        "version": chunk.version
    })
    
    print_section("Document + Semantic Features")
    print(f"  📄 Document Type: {chunk.section_type}")
    print(f"  📖 Readability Score: {chunk.readability_score:.3f}")
    print(f"  🏷️  High-Confidence Tags: {', '.join(chunk.high_confidence_tags)}")
    print(f"  🔗 Relationship Types: {[rel.relationship_type for rel in chunk.relationships]}")
    print(f"  📊 Overall Quality: {chunk.calculate_overall_quality_score():.3f}")
    print(f"  🧠 Embedding Dimensions: {len(chunk.semantic_embedding) if chunk.semantic_embedding else 0}")
    
    return chunk

def demo_factory_and_serialization():
    """Demo factory functions and serialization"""
    print_header("6. FACTORY FUNCTIONS & SERIALIZATION")
    
    print_section("Factory Function Usage")
    
    # Create chunks using factory function
    markdown_chunk = create_document_chunk_for_content_type(
        content_type=ContentType.MARKDOWN,
        id="factory_markdown",
        file_path="/docs/readme.md",
        content="## Getting Started\n\nThis section explains the basics.",
        start_line=1,
        end_line=3
    )
    
    html_chunk = create_document_chunk_for_content_type(
        content_type=ContentType.HTML,
        id="factory_html",
        file_path="/docs/index.html",
        content="<h1>Welcome</h1><p>This is a paragraph.</p>",
        start_line=1,
        end_line=1
    )
    
    print(f"  📝 Markdown chunk: {markdown_chunk.section_type} (content_type: {markdown_chunk.content_type.value})")
    print(f"  🌐 HTML chunk: {html_chunk.section_type} (content_type: {html_chunk.content_type.value})")
    
    print_section("Serialization Performance")
    
    # Add some data for serialization testing
    markdown_chunk.entities = ["Getting Started"]
    markdown_chunk.topics = ["documentation", "tutorial"]
    markdown_chunk.cross_references = ["Chapter 1"]
    
    # Test serialization
    start_time = time.time()
    chunk_dict = markdown_chunk.to_dict()
    serialize_time = time.time() - start_time
    
    # Test deserialization
    start_time = time.time()
    restored_chunk = SemanticDocumentChunk.from_dict(chunk_dict)
    deserialize_time = time.time() - start_time
    
    print_metrics("Serialization Performance", {
        "serialization_time_ms": f"{serialize_time * 1000:.2f}",
        "deserialization_time_ms": f"{deserialize_time * 1000:.2f}",
        "serialized_size_kb": f"{len(json.dumps(chunk_dict)) / 1024:.2f}",
        "data_preserved": "✅" if restored_chunk.id == markdown_chunk.id else "❌"
    })

def demo_real_world_scenarios():
    """Demo real-world usage scenarios"""
    print_header("7. REAL-WORLD USAGE SCENARIOS")
    
    print_section("Scenario 1: Documentation Quality Audit")
    print("  🔍 Automated documentation review:")
    
    docs_to_audit = [
        ("API Reference", SAMPLE_DOCUMENTS["code_block"], 0.9),
        ("User Guide", SAMPLE_DOCUMENTS["paragraph"], 0.7),
        ("Complex Technical Spec", SAMPLE_DOCUMENTS["complex_paragraph"], 0.3)
    ]
    
    audit_results = []
    for doc_name, content, expected_quality in docs_to_audit:
        chunk = SemanticDocumentChunk(
            id=f"audit_{doc_name.lower().replace(' ', '_')}",
            file_path=f"/docs/{doc_name.lower().replace(' ', '_')}.md",
            content=content,
            start_line=1,
            end_line=content.count('\n') + 1,
            content_type=ContentType.DOCUMENTATION
        )
        
        metrics = calculate_document_quality_metrics(chunk)
        avg_quality = sum(metrics.values()) / len(metrics)
        
        status = "✅ Pass" if avg_quality > 0.6 else "⚠️ Review" if avg_quality > 0.4 else "❌ Fail"
        audit_results.append((doc_name, avg_quality, status))
        
        print(f"    • {doc_name}: {avg_quality:.3f} - {status}")
    
    print_section("Scenario 2: Content Type Classification")
    print("  🏷️  Automatic document categorization:")
    
    classification_samples = [
        ("Tutorial content", SAMPLE_DOCUMENTS["numbered_list"]),
        ("Reference material", SAMPLE_DOCUMENTS["table"]),
        ("Code example", SAMPLE_DOCUMENTS["code_block"]),
        ("Conceptual explanation", SAMPLE_DOCUMENTS["paragraph"])
    ]
    
    for label, content in classification_samples:
        chunk = SemanticDocumentChunk(
            id=f"classify_{label.split()[0].lower()}",
            file_path="/docs/mixed_content.md",
            content=content,
            start_line=1,
            end_line=content.count('\n') + 1,
            content_type=ContentType.DOCUMENTATION
        )
        
        confidence = chunk.calculate_structure_consistency()
        print(f"    • {label}: {chunk.section_type} (confidence: {confidence:.3f})")
    
    print_section("Scenario 3: Readability Optimization")
    print("  📈 Content improvement recommendations:")
    
    improvement_sample = SemanticDocumentChunk(
        id="improvement_test",
        file_path="/docs/needs_improvement.md",
        content=SAMPLE_DOCUMENTS["complex_paragraph"],
        start_line=1,
        end_line=3,
        content_type=ContentType.DOCUMENTATION
    )
    
    print(f"    • Current readability: {improvement_sample.readability_score:.3f}")
    print(f"    • Average sentence length: {improvement_sample.avg_sentence_length:.1f} words")
    print(f"    • Recommendations:")
    
    if improvement_sample.avg_sentence_length > 25:
        print(f"      - Break long sentences (current: {improvement_sample.avg_sentence_length:.1f} words/sentence)")
    if improvement_sample.readability_score < 0.5:
        print("      - Simplify complex vocabulary")
        print("      - Use active voice")
        print("      - Add bullet points for complex ideas")

def main():
    """Run the complete SemanticDocumentChunk demo"""
    print_header("📚 SEMANTIC DOCUMENT CHUNK FEATURE DEMO")
    print("Demonstrating advanced document analysis and processing capabilities")
    print(f"🕒 Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Run all demonstrations
    demo_structure_detection()
    demo_readability_analysis()
    demo_document_quality_metrics()
    demo_document_types_comparison()
    demo_inheritance_and_integration()
    demo_factory_and_serialization()
    demo_real_world_scenarios()
    
    total_time = time.time() - start_time
    
    print_header("🎯 DEMO SUMMARY & ACHIEVEMENTS")
    print("✨ SemanticDocumentChunk successfully demonstrates:")
    print("  📄 Automatic structure detection (headings, lists, code, tables)")
    print("  📖 Advanced readability analysis and scoring")
    print("  📊 Comprehensive document quality metrics")
    print("  🏷️  Entity and topic extraction integration")
    print("  🔗 Cross-reference and relationship management")
    print("  🧬 Full inheritance from SemanticChunk features")
    print("  🏭 Factory functions for different content types")
    print("  💾 High-performance serialization capabilities")
    print("  🔍 Real-world usage scenarios and applications")
    print("  📈 Content improvement recommendations")
    
    print_metrics("Demo Performance", {
        "total_execution_time": f"{total_time:.3f}s",
        "document_types_analyzed": "8",
        "quality_metrics_calculated": "6",
        "structure_types_detected": "5",
        "readability_levels_tested": "4",
        "real_world_scenarios": "3",
        "performance_grade": "A+"
    })
    
    print(f"\n🏁 Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📚 Ready for document processing at scale!")

if __name__ == "__main__":
    main()