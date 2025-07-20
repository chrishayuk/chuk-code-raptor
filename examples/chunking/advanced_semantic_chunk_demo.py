#!/usr/bin/env python3
"""
Advanced SemanticChunk Integration Demo
======================================

Demonstrates how SemanticChunk's fingerprinting enables:
- Merkle tree-based incremental updates
- Code property graph integration  
- Hierarchical indexing with change propagation
- Impact analysis and dependency tracking

This shows the "future-ready" architecture in action.
"""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Set, Any, Optional
from collections import defaultdict

# Import your actual SemanticChunk classes
from chuk_code_raptor.chunking.semantic_chunk import (
    SemanticChunk, ChunkRelationship, SemanticTag, CodePattern,
    QualityMetric, ContentType, ChunkComplexity, create_chunk_id,
    calculate_chunk_similarity, find_related_chunks
)
from chuk_code_raptor.core.models import ChunkType

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

def print_section(title: str):
    """Print a section header"""
    print(f"\n--- {title} ---")

class MockMerkleIndexer:
    """Mock Merkle tree indexer using SemanticChunk fingerprints"""
    
    def __init__(self):
        self.chunk_fingerprints = {}  # chunk_id -> fingerprint
        self.dependency_graph = defaultdict(set)  # chunk_id -> set of dependent chunks
        self.stale_chunks = set()
        
    def register_chunk(self, chunk: SemanticChunk):
        """Register a chunk with its fingerprint"""
        self.chunk_fingerprints[chunk.id] = chunk.combined_fingerprint
        
        # Build reverse dependency graph
        for dep_id in chunk.dependencies:
            self.dependency_graph[dep_id].add(chunk.id)
    
    def detect_changes(self, chunk: SemanticChunk) -> bool:
        """Detect if chunk has changed using fingerprints"""
        old_fingerprint = self.chunk_fingerprints.get(chunk.id)
        current_fingerprint = chunk.combined_fingerprint
        
        if old_fingerprint != current_fingerprint:
            self.chunk_fingerprints[chunk.id] = current_fingerprint
            self._propagate_staleness(chunk.id)
            return True
        return False
    
    def _propagate_staleness(self, changed_chunk_id: str):
        """Propagate staleness up the dependency tree"""
        to_process = {changed_chunk_id}
        processed = set()
        
        while to_process:
            chunk_id = to_process.pop()
            if chunk_id in processed:
                continue
                
            processed.add(chunk_id)
            self.stale_chunks.add(chunk_id)
            
            # Add dependents to processing queue
            dependents = self.dependency_graph.get(chunk_id, set())
            to_process.update(dependents - processed)
    
    def get_impact_set(self, chunk_id: str) -> Set[str]:
        """Get all chunks affected by changes to given chunk"""
        impact_set = set()
        to_visit = {chunk_id}
        
        while to_visit:
            current = to_visit.pop()
            if current in impact_set:
                continue
                
            impact_set.add(current)
            dependents = self.dependency_graph.get(current, set())
            to_visit.update(dependents - impact_set)
            
        return impact_set

class MockCodePropertyGraph:
    """Mock code property graph using SemanticChunk relationships"""
    
    def __init__(self):
        self.nodes = {}  # chunk_id -> SemanticChunk
        self.edges = []  # List of (source, target, relationship_type, metadata)
        self.topic_clusters = defaultdict(list)  # topic -> [chunk_ids]
        
    def add_chunk(self, chunk: SemanticChunk):
        """Add chunk as a graph node"""
        self.nodes[chunk.id] = chunk
        
        # Add edges from chunk relationships
        for rel in chunk.relationships:
            self.edges.append((
                rel.source_chunk_id,
                rel.target_chunk_id, 
                rel.relationship_type,
                {
                    'strength': rel.strength,
                    'context': rel.context,
                    'line_number': rel.line_number
                }
            ))
        
        # Group by semantic tags for topic clustering
        for tag in chunk.high_confidence_tags:
            self.topic_clusters[tag].append(chunk.id)
    
    def find_related_by_structure(self, chunk_id: str, max_hops: int = 2) -> List[str]:
        """Find structurally related chunks via graph traversal"""
        related = set()
        current_level = {chunk_id}
        visited = set()
        
        for hop in range(max_hops):
            next_level = set()
            
            for node in current_level:
                if node in visited:
                    continue
                visited.add(node)
                
                # Find neighbors via edges
                for source, target, rel_type, metadata in self.edges:
                    if source == node and target not in visited:
                        next_level.add(target)
                        related.add(target)
                    elif target == node and source not in visited:
                        next_level.add(source)
                        related.add(source)
                        
            current_level = next_level
            
        return list(related)
    
    def find_related_by_semantics(self, chunk_id: str, similarity_threshold: float = 0.7) -> List[str]:
        """Find semantically related chunks via embeddings"""
        if chunk_id not in self.nodes:
            return []
            
        target_chunk = self.nodes[chunk_id]
        if not target_chunk.has_semantic_embedding:
            return []
            
        related = []
        for other_id, other_chunk in self.nodes.items():
            if other_id == chunk_id or not other_chunk.has_semantic_embedding:
                continue
                
            similarity = calculate_chunk_similarity(target_chunk, other_chunk, method="embedding")
            if similarity >= similarity_threshold:
                related.append((other_id, similarity))
                
        # Sort by similarity and return IDs
        related.sort(key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, _ in related]
    
    def get_quality_distribution(self) -> Dict[str, float]:
        """Analyze quality distribution across the graph"""
        if not self.nodes:
            return {}
            
        quality_scores = [chunk.calculate_overall_quality_score() for chunk in self.nodes.values()]
        
        return {
            'mean': sum(quality_scores) / len(quality_scores),
            'min': min(quality_scores),
            'max': max(quality_scores),
            'high_quality_percent': len([s for s in quality_scores if s > 0.8]) / len(quality_scores) * 100
        }

def create_sample_codebase():
    """Create a sample codebase with related chunks"""
    chunks = []
    
    # Database layer
    db_manager = SemanticChunk(
        id=create_chunk_id("src/db/manager.py", 1, ChunkType.CLASS, "DatabaseManager"),
        file_path="src/db/manager.py",
        content="""class DatabaseManager:
    def __init__(self, connection_string):
        self.connection = connect(connection_string)
        self.cache = LRUCache(1000)
    
    def execute_query(self, query, params=None):
        return self.connection.execute(query, params)""",
        start_line=1, end_line=7,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    # Service layer
    user_service = SemanticChunk(
        id=create_chunk_id("src/services/user.py", 1, ChunkType.CLASS, "UserService"),
        file_path="src/services/user.py", 
        content="""class UserService:
    def __init__(self, db_manager):
        self.db = db_manager
    
    def get_user(self, user_id):
        query = "SELECT * FROM users WHERE id = ?"
        return self.db.execute_query(query, [user_id])
    
    def create_user(self, user_data):
        return self.db.execute_query(
            "INSERT INTO users VALUES (?)", user_data
        )""",
        start_line=1, end_line=11,
        content_type=ContentType.CODE,
        language="python", 
        chunk_type=ChunkType.CLASS
    )
    
    # API layer
    user_api = SemanticChunk(
        id=create_chunk_id("src/api/user.py", 1, ChunkType.FUNCTION, "get_user_endpoint"),
        file_path="src/api/user.py",
        content="""@app.route('/users/<int:user_id>')
def get_user_endpoint(user_id):
    user_service = UserService(db_manager)
    user = user_service.get_user(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify(user.to_dict())""",
        start_line=1, end_line=8,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.FUNCTION
    )
    
    # Add relationships
    user_service.add_relationship(
        db_manager.id, "depends_on", strength=0.9, 
        context="uses for database operations", line_number=3
    )
    
    user_api.add_relationship(
        user_service.id, "depends_on", strength=0.85,
        context="uses for business logic", line_number=3
    )
    
    user_api.add_relationship(
        db_manager.id, "depends_on", strength=0.7,
        context="indirect dependency via service", line_number=3
    )
    
    # Add semantic tags
    for chunk in [db_manager, user_service, user_api]:
        chunk.add_semantic_tag("user-management", confidence=0.9, source="analysis")
        chunk.add_semantic_tag("web-api", confidence=0.8, source="pattern")
    
    db_manager.add_semantic_tag("database", confidence=0.95, source="ast")
    db_manager.add_semantic_tag("caching", confidence=0.85, source="pattern")
    
    user_service.add_semantic_tag("business-logic", confidence=0.9, source="analysis")
    user_service.add_semantic_tag("service-layer", confidence=0.88, source="architecture")
    
    user_api.add_semantic_tag("rest-api", confidence=0.92, source="pattern")
    user_api.add_semantic_tag("endpoint", confidence=0.95, source="ast")
    
    # Add embeddings (mock)
    for i, chunk in enumerate([db_manager, user_service, user_api]):
        # Mock embeddings that are similar (user management domain)
        base_embedding = [0.1, 0.8, 0.2] * 100  # 300-dim
        # Add slight variations
        embedding = [x + (i * 0.05) for x in base_embedding]
        chunk.set_embedding(embedding, "text-embedding-ada-002", 1)
    
    # Set quality scores
    db_manager.set_quality_score(QualityMetric.MAINTAINABILITY, 0.85)
    db_manager.set_quality_score(QualityMetric.READABILITY, 0.78)
    
    user_service.set_quality_score(QualityMetric.MAINTAINABILITY, 0.82)
    user_service.set_quality_score(QualityMetric.SEMANTIC_COHERENCE, 0.90)
    
    user_api.set_quality_score(QualityMetric.MAINTAINABILITY, 0.75)
    user_api.set_quality_score(QualityMetric.COMPLETENESS, 0.88)
    
    return [db_manager, user_service, user_api]

def demo_merkle_change_detection():
    """Demo Merkle tree-based change detection"""
    print_header("MERKLE TREE CHANGE DETECTION")
    
    chunks = create_sample_codebase()
    indexer = MockMerkleIndexer()
    
    # Register all chunks
    for chunk in chunks:
        indexer.register_chunk(chunk)
    
    print(f"Registered {len(chunks)} chunks in Merkle indexer")
    print(f"Dependency relationships: {len(indexer.dependency_graph)} nodes")
    
    print_section("Simulating Code Change")
    
    # Simulate change to database manager
    db_chunk = chunks[0]  # DatabaseManager
    original_fingerprint = db_chunk.combined_fingerprint
    
    print(f"Original fingerprint: {original_fingerprint[:16]}...")
    
    # Modify content (add logging)
    db_chunk.content += "\n        self.logger.info(f'Executing query: {query}')"
    db_chunk.update_fingerprints()
    
    print(f"New fingerprint: {db_chunk.combined_fingerprint[:16]}...")
    
    # Detect changes
    changed = indexer.detect_changes(db_chunk)
    print(f"Change detected: {changed}")
    print(f"Stale chunks: {indexer.stale_chunks}")
    
    # Get impact set
    impact_set = indexer.get_impact_set(db_chunk.id)
    print(f"Impact set: {impact_set}")
    
    return chunks, indexer

def demo_property_graph_integration(chunks):
    """Demo code property graph integration"""
    print_header("CODE PROPERTY GRAPH INTEGRATION")
    
    graph = MockCodePropertyGraph()
    
    # Add all chunks to graph
    for chunk in chunks:
        graph.add_chunk(chunk)
    
    print(f"Graph nodes: {len(graph.nodes)}")
    print(f"Graph edges: {len(graph.edges)}")
    print(f"Topic clusters: {len(graph.topic_clusters)}")
    
    print_section("Graph Structure Analysis")
    
    for source, target, rel_type, metadata in graph.edges:
        print(f"  {source} --[{rel_type}]--> {target} (strength: {metadata['strength']})")
    
    print_section("Topic Clustering")
    
    for topic, chunk_ids in graph.topic_clusters.items():
        print(f"  {topic}: {len(chunk_ids)} chunks")
        for chunk_id in chunk_ids:
            print(f"    - {chunk_id}")
    
    print_section("Structural Relationships")
    
    # Find related chunks via graph traversal
    user_service_id = chunks[1].id  # UserService
    structural_related = graph.find_related_by_structure(user_service_id, max_hops=2)
    print(f"Chunks related to {user_service_id} (structural):")
    for related_id in structural_related:
        print(f"  - {related_id}")
    
    print_section("Semantic Relationships")
    
    # Find related chunks via embeddings
    semantic_related = graph.find_related_by_semantics(user_service_id, similarity_threshold=0.8)
    print(f"Chunks related to {user_service_id} (semantic):")
    for related_id in semantic_related:
        similarity = calculate_chunk_similarity(chunks[1], graph.nodes[related_id], method="embedding")
        print(f"  - {related_id} (similarity: {similarity:.3f})")
    
    return graph

def demo_hierarchical_analysis(chunks, graph):
    """Demo hierarchical analysis capabilities"""
    print_header("HIERARCHICAL ANALYSIS")
    
    print_section("Quality Distribution Analysis")
    
    quality_dist = graph.get_quality_distribution()
    print(f"Quality distribution across codebase:")
    for metric, value in quality_dist.items():
        if 'percent' in metric:
            print(f"  {metric}: {value:.1f}%")
        else:
            print(f"  {metric}: {value:.3f}")
    
    print_section("Complexity Analysis")
    
    complexity_distribution = {}
    for chunk in chunks:
        complexity = chunk.complexity_level.value
        complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
    
    print("Complexity distribution:")
    for complexity, count in complexity_distribution.items():
        print(f"  {complexity}: {count} chunks")
    
    print_section("Pattern Detection Summary")
    
    all_patterns = []
    for chunk in chunks:
        all_patterns.extend([p.pattern_name for p in chunk.detected_patterns])
    
    pattern_counts = {}
    for pattern in all_patterns:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print("Detected patterns across codebase:")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} occurrences")

def demo_incremental_update_simulation(chunks, indexer):
    """Demo incremental update simulation"""
    print_header("INCREMENTAL UPDATE SIMULATION")
    
    print_section("Initial State")
    print(f"Clean chunks: {len(chunks) - len(indexer.stale_chunks)}")
    print(f"Stale chunks: {len(indexer.stale_chunks)}")
    
    print_section("Batch Update Simulation")
    
    # Simulate fixing the stale chunks
    stale_chunk_ids = list(indexer.stale_chunks)
    print(f"Processing {len(stale_chunk_ids)} stale chunks...")
    
    for chunk_id in stale_chunk_ids:
        # Find the chunk
        chunk = next((c for c in chunks if c.id == chunk_id), None)
        if chunk:
            # Simulate re-processing
            chunk.update_fingerprints()
            print(f"  ✅ Reprocessed {chunk_id}")
            
            # Update indexer
            indexer.chunk_fingerprints[chunk_id] = chunk.combined_fingerprint
    
    # Clear stale set
    indexer.stale_chunks.clear()
    
    print_section("Final State")
    print(f"Clean chunks: {len(chunks)}")
    print(f"Stale chunks: {len(indexer.stale_chunks)}")
    print("✅ All chunks up to date!")

def demo_advanced_search_simulation(chunks, graph):
    """Demo advanced search capabilities"""
    print_header("ADVANCED SEARCH SIMULATION")
    
    query = "database user management"
    print(f"Search query: '{query}'")
    
    print_section("Multi-Modal Search Results")
    
    # 1. Semantic search (via embeddings)
    print("1. Semantic Search (via embeddings):")
    semantic_results = []
    for chunk in chunks:
        if chunk.has_semantic_embedding:
            # Mock semantic scoring based on tags
            score = 0.0
            for tag in chunk.tag_names:
                if any(word in tag for word in query.split()):
                    score += 0.3
            semantic_results.append((chunk.id, score))
    
    semantic_results.sort(key=lambda x: x[1], reverse=True)
    for chunk_id, score in semantic_results[:3]:
        print(f"   - {chunk_id} (semantic score: {score:.2f})")
    
    # 2. Graph-based expansion
    print("\n2. Graph-based Expansion:")
    if semantic_results:
        top_result_id = semantic_results[0][0]
        expanded = graph.find_related_by_structure(top_result_id, max_hops=1)
        print(f"   Expanding from {top_result_id}:")
        for related_id in expanded:
            print(f"   - {related_id}")
    
    # 3. Quality-weighted ranking
    print("\n3. Quality-weighted Ranking:")
    final_results = []
    all_candidates = {result[0] for result in semantic_results} | set(expanded if semantic_results else [])
    
    for chunk_id in all_candidates:
        chunk = next((c for c in chunks if c.id == chunk_id), None)
        if chunk:
            quality_score = chunk.calculate_overall_quality_score()
            semantic_score = next((s for cid, s in semantic_results if cid == chunk_id), 0.0)
            combined_score = (semantic_score * 0.7) + (quality_score * 0.3)
            final_results.append((chunk_id, combined_score, quality_score))
    
    final_results.sort(key=lambda x: x[1], reverse=True)
    print("   Final ranked results:")
    for chunk_id, combined_score, quality_score in final_results:
        print(f"   - {chunk_id} (combined: {combined_score:.3f}, quality: {quality_score:.3f})")

def main():
    """Run the complete advanced SemanticChunk demo"""
    print_header("ADVANCED SEMANTIC CHUNK INTEGRATION DEMO")
    print("Demonstrating Merkle trees + Property graphs + Hierarchical indexing")
    
    # Demo Merkle tree change detection
    chunks, indexer = demo_merkle_change_detection()
    
    # Demo property graph integration
    graph = demo_property_graph_integration(chunks)
    
    # Demo hierarchical analysis
    demo_hierarchical_analysis(chunks, graph)
    
    # Demo incremental updates
    demo_incremental_update_simulation(chunks, indexer)
    
    # Demo advanced search
    demo_advanced_search_simulation(chunks, graph)
    
    print_header("INTEGRATION DEMO COMPLETE")
    print("SemanticChunk enables:")
    print("✅ Merkle tree-based incremental updates")
    print("✅ Rich code property graphs")
    print("✅ Multi-modal search (semantic + structural)")
    print("✅ Quality-aware ranking")
    print("✅ Change impact analysis")
    print("✅ Hierarchical code organization")
    print("✅ Enterprise-ready scalability")

if __name__ == "__main__":
    main()