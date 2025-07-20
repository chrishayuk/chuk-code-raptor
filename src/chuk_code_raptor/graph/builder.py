#!/usr/bin/env python3
# src/chuk_code_raptor/graph/builder.py
"""
Code Property Graph Builder
===========================

High-level builder for constructing and updating Code Property Graphs
from SemanticChunks with incremental update capabilities.

Path: chuk_code_raptor/graph/builder.py
"""

import numpy as np
from typing import List, Set, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging

from chuk_code_raptor.chunking.semantic_chunk import SemanticChunk, calculate_chunk_similarity
from .core import CodePropertyGraph
from .models import GraphType, EdgeType, GraphEdge
from .analytics import GraphAnalytics

logger = logging.getLogger(__name__)

class CPGBuilder:
    """
    High-level builder for Code Property Graphs
    
    Features:
    - Incremental updates from SemanticChunk changes
    - Semantic relationship detection via embeddings
    - Quality-based relationship inference
    - Pattern-based relationship detection
    - Batch processing for large codebases
    """
    
    def __init__(self, cpg: Optional[CodePropertyGraph] = None):
        self.cpg = cpg or CodePropertyGraph()
        self.analytics = GraphAnalytics(self.cpg)
        
        # Configuration
        self.semantic_similarity_threshold = 0.8
        self.quality_similarity_threshold = 0.7
        self.pattern_similarity_threshold = 0.75
        
    def build_from_chunks(self, chunks: List[SemanticChunk], 
                         include_semantic_edges: bool = True,
                         include_quality_edges: bool = True,
                         include_pattern_edges: bool = True) -> CodePropertyGraph:
        """
        Build a complete CPG from a list of SemanticChunks
        
        Args:
            chunks: List of SemanticChunks to process
            include_semantic_edges: Add semantic similarity edges
            include_quality_edges: Add quality-based edges
            include_pattern_edges: Add pattern similarity edges
            
        Returns:
            The constructed CodePropertyGraph
        """
        logger.info(f"Building CPG from {len(chunks)} chunks")
        
        # Add all chunks first
        for chunk in chunks:
            self.cpg.add_or_update_chunk(chunk)
        
        # Add derived relationships
        if include_semantic_edges:
            self._add_semantic_relationships(chunks)
            
        if include_quality_edges:
            self._add_quality_relationships(chunks)
            
        if include_pattern_edges:
            self._add_pattern_relationships(chunks)
        
        logger.info(f"CPG built with {len(self.cpg.nodes)} nodes and {len(self.cpg.edges)} edges")
        return self.cpg
    
    def update_from_changed_chunks(self, changed_chunks: List[SemanticChunk]) -> Set[str]:
        """
        Incrementally update the CPG from changed chunks
        
        Args:
            changed_chunks: List of chunks that have changed
            
        Returns:
            Set of node IDs that were affected by the changes
        """
        logger.info(f"Updating CPG with {len(changed_chunks)} changed chunks")
        
        affected_nodes = set()
        
        for chunk in changed_chunks:
            # Get impact set before updating
            if chunk.id in self.cpg.chunk_index:
                impact = self.cpg.get_impact_set(chunk.id)
                affected_nodes.update(impact.total_affected)
            
            # Update the chunk
            self.cpg.add_or_update_chunk(chunk)
            affected_nodes.add(chunk.id)
        
        # Recompute derived relationships for affected areas
        self._recompute_derived_relationships(affected_nodes)
        
        logger.info(f"CPG updated, {len(affected_nodes)} nodes affected")
        return affected_nodes
    
    def _add_semantic_relationships(self, chunks: List[SemanticChunk]):
        """Add semantic similarity relationships based on embeddings"""
        logger.info("Adding semantic relationships...")
        
        # Filter chunks with embeddings
        embedded_chunks = [c for c in chunks if c.has_semantic_embedding]
        
        if len(embedded_chunks) < 2:
            logger.warning("Not enough chunks with embeddings for semantic relationships")
            return
        
        relationships_added = 0
        
        for i, chunk1 in enumerate(embedded_chunks):
            for chunk2 in embedded_chunks[i+1:]:
                similarity = calculate_chunk_similarity(chunk1, chunk2, method="embedding")
                
                if similarity >= self.semantic_similarity_threshold:
                    self._add_semantic_edge(chunk1.id, chunk2.id, similarity, "embedding_similarity")
                    relationships_added += 1
        
        logger.info(f"Added {relationships_added} semantic relationships")
    
    def _add_quality_relationships(self, chunks: List[SemanticChunk]):
        """Add quality-based relationships"""
        logger.info("Adding quality relationships...")
        
        # Group chunks by quality ranges
        quality_groups = defaultdict(list)
        
        for chunk in chunks:
            quality = chunk.calculate_overall_quality_score()
            quality_bucket = int(quality * 10) / 10  # Round to nearest 0.1
            quality_groups[quality_bucket].append(chunk)
        
        relationships_added = 0
        
        # Connect chunks with similar quality scores
        for quality_bucket, chunk_group in quality_groups.items():
            if len(chunk_group) > 1:
                for i, chunk1 in enumerate(chunk_group):
                    for chunk2 in chunk_group[i+1:]:
                        quality_diff = abs(
                            chunk1.calculate_overall_quality_score() - 
                            chunk2.calculate_overall_quality_score()
                        )
                        
                        if quality_diff <= (1.0 - self.quality_similarity_threshold):
                            similarity = 1.0 - quality_diff
                            self._add_quality_edge(chunk1.id, chunk2.id, similarity, "quality_similarity")
                            relationships_added += 1
        
        logger.info(f"Added {relationships_added} quality relationships")
    
    def _add_pattern_relationships(self, chunks: List[SemanticChunk]):
        """Add pattern-based relationships"""
        logger.info("Adding pattern relationships...")
        
        # Group chunks by detected patterns
        pattern_groups = defaultdict(list)
        
        for chunk in chunks:
            for pattern in chunk.detected_patterns:
                pattern_groups[pattern.pattern_name].append((chunk, pattern.confidence))
        
        relationships_added = 0
        
        # Connect chunks with shared patterns
        for pattern_name, chunk_pattern_pairs in pattern_groups.items():
            if len(chunk_pattern_pairs) > 1:
                for i, (chunk1, conf1) in enumerate(chunk_pattern_pairs):
                    for (chunk2, conf2) in chunk_pattern_pairs[i+1:]:
                        # Similarity based on pattern confidence
                        similarity = (conf1 + conf2) / 2
                        
                        if similarity >= self.pattern_similarity_threshold:
                            self._add_pattern_edge(
                                chunk1.id, chunk2.id, similarity, 
                                f"shared_pattern:{pattern_name}"
                            )
                            relationships_added += 1
        
        logger.info(f"Added {relationships_added} pattern relationships")
    
    def _add_semantic_edge(self, source_id: str, target_id: str, 
                          similarity: float, context: str):
        """Add a semantic similarity edge"""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.SIMILAR_TO,
            graph_type=GraphType.SEMANTIC_GRAPH,
            weight=similarity,
            confidence=similarity,
            context=context,
            metadata={'similarity_method': 'embedding'}
        )
        
        edge_id = f"{source_id}->{target_id}:similar_to"
        self.cpg.edges[edge_id] = edge
        
        # Add to graph
        self.cpg.graphs[GraphType.SEMANTIC_GRAPH].add_edge(
            source_id, target_id,
            edge_type=EdgeType.SIMILAR_TO.value,
            weight=similarity,
            similarity=similarity,
            context=context
        )
    
    def _add_quality_edge(self, source_id: str, target_id: str, 
                         similarity: float, context: str):
        """Add a quality-based relationship edge"""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.QUALITY_SIMILAR,
            graph_type=GraphType.QUALITY_GRAPH,
            weight=similarity,
            confidence=similarity,
            context=context,
            metadata={'similarity_method': 'quality_score'}
        )
        
        edge_id = f"{source_id}->{target_id}:quality_similar"
        self.cpg.edges[edge_id] = edge
        
        # Add to graph
        self.cpg.graphs[GraphType.QUALITY_GRAPH].add_edge(
            source_id, target_id,
            edge_type=EdgeType.QUALITY_SIMILAR.value,
            weight=similarity,
            quality_similarity=similarity,
            context=context
        )
    
    def _add_pattern_edge(self, source_id: str, target_id: str,
                         similarity: float, context: str):
        """Add a pattern-based relationship edge"""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.PATTERN_SIMILAR,
            graph_type=GraphType.QUALITY_GRAPH,
            weight=similarity,
            confidence=similarity,
            context=context,
            metadata={'similarity_method': 'pattern_detection'}
        )
        
        edge_id = f"{source_id}->{target_id}:pattern_similar"
        self.cpg.edges[edge_id] = edge
        
        # Add to graph
        self.cpg.graphs[GraphType.QUALITY_GRAPH].add_edge(
            source_id, target_id,
            edge_type=EdgeType.PATTERN_SIMILAR.value,
            weight=similarity,
            pattern_similarity=similarity,
            context=context
        )
    
    def _recompute_derived_relationships(self, affected_nodes: Set[str]):
        """Recompute derived relationships for affected nodes"""
        logger.info(f"Recomputing derived relationships for {len(affected_nodes)} nodes")
        
        # Remove existing derived edges for affected nodes
        edges_to_remove = []
        for edge_id, edge in self.cpg.edges.items():
            if (edge.source_id in affected_nodes or edge.target_id in affected_nodes) and \
               edge.graph_type in [GraphType.SEMANTIC_GRAPH, GraphType.QUALITY_GRAPH]:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            edge = self.cpg.edges[edge_id]
            graph = self.cpg.graphs[edge.graph_type]
            if graph.has_edge(edge.source_id, edge.target_id):
                graph.remove_edge(edge.source_id, edge.target_id)
            del self.cpg.edges[edge_id]
        
        # Get affected chunks
        affected_chunks = [
            self.cpg.chunk_index[node_id] 
            for node_id in affected_nodes 
            if node_id in self.cpg.chunk_index
        ]
        
        # Recompute relationships
        if len(affected_chunks) > 1:
            self._add_semantic_relationships(affected_chunks)
            self._add_quality_relationships(affected_chunks)
            self._add_pattern_relationships(affected_chunks)
    
    def add_file_hierarchy_relationships(self, file_structure: Dict[str, List[str]]):
        """
        Add hierarchical relationships based on file structure
        
        Args:
            file_structure: Dict mapping directory paths to lists of file paths
        """
        logger.info("Adding file hierarchy relationships...")
        
        relationships_added = 0
        
        for directory, files in file_structure.items():
            # Find chunks for each file
            dir_chunks = []
            for file_path in files:
                file_chunks = [
                    chunk for chunk in self.cpg.chunk_index.values()
                    if chunk.file_path == file_path
                ]
                dir_chunks.extend(file_chunks)
            
            # Add containment relationships within the same directory
            for i, chunk1 in enumerate(dir_chunks):
                for chunk2 in dir_chunks[i+1:]:
                    if chunk1.file_path == chunk2.file_path:
                        # Same file - check line numbers for containment
                        if self._is_contained(chunk1, chunk2):
                            self._add_hierarchy_edge(chunk2.id, chunk1.id, "file_containment")
                            relationships_added += 1
                        elif self._is_contained(chunk2, chunk1):
                            self._add_hierarchy_edge(chunk1.id, chunk2.id, "file_containment")
                            relationships_added += 1
                    else:
                        # Different files in same directory
                        self._add_hierarchy_edge(chunk1.id, chunk2.id, "directory_sibling")
                        relationships_added += 1
        
        logger.info(f"Added {relationships_added} hierarchy relationships")
    
    def _is_contained(self, container: SemanticChunk, contained: SemanticChunk) -> bool:
        """Check if one chunk contains another based on line numbers"""
        return (container.start_line <= contained.start_line and 
                container.end_line >= contained.end_line and
                container.start_line < contained.start_line)  # Avoid self-containment
    
    def _add_hierarchy_edge(self, source_id: str, target_id: str, context: str):
        """Add a hierarchical relationship edge"""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.CONTAINS,
            graph_type=GraphType.HIERARCHY_GRAPH,
            weight=1.0,
            confidence=1.0,
            context=context,
            metadata={'hierarchy_type': context}
        )
        
        edge_id = f"{source_id}->{target_id}:contains"
        self.cpg.edges[edge_id] = edge
        
        # Add to graph
        self.cpg.graphs[GraphType.HIERARCHY_GRAPH].add_edge(
            source_id, target_id,
            edge_type=EdgeType.CONTAINS.value,
            weight=1.0,
            context=context
        )
    
    def get_build_summary(self) -> Dict[str, Any]:
        """Get summary of the CPG build process"""
        summary = self.cpg.get_summary()
        
        # Add builder-specific metrics
        summary['builder_config'] = {
            'semantic_threshold': self.semantic_similarity_threshold,
            'quality_threshold': self.quality_similarity_threshold,
            'pattern_threshold': self.pattern_similarity_threshold
        }
        
        # Relationship type breakdown
        relationship_types = defaultdict(int)
        for edge in self.cpg.edges.values():
            relationship_types[edge.edge_type.value] += 1
        
        summary['relationship_breakdown'] = dict(relationship_types)
        
        return summary

def create_cpg_from_chunks(chunks: List[SemanticChunk], 
                          include_semantic_edges: bool = True,
                          include_quality_edges: bool = True,
                          include_pattern_edges: bool = True) -> CodePropertyGraph:
    """
    Convenience function to create a CPG from chunks
    
    Args:
        chunks: List of SemanticChunks
        include_semantic_edges: Add semantic similarity edges
        include_quality_edges: Add quality-based edges  
        include_pattern_edges: Add pattern similarity edges
        
    Returns:
        Constructed CodePropertyGraph
    """
    builder = CPGBuilder()
    return builder.build_from_chunks(
        chunks, 
        include_semantic_edges=include_semantic_edges,
        include_quality_edges=include_quality_edges,
        include_pattern_edges=include_pattern_edges
    )

def update_cpg_from_changes(cpg: CodePropertyGraph, 
                           changed_chunks: List[SemanticChunk]) -> Set[str]:
    """
    Convenience function to update a CPG with changed chunks
    
    Args:
        cpg: Existing CodePropertyGraph
        changed_chunks: List of changed SemanticChunks
        
    Returns:
        Set of affected node IDs
    """
    builder = CPGBuilder(cpg)
    return builder.update_from_changed_chunks(changed_chunks)