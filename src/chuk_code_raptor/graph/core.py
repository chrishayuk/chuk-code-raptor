#!/usr/bin/env python3
# src/chuk_code_raptor/graph/core.py
"""
Code Property Graph Core
=======================

Core Code Property Graph implementation with incremental updates.
Handles graph construction, updates, and basic operations.

Path: chuk_code_raptor/graph/core.py
"""

import hashlib
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict
import networkx as nx

from chuk_code_raptor.chunking.semantic_chunk import SemanticChunk
from .models import (
    GraphType, EdgeType, GraphEdge, GraphNode, GraphMetrics, ChangeImpact,
    classify_relationship, create_edge_id
)

class CodePropertyGraph:
    """
    Core Code Property Graph with incremental update capabilities
    
    Features:
    - Multiple graph types with typed edges
    - Incremental updates via SemanticChunk fingerprints
    - Change tracking and impact analysis
    - Graph analytics and metrics
    """
    
    def __init__(self):
        # Core graph storage - one NetworkX graph per type
        self.graphs: Dict[GraphType, nx.MultiDiGraph] = {
            graph_type: nx.MultiDiGraph() for graph_type in GraphType
        }
        
        # Node and edge storage
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        
        # Chunk tracking for incremental updates
        self.chunk_fingerprints: Dict[str, str] = {}  # chunk_id -> fingerprint
        self.chunk_index: Dict[str, SemanticChunk] = {}  # chunk_id -> chunk
        
        # Change tracking
        self.dirty_nodes: Set[str] = set()
        self.version: int = 1
        
        # Analytics cache
        self._metrics_cache: Dict[GraphType, GraphMetrics] = {}
        self._cache_dirty = True
        
    def add_or_update_chunk(self, chunk: SemanticChunk) -> bool:
        """
        Add or update a chunk in the graph
        
        Args:
            chunk: SemanticChunk to add/update
            
        Returns:
            True if chunk was added/updated, False if no changes
        """
        chunk_id = chunk.id
        old_fingerprint = self.chunk_fingerprints.get(chunk_id)
        new_fingerprint = chunk.combined_fingerprint
        
        # Check if chunk actually changed
        if old_fingerprint == new_fingerprint:
            return False
        
        # Remove old chunk if it existed
        if chunk_id in self.chunk_index:
            self._remove_chunk_edges(chunk_id)
        
        # Update chunk storage
        self.chunk_index[chunk_id] = chunk
        self.chunk_fingerprints[chunk_id] = new_fingerprint
        
        # Create/update graph node
        self._create_or_update_node(chunk)
        
        # Process chunk relationships into edges
        self._process_chunk_relationships(chunk)
        
        # Mark for recomputation
        self.dirty_nodes.add(chunk_id)
        self._invalidate_cache()
        self.version += 1
        
        return True
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove a chunk from the graph
        
        Args:
            chunk_id: ID of chunk to remove
            
        Returns:
            True if chunk was removed, False if not found
        """
        if chunk_id not in self.chunk_index:
            return False
        
        # Remove edges
        self._remove_chunk_edges(chunk_id)
        
        # Remove from all graphs
        for graph in self.graphs.values():
            if graph.has_node(chunk_id):
                graph.remove_node(chunk_id)
        
        # Remove from storage
        del self.chunk_index[chunk_id]
        del self.chunk_fingerprints[chunk_id]
        if chunk_id in self.nodes:
            del self.nodes[chunk_id]
        
        self.dirty_nodes.discard(chunk_id)
        self._invalidate_cache()
        self.version += 1
        
        return True
    
    def _create_or_update_node(self, chunk: SemanticChunk):
        """Create or update a graph node from a chunk"""
        chunk_id = chunk.id
        
        # Create node metadata
        node_metadata = {
            'chunk_type': chunk.chunk_type.value,
            'file_path': chunk.file_path,
            'content_type': chunk.content_type.value,
            'language': chunk.language,
            'complexity': chunk.complexity_level.value,
            'tags': chunk.tag_names,
            'line_count': chunk.line_count,
            'character_count': chunk.character_count,
            'has_embedding': chunk.has_semantic_embedding,
            'version': chunk.version
        }
        
        # Create or update GraphNode
        if chunk_id not in self.nodes:
            self.nodes[chunk_id] = GraphNode(
                node_id=chunk_id,
                node_type="chunk",
                quality_score=chunk.calculate_overall_quality_score(),
                importance_score=chunk.importance_score,
                metadata=node_metadata
            )
        else:
            self.nodes[chunk_id].quality_score = chunk.calculate_overall_quality_score()
            self.nodes[chunk_id].importance_score = chunk.importance_score
            self.nodes[chunk_id].metadata.update(node_metadata)
            self.nodes[chunk_id].last_updated = datetime.now()
        
        # Add node to all graph types
        for graph in self.graphs.values():
            if not graph.has_node(chunk_id):
                graph.add_node(chunk_id, **node_metadata)
            else:
                # Update node attributes
                for key, value in node_metadata.items():
                    graph.nodes[chunk_id][key] = value
    
    def _remove_chunk_edges(self, chunk_id: str):
        """Remove all edges related to a chunk"""
        edges_to_remove = []
        
        for edge_id, edge in self.edges.items():
            if edge.source_id == chunk_id or edge.target_id == chunk_id:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            edge = self.edges[edge_id]
            graph = self.graphs[edge.graph_type]
            
            if graph.has_edge(edge.source_id, edge.target_id):
                graph.remove_edge(edge.source_id, edge.target_id)
            
            del self.edges[edge_id]
    
    def _process_chunk_relationships(self, chunk: SemanticChunk):
        """Process SemanticChunk relationships into graph edges"""
        for relationship in chunk.relationships:
            self._add_relationship_edge(
                source_id=relationship.source_chunk_id,
                target_id=relationship.target_chunk_id,
                relationship_type=relationship.relationship_type,
                strength=relationship.strength,
                context=relationship.context,
                line_number=relationship.line_number,
                metadata=relationship.metadata
            )
    
    def _add_relationship_edge(self, source_id: str, target_id: str, 
                             relationship_type: str, strength: float = 1.0,
                             context: str = "", line_number: Optional[int] = None,
                             metadata: Dict[str, Any] = None):
        """Add a relationship edge to appropriate graphs"""
        metadata = metadata or {}
        
        # Classify relationship
        edge_type, graph_type = classify_relationship(relationship_type)
        
        # Create edge
        edge_id = create_edge_id(source_id, target_id, edge_type)
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            graph_type=graph_type,
            weight=strength,
            source_line=line_number,
            context=context,
            metadata=metadata
        )
        
        self.edges[edge_id] = edge
        
        # Add to appropriate graph
        graph = self.graphs[graph_type]
        graph.add_edge(
            source_id, target_id,
            edge_type=edge_type.value,
            weight=strength,
            context=context,
            line=line_number,
            **metadata
        )
    
    def get_neighbors(self, node_id: str, graph_type: GraphType = GraphType.CALL_GRAPH,
                     direction: str = "both", max_distance: int = 1) -> Set[str]:
        """
        Get neighbors within max_distance hops
        
        Args:
            node_id: Starting node
            graph_type: Which graph to traverse
            direction: "in", "out", or "both"
            max_distance: Maximum hops
            
        Returns:
            Set of neighbor node IDs
        """
        graph = self.graphs[graph_type]
        
        if not graph.has_node(node_id):
            return set()
        
        # Choose graph direction
        if direction == "out":
            traverse_graph = graph
        elif direction == "in":
            traverse_graph = graph.reverse()
        else:  # both
            traverse_graph = graph.to_undirected()
        
        neighbors = set()
        current_level = {node_id}
        visited = {node_id}
        
        for _ in range(max_distance):
            next_level = set()
            for node in current_level:
                if node in traverse_graph:
                    for neighbor in traverse_graph.neighbors(node):
                        if neighbor not in visited:
                            next_level.add(neighbor)
                            neighbors.add(neighbor)
                            visited.add(neighbor)
            current_level = next_level
            
        return neighbors
    
    def find_shortest_path(self, source_id: str, target_id: str, 
                          graph_type: GraphType = GraphType.CALL_GRAPH) -> Optional[List[str]]:
        """Find shortest path between two nodes"""
        graph = self.graphs[graph_type]
        
        if not (graph.has_node(source_id) and graph.has_node(target_id)):
            return None
        
        try:
            return nx.shortest_path(graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return None
    
    def get_impact_set(self, changed_node_id: str, max_hops: int = 3) -> ChangeImpact:
        """
        Get comprehensive impact analysis for a changed node
        
        Args:
            changed_node_id: Node that changed
            max_hops: Maximum traversal distance
            
        Returns:
            ChangeImpact with detailed analysis
        """
        impact = ChangeImpact(changed_nodes=[changed_node_id])
        
        # Direct impact (1 hop)
        for graph_type in [GraphType.CALL_GRAPH, GraphType.DATA_FLOW, GraphType.IMPORT_GRAPH]:
            direct_neighbors = self.get_neighbors(
                changed_node_id, graph_type, direction="out", max_distance=1
            )
            impact.directly_affected.update(direct_neighbors)
            
            if graph_type.value not in impact.impact_by_graph_type:
                impact.impact_by_graph_type[graph_type.value] = set()
            impact.impact_by_graph_type[graph_type.value].update(direct_neighbors)
        
        # Transitive impact (2+ hops)
        for graph_type in [GraphType.CALL_GRAPH, GraphType.DATA_FLOW, GraphType.IMPORT_GRAPH]:
            if max_hops > 1:
                transitive_neighbors = self.get_neighbors(
                    changed_node_id, graph_type, direction="out", max_distance=max_hops
                )
                # Remove direct neighbors to get only transitive
                transitive_only = transitive_neighbors - impact.directly_affected
                impact.transitively_affected.update(transitive_only)
                impact.impact_by_graph_type[graph_type.value].update(transitive_only)
        
        # Quality impact analysis
        if changed_node_id in self.nodes:
            changed_quality = self.nodes[changed_node_id].quality_score
            for affected_id in impact.total_affected:
                if affected_id in self.nodes:
                    affected_quality = self.nodes[affected_id].quality_score
                    impact.quality_impact[affected_id] = abs(changed_quality - affected_quality)
        
        return impact
    
    def _invalidate_cache(self):
        """Invalidate analytics cache"""
        self._metrics_cache.clear()
        self._cache_dirty = True
    
    def get_graph_metrics(self, graph_type: GraphType) -> GraphMetrics:
        """
        Get comprehensive metrics for a specific graph type
        
        Args:
            graph_type: Which graph to analyze
            
        Returns:
            GraphMetrics with detailed analysis
        """
        if graph_type in self._metrics_cache and not self._cache_dirty:
            return self._metrics_cache[graph_type]
        
        graph = self.graphs[graph_type]
        metrics = GraphMetrics()
        
        # Basic metrics
        metrics.node_count = graph.number_of_nodes()
        metrics.edge_count = graph.number_of_edges()
        
        if metrics.node_count > 1:
            metrics.density = nx.density(graph)
            metrics.is_connected = nx.is_weakly_connected(graph)
            metrics.component_count = nx.number_weakly_connected_components(graph)
            
            try:
                # These can be expensive for large graphs
                if metrics.node_count < 1000:
                    undirected = graph.to_undirected()
                    metrics.average_clustering = nx.average_clustering(undirected)
                    
                    if metrics.is_connected:
                        metrics.diameter = nx.diameter(undirected)
            except:
                pass  # Skip expensive computations if they fail
        
        # Quality metrics from chunks
        quality_scores = []
        complexity_dist = defaultdict(int)
        
        for node_id in graph.nodes():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                quality_scores.append(node.quality_score)
                
                if 'complexity' in node.metadata:
                    complexity_dist[node.metadata['complexity']] += 1
        
        if quality_scores:
            metrics.average_quality = sum(quality_scores) / len(quality_scores)
            metrics.quality_variance = sum(
                (q - metrics.average_quality) ** 2 for q in quality_scores
            ) / len(quality_scores)
            metrics.high_quality_percentage = (
                len([q for q in quality_scores if q > 0.8]) / len(quality_scores) * 100
            )
        
        metrics.complexity_distribution = dict(complexity_dist)
        
        # Cache the result
        self._metrics_cache[graph_type] = metrics
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive graph summary"""
        summary = {
            'version': self.version,
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'dirty_nodes': len(self.dirty_nodes),
            'graph_metrics': {},
            'most_connected_nodes': [],
            'isolated_nodes': []
        }
        
        # Get metrics for each graph type
        for graph_type in GraphType:
            metrics = self.get_graph_metrics(graph_type)
            summary['graph_metrics'][graph_type.value] = metrics.to_dict()
        
        # Find most connected nodes (across all graphs)
        node_connections = defaultdict(int)
        for edge in self.edges.values():
            node_connections[edge.source_id] += 1
            node_connections[edge.target_id] += 1
        
        # Sort by connection count
        sorted_nodes = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)
        summary['most_connected_nodes'] = sorted_nodes[:10]
        
        # Find isolated nodes
        connected_nodes = set(node_connections.keys())
        all_nodes = set(self.nodes.keys())
        isolated = all_nodes - connected_nodes
        summary['isolated_nodes'] = list(isolated)
        
        return summary
    
    def export_graph(self, graph_type: GraphType, format: str = "networkx") -> Any:
        """
        Export a specific graph in various formats
        
        Args:
            graph_type: Which graph to export
            format: Export format ("networkx", "dict", "gexf", "graphml")
            
        Returns:
            Graph in requested format
        """
        graph = self.graphs[graph_type]
        
        if format == "networkx":
            return graph
        elif format == "dict":
            return nx.to_dict_of_dicts(graph)
        elif format == "gexf":
            return nx.generate_gexf(graph)
        elif format == "graphml":
            return nx.generate_graphml(graph)
        else:
            raise ValueError(f"Unsupported export format: {format}")