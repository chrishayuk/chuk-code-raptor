#!/usr/bin/env python3
# src/chuk_code_raptor/raptor/builder.py
"""
RAPTOR Builder
==============

High-level builder for constructing RAPTOR hierarchies from SemanticChunks
with integration to Code Property Graphs and incremental updates.

Path: chuk_code_raptor/raptor/builder.py
"""

import logging
from typing import List, Set, Dict, Any, Optional
from datetime import datetime

from chuk_code_raptor.chunking.semantic_chunk import SemanticChunk
from chuk_code_raptor.graph.core import CodePropertyGraph
from .core import RaptorTree
from .models import HierarchyLevel, QueryResult, classify_query_type

logger = logging.getLogger(__name__)

class RaptorBuilder:
    """
    High-level builder for RAPTOR hierarchies
    
    Features:
    - Integration with SemanticChunk fingerprints
    - Code Property Graph integration for relationships
    - Intelligent query routing across hierarchy + graph
    - Incremental updates and change tracking
    """
    
    def __init__(self, cpg: Optional[CodePropertyGraph] = None):
        self.raptor_tree = RaptorTree()
        self.cpg = cpg
        
        # Configuration
        self.enable_cpg_integration = cpg is not None
        
    def build_from_chunks(self, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """
        Build complete RAPTOR hierarchy from SemanticChunks
        
        Args:
            chunks: List of SemanticChunks to process
            
        Returns:
            Build summary with statistics
        """
        start_time = datetime.now()
        logger.info(f"Building RAPTOR hierarchy from {len(chunks)} chunks")
        
        # Build the hierarchy
        stats = self.raptor_tree.build_from_chunks(chunks)
        
        # Integrate with CPG if available
        if self.enable_cpg_integration:
            self._integrate_with_cpg(chunks)
        
        build_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'hierarchy_stats': stats.to_dict(),
            'total_build_time': build_time,
            'cpg_integration': self.enable_cpg_integration,
            'performance_metrics': self._calculate_performance_metrics()
        }
    
    def update_from_changes(self, changed_chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """
        Incrementally update RAPTOR from changed chunks
        
        Args:
            changed_chunks: List of chunks that have changed
            
        Returns:
            Update summary with affected nodes
        """
        start_time = datetime.now()
        logger.info(f"Updating RAPTOR with {len(changed_chunks)} changed chunks")
        
        # Update hierarchy
        affected_raptor_nodes = self.raptor_tree.update_from_changed_chunks(changed_chunks)
        
        # Update CPG if available
        affected_cpg_nodes = set()
        if self.enable_cpg_integration and self.cpg:
            for chunk in changed_chunks:
                self.cpg.add_or_update_chunk(chunk)
            
            # Get CPG impact
            for chunk in changed_chunks:
                impact = self.cpg.get_impact_set(chunk.id)
                affected_cpg_nodes.update(impact.total_affected)
        
        update_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'affected_raptor_nodes': len(affected_raptor_nodes),
            'affected_cpg_nodes': len(affected_cpg_nodes),
            'total_update_time': update_time,
            'efficiency_ratio': len(changed_chunks) / (len(affected_raptor_nodes) + len(affected_cpg_nodes)) if affected_raptor_nodes or affected_cpg_nodes else 1.0
        }
    
    def intelligent_search(self, query: str, max_results: int = 10, 
                          max_tokens: int = 8000) -> Dict[str, Any]:
        """
        Intelligent search combining RAPTOR hierarchy and CPG relationships
        
        Args:
            query: Search query
            max_results: Maximum number of results
            max_tokens: Token budget for results
            
        Returns:
            Search results with metadata
        """
        query_type = classify_query_type(query)
        logger.debug(f"Processing {query_type} query: {query}")
        
        if query_type == 'relationship' and self.enable_cpg_integration:
            return self._relationship_search(query, max_results)
        elif query_type == 'architectural':
            return self._architectural_search(query, max_results)
        elif query_type == 'implementation':
            return self._implementation_search(query, max_results, max_tokens)
        else:
            return self._general_search(query, max_results, max_tokens)
    
    def get_contextual_summary(self, node_id: str, context_type: str = "functional") -> Optional[str]:
        """Get contextual summary for a specific node"""
        from .models import SummaryType
        
        summary_type_map = {
            'functional': SummaryType.FUNCTIONAL,
            'architectural': SummaryType.ARCHITECTURAL,
            'api': SummaryType.API,
            'quality': SummaryType.QUALITY,
            'dependencies': SummaryType.DEPENDENCIES
        }
        
        summary_type = summary_type_map.get(context_type, SummaryType.FUNCTIONAL)
        return self.raptor_tree.get_node_summary(node_id, summary_type)
    
    def get_hierarchical_context(self, chunk_id: str) -> Dict[str, Any]:
        """
        Get complete hierarchical context for a chunk
        
        Args:
            chunk_id: SemanticChunk ID
            
        Returns:
            Hierarchical context with summaries at each level
        """
        # Find the chunk node
        chunk_node_id = self.raptor_tree._get_chunk_node_id(chunk_id)
        if not chunk_node_id:
            return {'error': 'Chunk not found in hierarchy'}
        
        # Get path to root
        hierarchy_path = self.raptor_tree.get_hierarchy_path(chunk_node_id)
        
        context = {
            'chunk_id': chunk_id,
            'hierarchy_path': [],
            'related_chunks': [],
            'architectural_context': {}
        }
        
        # Build context for each level
        for node_id in hierarchy_path:
            if node_id in self.raptor_tree.nodes:
                node = self.raptor_tree.nodes[node_id]
                
                context['hierarchy_path'].append({
                    'node_id': node_id,
                    'level': node.level.value,
                    'title': node.title,
                    'summary': node.primary_summary,
                    'keywords': node.topic_keywords[:5]
                })
        
        # Get related chunks via CPG if available
        if self.enable_cpg_integration and self.cpg:
            related_nodes = self.cpg.get_neighbors(chunk_id, max_distance=2)
            context['related_chunks'] = list(related_nodes)[:10]
            
            # Get architectural context
            impact = self.cpg.get_impact_set(chunk_id, max_hops=2)
            context['architectural_context'] = {
                'impact_scope': len(impact.total_affected),
                'direct_dependencies': len(impact.directly_affected),
                'importance_level': self._assess_importance(chunk_id)
            }
        
        return context
    
    def _integrate_with_cpg(self, chunks: List[SemanticChunk]):
        """Integrate RAPTOR with Code Property Graph"""
        if not self.cpg:
            return
        
        logger.info("Integrating RAPTOR with Code Property Graph")
        
        # Add chunks to CPG
        for chunk in chunks:
            self.cpg.add_or_update_chunk(chunk)
        
        # Add semantic relationships based on hierarchy
        self._add_hierarchical_relationships()
    
    def _add_hierarchical_relationships(self):
        """Add hierarchical relationships to CPG"""
        if not self.cpg:
            return
        
        # Add containment relationships
        for node_id, node in self.raptor_tree.nodes.items():
            for child_id in node.child_ids:
                child_node = self.raptor_tree.nodes.get(child_id)
                if child_node and child_node.level == HierarchyLevel.CHUNK:
                    # Add hierarchical relationship to CPG
                    for chunk_id in child_node.source_chunk_ids:
                        # This would add a "part_of" relationship in the CPG
                        pass
    
    def _relationship_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search focused on code relationships via CPG"""
        if not self.cpg:
            return {'error': 'CPG not available for relationship queries'}
        
        # Extract entities from query (simplified)
        # In practice, use NER or more sophisticated parsing
        entities = [word for word in query.split() if len(word) > 3]
        
        results = []
        for entity in entities[:3]:  # Limit entity search
            # Search for nodes containing the entity
            matching_chunks = []
            for chunk_id, chunk in self.cpg.chunk_index.items():
                if entity.lower() in chunk.content.lower():
                    matching_chunks.append(chunk_id)
            
            # For each match, get its relationships
            for chunk_id in matching_chunks[:2]:  # Top 2 matches per entity
                neighbors = self.cpg.get_neighbors(chunk_id, max_distance=2)
                
                for neighbor_id in list(neighbors)[:3]:  # Top 3 neighbors
                    if neighbor_id in self.cpg.chunk_index:
                        neighbor_chunk = self.cpg.chunk_index[neighbor_id]
                        results.append(QueryResult(
                            node_id=neighbor_id,
                            level=HierarchyLevel.CHUNK,
                            content=neighbor_chunk.content[:300],
                            summary=neighbor_chunk.summary or "No summary available",
                            score=0.8,  # Relationship-based score
                            file_path=neighbor_chunk.file_path
                        ))
        
        return {
            'query_type': 'relationship',
            'results': [r.to_dict() for r in results[:max_results]],
            'total_found': len(results),
            'search_method': 'cpg_traversal'
        }
    
    def _architectural_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search focused on architectural understanding"""
        # Start at module/repository level
        results = self.raptor_tree.search(query, max_results, HierarchyLevel.MODULE)
        
        # If no module results, try repository level
        if not results:
            results = self.raptor_tree.search(query, max_results, HierarchyLevel.REPOSITORY)
        
        return {
            'query_type': 'architectural',
            'results': [r.to_dict() for r in results],
            'total_found': len(results),
            'search_method': 'hierarchical_raptor'
        }
    
    def _implementation_search(self, query: str, max_results: int, max_tokens: int) -> Dict[str, Any]:
        """Search for specific implementation details"""
        # Start at file level, then drill to chunks
        file_results = self.raptor_tree.search(query, max_results // 2, HierarchyLevel.FILE)
        
        # Get detailed implementations for top file matches
        detailed_results = []
        token_budget = max_tokens
        
        for file_result in file_results[:3]:
            if token_budget <= 0:
                break
            
            # Get chunk-level details
            chunk_results = self.raptor_tree.search(query, 3, HierarchyLevel.CHUNK)
            
            for chunk_result in chunk_results:
                if token_budget <= 0:
                    break
                
                content_tokens = len(chunk_result.content.split())
                if content_tokens <= token_budget:
                    detailed_results.append(chunk_result)
                    token_budget -= content_tokens
        
        all_results = file_results + detailed_results
        all_results.sort(key=lambda r: r.score, reverse=True)
        
        return {
            'query_type': 'implementation',
            'results': [r.to_dict() for r in all_results[:max_results]],
            'total_found': len(all_results),
            'tokens_used': max_tokens - token_budget,
            'search_method': 'hierarchical_drill_down'
        }
    
    def _general_search(self, query: str, max_results: int, max_tokens: int) -> Dict[str, Any]:
        """General search across multiple levels"""
        # Multi-level search with token budgeting
        results = []
        token_budget = max_tokens
        
        # Start at file level (good balance of detail and context)
        file_results = self.raptor_tree.search(query, max_results // 2, HierarchyLevel.FILE)
        
        for result in file_results:
            content_tokens = len(result.content.split())
            if content_tokens <= token_budget:
                results.append(result)
                token_budget -= content_tokens
        
        # If we have budget left, get some module-level context
        if token_budget > 500:
            module_results = self.raptor_tree.search(query, 3, HierarchyLevel.MODULE)
            for result in module_results:
                content_tokens = len(result.content.split())
                if content_tokens <= token_budget:
                    results.append(result)
                    token_budget -= content_tokens
        
        # If still have budget, get some chunk details
        if token_budget > 300:
            chunk_results = self.raptor_tree.search(query, 5, HierarchyLevel.CHUNK)
            for result in chunk_results:
                content_tokens = len(result.content.split())
                if content_tokens <= token_budget:
                    results.append(result)
                    token_budget -= content_tokens
        
        # Sort by relevance
        results.sort(key=lambda r: r.score, reverse=True)
        
        return {
            'query_type': 'general',
            'results': [r.to_dict() for r in results[:max_results]],
            'total_found': len(results),
            'tokens_used': max_tokens - token_budget,
            'search_method': 'multi_level_budget'
        }
    
    def _assess_importance(self, chunk_id: str) -> str:
        """Assess the importance of a chunk based on CPG centrality"""
        if not self.cpg or chunk_id not in self.cpg.nodes:
            return 'unknown'
        
        node = self.cpg.nodes[chunk_id]
        pagerank = node.centrality_scores.get('call_graph_pagerank', 0.0)
        
        if pagerank > 0.1:
            return 'critical'
        elif pagerank > 0.05:
            return 'high'
        elif pagerank > 0.01:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for the RAPTOR system"""
        stats = self.raptor_tree.stats
        
        return {
            'hierarchy_depth': max(stats.nodes_by_level.keys()) if stats.nodes_by_level else 0,
            'compression_ratio': stats.compression_ratio,
            'nodes_per_level': stats.nodes_by_level,
            'average_quality': sum(stats.average_quality_by_level.values()) / len(stats.average_quality_by_level) if stats.average_quality_by_level else 0.0,
            'scalability_score': self._calculate_scalability_score()
        }
    
    def _calculate_scalability_score(self) -> float:
        """Calculate how well the hierarchy scales"""
        stats = self.raptor_tree.stats
        
        if not stats.nodes_by_level:
            return 0.0
        
        # Good scalability = logarithmic reduction at each level
        levels = sorted(stats.nodes_by_level.keys())
        ratios = []
        
        for i in range(len(levels) - 1):
            current_count = stats.nodes_by_level[levels[i]]
            next_count = stats.nodes_by_level[levels[i + 1]]
            
            if next_count > 0:
                ratio = current_count / next_count
                ratios.append(ratio)
        
        # Good ratio is 5-20 (reasonable summarization)
        good_ratios = [r for r in ratios if 5 <= r <= 20]
        return len(good_ratios) / len(ratios) if ratios else 0.0
    
    def export_hierarchy_summary(self) -> Dict[str, Any]:
        """Export a comprehensive summary of the RAPTOR hierarchy"""
        stats = self.raptor_tree.stats
        
        # Get sample nodes from each level
        level_samples = {}
        for level in HierarchyLevel:
            level_nodes = self.raptor_tree.nodes_by_level[level]
            if level_nodes:
                sample_node = self.raptor_tree.nodes[level_nodes[0]]
                level_samples[level.value] = {
                    'title': sample_node.title,
                    'summary': sample_node.primary_summary[:200] + "..." if len(sample_node.primary_summary) > 200 else sample_node.primary_summary,
                    'keywords': sample_node.topic_keywords[:5]
                }
        
        return {
            'hierarchy_overview': {
                'total_nodes': stats.total_nodes,
                'total_chunks': stats.total_chunks,
                'build_time': stats.build_time_seconds,
                'compression_ratio': stats.compression_ratio
            },
            'level_breakdown': stats.nodes_by_level,
            'quality_by_level': stats.average_quality_by_level,
            'level_samples': level_samples,
            'performance_metrics': self._calculate_performance_metrics(),
            'integration_status': {
                'cpg_enabled': self.enable_cpg_integration,
                'total_relationships': len(self.cpg.edges) if self.cpg else 0
            }
        }

# Convenience functions

def build_raptor_from_chunks(chunks: List[SemanticChunk], 
                           cpg: Optional[CodePropertyGraph] = None) -> RaptorBuilder:
    """
    Convenience function to build RAPTOR hierarchy from chunks
    
    Args:
        chunks: List of SemanticChunks
        cpg: Optional CodePropertyGraph for integration
        
    Returns:
        Configured RaptorBuilder
    """
    builder = RaptorBuilder(cpg)
    builder.build_from_chunks(chunks)
    return builder

def hybrid_search(raptor_builder: RaptorBuilder, query: str, 
                 max_results: int = 10, max_tokens: int = 8000) -> Dict[str, Any]:
    """
    Convenience function for hybrid RAPTOR + CPG search
    
    Args:
        raptor_builder: Configured RaptorBuilder
        query: Search query
        max_results: Maximum results
        max_tokens: Token budget
        
    Returns:
        Search results with metadata
    """
    return raptor_builder.intelligent_search(query, max_results, max_tokens)

def get_chunk_context(raptor_builder: RaptorBuilder, chunk_id: str) -> Dict[str, Any]:
    """
    Get comprehensive context for a chunk using RAPTOR + CPG
    
    Args:
        raptor_builder: Configured RaptorBuilder
        chunk_id: SemanticChunk ID
        
    Returns:
        Hierarchical and relational context
    """
    return raptor_builder.get_hierarchical_context(chunk_id)