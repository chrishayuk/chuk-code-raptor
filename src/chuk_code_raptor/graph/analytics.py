#!/usr/bin/env python3
# src/chuk_code_raptor/graph/analytics.py
"""
Code Property Graph Analytics
============================

Advanced analytics and metrics for Code Property Graphs.
Provides centrality analysis, community detection, and quality insights.

Path: chuk_code_raptor/graph/analytics.py
"""

import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import logging

from .core import CodePropertyGraph
from .models import GraphType, GraphMetrics

logger = logging.getLogger(__name__)

class GraphAnalytics:
    """
    Advanced analytics for Code Property Graphs
    
    Provides:
    - Centrality analysis (PageRank, betweenness, closeness)
    - Community detection
    - Quality analysis
    - Architectural insights
    - Change impact prediction
    """
    
    def __init__(self, cpg: CodePropertyGraph):
        self.cpg = cpg
        self._centrality_cache: Dict[GraphType, Dict[str, Dict[str, float]]] = {}
        self._community_cache: Dict[GraphType, Dict[str, int]] = {}
        
    def calculate_centrality_scores(self, graph_type: GraphType, 
                                   force_refresh: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate various centrality scores for nodes
        
        Args:
            graph_type: Which graph to analyze
            force_refresh: Force recalculation even if cached
            
        Returns:
            Dict mapping node_id to centrality scores
        """
        if not force_refresh and graph_type in self._centrality_cache:
            return self._centrality_cache[graph_type]
        
        graph = self.cpg.graphs[graph_type]
        
        if graph.number_of_nodes() == 0:
            return {}
        
        centrality_scores = defaultdict(dict)
        
        try:
            # PageRank - measures global importance
            pagerank = nx.pagerank(graph, weight='weight')
            for node_id, score in pagerank.items():
                centrality_scores[node_id]['pagerank'] = score
            
            # In-degree and out-degree centrality
            in_degree = dict(graph.in_degree())
            out_degree = dict(graph.out_degree())
            
            max_in = max(in_degree.values()) if in_degree else 1
            max_out = max(out_degree.values()) if out_degree else 1
            
            for node_id in graph.nodes():
                centrality_scores[node_id]['in_degree'] = in_degree.get(node_id, 0)
                centrality_scores[node_id]['out_degree'] = out_degree.get(node_id, 0)
                centrality_scores[node_id]['in_degree_normalized'] = in_degree.get(node_id, 0) / max_in
                centrality_scores[node_id]['out_degree_normalized'] = out_degree.get(node_id, 0) / max_out
            
            # Betweenness centrality - measures bridging importance
            if graph.number_of_nodes() < 1000:  # Expensive for large graphs
                betweenness = nx.betweenness_centrality(graph, weight='weight')
                for node_id, score in betweenness.items():
                    centrality_scores[node_id]['betweenness'] = score
            
            # Closeness centrality - measures reachability
            if graph.number_of_nodes() < 500:  # Very expensive for large graphs
                try:
                    closeness = nx.closeness_centrality(graph)
                    for node_id, score in closeness.items():
                        centrality_scores[node_id]['closeness'] = score
                except:
                    logger.warning(f"Could not calculate closeness centrality for {graph_type}")
            
            # Eigenvector centrality - measures influence
            if graph.number_of_nodes() < 1000:
                try:
                    eigenvector = nx.eigenvector_centrality(graph, weight='weight', max_iter=100)
                    for node_id, score in eigenvector.items():
                        centrality_scores[node_id]['eigenvector'] = score
                except:
                    logger.warning(f"Could not calculate eigenvector centrality for {graph_type}")
                    
        except Exception as e:
            logger.error(f"Error calculating centrality for {graph_type}: {e}")
        
        # Update node centrality scores in the CPG
        for node_id, scores in centrality_scores.items():
            if node_id in self.cpg.nodes:
                for metric, score in scores.items():
                    self.cpg.nodes[node_id].centrality_scores[f"{graph_type.value}_{metric}"] = score
        
        # Cache the results
        self._centrality_cache[graph_type] = dict(centrality_scores)
        
        return dict(centrality_scores)
    
    def detect_communities(self, graph_type: GraphType, 
                          algorithm: str = "louvain") -> Dict[str, int]:
        """
        Detect communities/clusters in the graph
        
        Args:
            graph_type: Which graph to analyze
            algorithm: Community detection algorithm ("louvain", "label_propagation", "greedy_modularity")
            
        Returns:
            Dict mapping node_id to community_id
        """
        graph = self.cpg.graphs[graph_type].to_undirected()
        
        if graph.number_of_nodes() < 2:
            return {}
        
        communities = {}
        
        try:
            if algorithm == "louvain":
                try:
                    import networkx.algorithms.community as nx_comm
                    community_generator = nx_comm.louvain_communities(graph, seed=42)
                    for i, community in enumerate(community_generator):
                        for node_id in community:
                            communities[node_id] = i
                except ImportError:
                    logger.warning("Louvain algorithm not available, falling back to label propagation")
                    algorithm = "label_propagation"
            
            if algorithm == "label_propagation":
                try:
                    import networkx.algorithms.community as nx_comm
                    community_generator = nx_comm.label_propagation_communities(graph)
                    for i, community in enumerate(community_generator):
                        for node_id in community:
                            communities[node_id] = i
                except ImportError:
                    logger.warning("Label propagation not available, falling back to greedy modularity")
                    algorithm = "greedy_modularity"
            
            if algorithm == "greedy_modularity":
                try:
                    import networkx.algorithms.community as nx_comm
                    community_generator = nx_comm.greedy_modularity_communities(graph)
                    for i, community in enumerate(community_generator):
                        for node_id in community:
                            communities[node_id] = i
                except ImportError:
                    logger.error("No community detection algorithms available")
                    return {}
        
        except Exception as e:
            logger.error(f"Error detecting communities for {graph_type}: {e}")
            return {}
        
        # Update node community assignments
        for node_id, community_id in communities.items():
            if node_id in self.cpg.nodes:
                self.cpg.nodes[node_id].community_id = f"{graph_type.value}_community_{community_id}"
        
        # Cache the results
        self._community_cache[graph_type] = communities
        
        return communities
    
    def get_architectural_insights(self) -> Dict[str, Any]:
        """
        Get architectural insights from the code structure
        
        Returns:
            Dict with architectural analysis
        """
        insights = {
            'layer_analysis': self._analyze_layers(),
            'coupling_analysis': self._analyze_coupling(),
            'quality_hotspots': self._find_quality_hotspots(),
            'central_components': self._find_central_components(),
            'architectural_violations': self._detect_violations()
        }
        
        return insights
    
    def _analyze_layers(self) -> Dict[str, Any]:
        """Analyze architectural layers"""
        layers = defaultdict(list)
        
        # Group chunks by file path patterns (heuristic layer detection)
        for node_id, node in self.cpg.nodes.items():
            if 'file_path' in node.metadata:
                file_path = node.metadata['file_path']
                
                if '/api/' in file_path or '/controllers/' in file_path:
                    layers['presentation'].append(node_id)
                elif '/services/' in file_path or '/business/' in file_path:
                    layers['business'].append(node_id)
                elif '/db/' in file_path or '/data/' in file_path or '/models/' in file_path:
                    layers['data'].append(node_id)
                elif '/utils/' in file_path or '/helpers/' in file_path:
                    layers['utility'].append(node_id)
                else:
                    layers['other'].append(node_id)
        
        # Analyze layer interactions
        layer_interactions = defaultdict(int)
        
        for edge in self.cpg.edges.values():
            source_layer = self._get_node_layer(edge.source_id, layers)
            target_layer = self._get_node_layer(edge.target_id, layers)
            
            if source_layer != target_layer:
                interaction_key = f"{source_layer} -> {target_layer}"
                layer_interactions[interaction_key] += 1
        
        return {
            'layers': {k: len(v) for k, v in layers.items()},
            'layer_interactions': dict(layer_interactions),
            'layering_score': self._calculate_layering_score(layers)
        }
    
    def _get_node_layer(self, node_id: str, layers: Dict[str, List[str]]) -> str:
        """Get the layer for a node"""
        for layer, nodes in layers.items():
            if node_id in nodes:
                return layer
        return 'unknown'
    
    def _calculate_layering_score(self, layers: Dict[str, List[str]]) -> float:
        """Calculate how well the code follows layered architecture"""
        # Simple heuristic: fewer cross-layer dependencies = better layering
        total_edges = len(self.cpg.edges)
        if total_edges == 0:
            return 1.0
        
        cross_layer_edges = 0
        for edge in self.cpg.edges.values():
            source_layer = self._get_node_layer(edge.source_id, layers)
            target_layer = self._get_node_layer(edge.target_id, layers)
            if source_layer != target_layer:
                cross_layer_edges += 1
        
        return 1.0 - (cross_layer_edges / total_edges)
    
    def _analyze_coupling(self) -> Dict[str, Any]:
        """Analyze coupling between components"""
        coupling_metrics = {}
        
        # Calculate fan-in and fan-out for each node
        for node_id in self.cpg.nodes:
            fan_in = len(self.cpg.get_neighbors(node_id, GraphType.CALL_GRAPH, "in", 1))
            fan_out = len(self.cpg.get_neighbors(node_id, GraphType.CALL_GRAPH, "out", 1))
            
            coupling_metrics[node_id] = {
                'fan_in': fan_in,
                'fan_out': fan_out,
                'total_coupling': fan_in + fan_out
            }
        
        # Find highly coupled components
        highly_coupled = sorted(
            coupling_metrics.items(),
            key=lambda x: x[1]['total_coupling'],
            reverse=True
        )[:10]
        
        return {
            'coupling_metrics': coupling_metrics,
            'highly_coupled_components': highly_coupled,
            'average_coupling': sum(m['total_coupling'] for m in coupling_metrics.values()) / len(coupling_metrics) if coupling_metrics else 0
        }
    
    def _find_quality_hotspots(self) -> Dict[str, Any]:
        """Find quality hotspots (high impact, low quality)"""
        hotspots = []
        
        for node_id, node in self.cpg.nodes.items():
            if node.quality_score < 0.6:  # Low quality threshold
                # Calculate impact (centrality in call graph)
                centrality_scores = node.centrality_scores
                pagerank = centrality_scores.get('call_graph_pagerank', 0.0)
                
                if pagerank > 0.01:  # High impact threshold
                    hotspots.append({
                        'node_id': node_id,
                        'quality_score': node.quality_score,
                        'impact_score': pagerank,
                        'hotspot_score': pagerank / (node.quality_score + 0.1)  # Avoid division by zero
                    })
        
        # Sort by hotspot score
        hotspots.sort(key=lambda x: x['hotspot_score'], reverse=True)
        
        return {
            'quality_hotspots': hotspots[:20],  # Top 20 hotspots
            'hotspot_count': len(hotspots)
        }
    
    def _find_central_components(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find most central components in different graphs"""
        central_components = {}
        
        for graph_type in [GraphType.CALL_GRAPH, GraphType.IMPORT_GRAPH, GraphType.SEMANTIC_GRAPH]:
            centrality_scores = self.calculate_centrality_scores(graph_type)
            
            # Sort by PageRank
            if centrality_scores:
                sorted_nodes = sorted(
                    centrality_scores.items(),
                    key=lambda x: x[1].get('pagerank', 0.0),
                    reverse=True
                )
                
                central_components[graph_type.value] = [
                    {
                        'node_id': node_id,
                        'pagerank': scores.get('pagerank', 0.0),
                        'in_degree': scores.get('in_degree', 0),
                        'out_degree': scores.get('out_degree', 0)
                    }
                    for node_id, scores in sorted_nodes[:10]
                ]
        
        return central_components
    
    def _detect_violations(self) -> List[Dict[str, Any]]:
        """Detect potential architectural violations"""
        violations = []
        
        # Detect circular dependencies
        for graph_type in [GraphType.CALL_GRAPH, GraphType.IMPORT_GRAPH]:
            graph = self.cpg.graphs[graph_type]
            
            try:
                cycles = list(nx.simple_cycles(graph))
                for cycle in cycles[:10]:  # Limit to first 10 cycles
                    violations.append({
                        'type': 'circular_dependency',
                        'graph_type': graph_type.value,
                        'cycle': cycle,
                        'severity': 'high' if len(cycle) <= 3 else 'medium'
                    })
            except:
                pass  # nx.simple_cycles can be expensive for large graphs
        
        # Detect high fan-out (potential God classes/functions)
        for node_id, node in self.cpg.nodes.items():
            out_degree = node.centrality_scores.get('call_graph_out_degree', 0)
            if out_degree > 10:  # Threshold for high fan-out
                violations.append({
                    'type': 'high_fan_out',
                    'node_id': node_id,
                    'fan_out': out_degree,
                    'severity': 'high' if out_degree > 20 else 'medium'
                })
        
        return violations
    
    def predict_change_impact(self, node_id: str, max_hops: int = 3) -> Dict[str, Any]:
        """
        Predict the impact of changing a specific node
        
        Args:
            node_id: Node to analyze
            max_hops: Maximum traversal distance
            
        Returns:
            Impact prediction analysis
        """
        if node_id not in self.cpg.nodes:
            return {'error': 'Node not found'}
        
        impact = self.cpg.get_impact_set(node_id, max_hops)
        
        # Calculate impact severity
        total_affected = len(impact.total_affected)
        
        # Weight by quality of affected nodes
        quality_weighted_impact = 0.0
        for affected_id in impact.total_affected:
            if affected_id in self.cpg.nodes:
                quality = self.cpg.nodes[affected_id].quality_score
                quality_weighted_impact += (1.0 - quality)  # Higher impact for lower quality
        
        # Calculate centrality impact
        centrality_impact = 0.0
        for affected_id in impact.total_affected:
            if affected_id in self.cpg.nodes:
                pagerank = self.cpg.nodes[affected_id].centrality_scores.get('call_graph_pagerank', 0.0)
                centrality_impact += pagerank
        
        # Determine impact severity
        if total_affected > 20 or centrality_impact > 0.1:
            severity = 'high'
        elif total_affected > 10 or centrality_impact > 0.05:
            severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'changed_node': node_id,
            'total_affected_nodes': total_affected,
            'directly_affected': len(impact.directly_affected),
            'transitively_affected': len(impact.transitively_affected),
            'quality_weighted_impact': quality_weighted_impact,
            'centrality_impact': centrality_impact,
            'severity': severity,
            'impact_by_graph': impact.impact_by_graph_type,
            'recommendations': self._generate_impact_recommendations(impact, severity)
        }
    
    def _generate_impact_recommendations(self, impact: 'ChangeImpact', severity: str) -> List[str]:
        """Generate recommendations based on impact analysis"""
        recommendations = []
        
        if severity == 'high':
            recommendations.append("Consider breaking this change into smaller, incremental changes")
            recommendations.append("Implement comprehensive testing for all affected components")
            recommendations.append("Consider creating feature flags for gradual rollout")
        
        if len(impact.directly_affected) > 15:
            recommendations.append("High number of direct dependencies - consider refactoring to reduce coupling")
        
        if len(impact.transitively_affected) > 30:
            recommendations.append("High transitive impact - review architectural boundaries")
        
        if not recommendations:
            recommendations.append("Impact appears manageable with standard testing practices")
        
        return recommendations
    
    def get_quality_trends(self) -> Dict[str, Any]:
        """Analyze quality trends across the codebase"""
        quality_by_file = defaultdict(list)
        quality_by_type = defaultdict(list)
        quality_by_complexity = defaultdict(list)
        
        for node_id, node in self.cpg.nodes.items():
            if 'file_path' in node.metadata:
                quality_by_file[node.metadata['file_path']].append(node.quality_score)
            
            if 'chunk_type' in node.metadata:
                quality_by_type[node.metadata['chunk_type']].append(node.quality_score)
            
            if 'complexity' in node.metadata:
                quality_by_complexity[node.metadata['complexity']].append(node.quality_score)
        
        # Calculate averages
        avg_quality_by_file = {
            file: sum(scores) / len(scores) 
            for file, scores in quality_by_file.items()
        }
        
        avg_quality_by_type = {
            chunk_type: sum(scores) / len(scores)
            for chunk_type, scores in quality_by_type.items()
        }
        
        avg_quality_by_complexity = {
            complexity: sum(scores) / len(scores)
            for complexity, scores in quality_by_complexity.items()
        }
        
        # Find files with quality issues
        low_quality_files = [
            (file, score) for file, score in avg_quality_by_file.items()
            if score < 0.6
        ]
        low_quality_files.sort(key=lambda x: x[1])
        
        return {
            'average_quality_by_file': avg_quality_by_file,
            'average_quality_by_type': avg_quality_by_type,
            'average_quality_by_complexity': avg_quality_by_complexity,
            'low_quality_files': low_quality_files[:10],
            'overall_quality_score': sum(node.quality_score for node in self.cpg.nodes.values()) / len(self.cpg.nodes) if self.cpg.nodes else 0.0
        }
    
    def find_refactoring_opportunities(self) -> Dict[str, Any]:
        """Find potential refactoring opportunities"""
        opportunities = {
            'duplicate_patterns': self._find_duplicate_patterns(),
            'god_classes': self._find_god_classes(),
            'dead_code': self._find_potential_dead_code(),
            'high_coupling': self._find_high_coupling_candidates(),
            'long_methods': self._find_long_methods()
        }
        
        return opportunities
    
    def _find_duplicate_patterns(self) -> List[Dict[str, Any]]:
        """Find chunks with similar patterns (potential duplicates)"""
        pattern_groups = defaultdict(list)
        
        # Group by detected patterns
        for node_id, chunk in self.cpg.chunk_index.items():
            for pattern in chunk.detected_patterns:
                pattern_groups[pattern.pattern_name].append((node_id, pattern.confidence))
        
        duplicates = []
        for pattern_name, chunk_list in pattern_groups.items():
            if len(chunk_list) > 2:  # Potential duplicates
                high_confidence = [
                    (node_id, conf) for node_id, conf in chunk_list 
                    if conf > 0.8
                ]
                if len(high_confidence) > 1:
                    duplicates.append({
                        'pattern': pattern_name,
                        'instances': high_confidence,
                        'refactor_potential': 'high' if len(high_confidence) > 3 else 'medium'
                    })
        
        return duplicates
    
    def _find_god_classes(self) -> List[Dict[str, Any]]:
        """Find potential God classes (high complexity + high coupling)"""
        god_classes = []
        
        for node_id, node in self.cpg.nodes.items():
            if node.metadata.get('chunk_type') == 'class':
                # High out-degree (many dependencies)
                out_degree = node.centrality_scores.get('call_graph_out_degree', 0)
                
                # High complexity
                complexity = node.metadata.get('complexity', 'simple')
                is_complex = complexity in ['complex', 'very_complex']
                
                # Large size
                line_count = node.metadata.get('line_count', 0)
                
                if out_degree > 8 and is_complex and line_count > 50:
                    god_classes.append({
                        'node_id': node_id,
                        'out_degree': out_degree,
                        'complexity': complexity,
                        'line_count': line_count,
                        'god_score': out_degree * (1 if is_complex else 0.5) * (line_count / 100)
                    })
        
        god_classes.sort(key=lambda x: x['god_score'], reverse=True)
        return god_classes[:10]
    
    def _find_potential_dead_code(self) -> List[str]:
        """Find potentially unused code (no incoming edges)"""
        dead_code = []
        
        for node_id, node in self.cpg.nodes.items():
            in_degree = node.centrality_scores.get('call_graph_in_degree', 0)
            
            # No incoming calls and not an entry point
            if in_degree == 0:
                chunk_type = node.metadata.get('chunk_type', '')
                # Skip entry points like main functions or API endpoints
                if chunk_type not in ['function'] or 'main' not in node_id.lower():
                    dead_code.append(node_id)
        
        return dead_code
    
    def _find_high_coupling_candidates(self) -> List[Dict[str, Any]]:
        """Find components with high coupling that could benefit from refactoring"""
        high_coupling = []
        
        for node_id, node in self.cpg.nodes.items():
            fan_in = node.centrality_scores.get('call_graph_in_degree', 0)
            fan_out = node.centrality_scores.get('call_graph_out_degree', 0)
            total_coupling = fan_in + fan_out
            
            if total_coupling > 10:  # High coupling threshold
                high_coupling.append({
                    'node_id': node_id,
                    'fan_in': fan_in,
                    'fan_out': fan_out,
                    'total_coupling': total_coupling,
                    'refactor_priority': 'high' if total_coupling > 20 else 'medium'
                })
        
        high_coupling.sort(key=lambda x: x['total_coupling'], reverse=True)
        return high_coupling[:15]
    
    def _find_long_methods(self) -> List[Dict[str, Any]]:
        """Find long methods that might benefit from decomposition"""
        long_methods = []
        
        for node_id, node in self.cpg.nodes.items():
            chunk_type = node.metadata.get('chunk_type', '')
            line_count = node.metadata.get('line_count', 0)
            
            if chunk_type in ['function', 'method'] and line_count > 30:
                complexity = node.metadata.get('complexity', 'simple')
                long_methods.append({
                    'node_id': node_id,
                    'line_count': line_count,
                    'complexity': complexity,
                    'decomposition_priority': 'high' if line_count > 50 else 'medium'
                })
        
        long_methods.sort(key=lambda x: x['line_count'], reverse=True)
        return long_methods[:10]
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        return {
            'summary': self.cpg.get_summary(),
            'centrality_analysis': {
                graph_type.value: self.calculate_centrality_scores(graph_type)
                for graph_type in [GraphType.CALL_GRAPH, GraphType.IMPORT_GRAPH, GraphType.SEMANTIC_GRAPH]
            },
            'architectural_insights': self.get_architectural_insights(),
            'quality_trends': self.get_quality_trends(),
            'refactoring_opportunities': self.find_refactoring_opportunities(),
            'community_structure': {
                graph_type.value: self.detect_communities(graph_type)
                for graph_type in [GraphType.SEMANTIC_GRAPH, GraphType.QUALITY_GRAPH]
            }
        }