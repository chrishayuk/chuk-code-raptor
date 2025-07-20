#!/usr/bin/env python3
# src/chuk_code_raptor/graph/models.py
"""
Code Property Graph Models
=========================

Core data models for the Code Property Graph system.
Defines nodes, edges, and graph metadata structures.

Path: chuk_code_raptor/graph/models.py
"""

from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json

class GraphType(Enum):
    """Types of graphs in the CPG"""
    CALL_GRAPH = "call_graph"
    DATA_FLOW = "data_flow" 
    CONTROL_FLOW = "control_flow"
    IMPORT_GRAPH = "import_graph"
    SEMANTIC_GRAPH = "semantic_graph"
    QUALITY_GRAPH = "quality_graph"
    HIERARCHY_GRAPH = "hierarchy_graph"

class EdgeType(Enum):
    """Types of edges in the CPG"""
    # Call relationships
    CALLS = "calls"
    CALLED_BY = "called_by"
    
    # Data flow
    DEFINES = "defines"
    USES = "uses"
    DATA_FLOW = "data_flow"
    
    # Control flow
    CONTROL_FLOW = "control_flow"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    
    # Import relationships
    IMPORTS = "imports"
    IMPORTED_BY = "imported_by"
    DEPENDS_ON = "depends_on"
    
    # Semantic relationships
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"
    IMPLEMENTS = "implements"
    OVERRIDES = "overrides"
    
    # Quality relationships
    QUALITY_SIMILAR = "quality_similar"
    REFACTOR_CANDIDATE = "refactor_candidate"
    PATTERN_SIMILAR = "pattern_similar"
    
    # Hierarchy
    CONTAINS = "contains"
    PART_OF = "part_of"

@dataclass
class GraphEdge:
    """Enhanced edge with rich metadata"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    graph_type: GraphType
    weight: float = 1.0
    confidence: float = 1.0
    
    # Context information
    source_line: Optional[int] = None
    target_line: Optional[int] = None
    context: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type.value,
            'graph_type': self.graph_type.value,
            'weight': self.weight,
            'confidence': self.confidence,
            'source_line': self.source_line,
            'target_line': self.target_line,
            'context': self.context,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEdge':
        """Create from dictionary"""
        data = data.copy()
        data['edge_type'] = EdgeType(data['edge_type'])
        data['graph_type'] = GraphType(data['graph_type'])
        if isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

@dataclass
class GraphNode:
    """Enhanced node with rich metadata"""
    node_id: str
    node_type: str = "chunk"
    
    # Graph properties
    in_degree: int = 0
    out_degree: int = 0
    centrality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Clustering information
    cluster_id: Optional[str] = None
    community_id: Optional[str] = None
    
    # Quality and importance
    quality_score: float = 0.0
    importance_score: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'in_degree': self.in_degree,
            'out_degree': self.out_degree,
            'centrality_scores': self.centrality_scores,
            'cluster_id': self.cluster_id,
            'community_id': self.community_id,
            'quality_score': self.quality_score,
            'importance_score': self.importance_score,
            'metadata': self.metadata,
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """Create from dictionary"""
        data = data.copy()
        if isinstance(data['last_updated'], str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

@dataclass
class GraphMetrics:
    """Graph analysis metrics"""
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    is_connected: bool = False
    component_count: int = 0
    average_clustering: float = 0.0
    diameter: Optional[int] = None
    
    # Quality metrics
    average_quality: float = 0.0
    quality_variance: float = 0.0
    high_quality_percentage: float = 0.0
    
    # Complexity metrics
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'node_count': self.node_count,
            'edge_count': self.edge_count,
            'density': self.density,
            'is_connected': self.is_connected,
            'component_count': self.component_count,
            'average_clustering': self.average_clustering,
            'diameter': self.diameter,
            'average_quality': self.average_quality,
            'quality_variance': self.quality_variance,
            'high_quality_percentage': self.high_quality_percentage,
            'complexity_distribution': self.complexity_distribution
        }

@dataclass
class ChangeImpact:
    """Change impact analysis results"""
    changed_nodes: List[str]
    directly_affected: Set[str] = field(default_factory=set)
    transitively_affected: Set[str] = field(default_factory=set)
    impact_by_graph_type: Dict[str, Set[str]] = field(default_factory=dict)
    critical_paths: List[List[str]] = field(default_factory=list)
    quality_impact: Dict[str, float] = field(default_factory=dict)
    
    @property
    def total_affected(self) -> Set[str]:
        """Get total set of affected nodes"""
        return self.directly_affected | self.transitively_affected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'changed_nodes': self.changed_nodes,
            'directly_affected': list(self.directly_affected),
            'transitively_affected': list(self.transitively_affected),
            'impact_by_graph_type': {
                k: list(v) for k, v in self.impact_by_graph_type.items()
            },
            'critical_paths': self.critical_paths,
            'quality_impact': self.quality_impact,
            'total_affected_count': len(self.total_affected)
        }

# Utility functions for relationship classification

def classify_relationship(relationship_type: str) -> tuple[EdgeType, GraphType]:
    """
    Classify a relationship string into EdgeType and GraphType
    
    Args:
        relationship_type: String describing the relationship
        
    Returns:
        Tuple of (EdgeType, GraphType)
    """
    mapping = {
        'calls': (EdgeType.CALLS, GraphType.CALL_GRAPH),
        'called_by': (EdgeType.CALLED_BY, GraphType.CALL_GRAPH),
        'depends_on': (EdgeType.DEPENDS_ON, GraphType.IMPORT_GRAPH),
        'imports': (EdgeType.IMPORTS, GraphType.IMPORT_GRAPH),
        'imported_by': (EdgeType.IMPORTED_BY, GraphType.IMPORT_GRAPH),
        'uses': (EdgeType.USES, GraphType.DATA_FLOW),
        'defines': (EdgeType.DEFINES, GraphType.DATA_FLOW),
        'data_flow': (EdgeType.DATA_FLOW, GraphType.DATA_FLOW),
        'contains': (EdgeType.CONTAINS, GraphType.HIERARCHY_GRAPH),
        'part_of': (EdgeType.PART_OF, GraphType.HIERARCHY_GRAPH),
        'similar_to': (EdgeType.SIMILAR_TO, GraphType.SEMANTIC_GRAPH),
        'related_to': (EdgeType.RELATED_TO, GraphType.SEMANTIC_GRAPH),
        'implements': (EdgeType.IMPLEMENTS, GraphType.SEMANTIC_GRAPH),
        'overrides': (EdgeType.OVERRIDES, GraphType.SEMANTIC_GRAPH),
        'quality_similar': (EdgeType.QUALITY_SIMILAR, GraphType.QUALITY_GRAPH),
        'refactor_candidate': (EdgeType.REFACTOR_CANDIDATE, GraphType.QUALITY_GRAPH),
        'pattern_similar': (EdgeType.PATTERN_SIMILAR, GraphType.QUALITY_GRAPH),
    }
    
    return mapping.get(relationship_type, (EdgeType.RELATED_TO, GraphType.SEMANTIC_GRAPH))

def create_edge_id(source_id: str, target_id: str, edge_type: EdgeType) -> str:
    """Create a unique edge identifier"""
    return f"{source_id}->{target_id}:{edge_type.value}"