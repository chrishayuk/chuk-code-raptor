#!/usr/bin/env python3
# src/chuk_code_raptor/raptor/core.py
"""
RAPTOR Core Implementation
=========================

Core RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
implementation with incremental updates and SemanticChunk integration.

Path: chuk_code_raptor/raptor/core.py
"""

import networkx as nx
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict
import logging

from chuk_code_raptor.chunking.semantic_chunk import SemanticChunk
from .models import (
    RaptorNode, HierarchyLevel, SummaryType, QueryResult, 
    HierarchyStats, create_raptor_node_id, extract_module_path,
    classify_query_type
)

logger = logging.getLogger(__name__)

class RaptorTree:
    """
    Core RAPTOR implementation with hierarchical abstractions
    
    Features:
    - Multi-level hierarchy from chunks to repository overview
    - Incremental updates with change detection
    - Smart query routing based on abstraction level
    - Integration with SemanticChunk fingerprints
    """
    
    def __init__(self):
        # Core tree structure
        self.tree = nx.DiGraph()
        self.nodes: Dict[str, RaptorNode] = {}
        
        # Level-based indexes for fast access
        self.nodes_by_level: Dict[HierarchyLevel, List[str]] = defaultdict(list)
        self.nodes_by_file: Dict[str, List[str]] = defaultdict(list)
        self.nodes_by_module: Dict[str, List[str]] = defaultdict(list)
        
        # Change tracking
        self.chunk_fingerprints: Dict[str, str] = {}  # chunk_id -> fingerprint
        self.node_hashes: Dict[str, str] = {}  # node_id -> combined_hash
        self.dirty_nodes: Set[str] = set()
        
        # Statistics
        self.stats = HierarchyStats()
        self.version = 1
        
    def build_from_chunks(self, chunks: List[SemanticChunk]) -> HierarchyStats:
        """
        Build the complete RAPTOR hierarchy from SemanticChunks
        
        Args:
            chunks: List of SemanticChunks to process
            
        Returns:
            HierarchyStats with build information
        """
        start_time = datetime.now()
        logger.info(f"Building RAPTOR hierarchy from {len(chunks)} chunks")
        
        # Clear existing structure
        self._reset()
        
        # Level 0: Add raw chunks as leaf nodes
        self._add_chunk_nodes(chunks)
        
        # Level 1: Build file-level summaries
        self._build_file_summaries()
        
        # Level 2: Build module-level summaries
        self._build_module_summaries()
        
        # Level 3: Build repository overview
        self._build_repository_summary()
        
        # Update statistics
        build_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(build_time)
        
        logger.info(f"RAPTOR hierarchy built: {self.stats.total_nodes} nodes in {build_time:.2f}s")
        return self.stats
    
    def update_from_changed_chunks(self, changed_chunks: List[SemanticChunk]) -> Set[str]:
        """
        Incrementally update the hierarchy from changed chunks
        
        Args:
            changed_chunks: List of chunks that have changed
            
        Returns:
            Set of node IDs that were affected
        """
        logger.info(f"Updating RAPTOR hierarchy with {len(changed_chunks)} changed chunks")
        
        affected_nodes = set()
        
        for chunk in changed_chunks:
            # Check if chunk actually changed
            old_fingerprint = self.chunk_fingerprints.get(chunk.id)
            new_fingerprint = chunk.combined_fingerprint
            
            if old_fingerprint == new_fingerprint:
                continue
            
            # Find and update the chunk node
            chunk_node_id = self._get_chunk_node_id(chunk.id)
            if chunk_node_id:
                # Invalidate ancestors up the tree
                ancestors = self._get_ancestors(chunk_node_id)
                affected_nodes.update(ancestors)
                affected_nodes.add(chunk_node_id)
                
                # Update the chunk node
                self._update_chunk_node(chunk)
        
        # Regenerate affected summaries
        self._regenerate_summaries(affected_nodes)
        
        logger.info(f"Updated {len(affected_nodes)} nodes in hierarchy")
        return affected_nodes
    
    def search(self, query: str, max_results: int = 10, 
              target_level: Optional[HierarchyLevel] = None) -> List[QueryResult]:
        """
        Smart search across the hierarchy with automatic level routing
        
        Args:
            query: Search query
            max_results: Maximum number of results
            target_level: Specific level to search (None for auto-routing)
            
        Returns:
            List of QueryResult objects
        """
        if target_level is None:
            target_level = self._route_query(query)
        
        logger.debug(f"Searching at level {target_level} for: {query}")
        
        # Search at the target level
        level_results = self._search_level(query, target_level, max_results)
        
        # If we have room and results suggest drilling down, get more detail
        if len(level_results) < max_results and target_level.value > HierarchyLevel.CHUNK.value:
            detail_level = HierarchyLevel(target_level.value - 1)
            
            # Get detailed results for top matches
            detailed_results = []
            for result in level_results[:3]:  # Top 3 candidates
                child_results = self._get_child_details(result.node_id, detail_level, query)
                detailed_results.extend(child_results[:2])  # 2 details per parent
            
            # Combine and re-rank
            all_results = level_results + detailed_results
            all_results.sort(key=lambda r: r.score, reverse=True)
            return all_results[:max_results]
        
        return level_results
    
    def get_node_summary(self, node_id: str, summary_type: SummaryType = SummaryType.FUNCTIONAL) -> Optional[str]:
        """Get a specific summary for a node"""
        if node_id in self.nodes:
            return self.nodes[node_id].summaries.get(summary_type, self.nodes[node_id].primary_summary)
        return None
    
    def get_hierarchy_path(self, node_id: str) -> List[str]:
        """Get the path from root to the specified node"""
        if node_id not in self.nodes:
            return []
        
        path = []
        current = node_id
        
        while current:
            path.append(current)
            node = self.nodes[current]
            current = node.parent_id
        
        return list(reversed(path))
    
    def get_subtree(self, node_id: str, max_depth: int = 2) -> List[str]:
        """Get all descendants of a node up to max_depth"""
        if node_id not in self.nodes:
            return []
        
        descendants = []
        current_level = {node_id}
        
        for depth in range(max_depth):
            next_level = set()
            for node in current_level:
                if node in self.nodes:
                    children = self.nodes[node].child_ids
                    descendants.extend(children)
                    next_level.update(children)
            current_level = next_level
        
        return descendants
    
    def _reset(self):
        """Reset the tree structure"""
        self.tree.clear()
        self.nodes.clear()
        self.nodes_by_level.clear()
        self.nodes_by_file.clear()
        self.nodes_by_module.clear()
        self.dirty_nodes.clear()
        
    def _add_chunk_nodes(self, chunks: List[SemanticChunk]):
        """Add SemanticChunks as Level 0 nodes"""
        for chunk in chunks:
            node_id = create_raptor_node_id(HierarchyLevel.CHUNK, chunk.id)
            
            node = RaptorNode(
                node_id=node_id,
                level=HierarchyLevel.CHUNK,
                raw_content=chunk.content,
                title=f"{chunk.chunk_type.value}: {chunk.id}",
                file_path=chunk.file_path,
                module_path=extract_module_path(chunk.file_path),
                language=chunk.language,
                source_chunk_ids=[chunk.id],
                topic_keywords=chunk.tag_names,
                importance_score=chunk.importance_score,
                quality_score=chunk.calculate_overall_quality_score()
            )
            
            # Add functional summary from chunk
            if chunk.summary:
                node.add_summary(SummaryType.FUNCTIONAL, chunk.summary)
            
            self._add_node(node)
            self.chunk_fingerprints[chunk.id] = chunk.combined_fingerprint
    
    def _build_file_summaries(self):
        """Build Level 1 file summaries"""
        # Group chunks by file
        files_chunks = defaultdict(list)
        for node_id in self.nodes_by_level[HierarchyLevel.CHUNK]:
            node = self.nodes[node_id]
            if node.file_path:
                files_chunks[node.file_path].append(node)
        
        for file_path, chunk_nodes in files_chunks.items():
            if len(chunk_nodes) == 1:
                # Single chunk file - just promote the chunk
                self._promote_single_chunk_to_file(chunk_nodes[0], file_path)
            else:
                # Multi-chunk file - create summary
                self._create_file_summary_node(file_path, chunk_nodes)
    
    def _create_file_summary_node(self, file_path: str, chunk_nodes: List[RaptorNode]):
        """Create a file-level summary node"""
        from pathlib import Path
        
        file_stem = Path(file_path).stem
        node_id = create_raptor_node_id(HierarchyLevel.FILE, file_stem, file_path)
        
        # Combine chunk content for summarization
        combined_content = "\n\n".join([
            f"## {node.title}\n{node.raw_content[:500]}..." 
            if len(node.raw_content) > 500 else f"## {node.title}\n{node.raw_content}"
            for node in chunk_nodes
        ])
        
        # Create summary (in real implementation, use LLM)
        file_summary = self._generate_file_summary(file_path, chunk_nodes)
        
        # Aggregate metadata
        all_keywords = []
        total_quality = 0.0
        for chunk_node in chunk_nodes:
            all_keywords.extend(chunk_node.topic_keywords)
            total_quality += chunk_node.quality_score
        
        # Remove duplicates and get top keywords
        unique_keywords = list(set(all_keywords))[:10]
        avg_quality = total_quality / len(chunk_nodes)
        
        file_node = RaptorNode(
            node_id=node_id,
            level=HierarchyLevel.FILE,
            raw_content=combined_content,
            title=f"File: {Path(file_path).name}",
            file_path=file_path,
            module_path=extract_module_path(file_path),
            language=chunk_nodes[0].language,
            source_chunk_ids=[cid for node in chunk_nodes for cid in node.source_chunk_ids],
            topic_keywords=unique_keywords,
            quality_score=avg_quality,
            child_ids=[node.node_id for node in chunk_nodes]
        )
        
        file_node.add_summary(SummaryType.FUNCTIONAL, file_summary)
        self._add_node(file_node)
        
        # Set parent relationships
        for chunk_node in chunk_nodes:
            chunk_node.parent_id = node_id
            self.tree.add_edge(node_id, chunk_node.node_id)
    
    def _promote_single_chunk_to_file(self, chunk_node: RaptorNode, file_path: str):
        """Promote a single chunk to represent the entire file"""
        from pathlib import Path
        
        file_stem = Path(file_path).stem
        new_node_id = create_raptor_node_id(HierarchyLevel.FILE, file_stem, file_path)
        
        # Create new file-level node based on chunk
        file_node = RaptorNode(
            node_id=new_node_id,
            level=HierarchyLevel.FILE,
            raw_content=chunk_node.raw_content,
            title=f"File: {Path(file_path).name}",
            file_path=file_path,
            module_path=extract_module_path(file_path),
            language=chunk_node.language,
            source_chunk_ids=chunk_node.source_chunk_ids,
            topic_keywords=chunk_node.topic_keywords,
            quality_score=chunk_node.quality_score,
            child_ids=[chunk_node.node_id]
        )
        
        # Copy summaries
        file_node.summaries = chunk_node.summaries.copy()
        file_node.primary_summary = chunk_node.primary_summary
        
        self._add_node(file_node)
        
        # Update relationships
        chunk_node.parent_id = new_node_id
        self.tree.add_edge(new_node_id, chunk_node.node_id)
    
    def _build_module_summaries(self):
        """Build Level 2 module summaries"""
        # Group files by module
        modules_files = defaultdict(list)
        for node_id in self.nodes_by_level[HierarchyLevel.FILE]:
            node = self.nodes[node_id]
            if node.module_path:
                modules_files[node.module_path].append(node)
        
        for module_path, file_nodes in modules_files.items():
            if len(file_nodes) == 1:
                # Single file module - promote to module level
                self._promote_single_file_to_module(file_nodes[0], module_path)
            else:
                # Multi-file module - create summary
                self._create_module_summary_node(module_path, file_nodes)
    
    def _create_module_summary_node(self, module_path: str, file_nodes: List[RaptorNode]):
        """Create a module-level summary node"""
        module_name = module_path.split('.')[-1]
        node_id = create_raptor_node_id(HierarchyLevel.MODULE, module_name)
        
        # Generate module summary
        module_summary = self._generate_module_summary(module_path, file_nodes)
        
        # Aggregate metadata
        all_keywords = []
        total_quality = 0.0
        all_source_chunks = []
        
        for file_node in file_nodes:
            all_keywords.extend(file_node.topic_keywords)
            total_quality += file_node.quality_score
            all_source_chunks.extend(file_node.source_chunk_ids)
        
        unique_keywords = list(set(all_keywords))[:15]
        avg_quality = total_quality / len(file_nodes)
        
        module_node = RaptorNode(
            node_id=node_id,
            level=HierarchyLevel.MODULE,
            title=f"Module: {module_path}",
            module_path=module_path,
            language=file_nodes[0].language,
            source_chunk_ids=all_source_chunks,
            topic_keywords=unique_keywords,
            quality_score=avg_quality,
            child_ids=[node.node_id for node in file_nodes]
        )
        
        module_node.add_summary(SummaryType.FUNCTIONAL, module_summary)
        self._add_node(module_node)
        
        # Set parent relationships
        for file_node in file_nodes:
            file_node.parent_id = node_id
            self.tree.add_edge(node_id, file_node.node_id)
    
    def _promote_single_file_to_module(self, file_node: RaptorNode, module_path: str):
        """Promote a single file to represent the entire module"""
        module_name = module_path.split('.')[-1]
        new_node_id = create_raptor_node_id(HierarchyLevel.MODULE, module_name)
        
        module_node = RaptorNode(
            node_id=new_node_id,
            level=HierarchyLevel.MODULE,
            raw_content=file_node.raw_content,
            title=f"Module: {module_path}",
            module_path=module_path,
            language=file_node.language,
            source_chunk_ids=file_node.source_chunk_ids,
            topic_keywords=file_node.topic_keywords,
            quality_score=file_node.quality_score,
            child_ids=[file_node.node_id]
        )
        
        # Copy summaries
        module_node.summaries = file_node.summaries.copy()
        module_node.primary_summary = file_node.primary_summary
        
        self._add_node(module_node)
        
        # Update relationships
        file_node.parent_id = new_node_id
        self.tree.add_edge(new_node_id, file_node.node_id)
    
    def _build_repository_summary(self):
        """Build Level 3 repository overview"""
        module_nodes = [self.nodes[node_id] for node_id in self.nodes_by_level[HierarchyLevel.MODULE]]
        
        if not module_nodes:
            return
        
        node_id = create_raptor_node_id(HierarchyLevel.REPOSITORY, "overview")
        
        # Generate repository summary
        repo_summary = self._generate_repository_summary(module_nodes)
        
        # Aggregate repository-level metadata
        all_keywords = []
        total_quality = 0.0
        all_source_chunks = []
        all_languages = set()
        
        for module_node in module_nodes:
            all_keywords.extend(module_node.topic_keywords)
            total_quality += module_node.quality_score
            all_source_chunks.extend(module_node.source_chunk_ids)
            if module_node.language:
                all_languages.add(module_node.language)
        
        unique_keywords = list(set(all_keywords))[:20]
        avg_quality = total_quality / len(module_nodes)
        primary_language = max(all_languages, key=lambda lang: sum(1 for node in module_nodes if node.language == lang))
        
        repo_node = RaptorNode(
            node_id=node_id,
            level=HierarchyLevel.REPOSITORY,
            title="Repository Overview",
            language=primary_language,
            source_chunk_ids=all_source_chunks,
            topic_keywords=unique_keywords,
            quality_score=avg_quality,
            child_ids=[node.node_id for node in module_nodes]
        )
        
        repo_node.add_summary(SummaryType.FUNCTIONAL, repo_summary)
        repo_node.add_summary(SummaryType.ARCHITECTURAL, self._generate_architectural_summary(module_nodes))
        
        self._add_node(repo_node)
        
        # Set parent relationships
        for module_node in module_nodes:
            module_node.parent_id = node_id
            self.tree.add_edge(node_id, module_node.node_id)
    
    def _add_node(self, node: RaptorNode):
        """Add a node to the tree and indexes"""
        self.nodes[node.node_id] = node
        self.tree.add_node(node.node_id)
        
        # Update indexes
        self.nodes_by_level[node.level].append(node.node_id)
        if node.file_path:
            self.nodes_by_file[node.file_path].append(node.node_id)
        if node.module_path:
            self.nodes_by_module[node.module_path].append(node.node_id)
        
        self.node_hashes[node.node_id] = node.combined_hash
    
    # Placeholder methods for LLM integration (implement with actual LLM)
    def _generate_file_summary(self, file_path: str, chunk_nodes: List[RaptorNode]) -> str:
        """Generate file summary (placeholder - implement with LLM)"""
        from pathlib import Path
        
        file_name = Path(file_path).name
        chunk_types = [node.title.split(':')[0] for node in chunk_nodes]
        unique_types = list(set(chunk_types))
        
        return f"File {file_name} contains {len(chunk_nodes)} components: {', '.join(unique_types)}. " \
               f"Primary functionality includes: {', '.join(chunk_nodes[0].topic_keywords[:3])}."
    
    def _generate_module_summary(self, module_path: str, file_nodes: List[RaptorNode]) -> str:
        """Generate module summary (placeholder - implement with LLM)"""
        file_count = len(file_nodes)
        common_keywords = self._get_common_keywords([node.topic_keywords for node in file_nodes])
        
        return f"Module {module_path} consists of {file_count} files. " \
               f"Core functionality: {', '.join(common_keywords[:5])}. " \
               f"Average quality score: {sum(node.quality_score for node in file_nodes) / file_count:.2f}."
    
    def _generate_repository_summary(self, module_nodes: List[RaptorNode]) -> str:
        """Generate repository summary (placeholder - implement with LLM)"""
        module_count = len(module_nodes)
        primary_language = module_nodes[0].language if module_nodes else "unknown"
        common_keywords = self._get_common_keywords([node.topic_keywords for node in module_nodes])
        
        return f"Repository contains {module_count} modules in {primary_language}. " \
               f"Main domains: {', '.join(common_keywords[:7])}. " \
               f"Overall architecture follows standard patterns with modules for: " \
               f"{', '.join([node.module_path.split('.')[-1] for node in module_nodes[:5]])}."
    
    def _generate_architectural_summary(self, module_nodes: List[RaptorNode]) -> str:
        """Generate architectural summary (placeholder - implement with LLM)"""
        layers = self._identify_architectural_layers(module_nodes)
        return f"Architecture follows layered design: {', '.join(layers)}. " \
               f"Dependencies flow from presentation through business logic to data access layers."
    
    def _get_common_keywords(self, keyword_lists: List[List[str]]) -> List[str]:
        """Get most common keywords across lists"""
        from collections import Counter
        
        all_keywords = [kw for kw_list in keyword_lists for kw in kw_list]
        counter = Counter(all_keywords)
        return [kw for kw, count in counter.most_common(10)]
    
    def _identify_architectural_layers(self, module_nodes: List[RaptorNode]) -> List[str]:
        """Identify architectural layers from module names"""
        layer_keywords = {
            'presentation': ['api', 'web', 'ui', 'controller', 'endpoint'],
            'business': ['service', 'business', 'logic', 'domain', 'core'],
            'data': ['data', 'db', 'database', 'repository', 'dao', 'model']
        }
        
        found_layers = set()
        for node in module_nodes:
            module_name = node.module_path.lower()
            for layer, keywords in layer_keywords.items():
                if any(keyword in module_name for keyword in keywords):
                    found_layers.add(layer)
        
        return list(found_layers)
    
    # Additional methods for search, updates, etc. would continue here...
    def _route_query(self, query: str) -> HierarchyLevel:
        """Route query to appropriate hierarchy level"""
        query_type = classify_query_type(query)
        
        routing_map = {
            'architectural': HierarchyLevel.MODULE,
            'relationship': HierarchyLevel.FILE,
            'implementation': HierarchyLevel.CHUNK,
            'api': HierarchyLevel.FILE,
            'quality': HierarchyLevel.MODULE,
            'general': HierarchyLevel.FILE
        }
        
        return routing_map.get(query_type, HierarchyLevel.FILE)
    
    def _search_level(self, query: str, level: HierarchyLevel, max_results: int) -> List[QueryResult]:
        """Search at a specific hierarchy level (placeholder - implement with embeddings)"""
        # Placeholder implementation - in reality, use vector search
        candidates = self.nodes_by_level[level]
        results = []
        
        query_lower = query.lower()
        
        for node_id in candidates:
            node = self.nodes[node_id]
            search_text = node.get_search_text().lower()
            
            # Simple keyword matching (replace with embedding similarity)
            score = sum(1 for word in query_lower.split() if word in search_text)
            score = score / len(query_lower.split()) if query_lower else 0
            
            if score > 0:
                results.append(QueryResult(
                    node_id=node_id,
                    level=level,
                    content=node.raw_content[:500] + "..." if len(node.raw_content) > 500 else node.raw_content,
                    summary=node.primary_summary,
                    score=score,
                    file_path=node.file_path,
                    module_path=node.module_path
                ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]
    
    def _get_child_details(self, parent_node_id: str, detail_level: HierarchyLevel, query: str) -> List[QueryResult]:
        """Get detailed results from children of a parent node"""
        if parent_node_id not in self.nodes:
            return []
        
        parent_node = self.nodes[parent_node_id]
        child_candidates = [child_id for child_id in parent_node.child_ids 
                           if child_id in self.nodes and self.nodes[child_id].level == detail_level]
        
        results = []
        query_lower = query.lower()
        
        for child_id in child_candidates:
            child_node = self.nodes[child_id]
            search_text = child_node.get_search_text().lower()
            
            score = sum(1 for word in query_lower.split() if word in search_text)
            score = score / len(query_lower.split()) if query_lower else 0
            
            if score > 0:
                results.append(QueryResult(
                    node_id=child_id,
                    level=detail_level,
                    content=child_node.raw_content[:300] + "..." if len(child_node.raw_content) > 300 else child_node.raw_content,
                    summary=child_node.primary_summary,
                    score=score * 0.8,  # Slightly lower score for detail results
                    file_path=child_node.file_path,
                    module_path=child_node.module_path
                ))
        
        return sorted(results, key=lambda r: r.score, reverse=True)
    
    def _update_stats(self, build_time: float):
        """Update hierarchy statistics"""
        self.stats.total_nodes = len(self.nodes)
        self.stats.total_chunks = len(self.nodes_by_level[HierarchyLevel.CHUNK])
        self.stats.build_time_seconds = build_time
        self.stats.last_update_time = datetime.now()
        
        # Nodes by level
        for level in HierarchyLevel:
            self.stats.nodes_by_level[level.value] = len(self.nodes_by_level[level])
        
        # Quality by level
        for level in HierarchyLevel:
            level_nodes = [self.nodes[node_id] for node_id in self.nodes_by_level[level]]
            if level_nodes:
                avg_quality = sum(node.quality_score for node in level_nodes) / len(level_nodes)
                self.stats.average_quality_by_level[level.value] = avg_quality
        
        # Size metrics
        self.stats.total_content_size = sum(len(node.raw_content) for node in self.nodes.values())
        self.stats.total_summary_size = sum(
            len(summary) for node in self.nodes.values() 
            for summary in node.summaries.values()
        )
        
        if self.stats.total_content_size > 0:
            self.stats.compression_ratio = self.stats.total_summary_size / self.stats.total_content_size
    
    def _get_chunk_node_id(self, chunk_id: str) -> Optional[str]:
        """Get the node ID for a chunk"""
        for node_id in self.nodes_by_level[HierarchyLevel.CHUNK]:
            node = self.nodes[node_id]
            if chunk_id in node.source_chunk_ids:
                return node_id
        return None
    
    def _get_ancestors(self, node_id: str) -> Set[str]:
        """Get all ancestors of a node"""
        ancestors = set()
        current = self.nodes.get(node_id)
        
        while current and current.parent_id:
            ancestors.add(current.parent_id)
            current = self.nodes.get(current.parent_id)
        
        return ancestors
    
    def _update_chunk_node(self, chunk: SemanticChunk):
        """Update a chunk node with new chunk data"""
        chunk_node_id = self._get_chunk_node_id(chunk.id)
        if not chunk_node_id:
            return
        
        node = self.nodes[chunk_node_id]
        node.raw_content = chunk.content
        node.topic_keywords = chunk.tag_names
        node.quality_score = chunk.calculate_overall_quality_score()
        
        if chunk.summary:
            node.add_summary(SummaryType.FUNCTIONAL, chunk.summary)
        
        node._update_hashes()
        self.chunk_fingerprints[chunk.id] = chunk.combined_fingerprint
        self.node_hashes[chunk_node_id] = node.combined_hash
    
    def _regenerate_summaries(self, affected_nodes: Set[str]):
        """Regenerate summaries for affected nodes"""
        # Sort nodes by level (bottom-up regeneration)
        sorted_nodes = sorted(affected_nodes, key=lambda nid: self.nodes[nid].level.value)
        
        for node_id in sorted_nodes:
            if node_id not in self.nodes:
                continue
            
            node = self.nodes[node_id]
            
            if node.level == HierarchyLevel.FILE:
                self._regenerate_file_summary(node)
            elif node.level == HierarchyLevel.MODULE:
                self._regenerate_module_summary(node)
            elif node.level == HierarchyLevel.REPOSITORY:
                self._regenerate_repository_summary(node)
    
    def _regenerate_file_summary(self, file_node: RaptorNode):
        """Regenerate file-level summary"""
        child_nodes = [self.nodes[child_id] for child_id in file_node.child_ids if child_id in self.nodes]
        new_summary = self._generate_file_summary(file_node.file_path, child_nodes)
        file_node.add_summary(SummaryType.FUNCTIONAL, new_summary)
    
    def _regenerate_module_summary(self, module_node: RaptorNode):
        """Regenerate module-level summary"""
        child_nodes = [self.nodes[child_id] for child_id in module_node.child_ids if child_id in self.nodes]
        new_summary = self._generate_module_summary(module_node.module_path, child_nodes)
        module_node.add_summary(SummaryType.FUNCTIONAL, new_summary)
    
    def _regenerate_repository_summary(self, repo_node: RaptorNode):
        """Regenerate repository-level summary"""
        child_nodes = [self.nodes[child_id] for child_id in repo_node.child_ids if child_id in self.nodes]
        new_summary = self._generate_repository_summary(child_nodes)
        repo_node.add_summary(SummaryType.FUNCTIONAL, new_summary)