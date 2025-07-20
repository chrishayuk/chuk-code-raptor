#!/usr/bin/env python3
# examples/cpg_demo.py
"""
Code Property Graph Demo
=======================

Comprehensive demonstration of the modular Code Property Graph system.
Shows building, updating, and analyzing graphs from SemanticChunks.

Usage:
    python examples/cpg_demo.py
"""

import json
from typing import List

# Import CPG modules
from chuk_code_raptor.graph.builder import CPGBuilder, create_cpg_from_chunks
from chuk_code_raptor.graph.analytics import GraphAnalytics
from chuk_code_raptor.graph.models import GraphType

# Import SemanticChunk infrastructure
from chuk_code_raptor.chunking.semantic_chunk import (
    SemanticChunk, create_chunk_id, QualityMetric, ContentType, CodePattern
)
from chuk_code_raptor.core.models import ChunkType

def create_sample_codebase() -> List[SemanticChunk]:
    """Create a realistic codebase with complex relationships"""
    chunks = []
    
    # 1. Database Layer
    db_connection = SemanticChunk(
        id=create_chunk_id("src/db/connection.py", 1, ChunkType.CLASS, "DatabaseConnection"),
        file_path="src/db/connection.py",
        content="""class DatabaseConnection:
    def __init__(self, connection_string):
        self.conn = create_connection(connection_string)
        self.transaction_count = 0
        self.connection_pool = ConnectionPool(max_size=20)
    
    def execute(self, query, params=None):
        with self.connection_pool.get_connection() as conn:
            self.transaction_count += 1
            return conn.execute(query, params)
    
    def close(self):
        return self.connection_pool.close_all()""",
        start_line=1, end_line=12,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    db_manager = SemanticChunk(
        id=create_chunk_id("src/db/manager.py", 1, ChunkType.CLASS, "DatabaseManager"),
        file_path="src/db/manager.py", 
        content="""class DatabaseManager:
    def __init__(self, connection):
        self.connection = connection
        self.cache = LRUCache(1000)
        self.query_stats = QueryStats()
        self.retry_policy = RetryPolicy(max_attempts=3)
    
    def execute_query(self, query, params=None):
        cache_key = self._generate_cache_key(query, params)
        if cache_key in self.cache:
            self.query_stats.record_cache_hit()
            return self.cache[cache_key]
        
        result = self.retry_policy.execute(
            lambda: self.connection.execute(query, params)
        )
        self.cache[cache_key] = result
        self.query_stats.record_query(query)
        return result
    
    def bulk_insert(self, table, records):
        return self.connection.execute_many(
            f"INSERT INTO {table} VALUES (?)", records
        )""",
        start_line=1, end_line=22,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    # 2. Business Logic Layer
    user_service = SemanticChunk(
        id=create_chunk_id("src/services/user_service.py", 1, ChunkType.CLASS, "UserService"),
        file_path="src/services/user_service.py",
        content="""class UserService:
    def __init__(self, db_manager, email_service, auth_service):
        self.db = db_manager
        self.email_service = email_service
        self.auth_service = auth_service
        self.user_cache = TTLCache(maxsize=500, ttl=3600)
        self.validation_rules = UserValidationRules()
    
    def create_user(self, user_data):
        self.validation_rules.validate(user_data)
        
        hashed_password = self.auth_service.hash_password(user_data['password'])
        user_data['password'] = hashed_password
        
        user_id = self.db.execute_query(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            [user_data['name'], user_data['email'], user_data['password']]
        )
        
        self.email_service.send_welcome_email(user_data['email'])
        self._invalidate_user_cache()
        return user_id
    
    def get_user(self, user_id):
        if user_id in self.user_cache:
            return self.user_cache[user_id]
        
        user = self.db.execute_query(
            "SELECT * FROM users WHERE id = ?", [user_id]
        )
        
        if user:
            self.user_cache[user_id] = user
        
        return user
    
    def update_user_profile(self, user_id, profile_data):
        self.validation_rules.validate_profile(profile_data)
        
        result = self.db.execute_query(
            "UPDATE users SET profile = ? WHERE id = ?",
            [json.dumps(profile_data), user_id]
        )
        
        self.user_cache.pop(user_id, None)
        return result""",
        start_line=1, end_line=40,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    email_service = SemanticChunk(
        id=create_chunk_id("src/services/email_service.py", 1, ChunkType.CLASS, "EmailService"),
        file_path="src/services/email_service.py",
        content="""class EmailService:
    def __init__(self, smtp_config, template_engine):
        self.smtp_client = SMTPClient(smtp_config)
        self.template_engine = template_engine
        self.rate_limiter = RateLimiter(max_per_minute=100)
        self.delivery_tracker = DeliveryTracker()
    
    def send_welcome_email(self, email_address):
        if not self.rate_limiter.allow_request():
            raise RateLimitExceeded("Email rate limit exceeded")
        
        template = self.template_engine.get_template('welcome')
        content = template.render({'email': email_address})
        
        message_id = self.smtp_client.send(email_address, content)
        self.delivery_tracker.track_send(message_id, email_address)
        return message_id
    
    def send_notification(self, email_address, message):
        if not self.rate_limiter.allow_request():
            raise RateLimitExceeded("Email rate limit exceeded")
            
        message_id = self.smtp_client.send(email_address, message)
        self.delivery_tracker.track_send(message_id, email_address)
        return message_id
    
    def get_delivery_status(self, message_id):
        return self.delivery_tracker.get_status(message_id)""",
        start_line=1, end_line=26,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    auth_service = SemanticChunk(
        id=create_chunk_id("src/services/auth_service.py", 1, ChunkType.CLASS, "AuthService"),
        file_path="src/services/auth_service.py",
        content="""class AuthService:
    def __init__(self, secret_key, token_expiry=3600):
        self.secret_key = secret_key
        self.token_expiry = token_expiry
        self.password_hasher = PasswordHasher()
        self.token_blacklist = TokenBlacklist()
    
    def hash_password(self, password):
        return self.password_hasher.hash(password)
    
    def verify_password(self, password, hash):
        return self.password_hasher.verify(password, hash)
    
    def generate_token(self, user_id):
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token):
        if self.token_blacklist.is_blacklisted(token):
            raise InvalidToken("Token has been revoked")
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload['user_id']
        except jwt.ExpiredSignatureError:
            raise InvalidToken("Token has expired")""",
        start_line=1, end_line=27,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    # 3. API Layer
    user_api = SemanticChunk(
        id=create_chunk_id("src/api/user_api.py", 1, ChunkType.FUNCTION, "create_user_endpoint"),
        file_path="src/api/user_api.py",
        content="""@app.route('/users', methods=['POST'])
@rate_limit(requests_per_minute=10)
@require_auth
def create_user_endpoint():
    try:
        user_data = request.get_json()
        validate_json_schema(user_data, USER_CREATION_SCHEMA)
        
        user_service = get_user_service()
        user_id = user_service.create_user(user_data)
        
        audit_logger.log_user_creation(user_id, request.remote_addr)
        
        return jsonify({
            'user_id': user_id,
            'status': 'created',
            'message': 'User created successfully'
        }), 201
        
    except ValidationError as e:
        return jsonify({'error': f'Validation failed: {str(e)}'}), 400
    except RateLimitExceeded as e:
        return jsonify({'error': 'Rate limit exceeded'}), 429
    except Exception as e:
        logger.error(f'User creation failed: {e}', exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500""",
        start_line=1, end_line=22,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.FUNCTION
    )
    
    get_user_api = SemanticChunk(
        id=create_chunk_id("src/api/user_api.py", 25, ChunkType.FUNCTION, "get_user_endpoint"),
        file_path="src/api/user_api.py",
        content="""@app.route('/users/<int:user_id>', methods=['GET'])
@rate_limit(requests_per_minute=100)
@require_auth
def get_user_endpoint(user_id):
    try:
        current_user = get_current_user()
        if not can_access_user(current_user, user_id):
            return jsonify({'error': 'Access denied'}), 403
        
        user_service = get_user_service()
        user = user_service.get_user(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Remove sensitive information
        safe_user = {k: v for k, v in user.items() if k != 'password'}
        
        return jsonify(safe_user)
        
    except Exception as e:
        logger.error(f'User retrieval failed: {e}', exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500""",
        start_line=25, end_line=46,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.FUNCTION
    )
    
    chunks = [db_connection, db_manager, user_service, email_service, auth_service, user_api, get_user_api]
    
    # Add relationships
    db_manager.add_relationship(
        db_connection.id, "depends_on", strength=0.95,
        context="primary database connection", line_number=3
    )
    
    user_service.add_relationship(
        db_manager.id, "depends_on", strength=0.9,
        context="database operations", line_number=3
    )
    
    user_service.add_relationship(
        email_service.id, "depends_on", strength=0.7,
        context="email notifications", line_number=4
    )
    
    user_service.add_relationship(
        auth_service.id, "depends_on", strength=0.8,
        context="password hashing", line_number=5
    )
    
    user_api.add_relationship(
        user_service.id, "calls", strength=0.85,
        context="business logic delegation", line_number=9
    )
    
    get_user_api.add_relationship(
        user_service.id, "calls", strength=0.85,
        context="user data retrieval", line_number=10
    )
    
    # Add semantic tags
    for chunk in chunks:
        chunk.add_semantic_tag("user-management", confidence=0.9, source="analysis")
        chunk.add_semantic_tag("web-application", confidence=0.8, source="domain")
    
    # Layer-specific tags
    db_connection.add_semantic_tag("database", confidence=0.95, source="ast")
    db_connection.add_semantic_tag("connection-pooling", confidence=0.88)
    
    db_manager.add_semantic_tag("database", confidence=0.95, source="ast")
    db_manager.add_semantic_tag("caching", confidence=0.85, source="pattern")
    db_manager.add_semantic_tag("performance", confidence=0.8)
    db_manager.add_semantic_tag("resilience", confidence=0.75)
    
    user_service.add_semantic_tag("business-logic", confidence=0.92, source="architecture")
    user_service.add_semantic_tag("service-layer", confidence=0.9)
    user_service.add_semantic_tag("validation", confidence=0.85)
    
    email_service.add_semantic_tag("email", confidence=0.95, source="domain")
    email_service.add_semantic_tag("notification", confidence=0.88)
    email_service.add_semantic_tag("rate-limiting", confidence=0.8)
    
    auth_service.add_semantic_tag("authentication", confidence=0.95, source="domain")
    auth_service.add_semantic_tag("security", confidence=0.9)
    auth_service.add_semantic_tag("jwt", confidence=0.85)
    
    user_api.add_semantic_tag("rest-api", confidence=0.95, source="pattern")
    user_api.add_semantic_tag("endpoint", confidence=0.95, source="ast")
    user_api.add_semantic_tag("rate-limiting", confidence=0.8)
    
    get_user_api.add_semantic_tag("rest-api", confidence=0.95, source="pattern")
    get_user_api.add_semantic_tag("endpoint", confidence=0.95, source="ast")
    get_user_api.add_semantic_tag("authorization", confidence=0.85)
    
    # Add detected patterns
    patterns_data = [
        (db_manager, "Manager Pattern", 0.85, ["class name contains 'Manager'", "manages database operations"]),
        (db_manager, "Caching Pattern", 0.8, ["cache implementation", "cache key generation"]),
        (user_service, "Service Layer Pattern", 0.9, ["encapsulates business logic", "coordinates between layers"]),
        (email_service, "Rate Limiting Pattern", 0.75, ["rate limiter implementation", "request throttling"]),
        (auth_service, "Strategy Pattern", 0.7, ["password hashing strategy", "token generation strategy"])
    ]
    
    for chunk, pattern_name, confidence, evidence in patterns_data:
        pattern = CodePattern(
            pattern_name=pattern_name,
            confidence=confidence,
            evidence=evidence,
            category="design_pattern"
        )
        chunk.detected_patterns.append(pattern)
    
    # Add quality scores
    quality_data = [
        (db_connection, 0.82, 0.78, 0.85),
        (db_manager, 0.85, 0.80, 0.88),
        (user_service, 0.80, 0.85, 0.82),
        (email_service, 0.88, 0.82, 0.86),
        (auth_service, 0.90, 0.88, 0.92),
        (user_api, 0.75, 0.70, 0.78),
        (get_user_api, 0.78, 0.75, 0.80)
    ]
    
    for chunk, maintainability, readability, coherence in quality_data:
        chunk.set_quality_score(QualityMetric.MAINTAINABILITY, maintainability)
        chunk.set_quality_score(QualityMetric.READABILITY, readability)
        chunk.set_quality_score(QualityMetric.SEMANTIC_COHERENCE, coherence)
    
    # Add embeddings
    base_embeddings = {
        'database': [0.1, 0.8, 0.2] * 100,
        'service': [0.3, 0.6, 0.4] * 100,
        'api': [0.5, 0.4, 0.6] * 100,
        'auth': [0.7, 0.3, 0.5] * 100
    }
    
    embedding_mapping = [
        (db_connection, 'database'),
        (db_manager, 'database'),
        (user_service, 'service'),
        (email_service, 'service'),
        (auth_service, 'auth'),
        (user_api, 'api'),
        (get_user_api, 'api')
    ]
    
    for chunk, embedding_type in embedding_mapping:
        base = base_embeddings[embedding_type]
        # Add slight variations for uniqueness
        embedding = [x + (hash(chunk.id) % 10) / 200 for x in base]
        chunk.set_embedding(embedding, "text-embedding-ada-002", 1)
    
    return chunks

def demo_cpg_construction():
    """Demo CPG construction with the builder"""
    print("="*70)
    print(" CPG CONSTRUCTION WITH BUILDER")
    print("="*70)
    
    chunks = create_sample_codebase()
    print(f"Created sample codebase with {len(chunks)} chunks")
    
    # Build CPG using the builder
    builder = CPGBuilder()
    cpg = builder.build_from_chunks(
        chunks,
        include_semantic_edges=True,
        include_quality_edges=True,
        include_pattern_edges=True
    )
    
    summary = builder.get_build_summary()
    print(f"Built CPG: {summary['total_nodes']} nodes, {summary['total_edges']} edges")
    
    print("\n--- Relationship Breakdown ---")
    for rel_type, count in summary['relationship_breakdown'].items():
        print(f"  {rel_type}: {count}")
    
    print("\n--- Graph Statistics ---")
    for graph_type, metrics in summary['graph_metrics'].items():
        print(f"  {graph_type}: {metrics['node_count']} nodes, {metrics['edge_count']} edges")
    
    return cpg, chunks

def demo_graph_analytics(cpg):
    """Demo advanced graph analytics"""
    print("\n" + "="*70)
    print(" ADVANCED GRAPH ANALYTICS")
    print("="*70)
    
    analytics = GraphAnalytics(cpg)
    
    # Calculate centrality for call graph
    print("\n--- Centrality Analysis (Call Graph) ---")
    centrality = analytics.calculate_centrality_scores(GraphType.CALL_GRAPH)
    
    # Show top nodes by PageRank
    pagerank_scores = [(node_id, scores.get('pagerank', 0.0)) for node_id, scores in centrality.items()]
    pagerank_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("Top nodes by PageRank:")
    for node_id, score in pagerank_scores[:5]:
        print(f"  {node_id}: {score:.4f}")
    
    # Community detection
    print("\n--- Community Detection (Semantic Graph) ---")
    communities = analytics.detect_communities(GraphType.SEMANTIC_GRAPH)
    
    community_groups = {}
    for node_id, community_id in communities.items():
        if community_id not in community_groups:
            community_groups[community_id] = []
        community_groups[community_id].append(node_id)
    
    for community_id, members in community_groups.items():
        print(f"Community {community_id}: {len(members)} members")
        for member in members[:3]:  # Show first 3 members
            print(f"  - {member}")
        if len(members) > 3:
            print(f"  ... and {len(members) - 3} more")
    
    return analytics

def demo_architectural_insights(analytics):
    """Demo architectural analysis"""
    print("\n" + "="*70)
    print(" ARCHITECTURAL INSIGHTS")
    print("="*70)
    
    insights = analytics.get_architectural_insights()
    
    print("--- Layer Analysis ---")
    layer_info = insights['layer_analysis']
    for layer, count in layer_info['layers'].items():
        print(f"  {layer}: {count} components")
    
    print(f"\nLayering Score: {layer_info['layering_score']:.3f}")
    
    print("\n--- Coupling Analysis ---")
    coupling_info = insights['coupling_analysis']
    print(f"Average Coupling: {coupling_info['average_coupling']:.2f}")
    
    print("\nHighly Coupled Components:")
    for node_id, metrics in coupling_info['highly_coupled_components'][:3]:
        print(f"  {node_id}: total coupling = {metrics['total_coupling']}")
    
    print("\n--- Quality Hotspots ---")
    hotspots = insights['quality_hotspots']['quality_hotspots'][:3]
    for hotspot in hotspots:
        print(f"  {hotspot['node_id']}: quality={hotspot['quality_score']:.2f}, impact={hotspot['impact_score']:.4f}")

def demo_change_impact_analysis(cpg, analytics):
    """Demo change impact analysis"""
    print("\n" + "="*70)
    print(" CHANGE IMPACT ANALYSIS")
    print("="*70)
    
    # Analyze impact of changing the DatabaseManager
    changed_node = "manager:class:DatabaseManager:1"
    impact_analysis = analytics.predict_change_impact(changed_node, max_hops=3)
    
    print(f"Impact analysis for changing: {changed_node}")
    print(f"Total affected nodes: {impact_analysis['total_affected_nodes']}")
    print(f"Directly affected: {impact_analysis['directly_affected']}")
    print(f"Transitively affected: {impact_analysis['transitively_affected']}")
    print(f"Impact severity: {impact_analysis['severity']}")
    
    print("\n--- Recommendations ---")
    for rec in impact_analysis['recommendations']:
        print(f"  • {rec}")

def demo_incremental_updates(cpg, chunks):
    """Demo incremental updates"""
    print("\n" + "="*70)
    print(" INCREMENTAL UPDATES")
    print("="*70)
    
    # Simulate a change to UserService
    user_service = next(c for c in chunks if "UserService" in c.id)
    
    print(f"Original version: {user_service.version}")
    print(f"Original fingerprint: {user_service.combined_fingerprint[:16]}...")
    
    # Modify the chunk
    user_service.content += "\n    def delete_user(self, user_id):\n        return self.db.execute_query('DELETE FROM users WHERE id = ?', [user_id])"
    user_service.update_fingerprints()
    
    print(f"Modified version: {user_service.version}")
    print(f"Modified fingerprint: {user_service.combined_fingerprint[:16]}...")
    
    # Update CPG
    builder = CPGBuilder(cpg)
    affected_nodes = builder.update_from_changed_chunks([user_service])
    
    print(f"Affected nodes: {len(affected_nodes)}")
    for node_id in list(affected_nodes)[:5]:
        print(f"  - {node_id}")

def demo_refactoring_opportunities(analytics):
    """Demo refactoring opportunity detection"""
    print("\n" + "="*70)
    print(" REFACTORING OPPORTUNITIES")
    print("="*70)
    
    opportunities = analytics.find_refactoring_opportunities()
    
    print("--- God Classes ---")
    for god_class in opportunities['god_classes'][:3]:
        print(f"  {god_class['node_id']}: score = {god_class['god_score']:.2f}")
    
    print("\n--- High Coupling Candidates ---")
    for candidate in opportunities['high_coupling'][:3]:
        print(f"  {candidate['node_id']}: coupling = {candidate['total_coupling']}")
    
    print("\n--- Long Methods ---")
    for method in opportunities['long_methods'][:3]:
        print(f"  {method['node_id']}: {method['line_count']} lines")

def main():
    """Run the complete CPG demo"""
    print("MODULAR CODE PROPERTY GRAPH DEMONSTRATION")
    print("Showcasing the new modular architecture")
    
    # Build CPG
    cpg, chunks = demo_cpg_construction()
    
    # Analytics
    analytics = demo_graph_analytics(cpg)
    demo_architectural_insights(analytics)
    demo_change_impact_analysis(cpg, analytics)
    
    # Incremental updates
    demo_incremental_updates(cpg, chunks)
    
    # Refactoring opportunities
    demo_refactoring_opportunities(analytics)
    
    print("\n" + "="*70)
    print(" DEMO COMPLETE")
    print("="*70)
    print("Modular CPG capabilities demonstrated:")
    print("✅ Separated concerns: models, core, builder, analytics")
    print("✅ Incremental updates with SemanticChunk fingerprints")
    print("✅ Advanced analytics and architectural insights")
    print("✅ Change impact prediction")
    print("✅ Refactoring opportunity detection")
    print("✅ Community detection and centrality analysis")
    print("✅ Quality-aware graph construction")

if __name__ == "__main__":
    main()