#!/usr/bin/env python3
"""
RAPTOR Integration Demo
======================

Comprehensive demonstration of RAPTOR (Recursive Abstractive Processing 
for Tree-Organized Retrieval) integrated with SemanticChunks and CPG.

Shows hierarchical abstractions, intelligent query routing, and scalable search.
"""

import json
from typing import List

# Import RAPTOR modules
from chuk_code_raptor.raptor.builder import RaptorBuilder, build_raptor_from_chunks, hybrid_search
from chuk_code_raptor.raptor.models import HierarchyLevel, classify_query_type

# Import existing infrastructure
from chuk_code_raptor.graph.builder import CPGBuilder
from chuk_code_raptor.chunking.semantic_chunk import (
    SemanticChunk, create_chunk_id, QualityMetric, ContentType, CodePattern
)
from chuk_code_raptor.core.models import ChunkType

def create_large_codebase_simulation() -> List[SemanticChunk]:
    """Create a larger, more realistic codebase for RAPTOR demo"""
    chunks = []
    
    # === Authentication Module ===
    auth_models = SemanticChunk(
        id=create_chunk_id("src/auth/models.py", 1, ChunkType.CLASS, "UserModel"),
        file_path="src/auth/models.py",
        content="""class UserModel:
    def __init__(self, user_id, email, hashed_password):
        self.user_id = user_id
        self.email = email
        self.hashed_password = hashed_password
        self.created_at = datetime.now()
        self.last_login = None
        self.is_active = True
        self.roles = []
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password)
    
    def add_role(self, role):
        if role not in self.roles:
            self.roles.append(role)
    
    def has_permission(self, permission):
        return any(role.has_permission(permission) for role in self.roles)""",
        start_line=1, end_line=18,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    auth_service = SemanticChunk(
        id=create_chunk_id("src/auth/service.py", 1, ChunkType.CLASS, "AuthenticationService"),
        file_path="src/auth/service.py",
        content="""class AuthenticationService:
    def __init__(self, user_repository, token_service, password_policy):
        self.user_repo = user_repository
        self.token_service = token_service
        self.password_policy = password_policy
        self.failed_attempts = {}
        self.lockout_duration = timedelta(minutes=15)
    
    def authenticate(self, email, password):
        if self._is_locked_out(email):
            raise AccountLockedException(f"Account locked until {self._get_lockout_expiry(email)}")
        
        user = self.user_repo.find_by_email(email)
        if not user or not user.check_password(password):
            self._record_failed_attempt(email)
            raise InvalidCredentialsException("Invalid email or password")
        
        self._clear_failed_attempts(email)
        user.last_login = datetime.now()
        self.user_repo.save(user)
        
        return self.token_service.generate_access_token(user)
    
    def register_user(self, email, password, profile_data):
        if not self.password_policy.validate(password):
            raise WeakPasswordException("Password does not meet security requirements")
        
        if self.user_repo.email_exists(email):
            raise EmailAlreadyExistsException("Email address already registered")
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = UserModel(None, email, hashed_password)
        
        return self.user_repo.create(user)""",
        start_line=1, end_line=32,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    # === Payment Module ===
    payment_processor = SemanticChunk(
        id=create_chunk_id("src/payment/processor.py", 1, ChunkType.CLASS, "PaymentProcessor"),
        file_path="src/payment/processor.py",
        content="""class PaymentProcessor:
    def __init__(self, stripe_client, fraud_detector, audit_logger):
        self.stripe = stripe_client
        self.fraud_detector = fraud_detector
        self.audit_logger = audit_logger
        self.retry_policy = ExponentialBackoffRetry(max_attempts=3)
    
    def process_payment(self, payment_request):
        # Fraud detection
        fraud_score = self.fraud_detector.assess(payment_request)
        if fraud_score > 0.8:
            self.audit_logger.log_suspicious_activity(payment_request)
            raise FraudDetectedException(f"High fraud risk: {fraud_score}")
        
        # Payment processing with retry
        return self.retry_policy.execute(
            lambda: self._charge_customer(payment_request)
        )
    
    def _charge_customer(self, payment_request):
        stripe_response = self.stripe.charges.create(
            amount=payment_request.amount_cents,
            currency=payment_request.currency,
            source=payment_request.payment_method_token,
            description=payment_request.description,
            metadata=payment_request.metadata
        )
        
        self.audit_logger.log_payment_processed(payment_request, stripe_response)
        return PaymentResult.from_stripe_response(stripe_response)""",
        start_line=1, end_line=27,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    # === API Gateway Module ===
    api_gateway = SemanticChunk(
        id=create_chunk_id("src/gateway/middleware.py", 1, ChunkType.CLASS, "AuthenticationMiddleware"),
        file_path="src/gateway/middleware.py",
        content="""class AuthenticationMiddleware:
    def __init__(self, auth_service, rate_limiter, cors_policy):
        self.auth_service = auth_service
        self.rate_limiter = rate_limiter
        self.cors_policy = cors_policy
        self.exempt_paths = ['/health', '/metrics', '/auth/login']
    
    def process_request(self, request):
        # CORS handling
        if request.method == 'OPTIONS':
            return self.cors_policy.create_preflight_response()
        
        # Rate limiting
        if not self.rate_limiter.allow_request(request.remote_addr):
            raise RateLimitExceededException("Too many requests")
        
        # Authentication bypass for exempt paths
        if request.path in self.exempt_paths:
            return None
        
        # Extract and validate token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise UnauthorizedException("Missing or invalid authorization header")
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        try:
            user_id = self.auth_service.validate_token(token)
            request.current_user_id = user_id
            return None
        except InvalidTokenException:
            raise UnauthorizedException("Invalid or expired token")""",
        start_line=1, end_line=29,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    # === Data Access Layer ===
    user_repository = SemanticChunk(
        id=create_chunk_id("src/data/repositories/user_repository.py", 1, ChunkType.CLASS, "UserRepository"),
        file_path="src/data/repositories/user_repository.py",
        content="""class UserRepository:
    def __init__(self, database_connection, cache_manager):
        self.db = database_connection
        self.cache = cache_manager
        self.table_name = 'users'
    
    def find_by_email(self, email):
        cache_key = f"user:email:{email}"
        cached_user = self.cache.get(cache_key)
        if cached_user:
            return UserModel.from_dict(cached_user)
        
        query = f"SELECT * FROM {self.table_name} WHERE email = ? AND is_active = TRUE"
        row = self.db.fetch_one(query, [email])
        
        if row:
            user = UserModel.from_row(row)
            self.cache.set(cache_key, user.to_dict(), ttl=300)
            return user
        
        return None
    
    def create(self, user):
        query = f"""
        INSERT INTO {self.table_name} 
        (email, hashed_password, created_at, is_active) 
        VALUES (?, ?, ?, ?)
        """
        
        user_id = self.db.execute_returning_id(
            query, 
            [user.email, user.hashed_password, user.created_at, user.is_active]
        )
        
        user.user_id = user_id
        self._invalidate_cache(user.email)
        return user""",
        start_line=1, end_line=32,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    # === Business Logic - Order Processing ===
    order_service = SemanticChunk(
        id=create_chunk_id("src/business/order_service.py", 1, ChunkType.CLASS, "OrderService"),
        file_path="src/business/order_service.py",
        content="""class OrderService:
    def __init__(self, inventory_service, payment_processor, notification_service):
        self.inventory = inventory_service
        self.payment = payment_processor
        self.notifications = notification_service
        self.order_repository = OrderRepository()
    
    def create_order(self, user_id, cart_items, shipping_address):
        # Validate inventory availability
        for item in cart_items:
            if not self.inventory.is_available(item.product_id, item.quantity):
                raise InsufficientInventoryException(f"Not enough {item.product_id} in stock")
        
        # Calculate total and create order
        total_amount = sum(item.price * item.quantity for item in cart_items)
        order = Order(
            user_id=user_id,
            items=cart_items,
            total_amount=total_amount,
            shipping_address=shipping_address,
            status=OrderStatus.PENDING
        )
        
        order = self.order_repository.save(order)
        
        # Reserve inventory
        for item in cart_items:
            self.inventory.reserve(item.product_id, item.quantity, order.order_id)
        
        # Process payment
        try:
            payment_result = self.payment.process_payment(
                PaymentRequest.from_order(order)
            )
            order.payment_id = payment_result.payment_id
            order.status = OrderStatus.PAID
            
        except PaymentFailedException as e:
            # Release reserved inventory
            self._release_inventory_reservation(order)
            order.status = OrderStatus.PAYMENT_FAILED
            raise OrderProcessingException(f"Payment failed: {e}")
        
        # Send confirmation
        self.notifications.send_order_confirmation(order)
        return self.order_repository.save(order)""",
        start_line=1, end_line=40,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.CLASS
    )
    
    # === API Endpoints ===
    user_api = SemanticChunk(
        id=create_chunk_id("src/api/user_controller.py", 1, ChunkType.FUNCTION, "create_user"),
        file_path="src/api/user_controller.py",
        content="""@router.post('/users')
@rate_limit(requests_per_minute=5)
def create_user(request: CreateUserRequest):
    try:
        # Validate input
        if not request.email or not request.password:
            return error_response("Email and password are required", 400)
        
        # Register user
        auth_service = get_auth_service()
        user = auth_service.register_user(
            email=request.email,
            password=request.password,
            profile_data=request.profile
        )
        
        # Generate welcome token
        welcome_token = auth_service.generate_welcome_token(user)
        
        # Send welcome email
        notification_service = get_notification_service()
        notification_service.send_welcome_email(user, welcome_token)
        
        return success_response({
            'user_id': user.user_id,
            'email': user.email,
            'message': 'User created successfully. Check email for verification.'
        }, 201)
        
    except EmailAlreadyExistsException:
        return error_response("Email address already registered", 409)
    except WeakPasswordException as e:
        return error_response(f"Password validation failed: {e}", 400)
    except Exception as e:
        logger.error(f"User creation failed: {e}", exc_info=True)
        return error_response("Internal server error", 500)""",
        start_line=1, end_line=31,
        content_type=ContentType.CODE,
        language="python",
        chunk_type=ChunkType.FUNCTION
    )
    
    chunks = [auth_models, auth_service, payment_processor, api_gateway, 
              user_repository, order_service, user_api]
    
    # Add relationships
    auth_service.add_relationship(
        auth_models.id, "depends_on", strength=0.9,
        context="user model operations", line_number=13
    )
    
    order_service.add_relationship(
        payment_processor.id, "depends_on", strength=0.85,
        context="payment processing", line_number=24
    )
    
    user_api.add_relationship(
        auth_service.id, "calls", strength=0.9,
        context="user registration", line_number=9
    )
    
    api_gateway.add_relationship(
        auth_service.id, "depends_on", strength=0.8,
        context="token validation", line_number=26
    )
    
    # Add comprehensive semantic tags
    tag_mappings = [
        (auth_models, ["authentication", "user-model", "security", "data-model"]),
        (auth_service, ["authentication", "security", "business-logic", "service-layer"]),
        (payment_processor, ["payment", "financial", "fraud-detection", "external-integration"]),
        (api_gateway, ["middleware", "security", "rate-limiting", "cors"]),
        (user_repository, ["data-access", "caching", "persistence", "repository-pattern"]),
        (order_service, ["business-logic", "order-processing", "inventory", "workflow"]),
        (user_api, ["rest-api", "endpoint", "user-management", "validation"])
    ]
    
    for chunk, tags in tag_mappings:
        for tag in tags:
            chunk.add_semantic_tag(tag, confidence=0.9, source="analysis")
    
    # Add quality scores
    quality_mappings = [
        (auth_models, 0.85, 0.80, 0.88),
        (auth_service, 0.82, 0.85, 0.90),
        (payment_processor, 0.88, 0.82, 0.85),
        (api_gateway, 0.80, 0.85, 0.82),
        (user_repository, 0.90, 0.88, 0.92),
        (order_service, 0.78, 0.80, 0.85),
        (user_api, 0.75, 0.78, 0.80)
    ]
    
    for chunk, maintainability, readability, coherence in quality_mappings:
        chunk.set_quality_score(QualityMetric.MAINTAINABILITY, maintainability)
        chunk.set_quality_score(QualityMetric.READABILITY, readability)
        chunk.set_quality_score(QualityMetric.SEMANTIC_COHERENCE, coherence)
    
    # Add embeddings (mock)
    base_embeddings = {
        'auth': [0.2, 0.8, 0.3] * 100,
        'payment': [0.7, 0.4, 0.6] * 100,
        'api': [0.5, 0.6, 0.4] * 100,
        'data': [0.3, 0.7, 0.5] * 100
    }
    
    embedding_mappings = [
        (auth_models, 'auth'), (auth_service, 'auth'),
        (payment_processor, 'payment'), (order_service, 'payment'),
        (api_gateway, 'api'), (user_api, 'api'),
        (user_repository, 'data')
    ]
    
    for chunk, embedding_type in embedding_mappings:
        base = base_embeddings[embedding_type]
        embedding = [x + (hash(chunk.id) % 10) / 200 for x in base]
        chunk.set_embedding(embedding, "text-embedding-ada-002", 1)
    
    return chunks

def demo_raptor_construction():
    """Demo RAPTOR hierarchy construction"""
    print("="*70)
    print(" RAPTOR HIERARCHY CONSTRUCTION")
    print("="*70)
    
    chunks = create_large_codebase_simulation()
    print(f"Created codebase simulation with {len(chunks)} chunks")
    
    # Build CPG first
    cpg_builder = CPGBuilder()
    cpg = cpg_builder.build_from_chunks(chunks)
    print(f"Built CPG: {len(cpg.nodes)} nodes, {len(cpg.edges)} edges")
    
    # Build RAPTOR with CPG integration
    raptor_builder = RaptorBuilder(cpg)
    build_summary = raptor_builder.build_from_chunks(chunks)
    
    print(f"Built RAPTOR hierarchy:")
    hierarchy_stats = build_summary['hierarchy_stats']
    print(f"  Total nodes: {hierarchy_stats['total_nodes']}")
    print(f"  Build time: {build_summary['total_build_time']:.2f}s")
    print(f"  Compression ratio: {hierarchy_stats['compression_ratio']:.3f}")
    
    print("\n--- Hierarchy Levels ---")
    for level, count in hierarchy_stats['nodes_by_level'].items():
        level_name = HierarchyLevel(level).name
        print(f"  Level {level} ({level_name}): {count} nodes")
    
    return raptor_builder, chunks

def demo_intelligent_query_routing(raptor_builder):
    """Demo intelligent query routing based on query type"""
    print("\n" + "="*70)
    print(" INTELLIGENT QUERY ROUTING")
    print("="*70)
    
    queries = [
        ("How does the authentication system work?", "architectural"),
        ("Show me the payment processing implementation", "implementation"),
        ("What calls the UserRepository?", "relationship"),
        ("What's the API for user creation?", "api"),
        ("Where are the quality issues?", "quality")
    ]
    
    for query, expected_type in queries:
        print(f"\n--- Query: '{query}' ---")
        
        detected_type = classify_query_type(query)
        print(f"Detected type: {detected_type} (expected: {expected_type})")
        
        # Perform search
        results = raptor_builder.intelligent_search(query, max_results=3)
        
        print(f"Search method: {results['search_method']}")
        print(f"Results found: {results['total_found']}")
        
        for i, result in enumerate(results['results'][:2]):
            level_name = HierarchyLevel(result['level']).name
            print(f"  {i+1}. [{level_name}] {result.get('file_path', 'N/A')}")
            print(f"     Score: {result['score']:.3f}")
            print(f"     Summary: {result['summary'][:100]}...")

def demo_hierarchical_context(raptor_builder, chunks):
    """Demo hierarchical context retrieval"""
    print("\n" + "="*70)
    print(" HIERARCHICAL CONTEXT RETRIEVAL")
    print("="*70)
    
    # Get context for the AuthenticationService
    auth_service_chunk = next(c for c in chunks if "AuthenticationService" in c.id)
    
    print(f"Getting hierarchical context for: {auth_service_chunk.id}")
    
    context = raptor_builder.get_hierarchical_context(auth_service_chunk.id)
    
    print("\n--- Hierarchy Path ---")
    for level_info in context['hierarchy_path']:
        level_name = HierarchyLevel(level_info['level']).name
        print(f"  Level {level_info['level']} ({level_name}): {level_info['title']}")
        print(f"    Keywords: {', '.join(level_info['keywords'])}")
        if level_info['summary']:
            print(f"    Summary: {level_info['summary'][:150]}...")
    
    print(f"\n--- Related Components ---")
    print(f"Related chunks: {len(context['related_chunks'])}")
    for related_id in context['related_chunks'][:5]:
        print(f"  - {related_id}")
    
    print(f"\n--- Architectural Context ---")
    arch_context = context['architectural_context']
    print(f"  Impact scope: {arch_context['impact_scope']} components")
    print(f"  Direct dependencies: {arch_context['direct_dependencies']}")
    print(f"  Importance level: {arch_context['importance_level']}")

def demo_token_efficient_search(raptor_builder):
    """Demo token-efficient search with budget management"""
    print("\n" + "="*70)
    print(" TOKEN-EFFICIENT SEARCH")
    print("="*70)
    
    queries_and_budgets = [
        ("How does user registration work?", 2000),
        ("Explain the payment processing architecture", 4000),
        ("Show me all authentication-related code", 8000)
    ]
    
    for query, token_budget in queries_and_budgets:
        print(f"\n--- Query: '{query}' (Budget: {token_budget} tokens) ---")
        
        results = raptor_builder.intelligent_search(
            query, max_results=10, max_tokens=token_budget
        )
        
        tokens_used = results.get('tokens_used', 0)
        efficiency = (tokens_used / token_budget) * 100 if token_budget > 0 else 0
        
        print(f"Tokens used: {tokens_used}/{token_budget} ({efficiency:.1f}%)")
        print(f"Results returned: {len(results['results'])}")
        print(f"Search method: {results['search_method']}")
        
        # Show level distribution of results
        level_counts = {}
        for result in results['results']:
            level_name = HierarchyLevel(result['level']).name
            level_counts[level_name] = level_counts.get(level_name, 0) + 1
        
        print(f"Level distribution: {level_counts}")

def demo_incremental_updates(raptor_builder, chunks):
    """Demo incremental RAPTOR updates"""
    print("\n" + "="*70)
    print(" INCREMENTAL RAPTOR UPDATES")
    print("="*70)
    
    # Simulate changes to the PaymentProcessor
    payment_chunk = next(c for c in chunks if "PaymentProcessor" in c.id)
    
    print(f"Original chunk version: {payment_chunk.version}")
    print(f"Original fingerprint: {payment_chunk.combined_fingerprint[:16]}...")
    
    # Modify the chunk
    payment_chunk.content += "\n\n    def refund_payment(self, payment_id, amount):\n        return self.stripe.refunds.create(charge=payment_id, amount=amount)"
    payment_chunk.update_fingerprints()
    
    print(f"Modified version: {payment_chunk.version}")
    print(f"Modified fingerprint: {payment_chunk.combined_fingerprint[:16]}...")
    
    # Update RAPTOR
    update_summary = raptor_builder.update_from_changes([payment_chunk])
    
    print(f"\nUpdate Results:")
    print(f"  Affected RAPTOR nodes: {update_summary['affected_raptor_nodes']}")
    print(f"  Affected CPG nodes: {update_summary['affected_cpg_nodes']}")
    print(f"  Update time: {update_summary['total_update_time']:.3f}s")
    print(f"  Efficiency ratio: {update_summary['efficiency_ratio']:.2f}")

def demo_scalability_analysis(raptor_builder):
    """Demo scalability metrics and performance analysis"""
    print("\n" + "="*70)
    print(" SCALABILITY ANALYSIS")
    print("="*70)
    
    hierarchy_summary = raptor_builder.export_hierarchy_summary()
    
    print("--- Hierarchy Overview ---")
    overview = hierarchy_summary['hierarchy_overview']
    print(f"  Total nodes: {overview['total_nodes']}")
    print(f"  Total chunks: {overview['total_chunks']}")
    print(f"  Build time: {overview['build_time']:.2f}s")
    print(f"  Compression ratio: {overview['compression_ratio']:.3f}")
    
    print("\n--- Performance Metrics ---")
    perf = hierarchy_summary['performance_metrics']
    print(f"  Hierarchy depth: {perf['hierarchy_depth']}")
    print(f"  Scalability score: {perf['scalability_score']:.3f}")
    print(f"  Average quality: {perf['average_quality']:.3f}")
    
    print("\n--- Level Samples ---")
    for level, sample in hierarchy_summary['level_samples'].items():
        level_name = HierarchyLevel(level).name
        print(f"  Level {level} ({level_name}):")
        print(f"    Title: {sample['title']}")
        print(f"    Keywords: {', '.join(sample['keywords'])}")

def main():
    """Run the complete RAPTOR demo"""
    print("RAPTOR (RECURSIVE ABSTRACTIVE PROCESSING) INTEGRATION DEMO")
    print("Showcasing hierarchical abstractions with SemanticChunk + CPG integration")
    
    # Build hierarchy
    raptor_builder, chunks = demo_raptor_construction()
    
    # Demo intelligent features
    demo_intelligent_query_routing(raptor_builder)
    demo_hierarchical_context(raptor_builder, chunks)
    demo_token_efficient_search(raptor_builder)
    demo_incremental_updates(raptor_builder, chunks)
    demo_scalability_analysis(raptor_builder)
    
    print("\n" + "="*70)
    print(" RAPTOR DEMO COMPLETE")
    print("="*70)
    print("RAPTOR capabilities demonstrated:")
    print("✅ Hierarchical abstraction (chunks → files → modules → repository)")
    print("✅ Intelligent query routing based on query type")
    print("✅ Token-efficient search with budget management")
    print("✅ Incremental updates with change propagation")
    print("✅ Integration with SemanticChunk fingerprints")
    print("✅ CPG relationship enhancement")
    print("✅ Scalable architecture (O(log n) memory usage)")
    print("✅ Multi-level context retrieval")
    
    print("\n🚀 RAPTOR enables:")
    print("  • Netflix-scale codebase analysis")
    print("  • Smart query routing (architectural vs implementation)")
    print("  • Token-efficient LLM context")
    print("  • Incremental knowledge updates")
    print("  • Hierarchical code understanding")

if __name__ == "__main__":
    main()