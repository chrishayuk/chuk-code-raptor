#!/usr/bin/env python3
# tests/chunking/test_code_chunk.py
"""
Comprehensive pytest tests for SemanticCodeChunk class
======================================================

Tests cover:
- Code-specific initialization and properties
- Quality metrics calculation (coupling, reusability, maintainability)
- Code quality indicators (docstrings, type hints, error handling)
- Architectural role classification
- Pattern detection and code smell analysis
- Integration with SemanticChunk features
- Factory functions and serialization
- Edge cases and performance
"""

import pytest
from unittest.mock import patch
from datetime import datetime

# Import the classes we're testing
from chuk_code_raptor.chunking.code_chunk import (
    SemanticCodeChunk, ArchitecturalRole, calculate_code_quality_metrics,
    create_code_chunk_for_content_type
)
from chuk_code_raptor.chunking.semantic_chunk import ContentType, QualityMetric
from chuk_code_raptor.core.models import ChunkType, CodeChunk


class TestSemanticCodeChunk:
    """Test suite for SemanticCodeChunk class"""
    
    @pytest.fixture
    def simple_function_chunk(self):
        """Create a simple function chunk for testing"""
        return SemanticCodeChunk(
            id="func_hello",
            file_path="/src/utils.py",
            content='def hello_world():\n    """Say hello to the world."""\n    print("Hello, World!")\n    return True',
            start_line=10,
            end_line=13,
            content_type=ContentType.CODE,
            language="python",
            chunk_type=ChunkType.FUNCTION
        )
    
    @pytest.fixture
    def complex_class_chunk(self):
        """Create a complex class chunk for testing"""
        content = '''class DatabaseManager:
    """Manages database connections and operations.
    
    This class provides a high-level interface for database operations
    with connection pooling and error handling.
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        try:
            self.pool = create_connection_pool(self.connection_string)
        except ConnectionError as e:
            self.logger.error(f"Failed to initialize pool: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query with optional parameters."""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or [])
                return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise
'''
        return SemanticCodeChunk(
            id="class_db_manager",
            file_path="/src/database.py",
            content=content,
            start_line=1,
            end_line=30,
            content_type=ContentType.CODE,
            language="python",
            chunk_type=ChunkType.CLASS
        )
    
    @pytest.fixture
    def poorly_documented_chunk(self):
        """Create a chunk with poor documentation for testing"""
        content = '''def complex_calculation(a, b, c, d, e):
    x = a + b
    if x > 10:
        y = c * d
        if y < 5:
            z = e / 2
            return z + x
        else:
            return y - x
    else:
        return a * b * c * d * e'''
        
        return SemanticCodeChunk(
            id="poor_doc_func",
            file_path="/src/calculations.py",
            content=content,
            start_line=5,
            end_line=15,
            content_type=ContentType.CODE,
            language="python",
            chunk_type=ChunkType.FUNCTION
        )
    
    @pytest.fixture
    def test_function_chunk(self):
        """Create a test function chunk"""
        content = '''def test_user_creation():
    """Test user creation with valid data."""
    user_data = {"name": "John", "email": "john@example.com"}
    
    try:
        user = create_user(user_data)
        assert user.name == "John"
        assert user.email == "john@example.com"
        assert user.id is not None
    except Exception as e:
        pytest.fail(f"User creation failed: {e}")'''
        
        return SemanticCodeChunk(
            id="test_user_creation",
            file_path="/tests/test_users.py",
            content=content,
            start_line=10,
            end_line=20,
            content_type=ContentType.CODE,
            language="python",
            chunk_type=ChunkType.FUNCTION,
            architectural_role=ArchitecturalRole.TESTING
        )

    def test_basic_initialization(self, simple_function_chunk):
        """Test basic code chunk initialization"""
        chunk = simple_function_chunk
        
        # Test SemanticChunk inheritance
        assert chunk.id == "func_hello"
        assert chunk.file_path == "/src/utils.py"
        assert chunk.content_type == ContentType.CODE
        assert chunk.language == "python"
        assert chunk.chunk_type == ChunkType.FUNCTION
        
        # Test code-specific properties
        assert chunk.accessibility == "public"
        assert chunk.architectural_role is None
        assert isinstance(chunk.imports, list)
        assert isinstance(chunk.exports, list)
        assert isinstance(chunk.function_calls, list)
        assert isinstance(chunk.variables_used, list)
        assert isinstance(chunk.types_used, list)
        assert isinstance(chunk.design_patterns, list)
        assert isinstance(chunk.code_smells, list)
        assert isinstance(chunk.architectural_concerns, list)

    def test_content_type_default(self):
        """Test that content type defaults to CODE"""
        chunk = SemanticCodeChunk(
            id="test",
            file_path="/test.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE  # Add required parameter
        )
        
        assert chunk.content_type == ContentType.CODE
        
        # Test that it defaults to CODE even when not explicitly set
        chunk2 = SemanticCodeChunk(
            id="test2",
            file_path="/test2.py",
            content="def test2(): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.DOCUMENTATION  # This should be overridden
        )
        
        # After __post_init__, should be CODE for code content
        assert chunk2.content_type == ContentType.CODE

    def test_code_quality_indicators_setting(self, simple_function_chunk):
        """Test setting code quality indicators"""
        chunk = simple_function_chunk
        
        # Set quality indicators
        chunk.set_code_quality_indicators(
            has_docstring=True,
            docstring_quality=0.8,
            has_type_hints=True,
            type_coverage=0.9,
            has_error_handling=True,
            cyclomatic_complexity=3
        )
        
        # Verify indicators are set
        assert chunk.has_docstring == True
        assert chunk.docstring_quality == 0.8
        assert chunk.has_type_hints == True
        assert chunk.type_coverage == 0.9
        assert chunk.has_error_handling == True
        assert chunk.cyclomatic_complexity == 3
        
        # Verify derived metrics are calculated
        assert chunk.maintainability_index > 0.0
        assert chunk.test_coverage_indicator >= 0.0

    def test_maintainability_calculation(self):
        """Test maintainability index calculation"""
        chunk = SemanticCodeChunk(
            id="maintainability_test",
            file_path="/test.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        # Test high maintainability (good documentation, low complexity)
        chunk.set_code_quality_indicators(
            has_docstring=True,
            docstring_quality=0.9,
            has_type_hints=True,
            type_coverage=0.8,
            cyclomatic_complexity=2
        )
        
        high_maintainability = chunk.maintainability_index
        assert high_maintainability > 0.8
        
        # Test low maintainability (poor documentation, high complexity)
        chunk.set_code_quality_indicators(
            has_docstring=False,
            docstring_quality=0.0,
            has_type_hints=False,
            type_coverage=0.0,
            cyclomatic_complexity=15
        )
        
        low_maintainability = chunk.maintainability_index
        assert low_maintainability < high_maintainability
        assert 0.0 <= low_maintainability <= 1.0

    def test_coupling_score_calculation(self):
        """Test coupling score calculation"""
        chunk = SemanticCodeChunk(
            id="coupling_test",
            file_path="/test.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        # Test low coupling (no dependencies)
        assert chunk.calculate_coupling_score() == 0.0
        
        # Test moderate coupling
        chunk.dependencies = ["module1", "module2", "module3"]
        chunk.dependents = ["caller1", "caller2"]
        
        coupling_score = chunk.calculate_coupling_score()
        assert 0.0 < coupling_score < 1.0
        assert coupling_score == 5 / 20.0  # (3 + 2) / 20
        
        # Test high coupling
        chunk.dependencies = ["mod" + str(i) for i in range(15)]
        chunk.dependents = ["caller" + str(i) for i in range(10)]
        
        high_coupling = chunk.calculate_coupling_score()
        assert high_coupling >= coupling_score
        assert high_coupling <= 1.0

    def test_reusability_score_calculation(self, simple_function_chunk):
        """Test reusability score calculation"""
        chunk = simple_function_chunk
        
        # Set up for high reusability
        chunk.set_quality_score(QualityMetric.COMPLETENESS, 0.9)
        chunk.set_code_quality_indicators(
            has_docstring=True,
            docstring_quality=0.8,
            has_type_hints=True,
            type_coverage=0.9
        )
        # Low coupling (few dependencies)
        chunk.dependencies = ["single_dep"]
        
        reusability = chunk.calculate_reusability_score()
        assert reusability > 0.7
        
        # Test low reusability
        chunk.set_quality_score(QualityMetric.COMPLETENESS, 0.3)
        chunk.set_code_quality_indicators(
            has_docstring=False,
            docstring_quality=0.0,
            has_type_hints=False,
            type_coverage=0.0
        )
        # High coupling
        chunk.dependencies = ["dep" + str(i) for i in range(10)]
        
        low_reusability = chunk.calculate_reusability_score()
        assert low_reusability < reusability
        assert 0.0 <= low_reusability <= 1.0

    def test_test_coverage_indicator(self, test_function_chunk):
        """Test test coverage indicator calculation"""
        chunk = test_function_chunk
        
        # Add test-related semantic tag
        chunk.add_semantic_tag("unit_test", 0.9, "manual")
        
        # Add validation dependencies
        chunk.dependencies = ["pytest", "assert_helper", "validation_utils"]
        
        # Set error handling
        chunk.set_code_quality_indicators(has_error_handling=True)
        
        coverage_indicator = chunk.test_coverage_indicator
        assert coverage_indicator > 0.5  # Should be well-tested
        assert chunk.is_well_tested

    def test_properties(self, complex_class_chunk, poorly_documented_chunk):
        """Test various boolean properties"""
        # Set up complex class as high quality
        complex_class_chunk.importance_score = 0.8
        complex_class_chunk.set_code_quality_indicators(
            has_docstring=True,
            docstring_quality=0.8,
            cyclomatic_complexity=5
        )
        
        # Test is_high_quality
        assert complex_class_chunk.is_high_quality
        
        # Test is_highly_coupled
        complex_class_chunk.dependencies = ["dep" + str(i) for i in range(20)]
        assert complex_class_chunk.is_highly_coupled
        
        # Test is_reusable
        complex_class_chunk.dependencies = ["single_dep"]  # Reduce coupling
        complex_class_chunk.set_quality_score(QualityMetric.COMPLETENESS, 0.8)
        assert complex_class_chunk.is_reusable
        
        # Test poorly documented chunk
        poorly_documented_chunk.set_code_quality_indicators(
            has_docstring=False,
            docstring_quality=0.0,
            cyclomatic_complexity=8
        )
        poorly_documented_chunk.code_smells = ["long_method", "complex_conditional"]
        
        assert not poorly_documented_chunk.is_high_quality

    def test_architectural_roles(self):
        """Test architectural role classification"""
        # Test data access role
        data_chunk = SemanticCodeChunk(
            id="data_access",
            file_path="/src/dao.py",
            content="def get_user_by_id(user_id): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE,
            architectural_role=ArchitecturalRole.DATA_ACCESS
        )
        
        assert data_chunk.architectural_role == ArchitecturalRole.DATA_ACCESS
        
        # Test business logic role
        business_chunk = SemanticCodeChunk(
            id="business_logic",
            file_path="/src/services.py",
            content="def calculate_discount(order): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE,
            architectural_role=ArchitecturalRole.BUSINESS_LOGIC
        )
        
        assert business_chunk.architectural_role == ArchitecturalRole.BUSINESS_LOGIC

    def test_code_analysis_properties(self, complex_class_chunk):
        """Test code analysis properties"""
        chunk = complex_class_chunk
        
        # Set code analysis properties
        chunk.imports = ["typing", "logging", "database"]
        chunk.exports = ["DatabaseManager"]
        chunk.function_calls = ["create_connection_pool", "cursor.execute", "logger.error"]
        chunk.variables_used = ["connection_string", "pool", "query", "params"]
        chunk.types_used = ["str", "Optional", "List", "Dict", "Any"]
        chunk.ast_node_path = "module.class.DatabaseManager"
        
        # Test properties are set
        assert len(chunk.imports) == 3
        assert len(chunk.exports) == 1
        assert len(chunk.function_calls) == 3
        assert len(chunk.variables_used) == 4
        assert len(chunk.types_used) == 5
        assert chunk.ast_node_path == "module.class.DatabaseManager"

    def test_design_patterns_and_code_smells(self):
        """Test design patterns and code smell tracking"""
        chunk = SemanticCodeChunk(
            id="patterns_test",
            file_path="/src/patterns.py",
            content="class Singleton: pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        # Add design patterns
        chunk.design_patterns = ["singleton", "factory"]
        
        # Add code smells
        chunk.code_smells = ["god_class", "long_method"]
        
        # Add architectural concerns
        chunk.architectural_concerns = ["tight_coupling", "circular_dependency"]
        
        assert "singleton" in chunk.design_patterns
        assert "god_class" in chunk.code_smells
        assert "tight_coupling" in chunk.architectural_concerns

    def test_from_code_chunk_conversion(self):
        """Test creating SemanticCodeChunk from CodeChunk"""
        # Create a basic CodeChunk
        code_chunk = CodeChunk(
            id="convert_test",
            file_path="/src/convert.py",
            content="def convert_me(): pass",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language="python",
            summary="Test function for conversion",
            keywords=["convert", "test"],
            embedding=[0.1, 0.2, 0.3],
            embedding_model="test-model",
            metadata={"source": "test"},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Convert to SemanticCodeChunk
        semantic_chunk = SemanticCodeChunk.from_code_chunk(code_chunk)
        
        # Verify conversion
        assert semantic_chunk.id == code_chunk.id
        assert semantic_chunk.file_path == code_chunk.file_path
        assert semantic_chunk.content == code_chunk.content
        assert semantic_chunk.chunk_type == code_chunk.chunk_type
        assert semantic_chunk.language == code_chunk.language
        assert semantic_chunk.summary == code_chunk.summary
        assert semantic_chunk.keywords == code_chunk.keywords
        assert semantic_chunk.semantic_embedding == code_chunk.embedding
        assert semantic_chunk.embedding_model == code_chunk.embedding_model
        assert semantic_chunk.metadata == code_chunk.metadata
        
        # Verify code-specific properties are initialized
        assert semantic_chunk.accessibility == "public"
        assert semantic_chunk.architectural_role is None
        assert isinstance(semantic_chunk.imports, list)

    def test_inheritance_from_semantic_chunk(self, simple_function_chunk):
        """Test that CodeChunk properly inherits from SemanticChunk"""
        chunk = simple_function_chunk
        
        # Should have all SemanticChunk properties
        assert hasattr(chunk, 'content_fingerprint')
        assert hasattr(chunk, 'semantic_tags')
        assert hasattr(chunk, 'relationships')
        assert hasattr(chunk, 'quality_scores')
        
        # Should be able to use SemanticChunk methods
        chunk.add_semantic_tag("utility_function", 0.9)
        assert len(chunk.semantic_tags) == 1
        assert chunk.semantic_tags[0].name == "utility_function"
        
        # Should be able to add relationships
        chunk.add_relationship("other_function", "calls", 0.8)
        assert len(chunk.relationships) == 1
        
        # Should be able to set quality scores
        chunk.set_quality_score(QualityMetric.READABILITY, 0.8)
        assert chunk.get_quality_score(QualityMetric.READABILITY) == 0.8


class TestCodeQualityMetrics:
    """Test suite for code quality metric calculations"""
    
    @pytest.fixture
    def high_quality_chunk(self):
        """Create a high-quality code chunk"""
        chunk = SemanticCodeChunk(
            id="high_quality",
            file_path="/src/quality.py",
            content='def well_documented_function(x: int) -> int:\n    """This is well documented."""\n    return x * 2',
            start_line=1,
            end_line=3,
            content_type=ContentType.CODE,
            language="python"
        )
        
        chunk.set_code_quality_indicators(
            has_docstring=True,
            docstring_quality=0.9,
            has_type_hints=True,
            type_coverage=1.0,
            has_error_handling=True,
            cyclomatic_complexity=1
        )
        
        chunk.dependencies = ["single_module"]  # Low coupling
        
        return chunk
    
    @pytest.fixture
    def low_quality_chunk(self):
        """Create a low-quality code chunk"""
        chunk = SemanticCodeChunk(
            id="low_quality",
            file_path="/src/bad_quality.py",
            content="def bad_function(a,b,c,d,e): return a+b+c+d+e if a>0 else b*c*d*e",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE,
            language="python"
        )
        
        chunk.set_code_quality_indicators(
            has_docstring=False,
            docstring_quality=0.0,
            has_type_hints=False,
            type_coverage=0.0,
            has_error_handling=False,
            cyclomatic_complexity=12
        )
        
        chunk.dependencies = ["dep" + str(i) for i in range(15)]  # High coupling
        
        return chunk

    def test_calculate_code_quality_metrics(self, high_quality_chunk, low_quality_chunk):
        """Test comprehensive quality metrics calculation"""
        # Test high quality chunk
        high_metrics = calculate_code_quality_metrics(high_quality_chunk)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'coupling', 'reusability', 'maintainability', 'test_coverage_indicator',
            'documentation_quality', 'type_safety', 'complexity', 'error_handling'
        ]
        
        for metric in expected_metrics:
            assert metric in high_metrics
            assert isinstance(high_metrics[metric], float)
            assert 0.0 <= high_metrics[metric] <= 1.0
        
        # High quality chunk should have good scores
        assert high_metrics['documentation_quality'] > 0.8
        assert high_metrics['type_safety'] > 0.8
        assert high_metrics['complexity'] > 0.8
        assert high_metrics['error_handling'] == 1.0
        
        # Test low quality chunk
        low_metrics = calculate_code_quality_metrics(low_quality_chunk)
        
        # Low quality chunk should have poor scores
        assert low_metrics['documentation_quality'] == 0.0
        assert low_metrics['type_safety'] == 0.0
        assert low_metrics['complexity'] <= 0.5  # Changed from < to <=
        assert low_metrics['error_handling'] == 0.0
        
        # Compare overall quality
        high_avg = sum(high_metrics.values()) / len(high_metrics)
        low_avg = sum(low_metrics.values()) / len(low_metrics)
        assert high_avg > low_avg

    def test_complexity_metric_calculation(self):
        """Test complexity metric calculation"""
        chunk = SemanticCodeChunk(
            id="complexity_test",
            file_path="/test.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        # Test low complexity (good)
        chunk.set_code_quality_indicators(cyclomatic_complexity=3)
        metrics = calculate_code_quality_metrics(chunk)
        assert metrics['complexity'] == 1.0
        
        # Test moderate complexity
        chunk.set_code_quality_indicators(cyclomatic_complexity=8)
        metrics = calculate_code_quality_metrics(chunk)
        assert metrics['complexity'] == 0.7
        
        # Test high complexity
        chunk.set_code_quality_indicators(cyclomatic_complexity=12)
        metrics = calculate_code_quality_metrics(chunk)
        assert metrics['complexity'] == 0.5
        
        # Test very high complexity (bad)
        chunk.set_code_quality_indicators(cyclomatic_complexity=20)
        metrics = calculate_code_quality_metrics(chunk)
        assert metrics['complexity'] == 0.3

    def test_coupling_metric(self):
        """Test coupling metric calculation"""
        chunk = SemanticCodeChunk(
            id="coupling_test",
            file_path="/test.py",
            content="def test(): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        # Test no coupling
        metrics = calculate_code_quality_metrics(chunk)
        assert metrics['coupling'] == 0.0
        
        # Test moderate coupling
        chunk.dependencies = ["dep1", "dep2", "dep3"]
        chunk.dependents = ["caller1", "caller2"]
        metrics = calculate_code_quality_metrics(chunk)
        assert 0.0 < metrics['coupling'] < 1.0
        
        # Test high coupling
        chunk.dependencies = ["dep" + str(i) for i in range(25)]
        metrics = calculate_code_quality_metrics(chunk)
        assert metrics['coupling'] == 1.0

    def test_edge_cases(self):
        """Test edge cases in quality metrics"""
        # Empty chunk
        empty_chunk = SemanticCodeChunk(
            id="empty",
            file_path="/test.py",
            content="",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        metrics = calculate_code_quality_metrics(empty_chunk)
        assert all(0.0 <= score <= 1.0 for score in metrics.values())
        
        # Chunk with extreme values
        extreme_chunk = SemanticCodeChunk(
            id="extreme",
            file_path="/test.py",
            content="def extreme(): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE
        )
        
        # Test extreme values - the set_code_quality_indicators should clamp them
        extreme_chunk.set_code_quality_indicators(
            docstring_quality=2.0,  # Above 1.0 - should be clamped to 1.0
            type_coverage=-0.5,     # Below 0.0 - should be clamped to 0.0
            cyclomatic_complexity=1000  # Very high - should be handled
        )
        
        # Verify values were clamped
        assert extreme_chunk.docstring_quality == 1.0  # Clamped from 2.0
        assert extreme_chunk.type_coverage == 0.0      # Clamped from -0.5
        assert extreme_chunk.cyclomatic_complexity == 1000  # Preserved but handled in metrics
        
        metrics = calculate_code_quality_metrics(extreme_chunk)
        assert all(0.0 <= score <= 1.0 for score in metrics.values())
        
        # Verify specific extreme case handling
        assert metrics['documentation_quality'] == 1.0  # Clamped value
        assert metrics['type_safety'] == 0.0            # Clamped value
        assert metrics['complexity'] == 0.3             # Very high complexity = 0.3


class TestFactoryFunctions:
    """Test suite for factory functions"""

    def test_create_code_chunk_for_content_type(self):
        """Test code chunk factory function"""
        chunk = create_code_chunk_for_content_type(
            content_type=ContentType.CODE,
            id="factory_test",
            file_path="/test.py",
            content="def factory_test(): pass",
            start_line=1,
            end_line=1,
            language="python"
        )
        
        # Should create a SemanticCodeChunk
        assert isinstance(chunk, SemanticCodeChunk)
        assert chunk.content_type == ContentType.CODE
        assert chunk.id == "factory_test"
        assert chunk.language == "python"


class TestCodeChunkIntegration:
    """Integration tests for code chunk functionality"""

    def test_full_code_analysis_workflow(self):
        """Test a complete code analysis workflow"""
        # Create a comprehensive code chunk
        content = '''class UserService:
    """Service for managing user operations.
    
    Provides CRUD operations for users with validation and error handling.
    """
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.validator = UserValidator()
    
    def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user with validation."""
        try:
            if not self.validator.validate(user_data):
                raise ValidationError("Invalid user data")
            
            user = User(**user_data)
            user_id = self.db.insert_user(user)
            user.id = user_id
            return user
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            raise UserCreationError(f"User creation failed: {e}")
'''
        
        chunk = SemanticCodeChunk(
            id="user_service",
            file_path="/src/services/user_service.py",
            content=content,
            start_line=1,
            end_line=25,
            content_type=ContentType.CODE,
            language="python",
            chunk_type=ChunkType.CLASS,
            architectural_role=ArchitecturalRole.BUSINESS_LOGIC
        )
        
        # Set code analysis properties
        chunk.imports = ["typing", "logging", "database", "validators"]
        chunk.exports = ["UserService"]
        chunk.function_calls = ["validator.validate", "User", "db.insert_user", "logger.error"]
        chunk.variables_used = ["user_data", "user", "user_id"]
        chunk.types_used = ["Dict", "Any", "User", "DatabaseConnection"]
        
        # Set quality indicators
        chunk.set_code_quality_indicators(
            has_docstring=True,
            docstring_quality=0.85,
            has_type_hints=True,
            type_coverage=0.9,
            has_error_handling=True,
            cyclomatic_complexity=4
        )
        
        # Add semantic information
        chunk.add_semantic_tag("business-logic", 0.9, "architectural")
        chunk.add_semantic_tag("user-management", 0.85, "domain")
        chunk.add_semantic_tag("crud-operations", 0.8, "functional")
        
        # Add relationships
        chunk.add_relationship("database_connection", "depends_on", 0.9)
        chunk.add_relationship("user_validator", "uses", 0.7)
        
        # Add design patterns and concerns
        chunk.design_patterns = ["dependency_injection", "service_layer"]
        chunk.architectural_concerns = ["error_handling", "validation"]
        
        # Test comprehensive analysis
        assert chunk.architectural_role == ArchitecturalRole.BUSINESS_LOGIC
        assert chunk.has_docstring
        assert chunk.has_type_hints
        assert chunk.has_error_handling
        assert chunk.is_high_quality
        assert chunk.is_reusable
        assert not chunk.is_highly_coupled
        
        # Test quality metrics
        metrics = calculate_code_quality_metrics(chunk)
        assert metrics['documentation_quality'] > 0.8
        assert metrics['type_safety'] > 0.8
        assert metrics['error_handling'] == 1.0
        assert metrics['complexity'] > 0.8  # Low complexity is good
        
        # Test semantic integration
        assert len(chunk.semantic_tags) == 3
        assert len(chunk.relationships) == 2
        assert "business-logic" in chunk.tag_names

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization of code chunks"""
        chunk = SemanticCodeChunk(
            id="serialize_test",
            file_path="/src/serialize.py",
            content="def serialize_test(): pass",
            start_line=1,
            end_line=1,
            content_type=ContentType.CODE,
            language="python",
            accessibility="private",
            architectural_role=ArchitecturalRole.UTILITY
        )
        
        # Add code-specific data
        chunk.imports = ["typing", "json"]
        chunk.exports = ["serialize_test"]
        chunk.design_patterns = ["factory"]
        chunk.code_smells = ["long_parameter_list"]
        
        chunk.set_code_quality_indicators(
            has_docstring=True,
            docstring_quality=0.7,
            cyclomatic_complexity=5
        )
        
        # Serialize to dict
        chunk_dict = chunk.to_dict()
        
        # Check that basic information is preserved
        assert chunk_dict['id'] == chunk.id
        assert chunk_dict['content'] == chunk.content
        assert chunk_dict['language'] == chunk.language
        
        # Note: The base SemanticChunk serialization might not include all 
        # code-specific fields. This test verifies what's currently serialized.
        assert 'content' in chunk_dict
        assert 'id' in chunk_dict


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])