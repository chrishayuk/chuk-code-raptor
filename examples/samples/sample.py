#!/usr/bin/env python3
# examples/samples/sample.py
"""
Enhanced Sample Python File
===========================

Comprehensive demonstration of Python language features for semantic chunking:
- Advanced async/await patterns with context managers
- Dataclasses with validation and post-init processing
- Abstract base classes and protocols
- Sophisticated error handling and retry logic
- Decorators with parameters and functools
- File I/O operations and path handling
- Dependency injection patterns
- Factory methods and builder patterns
- Type hints with generics and unions
- Logging and configuration management
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps, lru_cache, partial
from pathlib import Path
from typing import (
    Dict, List, Optional, Protocol, Union, Any, Callable, 
    TypeVar, Generic, Iterator, AsyncIterator, Tuple
)
import weakref
from collections import defaultdict, deque

# Module-level constants and configuration
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3
BATCH_SIZE_LIMIT = 1000
VERSION = "2.1.0"

# Global configuration registry
_global_config = {
    'log_level': 'INFO',
    'max_workers': 10,
    'enable_metrics': True,
    'cache_size': 128
}

logger = logging.getLogger(__name__)

T = TypeVar('T')
ProcessorResult = TypeVar('ProcessorResult')

class ProcessingStatus(Enum):
    """Enumeration of processing statuses."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.timestamp = time.time()

class ProcessingError(Exception):
    """Custom exception for processing errors."""
    def __init__(self, message: str, error_code: int = None, context: Dict = None):
        super().__init__(message)
        self.error_code = error_code or 500
        self.context = context or {}
        self.timestamp = time.time()

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def retry_with_backoff(max_attempts: int = 3, base_delay: float = 1.0, 
                       exponential_base: float = 2.0):
    """Decorator for retrying operations with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise
                    
                    delay = base_delay * (exponential_base ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise
                    
                    delay = base_delay * (exponential_base ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

@dataclass
class ProcessingConfig:
    """Advanced configuration for data processing operations."""
    batch_size: int = 100
    timeout: float = DEFAULT_TIMEOUT
    retries: int = MAX_RETRIES
    debug: bool = False
    enable_metrics: bool = True
    worker_pool_size: int = 5
    cache_enabled: bool = True
    compression_enabled: bool = False
    options: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.validate():
            raise ValidationError("Invalid configuration parameters")
        
        # Apply default options
        default_options = {
            'priority': 'normal',
            'encoding': 'utf-8',
            'buffer_size': 8192
        }
        
        for key, value in default_options.items():
            self.options.setdefault(key, value)
    
    def validate(self) -> bool:
        """Comprehensive validation of configuration parameters."""
        validations = [
            (self.batch_size > 0, "batch_size must be positive"),
            (self.batch_size <= BATCH_SIZE_LIMIT, f"batch_size cannot exceed {BATCH_SIZE_LIMIT}"),
            (self.timeout > 0, "timeout must be positive"),
            (self.retries >= 0, "retries cannot be negative"),
            (self.worker_pool_size > 0, "worker_pool_size must be positive"),
            (self.worker_pool_size <= 50, "worker_pool_size cannot exceed 50")
        ]
        
        for condition, message in validations:
            if not condition:
                logger.error(f"Configuration validation failed: {message}")
                return False
        
        return True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'ProcessingConfig':
        """Create configuration from JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return cls.from_dict(config_data)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in configuration file: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'batch_size': self.batch_size,
            'timeout': self.timeout,
            'retries': self.retries,
            'debug': self.debug,
            'enable_metrics': self.enable_metrics,
            'worker_pool_size': self.worker_pool_size,
            'cache_enabled': self.cache_enabled,
            'compression_enabled': self.compression_enabled,
            'options': self.options.copy(),
            'tags': self.tags.copy()
        }

@dataclass
class ProcessingMetrics:
    """Metrics tracking for processing operations."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    items_processed: int = 0
    items_failed: int = 0
    retry_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage_mb: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Calculate processing duration."""
        end = self.end_time or time.time()
        return max(0, end - self.start_time)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.items_processed + self.items_failed
        return (self.items_processed / total * 100) if total > 0 else 0.0
    
    @property
    def throughput(self) -> float:
        """Calculate items per second."""
        return self.items_processed / self.duration if self.duration > 0 else 0.0
    
    def finalize(self) -> None:
        """Mark metrics as complete."""
        if self.end_time is None:
            self.end_time = time.time()

class ProcessorProtocol(Protocol):
    """Protocol defining the interface for data processors."""
    
    async def process(self, data: List[Dict[str, Any]]) -> Optional[ProcessorResult]:
        """Process data and return results."""
        ...
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get processing metrics."""
        ...
    
    async def health_check(self) -> bool:
        """Check if processor is healthy."""
        ...

class CacheProtocol(Protocol):
    """Protocol for caching implementations."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        ...
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        ...

class MemoryCache:
    """Simple in-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_order = deque()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                
                if time.time() < expiry:
                    # Move to end (most recently used)
                    self._access_order.remove(key)
                    self._access_order.append(key)
                    return value
                else:
                    # Expired
                    del self._cache[key]
                    self._access_order.remove(key)
            
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        async with self._lock:
            # Remove if already exists
            if key in self._cache:
                self._access_order.remove(key)
            
            # Add new entry
            self._cache[key] = (value, expiry)
            self._access_order.append(key)
            
            # Evict if over size limit
            while len(self._cache) > self.max_size:
                oldest_key = self._access_order.popleft()
                del self._cache[oldest_key]
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
                return True
            return False

class BaseProcessor(ABC):
    """Abstract base class for all data processors with enhanced features."""
    
    def __init__(self, config: ProcessingConfig, cache: Optional[CacheProtocol] = None):
        self.config = config
        self.cache = cache
        self.metrics = ProcessingMetrics()
        self._is_initialized = False
        self._is_shutdown = False
        self._processors_registry = weakref.WeakSet()
        
        # Register this processor
        self._processors_registry.add(self)
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config.to_dict()}")
    
    @abstractmethod
    async def process(self, data: List[Dict[str, Any]]) -> Optional[ProcessorResult]:
        """Process data - must be implemented by subclasses."""
        pass
    
    @property
    def is_ready(self) -> bool:
        """Check if processor is ready for operations."""
        return self._is_initialized and not self._is_shutdown and self.config.validate()
    
    @property
    def status(self) -> ProcessingStatus:
        """Get current processor status."""
        if self._is_shutdown:
            return ProcessingStatus.CANCELLED
        elif not self._is_initialized:
            return ProcessingStatus.PENDING
        else:
            return ProcessingStatus.RUNNING
    
    async def health_check(self) -> bool:
        """Perform health check."""
        try:
            # Basic health checks
            if not self.is_ready:
                return False
            
            # Test cache if available
            if self.cache:
                test_key = f"health_check_{int(time.time())}"
                await self.cache.set(test_key, "test", ttl=1)
                test_value = await self.cache.get(test_key)
                if test_value != "test":
                    logger.warning("Cache health check failed")
                    return False
                await self.cache.delete(test_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics."""
        return self.metrics
    
    @timing_decorator
    async def initialize(self) -> None:
        """Initialize the processor with comprehensive setup."""
        if self._is_initialized:
            logger.warning("Processor already initialized")
            return
        
        try:
            if self.config.debug:
                logger.debug("Starting processor initialization...")
            
            # Simulate initialization work
            await asyncio.sleep(0.1)
            
            # Initialize cache if not provided
            if self.cache is None and self.config.cache_enabled:
                self.cache = MemoryCache(
                    max_size=self.config.options.get('cache_size', _global_config['cache_size'])
                )
                logger.debug("Initialized default memory cache")
            
            # Setup metrics tracking
            if self.config.enable_metrics:
                self.metrics = ProcessingMetrics()
                logger.debug("Metrics tracking enabled")
            
            self._is_initialized = True
            logger.info(f"{self.__class__.__name__} initialized successfully")
            
        except Exception as e:
            logger.error(f"Processor initialization failed: {e}")
            self._is_initialized = False
            raise ProcessingError(f"Initialization failed: {e}", context={'processor': self.__class__.__name__})
    
    @timing_decorator
    async def shutdown(self) -> None:
        """Gracefully shutdown the processor."""
        if self._is_shutdown:
            return
        
        logger.info(f"Shutting down {self.__class__.__name__}")
        
        try:
            # Finalize metrics
            if self.config.enable_metrics:
                self.metrics.finalize()
            
            # Clear cache if owned
            if self.cache and isinstance(self.cache, MemoryCache):
                # Could implement cache persistence here
                pass
            
            self._is_shutdown = True
            logger.info("Processor shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise

class AsyncDataProcessor(BaseProcessor, Generic[T]):
    """Advanced asynchronous data processor with comprehensive features."""
    
    def __init__(self, config: ProcessingConfig, cache: Optional[CacheProtocol] = None):
        super().__init__(config, cache)
        self._semaphore = asyncio.Semaphore(config.worker_pool_size)
        self._session_pool = {}
        self._active_tasks = set()
        self._task_counter = 0
    
    @timing_decorator
    @retry_with_backoff(max_attempts=3)
    async def process(self, data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Process data with advanced features and monitoring."""
        if not self.is_ready:
            raise ProcessingError("Processor not ready", error_code=503)
        
        if not data:
            logger.warning("No data provided for processing")
            return None
        
        processing_id = f"proc_{int(time.time())}_{self._task_counter}"
        self._task_counter += 1
        
        logger.info(f"Starting processing job {processing_id} with {len(data)} items")
        
        try:
            async with self._get_processing_session(processing_id) as session:
                # Check cache for previously processed data
                cache_key = self._generate_cache_key(data)
                cached_result = None
                
                if self.cache and self.config.cache_enabled:
                    cached_result = await self.cache.get(cache_key)
                    if cached_result:
                        self.metrics.cache_hits += 1
                        logger.debug(f"Cache hit for {processing_id}")
                        return cached_result
                    else:
                        self.metrics.cache_misses += 1
                
                # Process data in batches
                results = []
                total_batches = (len(data) + self.config.batch_size - 1) // self.config.batch_size
                
                for batch_idx in range(0, len(data), self.config.batch_size):
                    batch_num = (batch_idx // self.config.batch_size) + 1
                    batch = data[batch_idx:batch_idx + self.config.batch_size]
                    
                    logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
                    
                    try:
                        batch_result = await self._process_batch_with_monitoring(
                            batch, session, f"{processing_id}_batch_{batch_num}"
                        )
                        
                        if batch_result:
                            results.extend(batch_result)
                            self.metrics.items_processed += len(batch_result)
                    
                    except Exception as e:
                        logger.error(f"Batch {batch_num} failed: {e}")
                        self.metrics.items_failed += len(batch)
                        
                        # Continue with other batches or fail completely based on config
                        if not self.config.options.get('continue_on_batch_failure', True):
                            raise
                
                # Create result
                result = {
                    "processing_id": processing_id,
                    "success": True,
                    "processed_count": len(results),
                    "failed_count": self.metrics.items_failed,
                    "total_batches": total_batches,
                    "cache_hit": cached_result is not None,
                    "results": results,
                    "metadata": {
                        "processor_version": VERSION,
                        "config_tags": self.config.tags,
                        "processing_time": self.metrics.duration
                    }
                }
                
                # Cache the result
                if self.cache and self.config.cache_enabled and not cached_result:
                    cache_ttl = self.config.options.get('cache_ttl', 3600)
                    await self.cache.set(cache_key, result, ttl=cache_ttl)
                
                logger.info(f"Processing job {processing_id} completed successfully")
                return result
        
        except Exception as e:
            logger.error(f"Processing job {processing_id} failed: {e}")
            
            error_result = {
                "processing_id": processing_id,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "failed_count": len(data)
            }
            
            return error_result
    
    async def _process_batch_with_monitoring(self, batch: List[Dict], session: Dict, 
                                           batch_id: str) -> List[Dict]:
        """Process a batch with comprehensive monitoring."""
        start_time = time.time()
        
        try:
            # Create task for monitoring
            task = asyncio.create_task(self._process_batch_internal(batch, session))
            self._active_tasks.add(task)
            
            try:
                result = await asyncio.wait_for(task, timeout=self.config.timeout)
                processing_time = time.time() - start_time
                
                logger.debug(f"Batch {batch_id} completed in {processing_time:.3f}s")
                return result
                
            finally:
                self._active_tasks.discard(task)
        
        except asyncio.TimeoutError:
            logger.warning(f"Batch {batch_id} timed out after {self.config.timeout}s")
            raise ProcessingError(f"Batch processing timeout: {batch_id}", error_code=408)
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Batch {batch_id} failed after {processing_time:.3f}s: {e}")
            raise
    
    async def _process_batch_internal(self, batch: List[Dict], session: Dict) -> List[Dict]:
        """Internal batch processing with semaphore control."""
        async with self._semaphore:
            # Create tasks for concurrent processing
            tasks = [
                self._process_item_with_retry(item, session, f"item_{i}")
                for i, item in enumerate(batch)
            ]
            
            # Process with exception handling
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            successful_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Item processing failed: {result}")
                    self.metrics.items_failed += 1
                else:
                    successful_results.append(result)
            
            return successful_results
    
    async def _process_item_with_retry(self, item: Dict, session: Dict, item_id: str) -> Dict:
        """Process a single item with retry logic."""
        for attempt in range(self.config.retries + 1):
            try:
                return await self._process_single_item(item, session, item_id)
            
            except Exception as e:
                if attempt == self.config.retries:
                    logger.error(f"Item {item_id} failed after {self.config.retries} retries: {e}")
                    raise
                
                retry_delay = min(2 ** attempt, 10)  # Cap at 10 seconds
                logger.debug(f"Retrying item {item_id} in {retry_delay}s (attempt {attempt + 1})")
                await asyncio.sleep(retry_delay)
                self.metrics.retry_count += 1
    
    async def _process_single_item(self, item: Dict, session: Dict, item_id: str) -> Dict:
        """Process a single item with business logic."""
        # Simulate processing time based on item complexity
        complexity = item.get('complexity', 1.0)
        processing_time = min(0.01 * complexity, 0.1)
        await asyncio.sleep(processing_time)
        
        # Simulate different processing outcomes
        if item.get('should_fail', False):
            raise ProcessingError(f"Simulated failure for {item_id}")
        
        # Create processed result with enriched data
        result = {
            **item,
            "processed": True,
            "processed_at": time.time(),
            "processor_id": id(self),
            "session_id": session.get('session_id'),
            "processing_version": VERSION,
            "item_id": item_id
        }
        
        # Add computed fields based on processing logic
        if 'value' in item:
            result['processed_value'] = item['value'] * 2
        
        if 'category' in item:
            result['category_normalized'] = item['category'].lower().strip()
        
        return result
    
    @asynccontextmanager
    async def _get_processing_session(self, session_id: str):
        """Context manager for processing sessions."""
        session = {
            'session_id': session_id,
            'created_at': time.time(),
            'processor_id': id(self),
            'config_snapshot': self.config.to_dict()
        }
        
        self._session_pool[session_id] = session
        logger.debug(f"Created processing session: {session_id}")
        
        try:
            yield session
        finally:
            # Session cleanup
            if session_id in self._session_pool:
                session_duration = time.time() - session['created_at']
                logger.debug(f"Closed session {session_id} after {session_duration:.3f}s")
                del self._session_pool[session_id]
    
    def _generate_cache_key(self, data: List[Dict]) -> str:
        """Generate cache key for data."""
        import hashlib
        
        # Create deterministic hash from data
        data_str = json.dumps(data, sort_keys=True, default=str)
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        
        combined = f"{data_str}:{config_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def cancel_all_tasks(self) -> None:
        """Cancel all active processing tasks."""
        if self._active_tasks:
            logger.info(f"Cancelling {len(self._active_tasks)} active tasks")
            
            for task in self._active_tasks.copy():
                task.cancel()
            
            # Wait for tasks to complete cancellation
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()

class FileProcessor(AsyncDataProcessor[Dict]):
    """Specialized processor for file operations with advanced features."""
    
    def __init__(self, config: ProcessingConfig, output_dir: Optional[Path] = None, 
                 supported_formats: Optional[List[str]] = None):
        super().__init__(config)
        self.output_dir = Path(output_dir or "./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.supported_formats = supported_formats or ['.txt', '.json', '.csv', '.md']
        self._file_locks = defaultdict(asyncio.Lock)
    
    async def process_files(self, file_paths: List[Union[str, Path]], 
                          output_format: str = 'json') -> Dict[str, Any]:
        """Process multiple files with comprehensive error handling."""
        if not file_paths:
            return {"error": "No file paths provided"}
        
        processing_results = {}
        valid_files = []
        
        # Validate files first
        for file_path in file_paths:
            file_path = Path(file_path)
            
            if not file_path.exists():
                processing_results[str(file_path)] = {
                    "success": False,
                    "error": "File not found",
                    "error_code": 404
                }
                continue
            
            if file_path.suffix.lower() not in self.supported_formats:
                processing_results[str(file_path)] = {
                    "success": False,
                    "error": f"Unsupported format: {file_path.suffix}",
                    "error_code": 415
                }
                continue
            
            valid_files.append(file_path)
        
        logger.info(f"Processing {len(valid_files)} valid files out of {len(file_paths)} total")
        
        # Process valid files concurrently
        if valid_files:
            tasks = [
                self._process_single_file(file_path, output_format)
                for file_path in valid_files
            ]
            
            file_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for file_path, result in zip(valid_files, file_results):
                if isinstance(result, Exception):
                    processing_results[str(file_path)] = {
                        "success": False,
                        "error": str(result),
                        "error_type": type(result).__name__
                    }
                else:
                    processing_results[str(file_path)] = result
        
        # Generate summary
        successful = sum(1 for r in processing_results.values() if r.get("success", False))
        failed = len(processing_results) - successful
        
        return {
            "summary": {
                "total_files": len(file_paths),
                "processed": successful,
                "failed": failed,
                "success_rate": (successful / len(file_paths) * 100) if file_paths else 0
            },
            "files": processing_results,
            "processor_metrics": self.get_metrics().custom_metrics
        }
    
    @timing_decorator
    async def _process_single_file(self, file_path: Path, output_format: str) -> Dict[str, Any]:
        """Process a single file with file-specific logic."""
        async with self._file_locks[str(file_path)]:
            try:
                # Read file content
                content = await self._read_file_async(file_path)
                
                # Create processing data
                file_data = [{
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "content": content,
                    "format": file_path.suffix.lower()
                }]
                
                # Process using parent class
                process_result = await self.process(file_data)
                
                if process_result and process_result.get("success"):
                    # Save result to output file
                    output_file = self._generate_output_path(file_path, output_format)
                    await self._write_result_async(output_file, process_result)
                    
                    return {
                        "success": True,
                        "input_file": str(file_path),
                        "output_file": str(output_file),
                        "processed_items": process_result.get("processed_count", 0),
                        "file_size_bytes": file_path.stat().st_size,
                        "processing_time": process_result.get("metadata", {}).get("processing_time", 0)
                    }
                else:
                    return process_result or {"success": False, "error": "Processing failed"}
            
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
    
    async def _read_file_async(self, file_path: Path) -> str:
        """Asynchronously read file content with encoding detection."""
        loop = asyncio.get_event_loop()
        
        def read_file():
            try:
                # Try UTF-8 first
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                # Fallback to other encodings
                for encoding in ['latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                
                # Final fallback - read as binary and decode with errors ignored
                with open(file_path, 'rb') as f:
                    return f.read().decode('utf-8', errors='ignore')
        
        return await loop.run_in_executor(None, read_file)
    
    async def _write_result_async(self, output_file: Path, result: Dict) -> None:
        """Asynchronously write processing result with formatting."""
        loop = asyncio.get_event_loop()
        
        def write_file():
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                if output_file.suffix.lower() == '.json':
                    json.dump(result, f, indent=2, default=str, ensure_ascii=False)
                else:
                    f.write(str(result))
        
        await loop.run_in_executor(None, write_file)
    
    def _generate_output_path(self, input_path: Path, output_format: str) -> Path:
        """Generate output file path based on input and format."""
        timestamp = int(time.time())
        base_name = f"{input_path.stem}_processed_{timestamp}"
        
        if output_format.lower() == 'json':
            extension = '.json'
        elif output_format.lower() == 'txt':
            extension = '.txt'
        else:
            extension = '.json'  # Default
        
        return self.output_dir / f"{base_name}{extension}"

# Factory and utility functions
class ProcessorFactory:
    """Factory for creating different types of processors."""
    
    _processor_types = {
        'async': AsyncDataProcessor,
        'file': FileProcessor
    }
    
    @classmethod
    def create_processor(cls, processor_type: str, config: Optional[ProcessingConfig] = None, 
                        **kwargs) -> BaseProcessor:
        """Create a processor of the specified type."""
        if processor_type not in cls._processor_types:
            available_types = list(cls._processor_types.keys())
            raise ValueError(f"Unknown processor type '{processor_type}'. Available: {available_types}")
        
        config = config or ProcessingConfig()
        processor_class = cls._processor_types[processor_type]
        
        return processor_class(config, **kwargs)
    
    @classmethod
    def register_processor_type(cls, name: str, processor_class: type) -> None:
        """Register a new processor type."""
        if not issubclass(processor_class, BaseProcessor):
            raise ValueError("Processor class must inherit from BaseProcessor")
        
        cls._processor_types[name] = processor_class
        logger.info(f"Registered processor type: {name}")
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available processor types."""
        return list(cls._processor_types.keys())

@lru_cache(maxsize=128)
def get_default_config(processor_type: str = "async") -> ProcessingConfig:
    """Get default configuration for processor type (cached)."""
    base_config = ProcessingConfig()
    
    if processor_type == "file":
        base_config.batch_size = 10  # Smaller batches for file processing
        base_config.options['buffer_size'] = 16384
        
    elif processor_type == "async":
        base_config.worker_pool_size = 10
        base_config.enable_metrics = True
    
    return base_config

@contextmanager
def temporary_config_override(**overrides) -> Iterator[None]:
    """Context manager for temporarily overriding global configuration."""
    original_values = {}
    
    for key, value in overrides.items():
        if key in _global_config:
            original_values[key] = _global_config[key]
        _global_config[key] = value
    
    try:
        yield
    finally:
        # Restore original values
        for key, value in original_values.items():
            _global_config[key] = value
        
        # Remove any new keys that weren't in original config
        for key in overrides:
            if key not in original_values:
                _global_config.pop(key, None)

async def cleanup_all_processors() -> None:
    """Cleanup function to shutdown all active processors."""
    # This would need to be implemented with proper registry tracking
    logger.info("Cleaning up all processors...")

# Legacy coroutine support (for demonstration)
@asyncio.coroutine
def legacy_processing_example(data: List[Dict]) -> Any:
    """Example of legacy coroutine syntax for compatibility testing."""
    yield from asyncio.sleep(0.1)
    
    processed_data = []
    for item in data:
        processed_item = {**item, "legacy_processed": True}
        processed_data.append(processed_item)
    
    return {"results": processed_data, "legacy": True}

# Main execution and demonstration
if __name__ == "__main__":
    async def comprehensive_demo():
        """Comprehensive demonstration of all features."""
        print("🚀 Starting Comprehensive Processing Demo")
        print("=" * 60)
        
        # Create advanced configuration
        config = ProcessingConfig(
            batch_size=5,
            timeout=10.0,
            debug=True,
            enable_metrics=True,
            worker_pool_size=3,
            cache_enabled=True,
            tags=["demo", "comprehensive", "async"],
            options={
                'priority': 'high',
                'cache_ttl': 300,
                'continue_on_batch_failure': True
            }
        )
        
        print(f"⚙️  Configuration: {config.to_dict()}")
        
        # Create processors using factory
        try:
            async_processor = ProcessorFactory.create_processor("async", config)
            await async_processor.initialize()
            
            # Perform health check
            health_status = await async_processor.health_check()
            print(f"🏥 Health check: {'✅ PASS' if health_status else '❌ FAIL'}")
            
            if not health_status:
                print("Cannot proceed - processor not healthy")
                return
            
            # Create test data with various scenarios
            test_data = [
                {"id": i, "value": i * 10, "category": f"group_{i % 3}", "complexity": 1.0}
                for i in range(1, 13)
            ]
            
            # Add some failure scenarios for testing
            test_data.append({"id": 999, "should_fail": True, "category": "error_test"})
            
            print(f"📊 Test data: {len(test_data)} items")
            
            # Process data
            print("\n🔄 Processing data...")
            result = await async_processor.process(test_data)
            
            if result and result.get("success"):
                print(f"✅ Processing completed successfully!")
                print(f"   Processed: {result['processed_count']} items")
                print(f"   Failed: {result['failed_count']} items")
                print(f"   Cache hit: {result['cache_hit']}")
                print(f"   Processing ID: {result['processing_id']}")
                
                # Show some sample results
                if result['results']:
                    print(f"   Sample result: {result['results'][0]}")
            else:
                print(f"❌ Processing failed: {result}")
            
            # Get detailed metrics
            metrics = async_processor.get_metrics()
            print(f"\n📈 Metrics:")
            print(f"   Duration: {metrics.duration:.3f}s")
            print(f"   Success rate: {metrics.success_rate:.1f}%")
            print(f"   Throughput: {metrics.throughput:.1f} items/sec")
            print(f"   Cache hits: {metrics.cache_hits}")
            print(f"   Cache misses: {metrics.cache_misses}")
            print(f"   Retry count: {metrics.retry_count}")
            
            # Test file processor
            print(f"\n📁 Testing file processor...")
            file_config = get_default_config("file")
            file_processor = ProcessorFactory.create_processor("file", file_config)
            await file_processor.initialize()
            
            # Create a test file for processing
            test_file = Path("test_input.txt")
            test_file.write_text("This is test content for file processing.\nLine 2\nLine 3")
            
            try:
                file_result = await file_processor.process_files([test_file])
                print(f"   File processing result: {file_result['summary']}")
            finally:
                # Cleanup test file
                if test_file.exists():
                    test_file.unlink()
            
            # Shutdown processors
            await async_processor.shutdown()
            await file_processor.shutdown()
            
        except Exception as e:
            print(f"💥 Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    def main():
        """Main entry point with configuration and error handling."""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        print(f"🐍 Enhanced Python Sample - Version {VERSION}")
        print(f"Available processor types: {ProcessorFactory.get_available_types()}")
        
        try:
            # Use temporary config override for demo
            with temporary_config_override(log_level='DEBUG', max_workers=5):
                asyncio.run(comprehensive_demo())
                
            print("\n🎉 Demo completed successfully!")
            
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n💥 Demo failed: {e}")
            logging.exception("Demo error details:")
    
    main()