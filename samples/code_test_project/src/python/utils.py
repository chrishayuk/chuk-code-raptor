"""
Utility functions and helpers.

Common functionality used throughout the application.
"""

import os
import yaml
import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime


def setup_logging(verbose: bool = False) -> None:
    """Setup application logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_config(config_path: Union[str, Path]) -> bool:
    """Validate configuration file."""
    try:
        config = load_config(config_path)
        
        # Check required keys
        required_keys = ['database', 'auth', 'logging']
        for key in required_keys:
            if key not in config:
                print(f"Missing required config key: {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Config validation error: {e}")
        return False


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Get file size in megabytes."""
    return Path(file_path).stat().st_size / (1024 * 1024)


def format_timestamp(dt: datetime) -> str:
    """Format datetime as string."""
    return dt.strftime('%Y-%m-%d %H:%M:%S')


class Timer:
    """Simple timing context manager."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"{self.description} took {duration:.2f} seconds")


def retry_operation(func, max_attempts: int = 3, delay: float = 1.0):
    """Retry an operation with exponential backoff."""
    import time
    
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay * (2 ** attempt))


# Environment helpers
def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default."""
    return os.environ.get(name, default)


def is_development() -> bool:
    """Check if running in development mode."""
    return get_env_var('ENV', 'development').lower() == 'development'


def is_production() -> bool:
    """Check if running in production mode."""
    return get_env_var('ENV', 'development').lower() == 'production'
