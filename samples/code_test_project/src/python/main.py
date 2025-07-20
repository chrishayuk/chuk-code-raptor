#!/usr/bin/env python3
"""
Main application entry point.

This is the primary module that starts the application and handles
command-line arguments and basic setup.
"""

import sys
import argparse
from typing import Optional, List
from .utils import setup_logging, validate_config
from .auth import authenticate_user, check_permissions
from .database import DatabaseManager


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CodeRaptor Test Application",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Show what would be done without executing"
    )
    
    return parser.parse_args()


class Application:
    """Main application class."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.db_manager = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize the application."""
        try:
            # Validate configuration
            if not validate_config(self.config_path):
                print(f"Error: Invalid configuration file: {self.config_path}")
                return False
            
            # Setup database connection
            self.db_manager = DatabaseManager(self.config_path)
            self.db_manager.connect()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False
    
    def run(self, args: argparse.Namespace) -> int:
        """Run the main application logic."""
        if not self.is_initialized:
            return 1
        
        try:
            # Authenticate user
            user = authenticate_user()
            if not user:
                print("Authentication failed")
                return 1
            
            # Check permissions
            if not check_permissions(user, "read"):
                print("Insufficient permissions")
                return 1
            
            print(f"Welcome, {user.name}!")
            
            # Main application logic would go here
            print("Application running successfully...")
            
            return 0
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 1
        except Exception as e:
            print(f"Runtime error: {e}")
            return 1
        finally:
            if self.db_manager:
                self.db_manager.disconnect()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Create and run application
    app = Application(args.config)
    
    if not app.initialize():
        return 1
    
    return app.run(args)


if __name__ == "__main__":
    sys.exit(main())
