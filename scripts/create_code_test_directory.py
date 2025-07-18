#!/usr/bin/env python3
# scripts/create_code_test_directory.py
"""
Create Test Directory Script
============================

Creates a comprehensive test directory structure with sample files
for testing the CodeRaptor file processor and chunking systems.

This generates realistic code files in multiple languages with
various patterns to test all aspects of the system under a samples folder.

Usage:
    python create_code_test_directory.py [output_dir]
    
Example:
    python create_code_test_directory.py ./my_project
    # Creates: ./my_project/samples/code_test_project/
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

def create_python_files(base_dir: Path) -> None:
    """Create Python source files"""
    python_dir = base_dir / "src" / "python"
    python_dir.mkdir(parents=True, exist_ok=True)
    
    # Main application file
    main_py = python_dir / "main.py"
    main_py.write_text('''#!/usr/bin/env python3
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
            print("\\nOperation cancelled by user")
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
''')

    # Authentication module
    auth_py = python_dir / "auth.py"
    auth_py.write_text('''"""
Authentication and authorization module.

Handles user authentication, session management, and permission checking.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class User:
    """User data model."""
    id: int
    name: str
    email: str
    password_hash: str
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None


@dataclass 
class Session:
    """User session data."""
    token: str
    user_id: int
    created_at: datetime
    expires_at: datetime
    is_active: bool = True


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class PermissionError(Exception):
    """Raised when user lacks required permissions."""
    pass


class AuthManager:
    """Manages authentication and sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.users: Dict[int, User] = self._load_test_users()
    
    def _load_test_users(self) -> Dict[int, User]:
        """Load test users for demo purposes."""
        return {
            1: User(
                id=1,
                name="Admin User",
                email="admin@example.com",
                password_hash=self._hash_password("admin123"),
                permissions=["read", "write", "admin"],
                created_at=datetime.now()
            ),
            2: User(
                id=2,
                name="Regular User", 
                email="user@example.com",
                password_hash=self._hash_password("user123"),
                permissions=["read"],
                created_at=datetime.now()
            )
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        salt = "code_raptor_salt"  # In real app, use random salt per user
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def authenticate(self, email: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        # Find user by email
        user = None
        for u in self.users.values():
            if u.email == email:
                user = u
                break
        
        if not user:
            raise AuthenticationError("User not found")
        
        # Check password
        password_hash = self._hash_password(password)
        if user.password_hash != password_hash:
            raise AuthenticationError("Invalid password")
        
        # Create session
        token = secrets.token_urlsafe(32)
        session = Session(
            token=token,
            user_id=user.id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        self.sessions[token] = session
        user.last_login = datetime.now()
        
        return token
    
    def get_user_by_token(self, token: str) -> Optional[User]:
        """Get user by session token."""
        session = self.sessions.get(token)
        if not session or not session.is_active:
            return None
        
        if datetime.now() > session.expires_at:
            session.is_active = False
            return None
        
        return self.users.get(session.user_id)
    
    def logout(self, token: str) -> bool:
        """Logout user by invalidating session."""
        session = self.sessions.get(token)
        if session:
            session.is_active = False
            return True
        return False


# Global auth manager instance
_auth_manager = AuthManager()


def authenticate_user() -> Optional[User]:
    """Interactive user authentication."""
    try:
        email = input("Email: ").strip()
        password = input("Password: ").strip()
        
        token = _auth_manager.authenticate(email, password)
        return _auth_manager.get_user_by_token(token)
        
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
        return None


def check_permissions(user: User, required_permission: str) -> bool:
    """Check if user has required permission."""
    return required_permission in user.permissions


def get_current_user(token: str) -> Optional[User]:
    """Get current user from token."""
    return _auth_manager.get_user_by_token(token)
''')

    # Utilities module
    utils_py = python_dir / "utils.py"
    utils_py.write_text('''"""
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
''')

    # Database module
    database_py = python_dir / "database.py"
    database_py.write_text('''"""
Database management module.

Handles database connections, queries, and data persistence.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from .utils import load_config


class DatabaseError(Exception):
    """Database operation error."""
    pass


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.db_path = self.config.get('database', {}).get('path', 'app.db')
        self.connection = None
    
    def connect(self) -> None:
        """Connect to database and initialize schema."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            self._initialize_schema()
        except Exception as e:
            raise DatabaseError(f"Failed to connect to database: {e}")
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            language TEXT,
            size_bytes INTEGER,
            line_count INTEGER,
            content_hash TEXT,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            file_id INTEGER,
            content TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            chunk_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES files (id)
        );
        """
        
        self.connection.executescript(schema_sql)
        self.connection.commit()
    
    @contextmanager
    def transaction(self):
        """Database transaction context manager."""
        if not self.connection:
            raise DatabaseError("Not connected to database")
        
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
    
    def execute_query(self, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results."""
        if not self.connection:
            raise DatabaseError("Not connected to database")
        
        cursor = self.connection.execute(sql, params)
        return cursor.fetchall()
    
    def execute_update(self, sql: str, params: Tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        if not self.connection:
            raise DatabaseError("Not connected to database")
        
        cursor = self.connection.execute(sql, params)
        self.connection.commit()
        return cursor.rowcount
    
    def insert_file(self, path: str, language: str, size_bytes: int, 
                   line_count: int, content_hash: str) -> int:
        """Insert a file record and return the ID."""
        sql = """
        INSERT OR REPLACE INTO files 
        (path, language, size_bytes, line_count, content_hash)
        VALUES (?, ?, ?, ?, ?)
        """
        
        cursor = self.connection.execute(
            sql, (path, language, size_bytes, line_count, content_hash)
        )
        self.connection.commit()
        return cursor.lastrowid
    
    def insert_chunk(self, chunk_id: str, file_id: int, content: str,
                    start_line: int, end_line: int, chunk_type: str) -> None:
        """Insert a chunk record."""
        sql = """
        INSERT OR REPLACE INTO chunks
        (id, file_id, content, start_line, end_line, chunk_type)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        self.execute_update(sql, (chunk_id, file_id, content, start_line, end_line, chunk_type))
    
    def get_files_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Get all files for a specific language."""
        sql = "SELECT * FROM files WHERE language = ? ORDER BY path"
        rows = self.execute_query(sql, (language,))
        return [dict(row) for row in rows]
    
    def get_chunks_by_file(self, file_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a specific file."""
        sql = "SELECT * FROM chunks WHERE file_id = ? ORDER BY start_line"
        rows = self.execute_query(sql, (file_id,))
        return [dict(row) for row in rows]
    
    def search_chunks(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search chunks by content."""
        sql = """
        SELECT c.*, f.path, f.language
        FROM chunks c
        JOIN files f ON c.file_id = f.id
        WHERE c.content LIKE ?
        ORDER BY c.chunk_type, f.path
        LIMIT ?
        """
        
        rows = self.execute_query(sql, (f'%{search_term}%', limit))
        return [dict(row) for row in rows]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        # File counts by language
        sql = "SELECT language, COUNT(*) as count FROM files GROUP BY language"
        language_counts = {row['language']: row['count'] for row in self.execute_query(sql)}
        stats['files_by_language'] = language_counts
        
        # Chunk counts by type  
        sql = "SELECT chunk_type, COUNT(*) as count FROM chunks GROUP BY chunk_type"
        chunk_counts = {row['chunk_type']: row['count'] for row in self.execute_query(sql)}
        stats['chunks_by_type'] = chunk_counts
        
        # Total counts
        stats['total_files'] = sum(language_counts.values())
        stats['total_chunks'] = sum(chunk_counts.values())
        
        return stats
''')

def create_javascript_files(base_dir: Path) -> None:
    """Create JavaScript/TypeScript source files"""
    js_dir = base_dir / "src" / "frontend"
    js_dir.mkdir(parents=True, exist_ok=True)
    
    # Main React component
    app_tsx = js_dir / "App.tsx"
    app_tsx.write_text('''import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import FileExplorer from './pages/FileExplorer';
import SearchResults from './pages/SearchResults';
import Settings from './pages/Settings';
import Login from './pages/Login';
import './App.css';

interface AppProps {
  title?: string;
}

const App: React.FC<AppProps> = ({ title = "CodeRaptor" }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate app initialization
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner"></div>
        <p>Loading {title}...</p>
      </div>
    );
  }

  return (
    <ThemeProvider>
      <AuthProvider>
        <Router>
          <div className="app">
            <AppContent 
              title={title}
              sidebarOpen={sidebarOpen}
              setSidebarOpen={setSidebarOpen}
            />
          </div>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
};

interface AppContentProps {
  title: string;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

const AppContent: React.FC<AppContentProps> = ({ title, sidebarOpen, setSidebarOpen }) => {
  const { user, isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Login />;
  }

  return (
    <>
      <Header 
        title={title}
        user={user}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
      />
      
      <div className="app-body">
        <Sidebar 
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />
        
        <main className={`main-content ${sidebarOpen ? 'with-sidebar' : 'full-width'}`}>
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/files" element={<FileExplorer />} />
            <Route path="/search" element={<SearchResults />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </main>
      </div>
    </>
  );
};

export default App;
''')

    # Auth context
    auth_context_tsx = js_dir / "contexts" / "AuthContext.tsx"
    auth_context_tsx.parent.mkdir(exist_ok=True)
    auth_context_tsx.write_text('''import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: number;
  name: string;
  email: string;
  permissions: string[];
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for existing session
    const checkSession = async () => {
      try {
        const token = localStorage.getItem('auth_token');
        if (token) {
          const userData = await validateToken(token);
          if (userData) {
            setUser(userData);
          }
        }
      } catch (error) {
        console.error('Session validation failed:', error);
        localStorage.removeItem('auth_token');
      } finally {
        setLoading(false);
      }
    };

    checkSession();
  }, []);

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      setLoading(true);
      
      // Mock authentication - replace with real API call
      const response = await mockLogin(email, password);
      
      if (response.success) {
        setUser(response.user);
        localStorage.setItem('auth_token', response.token);
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Login failed:', error);
      return false;
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('auth_token');
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    login,
    logout,
    loading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Mock authentication functions
const mockLogin = async (email: string, password: string) => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  // Mock users
  const users = [
    {
      id: 1,
      name: 'Admin User',
      email: 'admin@example.com',
      password: 'admin123',
      permissions: ['read', 'write', 'admin']
    },
    {
      id: 2,
      name: 'Regular User',
      email: 'user@example.com', 
      password: 'user123',
      permissions: ['read']
    }
  ];
  
  const user = users.find(u => u.email === email && u.password === password);
  
  if (user) {
    const { password: _, ...userWithoutPassword } = user;
    return {
      success: true,
      user: userWithoutPassword,
      token: 'mock_token_' + Date.now()
    };
  }
  
  return { success: false };
};

const validateToken = async (token: string): Promise<User | null> => {
  // Mock token validation
  if (token.startsWith('mock_token_')) {
    return {
      id: 1,
      name: 'Mock User',
      email: 'user@example.com',
      permissions: ['read']
    };
  }
  return null;
};
''')

    # Utility functions
    utils_ts = js_dir / "utils" / "helpers.ts"
    utils_ts.parent.mkdir(exist_ok=True)
    utils_ts.write_text('''/**
 * Utility functions and helpers for the frontend application.
 */

// Type definitions
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface FileMetadata {
  path: string;
  name: string;
  size: number;
  language: string;
  lastModified: Date;
  lineCount: number;
}

export interface SearchResult {
  id: string;
  content: string;
  filePath: string;
  language: string;
  startLine: number;
  endLine: number;
  score: number;
}

// String utilities
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
};

export const formatFileSize = (bytes: number): string => {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${size.toFixed(1)} ${units[unitIndex]}`;
};

export const capitalizeFirst = (str: string): string => {
  return str.charAt(0).toUpperCase() + str.slice(1);
};

// Date utilities
export const formatDate = (date: Date | string): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

export const getRelativeTime = (date: Date | string): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  
  if (diffMinutes < 1) return 'just now';
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  
  return formatDate(d);
};

// Array utilities
export const groupBy = <T, K extends keyof T>(
  array: T[],
  key: K
): Record<string, T[]> => {
  return array.reduce((groups, item) => {
    const groupKey = String(item[key]);
    groups[groupKey] = groups[groupKey] || [];
    groups[groupKey].push(item);
    return groups;
  }, {} as Record<string, T[]>);
};

export const sortBy = <T>(
  array: T[],
  key: keyof T,
  direction: 'asc' | 'desc' = 'asc'
): T[] => {
  return [...array].sort((a, b) => {
    const aVal = a[key];
    const bVal = b[key];
    
    if (aVal < bVal) return direction === 'asc' ? -1 : 1;
    if (aVal > bVal) return direction === 'asc' ? 1 : -1;
    return 0;
  });
};

// API utilities
export const fetchApi = async <T = any>(
  url: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> => {
  try {
    const token = localStorage.getItem('auth_token');
    
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      return {
        success: false,
        error: data.message || 'Request failed',
      };
    }
    
    return {
      success: true,
      data,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
};

// Local storage utilities
export const storage = {
  get<T>(key: string, defaultValue?: T): T | undefined {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch {
      return defaultValue;
    }
  },
  
  set<T>(key: string, value: T): void {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error('Failed to save to localStorage:', error);
    }
  },
  
  remove(key: string): void {
    localStorage.removeItem(key);
  },
  
  clear(): void {
    localStorage.clear();
  }
};

// Debounce utility
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  delay: number
): ((...args: Parameters<T>) => void) => {
  let timeoutId: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
};

// Theme utilities
export const getLanguageColor = (language: string): string => {
  const colors: Record<string, string> = {
    python: '#3776ab',
    javascript: '#f7df1e',
    typescript: '#3178c6',
    java: '#ed8b00',
    cpp: '#00599c',
    rust: '#ce422b',
    go: '#00add8',
    ruby: '#cc342d',
    php: '#777bb4',
    html: '#e34f26',
    css: '#1572b6',
    markdown: '#083fa1',
    json: '#292929',
    yaml: '#cb171e',
    unknown: '#6c757d'
  };
  
  return colors[language.toLowerCase()] || colors.unknown;
};

export const getFileIcon = (language: string): string => {
  const icons: Record<string, string> = {
    python: '🐍',
    javascript: '📜',
    typescript: '📘',
    java: '☕',
    cpp: '⚙️',
    rust: '🦀',
    go: '🐹',
    ruby: '💎',
    php: '🐘',
    html: '🌐',
    css: '🎨',
    markdown: '📝',
    json: '📋',
    yaml: '⚙️',
    unknown: '📄'
  };
  
  return icons[language.toLowerCase()] || icons.unknown;
};
''')

def create_documentation_files(base_dir: Path) -> None:
    """Create documentation files"""
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Main README
    readme_md = base_dir / "README.md"
    readme_md.write_text('''# CodeRaptor Test Project

This is a test project created to demonstrate and test the CodeRaptor file processing and indexing system.

## Project Structure

```
samples/code_test_project/
├── src/
│   ├── python/          # Python backend code
│   │   ├── main.py     # Application entry point
│   │   ├── auth.py     # Authentication module
│   │   ├── utils.py    # Utility functions
│   │   └── database.py # Database operations
│   └── frontend/        # TypeScript/React frontend
│       ├── App.tsx     # Main React component
│       ├── contexts/   # React contexts
│       └── utils/      # Frontend utilities
├── docs/               # Documentation
├── config/             # Configuration files
├── tests/              # Test files
└── scripts/            # Utility scripts

```

## Features

This test project includes:

- **Python Backend**: Authentication, database operations, utilities
- **TypeScript Frontend**: React components, contexts, utilities  
- **Documentation**: README files and API docs
- **Configuration**: YAML and JSON config files
- **Tests**: Unit and integration tests
- **Scripts**: Build and deployment scripts

## Languages Covered

- Python (backend logic)
- TypeScript/JavaScript (frontend)
- Markdown (documentation)
- YAML (configuration)
- JSON (data files)
- Shell scripts (automation)

## Testing CodeRaptor

Use this project to test:

1. **File Processing**: Multi-language detection and parsing
2. **Chunking**: Semantic chunking of functions, classes, components
3. **Hierarchy Building**: RAPTOR hierarchical organization
4. **Search**: Finding relevant code across languages
5. **Statistics**: Language distribution and file metrics

## Usage

```bash
# Index this test project
code-raptor index ./samples/code_test_project

# Search for authentication code  
code-raptor search "authentication"

# Search for React components
code-raptor search "React component"

# Get project statistics
code-raptor stats ./samples/code_test_project
```

## Expected Results

When processing this project, CodeRaptor should:

- Detect 6+ programming languages
- Create 20+ semantic code chunks
- Build 3-4 level hierarchy
- Generate meaningful summaries
- Enable cross-language search

This provides a comprehensive test case for all CodeRaptor features.
''')

    # API documentation
    api_md = docs_dir / "api.md"
    api_md.write_text('''# API Documentation

## Authentication Endpoints

### POST /api/auth/login
Authenticate user and return session token.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "name": "User Name",
    "email": "user@example.com",
    "permissions": ["read", "write"]
  }
}
```

### POST /api/auth/logout
Invalidate session token.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

## File Management Endpoints

### GET /api/files
List all indexed files.

**Query Parameters:**
- `language` - Filter by programming language
- `limit` - Maximum number of results (default: 50)
- `offset` - Pagination offset (default: 0)

**Response:**
```json
{
  "success": true,
  "data": {
    "files": [
      {
        "id": 1,
        "path": "src/main.py",
        "language": "python", 
        "size": 2048,
        "lineCount": 85,
        "lastModified": "2025-01-17T10:30:00Z"
      }
    ],
    "total": 1,
    "hasMore": false
  }
}
```

### GET /api/files/:id
Get detailed information about a specific file.

**Response:**
```json
{
  "success": true,
  "data": {
    "file": {
      "id": 1,
      "path": "src/main.py",
      "language": "python",
      "content": "#!/usr/bin/env python3...",
      "chunks": [
        {
          "id": "main.py:function:main:10",
          "type": "function",
          "startLine": 10,
          "endLine": 25,
          "content": "def main():"
        }
      ]
    }
  }
}
```

## Search Endpoints

### GET /api/search
Search code chunks by content.

**Query Parameters:**
- `q` - Search query (required)
- `language` - Filter by language
- `type` - Filter by chunk type (function, class, etc.)
- `limit` - Maximum results (default: 10)

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "main.py:function:authenticate:15",
        "content": "def authenticate(email, password):",
        "filePath": "src/auth.py",
        "language": "python",
        "startLine": 15,
        "endLine": 30,
        "score": 0.95,
        "context": "Authentication module"
      }
    ],
    "total": 1,
    "query": "authenticate",
    "took": 45
  }
}
```

## Statistics Endpoints

### GET /api/stats
Get project statistics.

**Response:**
```json
{
  "success": true,
  "data": {
    "files": {
      "total": 25,
      "byLanguage": {
        "python": 8,
        "typescript": 12,
        "markdown": 3,
        "yaml": 2
      }
    },
    "chunks": {
      "total": 156,
      "byType": {
        "function": 45,
        "class": 12,
        "component": 18,
        "section": 15
      }
    },
    "size": {
      "totalBytes": 524288,
      "totalLines": 3421
    }
  }
}
```

## Error Responses

All endpoints return errors in this format:

```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

### Common Error Codes
- `UNAUTHORIZED` - Invalid or missing authentication
- `FORBIDDEN` - Insufficient permissions
- `NOT_FOUND` - Resource not found
- `VALIDATION_ERROR` - Invalid request data
- `INTERNAL_ERROR` - Server error
''')

def create_config_files(base_dir: Path) -> None:
    """Create configuration files"""
    config_dir = base_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Application config
    app_yaml = config_dir / "app.yaml"
    app_yaml.write_text('''# Application Configuration

database:
  path: "app.db"
  pool_size: 10
  timeout: 30

auth:
  secret_key: "your-secret-key-here"
  token_expiry_hours: 24
  bcrypt_rounds: 12

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "app.log"
  max_size_mb: 10
  backup_count: 5

server:
  host: "0.0.0.0"
  port: 8000
  debug: false
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:8080"

features:
  enable_search: true
  enable_file_upload: true
  enable_admin_panel: true
  max_file_size_mb: 10

cache:
  type: "memory"  # memory, redis, memcached
  ttl_seconds: 3600
  max_size: 1000
''')

    # Development environment
    dev_json = config_dir / "development.json"
    dev_json.write_text('''{
  "environment": "development",
  "debug": true,
  "database": {
    "path": "dev.db",
    "echo_sql": true
  },
  "logging": {
    "level": "DEBUG"
  },
  "server": {
    "port": 8000,
    "reload": true
  },
  "features": {
    "enable_debug_routes": true,
    "enable_profiling": true
  }
}
''')

def create_test_files(base_dir: Path) -> None:
    """Create test files"""
    tests_dir = base_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    
    # Python test
    test_auth_py = tests_dir / "test_auth.py"
    test_auth_py.write_text('''"""
Tests for authentication module.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# In a real project, these would be proper imports
# from src.python.auth import AuthManager, User, Session, AuthenticationError


class MockUser:
    def __init__(self, id, name, email, permissions):
        self.id = id
        self.name = name
        self.email = email
        self.permissions = permissions


class TestAuthManager(unittest.TestCase):
    """Test cases for AuthManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.auth_manager = None  # Would initialize real AuthManager
    
    def test_authenticate_valid_user(self):
        """Test authentication with valid credentials."""
        # Mock test - in real implementation would test actual auth
        email = "test@example.com"
        password = "password123"
        
        # This would be a real test
        expected_result = True
        actual_result = True  # Mock result
        
        self.assertEqual(expected_result, actual_result)
    
    def test_authenticate_invalid_user(self):
        """Test authentication with invalid credentials."""
        email = "invalid@example.com"
        password = "wrongpassword"
        
        # This would test actual authentication failure
        with self.assertRaises(Exception):  # Would be AuthenticationError
            pass  # Would call auth_manager.authenticate(email, password)
    
    def test_session_expiry(self):
        """Test that sessions expire correctly."""
        # Mock session creation and expiry
        session_token = "mock_token_123"
        
        # Would test actual session management
        self.assertTrue(True)  # Placeholder assertion
    
    def test_permission_checking(self):
        """Test permission validation."""
        user = MockUser(1, "Test User", "test@example.com", ["read", "write"])
        
        # Test permission checking logic
        self.assertIn("read", user.permissions)
        self.assertIn("write", user.permissions)
        self.assertNotIn("admin", user.permissions)
    
    @patch('datetime.datetime')
    def test_token_generation(self, mock_datetime):
        """Test secure token generation."""
        mock_datetime.now.return_value = datetime(2025, 1, 17, 12, 0, 0)
        
        # Would test actual token generation
        token = "mock_generated_token"
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 10)


class TestUserModel(unittest.TestCase):
    """Test cases for User model."""
    
    def test_user_creation(self):
        """Test user object creation."""
        user = MockUser(
            id=1,
            name="Test User",
            email="test@example.com", 
            permissions=["read"]
        )
        
        self.assertEqual(user.id, 1)
        self.assertEqual(user.name, "Test User")
        self.assertEqual(user.email, "test@example.com")
        self.assertEqual(user.permissions, ["read"])
    
    def test_user_has_permission(self):
        """Test permission checking."""
        user = MockUser(1, "User", "user@test.com", ["read", "write"])
        
        # Would test actual permission checking method
        self.assertIn("read", user.permissions)
        self.assertNotIn("admin", user.permissions)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
''')

    # JavaScript test
    test_utils_js = tests_dir / "test_utils.js"
    test_utils_js.write_text('''/**
 * Tests for utility functions.
 */

// Mock testing framework (would use Jest/Mocha in real project)
const assert = (condition, message) => {
  if (!condition) {
    throw new Error(message || 'Assertion failed');
  }
};

const describe = (name, fn) => {
  console.log(`\\n--- ${name} ---`);
  fn();
};

const test = (name, fn) => {
  try {
    fn();
    console.log(`✓ ${name}`);
  } catch (error) {
    console.log(`✗ ${name}: ${error.message}`);
  }
};

// Mock utility functions (would import from actual modules)
const formatFileSize = (bytes) => {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${size.toFixed(1)} ${units[unitIndex]}`;
};

const truncateText = (text, maxLength) => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
};

const debounce = (func, delay) => {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
};

// Test suites
describe('File Size Formatting', () => {
  test('formats bytes correctly', () => {
    assert(formatFileSize(500) === '500.0 B');
    assert(formatFileSize(1024) === '1.0 KB');
    assert(formatFileSize(1048576) === '1.0 MB');
    assert(formatFileSize(1073741824) === '1.0 GB');
  });
  
  test('handles edge cases', () => {
    assert(formatFileSize(0) === '0.0 B');
    assert(formatFileSize(1) === '1.0 B');
    assert(formatFileSize(1023) === '1023.0 B');
  });
});

describe('Text Truncation', () => {
  test('truncates long text', () => {
    const longText = 'This is a very long text that should be truncated';
    const result = truncateText(longText, 20);
    assert(result === 'This is a very lo...');
    assert(result.length === 20);
  });
  
  test('preserves short text', () => {
    const shortText = 'Short text';
    const result = truncateText(shortText, 20);
    assert(result === shortText);
  });
  
  test('handles edge cases', () => {
    assert(truncateText('', 10) === '');
    assert(truncateText('abc', 3) === 'abc');
    assert(truncateText('abcd', 3) === '...');
  });
});

describe('Debounce Function', () => {
  test('debounces function calls', (done) => {
    let callCount = 0;
    const increment = () => callCount++;
    const debouncedIncrement = debounce(increment, 100);
    
    // Call multiple times rapidly
    debouncedIncrement();
    debouncedIncrement();
    debouncedIncrement();
    
    // Should only execute once after delay
    setTimeout(() => {
      assert(callCount === 1);
      if (done) done();
    }, 150);
  });
});

// Run tests
console.log('Running JavaScript utility tests...');
try {
  // Note: In real tests, these would be properly structured with async support
  console.log('\\n✓ All tests completed');
} catch (error) {
  console.log(`\\n✗ Test failed: ${error.message}`);
}
''')

def create_script_files(base_dir: Path) -> None:
    """Create utility scripts"""
    scripts_dir = base_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    # Build script
    build_sh = scripts_dir / "build.sh"
    build_sh.write_text('''#!/bin/bash
# Build script for CodeRaptor test project

set -e  # Exit on error

echo "🔨 Building CodeRaptor Test Project..."

# Check dependencies
echo "📋 Checking dependencies..."
if command -v python3 >/dev/null 2>&1; then
    echo "✓ Python 3 found: $(python3 --version)"
else
    echo "❌ Python 3 not found"
    exit 1
fi

if command -v node >/dev/null 2>&1; then
    echo "✓ Node.js found: $(node --version)"
else
    echo "❌ Node.js not found"
    exit 1
fi

# Create virtual environment for Python
echo "🐍 Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Created virtual environment"
fi

source .venv/bin/activate
pip install -q --upgrade pip

# Install Python dependencies (if requirements.txt exists)
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo "✓ Installed Python dependencies"
fi

# Install Node.js dependencies (if package.json exists)
if [ -f "package.json" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install --silent
    echo "✓ Installed Node.js dependencies"
fi

# Run tests
echo "🧪 Running tests..."
if [ -d "tests" ]; then
    python3 -m pytest tests/ -v 2>/dev/null || echo "⚠️  Python tests not configured"
    npm test 2>/dev/null || echo "⚠️  JavaScript tests not configured"
fi

# Build documentation
echo "📚 Building documentation..."
if [ -d "docs" ]; then
    echo "✓ Documentation ready"
fi

echo "✅ Build completed successfully!"
echo ""
echo "📁 Project structure:"
find . -type f -name "*.py" -o -name "*.ts" -o -name "*.js" -o -name "*.md" | head -10
echo ""
echo "🎯 Ready for CodeRaptor indexing!"
''')
    build_sh.chmod(0o755)  # Make executable

    # Deployment script
    deploy_py = scripts_dir / "deploy.py"
    deploy_py.write_text('''#!/usr/bin/env python3
"""
Deployment script for CodeRaptor test project.

Handles building, testing, and deploying the application.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ {' '.join(cmd)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {' '.join(cmd)} failed:")
        print(f"   {e.stderr}")
        return False


def check_dependencies() -> bool:
    """Check that required tools are available."""
    dependencies = [
        (["python3", "--version"], "Python 3"),
        (["node", "--version"], "Node.js"), 
        (["git", "--version"], "Git")
    ]
    
    print("📋 Checking dependencies...")
    all_good = True
    
    for cmd, name in dependencies:
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✓ {name} available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {name} not found")
            all_good = False
    
    return all_good


def build_project() -> bool:
    """Build the project."""
    print("🔨 Building project...")
    
    # Run build script if it exists
    build_script = Path("scripts/build.sh")
    if build_script.exists():
        return run_command(["bash", str(build_script)])
    
    # Otherwise, run basic build steps
    steps = [
        (["python3", "-m", "py_compile"] + 
         [str(f) for f in Path("src/python").glob("*.py")], "Compile Python"),
    ]
    
    for cmd, desc in steps:
        print(f"🔧 {desc}...")
        if not run_command(cmd):
            return False
    
    return True


def run_tests() -> bool:
    """Run test suite."""
    print("🧪 Running tests...")
    
    test_commands = [
        (["python3", "-m", "unittest", "discover", "tests"], "Python tests"),
        (["node", "tests/test_utils.js"], "JavaScript tests")
    ]
    
    all_passed = True
    for cmd, desc in test_commands:
        print(f"🔍 {desc}...")
        if not run_command(cmd):
            print(f"⚠️  {desc} failed")
            all_passed = False
    
    return all_passed


def create_package() -> bool:
    """Create deployment package."""
    print("📦 Creating deployment package...")
    
    # Create dist directory
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()
    
    # Copy important files
    files_to_copy = [
        "src/",
        "config/",
        "docs/",
        "README.md"
    ]
    
    for item in files_to_copy:
        src = Path(item)
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, dist_dir / src.name)
            else:
                shutil.copy2(src, dist_dir)
            print(f"✓ Copied {item}")
    
    print(f"✓ Package created in {dist_dir}")
    return True


def deploy(environment: str = "staging") -> bool:
    """Deploy to specified environment."""
    print(f"🚀 Deploying to {environment}...")
    
    if environment == "staging":
        print("📤 Deploying to staging server...")
        # In real deployment, this would:
        # - Upload files to staging server
        # - Run deployment scripts
        # - Update configuration
        # - Restart services
        print("✓ Staged deployment completed")
        
    elif environment == "production":
        print("📤 Deploying to production...")
        # Production deployment steps
        print("✓ Production deployment completed")
        
    else:
        print(f"❌ Unknown environment: {environment}")
        return False
    
    return True


def main():
    """Main deployment workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy CodeRaptor test project")
    parser.add_argument(
        "--environment", "-e",
        choices=["staging", "production"],
        default="staging",
        help="Deployment environment"
    )
    parser.add_argument(
        "--skip-tests", 
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--build-only",
        action="store_true", 
        help="Only build, don't deploy"
    )
    
    args = parser.parse_args()
    
    print("🚀 CodeRaptor Test Project Deployment")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return 1
    
    # Build project
    if not build_project():
        print("❌ Build failed")
        return 1
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            print("⚠️  Tests failed, continuing anyway...")
    
    # Create package
    if not create_package():
        print("❌ Package creation failed")
        return 1
    
    # Deploy
    if not args.build_only:
        if not deploy(args.environment):
            print("❌ Deployment failed")
            return 1
    
    print("✅ Deployment completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''')

def main():
    """Main function to create test directory structure."""
    import sys
    
    # Get output directory from command line or use default
    if len(sys.argv) > 1:
        base_output_dir = Path(sys.argv[1])
    else:
        base_output_dir = Path("./")
    
    # Create samples folder structure
    samples_dir = base_output_dir / "samples"
    output_dir = samples_dir / "code_test_project"
    
    print(f"🚀 Creating CodeRaptor test directory: {output_dir}")
    
    # Create base directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create all file types
    print("📝 Creating Python files...")
    create_python_files(output_dir)
    
    print("🌐 Creating JavaScript/TypeScript files...")
    create_javascript_files(output_dir)
    
    print("📚 Creating documentation...")
    create_documentation_files(output_dir)
    
    print("⚙️  Creating configuration files...")
    create_config_files(output_dir)
    
    print("🧪 Creating test files...")
    create_test_files(output_dir)
    
    print("🔧 Creating utility scripts...")
    create_script_files(output_dir)
    
    # Create additional files
    gitignore = output_dir / ".gitignore"
    gitignore.write_text('''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.eslintcache

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
logs/
*.log

# Database
*.db
*.sqlite
*.sqlite3

# CodeRaptor
.raptor_index/
demo_index/
''')
    
    package_json = output_dir / "package.json"
    package_json.write_text('''{
  "name": "code-raptor-test-project",
  "version": "1.0.0",
  "description": "Test project for CodeRaptor file processing and indexing",
  "main": "src/frontend/App.tsx",
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "node tests/test_utils.js",
    "lint": "eslint src/frontend --ext .ts,.tsx,.js,.jsx",
    "deploy": "python3 scripts/deploy.py"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.8.0",
    "typescript": "^4.9.0"
  },
  "devDependencies": {
    "eslint": "^8.0.0",
    "jest": "^29.0.0",
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0"
  },
  "keywords": ["code", "indexing", "raptor", "test"],
  "author": "CodeRaptor Team",
  "license": "MIT"
}''')
    
    requirements_txt = output_dir / "requirements.txt"
    requirements_txt.write_text('''# Python dependencies for CodeRaptor test project

# Core dependencies
pyyaml>=6.0
sqlite3  # Usually included with Python

# Development dependencies
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=6.0.0
mypy>=1.0.0

# Optional dependencies for extended functionality
requests>=2.28.0
numpy>=1.21.0
pandas>=1.5.0

# Web framework (if building API)
flask>=2.2.0
flask-cors>=3.0.0
gunicorn>=20.1.0
''')
    
    # Create a simple Makefile
    makefile = output_dir / "Makefile"
    makefile.write_text('''# Makefile for CodeRaptor test project

.PHONY: install test build clean lint format help

# Default target
help:
	@echo "Available targets:"
	@echo "  install  - Install dependencies"
	@echo "  test     - Run tests"
	@echo "  build    - Build project"
	@echo "  clean    - Clean build artifacts"
	@echo "  lint     - Run linters"
	@echo "  format   - Format code"
	@echo "  raptor   - Run CodeRaptor indexing"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	python3 -m pip install -r requirements.txt
	@echo "Installing Node.js dependencies..."
	npm install
	@echo "✅ Dependencies installed"

# Run tests
test:
	@echo "Running Python tests..."
	python3 -m pytest tests/ -v
	@echo "Running JavaScript tests..."
	node tests/test_utils.js
	@echo "✅ Tests completed"

# Build project
build:
	@echo "Building project..."
	bash scripts/build.sh
	@echo "✅ Build completed"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf dist/
	rm -rf build/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	@echo "✅ Cleaned"

# Run linters
lint:
	@echo "Running Python linters..."
	flake8 src/python/ tests/
	mypy src/python/
	@echo "Running JavaScript linters..."
	npx eslint src/frontend/
	@echo "✅ Linting completed"

# Format code
format:
	@echo "Formatting Python code..."
	black src/python/ tests/
	@echo "Formatting JavaScript code..."
	npx prettier --write src/frontend/
	@echo "✅ Code formatted"

# Run CodeRaptor indexing
raptor:
	@echo "Running CodeRaptor indexing..."
	code-raptor index . --config file_processor_config.yaml
	@echo "✅ Indexing completed"

# Development server
dev:
	@echo "Starting development servers..."
	python3 src/python/main.py --config config/development.json &
	npm start &
	@echo "✅ Development servers started"
''')

    # Create a samples README
    samples_readme = samples_dir / "README.md"
    samples_readme.write_text('''# CodeRaptor Samples

This directory contains sample projects for testing and demonstrating CodeRaptor functionality.

## Test Project

The `code_test_project/` directory contains a comprehensive multi-language codebase designed to test all aspects of CodeRaptor:

### Features Tested
- **Multi-language Support**: Python, TypeScript, JavaScript, Markdown, YAML, JSON, Shell
- **Code Structures**: Functions, classes, components, modules
- **Documentation**: README files, API docs, inline comments
- **Configuration**: YAML configs, JSON settings
- **Build Systems**: Makefiles, package.json, requirements.txt
- **Testing**: Unit tests, integration tests

### Usage

```bash
# Navigate to samples directory
cd samples/

# Index the test project
code-raptor index code_test_project/

# Search across all languages
code-raptor search "authentication"

# Get statistics
code-raptor stats code_test_project/
```

### Expected Results

When processing the test project, CodeRaptor should:

1. **Detect Languages**: Python, TypeScript, JavaScript, Markdown, YAML, JSON, Shell
2. **Create Chunks**: 25+ semantic code chunks (functions, classes, components)
3. **Build Hierarchy**: 3-4 level RAPTOR tree structure
4. **Enable Search**: Cross-language semantic search
5. **Generate Stats**: Language distribution, file metrics

This provides a comprehensive test case for validating CodeRaptor's file processing, chunking, and indexing capabilities across multiple programming languages and file types.

## Adding More Samples

To add additional test cases:

1. Create new directories under `samples/`
2. Include diverse file types and structures
3. Add language-specific features to test
4. Document expected behavior and results

Each sample should focus on specific aspects of CodeRaptor functionality while maintaining realistic code patterns and project structures.
''')
    
    print("\n✅ Test directory structure created successfully!")
    
    # Print summary
    created_files = list(output_dir.rglob("*"))
    created_files = [f for f in created_files if f.is_file()]
    
    print(f"\n📊 Summary:")
    print(f"   📁 Created directory: {output_dir}")
    print(f"   📄 Total files: {len(created_files)}")
    
    # Count by language
    languages = {}
    for file in created_files:
        ext = file.suffix.lower()
        if ext == '.py':
            lang = 'Python'
        elif ext in ['.ts', '.tsx', '.js', '.jsx']:
            lang = 'TypeScript/JavaScript'
        elif ext == '.md':
            lang = 'Markdown'
        elif ext in ['.yaml', '.yml']:
            lang = 'YAML'
        elif ext == '.json':
            lang = 'JSON'
        elif ext == '.sh':
            lang = 'Shell'
        else:
            lang = 'Other'
        
        languages[lang] = languages.get(lang, 0) + 1
    
    print(f"\n🗣️  Languages:")
    for lang, count in sorted(languages.items()):
        print(f"   {lang}: {count} files")
    
    print(f"\n🎯 Usage:")
    print(f"   cd {output_dir}")
    print(f"   make install     # Install dependencies")
    print(f"   make test        # Run tests")
    print(f"   make raptor      # Index with CodeRaptor")
    
    print(f"\n📁 Structure:")
    print(f"   {base_output_dir}/")
    print(f"   └── samples/")
    print(f"       ├── README.md")
    print(f"       └── code_test_project/")
    print(f"           ├── src/")
    print(f"           ├── docs/")
    print(f"           ├── config/")
    print(f"           ├── tests/")
    print(f"           └── scripts/")
    
    print(f"\n🦖 Ready for CodeRaptor testing!")

if __name__ == "__main__":
    main()