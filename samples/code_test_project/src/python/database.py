"""
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
