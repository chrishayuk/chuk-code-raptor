"""
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
