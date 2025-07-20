# API Documentation

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
