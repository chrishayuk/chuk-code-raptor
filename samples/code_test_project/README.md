# CodeRaptor Test Project

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
