# CodeRaptor Samples

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
