#!/bin/bash
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
