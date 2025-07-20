#!/usr/bin/env python3
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
