#!/usr/bin/env python3
"""
Enhanced Chuk CLI with Rich Output and Better UX
================================================

Improved version with better visual output, error handling, and user experience.
"""

import click
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Rich imports for better CLI experience
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax
from rich.text import Text
from rich import box
from rich.prompt import Confirm

# File system monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Git operations
import git

# Your existing systems
from chuk_code_raptor.chunking import ChunkingEngine, ChunkingConfig, PRECISE_CONFIG
from chuk_code_raptor.graph.builder import CPGBuilder
from chuk_code_raptor.raptor.builder import RaptorBuilder

console = Console()

@dataclass
class IndexedRepository:
    """Represents an indexed repository with rich metadata"""
    name: str
    path: str
    repo_url: Optional[str]
    indexed_at: datetime
    file_count: int
    chunk_count: int
    languages: List[str]
    status: str
    last_modified: datetime
    quality_score: float = 0.0
    complexity_score: float = 0.0

class ChukCLIEnhanced:
    """Enhanced CLI with rich output and better UX"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".chuk"
        self.config_dir.mkdir(exist_ok=True)
        
        self.index_db_path = self.config_dir / "index.json"
        self.chunks_dir = self.config_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        
        # Initialize with progress indication
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing Chuk CLI...", total=None)
            
            self.chunking_engine = ChunkingEngine(PRECISE_CONFIG)
            self.indexed_repos: Dict[str, IndexedRepository] = {}
            self._load_index()
            
            progress.update(task, completed=True)
    
    def _load_index(self):
        """Load existing index with error handling"""
        if not self.index_db_path.exists():
            return
            
        try:
            with open(self.index_db_path, 'r') as f:
                data = json.load(f)
            
            for repo_data in data.get('repositories', []):
                repo = IndexedRepository(
                    name=repo_data['name'],
                    path=repo_data['path'],
                    repo_url=repo_data.get('repo_url'),
                    indexed_at=datetime.fromisoformat(repo_data['indexed_at']),
                    file_count=repo_data['file_count'],
                    chunk_count=repo_data['chunk_count'],
                    languages=repo_data.get('languages', []),
                    status=repo_data['status'],
                    last_modified=datetime.fromisoformat(repo_data['last_modified']),
                    quality_score=repo_data.get('quality_score', 0.0),
                    complexity_score=repo_data.get('complexity_score', 0.0)
                )
                self.indexed_repos[repo.name] = repo
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not load index: {e}[/yellow]")
    
    def _save_index(self):
        """Save index to disk with error handling"""
        try:
            data = {
                'repositories': [asdict(repo) for repo in self.indexed_repos.values()],
                'version': '0.1.0',
                'saved_at': datetime.now().isoformat()
            }
            
            # Convert datetime objects to strings
            for repo_data in data['repositories']:
                repo_data['indexed_at'] = repo_data['indexed_at'].isoformat() if isinstance(repo_data['indexed_at'], datetime) else repo_data['indexed_at']
                repo_data['last_modified'] = repo_data['last_modified'].isoformat() if isinstance(repo_data['last_modified'], datetime) else repo_data['last_modified']
            
            with open(self.index_db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            console.print(f"[red]❌ Error saving index: {e}[/red]")
    
    def _clone_repo(self, repo_url: str) -> Path:
        """Clone repository with progress indication"""
        temp_dir = Path(tempfile.mkdtemp(prefix="chuk_clone_"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Cloning {repo_url}...", total=None)
            
            try:
                git.Repo.clone_from(repo_url, temp_dir)
                progress.update(task, completed=True)
                return temp_dir
            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise click.ClickException(f"Failed to clone repository: {e}")
    
    def index_repository(self, path_or_url: str, name: Optional[str] = None) -> IndexedRepository:
        """Index repository with rich progress display"""
        is_remote = path_or_url.startswith(('http://', 'https://', 'git@'))
        repo_name = name or Path(path_or_url).stem
        
        console.print(f"\n[bold blue]🚀 Starting analysis of: {path_or_url}[/bold blue]")
        
        # Handle remote vs local
        if is_remote:
            local_path = self._clone_repo(path_or_url)
            repo_url = path_or_url
        else:
            local_path = Path(path_or_url).resolve()
            if not local_path.exists():
                raise click.ClickException(f"Path does not exist: {local_path}")
            repo_url = None
        
        try:
            # Collect files
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                scan_task = progress.add_task("Scanning files...", total=None)
                files = self._collect_files(local_path)
                progress.update(scan_task, completed=True)
            
            if not files:
                raise click.ClickException("No supported files found")
            
            # Show file statistics
            file_stats = self._analyze_file_types(files)
            self._display_file_stats(file_stats, len(files))
            
            # Process files with detailed progress
            all_chunks = []
            processed_files = 0
            errors = []
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("{task.completed}/{task.total} files"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing files...", total=len(files))
                
                for file_path in files:
                    try:
                        chunks = self.chunking_engine.chunk_file(str(file_path))
                        all_chunks.extend(chunks)
                        processed_files += 1
                    except Exception as e:
                        errors.append(f"{file_path}: {e}")
                    
                    progress.update(task, advance=1)
            
            if errors:
                console.print(f"[yellow]⚠️  {len(errors)} files had processing errors[/yellow]")
            
            # Build semantic structures
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                # CPG
                cpg_task = progress.add_task("Building code property graph...", total=None)
                cpg_builder = CPGBuilder()
                cpg = cpg_builder.build_from_chunks(all_chunks)
                progress.update(cpg_task, completed=True)
                
                # RAPTOR
                raptor_task = progress.add_task("Building RAPTOR hierarchy...", total=None)
                raptor_builder = RaptorBuilder(cpg)
                raptor_summary = raptor_builder.build_from_chunks(all_chunks)
                progress.update(raptor_task, completed=True)
            
            # Calculate quality metrics
            quality_scores = [chunk.calculate_overall_quality_score() for chunk in all_chunks if hasattr(chunk, 'calculate_overall_quality_score')]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            complexity_levels = [chunk.complexity_level.value for chunk in all_chunks if hasattr(chunk, 'complexity_level')]
            complex_ratio = len([c for c in complexity_levels if c in ['complex', 'very_complex']]) / len(complexity_levels) if complexity_levels else 0.0
            
            # Save data
            self._save_repository_data(repo_name, all_chunks, cpg, raptor_summary)
            
            # Create repository record
            repo = IndexedRepository(
                name=repo_name,
                path=str(local_path),
                repo_url=repo_url,
                indexed_at=datetime.now(),
                file_count=processed_files,
                chunk_count=len(all_chunks),
                languages=list(file_stats.keys()),
                status="indexed",
                last_modified=datetime.now(),
                quality_score=avg_quality,
                complexity_score=complex_ratio
            )
            
            self.indexed_repos[repo_name] = repo
            self._save_index()
            
            # Display success summary
            self._display_success_summary(repo, file_stats, len(errors))
            
            return repo
            
        finally:
            if is_remote and local_path.exists():
                shutil.rmtree(local_path, ignore_errors=True)
    
    def _collect_files(self, root_path: Path) -> List[Path]:
        """Collect files with improved filtering"""
        supported_extensions = set(self.chunking_engine.get_supported_extensions())
        files = []
        
        ignore_patterns = {
            '.git', 'node_modules', '__pycache__', '.pytest_cache',
            'venv', '.venv', 'build', 'dist', '.DS_Store', 'target',
            '.idea', '.vscode', 'coverage', '.nyc_output'
        }
        
        for file_path in root_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in supported_extensions:
                # Check if any parent directory matches ignore patterns
                if any(ignore in str(file_path) for ignore in ignore_patterns):
                    continue
                files.append(file_path)
        
        return files
    
    def _analyze_file_types(self, files: List[Path]) -> Dict[str, int]:
        """Analyze file types for statistics"""
        stats = {}
        for file_path in files:
            ext = file_path.suffix.lower()
            stats[ext] = stats.get(ext, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    
    def _display_file_stats(self, file_stats: Dict[str, int], total_files: int):
        """Display file statistics in a nice table"""
        table = Table(title="📊 File Analysis", box=box.ROUNDED)
        table.add_column("Extension", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")
        
        for ext, count in file_stats.items():
            percentage = (count / total_files) * 100
            table.add_row(ext, str(count), f"{percentage:.1f}%")
        
        console.print(table)
    
    def _save_repository_data(self, repo_name: str, chunks: List, cpg, raptor_summary):
        """Save repository analysis data"""
        repo_chunks_dir = self.chunks_dir / repo_name
        repo_chunks_dir.mkdir(exist_ok=True)
        
        # Save chunks
        chunks_file = repo_chunks_dir / "chunks.json"
        with open(chunks_file, 'w') as f:
            chunks_data = [chunk.to_dict() for chunk in chunks]
            json.dump(chunks_data, f, indent=2)
        
        # Save analysis
        analysis_file = repo_chunks_dir / "analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump({
                'cpg_summary': cpg.get_summary(),
                'raptor_summary': raptor_summary,
                'chunking_stats': self.chunking_engine.get_statistics(),
                'analysis_timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def _display_success_summary(self, repo: IndexedRepository, file_stats: Dict[str, int], error_count: int):
        """Display success summary with rich formatting"""
        panel_content = f"""[green]✅ Successfully indexed repository![/green]

[bold]Repository:[/bold] {repo.name}
[bold]Files processed:[/bold] {repo.file_count}
[bold]Semantic chunks:[/bold] {repo.chunk_count}
[bold]Languages:[/bold] {', '.join(file_stats.keys())}
[bold]Quality score:[/bold] {repo.quality_score:.2f}/1.0
[bold]Complexity ratio:[/bold] {repo.complexity_score:.2f}"""

        if error_count > 0:
            panel_content += f"\n[yellow]⚠️  {error_count} files had processing errors[/yellow]"
        
        console.print(Panel(panel_content, title="🎉 Indexing Complete", border_style="green"))

    def search_code(self, query: str, repo_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Enhanced search with better result formatting"""
        results = []
        repos_to_search = [repo_name] if repo_name else list(self.indexed_repos.keys())
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Searching...", total=len(repos_to_search))
            
            for name in repos_to_search:
                if name not in self.indexed_repos:
                    continue
                
                repo_chunks_dir = self.chunks_dir / name
                chunks_file = repo_chunks_dir / "chunks.json"
                
                if chunks_file.exists():
                    with open(chunks_file, 'r') as f:
                        chunks_data = json.load(f)
                    
                    query_lower = query.lower()
                    for chunk_data in chunks_data:
                        content = chunk_data.get('content', '')
                        if query_lower in content.lower():
                            # Calculate relevance score
                            score = content.lower().count(query_lower) / len(content.split())
                            
                            results.append({
                                'repository': name,
                                'file_path': chunk_data.get('file_path', ''),
                                'chunk_type': chunk_data.get('chunk_type', ''),
                                'chunk_id': chunk_data.get('id', ''),
                                'content': content,
                                'start_line': chunk_data.get('start_line', 0),
                                'end_line': chunk_data.get('end_line', 0),
                                'relevance_score': score
                            })
                
                progress.update(task, advance=1)
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:limit]
    
    def display_search_results(self, results: List[Dict[str, Any]], query: str):
        """Display search results with rich formatting"""
        if not results:
            console.print(f"[yellow]🤷 No results found for: '{query}'[/yellow]")
            return
        
        console.print(f"\n[bold green]🎯 Found {len(results)} results for: '{query}'[/bold green]\n")
        
        for i, result in enumerate(results, 1):
            # Create a panel for each result
            content_preview = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            
            panel_content = f"""[bold]File:[/bold] {result['file_path']}
[bold]Lines:[/bold] {result['start_line']}-{result['end_line']}
[bold]Type:[/bold] {result['chunk_type']}
[bold]Repository:[/bold] {result['repository']}

[dim]{content_preview}[/dim]"""
            
            console.print(Panel(
                panel_content,
                title=f"Result {i}",
                border_style="blue"
            ))
    
    def display_status(self):
        """Display repository status with rich table"""
        if not self.indexed_repos:
            console.print(Panel(
                "[yellow]No repositories indexed yet[/yellow]\n\nRun [bold]chuk index <path>[/bold] to get started!",
                title="📚 Repository Status",
                border_style="yellow"
            ))
            return
        
        table = Table(title="📚 Indexed Repositories", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Files", justify="right", style="blue")
        table.add_column("Chunks", justify="right", style="magenta")
        table.add_column("Quality", justify="right", style="yellow")
        table.add_column("Languages", style="dim")
        table.add_column("Indexed", style="dim")
        
        for repo in self.indexed_repos.values():
            status = "✅ Indexed" if repo.status == "indexed" else "⚠️ Error"
            quality_color = "green" if repo.quality_score > 0.7 else "yellow" if repo.quality_score > 0.5 else "red"
            
            table.add_row(
                repo.name,
                status,
                str(repo.file_count),
                str(repo.chunk_count),
                f"[{quality_color}]{repo.quality_score:.2f}[/{quality_color}]",
                ", ".join(repo.languages[:3]),  # Show first 3 languages
                repo.indexed_at.strftime('%Y-%m-%d')
            )
        
        console.print(table)

# CLI Commands with enhanced UX

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """🚀 [bold blue]Chuk Code Raptor[/bold blue] - Semantic Code Analysis CLI
    
    A powerful tool for understanding your codebase through semantic analysis.
    """
    pass

@cli.command()
@click.argument('path_or_url')
@click.option('--name', '-n', help='Custom name for the repository')
@click.option('--force', '-f', is_flag=True, help='Force re-indexing if already exists')
def index(path_or_url, name, force):
    """📚 Index a local folder or GitHub repository
    
    Examples:
      chuk index ./src
      chuk index https://github.com/user/repo
      chuk index ~/projects/my-app --name my-app
    """
    chuk = ChukCLIEnhanced()
    
    repo_name = name or Path(path_or_url).stem
    
    # Check if already indexed
    if repo_name in chuk.indexed_repos and not force:
        if Confirm.ask(f"Repository '{repo_name}' is already indexed. Re-index?"):
            force = True
        else:
            console.print("[yellow]Skipping indexing[/yellow]")
            return
    
    try:
        chuk.index_repository(path_or_url, name)
    except Exception as e:
        console.print(f"[red]❌ Indexing failed: {e}[/red]")
        raise click.Abort()

@cli.command()
@click.argument('query')
@click.option('--repo', '-r', help='Search specific repository')
@click.option('--limit', '-l', default=10, help='Maximum number of results')
def search(query, repo, limit):
    """🔍 Search indexed code using natural language
    
    Examples:
      chuk search "authentication logic"
      chuk search "error handling" --repo my-app
      chuk search "database connection" --limit 5
    """
    chuk = ChukCLIEnhanced()
    
    results = chuk.search_code(query, repo, limit)
    chuk.display_search_results(results, query)

@cli.command()
def status():
    """📊 Show repository indexing status"""
    chuk = ChukCLIEnhanced()
    chuk.display_status()

@cli.command()
@click.argument('path')
def monitor(path):
    """👀 Monitor a folder for changes (coming soon)"""
    console.print("[yellow]🚧 Monitoring feature coming soon![/yellow]")

if __name__ == '__main__':
    cli()