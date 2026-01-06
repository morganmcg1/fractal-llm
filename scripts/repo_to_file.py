#!/usr/bin/env python3
"""
Convert the entire repo into a single markdown file for LLM consumption.

Respects .gitignore (via git ls-files) and excludes third_party/ directory.
Output includes a tree structure and all file contents with clear markers.
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import simple_parsing as sp
from rich.console import Console
from rich.panel import Panel

console = Console()

# File extensions to include (text-based files an LLM can process)
TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".sh", ".bash", ".zsh", ".fish",
    ".html", ".css", ".js", ".ts", ".tsx", ".jsx",
    ".sql", ".graphql",
    ".dockerfile", ".env.example",
    ".gitignore", ".gitattributes",
    ".cfg", ".ini", ".conf",
    ".rst", ".tex",
    ".ipynb",
}

# Filenames without extension to include
TEXT_FILENAMES = {
    "Dockerfile", "Makefile", "LICENSE", "Procfile",
    ".env.example", ".gitignore", ".python-version",
}

# Directories to always exclude (in addition to third_party)
EXCLUDE_DIRS = {
    "third_party", ".git", "__pycache__", "node_modules", ".venv",
    "_nanochat_shims", "crwv_cli",
}

# Files to always exclude
EXCLUDE_FILES = {
    ".devcontainer.json",
    ".env.example",
    "AGENTS.md",
    "claude-research.md",
    "codex-research.md",
    "research.md",
    "nanochat_modal.py",
    "chat.py",
    "probe_batch_size.sh",
    "repro_check.sh",
    "the_boundary_of_neural_network_trainability_is_fractal.ipynb",
}

# Map file extensions to markdown language hints
EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".json": "json",
    ".jsonl": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".ipynb": "json",
    ".gitignore": "gitignore",
}


@dataclass
class Args:
    """Convert repo to a single markdown file for LLM consumption"""
    output: str = "repo_snapshot.md"  # Output file path
    exclude_dirs: list[str] | None = None  # Additional directories to exclude
    include_files: list[str] = field(default_factory=list)  # Extra files to include (even from excluded dirs)
    include_binary_names: bool = False  # Include binary file names (without content)


def get_tracked_files() -> list[str]:
    """Get list of git-tracked files (respects .gitignore)."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split("\n")


def should_include_file(filepath: str, exclude_dirs: set[str], exclude_files: set[str]) -> bool:
    """Check if file should be included based on path and extension."""
    path = Path(filepath)
    name = path.name

    # Check if file is explicitly excluded
    if name in exclude_files:
        return False

    # Check if in excluded directory
    for part in path.parts:
        if part in exclude_dirs:
            return False

    # Check extension or filename
    ext = path.suffix.lower()

    return ext in TEXT_EXTENSIONS or name in TEXT_FILENAMES


def build_tree_markdown(files: list[str]) -> str:
    """Build a markdown tree representation of the file structure."""
    lines = ["## Repository Structure", "", "```"]

    # Group by directory
    dirs: dict[str, list[str]] = {}
    for f in sorted(files):
        path = Path(f)
        parent = str(path.parent) if path.parent != Path(".") else "."
        if parent not in dirs:
            dirs[parent] = []
        dirs[parent].append(path.name)

    # Build tree output
    for dir_path in sorted(dirs.keys()):
        if dir_path == ".":
            lines.append("./")
        else:
            indent = "  " * (dir_path.count("/"))
            lines.append(f"{indent}{dir_path}/")

        for filename in sorted(dirs[dir_path]):
            indent = "  " * (dir_path.count("/") + 1) if dir_path != "." else "  "
            lines.append(f"{indent}{filename}")

    lines.append("```")
    return "\n".join(lines)


def read_file_content(filepath: str) -> str | None:
    """Read file content, return None if binary or unreadable."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, IOError):
        return None


def get_language_hint(filepath: str) -> str:
    """Get markdown code block language hint for a file."""
    ext = Path(filepath).suffix.lower()
    return EXT_TO_LANG.get(ext, "")


def format_file_section_markdown(filepath: str, content: str) -> str:
    """Format a file's content as a markdown section with code block."""
    lang = get_language_hint(filepath)
    return f"\n### `{filepath}`\n\n```{lang}\n{content}\n```\n"


def main():
    args = sp.parse(Args)

    console.rule("[bold blue]Repo to Markdown Converter")

    # Build exclusion set
    exclude_dirs = EXCLUDE_DIRS.copy()
    if args.exclude_dirs:
        exclude_dirs.update(args.exclude_dirs)

    console.print(f"Excluding directories: {sorted(exclude_dirs)}")

    # Get tracked files
    all_files = get_tracked_files()
    console.print(f"Total git-tracked files: {len(all_files)}")

    # Filter files
    included_files = [f for f in all_files if should_include_file(f, exclude_dirs, EXCLUDE_FILES)]

    # Add explicitly included files
    for extra_file in args.include_files:
        if extra_file not in included_files and Path(extra_file).exists():
            included_files.append(extra_file)
            console.print(f"[yellow]Added extra file:[/yellow] {extra_file}")

    console.print(f"Files after filtering: {len(included_files)}")

    # Build output
    output_parts = []

    # Header
    output_parts.append(f"""# Repository Snapshot: fractal-llm

> **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
>
> **Description:** Exploring fractal training dynamics in LLMs

This file contains the complete source code of the repository, formatted for LLM consumption.
""")

    # Tree structure
    output_parts.append(build_tree_markdown(included_files))

    # File contents header
    output_parts.append("\n---\n")
    output_parts.append("## File Contents\n")

    # File contents
    files_included = 0
    files_skipped = 0
    total_chars = 0

    for filepath in sorted(included_files):
        content = read_file_content(filepath)
        if content is not None:
            output_parts.append(format_file_section_markdown(filepath, content))
            files_included += 1
            total_chars += len(content)
        else:
            files_skipped += 1
            if args.include_binary_names:
                output_parts.append(f"\n### `{filepath}`\n\n*[Binary file]*\n")

    # Write output
    output_content = "\n".join(output_parts)
    output_path = Path(args.output)
    output_path.write_text(output_content, encoding="utf-8")

    # Summary
    console.print(Panel(
        f"[green]✓ Generated:[/green] {output_path}\n"
        f"[blue]Files included:[/blue] {files_included}\n"
        f"[yellow]Files skipped (binary):[/yellow] {files_skipped}\n"
        f"[cyan]Total characters:[/cyan] {total_chars:,}\n"
        f"[cyan]Output size:[/cyan] {len(output_content):,} chars",
        title="Summary"
    ))


if __name__ == "__main__":
    main()
