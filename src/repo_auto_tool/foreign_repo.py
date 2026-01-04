"""Foreign repository support for working with external git repositories.

This module handles:
- Cloning repositories from URLs (GitHub, GitLab, etc.)
- Detecting repository conventions (style, patterns, tooling)
- Setting up fork workflow for pull requests
- Managing temporary/workspace directories

All operations are defensively coded with robust error handling.
"""

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import (
    ConfigurationError,
    GitOperationError,
    RepoAutoToolError,
)

logger = logging.getLogger(__name__)


class ForeignRepoError(RepoAutoToolError):
    """Error related to foreign repository operations."""

    def __init__(self, message: str, url: str | None = None, details: str | None = None):
        super().__init__(message)
        self.url = url
        self.details = details


@dataclass
class RepoConventions:
    """Detected repository conventions and patterns.

    Auto-detected from the codebase to ensure changes follow existing patterns.
    """
    # Code style
    indent_style: str = "spaces"  # "spaces" or "tabs"
    indent_size: int = 4
    quote_style: str = "double"  # "single" or "double"
    line_ending: str = "lf"  # "lf" or "crlf"
    max_line_length: int = 88  # Default to Black's 88

    # Naming conventions
    naming_style: str = "snake_case"  # "snake_case", "camelCase", "PascalCase"
    uses_emojis: bool = False

    # Project type
    project_type: str = "unknown"  # "library", "cli", "web_app", "api", "monorepo"
    primary_language: str = "unknown"

    # Tooling (auto-detected)
    package_manager: str | None = None  # "pip", "poetry", "npm", "yarn", "pnpm", etc.
    test_framework: str | None = None  # "pytest", "unittest", "jest", "mocha", etc.
    linter: str | None = None  # "ruff", "flake8", "eslint", etc.
    formatter: str | None = None  # "black", "prettier", etc.
    build_tool: str | None = None  # "setuptools", "webpack", "cargo", etc.

    # Patterns
    has_ci: bool = False
    ci_platform: str | None = None  # "github_actions", "gitlab_ci", "circleci"
    has_pre_commit: bool = False
    has_type_hints: bool = False
    docstring_style: str | None = None  # "google", "numpy", "sphinx", None

    # File structure
    src_layout: bool = False  # Uses src/ directory
    test_location: str | None = None  # "tests/", "test/", "src/*/tests/", etc.

    # Metadata
    detected_from_files: list[str] = field(default_factory=list)
    confidence: float = 0.5  # 0.0 to 1.0

    def to_prompt_guidance(self) -> str:
        """Generate guidance for Claude based on detected conventions."""
        lines = ["Follow these repository conventions:"]

        # Code style
        if self.indent_style == "tabs":
            lines.append("- Use tabs for indentation")
        else:
            lines.append(f"- Use {self.indent_size} spaces for indentation")

        lines.append(f"- Use {self.quote_style} quotes for strings")
        lines.append(f"- Keep lines under {self.max_line_length} characters")

        # Emojis
        if not self.uses_emojis:
            lines.append("- Do NOT use emojis in code, comments, or commit messages")

        # Naming
        if self.naming_style == "snake_case":
            lines.append("- Use snake_case for variables and functions")
        elif self.naming_style == "camelCase":
            lines.append("- Use camelCase for variables and functions")

        # Type hints
        if self.has_type_hints:
            lines.append("- Add type hints to function signatures")
        else:
            lines.append("- Do not add type hints unless asked")

        # Tooling guidance
        if self.test_framework:
            lines.append(f"- Use {self.test_framework} for tests")
        if self.linter:
            lines.append(f"- Code must pass {self.linter}")
        if self.formatter:
            lines.append(f"- Code will be formatted with {self.formatter}")

        # Project structure
        if self.src_layout:
            lines.append("- Follow src/ layout convention")
        if self.test_location:
            lines.append(f"- Place tests in {self.test_location}")

        return "\n".join(lines)


@dataclass
class ClonedRepo:
    """Information about a cloned repository."""
    original_url: str
    local_path: Path
    owner: str | None = None
    repo_name: str | None = None
    default_branch: str = "main"
    is_fork: bool = False
    conventions: RepoConventions | None = None
    workspace_dir: Path | None = None  # Parent temp dir if using workspace


class ForeignRepoManager:
    """Manages operations on foreign (external) repositories.

    Handles cloning, convention detection, and cleanup.
    All methods are defensively coded to handle errors gracefully.
    """

    # URL patterns for various git hosts
    URL_PATTERNS = [
        # GitHub: https://github.com/owner/repo or git@github.com:owner/repo
        (r"(?:https?://)?(?:www\.)?github\.com[/:]([^/]+)/([^/.]+)(?:\.git)?/?", "github"),
        # GitLab: https://gitlab.com/owner/repo
        (r"(?:https?://)?(?:www\.)?gitlab\.com[/:]([^/]+)/([^/.]+)(?:\.git)?/?", "gitlab"),
        # Bitbucket: https://bitbucket.org/owner/repo
        (r"(?:https?://)?(?:www\.)?bitbucket\.org[/:]([^/]+)/([^/.]+)(?:\.git)?/?", "bitbucket"),
        # Generic git URL
        (r"(?:https?|git)://[^/]+/(.+?)(?:\.git)?/?$", "generic"),
    ]

    def __init__(
        self,
        workspace_dir: Path | None = None,
        cleanup_on_exit: bool = True,
    ):
        """Initialize the foreign repo manager.

        Args:
            workspace_dir: Directory to clone repos into. If None, uses temp dir.
            cleanup_on_exit: Whether to clean up cloned repos on exit.
        """
        self.workspace_dir = workspace_dir
        self.cleanup_on_exit = cleanup_on_exit
        self._cloned_repos: list[ClonedRepo] = []
        self._temp_dirs: list[Path] = []

    def parse_repo_url(self, url: str) -> tuple[str | None, str | None, str]:
        """Parse a repository URL to extract owner, repo name, and host.

        Args:
            url: Git repository URL (HTTPS or SSH).

        Returns:
            Tuple of (owner, repo_name, host_type).
            Owner and repo_name may be None for generic URLs.
        """
        if not url:
            return None, None, "unknown"

        url = url.strip()

        for pattern, host_type in self.URL_PATTERNS:
            match = re.match(pattern, url, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    return groups[0], groups[1], host_type
                elif len(groups) == 1:
                    # Generic URL - extract repo name from path
                    path_parts = groups[0].rstrip("/").split("/")
                    repo_name = path_parts[-1] if path_parts else None
                    return None, repo_name, host_type

        return None, None, "unknown"

    def is_url(self, path_or_url: str) -> bool:
        """Check if the given string is a URL rather than a local path.

        Args:
            path_or_url: String that might be a URL or local path.

        Returns:
            True if it looks like a URL.
        """
        if not path_or_url:
            return False

        path_or_url = path_or_url.strip()

        # Obvious URL patterns
        if path_or_url.startswith(("http://", "https://", "git://", "git@")):
            return True

        # GitHub/GitLab/Bitbucket short forms
        if re.match(r"^(github|gitlab|bitbucket)\.(com|org|io)/", path_or_url, re.IGNORECASE):
            return True

        # owner/repo format (GitHub shorthand)
        if re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$", path_or_url):
            return True

        return False

    def normalize_url(self, url: str) -> str:
        """Normalize a URL to a clonable HTTPS format.

        Args:
            url: Repository URL or shorthand.

        Returns:
            Normalized HTTPS URL.
        """
        url = url.strip()

        # Handle owner/repo shorthand (assume GitHub)
        if re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$", url):
            return f"https://github.com/{url}.git"

        # Add https:// if missing for domain URLs
        if re.match(r"^(github|gitlab|bitbucket)\.(com|org|io)/", url, re.IGNORECASE):
            return f"https://{url}"

        # Convert SSH to HTTPS
        ssh_match = re.match(r"git@([^:]+):(.+?)(?:\.git)?$", url)
        if ssh_match:
            host, path = ssh_match.groups()
            return f"https://{host}/{path}.git"

        # Ensure .git suffix for consistency
        if not url.endswith(".git") and not url.endswith("/"):
            url = f"{url}.git"

        return url

    def clone(
        self,
        url: str,
        target_dir: Path | None = None,
        branch: str | None = None,
        depth: int | None = None,
    ) -> ClonedRepo:
        """Clone a repository from URL.

        Args:
            url: Repository URL (will be normalized).
            target_dir: Where to clone. If None, creates temp directory.
            branch: Specific branch to clone. If None, uses default.
            depth: Shallow clone depth. If None, full clone.

        Returns:
            ClonedRepo with information about the cloned repo.

        Raises:
            ForeignRepoError: If cloning fails.
        """
        normalized_url = self.normalize_url(url)
        owner, repo_name, host_type = self.parse_repo_url(normalized_url)

        # Determine target directory
        workspace = None
        if target_dir is None:
            if self.workspace_dir:
                workspace = self.workspace_dir
                target_dir = workspace / (repo_name or "repo")
            else:
                workspace = Path(tempfile.mkdtemp(prefix="repo-auto-tool-"))
                self._temp_dirs.append(workspace)
                target_dir = workspace / (repo_name or "repo")

        target_dir = Path(target_dir).resolve()

        # Ensure parent exists
        try:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ForeignRepoError(
                f"Cannot create directory for clone: {e}",
                url=url,
            ) from e

        # Build clone command
        cmd = ["git", "clone"]
        if branch:
            cmd.extend(["--branch", branch])
        if depth:
            cmd.extend(["--depth", str(depth)])
        cmd.extend([normalized_url, str(target_dir)])

        logger.info(f"Cloning {url} to {target_dir}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=300,  # 5 minute timeout for large repos
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise ForeignRepoError(
                    f"Git clone failed: {error_msg.strip()}",
                    url=url,
                    details=error_msg,
                )
        except subprocess.TimeoutExpired as e:
            raise ForeignRepoError(
                "Clone timed out after 5 minutes",
                url=url,
            ) from e
        except OSError as e:
            raise ForeignRepoError(
                f"Failed to execute git: {e}",
                url=url,
            ) from e

        # Get default branch
        default_branch = self._get_default_branch(target_dir)

        # Create ClonedRepo record
        cloned = ClonedRepo(
            original_url=url,
            local_path=target_dir,
            owner=owner,
            repo_name=repo_name,
            default_branch=default_branch,
            workspace_dir=workspace,
        )

        self._cloned_repos.append(cloned)
        logger.info(f"Successfully cloned {repo_name or url} to {target_dir}")

        return cloned

    def _get_default_branch(self, repo_path: Path) -> str:
        """Get the default branch name for a repository.

        Args:
            repo_path: Path to the cloned repository.

        Returns:
            Default branch name ("main" as fallback).
        """
        try:
            # Try to get from remote HEAD
            result = subprocess.run(
                ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                # refs/remotes/origin/main -> main
                ref = result.stdout.strip()
                if ref:
                    return ref.split("/")[-1]

            # Fallback: check current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not determine default branch: {e}")

        return "main"

    def detect_conventions(self, repo_path: Path) -> RepoConventions:
        """Detect repository conventions from the codebase.

        Analyzes files to detect:
        - Code style (indentation, quotes, etc.)
        - Project type and primary language
        - Tooling (package manager, test framework, linter)
        - File structure patterns

        Args:
            repo_path: Path to the repository.

        Returns:
            RepoConventions with detected patterns.
        """
        conventions = RepoConventions()
        detected_files: list[str] = []

        repo_path = Path(repo_path).resolve()
        if not repo_path.exists():
            logger.warning(f"Repository path does not exist: {repo_path}")
            return conventions

        # Detect from config files
        conventions = self._detect_from_config_files(repo_path, conventions, detected_files)

        # Detect primary language
        conventions = self._detect_language(repo_path, conventions, detected_files)

        # Detect project type
        conventions = self._detect_project_type(repo_path, conventions, detected_files)

        # Detect code style from source files
        conventions = self._detect_code_style(repo_path, conventions, detected_files)

        # Detect emoji usage
        conventions = self._detect_emoji_usage(repo_path, conventions)

        conventions.detected_from_files = detected_files
        conventions.confidence = min(0.9, 0.3 + len(detected_files) * 0.1)

        logger.info(f"Detected conventions (confidence: {conventions.confidence:.1%})")
        return conventions

    def _detect_from_config_files(
        self,
        repo_path: Path,
        conventions: RepoConventions,
        detected_files: list[str],
    ) -> RepoConventions:
        """Detect conventions from config files."""

        # Python: pyproject.toml, setup.py, setup.cfg
        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            detected_files.append("pyproject.toml")
            try:
                content = pyproject.read_text(encoding="utf-8", errors="replace")

                # Detect build system
                if "[tool.poetry]" in content:
                    conventions.package_manager = "poetry"
                elif "[build-system]" in content:
                    conventions.package_manager = "pip"

                # Detect formatters/linters
                if "[tool.black]" in content:
                    conventions.formatter = "black"
                    # Extract line length
                    match = re.search(r"line-length\s*=\s*(\d+)", content)
                    if match:
                        conventions.max_line_length = int(match.group(1))

                if "[tool.ruff]" in content:
                    conventions.linter = "ruff"
                    # Extract line length
                    match = re.search(r"line-length\s*=\s*(\d+)", content)
                    if match:
                        conventions.max_line_length = int(match.group(1))

                if "[tool.pytest" in content or "[tool.pytest.ini_options]" in content:
                    conventions.test_framework = "pytest"

                if "[tool.mypy]" in content:
                    conventions.has_type_hints = True

            except Exception as e:
                logger.debug(f"Error parsing pyproject.toml: {e}")

        # Python: requirements.txt
        requirements = repo_path / "requirements.txt"
        if requirements.exists():
            detected_files.append("requirements.txt")
            if conventions.package_manager is None:
                conventions.package_manager = "pip"

        # Python: setup.py
        setup_py = repo_path / "setup.py"
        if setup_py.exists():
            detected_files.append("setup.py")
            conventions.build_tool = "setuptools"

        # Node.js: package.json
        package_json = repo_path / "package.json"
        if package_json.exists():
            detected_files.append("package.json")
            try:
                content = package_json.read_text(encoding="utf-8", errors="replace")
                data = json.loads(content)

                # Detect package manager
                if (repo_path / "yarn.lock").exists():
                    conventions.package_manager = "yarn"
                elif (repo_path / "pnpm-lock.yaml").exists():
                    conventions.package_manager = "pnpm"
                elif (repo_path / "package-lock.json").exists():
                    conventions.package_manager = "npm"

                # Detect from devDependencies
                dev_deps = data.get("devDependencies", {})
                deps = data.get("dependencies", {})
                all_deps = {**deps, **dev_deps}

                if "eslint" in all_deps:
                    conventions.linter = "eslint"
                if "prettier" in all_deps:
                    conventions.formatter = "prettier"
                if "jest" in all_deps:
                    conventions.test_framework = "jest"
                elif "mocha" in all_deps:
                    conventions.test_framework = "mocha"
                elif "vitest" in all_deps:
                    conventions.test_framework = "vitest"

                if "typescript" in all_deps:
                    conventions.has_type_hints = True

            except Exception as e:
                logger.debug(f"Error parsing package.json: {e}")

        # Rust: Cargo.toml
        cargo_toml = repo_path / "Cargo.toml"
        if cargo_toml.exists():
            detected_files.append("Cargo.toml")
            conventions.package_manager = "cargo"
            conventions.build_tool = "cargo"
            conventions.primary_language = "rust"

        # Go: go.mod
        go_mod = repo_path / "go.mod"
        if go_mod.exists():
            detected_files.append("go.mod")
            conventions.package_manager = "go"
            conventions.primary_language = "go"
            conventions.indent_style = "tabs"

        # EditorConfig
        editorconfig = repo_path / ".editorconfig"
        if editorconfig.exists():
            detected_files.append(".editorconfig")
            try:
                content = editorconfig.read_text(encoding="utf-8", errors="replace")

                if "indent_style = tab" in content.lower():
                    conventions.indent_style = "tabs"
                elif "indent_style = space" in content.lower():
                    conventions.indent_style = "spaces"

                match = re.search(r"indent_size\s*=\s*(\d+)", content, re.IGNORECASE)
                if match:
                    conventions.indent_size = int(match.group(1))

            except Exception as e:
                logger.debug(f"Error parsing .editorconfig: {e}")

        # Pre-commit
        if (repo_path / ".pre-commit-config.yaml").exists():
            detected_files.append(".pre-commit-config.yaml")
            conventions.has_pre_commit = True

        # CI/CD
        if (repo_path / ".github" / "workflows").exists():
            conventions.has_ci = True
            conventions.ci_platform = "github_actions"
        elif (repo_path / ".gitlab-ci.yml").exists():
            conventions.has_ci = True
            conventions.ci_platform = "gitlab_ci"
        elif (repo_path / ".circleci").exists():
            conventions.has_ci = True
            conventions.ci_platform = "circleci"

        return conventions

    def _detect_language(
        self,
        repo_path: Path,
        conventions: RepoConventions,
        detected_files: list[str],
    ) -> RepoConventions:
        """Detect primary programming language."""

        if conventions.primary_language != "unknown":
            return conventions

        # Count files by extension
        extensions: dict[str, int] = {}
        try:
            for file_path in repo_path.rglob("*"):
                if file_path.is_file() and not any(
                    part.startswith(".") for part in file_path.parts
                ):
                    ext = file_path.suffix.lower()
                    if ext:
                        extensions[ext] = extensions.get(ext, 0) + 1
        except Exception as e:
            logger.debug(f"Error counting files: {e}")
            return conventions

        # Map extensions to languages
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
        }

        lang_counts: dict[str, int] = {}
        for ext, count in extensions.items():
            lang = lang_map.get(ext)
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + count

        if lang_counts:
            primary = max(lang_counts, key=lambda k: lang_counts[k])
            conventions.primary_language = primary

            # Set language-specific defaults
            if primary == "python":
                conventions.naming_style = "snake_case"
                if conventions.test_framework is None:
                    conventions.test_framework = "pytest"
            elif primary in ("javascript", "typescript"):
                conventions.naming_style = "camelCase"
            elif primary == "go":
                conventions.indent_style = "tabs"
                conventions.naming_style = "camelCase"

        return conventions

    def _detect_project_type(
        self,
        repo_path: Path,
        conventions: RepoConventions,
        detected_files: list[str],
    ) -> RepoConventions:
        """Detect project type (library, CLI, web app, etc.)."""

        # Check for CLI indicators
        cli_indicators = [
            repo_path / "src" / "cli.py",
            repo_path / "cli.py",
            repo_path / "src" / "main.py",
            repo_path / "__main__.py",
        ]

        for indicator in cli_indicators:
            if indicator.exists():
                conventions.project_type = "cli"
                return conventions

        # Check pyproject.toml for scripts
        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8", errors="replace")
                if "[project.scripts]" in content or "[tool.poetry.scripts]" in content:
                    conventions.project_type = "cli"
                    return conventions
            except Exception:
                pass

        # Check package.json for bin
        package_json = repo_path / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text(encoding="utf-8", errors="replace"))
                if "bin" in data:
                    conventions.project_type = "cli"
                    return conventions
            except Exception:
                pass

        # Check for web app indicators
        web_indicators = [
            "app.py",
            "wsgi.py",
            "asgi.py",
            "manage.py",  # Django
            "next.config.js",
            "nuxt.config.js",
            "vite.config.js",
            "webpack.config.js",
        ]

        for indicator in web_indicators:
            if (repo_path / indicator).exists():
                conventions.project_type = "web_app"
                return conventions

        # Check for src layout (usually libraries)
        if (repo_path / "src").is_dir():
            conventions.src_layout = True
            if conventions.project_type == "unknown":
                conventions.project_type = "library"

        # Check for tests location
        for test_dir in ["tests", "test", "spec"]:
            if (repo_path / test_dir).is_dir():
                conventions.test_location = f"{test_dir}/"
                break

        return conventions

    def _detect_code_style(
        self,
        repo_path: Path,
        conventions: RepoConventions,
        detected_files: list[str],
    ) -> RepoConventions:
        """Detect code style from source files."""

        # Find source files to analyze
        source_patterns = {
            "python": "**/*.py",
            "javascript": "**/*.js",
            "typescript": "**/*.ts",
        }

        lang = conventions.primary_language
        if lang not in source_patterns:
            return conventions

        pattern = source_patterns[lang]
        sample_files: list[Path] = []

        try:
            for file_path in repo_path.glob(pattern):
                # Skip common non-source directories
                if any(part in str(file_path) for part in [
                    "node_modules", ".git", "__pycache__", ".venv", "venv",
                    "dist", "build", ".tox", ".pytest_cache"
                ]):
                    continue
                sample_files.append(file_path)
                if len(sample_files) >= 5:
                    break
        except Exception as e:
            logger.debug(f"Error finding source files: {e}")
            return conventions

        if not sample_files:
            return conventions

        # Analyze indentation and quotes
        indent_tabs = 0
        indent_spaces = 0
        single_quotes = 0
        double_quotes = 0
        type_hints = 0

        indent_2_space = 0
        indent_4_space = 0

        for file_path in sample_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.split("\n")

                for line in lines:
                    if line.startswith("\t"):
                        indent_tabs += 1
                    elif line.startswith("    "):
                        indent_spaces += 1
                        indent_4_space += 1
                    elif line.startswith("  ") and not line.startswith("    "):
                        indent_spaces += 1
                        indent_2_space += 1

                    # Count quotes (simple heuristic)
                    single_quotes += line.count("'") - line.count("\\'")
                    double_quotes += line.count('"') - line.count('\\"')

                # Check for type hints (Python)
                if lang == "python":
                    if re.search(r"def \w+\([^)]*:[^)]+\)", content):
                        type_hints += 1
                    if re.search(r"-> \w+:", content):
                        type_hints += 1

            except Exception as e:
                logger.debug(f"Error analyzing {file_path}: {e}")

        # Apply detected patterns
        if indent_tabs > indent_spaces:
            conventions.indent_style = "tabs"

        # Determine indent size based on which is more common
        if indent_2_space > indent_4_space:
            conventions.indent_size = 2
        elif indent_4_space > 0:
            conventions.indent_size = 4

        if single_quotes > double_quotes * 1.5:
            conventions.quote_style = "single"
        elif double_quotes > single_quotes * 1.5:
            conventions.quote_style = "double"

        if type_hints >= len(sample_files):
            conventions.has_type_hints = True

        return conventions

    def _detect_emoji_usage(
        self,
        repo_path: Path,
        conventions: RepoConventions,
    ) -> RepoConventions:
        """Detect if the repository uses emojis in code/commits."""

        # Check recent commit messages
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-20"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                # Simple emoji detection (common emoji ranges)
                emoji_pattern = re.compile(
                    "["
                    "\U0001F600-\U0001F64F"  # emoticons
                    "\U0001F300-\U0001F5FF"  # symbols & pictographs
                    "\U0001F680-\U0001F6FF"  # transport & map
                    "\U0001F1E0-\U0001F1FF"  # flags
                    "\U00002702-\U000027B0"
                    "\U000024C2-\U0001F251"
                    "]+",
                    flags=re.UNICODE,
                )
                if emoji_pattern.search(result.stdout):
                    conventions.uses_emojis = True
        except Exception as e:
            logger.debug(f"Error checking commit messages for emojis: {e}")

        return conventions

    def create_improvement_branch(
        self,
        repo: ClonedRepo,
        branch_name: str = "repo-auto-tool/improvements",
    ) -> str:
        """Create a branch for improvements.

        Args:
            repo: ClonedRepo to create branch in.
            branch_name: Name for the improvement branch.

        Returns:
            Name of the created branch.

        Raises:
            GitOperationError: If branch creation fails.
        """
        try:
            # Ensure we're on the default branch
            subprocess.run(
                ["git", "checkout", repo.default_branch],
                cwd=repo.local_path,
                capture_output=True,
                check=True,
            )

            # Create and checkout new branch
            result = subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=repo.local_path,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                # Branch might already exist
                if "already exists" in (result.stderr or ""):
                    subprocess.run(
                        ["git", "checkout", branch_name],
                        cwd=repo.local_path,
                        capture_output=True,
                        check=True,
                    )
                else:
                    raise GitOperationError(
                        operation="create branch",
                        reason=result.stderr or "Unknown error",
                    )

            logger.info(f"Created improvement branch: {branch_name}")
            return branch_name

        except subprocess.CalledProcessError as e:
            raise GitOperationError(
                operation="create branch",
                reason=str(e),
            ) from e

    def prepare_pr_info(self, repo: ClonedRepo) -> dict[str, Any]:
        """Prepare information for creating a pull request.

        Args:
            repo: ClonedRepo with improvements.

        Returns:
            Dictionary with PR information (title, body template, etc.).
        """
        info: dict[str, Any] = {
            "base_branch": repo.default_branch,
            "owner": repo.owner,
            "repo": repo.repo_name,
            "url": repo.original_url,
        }

        # Get current branch
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo.local_path,
                capture_output=True,
                text=True,
                check=True,
            )
            info["head_branch"] = result.stdout.strip()
        except Exception:
            info["head_branch"] = "repo-auto-tool/improvements"

        # Get commit log for PR body
        try:
            result = subprocess.run(
                ["git", "log", f"{repo.default_branch}..HEAD", "--oneline"],
                cwd=repo.local_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                commits = result.stdout.strip().split("\n")
                info["commits"] = [c for c in commits if c]
        except Exception:
            info["commits"] = []

        # Generate PR body template
        info["pr_body_template"] = """## Changes

{description}

## Commits

{commit_list}

---
*Generated by repo-auto-tool*
"""

        return info

    def cleanup(self, repo: ClonedRepo | None = None) -> None:
        """Clean up cloned repositories.

        Args:
            repo: Specific repo to clean up. If None, cleans all.
        """
        if repo:
            repos_to_clean = [repo]
        else:
            repos_to_clean = self._cloned_repos.copy()

        for r in repos_to_clean:
            if r.local_path and r.local_path.exists():
                try:
                    shutil.rmtree(r.local_path)
                    logger.info(f"Cleaned up: {r.local_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {r.local_path}: {e}")

            if r in self._cloned_repos:
                self._cloned_repos.remove(r)

        # Clean up temp directories
        if repo is None:
            for temp_dir in self._temp_dirs:
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp dir {temp_dir}: {e}")
            self._temp_dirs.clear()

    def __enter__(self) -> "ForeignRepoManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.cleanup_on_exit:
            self.cleanup()


def resolve_repo_path(
    path_or_url: str,
    workspace_dir: Path | None = None,
) -> tuple[Path, ClonedRepo | None, RepoConventions | None]:
    """Resolve a repository path or URL to a local path.

    This is the main entry point for foreign repo support.
    If given a URL, clones it and returns the local path.
    If given a local path, validates it exists.

    Args:
        path_or_url: Local path or repository URL.
        workspace_dir: Optional directory for cloning.

    Returns:
        Tuple of (local_path, cloned_repo_info, conventions).
        cloned_repo_info is None for local paths.
        conventions is None if detection fails.

    Raises:
        ConfigurationError: If path doesn't exist or URL is invalid.
        ForeignRepoError: If cloning fails.
    """
    manager = ForeignRepoManager(workspace_dir=workspace_dir, cleanup_on_exit=False)

    if manager.is_url(path_or_url):
        # Clone from URL
        try:
            cloned = manager.clone(path_or_url)
            conventions = manager.detect_conventions(cloned.local_path)
            cloned.conventions = conventions
            return cloned.local_path, cloned, conventions
        except ForeignRepoError:
            raise
        except Exception as e:
            raise ForeignRepoError(
                f"Failed to clone repository: {e}",
                url=path_or_url,
            ) from e
    else:
        # Local path
        local_path = Path(path_or_url).resolve()

        if not local_path.exists():
            raise ConfigurationError(f"Path does not exist: {local_path}")

        if not local_path.is_dir():
            raise ConfigurationError(f"Path is not a directory: {local_path}")

        # Detect conventions for local repos too
        try:
            conventions = manager.detect_conventions(local_path)
        except Exception as e:
            logger.warning(f"Failed to detect conventions: {e}")
            conventions = None

        return local_path, None, conventions
