"""Smart defaults and auto-understanding for minimal user input.

This module provides intelligent auto-detection and smart default behaviors
so that users only need to provide a repo path and a goal. Everything else
is automatically inferred from the codebase.

The module handles:
- Auto-detecting test frameworks, package managers, build tools
- Detecting code style (indentation, quotes, naming conventions)
- Inferring project type and adjusting approach accordingly
- Applying sensible defaults for validation and execution
- Detecting existing patterns to maintain consistency

All operations are defensively coded with robust error handling.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DetectedTools:
    """Detected development tools in a project.

    Attributes:
        test_framework: Detected test framework (pytest, unittest, jest, etc.)
        test_command: Command to run tests
        linter: Detected linter (ruff, flake8, eslint, etc.)
        lint_command: Command to run linter
        formatter: Detected formatter (black, prettier, etc.)
        format_command: Command to run formatter
        package_manager: Detected package manager (pip, poetry, npm, etc.)
        build_tool: Detected build tool (setuptools, webpack, cargo, etc.)
        type_checker: Detected type checker (mypy, pyright, etc.)
    """

    test_framework: str | None = None
    test_command: str | None = None
    linter: str | None = None
    lint_command: str | None = None
    formatter: str | None = None
    format_command: str | None = None
    package_manager: str | None = None
    build_tool: str | None = None
    type_checker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_framework": self.test_framework,
            "test_command": self.test_command,
            "linter": self.linter,
            "lint_command": self.lint_command,
            "formatter": self.formatter,
            "format_command": self.format_command,
            "package_manager": self.package_manager,
            "build_tool": self.build_tool,
            "type_checker": self.type_checker,
        }


@dataclass
class ProjectProfile:
    """Comprehensive profile of a project based on auto-detection.

    This class combines detected tools, code style, and project metadata
    to provide a complete picture of how the project should be handled.

    Attributes:
        project_type: Type of project (library, cli, web_app, api, etc.)
        primary_language: Main programming language
        detected_tools: Detected development tools
        code_style: Detected code style preferences
        has_ci: Whether CI/CD is configured
        has_pre_commit: Whether pre-commit hooks are configured
        warnings: Any warnings or issues detected
        recommendations: Suggestions for the improvement process
    """

    project_type: str = "unknown"
    primary_language: str = "unknown"
    detected_tools: DetectedTools = field(default_factory=DetectedTools)
    code_style: dict[str, Any] = field(default_factory=dict)
    has_ci: bool = False
    has_pre_commit: bool = False
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_type": self.project_type,
            "primary_language": self.primary_language,
            "detected_tools": self.detected_tools.to_dict(),
            "code_style": self.code_style,
            "has_ci": self.has_ci,
            "has_pre_commit": self.has_pre_commit,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def generate_guidance(self) -> str:
        """Generate guidance text for Claude prompts.

        Returns:
            Formatted guidance string.
        """
        lines = ["PROJECT CONTEXT (auto-detected):"]

        if self.primary_language != "unknown":
            lines.append(f"- Primary language: {self.primary_language}")
        if self.project_type != "unknown":
            lines.append(f"- Project type: {self.project_type}")

        # Code style guidance
        if self.code_style:
            style = self.code_style
            if style.get("indent_style"):
                indent = style.get("indent_style")
                size = style.get("indent_size", 4)
                if indent == "tabs":
                    lines.append("- Use tabs for indentation")
                else:
                    lines.append(f"- Use {size} spaces for indentation")

            if style.get("quote_style"):
                lines.append(f"- Use {style['quote_style']} quotes for strings")

            if style.get("max_line_length"):
                lines.append(f"- Keep lines under {style['max_line_length']} characters")

            if style.get("uses_emojis") is False:
                lines.append("- Do NOT use emojis in code, comments, or commits")

            if style.get("naming_style"):
                lines.append(f"- Use {style['naming_style']} for naming")

        # Tool guidance
        tools = self.detected_tools
        if tools.test_framework:
            lines.append(f"- Test framework: {tools.test_framework}")
        if tools.linter:
            lines.append(f"- Linter: {tools.linter} (code must pass)")
        if tools.formatter:
            lines.append(f"- Formatter: {tools.formatter}")

        # Warnings
        if self.warnings:
            lines.append("\nWARNINGS:")
            for warning in self.warnings[:3]:
                lines.append(f"  - {warning}")

        # Recommendations
        if self.recommendations:
            lines.append("\nRECOMMENDATIONS:")
            for rec in self.recommendations[:3]:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class SmartDefaults:
    """Provides intelligent auto-detection and smart defaults for projects.

    This class analyzes a project and determines sensible defaults for:
    - Test command and framework
    - Linter command and settings
    - Code style preferences
    - Project type-specific behaviors

    All detection is defensive and handles errors gracefully.
    """

    def __init__(self, repo_path: Path):
        """Initialize with a repository path.

        Args:
            repo_path: Path to the repository to analyze.
        """
        self.repo_path = Path(repo_path).resolve()
        self._profile: ProjectProfile | None = None
        self._cache: dict[str, Any] = {}

    def analyze(self) -> ProjectProfile:
        """Perform comprehensive project analysis.

        Returns:
            ProjectProfile with all detected information.
        """
        if self._profile is not None:
            return self._profile

        logger.info(f"Analyzing project: {self.repo_path}")

        profile = ProjectProfile()

        try:
            # Detect primary language
            profile.primary_language = self._detect_language()

            # Detect project type
            profile.project_type = self._detect_project_type()

            # Detect development tools
            profile.detected_tools = self._detect_tools()

            # Detect code style
            profile.code_style = self._detect_code_style()

            # Check for CI/CD
            profile.has_ci = self._has_ci()

            # Check for pre-commit
            profile.has_pre_commit = (self.repo_path / ".pre-commit-config.yaml").exists()

            # Generate recommendations
            profile.recommendations = self._generate_recommendations(profile)

            # Identify warnings
            profile.warnings = self._identify_warnings(profile)

        except Exception as e:
            logger.error(f"Error during project analysis: {e}")
            profile.warnings.append(f"Analysis incomplete due to error: {e}")

        self._profile = profile
        return profile

    def get_test_command(self) -> str | None:
        """Get the recommended test command.

        Returns:
            Test command string or None if no tests detected.
        """
        profile = self.analyze()
        return profile.detected_tools.test_command

    def get_lint_command(self) -> str | None:
        """Get the recommended lint command.

        Returns:
            Lint command string or None if no linter detected.
        """
        profile = self.analyze()
        return profile.detected_tools.lint_command

    def _detect_language(self) -> str:
        """Detect the primary programming language.

        Returns:
            Language name or 'unknown'.
        """
        if "language" in self._cache:
            return self._cache["language"]

        # Count files by extension
        ext_counts: dict[str, int] = {}
        try:
            for path in self.repo_path.rglob("*"):
                if not path.is_file():
                    continue
                # Skip common non-source directories
                if any(part.startswith(".") or part in (
                    "node_modules", "venv", ".venv", "env",
                    "__pycache__", "dist", "build", "target"
                ) for part in path.parts):
                    continue

                ext = path.suffix.lower()
                if ext:
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
        except OSError as e:
            logger.warning(f"Error scanning files: {e}")
            return "unknown"

        # Map extensions to languages
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".swift": "swift",
            ".kt": "kotlin",
        }

        lang_counts: dict[str, int] = {}
        for ext, count in ext_counts.items():
            if ext in lang_map:
                lang = lang_map[ext]
                lang_counts[lang] = lang_counts.get(lang, 0) + count

        if not lang_counts:
            return "unknown"

        # Return the most common language
        primary = max(lang_counts, key=lambda k: lang_counts[k])
        self._cache["language"] = primary
        return primary

    def _detect_project_type(self) -> str:
        """Detect the type of project.

        Returns:
            Project type string.
        """
        # Check for CLI indicators
        cli_indicators = [
            self.repo_path / "src" / "cli.py",
            self.repo_path / "cli.py",
            self.repo_path / "src" / "main.py",
            self.repo_path / "__main__.py",
        ]
        for indicator in cli_indicators:
            if indicator.exists():
                return "cli"

        # Check pyproject.toml for scripts
        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8", errors="replace")
                if "[project.scripts]" in content or "[tool.poetry.scripts]" in content:
                    return "cli"
            except Exception:
                pass

        # Check package.json for bin
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                import json
                data = json.loads(package_json.read_text(encoding="utf-8", errors="replace"))
                if "bin" in data:
                    return "cli"
            except Exception:
                pass

        # Check for web app indicators
        web_indicators = [
            "app.py", "wsgi.py", "asgi.py", "manage.py",
            "next.config.js", "nuxt.config.js", "vite.config.js",
            "vite.config.ts", "webpack.config.js",
        ]
        for indicator in web_indicators:
            if (self.repo_path / indicator).exists():
                return "web_app"

        # Check for API indicators
        api_indicators = ["api.py", "routes.py", "endpoints.py"]
        for indicator in api_indicators:
            for path in self.repo_path.rglob(indicator):
                if "test" not in str(path).lower():
                    return "api"

        # Check for src layout (usually libraries)
        if (self.repo_path / "src").is_dir():
            return "library"

        # Default
        return "unknown"

    def _detect_tools(self) -> DetectedTools:
        """Detect development tools in the project.

        Returns:
            DetectedTools with all detected tools.
        """
        tools = DetectedTools()
        lang = self._detect_language()

        # Detect based on language
        if lang == "python":
            tools = self._detect_python_tools()
        elif lang in ("javascript", "typescript"):
            tools = self._detect_js_tools()
        elif lang == "rust":
            tools.test_command = "cargo test"
            tools.lint_command = "cargo clippy"
            tools.build_tool = "cargo"
            tools.package_manager = "cargo"
        elif lang == "go":
            tools.test_command = "go test ./..."
            tools.lint_command = "golangci-lint run"
            tools.package_manager = "go"

        return tools

    def _detect_python_tools(self) -> DetectedTools:
        """Detect Python-specific development tools.

        Returns:
            DetectedTools configured for Python project.
        """
        tools = DetectedTools()

        # Detect package manager
        if (self.repo_path / "poetry.lock").exists():
            tools.package_manager = "poetry"
        elif (self.repo_path / "Pipfile.lock").exists():
            tools.package_manager = "pipenv"
        elif (self.repo_path / "requirements.txt").exists():
            tools.package_manager = "pip"
        elif (self.repo_path / "pyproject.toml").exists():
            tools.package_manager = "pip"  # Could be pip with pyproject.toml

        # Detect test framework
        pyproject = self.repo_path / "pyproject.toml"
        has_pytest_config = False
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8", errors="replace")
                if "[tool.pytest" in content:
                    has_pytest_config = True
            except Exception:
                pass

        # Check for pytest in dependencies or config
        if has_pytest_config or (self.repo_path / "pytest.ini").exists():
            tools.test_framework = "pytest"
            tools.test_command = "pytest"
        elif (self.repo_path / "setup.cfg").exists():
            try:
                setup_cfg = self.repo_path / "setup.cfg"
                content = setup_cfg.read_text(encoding="utf-8", errors="replace")
                if "[tool:pytest]" in content:
                    tools.test_framework = "pytest"
                    tools.test_command = "pytest"
            except Exception:
                pass

        # Check for test files to infer pytest if not found
        if not tools.test_framework:
            for pattern in ["test_*.py", "*_test.py"]:
                if list(self.repo_path.rglob(pattern)):
                    tools.test_framework = "pytest"
                    tools.test_command = "pytest"
                    break

        # Detect linter
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8", errors="replace")
                if "[tool.ruff]" in content:
                    tools.linter = "ruff"
                    tools.lint_command = "ruff check ."
                elif "[tool.flake8]" in content or (self.repo_path / ".flake8").exists():
                    tools.linter = "flake8"
                    tools.lint_command = "flake8 ."
            except Exception:
                pass

        # Check for ruff.toml
        if not tools.linter and (self.repo_path / "ruff.toml").exists():
            tools.linter = "ruff"
            tools.lint_command = "ruff check ."

        # Default to ruff if nothing found (it's the modern standard)
        if not tools.linter:
            # Check if ruff is installed
            if self._is_tool_installed("ruff"):
                tools.linter = "ruff"
                tools.lint_command = "ruff check ."

        # Detect formatter
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8", errors="replace")
                if "[tool.black]" in content:
                    tools.formatter = "black"
                    tools.format_command = "black ."
                elif "[tool.ruff.format]" in content:
                    tools.formatter = "ruff"
                    tools.format_command = "ruff format ."
            except Exception:
                pass

        # Detect type checker
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8", errors="replace")
                if "[tool.mypy]" in content:
                    tools.type_checker = "mypy"
                elif "[tool.pyright]" in content:
                    tools.type_checker = "pyright"
            except Exception:
                pass

        return tools

    def _detect_js_tools(self) -> DetectedTools:
        """Detect JavaScript/TypeScript development tools.

        Returns:
            DetectedTools configured for JS/TS project.
        """
        tools = DetectedTools()

        # Detect package manager
        if (self.repo_path / "yarn.lock").exists():
            tools.package_manager = "yarn"
        elif (self.repo_path / "pnpm-lock.yaml").exists():
            tools.package_manager = "pnpm"
        elif (self.repo_path / "package-lock.json").exists():
            tools.package_manager = "npm"

        # Read package.json for more info
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                import json
                data = json.loads(package_json.read_text(encoding="utf-8", errors="replace"))

                deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}

                # Detect test framework
                if "jest" in deps:
                    tools.test_framework = "jest"
                    tools.test_command = "npm test"
                elif "mocha" in deps:
                    tools.test_framework = "mocha"
                    tools.test_command = "npm test"
                elif "vitest" in deps:
                    tools.test_framework = "vitest"
                    tools.test_command = "npm test"

                # Detect linter
                if "eslint" in deps:
                    tools.linter = "eslint"
                    has_lint_script = "lint" in data.get("scripts", {})
                    tools.lint_command = "npm run lint" if has_lint_script else "npx eslint ."

                # Detect formatter
                if "prettier" in deps:
                    tools.formatter = "prettier"
                    tools.format_command = "npx prettier --write ."

                # Detect type checker
                if "typescript" in deps:
                    tools.type_checker = "typescript"

            except Exception as e:
                logger.debug(f"Error parsing package.json: {e}")

        return tools

    def _detect_code_style(self) -> dict[str, Any]:
        """Detect code style preferences from the codebase.

        Returns:
            Dictionary of style preferences.
        """
        style: dict[str, Any] = {}

        lang = self._detect_language()

        # Check for editorconfig
        editorconfig = self.repo_path / ".editorconfig"
        if editorconfig.exists():
            try:
                content = editorconfig.read_text(encoding="utf-8", errors="replace")
                if "indent_style = tab" in content.lower():
                    style["indent_style"] = "tabs"
                elif "indent_style = space" in content.lower():
                    style["indent_style"] = "spaces"

                match = re.search(r"indent_size\s*=\s*(\d+)", content, re.IGNORECASE)
                if match:
                    style["indent_size"] = int(match.group(1))
            except Exception:
                pass

        # Check for pyproject.toml settings
        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8", errors="replace")

                # Line length from black or ruff
                match = re.search(r"line-length\s*=\s*(\d+)", content)
                if match:
                    style["max_line_length"] = int(match.group(1))
            except Exception:
                pass

        # Check for emoji usage in commits
        style["uses_emojis"] = self._check_emoji_usage()

        # Detect naming style from source files
        if lang == "python":
            style["naming_style"] = "snake_case"
        elif lang in ("javascript", "typescript"):
            style["naming_style"] = "camelCase"
        elif lang == "go":
            style["naming_style"] = "camelCase"
            style["indent_style"] = "tabs"

        # Detect quote style from source files
        quote_style = self._detect_quote_style()
        if quote_style:
            style["quote_style"] = quote_style

        return style

    def _check_emoji_usage(self) -> bool:
        """Check if the repository uses emojis in commits/code.

        Returns:
            True if emojis are used.
        """
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-20"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode == 0:
                # Common emoji ranges
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
                return bool(emoji_pattern.search(result.stdout))
        except Exception as e:
            logger.debug(f"Error checking emoji usage: {e}")
        return False

    def _detect_quote_style(self) -> str | None:
        """Detect preferred quote style from source files.

        Returns:
            'single' or 'double' or None.
        """
        lang = self._detect_language()

        # Get file extension based on language
        ext_map = {
            "python": "*.py",
            "javascript": "*.js",
            "typescript": "*.ts",
        }
        pattern = ext_map.get(lang)
        if not pattern:
            return None

        single_count = 0
        double_count = 0

        try:
            files = list(self.repo_path.glob(pattern))[:5]  # Sample first 5 files
            for file_path in files:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    # Simple counting (ignoring escaped quotes)
                    single_count += content.count("'") - content.count("\\'")
                    double_count += content.count('"') - content.count('\\"')
                except Exception:
                    pass
        except Exception:
            pass

        if single_count > double_count * 1.5:
            return "single"
        elif double_count > single_count * 1.5:
            return "double"
        return None

    def _has_ci(self) -> bool:
        """Check if CI/CD is configured.

        Returns:
            True if CI is detected.
        """
        ci_paths = [
            self.repo_path / ".github" / "workflows",
            self.repo_path / ".gitlab-ci.yml",
            self.repo_path / ".circleci",
            self.repo_path / "Jenkinsfile",
            self.repo_path / ".travis.yml",
            self.repo_path / "azure-pipelines.yml",
        ]
        for ci_path in ci_paths:
            if ci_path.exists():
                return True
        return False

    def _is_tool_installed(self, tool_name: str) -> bool:
        """Check if a tool is installed and available.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool is available.
        """
        try:
            result = subprocess.run(
                [tool_name, "--version"],
                capture_output=True,
                check=False,
                timeout=5,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _generate_recommendations(self, profile: ProjectProfile) -> list[str]:
        """Generate recommendations based on detected profile.

        Args:
            profile: The detected project profile.

        Returns:
            List of recommendation strings.
        """
        recs = []

        tools = profile.detected_tools

        # Recommend testing if no tests found
        if not tools.test_command:
            if profile.primary_language == "python":
                recs.append("Consider adding pytest for testing")
            elif profile.primary_language in ("javascript", "typescript"):
                recs.append("Consider adding jest or vitest for testing")

        # Recommend linting if no linter found
        if not tools.linter:
            if profile.primary_language == "python":
                recs.append("Consider adding ruff for fast linting")
            elif profile.primary_language in ("javascript", "typescript"):
                recs.append("Consider adding eslint for linting")

        # Recommend CI if not present
        if not profile.has_ci:
            recs.append("Consider adding CI/CD (e.g., GitHub Actions)")

        # Recommend pre-commit if not present and has linting
        if not profile.has_pre_commit and tools.linter:
            recs.append("Consider adding pre-commit hooks for automated checks")

        return recs

    def _identify_warnings(self, profile: ProjectProfile) -> list[str]:
        """Identify potential issues or warnings.

        Args:
            profile: The detected project profile.

        Returns:
            List of warning strings.
        """
        warnings = []

        # Check for common issues
        if profile.primary_language == "unknown":
            warnings.append("Could not detect primary language - commands may need adjustment")

        # Check for missing virtual environment (Python)
        if profile.primary_language == "python":
            venv_paths = [
                self.repo_path / "venv",
                self.repo_path / ".venv",
                self.repo_path / "env",
            ]
            if not any(p.is_dir() for p in venv_paths) and not os.environ.get("VIRTUAL_ENV"):
                warnings.append("No virtual environment detected - recommend creating one")

        return warnings


def detect_smart_defaults(repo_path: Path) -> ProjectProfile:
    """Convenience function to detect smart defaults for a repository.

    Args:
        repo_path: Path to the repository.

    Returns:
        ProjectProfile with all detected information.
    """
    analyzer = SmartDefaults(repo_path)
    return analyzer.analyze()


def get_validation_commands(repo_path: Path) -> tuple[str | None, str | None]:
    """Get recommended test and lint commands for a repository.

    Args:
        repo_path: Path to the repository.

    Returns:
        Tuple of (test_command, lint_command), either may be None.
    """
    try:
        analyzer = SmartDefaults(repo_path)
        return analyzer.get_test_command(), analyzer.get_lint_command()
    except Exception as e:
        logger.warning(f"Error detecting validation commands: {e}")
        return None, None
