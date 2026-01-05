"""Smart validation that only tests what changed.

This module provides differential validation to significantly reduce validation
time by only running tests relevant to the changes made.

Strategy:
1. Analyze git diff to identify changed files
2. Build import dependency graph to find affected modules
3. Identify test files that test the changed modules
4. Run targeted tests when safe (small, isolated changes)
5. Fall back to full validation for major changes

Example:
    validator = SmartValidator(
        repo_path=repo_path,
        full_validator=full_validation_pipeline,
    )

    # Automatically decides whether to run targeted or full validation
    passed, results = validator.validate()
"""

from __future__ import annotations

import ast
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Thresholds for when to use smart vs full validation
MAX_FILES_FOR_SMART = 10  # Max changed files for smart validation
MAX_PERCENTAGE_CHANGED = 0.15  # Max 15% of files changed for smart validation


@dataclass
class ChangeAnalysis:
    """Analysis of changes in the repository.

    Attributes:
        changed_files: Set of changed file paths.
        changed_modules: Set of changed Python modules (dot notation).
        affected_test_files: Set of test files that should be run.
        is_major_change: Whether this is a major change requiring full validation.
        reason: Reason for the is_major_change decision.
        total_python_files: Total number of Python files in repo.
    """
    changed_files: set[Path] = field(default_factory=set)
    changed_modules: set[str] = field(default_factory=set)
    affected_test_files: set[Path] = field(default_factory=set)
    is_major_change: bool = False
    reason: str = ""
    total_python_files: int = 0


class DependencyAnalyzer:
    """Analyzes Python import dependencies to find affected modules.

    Attributes:
        repo_path: Path to the repository root.
        import_graph: Map of module to set of modules it imports.
        reverse_graph: Map of module to set of modules that import it.
    """

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.import_graph: dict[str, set[str]] = {}
        self.reverse_graph: dict[str, set[str]] = {}

    def extract_imports_from_file(self, file_path: Path) -> set[str]:
        """Extract import statements from a Python file.

        Args:
            file_path: Path to the Python file.

        Returns:
            Set of imported module names (in dot notation).
        """
        imports = set()

        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])

        except (SyntaxError, UnicodeDecodeError, OSError) as e:
            logger.debug(f"Could not parse {file_path}: {e}")

        return imports

    def get_module_name(self, file_path: Path) -> str | None:
        """Convert file path to Python module name.

        Args:
            file_path: Path to the Python file.

        Returns:
            Module name in dot notation, or None if not a valid module.
        """
        try:
            rel_path = file_path.relative_to(self.repo_path)
        except ValueError:
            return None

        # Remove .py extension and convert path to module notation
        parts = list(rel_path.parts)
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]

        # Skip __init__.py files (they represent the package itself)
        if parts[-1] == '__init__':
            parts = parts[:-1]

        if not parts:
            return None

        return '.'.join(parts)

    def find_affected_modules(self, changed_modules: set[str]) -> set[str]:
        """Find all modules affected by changes (transitive closure).

        Args:
            changed_modules: Set of directly changed modules.

        Returns:
            Set of all affected modules (including changed ones).
        """
        affected = set(changed_modules)
        to_process = list(changed_modules)

        while to_process:
            current = to_process.pop()

            # Find modules that import this module
            if current in self.reverse_graph:
                for importer in self.reverse_graph[current]:
                    if importer not in affected:
                        affected.add(importer)
                        to_process.append(importer)

        return affected


class SmartValidator:
    """Intelligent validation that targets tests based on changes.

    Analyzes what changed and runs only relevant tests when safe,
    significantly reducing validation time for small changes.

    Attributes:
        repo_path: Path to the repository.
        full_validator: Full validation pipeline to fall back to.
        dependency_analyzer: Analyzer for import dependencies.
        enable_smart_validation: Whether smart validation is enabled.
    """

    def __init__(
        self,
        repo_path: Path | str,
        full_validator: Any,
        enable_smart_validation: bool = True,
    ):
        self.repo_path = Path(repo_path)
        self.full_validator = full_validator
        self.enable_smart_validation = enable_smart_validation
        self.dependency_analyzer = DependencyAnalyzer(self.repo_path)

    def analyze_changes(self) -> ChangeAnalysis:
        """Analyze what changed in the repository.

        Returns:
            ChangeAnalysis with information about the changes.
        """
        analysis = ChangeAnalysis()

        # Get changed files from git
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                logger.warning("Could not get git diff, assuming major change")
                analysis.is_major_change = True
                analysis.reason = "Git diff failed"
                return analysis

            changed_file_paths = result.stdout.strip().split('\n')
            changed_file_paths = [p for p in changed_file_paths if p]

        except (FileNotFoundError, subprocess.SubprocessError) as e:
            logger.warning(f"Git not available: {e}")
            analysis.is_major_change = True
            analysis.reason = "Git not available"
            return analysis

        # Filter to Python files
        for path_str in changed_file_paths:
            path = self.repo_path / path_str
            if path.suffix == '.py' and path.exists():
                analysis.changed_files.add(path)

        # Count total Python files
        try:
            analysis.total_python_files = len(list(self.repo_path.rglob('*.py')))
        except OSError:
            analysis.total_python_files = 100  # Default estimate

        # Check if major change
        num_changed = len(analysis.changed_files)

        if num_changed == 0:
            # No changes - use full validation to be safe
            analysis.is_major_change = True
            analysis.reason = "No changed files detected"
            return analysis

        if num_changed > MAX_FILES_FOR_SMART:
            analysis.is_major_change = True
            analysis.reason = f"Too many files changed ({num_changed} > {MAX_FILES_FOR_SMART})"
            return analysis

        percentage_changed = num_changed / max(1, analysis.total_python_files)
        if percentage_changed > MAX_PERCENTAGE_CHANGED:
            analysis.is_major_change = True
            analysis.reason = (
                f"Large percentage changed ({percentage_changed:.0%} > "
                f"{MAX_PERCENTAGE_CHANGED:.0%})"
            )
            return analysis

        # Check for architectural changes (changes to core files)
        core_indicators = {
            '__init__.py', 'setup.py', 'pyproject.toml', 'config.py',
            'settings.py', 'base.py', 'main.py',
        }

        for file_path in analysis.changed_files:
            if file_path.name in core_indicators:
                analysis.is_major_change = True
                analysis.reason = f"Core file changed: {file_path.name}"
                return analysis

        # Extract changed modules
        for file_path in analysis.changed_files:
            module = self.dependency_analyzer.get_module_name(file_path)
            if module:
                analysis.changed_modules.add(module)

        # Find affected test files
        analysis.affected_test_files = self._find_test_files(
            analysis.changed_modules
        )

        logger.info(
            f"Change analysis: {num_changed} files, {len(analysis.changed_modules)} modules, "
            f"{len(analysis.affected_test_files)} test files"
        )

        return analysis

    def _find_test_files(self, changed_modules: set[str]) -> set[Path]:
        """Find test files that test the changed modules.

        Args:
            changed_modules: Set of changed module names.

        Returns:
            Set of test file paths.
        """
        test_files = set()

        # Common test directory patterns
        test_dirs = [
            self.repo_path / 'tests',
            self.repo_path / 'test',
        ]

        # Also look for test files in the same directories as changed files
        for module in changed_modules:
            # Convert module back to path
            module_path = self.repo_path / module.replace('.', '/')

            # Look for test files in same directory
            if module_path.is_file():
                parent = module_path.parent
            else:
                parent = module_path

            if parent.exists():
                for test_file in parent.glob('test_*.py'):
                    test_files.add(test_file)
                for test_file in parent.glob('*_test.py'):
                    test_files.add(test_file)

        # Look in test directories for files matching module names
        for test_dir in test_dirs:
            if not test_dir.exists():
                continue

            for module in changed_modules:
                # Look for test_<module>.py
                module_name = module.split('.')[-1]
                test_file = test_dir / f'test_{module_name}.py'
                if test_file.exists():
                    test_files.add(test_file)

            # If no specific tests found, include all tests in test dir
            if not test_files:
                test_files = set(test_dir.rglob('test_*.py'))

        return test_files

    def validate(self) -> tuple[bool, list[Any]]:
        """Run smart or full validation based on changes.

        Returns:
            Tuple of (all_passed, results) from validation.
        """
        if not self.enable_smart_validation:
            logger.info("Smart validation disabled, running full validation")
            return self.full_validator.validate(self.repo_path)

        analysis = self.analyze_changes()

        if analysis.is_major_change:
            logger.info(f"Running FULL validation: {analysis.reason}")
            return self.full_validator.validate(self.repo_path)

        # Run targeted validation
        logger.info(
            f"Running SMART validation on {len(analysis.affected_test_files)} test files"
        )

        if not analysis.affected_test_files:
            logger.warning("No test files found for changes, running full validation")
            return self.full_validator.validate(self.repo_path)

        # Run only the affected tests
        # For now, we'll fall back to full validation but log what we would do
        # In a future version, we could actually run targeted tests
        logger.info(f"Would run tests: {[str(f.name) for f in analysis.affected_test_files]}")

        # TODO: Implement targeted test running
        # For now, run full validation but log the optimization opportunity
        logger.info("(Smart validation not fully implemented yet, running full validation)")
        return self.full_validator.validate(self.repo_path)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about validation optimization potential.

        Returns:
            Dictionary with stats.
        """
        analysis = self.analyze_changes()

        return {
            "smart_validation_enabled": self.enable_smart_validation,
            "changed_files": len(analysis.changed_files),
            "changed_modules": len(analysis.changed_modules),
            "affected_test_files": len(analysis.affected_test_files),
            "is_major_change": analysis.is_major_change,
            "reason": analysis.reason,
            "would_use_smart": not analysis.is_major_change and self.enable_smart_validation,
        }
