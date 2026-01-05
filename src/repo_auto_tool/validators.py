"""Validators to ensure improvements don't break the codebase."""

import logging
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    validator_name: str
    message: str
    output: str = ""
    

class Validator(ABC):
    """Base class for validators."""
    
    name: str = "base"
    
    @abstractmethod
    def validate(self, repo_path: Path) -> ValidationResult:
        """Run validation and return result."""
        pass


class CommandValidator(Validator):
    """Validator that runs a shell command."""
    
    def __init__(self, name: str, command: str, timeout: int = 120):
        self.name = name
        self.command = command
        self.timeout = timeout
    
    def validate(self, repo_path: Path) -> ValidationResult:
        """Run the command and check exit code."""
        logger.info(f"Running validator '{self.name}': {self.command}")
        
        try:
            result = subprocess.run(
                self.command,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            
            if passed:
                logger.info(f"  [OK] {self.name} passed")
            else:
                logger.warning(f"  [FAIL] {self.name} failed")
                logger.debug(f"  Output: {output[:500]}")
            
            return ValidationResult(
                passed=passed,
                validator_name=self.name,
                message="Passed" if passed else f"Exit code {result.returncode}",
                output=output,
            )
            
        except subprocess.TimeoutExpired:
            logger.error(f"  [FAIL] {self.name} timed out")
            return ValidationResult(
                passed=False,
                validator_name=self.name,
                message=f"Timed out after {self.timeout}s",
            )
        except Exception as e:
            logger.error(f"  [FAIL] {self.name} error: {e}")
            return ValidationResult(
                passed=False,
                validator_name=self.name,
                message=str(e),
            )


class TestValidator(CommandValidator):
    """Run pytest or similar test suite."""
    
    def __init__(self, command: str = "pytest", timeout: int = 300):
        super().__init__("tests", command, timeout)


class LintValidator(CommandValidator):
    """Run linter (ruff, flake8, etc.)."""
    
    def __init__(self, command: str = "ruff check .", timeout: int = 60):
        super().__init__("linter", command, timeout)


class TypeCheckValidator(CommandValidator):
    """Run type checker (mypy, pyright)."""
    
    def __init__(self, command: str = "mypy .", timeout: int = 120):
        super().__init__("typecheck", command, timeout)


class SyntaxValidator(Validator):
    """Check Python files for syntax errors (fast, no deps)."""
    
    name = "syntax"
    
    def validate(self, repo_path: Path) -> ValidationResult:
        """Compile all Python files to check syntax."""
        import py_compile
        
        errors = []
        for py_file in repo_path.rglob("*.py"):
            # Skip common non-source directories
            if any(part.startswith(".") or part in ("venv", "node_modules", "__pycache__") 
                   for part in py_file.parts):
                continue
            
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"{py_file}: {e}")
        
        if errors:
            logger.warning(f"  [FAIL] Syntax errors in {len(errors)} files")
            return ValidationResult(
                passed=False,
                validator_name=self.name,
                message=f"Syntax errors in {len(errors)} files",
                output="\n".join(errors),
            )
        
        logger.info("  [OK] Syntax check passed")
        return ValidationResult(
            passed=True,
            validator_name=self.name,
            message="All Python files have valid syntax",
        )


class ValidationPipeline:
    """Run multiple validators, optionally in parallel.

    Parallel mode runs all validators concurrently using threads,
    which can significantly speed up validation when multiple slow
    validators (like tests and linting) are configured.
    """

    def __init__(
        self,
        validators: list[Validator] | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
    ):
        """Initialize the validation pipeline.

        Args:
            validators: List of validators to run.
            parallel: If True, run validators concurrently using threads.
            max_workers: Maximum number of parallel workers. Defaults to
                number of validators (one thread per validator).
        """
        self.validators = validators or []
        self.parallel = parallel
        self.max_workers = max_workers

    def add(self, validator: Validator) -> "ValidationPipeline":
        """Add a validator to the pipeline."""
        self.validators.append(validator)
        return self

    def validate(self, repo_path: Path) -> tuple[bool, list[ValidationResult]]:
        """Run all validators.

        If parallel mode is enabled, validators run concurrently.
        Results are always returned in the order validators were added.

        Returns:
            (all_passed, list of results)
        """
        if self.parallel and len(self.validators) > 1:
            return self._validate_parallel(repo_path)
        return self._validate_sequential(repo_path)

    def _validate_sequential(
        self, repo_path: Path
    ) -> tuple[bool, list[ValidationResult]]:
        """Run validators sequentially (original behavior)."""
        results = []
        all_passed = True

        for validator in self.validators:
            result = validator.validate(repo_path)
            results.append(result)
            if not result.passed:
                all_passed = False
                # Continue running other validators for full report

        return all_passed, results

    def _validate_parallel(
        self, repo_path: Path
    ) -> tuple[bool, list[ValidationResult]]:
        """Run validators in parallel using threads.

        Each validator runs in its own thread. Results are collected
        and returned in the original validator order.
        """
        workers = self.max_workers or len(self.validators)
        logger.info(f"Running {len(self.validators)} validators in parallel")

        # Map to track validator index for result ordering
        results_map: dict[int, ValidationResult] = {}

        def run_validator(
            index: int, validator: Validator
        ) -> tuple[int, ValidationResult]:
            """Wrapper to run a validator and return its index."""
            result = validator.validate(repo_path)
            return index, result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all validators
            futures = {
                executor.submit(run_validator, i, v): i
                for i, v in enumerate(self.validators)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results_map[index] = result
                except Exception as e:
                    # If a validator thread fails unexpectedly, record failure
                    index = futures[future]
                    validator_name = self.validators[index].name
                    logger.error(f"Validator '{validator_name}' raised exception: {e}")
                    results_map[index] = ValidationResult(
                        passed=False,
                        validator_name=validator_name,
                        message=f"Exception during validation: {e}",
                        output=str(e),
                    )

        # Reconstruct results in original order
        results = [results_map[i] for i in range(len(self.validators))]
        all_passed = all(r.passed for r in results)

        return all_passed, results

    @classmethod
    def default(
        cls,
        test_cmd: str = "pytest",
        lint_cmd: str = "ruff check .",
        parallel: bool = False,
    ) -> "ValidationPipeline":
        """Create a default validation pipeline.

        Args:
            test_cmd: Command to run tests.
            lint_cmd: Command to run linter.
            parallel: If True, run validators concurrently.
        """
        return cls(
            [
                SyntaxValidator(),
                LintValidator(lint_cmd),
                TestValidator(test_cmd),
            ],
            parallel=parallel,
        )

    def get_failure_summary(self, results: list[ValidationResult]) -> str:
        """Get a summary of failures for Claude context."""
        failures = [r for r in results if not r.passed]
        if not failures:
            return "All validations passed."

        parts = ["VALIDATION FAILURES (fix these):"]
        for f in failures:
            parts.append(f"\n## {f.validator_name.upper()} FAILED: {f.message}")
            if f.output:
                # Include more output so Claude can actually fix the issues
                output = f.output[:2000]
                if len(f.output) > 2000:
                    output += "\n... (truncated)"
                parts.append(f"```\n{output}\n```")

        return "\n".join(parts)
