"""Safety utilities for detecting secrets and dangerous commands."""

import re
from dataclasses import dataclass


@dataclass
class SecretMatch:
    """Represents a detected secret in text."""

    secret_type: str
    matched_text: str
    start_position: int
    end_position: int
    redacted_text: str


@dataclass
class DangerousCommand:
    """Represents a flagged dangerous command."""

    command: str
    pattern_name: str
    description: str
    severity: str  # "warning" or "critical"


class SecretsRedactor:
    """Detects and redacts secrets from text using regex patterns."""

    # Pattern definitions: (name, pattern, description)
    SECRET_PATTERNS: list[tuple[str, str, str]] = [
        (
            "aws_access_key",
            r"AKIA[0-9A-Z]{16}",
            "AWS Access Key ID",
        ),
        (
            "anthropic_api_key",
            r"sk-ant-api\d{2}-[A-Za-z0-9_-]{86}",
            "Anthropic API Key",
        ),
        (
            "anthropic_api_key_short",
            r"sk-ant-[A-Za-z0-9_-]{20,}",
            "Anthropic API Key (short form)",
        ),
        (
            "openai_api_key",
            r"sk-[A-Za-z0-9]{20,}",
            "OpenAI API Key",
        ),
        (
            "generic_api_key",
            r"(?i)(?:api[_-]?key|apikey)['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_-]{20,})['\"]?",
            "Generic API Key assignment",
        ),
        (
            "generic_token",
            r"(?i)(?:token|auth[_-]?token|access[_-]?token)['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_-]{20,})['\"]?",
            "Generic Token assignment",
        ),
        (
            "password_assignment",
            r"(?i)(?:password|passwd|pwd)['\"]?\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?",
            "Password assignment",
        ),
        (
            "bearer_token",
            r"Bearer\s+[A-Za-z0-9_-]{20,}",
            "Bearer Token",
        ),
        (
            "basic_auth",
            r"Basic\s+[A-Za-z0-9+/=]{20,}",
            "Basic Auth credentials",
        ),
        (
            "private_key",
            r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
            "Private Key header",
        ),
        (
            "github_token",
            r"gh[pousr]_[A-Za-z0-9_]{36,}",
            "GitHub Token",
        ),
    ]

    def __init__(self) -> None:
        """Initialize the redactor with compiled patterns."""
        self._compiled_patterns: list[tuple[str, re.Pattern[str], str]] = []
        for name, pattern, description in self.SECRET_PATTERNS:
            try:
                compiled = re.compile(pattern)
                self._compiled_patterns.append((name, compiled, description))
            except re.error:
                # Skip invalid patterns
                pass

    def find_secrets(self, text: str) -> list[SecretMatch]:
        """Find all secrets in the given text.

        Args:
            text: The text to scan for secrets.

        Returns:
            List of SecretMatch objects for each detected secret.
        """
        matches: list[SecretMatch] = []
        seen_positions: set[tuple[int, int]] = set()

        for name, pattern, _description in self._compiled_patterns:
            for match in pattern.finditer(text):
                # Get the actual matched secret (could be in a group)
                if match.lastindex:
                    # Pattern has groups, use the first captured group
                    matched_text = match.group(1)
                    start = match.start(1)
                    end = match.end(1)
                else:
                    matched_text = match.group(0)
                    start = match.start()
                    end = match.end()

                # Avoid duplicate matches at the same position
                pos_key = (start, end)
                if pos_key in seen_positions:
                    continue
                seen_positions.add(pos_key)

                # Create redacted version
                if len(matched_text) > 8:
                    redacted = matched_text[:4] + "*" * (len(matched_text) - 8) + matched_text[-4:]
                else:
                    redacted = "*" * len(matched_text)

                matches.append(
                    SecretMatch(
                        secret_type=name,
                        matched_text=matched_text,
                        start_position=start,
                        end_position=end,
                        redacted_text=redacted,
                    )
                )

        # Sort by position for consistent output
        matches.sort(key=lambda m: m.start_position)
        return matches

    def redact(self, text: str) -> str:
        """Redact all detected secrets from the text.

        Args:
            text: The text to redact secrets from.

        Returns:
            Text with all detected secrets replaced with redacted versions.
        """
        matches = self.find_secrets(text)
        if not matches:
            return text

        # Build result by replacing matches from end to start
        # (to preserve positions)
        result = text
        for match in reversed(matches):
            result = (
                result[: match.start_position]
                + match.redacted_text
                + result[match.end_position :]
            )
        return result


class DangerousCommandDetector:
    """Detects potentially dangerous commands (flags but does not block)."""

    # Pattern definitions: (name, pattern, description, severity)
    DANGEROUS_PATTERNS: list[tuple[str, str, str, str]] = [
        (
            "curl_pipe_bash",
            r"curl\s+[^\|]*\|\s*(?:ba)?sh",
            "Piping curl output to shell - potential code execution risk",
            "critical",
        ),
        (
            "wget_pipe_bash",
            r"wget\s+[^\|]*\|\s*(?:ba)?sh",
            "Piping wget output to shell - potential code execution risk",
            "critical",
        ),
        (
            "rm_rf_root",
            r"rm\s+-[rf]{1,2}[rf]?\s+/(?:\s|$)",
            "Recursive delete of root directory",
            "critical",
        ),
        (
            "rm_rf_wildcard",
            r"rm\s+-[rf]{1,2}[rf]?\s+\*",
            "Recursive delete with wildcard - may delete unintended files",
            "warning",
        ),
        (
            "pip_install_url",
            r"pip\s+install\s+https?://",
            "Installing package from URL - verify source before running",
            "warning",
        ),
        (
            "pip_install_git",
            r"pip\s+install\s+git\+",
            "Installing package from git - verify repository before running",
            "warning",
        ),
        (
            "chmod_777",
            r"chmod\s+777\s+",
            "Setting overly permissive file permissions",
            "warning",
        ),
        (
            "eval_command",
            r"eval\s+[\"'\$]",
            "Using eval with dynamic content - potential injection risk",
            "warning",
        ),
        (
            "dd_device",
            r"dd\s+.*of=/dev/",
            "Writing directly to device - potential data loss risk",
            "critical",
        ),
        (
            "mkfs_command",
            r"mkfs\s+",
            "Formatting filesystem - potential data loss risk",
            "critical",
        ),
    ]

    def __init__(self) -> None:
        """Initialize the detector with compiled patterns."""
        self._compiled_patterns: list[tuple[str, re.Pattern[str], str, str]] = []
        for name, pattern, description, severity in self.DANGEROUS_PATTERNS:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self._compiled_patterns.append((name, compiled, description, severity))
            except re.error:
                # Skip invalid patterns
                pass

    def detect(self, command: str) -> list[DangerousCommand]:
        """Detect dangerous patterns in a command.

        Args:
            command: The command string to analyze.

        Returns:
            List of DangerousCommand objects for each detected pattern.
        """
        detected: list[DangerousCommand] = []

        for name, pattern, description, severity in self._compiled_patterns:
            if pattern.search(command):
                detected.append(
                    DangerousCommand(
                        command=command,
                        pattern_name=name,
                        description=description,
                        severity=severity,
                    )
                )

        return detected

    def is_dangerous(self, command: str) -> bool:
        """Check if a command contains any dangerous patterns.

        Args:
            command: The command string to check.

        Returns:
            True if any dangerous patterns are detected.
        """
        return len(self.detect(command)) > 0


class SafetyManager:
    """Facade class coordinating secret redaction and dangerous command detection."""

    def __init__(
        self,
        redact_secrets: bool = True,
        detect_dangerous: bool = True,
    ) -> None:
        """Initialize the safety manager.

        Args:
            redact_secrets: Whether to enable secret redaction.
            detect_dangerous: Whether to enable dangerous command detection.
        """
        self._redact_secrets = redact_secrets
        self._detect_dangerous = detect_dangerous
        self._redactor = SecretsRedactor() if redact_secrets else None
        self._detector = DangerousCommandDetector() if detect_dangerous else None

    def process_text(self, text: str) -> tuple[str, list[SecretMatch]]:
        """Process text, redacting secrets if enabled.

        Args:
            text: The text to process.

        Returns:
            Tuple of (processed_text, list of detected secrets).
        """
        if not self._redact_secrets or self._redactor is None:
            return text, []

        matches = self._redactor.find_secrets(text)
        redacted = self._redactor.redact(text)
        return redacted, matches

    def check_command(self, command: str) -> list[DangerousCommand]:
        """Check a command for dangerous patterns.

        Args:
            command: The command to check.

        Returns:
            List of detected dangerous command patterns.
        """
        if not self._detect_dangerous or self._detector is None:
            return []

        return self._detector.detect(command)

    def redact(self, text: str) -> str:
        """Convenience method to just redact text.

        Args:
            text: The text to redact.

        Returns:
            Redacted text.
        """
        processed, _ = self.process_text(text)
        return processed

    def is_command_dangerous(self, command: str) -> bool:
        """Check if a command is dangerous.

        Args:
            command: The command to check.

        Returns:
            True if the command contains dangerous patterns.
        """
        return len(self.check_command(command)) > 0
