"""Circuit breaker pattern for fault tolerance.

This module provides a circuit breaker implementation to prevent cascading
failures when external dependencies (Claude API, git, file I/O) are failing
repeatedly. The circuit breaker has three states:

- CLOSED: Normal operation, requests go through
- OPEN: Failures exceeded threshold, requests fail immediately
- HALF_OPEN: Testing if the service recovered, allows limited requests

All code is defensively written with robust error handling.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """States of a circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, not allowing requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and call is rejected."""

    def __init__(
        self,
        circuit_name: str,
        failures: int,
        retry_after: float,
    ):
        self.circuit_name = circuit_name
        self.failures = failures
        self.retry_after = retry_after
        super().__init__(
            f"Circuit '{circuit_name}' is OPEN after {failures} failures. "
            f"Retry after {retry_after:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker.

    Attributes:
        failure_threshold: Number of failures to trigger open state
        success_threshold: Number of successes in half-open to close circuit
        timeout: Seconds to wait before transitioning from open to half-open
        half_open_max_calls: Max concurrent calls allowed in half-open state
        exclude_exceptions: Exception types that don't count as failures
        include_exceptions: If set, only these exceptions count as failures
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    half_open_max_calls: int = 1
    exclude_exceptions: tuple[type[Exception], ...] = field(default_factory=tuple)
    include_exceptions: tuple[type[Exception], ...] | None = None

    def __post_init__(self) -> None:
        """Validate configuration with defensive bounds."""
        self.failure_threshold = max(1, min(100, self.failure_threshold))
        self.success_threshold = max(1, min(50, self.success_threshold))
        self.timeout = max(1.0, min(3600.0, self.timeout))
        self.half_open_max_calls = max(1, min(10, self.half_open_max_calls))


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker.

    Attributes:
        total_calls: Total number of calls made
        successful_calls: Number of successful calls
        failed_calls: Number of failed calls
        rejected_calls: Number of calls rejected due to open circuit
        state_changes: Number of state transitions
        last_failure_time: Timestamp of last failure
        last_success_time: Timestamp of last success
        consecutive_failures: Current consecutive failure count
        consecutive_successes: Current consecutive success count in half-open
    """

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    def record_success(self) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.last_success_time = time.time()
        self.consecutive_failures = 0
        self.consecutive_successes += 1

    def record_failure(self) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.last_failure_time = time.time()
        self.consecutive_failures += 1
        self.consecutive_successes = 0

    def record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        self.rejected_calls += 1

    def record_state_change(self) -> None:
        """Record a state transition."""
        self.state_changes += 1

    def get_failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.failed_calls / self.total_calls) * 100

    def reset_consecutive_counts(self) -> None:
        """Reset consecutive counters (on state change)."""
        self.consecutive_failures = 0
        self.consecutive_successes = 0


class CircuitBreaker(Generic[T]):
    """Thread-safe circuit breaker implementation.

    Prevents cascading failures by failing fast when a dependency is unhealthy.

    Example:
        breaker = CircuitBreaker("claude_api")

        @breaker
        def call_claude(prompt: str) -> str:
            return claude_client.complete(prompt)

        # Or use as context manager
        with breaker:
            result = call_claude(prompt)

        # Or call directly
        try:
            result = breaker.call(lambda: api.request())
        except CircuitOpenError:
            result = use_fallback()
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        fallback: Callable[[], T] | None = None,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Name for this circuit (for logging/identification)
            config: Configuration options
            fallback: Optional fallback function when circuit is open
            on_state_change: Callback when state changes
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = time.time()
        self._half_open_calls = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        with self._lock:
            return self._stats

    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        with self._lock:
            return self._state == CircuitState.CLOSED

    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        with self._lock:
            return self._state == CircuitState.OPEN

    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        with self._lock:
            return self._state == CircuitState.HALF_OPEN

    def _should_allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                elapsed = time.time() - self._last_state_change
                if elapsed >= self.config.timeout:
                    # Transition to half-open
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return

            self._state = new_state
            self._last_state_change = time.time()
            self._stats.record_state_change()

            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._stats.reset_consecutive_counts()

            logger.info(
                f"Circuit '{self.name}' transitioned: {old_state.value} -> {new_state.value}"
            )

            if self.on_state_change:
                try:
                    self.on_state_change(self.name, old_state, new_state)
                except Exception as e:
                    logger.debug(f"State change callback error: {e}")

    def _should_count_exception(self, exc: Exception) -> bool:
        """Check if an exception should be counted as a failure."""
        # Check exclude list first
        if self.config.exclude_exceptions:
            if isinstance(exc, self.config.exclude_exceptions):
                return False

        # Check include list if set
        if self.config.include_exceptions is not None:
            return isinstance(exc, self.config.include_exceptions)

        return True

    def _handle_success(self) -> None:
        """Handle a successful call."""
        with self._lock:
            self._stats.record_success()

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    # Recovered - close the circuit
                    self._transition_to(CircuitState.CLOSED)

    def _handle_failure(self, exc: Exception) -> None:
        """Handle a failed call."""
        with self._lock:
            if not self._should_count_exception(exc):
                logger.debug(f"Exception {type(exc).__name__} excluded from failure count")
                return

            self._stats.record_failure()

            if self._state == CircuitState.HALF_OPEN:
                # Recovery failed - reopen circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    # Too many failures - open circuit
                    self._transition_to(CircuitState.OPEN)

    def call(self, func: Callable[[], T]) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: The function to execute

        Returns:
            The function's return value

        Raises:
            CircuitOpenError: If the circuit is open
            Exception: Any exception from the function
        """
        if not self._should_allow_request():
            self._stats.record_rejection()

            # Try fallback if available
            if self.fallback:
                try:
                    logger.debug(f"Circuit '{self.name}' open, using fallback")
                    return self.fallback()
                except Exception as e:
                    logger.debug(f"Fallback failed: {e}")

            elapsed = time.time() - self._last_state_change
            retry_after = max(0.0, self.config.timeout - elapsed)
            raise CircuitOpenError(
                self.name,
                self._stats.consecutive_failures,
                retry_after,
            )

        try:
            result = func()
            self._handle_success()
            return result
        except Exception as e:
            self._handle_failure(e)
            raise

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as a decorator.

        Example:
            @circuit_breaker
            def api_call():
                return client.request()
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.call(lambda: func(*args, **kwargs))

        return wrapper

    def __enter__(self) -> CircuitBreaker[T]:
        """Context manager entry."""
        if not self._should_allow_request():
            self._stats.record_rejection()
            elapsed = time.time() - self._last_state_change
            retry_after = max(0.0, self.config.timeout - elapsed)
            raise CircuitOpenError(
                self.name,
                self._stats.consecutive_failures,
                retry_after,
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Context manager exit."""
        if exc_type is None:
            self._handle_success()
        elif isinstance(exc_val, Exception):
            self._handle_failure(exc_val)
        return False  # Don't suppress exceptions

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._last_state_change = time.time()
            self._half_open_calls = 0
            self._stats.reset_consecutive_counts()

            logger.info(
                f"Circuit '{self.name}' manually reset from {old_state.value}"
            )

    def force_open(self) -> None:
        """Manually force the circuit to open state."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            logger.info(f"Circuit '{self.name}' manually forced open")

    def get_status_dict(self) -> dict[str, Any]:
        """Get a dictionary representation of the circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "stats": {
                    "total_calls": self._stats.total_calls,
                    "successful_calls": self._stats.successful_calls,
                    "failed_calls": self._stats.failed_calls,
                    "rejected_calls": self._stats.rejected_calls,
                    "failure_rate": self._stats.get_failure_rate(),
                    "consecutive_failures": self._stats.consecutive_failures,
                },
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout,
                },
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers.

    Provides centralized access and monitoring of all circuit breakers
    in the application.
    """

    _instance: CircuitBreakerRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> CircuitBreakerRegistry:
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._breakers: dict[str, CircuitBreaker[Any]] = {}
                cls._instance._registry_lock = threading.Lock()
            return cls._instance

    def register(self, breaker: CircuitBreaker[Any]) -> None:
        """Register a circuit breaker."""
        with self._registry_lock:
            self._breakers[breaker.name] = breaker
            logger.debug(f"Registered circuit breaker: {breaker.name}")

    def get(self, name: str) -> CircuitBreaker[Any] | None:
        """Get a circuit breaker by name."""
        with self._registry_lock:
            return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker[Any]:
        """Get an existing circuit breaker or create a new one."""
        with self._registry_lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def list_all(self) -> list[str]:
        """List all registered circuit breaker names."""
        with self._registry_lock:
            return list(self._breakers.keys())

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers."""
        with self._registry_lock:
            return {name: breaker.get_status_dict() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._registry_lock:
            for breaker in self._breakers.values():
                try:
                    breaker.reset()
                except Exception as e:
                    logger.debug(f"Error resetting {breaker.name}: {e}")

    def get_open_circuits(self) -> list[str]:
        """Get names of all open circuits."""
        with self._registry_lock:
            return [name for name, breaker in self._breakers.items() if breaker.is_open()]


# Global registry instance
registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker[Any]:
    """Get or create a circuit breaker from the global registry.

    This is the recommended way to access circuit breakers,
    ensuring consistent naming and centralized monitoring.

    Args:
        name: Name for the circuit breaker
        config: Optional configuration

    Returns:
        CircuitBreaker instance
    """
    return registry.get_or_create(name, config)


# Pre-configured circuit breakers for common operations
def create_claude_circuit_breaker() -> CircuitBreaker[Any]:
    """Create a circuit breaker configured for Claude API calls.

    Uses higher thresholds and longer timeout suitable for API calls.
    """
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=1,
        timeout=120.0,  # 2 minutes before retry
        half_open_max_calls=1,
    )
    breaker: CircuitBreaker[Any] = CircuitBreaker("claude_api", config)
    registry.register(breaker)
    return breaker


def create_git_circuit_breaker() -> CircuitBreaker[Any]:
    """Create a circuit breaker configured for git operations.

    Uses lower thresholds for faster recovery.
    """
    config = CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout=30.0,
        half_open_max_calls=2,
    )
    breaker: CircuitBreaker[Any] = CircuitBreaker("git_ops", config)
    registry.register(breaker)
    return breaker


def create_file_io_circuit_breaker() -> CircuitBreaker[Any]:
    """Create a circuit breaker configured for file I/O operations.

    Uses very low thresholds as file I/O should rarely fail.
    """
    config = CircuitBreakerConfig(
        failure_threshold=10,
        success_threshold=1,
        timeout=10.0,
        half_open_max_calls=3,
    )
    breaker: CircuitBreaker[Any] = CircuitBreaker("file_io", config)
    registry.register(breaker)
    return breaker
