"""Tests for the circuit breaker module."""

import time
from unittest.mock import MagicMock

import pytest

from repo_auto_tool.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    CircuitStats,
    get_circuit_breaker,
)


class TestCircuitStats:
    """Tests for the CircuitStats dataclass."""

    def test_default_values(self) -> None:
        """Test CircuitStats default values."""
        stats = CircuitStats()
        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.rejected_calls == 0

    def test_record_success(self) -> None:
        """Test recording a successful call."""
        stats = CircuitStats()
        stats.record_success()
        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 1

    def test_record_failure(self) -> None:
        """Test recording a failed call."""
        stats = CircuitStats()
        stats.record_failure()
        assert stats.total_calls == 1
        assert stats.failed_calls == 1
        assert stats.consecutive_failures == 1
        assert stats.consecutive_successes == 0

    def test_record_rejection(self) -> None:
        """Test recording a rejected call."""
        stats = CircuitStats()
        stats.record_rejection()
        assert stats.rejected_calls == 1
        assert stats.total_calls == 0  # Rejections don't count as calls

    def test_get_failure_rate_zero_calls(self) -> None:
        """Test failure rate with zero calls."""
        stats = CircuitStats()
        assert stats.get_failure_rate() == 0.0

    def test_get_failure_rate(self) -> None:
        """Test failure rate calculation."""
        stats = CircuitStats()
        stats.record_success()
        stats.record_failure()
        assert stats.get_failure_rate() == 50.0


class TestCircuitBreakerConfig:
    """Tests for the CircuitBreakerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 60.0
        assert config.half_open_max_calls == 1

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout=30.0,
            half_open_max_calls=2,
        )
        assert config.failure_threshold == 3
        assert config.success_threshold == 1
        assert config.timeout == 30.0
        assert config.half_open_max_calls == 2

    def test_bounds_validation(self) -> None:
        """Test that values are clamped to valid bounds."""
        config = CircuitBreakerConfig(
            failure_threshold=0,  # Too low
            timeout=0.5,  # Too low
        )
        assert config.failure_threshold == 1  # Clamped to min
        assert config.timeout == 1.0  # Clamped to min


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""

    def test_initial_state_is_closed(self) -> None:
        """Test that circuit breaker starts in closed state."""
        breaker = CircuitBreaker("test")
        assert breaker.is_closed()
        assert breaker.state == CircuitState.CLOSED

    def test_successful_call(self) -> None:
        """Test successful call through circuit breaker."""
        breaker = CircuitBreaker("test")
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.stats.successful_calls == 1

    def test_failed_call(self) -> None:
        """Test failed call through circuit breaker."""
        breaker = CircuitBreaker("test")
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("test error")))
        assert breaker.stats.failed_calls == 1

    def test_circuit_opens_after_threshold(self) -> None:
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test", config)

        # First failure
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        assert breaker.is_closed()

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        assert breaker.is_open()

    def test_circuit_rejects_when_open(self) -> None:
        """Test calls are rejected when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=60.0)
        breaker = CircuitBreaker("test", config)

        # Fail to open the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        assert breaker.is_open()

        # Next call should be rejected
        with pytest.raises(CircuitOpenError) as exc_info:
            breaker.call(lambda: "success")
        assert exc_info.value.circuit_name == "test"
        assert breaker.stats.rejected_calls == 1

    def test_fallback_on_open(self) -> None:
        """Test fallback function is called when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        fallback = MagicMock(return_value="fallback")
        breaker = CircuitBreaker("test", config, fallback=fallback)

        # Fail to open the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        # Fallback should be called
        result = breaker.call(lambda: "success")
        assert result == "fallback"
        fallback.assert_called_once()

    def test_decorator_usage(self) -> None:
        """Test using circuit breaker as a decorator."""
        breaker = CircuitBreaker("test")

        @breaker
        def my_function() -> str:
            return "decorated result"

        result = my_function()
        assert result == "decorated result"
        assert breaker.stats.successful_calls == 1

    def test_context_manager_success(self) -> None:
        """Test using circuit breaker as context manager for success."""
        breaker = CircuitBreaker("test")

        with breaker:
            pass  # Success

        assert breaker.stats.successful_calls == 1

    def test_context_manager_failure(self) -> None:
        """Test using circuit breaker as context manager for failure."""
        breaker = CircuitBreaker("test")

        with pytest.raises(ValueError):
            with breaker:
                raise ValueError("test error")

        assert breaker.stats.failed_calls == 1

    def test_manual_reset(self) -> None:
        """Test manually resetting the circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        assert breaker.is_open()

        # Reset it
        breaker.reset()
        assert breaker.is_closed()

    def test_force_open(self) -> None:
        """Test forcing the circuit open."""
        breaker = CircuitBreaker("test")
        assert breaker.is_closed()

        breaker.force_open()
        assert breaker.is_open()

    def test_get_status_dict(self) -> None:
        """Test getting status as dictionary."""
        breaker = CircuitBreaker("test")
        breaker.call(lambda: "success")

        status = breaker.get_status_dict()
        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["stats"]["successful_calls"] == 1

    def test_half_open_recovery(self) -> None:
        """Test recovery through half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=1,
            timeout=1.0,  # Minimum allowed timeout
        )
        breaker = CircuitBreaker("test", config)

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))
        assert breaker.is_open()

        # Wait for timeout
        time.sleep(1.1)

        # Next call should go through (half-open state) and close circuit
        result = breaker.call(lambda: "recovered")
        assert result == "recovered"
        assert breaker.is_closed()

    def test_excluded_exceptions(self) -> None:
        """Test that excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            exclude_exceptions=(ValueError,),
        )
        breaker = CircuitBreaker("test", config)

        # ValueError should not count as failure (excluded)
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("excluded")))
        assert breaker.is_closed()  # Still closed because excluded
        assert breaker.stats.failed_calls == 0  # Excluded exceptions not recorded
        assert breaker.stats.consecutive_failures == 0  # No consecutive failures


class TestCircuitBreakerRegistry:
    """Tests for the CircuitBreakerRegistry class."""

    def test_register_and_get(self) -> None:
        """Test registering and getting a circuit breaker."""
        registry = CircuitBreakerRegistry()
        breaker = CircuitBreaker("registry_test")
        registry.register(breaker)

        retrieved = registry.get("registry_test")
        assert retrieved is breaker

    def test_get_nonexistent(self) -> None:
        """Test getting a nonexistent circuit breaker returns None."""
        registry = CircuitBreakerRegistry()
        assert registry.get("nonexistent") is None

    def test_get_or_create(self) -> None:
        """Test get_or_create creates if not exists."""
        registry = CircuitBreakerRegistry()
        breaker = registry.get_or_create("get_or_create_test")
        assert breaker is not None
        assert breaker.name == "get_or_create_test"

        # Second call should return same instance
        breaker2 = registry.get_or_create("get_or_create_test")
        assert breaker is breaker2

    def test_list_all(self) -> None:
        """Test listing all circuit breakers."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("list_test_1")
        registry.get_or_create("list_test_2")

        names = registry.list_all()
        assert "list_test_1" in names
        assert "list_test_2" in names

    def test_reset_all(self) -> None:
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("reset_all_test", config)
        registry.register(breaker)

        # Open the circuit
        breaker.force_open()
        assert breaker.is_open()

        # Reset all
        registry.reset_all()
        assert breaker.is_closed()


class TestGetCircuitBreaker:
    """Tests for the get_circuit_breaker function."""

    def test_get_circuit_breaker(self) -> None:
        """Test getting a circuit breaker from global registry."""
        breaker = get_circuit_breaker("global_test")
        assert breaker is not None
        assert breaker.name == "global_test"

    def test_get_circuit_breaker_with_config(self) -> None:
        """Test getting a circuit breaker with custom config."""
        config = CircuitBreakerConfig(failure_threshold=10)
        breaker = get_circuit_breaker("global_config_test", config)
        assert breaker.config.failure_threshold == 10
