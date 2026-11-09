"""Circuit breaker pattern for LLM API calls.

This module provides a circuit breaker implementation to prevent cascading
failures when an LLM provider is experiencing issues. The circuit breaker
tracks failures and automatically transitions between closed, open, and
half-open states.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests go through
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass(slots=True)
class CircuitBreakerConfig:
    """Configuration for the circuit breaker."""

    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Successes needed to close from half-open
    timeout: float = 60.0  # Seconds before transitioning from open to half-open
    excluded_exceptions: tuple[type[BaseException], ...] = (
        asyncio.CancelledError,
        KeyboardInterrupt,
    )


@dataclass(slots=True)
class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    The circuit breaker wraps a callable and tracks its success/failure rate.
    When failures exceed the threshold, the circuit opens and subsequent calls
    fail immediately without calling the wrapped function. After a timeout,
    the circuit enters half-open state to test if the service has recovered.
    """

    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """Current state of the circuit breaker."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """True when the circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """True when the circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """True when the circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def _record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._success_count = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        self._success_count = 0

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._state = CircuitState.OPEN
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN

    def _check_timeout(self) -> bool:
        """Check if enough time has passed to transition from open to half-open."""
        if self._state != CircuitState.OPEN:
            return False
        return (time.monotonic() - self._last_failure_time) >= self.config.timeout

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> T:
        """Execute the function with circuit breaker protection.

        Args:
            func: The async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call

        Raises:
            CircuitOpenError: When the circuit is open and not ready to test
            The original exception from the function if it fails
        """
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._check_timeout():
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitOpenError(
                        f"Circuit breaker is open. Retry after "
                        f"{self.config.timeout - (time.monotonic() - self._last_failure_time):.1f}s"
                    )

        try:
            result = await func(*args, **kwargs)
        except self.config.excluded_exceptions:
            # Don't count excluded exceptions as failures
            raise
        except BaseException as exc:
            async with self._lock:
                self._record_failure()
            raise
        else:
            async with self._lock:
                self._record_success()
            return result

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and blocking requests."""

    pass