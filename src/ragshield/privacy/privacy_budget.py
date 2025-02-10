"""Privacy Budget Management for Differential Privacy.

Implements privacy budget tracking and management following the
composition theorems of differential privacy:
- Basic composition: epsilon accumulates additively
- Advanced composition: tighter bounds using moments accountant

Key concepts:
- epsilon (ε): Privacy loss parameter (smaller = more private)
- delta (δ): Probability of privacy breach (smaller = more private)
- Budget exhaustion: When accumulated privacy loss exceeds threshold
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
from datetime import datetime
import math
import threading


class CompositionType(Enum):
    """Composition theorem types for privacy accounting."""
    BASIC = "basic"  # Simple additive composition
    ADVANCED = "advanced"  # Strong composition theorem
    ZERO_CONCENTRATED = "zcdp"  # Zero-concentrated DP


class BudgetExceededError(Exception):
    """Raised when privacy budget would be exceeded."""
    pass


@dataclass
class PrivacySpend:
    """Record of a single privacy expenditure.

    Attributes:
        epsilon: Privacy parameter epsilon
        delta: Privacy parameter delta
        operation: Description of the operation
        timestamp: When the spend occurred
        metadata: Additional context
    """
    epsilon: float
    delta: float = 0.0
    operation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetStatus:
    """Current status of privacy budget.

    Attributes:
        total_epsilon: Accumulated epsilon
        total_delta: Accumulated delta
        remaining_epsilon: Epsilon remaining before exhaustion
        remaining_delta: Delta remaining
        is_exhausted: Whether budget is exhausted
        utilization: Fraction of budget used (0.0 to 1.0+)
    """
    total_epsilon: float
    total_delta: float
    remaining_epsilon: float
    remaining_delta: float
    is_exhausted: bool
    utilization: float


class PrivacyBudgetManager:
    """Manages differential privacy budget across operations.

    Tracks cumulative privacy loss and prevents operations that would
    exceed the allocated budget. Supports multiple composition methods.

    Example:
        >>> budget = PrivacyBudgetManager(epsilon_budget=1.0, delta_budget=1e-5)
        >>> budget.can_spend(0.1)  # Check if we can spend
        True
        >>> budget.spend(0.1, operation="retrieve_query_1")
        >>> budget.get_status().remaining_epsilon
        0.9
        >>> budget.spend(1.0)  # This would exceed budget
        BudgetExceededError: Would exceed epsilon budget
    """

    def __init__(
        self,
        epsilon_budget: float = 1.0,
        delta_budget: float = 1e-5,
        composition: CompositionType = CompositionType.BASIC,
        strict_mode: bool = True,
        on_exhausted: Optional[Callable[['PrivacyBudgetManager'], None]] = None,
        warning_threshold: float = 0.8,
    ):
        """Initialize privacy budget manager.

        Args:
            epsilon_budget: Total epsilon budget
            delta_budget: Total delta budget
            composition: Composition theorem to use
            strict_mode: If True, raise exception when budget exceeded
            on_exhausted: Callback when budget is exhausted
            warning_threshold: Fraction at which to trigger warning (0.0-1.0)
        """
        if epsilon_budget <= 0:
            raise ValueError("epsilon_budget must be positive")
        if delta_budget < 0:
            raise ValueError("delta_budget must be non-negative")
        if not 0 < warning_threshold <= 1:
            raise ValueError("warning_threshold must be in (0, 1]")

        self.epsilon_budget = epsilon_budget
        self.delta_budget = delta_budget
        self.composition = composition
        self.strict_mode = strict_mode
        self.on_exhausted = on_exhausted
        self.warning_threshold = warning_threshold

        # Tracking state
        self._spent_epsilon: float = 0.0
        self._spent_delta: float = 0.0
        self._spend_history: List[PrivacySpend] = []
        self._lock = threading.Lock()
        self._exhausted_notified = False
        self._warning_notified = False

    def spend(
        self,
        epsilon: float,
        delta: float = 0.0,
        operation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> PrivacySpend:
        """Record a privacy expenditure.

        Args:
            epsilon: Epsilon to spend
            delta: Delta to spend
            operation: Description of the operation
            metadata: Additional context
            force: If True, spend even if budget exceeded (only in non-strict mode)

        Returns:
            The recorded spend

        Raises:
            BudgetExceededError: If strict_mode and budget would be exceeded
            ValueError: If epsilon or delta is negative
        """
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if delta < 0:
            raise ValueError("delta must be non-negative")

        with self._lock:
            # Calculate new totals using composition
            new_epsilon = self._compose_epsilon(self._spent_epsilon, epsilon)
            new_delta = self._compose_delta(self._spent_delta, delta)

            # Check budget
            would_exceed = (
                new_epsilon > self.epsilon_budget or
                new_delta > self.delta_budget
            )

            if would_exceed and self.strict_mode and not force:
                raise BudgetExceededError(
                    f"Would exceed budget: "
                    f"epsilon {new_epsilon:.4f} > {self.epsilon_budget:.4f} or "
                    f"delta {new_delta:.2e} > {self.delta_budget:.2e}"
                )

            # Record spend
            spend = PrivacySpend(
                epsilon=epsilon,
                delta=delta,
                operation=operation,
                metadata=metadata or {},
            )
            self._spend_history.append(spend)
            self._spent_epsilon = new_epsilon
            self._spent_delta = new_delta

            # Check for warning threshold
            utilization = self._spent_epsilon / self.epsilon_budget
            if utilization >= self.warning_threshold and not self._warning_notified:
                self._warning_notified = True

            # Check for exhaustion
            if would_exceed and not self._exhausted_notified:
                self._exhausted_notified = True
                if self.on_exhausted:
                    self.on_exhausted(self)

            return spend

    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        """Check if a spend is within budget.

        Args:
            epsilon: Epsilon to check
            delta: Delta to check

        Returns:
            True if spend is within budget
        """
        with self._lock:
            new_epsilon = self._compose_epsilon(self._spent_epsilon, epsilon)
            new_delta = self._compose_delta(self._spent_delta, delta)
            return (
                new_epsilon <= self.epsilon_budget and
                new_delta <= self.delta_budget
            )

    def get_status(self) -> BudgetStatus:
        """Get current budget status.

        Returns:
            BudgetStatus with current state
        """
        with self._lock:
            remaining_eps = max(0, self.epsilon_budget - self._spent_epsilon)
            remaining_delta = max(0, self.delta_budget - self._spent_delta)
            utilization = self._spent_epsilon / self.epsilon_budget

            return BudgetStatus(
                total_epsilon=self._spent_epsilon,
                total_delta=self._spent_delta,
                remaining_epsilon=remaining_eps,
                remaining_delta=remaining_delta,
                is_exhausted=(remaining_eps <= 0 or remaining_delta < 0),
                utilization=utilization,
            )

    def get_history(self) -> List[PrivacySpend]:
        """Get spend history.

        Returns:
            List of all privacy spends
        """
        with self._lock:
            return list(self._spend_history)

    def reset(self) -> None:
        """Reset the budget to initial state."""
        with self._lock:
            self._spent_epsilon = 0.0
            self._spent_delta = 0.0
            self._spend_history.clear()
            self._exhausted_notified = False
            self._warning_notified = False

    def _compose_epsilon(self, current: float, new: float) -> float:
        """Compose epsilon values based on composition theorem.

        Args:
            current: Current accumulated epsilon
            new: New epsilon to add

        Returns:
            Composed epsilon value
        """
        if self.composition == CompositionType.BASIC:
            # Basic composition: simple addition
            return current + new

        elif self.composition == CompositionType.ADVANCED:
            # Strong composition theorem (simplified)
            # For k queries with epsilon each: O(sqrt(k) * epsilon)
            # We approximate by tracking individual epsilons
            n = len(self._spend_history) + 1
            all_epsilons = [s.epsilon for s in self._spend_history] + [new]
            sum_sq = sum(e**2 for e in all_epsilons)
            return math.sqrt(2 * sum_sq * math.log(1 / self.delta_budget)) + sum(all_epsilons) * (math.exp(new) - 1) / (math.exp(new) + 1)

        elif self.composition == CompositionType.ZERO_CONCENTRATED:
            # zCDP: rho-zCDP composes additively
            # Convert epsilon to rho: rho = epsilon^2 / 2
            current_rho = current**2 / 2
            new_rho = new**2 / 2
            total_rho = current_rho + new_rho
            # Convert back: epsilon = sqrt(2 * rho)
            return math.sqrt(2 * total_rho)

        else:
            return current + new

    def _compose_delta(self, current: float, new: float) -> float:
        """Compose delta values.

        For basic composition, delta accumulates additively.

        Args:
            current: Current accumulated delta
            new: New delta to add

        Returns:
            Composed delta value
        """
        return current + new

    def estimate_queries_remaining(self, epsilon_per_query: float) -> int:
        """Estimate how many more queries can be made.

        Args:
            epsilon_per_query: Expected epsilon per query

        Returns:
            Estimated number of queries remaining
        """
        if epsilon_per_query <= 0:
            return float('inf')

        status = self.get_status()
        return int(status.remaining_epsilon / epsilon_per_query)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize budget state to dictionary.

        Returns:
            Dictionary representation
        """
        status = self.get_status()
        return {
            "epsilon_budget": self.epsilon_budget,
            "delta_budget": self.delta_budget,
            "composition": self.composition.value,
            "spent_epsilon": self._spent_epsilon,
            "spent_delta": self._spent_delta,
            "remaining_epsilon": status.remaining_epsilon,
            "remaining_delta": status.remaining_delta,
            "utilization": status.utilization,
            "is_exhausted": status.is_exhausted,
            "num_operations": len(self._spend_history),
        }

    def __repr__(self) -> str:
        status = self.get_status()
        return (
            f"PrivacyBudgetManager("
            f"spent={status.total_epsilon:.4f}/{self.epsilon_budget:.4f}, "
            f"remaining={status.remaining_epsilon:.4f}, "
            f"utilization={status.utilization:.1%})"
        )


class PrivacyAccountant:
    """Advanced privacy accountant with multiple budget pools.

    Supports allocating separate budgets for different operations
    (e.g., retrieval vs. analytics) with independent tracking.

    Example:
        >>> accountant = PrivacyAccountant(total_epsilon=2.0)
        >>> accountant.allocate_pool("retrieval", epsilon=1.0)
        >>> accountant.allocate_pool("analytics", epsilon=1.0)
        >>> accountant.spend("retrieval", 0.1)
        >>> accountant.get_pool_status("retrieval").remaining_epsilon
        0.9
    """

    def __init__(
        self,
        total_epsilon: float = 1.0,
        total_delta: float = 1e-5,
    ):
        """Initialize accountant.

        Args:
            total_epsilon: Total epsilon across all pools
            total_delta: Total delta across all pools
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self._pools: Dict[str, PrivacyBudgetManager] = {}
        self._allocated_epsilon: float = 0.0
        self._allocated_delta: float = 0.0
        self._lock = threading.Lock()

    def allocate_pool(
        self,
        pool_name: str,
        epsilon: float,
        delta: Optional[float] = None,
        composition: CompositionType = CompositionType.BASIC,
    ) -> PrivacyBudgetManager:
        """Allocate a named budget pool.

        Args:
            pool_name: Unique name for the pool
            epsilon: Epsilon budget for this pool
            delta: Delta budget (proportional to total if None)
            composition: Composition type for the pool

        Returns:
            The created budget manager

        Raises:
            ValueError: If pool exists or allocation exceeds total
        """
        with self._lock:
            if pool_name in self._pools:
                raise ValueError(f"Pool '{pool_name}' already exists")

            if self._allocated_epsilon + epsilon > self.total_epsilon:
                raise ValueError(
                    f"Cannot allocate {epsilon:.4f} epsilon: "
                    f"only {self.total_epsilon - self._allocated_epsilon:.4f} remaining"
                )

            if delta is None:
                # Proportional delta allocation
                delta = (epsilon / self.total_epsilon) * self.total_delta

            if self._allocated_delta + delta > self.total_delta:
                raise ValueError(f"Cannot allocate {delta:.2e} delta")

            manager = PrivacyBudgetManager(
                epsilon_budget=epsilon,
                delta_budget=delta,
                composition=composition,
            )

            self._pools[pool_name] = manager
            self._allocated_epsilon += epsilon
            self._allocated_delta += delta

            return manager

    def get_pool(self, pool_name: str) -> PrivacyBudgetManager:
        """Get a budget pool by name.

        Args:
            pool_name: Name of the pool

        Returns:
            The budget manager

        Raises:
            KeyError: If pool doesn't exist
        """
        if pool_name not in self._pools:
            raise KeyError(f"Pool '{pool_name}' not found")
        return self._pools[pool_name]

    def spend(
        self,
        pool_name: str,
        epsilon: float,
        delta: float = 0.0,
        operation: str = "",
    ) -> PrivacySpend:
        """Spend from a named pool.

        Args:
            pool_name: Pool to spend from
            epsilon: Epsilon to spend
            delta: Delta to spend
            operation: Operation description

        Returns:
            The recorded spend
        """
        return self.get_pool(pool_name).spend(epsilon, delta, operation)

    def get_pool_status(self, pool_name: str) -> BudgetStatus:
        """Get status of a specific pool.

        Args:
            pool_name: Pool name

        Returns:
            Budget status
        """
        return self.get_pool(pool_name).get_status()

    def get_total_status(self) -> Dict[str, Any]:
        """Get status across all pools.

        Returns:
            Dictionary with total and per-pool status
        """
        total_spent_eps = sum(
            p.get_status().total_epsilon for p in self._pools.values()
        )
        total_spent_delta = sum(
            p.get_status().total_delta for p in self._pools.values()
        )

        return {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "allocated_epsilon": self._allocated_epsilon,
            "allocated_delta": self._allocated_delta,
            "spent_epsilon": total_spent_eps,
            "spent_delta": total_spent_delta,
            "pools": {
                name: pool.to_dict() for name, pool in self._pools.items()
            },
        }

    def list_pools(self) -> List[str]:
        """List all pool names.

        Returns:
            List of pool names
        """
        return list(self._pools.keys())
