"""
FinQA Agent Monitoring

Prometheus metrics and drift detection for production monitoring.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
from datetime import datetime
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from src.logger import get_logger

logger = get_logger(__name__)

# Try to import prometheus_client, make it optional
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client_not_available", msg="Install with: pip install prometheus-client")


class PrometheusMetrics:
    """Prometheus metrics for FinQA agent."""

    def __init__(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_metrics_disabled", reason="prometheus_client_not_installed")
            self.enabled = False
            return

        self.enabled = True

        # Query latency histogram
        self.query_latency_ms = Histogram(
            'finqa_query_latency_ms',
            'Query processing latency in milliseconds',
            buckets=[100, 500, 1000, 2000, 5000, 10000, 20000, 30000]
        )

        # Verification pass rate (0 or 1 per query)
        self.verification_pass = Counter(
            'finqa_verification_pass_total',
            'Total number of verification passes'
        )
        self.verification_fail = Counter(
            'finqa_verification_fail_total',
            'Total number of verification failures'
        )
        self.verification_uncertain = Counter(
            'finqa_verification_uncertain_total',
            'Total number of uncertain verifications'
        )

        # Retry metrics
        self.retry_triggered = Counter(
            'finqa_retry_triggered_total',
            'Total number of retries triggered'
        )
        self.retry_exhausted = Counter(
            'finqa_retry_exhausted_total',
            'Total number of retry exhaustions'
        )

        # Answer confidence (gauge for latest value)
        self.answer_confidence = Gauge(
            'finqa_answer_confidence',
            'Answer confidence score (1.0=HIGH, 0.5=MEDIUM, 0.0=LOW)'
        )

        # Calculator usage
        self.calculator_used = Counter(
            'finqa_calculator_used_total',
            'Total number of calculator usages'
        )
        self.calculator_skipped = Counter(
            'finqa_calculator_skipped_total',
            'Total number of calculator skips'
        )

        # Query counter
        self.query_total = Counter(
            'finqa_query_total',
            'Total number of queries processed'
        )

        # Error counter
        self.error_total = Counter(
            'finqa_error_total',
            'Total number of errors',
            ['error_type']
        )

        logger.info("prometheus_metrics_initialized")

    def record_query(
        self,
        latency_ms: float,
        verification_status: str,
        verification_confidence: str,
        retry_count: int,
        retry_exhausted: bool,
        calculator_used: bool,
        error: Optional[str] = None
    ):
        """
        Record metrics for a single query.

        Args:
            latency_ms: Query latency in milliseconds
            verification_status: PASS/FAIL/UNCERTAIN
            verification_confidence: HIGH/MEDIUM/LOW
            retry_count: Number of retries used
            retry_exhausted: Whether retries were exhausted
            calculator_used: Whether calculator was used
            error: Error message if any
        """
        if not self.enabled:
            return

        # Query counter
        self.query_total.inc()

        # Latency
        self.query_latency_ms.observe(latency_ms)

        # Verification
        if verification_status == "PASS":
            self.verification_pass.inc()
        elif verification_status == "FAIL":
            self.verification_fail.inc()
        elif verification_status == "UNCERTAIN":
            self.verification_uncertain.inc()

        # Retry
        if retry_count > 0:
            self.retry_triggered.inc()
        if retry_exhausted:
            self.retry_exhausted.inc()

        # Confidence (convert to numeric)
        confidence_map = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0}
        confidence_score = confidence_map.get(verification_confidence, 0.0)
        self.answer_confidence.set(confidence_score)

        # Calculator
        if calculator_used:
            self.calculator_used.inc()
        else:
            self.calculator_skipped.inc()

        # Errors
        if error:
            self.error_total.labels(error_type=type(error).__name__).inc()


class QueryDriftDetector:
    """
    Detects distribution drift in query embeddings using JS-divergence.

    Uses the same embedding model as retriever for consistency.
    """

    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.1,
        num_bins: int = 50
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Number of queries to keep in reference window
            drift_threshold: JS-divergence threshold for drift alert
            num_bins: Number of bins for histogram
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.num_bins = num_bins

        # Reference distribution (first N queries)
        self.reference_embeddings: Optional[np.ndarray] = None
        self.reference_distribution: Optional[np.ndarray] = None

        # Current window of embeddings
        self.current_embeddings: deque = deque(maxlen=window_size)

        # Drift history
        self.drift_scores: List[float] = []
        self.drift_alerts: List[Dict[str, Any]] = []

        logger.info(
            "drift_detector_initialized",
            window_size=window_size,
            threshold=drift_threshold,
            num_bins=num_bins
        )

    def _compute_histogram(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute histogram from embeddings.

        Uses L2 norms of embeddings as the distribution to track.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            Normalized histogram
        """
        if len(embeddings) == 0:
            return np.zeros(self.num_bins)

        # Compute L2 norms
        norms = np.linalg.norm(embeddings, axis=1)

        # Create histogram
        hist, _ = np.histogram(norms, bins=self.num_bins, density=True)

        # Normalize to probability distribution
        hist = hist + 1e-10  # Add small epsilon to avoid zeros
        hist = hist / hist.sum()

        return hist

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions.

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            JS divergence (0 to 1)
        """
        # JSD is symmetric, bounded [0, 1]
        return float(jensenshannon(p, q))

    def add_query_embedding(self, embedding: np.ndarray) -> Optional[float]:
        """
        Add a query embedding and check for drift.

        Args:
            embedding: Query embedding vector

        Returns:
            JS divergence score if drift check was performed, None otherwise
        """
        # Add to current window
        self.current_embeddings.append(embedding)

        # If we don't have reference yet, build it
        if self.reference_embeddings is None:
            if len(self.current_embeddings) >= self.window_size:
                self.reference_embeddings = np.array(list(self.current_embeddings))
                self.reference_distribution = self._compute_histogram(self.reference_embeddings)
                logger.info(
                    "reference_distribution_established",
                    num_samples=len(self.reference_embeddings)
                )
            return None

        # Check drift periodically (every 10 queries)
        if len(self.current_embeddings) % 10 == 0:
            # Compute current distribution
            current_embeddings_array = np.array(list(self.current_embeddings))
            current_distribution = self._compute_histogram(current_embeddings_array)

            # Compute JS divergence
            js_div = self._js_divergence(self.reference_distribution, current_distribution)
            self.drift_scores.append(js_div)

            logger.info(
                "drift_check",
                js_divergence=round(js_div, 4),
                threshold=self.drift_threshold,
                drift_detected=js_div > self.drift_threshold
            )

            # Alert if drift detected
            if js_div > self.drift_threshold:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "js_divergence": round(js_div, 4),
                    "threshold": self.drift_threshold,
                    "num_queries": len(self.current_embeddings),
                }
                self.drift_alerts.append(alert)

                logger.warning(
                    "drift_alert",
                    **alert
                )

            return js_div

        return None

    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get drift detection summary.

        Returns:
            Dictionary with drift statistics
        """
        return {
            "reference_established": self.reference_embeddings is not None,
            "current_window_size": len(self.current_embeddings),
            "num_drift_checks": len(self.drift_scores),
            "num_drift_alerts": len(self.drift_alerts),
            "latest_drift_score": self.drift_scores[-1] if self.drift_scores else None,
            "avg_drift_score": np.mean(self.drift_scores) if self.drift_scores else None,
            "max_drift_score": np.max(self.drift_scores) if self.drift_scores else None,
            "drift_alerts": self.drift_alerts,
        }


class AgentMonitor:
    """
    Combined monitoring for FinQA agent.

    Tracks Prometheus metrics and query drift.
    """

    def __init__(
        self,
        enable_drift_detection: bool = True,
        drift_window_size: int = 100,
        drift_threshold: float = 0.1
    ):
        """
        Initialize agent monitor.

        Args:
            enable_drift_detection: Whether to enable drift detection
            drift_window_size: Window size for drift detection
            drift_threshold: JS-divergence threshold for alerts
        """
        self.metrics = PrometheusMetrics()

        self.drift_detector = None
        if enable_drift_detection:
            self.drift_detector = QueryDriftDetector(
                window_size=drift_window_size,
                drift_threshold=drift_threshold
            )

        logger.info(
            "agent_monitor_initialized",
            prometheus_enabled=self.metrics.enabled,
            drift_detection_enabled=enable_drift_detection
        )

    def record_agent_run(
        self,
        result: Dict[str, Any],
        query_embedding: Optional[np.ndarray] = None,
        start_time: Optional[float] = None
    ):
        """
        Record metrics from an agent run.

        Args:
            result: Agent result dictionary
            query_embedding: Query embedding for drift detection
            start_time: Start time (from time.time()) for latency calculation
        """
        # Calculate latency
        if start_time is not None:
            latency_ms = (time.time() - start_time) * 1000
        else:
            # Use node traces if available
            node_traces = result.get("node_traces", [])
            latency_ms = sum(trace.get("duration_ms", 0) for trace in node_traces)

        # Extract fields
        verification_status = result.get("verification_status", "UNCERTAIN")
        verification_confidence = result.get("verification_confidence", "LOW")
        retry_count = result.get("retry_count", 0)
        retry_exhausted = result.get("retry_exhausted", False)

        # Check if calculator was used
        calculator_used = False
        for trace in result.get("trace", []):
            if trace.get("node") == "calculator" and not trace.get("skipped"):
                calculator_used = True
                break

        # Check for errors
        error = None
        for trace in result.get("trace", []):
            if trace.get("error"):
                error = trace.get("error")
                break

        # Record Prometheus metrics
        self.metrics.record_query(
            latency_ms=latency_ms,
            verification_status=verification_status,
            verification_confidence=verification_confidence,
            retry_count=retry_count,
            retry_exhausted=retry_exhausted,
            calculator_used=calculator_used,
            error=error
        )

        # Record drift if embedding provided
        if self.drift_detector is not None and query_embedding is not None:
            drift_score = self.drift_detector.add_query_embedding(query_embedding)
            if drift_score is not None:
                logger.info("drift_score_computed", score=round(drift_score, 4))

        logger.info(
            "agent_run_recorded",
            latency_ms=round(latency_ms, 2),
            verification_status=verification_status,
            retries=retry_count,
            calculator_used=calculator_used
        )

    def get_drift_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get drift detection summary.

        Returns:
            Drift summary dict or None if drift detection disabled
        """
        if self.drift_detector is None:
            return None
        return self.drift_detector.get_drift_summary()


# Global monitor instance (lazy init)
_global_monitor: Optional[AgentMonitor] = None


def get_monitor(
    enable_drift_detection: bool = True,
    drift_window_size: int = 100,
    drift_threshold: float = 0.1
) -> AgentMonitor:
    """
    Get or create global monitor instance.

    Args:
        enable_drift_detection: Whether to enable drift detection
        drift_window_size: Window size for drift detection
        drift_threshold: JS-divergence threshold for alerts

    Returns:
        Global AgentMonitor instance
    """
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = AgentMonitor(
            enable_drift_detection=enable_drift_detection,
            drift_window_size=drift_window_size,
            drift_threshold=drift_threshold
        )

    return _global_monitor


if __name__ == "__main__":
    """Test monitoring functionality."""
    import numpy as np

    print("Testing FinQA Agent Monitoring\n")

    # Test Prometheus metrics
    print("=" * 80)
    print("Testing Prometheus Metrics")
    print("=" * 80)

    metrics = PrometheusMetrics()
    if metrics.enabled:
        metrics.record_query(
            latency_ms=1500.5,
            verification_status="PASS",
            verification_confidence="HIGH",
            retry_count=0,
            retry_exhausted=False,
            calculator_used=True,
            error=None
        )
        print("✓ Metrics recorded successfully")
    else:
        print("⚠ Prometheus not available (install: pip install prometheus-client)")

    # Test drift detector
    print("\n" + "=" * 80)
    print("Testing Drift Detection")
    print("=" * 80)

    detector = QueryDriftDetector(window_size=50, drift_threshold=0.1, num_bins=20)

    # Simulate reference distribution (normal)
    print("\nBuilding reference distribution (50 embeddings)...")
    for i in range(50):
        emb = np.random.randn(384)  # 384-dim like sentence-transformers
        detector.add_query_embedding(emb)

    # Simulate current queries (no drift)
    print("\nAdding 20 similar queries (no drift expected)...")
    for i in range(20):
        emb = np.random.randn(384)
        drift_score = detector.add_query_embedding(emb)
        if drift_score is not None:
            print(f"  Drift check: JS-divergence = {drift_score:.4f}")

    # Simulate drift (shift distribution)
    print("\nAdding 20 shifted queries (drift expected)...")
    for i in range(20):
        emb = np.random.randn(384) + 2.0  # Shift mean
        drift_score = detector.add_query_embedding(emb)
        if drift_score is not None:
            print(f"  Drift check: JS-divergence = {drift_score:.4f}")

    # Print summary
    summary = detector.get_drift_summary()
    print("\nDrift Summary:")
    for key, value in summary.items():
        if key != "drift_alerts":
            print(f"  {key}: {value}")

    if summary["drift_alerts"]:
        print(f"\n  Drift alerts: {len(summary['drift_alerts'])}")
        for alert in summary["drift_alerts"]:
            print(f"    - {alert}")

    print("\n" + "=" * 80)
