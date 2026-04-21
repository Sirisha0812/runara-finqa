"""
FinQA Agent Monitoring

Production monitoring with Prometheus metrics, drift detection, and maintenance automation.

MAINTENANCE PLAN:
1. Auto-retrain trigger: If verification_pass_rate < 70% over 24hr rolling window,
   trigger alert for model fine-tuning or prompt engineering review.

2. Index refresh: Rebuild FAISS index on:
   - Weekly schedule (cron: 0 2 * * 0)
   - When new documents added to training set (webhook trigger)
   - After index corruption detected (validation check fails)

3. Model upgrade path:
   - Update VLLM_MODEL in .env (e.g., Qwen/Qwen2.5-7B-Instruct → Qwen2.5-32B)
   - Restart vLLM server (zero code changes needed)
   - Monitor verification_pass_rate for A/B comparison
   - Rollback if pass rate degrades by >5%

4. GPU monitoring:
   - Alert if gpu_memory_percent > 90% for 10min (OOM risk)
   - Alert if query_latency_p99 > 30s (user experience degradation)
   - Scale horizontally: add vLLM replicas behind load balancer

5. Drift response:
   - JS-divergence > 0.1: Log warning, increase monitoring
   - JS-divergence > 0.3: Critical alert, review query patterns
   - Consider retraining retriever embeddings if drift persists
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Deque
from collections import deque, defaultdict
from datetime import datetime, timedelta
from scipy.spatial.distance import jensenshannon

from src.logger import get_logger

logger = get_logger(__name__)

# Prometheus imports (optional dependency)
try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        start_http_server, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_not_available",
        msg="Install with: pip install prometheus-client"
    )


class FinQAMonitor:
    """
    Production monitoring for FinQA agent.

    Tracks Prometheus metrics, detects query drift, and provides maintenance insights.
    """

    def __init__(
        self,
        enable_prometheus: bool = True,
        enable_drift_detection: bool = True,
        drift_window_size: int = 100,
        drift_threshold: float = 0.1,
        metrics_port: int = 8001
    ):
        """
        Initialize FinQA monitor.

        Args:
            enable_prometheus: Enable Prometheus metrics
            enable_drift_detection: Enable drift detection
            drift_window_size: Sliding window size for drift detection
            drift_threshold: JS-divergence threshold for drift alerts
            metrics_port: Port for /metrics endpoint (default 8001)
        """
        self.drift_window_size = drift_window_size
        self.drift_threshold = drift_threshold
        self.metrics_port = metrics_port

        # Initialize Prometheus metrics
        if enable_prometheus and PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
            self.prometheus_enabled = True
        else:
            self.prometheus_enabled = False
            if enable_prometheus:
                logger.warning("prometheus_disabled", reason="library_not_installed")

        # Drift detection
        self.drift_detection_enabled = enable_drift_detection
        if enable_drift_detection:
            self._init_drift_detector()

        # Verification pass rate tracking (24hr window)
        self.verification_history: Deque[Dict[str, Any]] = deque(maxlen=1000)

        logger.info(
            "monitor_initialized",
            prometheus_enabled=self.prometheus_enabled,
            drift_detection_enabled=self.drift_detection_enabled,
            metrics_port=metrics_port
        )

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""

        # Query latency histogram (total agent run time)
        self.query_latency_ms = Histogram(
            'finqa_query_latency_ms',
            'Total agent run time in milliseconds',
            buckets=[100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 60000]
        )

        # Node-level latency histograms
        self.node_latency_ms = Histogram(
            'finqa_node_latency_ms',
            'Per-node latency in milliseconds',
            ['node'],  # Labels: retrieve, reason, calculator, verifier, answer
            buckets=[10, 50, 100, 500, 1000, 2000, 5000, 10000]
        )

        # Verification outcomes
        self.verification_pass = Counter(
            'finqa_verification_pass_total',
            'Verification PASS count'
        )
        self.verification_fail = Counter(
            'finqa_verification_fail_total',
            'Verification FAIL count'
        )
        self.verification_uncertain = Counter(
            'finqa_verification_uncertain_total',
            'Verification UNCERTAIN count'
        )
        self.verification_error = Counter(
            'finqa_verification_error_total',
            'Verification ERROR count'
        )

        # Retry metrics
        self.retry_triggered = Counter(
            'finqa_retry_triggered_total',
            'Number of retries triggered'
        )
        self.retry_exhausted = Counter(
            'finqa_retry_exhausted_total',
            'Number of retry exhaustions'
        )

        # Calculator usage
        self.calculator_used = Counter(
            'finqa_calculator_used_total',
            'Calculator used count'
        )
        self.calculator_skipped = Counter(
            'finqa_calculator_skipped_total',
            'Calculator skipped count'
        )

        # Answer confidence
        self.confidence_high = Counter(
            'finqa_confidence_high_total',
            'HIGH confidence answers'
        )
        self.confidence_medium = Counter(
            'finqa_confidence_medium_total',
            'MEDIUM confidence answers'
        )
        self.confidence_low = Counter(
            'finqa_confidence_low_total',
            'LOW confidence answers'
        )

        # GPU memory utilization
        self.gpu_memory_percent = Gauge(
            'finqa_gpu_memory_percent',
            'GPU memory utilization percentage'
        )

        # Total queries
        self.query_total = Counter(
            'finqa_query_total',
            'Total queries processed'
        )

        # Drift detection
        self.drift_score = Gauge(
            'finqa_drift_score',
            'Latest JS-divergence drift score'
        )
        self.drift_alerts = Counter(
            'finqa_drift_alerts_total',
            'Total drift alerts triggered'
        )

        logger.info("prometheus_metrics_initialized")

    def _init_drift_detector(self):
        """Initialize drift detection."""
        # Reference distribution (first N queries)
        self.reference_embeddings: Optional[np.ndarray] = None
        self.reference_distribution: Optional[np.ndarray] = None

        # Current sliding window
        self.embedding_window: Deque[np.ndarray] = deque(maxlen=self.drift_window_size)

        # Drift history
        self.drift_scores: List[float] = []
        self.drift_alert_history: List[Dict[str, Any]] = []

        # Histogram bins for distribution
        self.num_bins = 50

        logger.info(
            "drift_detector_initialized",
            window_size=self.drift_window_size,
            threshold=self.drift_threshold
        )

    def _compute_distribution(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute probability distribution from embeddings.

        Uses L2 norms as distribution feature (same pattern as retriever).

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            Normalized histogram (probability distribution)
        """
        if len(embeddings) == 0:
            return np.zeros(self.num_bins)

        # Compute L2 norms
        norms = np.linalg.norm(embeddings, axis=1)

        # Create histogram
        hist, _ = np.histogram(norms, bins=self.num_bins, density=True)

        # Normalize to probability distribution
        hist = hist + 1e-10  # Avoid zeros
        hist = hist / hist.sum()

        return hist

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence.

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            JS divergence (0 to 1)
        """
        return float(jensenshannon(p, q))

    def log_agent_run(
        self,
        result: Dict[str, Any],
        latency_ms: float,
        query_embedding: Optional[np.ndarray] = None,
        gpu_memory_percent: Optional[float] = None
    ):
        """
        Log metrics from an agent run.

        Args:
            result: Agent result dictionary
            latency_ms: Total agent run time in milliseconds
            query_embedding: Query embedding for drift detection
            gpu_memory_percent: GPU memory utilization percentage
        """
        # Increment query counter
        if self.prometheus_enabled:
            self.query_total.inc()

        # Total latency
        if self.prometheus_enabled:
            self.query_latency_ms.observe(latency_ms)

        # Node-level latencies
        if self.prometheus_enabled:
            for trace in result.get("node_traces", []):
                node = trace.get("node", "unknown")
                duration_ms = trace.get("duration_ms", 0)
                self.node_latency_ms.labels(node=node).observe(duration_ms)

        # Verification status
        verification_status = result.get("verification_status", "UNCERTAIN")
        if self.prometheus_enabled:
            if verification_status == "PASS":
                self.verification_pass.inc()
            elif verification_status == "FAIL":
                self.verification_fail.inc()
            elif verification_status == "UNCERTAIN":
                self.verification_uncertain.inc()
            elif verification_status == "ERROR":
                self.verification_error.inc()

        # Track verification for auto-retrain trigger
        self.verification_history.append({
            "timestamp": datetime.now(),
            "status": verification_status,
            "passed": verification_status == "PASS"
        })

        # Check auto-retrain trigger (24hr window)
        self._check_auto_retrain_trigger()

        # Retry metrics
        retry_count = result.get("retry_count", 0)
        retry_exhausted = result.get("retry_exhausted", False)
        if self.prometheus_enabled:
            if retry_count > 0:
                self.retry_triggered.inc()
            if retry_exhausted:
                self.retry_exhausted.inc()

        # Calculator usage
        calculator_used = False
        for trace in result.get("trace", []):
            if trace.get("node") == "calculator" and not trace.get("skipped"):
                calculator_used = True
                break
        if self.prometheus_enabled:
            if calculator_used:
                self.calculator_used.inc()
            else:
                self.calculator_skipped.inc()

        # Answer confidence
        confidence = result.get("verification_confidence", "LOW")
        if self.prometheus_enabled:
            if confidence == "HIGH":
                self.confidence_high.inc()
            elif confidence == "MEDIUM":
                self.confidence_medium.inc()
            else:
                self.confidence_low.inc()

        # GPU memory
        if gpu_memory_percent is not None and self.prometheus_enabled:
            self.gpu_memory_percent.set(gpu_memory_percent)

        # Drift detection
        if self.drift_detection_enabled and query_embedding is not None:
            drift_score = self._check_drift(query_embedding)
            if drift_score is not None and self.prometheus_enabled:
                self.drift_score.set(drift_score)

        logger.info(
            "agent_run_logged",
            latency_ms=round(latency_ms, 2),
            verification=verification_status,
            retries=retry_count,
            calculator_used=calculator_used,
            confidence=confidence
        )

    def _check_drift(self, embedding: np.ndarray) -> Optional[float]:
        """
        Check for distribution drift.

        Args:
            embedding: Query embedding

        Returns:
            JS-divergence score if check performed, None otherwise
        """
        # Add to window
        self.embedding_window.append(embedding)

        # Establish reference if not yet done
        if self.reference_embeddings is None:
            if len(self.embedding_window) >= self.drift_window_size:
                self.reference_embeddings = np.array(list(self.embedding_window))
                self.reference_distribution = self._compute_distribution(
                    self.reference_embeddings
                )
                logger.info(
                    "reference_distribution_established",
                    num_samples=len(self.reference_embeddings)
                )
            return None

        # Check drift every 10 queries
        if len(self.embedding_window) % 10 == 0:
            current_embeddings = np.array(list(self.embedding_window))
            current_distribution = self._compute_distribution(current_embeddings)

            js_div = self._js_divergence(self.reference_distribution, current_distribution)
            self.drift_scores.append(js_div)

            # Determine severity
            if js_div > self.drift_threshold:
                if js_div > 0.3:
                    severity = "CRITICAL"
                elif js_div > 0.2:
                    severity = "HIGH"
                else:
                    severity = "MEDIUM"

                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "js_divergence": round(js_div, 4),
                    "threshold": self.drift_threshold,
                    "severity": severity,
                    "num_queries": len(self.embedding_window)
                }
                self.drift_alert_history.append(alert)

                if self.prometheus_enabled:
                    self.drift_alerts.inc()

                logger.warning("drift_alert", **alert)
            else:
                logger.info(
                    "drift_check",
                    js_divergence=round(js_div, 4),
                    threshold=self.drift_threshold,
                    drift_detected=False
                )

            return js_div

        return None

    def _check_auto_retrain_trigger(self):
        """
        Check if auto-retrain should be triggered based on verification pass rate.

        Triggers alert if pass rate < 70% over 24hr window.
        """
        if len(self.verification_history) < 20:  # Need minimum samples
            return

        # Filter to last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [v for v in self.verification_history if v["timestamp"] >= cutoff]

        if len(recent) < 20:  # Need minimum samples in 24hr window
            return

        # Calculate pass rate
        passed = sum(1 for v in recent if v["passed"])
        pass_rate = passed / len(recent)

        # Trigger alert if < 70%
        if pass_rate < 0.70:
            logger.warning(
                "auto_retrain_trigger",
                pass_rate=round(pass_rate, 3),
                threshold=0.70,
                samples_24hr=len(recent),
                action="review_model_or_prompts"
            )

    def start_metrics_server(self):
        """
        Start HTTP server for /metrics endpoint.

        Prometheus can scrape metrics from http://localhost:{metrics_port}/metrics
        """
        if not self.prometheus_enabled:
            logger.warning(
                "metrics_server_not_started",
                reason="prometheus_not_enabled"
            )
            return

        try:
            start_http_server(self.metrics_port)
            logger.info(
                "metrics_server_started",
                port=self.metrics_port,
                endpoint=f"http://localhost:{self.metrics_port}/metrics"
            )
            print(f"\n✓ Prometheus metrics available at: http://localhost:{self.metrics_port}/metrics")
            print("  Configure Grafana to scrape this endpoint\n")
        except Exception as e:
            logger.error("metrics_server_failed", error=str(e))

    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get drift detection summary.

        Returns:
            Drift statistics
        """
        if not self.drift_detection_enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "reference_established": self.reference_embeddings is not None,
            "current_window_size": len(self.embedding_window),
            "num_drift_checks": len(self.drift_scores),
            "num_drift_alerts": len(self.drift_alert_history),
            "latest_drift_score": self.drift_scores[-1] if self.drift_scores else None,
            "avg_drift_score": np.mean(self.drift_scores) if self.drift_scores else None,
            "max_drift_score": np.max(self.drift_scores) if self.drift_scores else None,
            "recent_alerts": self.drift_alert_history[-5:] if self.drift_alert_history else []
        }

    def get_verification_summary(self) -> Dict[str, Any]:
        """
        Get verification pass rate summary.

        Returns:
            Verification statistics
        """
        if len(self.verification_history) == 0:
            return {
                "total_queries": 0,
                "pass_rate_24hr": None,
                "pass_rate_all_time": None
            }

        # All time
        total = len(self.verification_history)
        passed_all = sum(1 for v in self.verification_history if v["passed"])
        pass_rate_all = passed_all / total if total > 0 else 0

        # Last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [v for v in self.verification_history if v["timestamp"] >= cutoff]
        passed_24hr = sum(1 for v in recent if v["passed"])
        pass_rate_24hr = passed_24hr / len(recent) if len(recent) > 0 else None

        return {
            "total_queries": total,
            "queries_24hr": len(recent),
            "pass_rate_all_time": round(pass_rate_all, 3),
            "pass_rate_24hr": round(pass_rate_24hr, 3) if pass_rate_24hr is not None else None,
            "auto_retrain_triggered": pass_rate_24hr is not None and pass_rate_24hr < 0.70
        }


# Global monitor instance
_global_monitor: Optional[FinQAMonitor] = None


def get_monitor(
    enable_prometheus: bool = True,
    enable_drift_detection: bool = True,
    drift_window_size: int = 100,
    drift_threshold: float = 0.1,
    metrics_port: int = 8001
) -> FinQAMonitor:
    """
    Get or create global monitor instance.

    Args:
        enable_prometheus: Enable Prometheus metrics
        enable_drift_detection: Enable drift detection
        drift_window_size: Sliding window size
        drift_threshold: JS-divergence threshold
        metrics_port: Port for /metrics endpoint

    Returns:
        Global FinQAMonitor instance
    """
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = FinQAMonitor(
            enable_prometheus=enable_prometheus,
            enable_drift_detection=enable_drift_detection,
            drift_window_size=drift_window_size,
            drift_threshold=drift_threshold,
            metrics_port=metrics_port
        )

    return _global_monitor


if __name__ == "__main__":
    """Test monitoring and start metrics server."""
    import sys

    print("=" * 80)
    print("FinQA Agent Monitoring - Test & Metrics Server")
    print("=" * 80)

    # Create monitor
    monitor = get_monitor()

    # Simulate some agent runs
    print("\nSimulating 10 agent runs...")
    for i in range(10):
        # Simulate agent result
        result = {
            "verification_status": "PASS" if i % 3 != 0 else "UNCERTAIN",
            "verification_confidence": "HIGH" if i % 2 == 0 else "MEDIUM",
            "retry_count": 1 if i % 5 == 0 else 0,
            "retry_exhausted": False,
            "node_traces": [
                {"node": "retrieve", "duration_ms": 200 + i * 10},
                {"node": "reason", "duration_ms": 1500 + i * 100},
                {"node": "calculator", "duration_ms": 50},
                {"node": "verifier", "duration_ms": 800 + i * 50},
                {"node": "answer", "duration_ms": 600 + i * 30},
            ],
            "trace": [
                {"node": "calculator", "skipped": i % 2 == 0}
            ]
        }

        # Simulate embedding
        embedding = np.random.randn(384)

        # Log metrics
        monitor.log_agent_run(
            result=result,
            latency_ms=3150 + i * 190,
            query_embedding=embedding,
            gpu_memory_percent=85.0 + i * 0.5
        )

    # Print summaries
    print("\n" + "=" * 80)
    print("Drift Summary:")
    print("=" * 80)
    drift_summary = monitor.get_drift_summary()
    for key, value in drift_summary.items():
        if key != "recent_alerts":
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("Verification Summary:")
    print("=" * 80)
    ver_summary = monitor.get_verification_summary()
    for key, value in ver_summary.items():
        print(f"  {key}: {value}")

    # Start metrics server
    if monitor.prometheus_enabled:
        print("\n" + "=" * 80)
        print("Starting Prometheus metrics server...")
        print("=" * 80)
        monitor.start_metrics_server()
        print("\nServer running. Press Ctrl+C to stop.")
        try:
            # Keep server running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
    else:
        print("\n⚠ Prometheus not available. Install with: pip install prometheus-client")

    print("\n" + "=" * 80)
