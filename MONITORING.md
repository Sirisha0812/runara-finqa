# FinQA Monitoring Setup Guide

## Quick Start

### 1. Start Monitoring Stack

```bash
# Start Prometheus and Grafana
docker-compose up -d

# Verify containers are running
docker-compose ps
```

Expected output:
```
NAME                   STATUS              PORTS
finqa-prometheus       Up                  0.0.0.0:9090->9090/tcp
finqa-grafana          Up                  0.0.0.0:3000->3000/tcp
```

### 2. Start FinQA Agent Metrics Server

```bash
# Terminal 1: Start metrics endpoint
python -m src.monitoring

# Terminal 2: Run agent queries (metrics will be logged automatically)
python -m src.agent --question "what is the interest expense in 2009?"
```

### 3. Access Grafana Dashboard

1. Open http://localhost:3000
2. Login:
   - **Username**: `admin`
   - **Password**: `admin`
3. Navigate to: **Dashboards** → **FinQA Agent Dashboard**

## Dashboard Overview

The dashboard includes 9 panels across 4 categories:

### 📊 Performance Metrics
- **Query Latency**: p50, p95, p99 latencies (target: <10s)
- **Per-Node Latency**: LangGraph node breakdown (retrieve, reason, calculator, verifier, answer)

### ✅ Quality Metrics
- **Verification Pass Rate**: Gauge (target: >85%)
- **Verification Status Distribution**: PASS / FAIL / UNCERTAIN bar chart
- **Answer Confidence Distribution**: HIGH / MEDIUM / LOW bar chart

### 🔄 Reliability Metrics
- **Retry Rate**: Gauge (target: <20%)
- **Calculator Usage**: Used vs Skipped counts

### 🖥️ Infrastructure Metrics
- **GPU Memory Usage**: Gauge (alert: >90%)
- **Query Drift Score**: JS divergence (alert: >0.3)

## Example Workflow

### Running Evaluation with Monitoring

```bash
# Terminal 1: Start monitoring stack
docker-compose up -d
python -m src.monitoring

# Terminal 2: Run evaluation
python -m src.evaluator --num-examples 20

# Terminal 3: Watch Grafana dashboard
open http://localhost:3000
```

The dashboard will show:
- Real-time latency percentiles
- Verification pass rate trending
- GPU memory usage during inference
- Query drift detection over 20 examples

### Simulating Production Load

```bash
# Run multiple queries to populate metrics
for i in {1..50}; do
    python -m src.agent --question "what is the interest expense in 2009?"
    sleep 2
done
```

Watch the dashboard update in real-time with:
- Latency distributions
- Verification status changes
- Retry rate evolution

## Prometheus Metrics Reference

### Histograms
```promql
finqa_query_latency_ms       # Total query latency (buckets: 100ms-60s)
finqa_node_latency_ms        # Per-node latency (labels: node=retrieve|reason|calculator|verifier|answer)
```

### Counters
```promql
finqa_verification_pass_total        # Verification passed
finqa_verification_fail_total        # Verification failed
finqa_verification_uncertain_total   # Verification uncertain
finqa_retry_triggered_total          # Retry triggered
finqa_retry_exhausted_total          # Max retries reached
finqa_calculator_used_total          # Calculator used
finqa_calculator_skipped_total       # Calculator skipped
finqa_confidence_high_total          # High confidence answers
finqa_confidence_medium_total        # Medium confidence answers
finqa_confidence_low_total           # Low confidence answers
finqa_drift_alerts_total             # Drift alerts triggered
```

### Gauges
```promql
finqa_gpu_memory_percent    # GPU memory usage (0-100)
finqa_drift_score           # JS divergence (0-1)
```

## Alert Thresholds

### Critical Alerts (Red)
- **Verification Pass Rate** < 50%
- **Retry Rate** > 60%
- **GPU Memory** > 95%
- **Drift Score** > 0.3

### Warning Alerts (Yellow)
- **Verification Pass Rate** 70-85%
- **Retry Rate** 20-40%
- **GPU Memory** 75-85%
- **Drift Score** 0.1-0.3

## Production Maintenance

### Auto-Retrain Trigger
The monitoring system tracks verification pass rate over a 24-hour rolling window:

```python
# Trigger alert if pass rate < 70% over 24hr
if verification_pass_rate_24hr < 0.70:
    logger.warning("auto_retrain_trigger", pass_rate=verification_pass_rate_24hr)
```

Response actions:
1. Review recent queries in `prometheus-data`
2. Check drift score trends
3. Retrain embeddings if drift > 0.2
4. Fine-tune LLM if pass rate continues declining

### Index Refresh Schedule
```bash
# Weekly FAISS index rebuild (cron)
0 2 * * 0 python -m src.retriever --rebuild-index
```

### Model Upgrade Path
```bash
# 1. Update model in .env
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct  # Upgrade from 1.5B → 7B

# 2. Restart vLLM server (no code changes needed)
# 3. Monitor verification_pass_rate for A/B comparison
# 4. Rollback if pass rate degrades by >5%
```

## Troubleshooting

### No Data in Grafana
1. Check metrics endpoint: `curl http://localhost:8001/metrics`
2. Check Prometheus targets: http://localhost:9090/targets
3. Verify datasource: Grafana → Connections → Prometheus → Test

### Prometheus Cannot Scrape Metrics
Mac/Windows users: `host.docker.internal` may not resolve.

**Fix**:
1. Find your IP: `ifconfig | grep inet` (Mac) or `ipconfig` (Windows)
2. Update `monitoring/prometheus.yml`:
   ```yaml
   targets: ['192.168.1.X:8001']  # Your actual IP
   ```
3. Restart: `docker-compose restart prometheus`

### Dashboard Shows "No data"
- Default time range: Last 1 hour
- If testing with sparse data, adjust time range to "Last 5 minutes" in Grafana

## Stopping Monitoring

```bash
# Stop containers (keep data)
docker-compose down

# Stop containers and delete data
docker-compose down -v
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    FinQA Agent                          │
│  ┌────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐        │
│  │Retrieve│→│ Reason │→│Calculator│→│Verifier │→Answer │
│  └────────┘ └────────┘ └──────────┘ └─────────┘        │
│                                                          │
│  src/monitoring.py: FinQAMonitor.log_agent_run()        │
│       ↓                                                  │
│  /metrics endpoint (port 8001)                          │
└──────────────────┬──────────────────────────────────────┘
                   │ scrape every 10s
                   ▼
         ┌─────────────────┐
         │   Prometheus    │ :9090
         │  (Time Series)  │
         └────────┬────────┘
                  │ query
                  ▼
         ┌─────────────────┐
         │     Grafana     │ :3000
         │  (Visualization)│
         └─────────────────┘
```

## Next Steps

1. **Customize Dashboard**: Edit panels in Grafana to add project-specific metrics
2. **Set Up Alerting**: Configure Grafana alerts to send notifications (Slack, email, PagerDuty)
3. **Long-term Storage**: Configure Prometheus remote write to long-term storage (e.g., Thanos, Cortex)
4. **Production Deployment**: Use managed Prometheus (AWS AMP, GCP Managed Prometheus) for scalability

See `monitoring/README.md` for detailed configuration reference.
