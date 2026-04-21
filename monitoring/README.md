# FinQA Monitoring Stack

Production monitoring setup for FinQA agent with Prometheus and Grafana.

## Quick Start

### 1. Start the monitoring stack

```bash
docker-compose up -d
```

This starts:
- **Prometheus** on port 9090
- **Grafana** on port 3000

### 2. Start the FinQA agent metrics server

```bash
# In a separate terminal
python -m src.monitoring
```

This starts the `/metrics` endpoint on port 8001.

### 3. Access Grafana

1. Open http://localhost:3000
2. Login with:
   - Username: `admin`
   - Password: `admin`
3. Navigate to Dashboards → FinQA Agent Dashboard

## Dashboard Panels

The pre-configured dashboard includes:

### Performance Metrics
- **Query Latency**: p50, p95, p99 latencies for end-to-end queries
- **Per-Node Latency**: Breakdown by LangGraph nodes (retrieve, reason, calculator, verifier, answer)

### Quality Metrics
- **Verification Pass Rate**: Percentage of answers that pass verification
- **Verification Status Distribution**: PASS / FAIL / UNCERTAIN counts
- **Answer Confidence Distribution**: HIGH / MEDIUM / LOW confidence levels

### Reliability Metrics
- **Retry Rate**: Percentage of queries requiring retries
- **Calculator Usage**: Used vs Skipped counts

### Infrastructure Metrics
- **GPU Memory Usage**: Real-time GPU memory percentage
- **Query Drift Score**: JS divergence for distribution drift detection

## Prometheus Configuration

Prometheus scrapes the FinQA agent metrics every 10 seconds from `host.docker.internal:8001/metrics`.

To view raw metrics:
- Prometheus UI: http://localhost:9090
- Metrics endpoint: http://localhost:8001/metrics (when agent is running)

## Example Prometheus Queries

```promql
# Average query latency over 5 minutes
rate(finqa_query_latency_ms_sum[5m]) / rate(finqa_query_latency_ms_count[5m])

# Verification pass rate
100 * rate(finqa_verification_pass_total[5m]) /
  (rate(finqa_verification_pass_total[5m]) +
   rate(finqa_verification_fail_total[5m]) +
   rate(finqa_verification_uncertain_total[5m]))

# GPU memory usage
finqa_gpu_memory_percent

# Drift score
finqa_drift_score
```

## Alerts

The dashboard uses color-coded thresholds:

### Verification Pass Rate
- 🟢 Green: ≥85%
- 🟡 Yellow: 70-85%
- 🟠 Orange: 50-70%
- 🔴 Red: <50%

### Retry Rate
- 🟢 Green: <20%
- 🟡 Yellow: 20-40%
- 🟠 Orange: 40-60%
- 🔴 Red: >60%

### GPU Memory
- 🟢 Green: <75%
- 🟡 Yellow: 75-85%
- 🟠 Orange: 85-95%
- 🔴 Red: >95%

### Drift Score (JS Divergence)
- 🟢 Green: <0.1
- 🟡 Yellow: 0.1-0.3
- 🔴 Red: >0.3

## Stopping the Stack

```bash
docker-compose down
```

To remove volumes (deletes historical data):

```bash
docker-compose down -v
```

## Data Persistence

Prometheus and Grafana data is persisted in Docker volumes:
- `prometheus-data`: Time series metrics
- `grafana-data`: Dashboards and settings

## Architecture

```
┌──────────────┐
│ FinQA Agent  │ :8001/metrics
└──────┬───────┘
       │
       │ scrape every 10s
       ▼
┌──────────────┐
│  Prometheus  │ :9090
└──────┬───────┘
       │
       │ query
       ▼
┌──────────────┐
│   Grafana    │ :3000
└──────────────┘
```

## Troubleshooting

### Prometheus not scraping metrics

1. Check agent is running:
   ```bash
   curl http://localhost:8001/metrics
   ```

2. Check Prometheus targets:
   - Open http://localhost:9090/targets
   - Verify `finqa-agent` target is UP

### Grafana dashboard not showing data

1. Verify Prometheus datasource:
   - Grafana → Connections → Data Sources → Prometheus
   - Click "Test" to verify connection

2. Check time range:
   - Dashboard uses "Last 1 hour" by default
   - Adjust if testing with sparse data

### Docker network issues (Mac/Windows)

If `host.docker.internal` doesn't work:

1. Find your machine's IP: `ifconfig` (Mac) or `ipconfig` (Windows)
2. Update `monitoring/prometheus.yml`:
   ```yaml
   targets: ['YOUR_IP:8001']
   ```
3. Restart: `docker-compose restart prometheus`

## Integration with Agent

The agent automatically logs metrics via `src/monitoring.py`:

```python
from src.monitoring import get_monitor

monitor = get_monitor()
monitor.log_agent_run(result, latency_ms, query_embedding, gpu_memory_percent)
```

See `src/monitoring.py` for full API and maintenance plan.
