from prometheus_client import Histogram,Gauge,Counter

REQUEST_COUNT= Counter(
    "http_request_total",
    "Total number of HTTP requests",
    ["method", "path", "status"]
)
REQUEST_LATENCY= Histogram(
    "http_request_duration_seconds",
    "Latency of the HTTP request",
    ["path"]
)
GPU_UTIL= Gauge("gpu_utilization", "GPU used percent")
GPU_MEMORY= Gauge("gpu_memory_used_bytes", "GPU used percent")