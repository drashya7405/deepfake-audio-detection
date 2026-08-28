"""
In-memory Rate Limiter for Prediction Endpoint
Protects the inference server from overload on public demo deployments.
"""

import time
import threading
from collections import defaultdict, deque
from fastapi import HTTPException, Request

class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int = 15):
        self.limit = requests_per_minute
        self.window_seconds = 60.0
        self.client_records = defaultdict(deque)
        self.lock = threading.Lock()

    def check_rate_limit(self, request: Request):
        if self.limit <= 0:
            return  # Rate limiting disabled

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        with self.lock:
            timestamps = self.client_records[client_ip]
            
            # Remove timestamps older than window
            while timestamps and timestamps[0] < now - self.window_seconds:
                timestamps.popleft()

            if len(timestamps) >= self.limit:
                retry_after = int(self.window_seconds - (now - timestamps[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded ({self.limit} requests/min). Please wait {retry_after}s before trying again.",
                    headers={"Retry-After": str(retry_after)}
                )

            timestamps.append(now)

