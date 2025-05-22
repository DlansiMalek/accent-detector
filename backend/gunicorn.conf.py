import multiprocessing

# Gunicorn configuration for the accent detector API
bind = "0.0.0.0:$PORT"  # Use the PORT environment variable
workers = 1  # Only use 1 worker due to memory constraints with ML models
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 600  # 10 minute timeout for full video processing
keepalive = 5
threads = 1
preload_app = False  # Don't preload to avoid memory issues
max_requests = 10  # Restart workers periodically to free memory
max_requests_jitter = 3
