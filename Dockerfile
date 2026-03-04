FROM python:3.12-slim

# Install git (needed for repo operations)
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install with daemon extras
RUN pip install --no-cache-dir ".[daemon]"

# Create non-root user
RUN useradd --create-home appuser && \
    mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

# Default env vars
ENV UM_DAEMON_HOST=0.0.0.0
ENV UM_DAEMON_PORT=8080
ENV UM_DAEMON_DB_PATH=/app/data/daemon_tasks.db
ENV UM_DAEMON_CHECKPOINT_DIR=/app/data/checkpoints

EXPOSE 8080

CMD ["um-agent-daemon"]
