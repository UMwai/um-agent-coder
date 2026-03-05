FROM python:3.12-slim

# Install git + curl (needed for repo operations and Node.js setup)
RUN apt-get update && apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js 22 (for codex + gemini CLIs)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Codex and Gemini CLIs globally
RUN npm install -g @openai/codex @google/gemini-cli

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install with daemon extras
RUN pip install --no-cache-dir ".[daemon]"

# Create non-root user with credential mount points
RUN useradd --create-home appuser && \
    mkdir -p /app/data /home/appuser/.codex /home/appuser/.gemini && \
    chown -R appuser:appuser /app /home/appuser/.codex /home/appuser/.gemini
USER appuser

# Default env vars
ENV UM_DAEMON_HOST=0.0.0.0
ENV UM_DAEMON_PORT=8080
ENV UM_DAEMON_DB_PATH=/app/data/daemon_tasks.db
ENV UM_DAEMON_CHECKPOINT_DIR=/app/data/checkpoints

EXPOSE 8080

CMD ["um-agent-daemon"]
