# Daemon API Reference

The um-agent-daemon is a FastAPI service providing task management, webhook integrations, and query proxy capabilities. It runs as a Docker container or standalone process.

## Quick Start

```bash
# Standalone
pip install -e ".[daemon]"
um-agent-daemon

# Docker
docker compose up
```

Default: `http://localhost:8080`

## Authentication

All endpoints accept an optional `X-API-Key` header. Auth is only enforced when `UM_DAEMON_API_KEY` is set.

```bash
curl -H "X-API-Key: your-key" http://localhost:8080/api/tasks
```

---

## System

### `GET /health`

Returns service health and task counts.

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "tasks_pending": 3,
  "tasks_running": 1
}
```

---

## Tasks API

### `POST /api/tasks`

Create a new task for background processing.

**Request:**
```json
{
  "prompt": "Implement JWT authentication",
  "spec": {"language": "python"},
  "source": "api",
  "webhook_url": "https://example.com/callback",
  "priority": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | yes | — | Task prompt (min 1 char) |
| `spec` | object | no | null | Additional task specification |
| `source` | string | no | "api" | Origin identifier |
| `webhook_url` | string | no | null | Callback URL for completion notification |
| `priority` | int | no | 5 | Priority 1-10 (higher = more urgent) |

**Response (201):**
```json
{
  "id": "abc123",
  "prompt": "Implement JWT authentication",
  "spec": {"language": "python"},
  "status": "pending",
  "source": "api",
  "source_meta": null,
  "result": null,
  "error": null,
  "created_at": "2026-03-05T10:00:00Z",
  "started_at": null,
  "completed_at": null
}
```

### `GET /api/tasks`

List tasks with optional filtering and pagination.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | string | — | Filter: pending, running, completed, failed, cancelled |
| `limit` | int | 50 | Results per page (1-200) |
| `offset` | int | 0 | Pagination offset |

**Response:**
```json
{
  "tasks": [...],
  "total": 142,
  "limit": 50,
  "offset": 0
}
```

### `GET /api/tasks/{task_id}`

Get a single task by ID. Returns 404 if not found.

### `POST /api/tasks/{task_id}/cancel`

Cancel a running or pending task.

**Request:**
```json
{
  "reason": "No longer needed"
}
```

### `GET /api/tasks/{task_id}/logs`

Get execution logs for a task.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 100 | Max log entries (1-1000) |

**Response:**
```json
{
  "task_id": "abc123",
  "logs": [
    {
      "id": 1,
      "task_id": "abc123",
      "level": "info",
      "message": "Task created via api",
      "data": null,
      "created_at": "2026-03-05T10:00:00Z"
    }
  ]
}
```

---

## Query Proxy API

Routes queries through CLI subscriptions (Codex/Gemini) so billing goes through your existing subscription rather than per-token API charges.

### `POST /api/query`

Submit a query to a CLI provider.

**Request:**
```json
{
  "prompt": "Explain quantum computing",
  "provider": "gemini",
  "model": "gemini-2.5-pro",
  "timeout": 300
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | yes | — | Query text (max 100k chars) |
| `provider` | string | yes | — | `"gemini"` or `"codex"` |
| `model` | string | no | per-provider default | Model override |
| `timeout` | int | no | 300 | Timeout in seconds (10-1800) |

**Response:**
```json
{
  "id": "q-a1b2c3d4",
  "provider": "gemini",
  "model": "gemini-2.5-pro",
  "response": "Quantum computing uses quantum-mechanical phenomena...",
  "duration_ms": 2340
}
```

**Errors:**
| Code | Meaning |
|------|---------|
| 502 | CLI returned an error |
| 503 | CLI not installed in container |
| 504 | Query timed out |

### `GET /api/query/providers`

List available providers and their authentication status.

**Response:**
```json
{
  "providers": [
    {
      "name": "codex",
      "available": true,
      "default_model": "gpt-5.2",
      "authenticated": true
    },
    {
      "name": "gemini",
      "available": true,
      "default_model": "gemini-2.5-pro",
      "authenticated": true
    }
  ]
}
```

---

## Webhook Integrations

### GitHub — `POST /webhooks/github`

Responds to GitHub webhook events. Verifies `X-Hub-Signature-256` if `UM_DAEMON_GITHUB_WEBHOOK_SECRET` is set.

**Supported events:**

| Event | Trigger | Task Created |
|-------|---------|--------------|
| `ping` | Webhook setup | No (returns pong) |
| `issue_comment` | Comment contains `/agent` | Yes — extracts command after `/agent` |
| `pull_request` | PR opened or synchronized | Yes — includes PR details |

Task IDs: `gh-{uuid}` for issues, `gh-pr-{uuid}` for PRs.

### Slack — `POST /slack/events`

Responds to Slack Events API. Verifies `X-Slack-Signature` if `UM_DAEMON_SLACK_SIGNING_SECRET` is set.

**Supported events:**

| Event | Trigger | Task Created |
|-------|---------|--------------|
| `url_verification` | Slack challenge handshake | No |
| `app_mention` | Bot is @mentioned | Yes — strips mention, uses remaining text |

Task IDs: `slack-{uuid}`

### Discord — `POST /webhooks/discord`

Responds to Discord Interactions. Verifies Ed25519 signature if `UM_DAEMON_DISCORD_PUBLIC_KEY` is set.

**Supported interactions:**

| Type | Trigger | Task Created |
|------|---------|--------------|
| PING (1) | Discord verification | No (returns PONG) |
| APPLICATION_COMMAND (2) | `/agent prompt:...` slash command | Yes |

Task IDs: `discord-{uuid}`

---

## Configuration

All settings use the `UM_DAEMON_` environment variable prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `UM_DAEMON_HOST` | 0.0.0.0 | Bind address |
| `UM_DAEMON_PORT` | 8080 | Listen port |
| `UM_DAEMON_WORKERS` | 1 | Uvicorn workers |
| `UM_DAEMON_LOG_LEVEL` | info | Log level |
| `UM_DAEMON_DB_PATH` | daemon_tasks.db | SQLite database path |
| `UM_DAEMON_API_KEY` | — | API key (auth disabled if unset) |
| `UM_DAEMON_MAX_CONCURRENT_TASKS` | 2 | Max parallel background tasks |
| `UM_DAEMON_TASK_TIMEOUT_SECONDS` | 3600 | Background task timeout |
| `UM_DAEMON_CODEX_MODEL` | gpt-5.2 | Default Codex model |
| `UM_DAEMON_GEMINI_MODEL` | gemini-2.5-pro | Default Gemini model |
| `UM_DAEMON_QUERY_RATE_LIMIT` | 60 | Query requests/min/provider |
| `UM_DAEMON_GITHUB_WEBHOOK_SECRET` | — | GitHub HMAC secret |
| `UM_DAEMON_GITHUB_TOKEN` | — | GitHub API token |
| `UM_DAEMON_SLACK_SIGNING_SECRET` | — | Slack signing secret |
| `UM_DAEMON_SLACK_BOT_TOKEN` | — | Slack bot token |
| `UM_DAEMON_DISCORD_PUBLIC_KEY` | — | Discord Ed25519 public key |
| `UM_DAEMON_DISCORD_BOT_TOKEN` | — | Discord bot token |
| `UM_DAEMON_DEFAULT_WEBHOOK_URL` | — | Default notification webhook |
| `UM_DAEMON_DEFAULT_SLACK_WEBHOOK` | — | Default Slack webhook |
| `UM_DAEMON_DEFAULT_DISCORD_WEBHOOK` | — | Default Discord webhook |

---

## Database Schema

SQLite with two tables:

```
tasks
├── id (TEXT PK)
├── prompt (TEXT NOT NULL)
├── spec (JSON)
├── status (TEXT: pending/running/completed/failed/cancelled)
├── source (TEXT: api/github/slack/discord)
├── source_meta (JSON)
├── result (JSON)
├── error (TEXT)
├── created_at (ISO 8601)
├── started_at (ISO 8601)
└── completed_at (ISO 8601)

task_logs
├── id (INTEGER PK AUTOINCREMENT)
├── task_id (TEXT FK → tasks.id)
├── level (TEXT: info/error/warning)
├── message (TEXT)
├── data (JSON)
└── created_at (ISO 8601)
```
