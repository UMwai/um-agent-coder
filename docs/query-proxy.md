# Query Proxy Setup Guide

Route AI queries through the Google Code Assist API using your Google One AI Ultra subscription. The proxy calls the API directly from Python — no CLI subprocess or Node.js required.

## How It Works

```
Client → POST /api/query → FastAPI → Code Assist API (cloudcode-pa) → Response
                                          ↑
                              OAuth creds loaded from ~/.gemini/oauth_creds.json
                              Auto token refresh via Google OAuth2
                              Round-robin across Gemini 3 models
```

The client uses the Gemini CLI's OAuth refresh token to obtain access tokens, then calls Google's Code Assist API directly. Queries are billed through your Google One AI Ultra subscription — no per-token API charges.

## Supported Models

| Short Name | Model ID | Sustained req/min | Avg Latency |
|------------|----------|-------------------|-------------|
| `flash` | `gemini-3-flash-preview` | ~30 | ~1.8s |
| `pro` | `gemini-3-pro-preview` | ~18 | ~3.3s |
| `pro-3.1` | `gemini-3.1-pro-preview` | ~18 | ~3.3s |
| **`auto`** | **Round-robin all three** | **~66** | — |

Per-model rate limits are independent — `auto` mode round-robins across all models to maximize throughput.

## Prerequisites

- Docker and Docker Compose
- **Google One AI Ultra** subscription (or Gemini Advanced)
- Gemini CLI authenticated on your local machine:
  ```bash
  npx @google/gemini-cli auth    # creates ~/.gemini/oauth_creds.json
  ```

## Setup

### 1. Extract credentials

```bash
./scripts/extract-oauth-creds.sh
```

This copies your OAuth tokens into `secrets/` (gitignored):

```
secrets/
└── gemini/
    ├── oauth_creds.json  # OAuth tokens (refresh_token required)
    └── settings.json     # Optional settings
```

### 2. Build and run

```bash
docker compose up --build
```

### 3. Verify models

```bash
curl http://localhost:8080/api/query/models
```

Expected output:
```json
{
  "authenticated": true,
  "tier": "g1-ultra-tier",
  "models": [
    {"name": "flash", "model_id": "gemini-3-flash-preview", "available": true},
    {"name": "pro", "model_id": "gemini-3-pro-preview", "available": true},
    {"name": "pro-3.1", "model_id": "gemini-3.1-pro-preview", "available": true}
  ]
}
```

### 4. Send queries

```bash
# Default (auto round-robin)
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain TCP/IP in one paragraph"}'

# Specific model
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python fibonacci function", "model": "pro"}'

# With system prompt and temperature
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "model": "flash", "system_prompt": "Reply in French", "temperature": 0.3}'
```

Response:
```json
{
  "id": "q-a1b2c3d4",
  "model": "gemini-3-flash-preview",
  "response": "...",
  "duration_ms": 1823,
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 156,
    "total_tokens": 168
  }
}
```

## API Authentication

Set `UM_DAEMON_API_KEY` to require an API key on all endpoints:

```bash
# In .env or docker-compose override
UM_DAEMON_API_KEY=my-secret-key
```

```bash
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-secret-key" \
  -d '{"prompt": "Hello", "model": "flash"}'
```

## Cloud Run Deployment

### 1. Store credentials as GCP secrets

```bash
gcloud secrets create gemini-oauth-creds --data-file=secrets/gemini/oauth_creds.json
```

### 2. Deploy

```bash
GCP_PROJECT_ID=your-project ./deploy/deploy.sh
```

The deploy script mounts secrets as files at `/home/appuser/.gemini/`.

### 3. Set API key

```bash
gcloud run services update um-agent-daemon \
  --set-env-vars "UM_DAEMON_API_KEY=your-production-key"
```

## Configuration Reference

| Env Variable | Default | Description |
|--------------|---------|-------------|
| `UM_DAEMON_GEMINI_MODEL` | `gemini-3-flash-preview` | Default model for queries |
| `UM_DAEMON_GEMINI_AUTO_MODELS` | `gemini-3-flash-preview,gemini-3-pro-preview,gemini-3.1-pro-preview` | Models for auto round-robin |
| `UM_DAEMON_QUERY_RATE_LIMIT` | 60 | Max requests/min |
| `UM_DAEMON_API_KEY` | — | API key (disabled if unset) |

## Re-Authentication

If your Gemini refresh token gets revoked (password change, manual revocation), queries will fail with auth errors. To re-authenticate:

### Via Web UI

1. Visit `https://<your-daemon-url>/api/auth/gemini` in your browser
2. Follow the on-screen instructions to run `npx @google/gemini-cli auth` locally
3. Extract your refresh token and paste it into the form
4. The daemon updates the GCP secret and hot-reloads the client automatically

### Via API

```bash
# Check auth health
curl https://<your-daemon-url>/api/auth/gemini/status

# Submit new token
curl -X POST https://<your-daemon-url>/api/auth/gemini \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"refresh_token": "1//0abc..."}'
```

### Configuration

| Env Variable | Default | Description |
|---|---|---|
| `UM_DAEMON_GCP_PROJECT_ID` | — | GCP project for Secret Manager updates |

When `UM_DAEMON_GCP_PROJECT_ID` is set, submitting a new token also creates a new version of the `gemini-oauth-creds` secret in GCP Secret Manager. This ensures the token persists across Cloud Run deployments.

## Troubleshooting

**`"authenticated": false` in /api/query/models**
- Re-run `./scripts/extract-oauth-creds.sh` to refresh credentials
- Verify `secrets/gemini/oauth_creds.json` contains a `refresh_token`

**502 "Gemini API error"**
- Check container logs: `docker compose logs daemon`
- Verify your Google One AI Ultra subscription is active
- Try refreshing credentials: re-authenticate with `npx @google/gemini-cli auth`

**429 rate limit**
- Use `model: "auto"` to spread load across all models
- Reduce request frequency (sustained ~30 req/min per model)
- Rate limits reset every ~30 seconds

**503 "credentials not found"**
- Ensure `secrets/gemini/oauth_creds.json` is mounted in Docker
- For local dev: run `npx @google/gemini-cli auth` to create `~/.gemini/oauth_creds.json`

**504 timeout**
- Increase timeout in the request body: `"timeout": 600`
- Complex queries may take longer on first run
