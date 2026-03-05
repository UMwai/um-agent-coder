# Query Proxy Setup Guide

Route AI queries through your existing Codex and Gemini CLI subscriptions instead of paying per-token API charges. The proxy runs both CLIs inside a Docker container with your OAuth credentials mounted in.

## How It Works

```
Client → POST /api/query → FastAPI → subprocess(codex/gemini CLI) → Response
                                          ↑
                              OAuth creds mounted from host
                              CLIs handle token refresh automatically
```

The CLIs authenticate using OAuth refresh tokens and bill queries through your subscription. You don't need API keys — the CLIs manage token refresh internally.

## Prerequisites

- Docker and Docker Compose
- Active subscriptions:
  - **OpenAI**: ChatGPT Plus/Pro/Team with Codex access
  - **Google**: Gemini Advanced or Google One AI Premium
- CLIs authenticated on your local machine:
  - `npx @openai/codex auth` (creates `~/.codex/auth.json`)
  - `npx @google/gemini-cli auth` (creates `~/.gemini/oauth_creds.json`)

## Setup

### 1. Extract credentials

```bash
./scripts/extract-oauth-creds.sh
```

This copies your local OAuth tokens into `secrets/` (gitignored):

```
secrets/
├── codex/
│   ├── auth.json        # OAuth tokens (access + refresh)
│   └── config.toml      # Model settings
└── gemini/
    ├── oauth_creds.json  # OAuth tokens
    └── settings.json     # Model settings
```

### 2. Build and run

```bash
docker compose up --build
```

### 3. Verify providers

```bash
curl http://localhost:8080/api/query/providers
```

Expected output:
```json
{
  "providers": [
    {"name": "codex", "available": true, "default_model": "gpt-5.2", "authenticated": true},
    {"name": "gemini", "available": true, "default_model": "gemini-2.5-pro", "authenticated": true}
  ]
}
```

### 4. Send queries

```bash
# Gemini query
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain TCP/IP in one paragraph", "provider": "gemini"}'

# Codex query
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python fibonacci function", "provider": "codex"}'

# With model override
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "provider": "gemini", "model": "gemini-2.5-flash"}'
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
  -d '{"prompt": "Hello", "provider": "gemini"}'
```

## Cloud Run Deployment

### 1. Store credentials as GCP secrets

```bash
gcloud secrets create codex-auth --data-file=secrets/codex/auth.json
gcloud secrets create codex-config --data-file=secrets/codex/config.toml
gcloud secrets create gemini-oauth-creds --data-file=secrets/gemini/oauth_creds.json
gcloud secrets create gemini-settings --data-file=secrets/gemini/settings.json
```

### 2. Deploy

```bash
GCP_PROJECT_ID=your-project ./deploy/deploy.sh
```

The deploy script mounts secrets as files at the paths the CLIs expect (`/home/appuser/.codex/`, `/home/appuser/.gemini/`).

### 3. Set API key

```bash
gcloud run services update um-agent-daemon \
  --set-env-vars "UM_DAEMON_API_KEY=your-production-key"
```

## Token Refresh

The CLIs handle OAuth token refresh automatically:
- **Codex**: Refreshes when `auth.json` access_token expires, writes updated tokens back
- **Gemini**: Refreshes when `oauth_creds.json` access_token expires

The credential files are mounted read-only in Docker. If a CLI needs to write refreshed tokens, mount read-write instead:

```yaml
# docker-compose.override.yml
services:
  daemon:
    volumes:
      - ./secrets/codex:/home/appuser/.codex    # Remove :ro
      - ./secrets/gemini:/home/appuser/.gemini  # Remove :ro
```

On Cloud Run, the CLI will use the initial token and refresh via the OAuth provider's token endpoint directly in memory — the secret file itself isn't rewritten.

## Configuration Reference

| Env Variable | Default | Description |
|--------------|---------|-------------|
| `UM_DAEMON_CODEX_MODEL` | gpt-5.2 | Default model for Codex queries |
| `UM_DAEMON_GEMINI_MODEL` | gemini-2.5-pro | Default model for Gemini queries |
| `UM_DAEMON_QUERY_RATE_LIMIT` | 60 | Max requests/min per provider |
| `UM_DAEMON_API_KEY` | — | API key (disabled if unset) |

## Troubleshooting

**Provider shows `"authenticated": false`**
- Re-run `./scripts/extract-oauth-creds.sh` to refresh credentials
- Verify the credential files contain `refresh_token`

**502 error from CLI**
- Check container logs: `docker compose logs daemon`
- The CLI may need a read-write mount to refresh tokens (see Token Refresh section)

**503 "CLI not installed"**
- Rebuild the image: `docker compose build --no-cache`
- Verify CLIs are installed: `docker compose exec daemon which codex gemini`

**504 timeout**
- Increase timeout in the request body: `"timeout": 600`
- Complex queries may take longer on first run (model cold start)
