#!/usr/bin/env bash
# Deploy um-agent-daemon to Google Cloud Run
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-um-agent-daemon}"

echo "Deploying ${SERVICE_NAME} to ${REGION}..."

gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --min-instances 1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --no-cpu-throttling \
  --set-env-vars "UM_DAEMON_DB_PATH=/app/data/daemon_tasks.db" \
  --update-secrets="/home/appuser/.codex/auth.json=codex-auth:latest" \
  --update-secrets="/home/appuser/.codex/config.toml=codex-config:latest" \
  --update-secrets="/home/appuser/.gemini/oauth_creds.json=gemini-oauth-creds:latest" \
  --update-secrets="/home/appuser/.gemini/settings.json=gemini-settings:latest"

echo ""
echo "Deployed! Service URL:"
gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format 'value(status.url)'
