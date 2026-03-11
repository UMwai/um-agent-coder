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
  --min-instances 0 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --no-cpu-throttling \
  --set-env-vars "^##^UM_DAEMON_DB_PATH=/app/data/daemon_tasks.db##UM_DAEMON_GCP_PROJECT_ID=${PROJECT_ID}##UM_DAEMON_GEMINI_FIRESTORE_ENABLED=true##UM_DAEMON_WORLD_AGENT_ENABLED=true##UM_DAEMON_WORLD_AGENT_GITHUB_REPOS=UMwai/um-agent-coder,UMwai/um_ai-hedge-fund" \
  --update-secrets="/home/appuser/.gemini/oauth_creds.json=gemini-oauth-creds:latest,UM_DAEMON_GITHUB_TOKEN=github-pat:latest"

# Grant service account permission to add secret versions
SA_EMAIL=$(gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format 'value(spec.template.spec.serviceAccountName)')
if [ -n "${SA_EMAIL}" ]; then
  echo "Granting secretVersionAdder on gemini-oauth-creds to ${SA_EMAIL}..."
  gcloud secrets add-iam-policy-binding gemini-oauth-creds \
    --project "${PROJECT_ID}" \
    --member "serviceAccount:${SA_EMAIL}" \
    --role "roles/secretmanager.secretVersionAdder" \
    --quiet 2>/dev/null || echo "  (IAM binding may already exist)"
fi

echo ""
echo "Deployed! Service URL:"
gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format 'value(status.url)'
