"""Gemini OAuth re-authentication endpoints.

Provides a web form for pasting a new Gemini CLI refresh token when the
existing token gets revoked. Updates GCP Secret Manager and hot-reloads
the Gemini client singleton.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from um_agent_coder.daemon.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth/gemini", tags=["auth"])

# Gemini CLI OAuth app credentials (public, embedded in CLI source)
_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"


# --- Models ---


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10, max_length=2048)


class AuthStatusResponse(BaseModel):
    healthy: bool
    tier: Optional[str] = None
    error: Optional[str] = None


class ReauthResponse(BaseModel):
    success: bool
    tier: Optional[str] = None
    secret_updated: bool = False
    error: Optional[str] = None


# --- HTML Form ---

_FORM_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemini Re-Authentication</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 20px; color: #333; }
  h1 { font-size: 1.4rem; }
  .steps { background: #f5f5f5; padding: 16px 20px; border-radius: 8px; margin: 16px 0; }
  .steps ol { margin: 0; padding-left: 20px; }
  .steps li { margin: 8px 0; }
  code { background: #e8e8e8; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
  pre { background: #1e1e1e; color: #d4d4d4; padding: 12px 16px; border-radius: 8px; overflow-x: auto; font-size: 0.85em; }
  textarea { width: 100%; height: 80px; font-family: monospace; font-size: 0.9em; padding: 10px; border: 2px solid #ccc; border-radius: 6px; resize: vertical; }
  textarea:focus { border-color: #4285f4; outline: none; }
  button { background: #4285f4; color: white; border: none; padding: 10px 24px; border-radius: 6px; font-size: 1em; cursor: pointer; margin-top: 8px; }
  button:hover { background: #3367d6; }
  .result { margin-top: 16px; padding: 12px 16px; border-radius: 6px; display: none; }
  .result.ok { background: #e8f5e9; color: #2e7d32; display: block; }
  .result.err { background: #ffebee; color: #c62828; display: block; }
  .status { margin-top: 20px; padding: 12px; background: #e3f2fd; border-radius: 6px; }
  .status.unhealthy { background: #fff3e0; }
</style>
</head>
<body>
<h1>Gemini Re-Authentication</h1>

<div id="status-box" class="status" style="display:none"></div>

<div class="steps">
<ol>
  <li>Run locally: <code>npx @google/gemini-cli auth</code></li>
  <li>Extract your refresh token:
    <pre>python3 -c "import json; print(json.load(open('$HOME/.gemini/oauth_creds.json'))['refresh_token'])"</pre>
  </li>
  <li>Paste the refresh token below and submit.</li>
</ol>
</div>

<form id="reauth-form">
  <label for="token"><strong>Refresh Token</strong></label><br>
  <textarea id="token" name="refresh_token" placeholder="1//0abc..." required></textarea><br>
  <button type="submit">Update Token</button>
</form>

<div id="result" class="result"></div>

<script>
// Check auth status on load
fetch(window.location.pathname + '/status', {
  headers: window._apiKey ? {'X-API-Key': window._apiKey} : {}
}).then(r => r.json()).then(data => {
  const box = document.getElementById('status-box');
  box.style.display = 'block';
  if (data.healthy) {
    box.textContent = '\\u2705 Auth is healthy (tier: ' + (data.tier || 'unknown') + ')';
  } else {
    box.className = 'status unhealthy';
    box.textContent = '\\u26a0\\ufe0f Auth is unhealthy: ' + (data.error || 'unknown error');
  }
}).catch(() => {});

document.getElementById('reauth-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const token = document.getElementById('token').value.trim();
  if (!token) return;

  const result = document.getElementById('result');
  result.className = 'result';
  result.style.display = 'none';

  try {
    const headers = {'Content-Type': 'application/json'};
    // Forward API key from query param if present
    const params = new URLSearchParams(window.location.search);
    if (params.get('key')) headers['X-API-Key'] = params.get('key');

    const resp = await fetch(window.location.pathname, {
      method: 'POST',
      headers,
      body: JSON.stringify({refresh_token: token}),
    });
    const data = await resp.json();
    if (data.success) {
      result.className = 'result ok';
      result.textContent = '\\u2705 Token updated successfully!' +
        (data.tier ? ' Tier: ' + data.tier : '') +
        (data.secret_updated ? ' (GCP secret updated)' : '');
    } else {
      result.className = 'result err';
      result.textContent = '\\u274c Failed: ' + (data.error || 'Unknown error');
    }
  } catch (err) {
    result.className = 'result err';
    result.textContent = '\\u274c Request failed: ' + err.message;
  }
});
</script>
</body>
</html>
"""


# --- Helpers ---


async def _validate_refresh_token(refresh_token: str) -> tuple[bool, str]:
    """Validate a refresh token by attempting a token refresh.

    Returns (success, error_message).
    """
    from um_agent_coder.daemon.gemini_client import _load_cli_oauth_app

    try:
        client_id, client_secret = _load_cli_oauth_app()
    except RuntimeError as e:
        return False, f"Cannot load OAuth app credentials: {e}"

    async with httpx.AsyncClient(timeout=15.0) as http:
        try:
            resp = await http.post(
                _OAUTH_TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            if resp.status_code != 200:
                detail = resp.json().get("error_description", resp.text)
                return False, f"Token refresh failed ({resp.status_code}): {detail}"
            return True, ""
        except Exception as e:
            return False, f"Token refresh request failed: {e}"


async def _update_gcp_secret(refresh_token: str) -> tuple[bool, str]:
    """Update the gemini-oauth-creds secret in GCP Secret Manager.

    Returns (success, error_message). Returns (False, "") if GCP project not configured.
    """
    from um_agent_coder.daemon.app import get_settings

    settings = get_settings()
    if not settings.gcp_project_id:
        return False, ""

    try:
        from google.cloud import secretmanager  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("google-cloud-secret-manager not installed, skipping secret update")
        return False, ""

    import json

    secret_name = f"projects/{settings.gcp_project_id}/secrets/gemini-oauth-creds"

    try:
        client = secretmanager.SecretManagerServiceClient()

        # Read existing secret to preserve client_id / client_secret
        try:
            existing = client.access_secret_version(
                request={"name": f"{secret_name}/versions/latest"}
            )
            creds = json.loads(existing.payload.data.decode())
        except Exception:
            creds = {}

        creds["refresh_token"] = refresh_token

        # Add new version
        client.add_secret_version(
            request={
                "parent": secret_name,
                "payload": {"data": json.dumps(creds).encode()},
            }
        )
        logger.info("Updated GCP secret %s with new refresh token", secret_name)
        return True, ""
    except Exception as e:
        logger.error("Failed to update GCP secret: %s", e)
        return False, str(e)


# --- Endpoints ---


@router.get("", response_class=HTMLResponse)
async def reauth_form(
    _key: Optional[str] = Depends(verify_api_key),
):
    """Serve the re-authentication HTML form."""
    return HTMLResponse(_FORM_HTML)


@router.post("", response_model=ReauthResponse)
async def reauth_submit(
    req: RefreshTokenRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Accept a new refresh token, validate it, update secret, and hot-reload client."""
    # 1. Validate the token
    valid, err = await _validate_refresh_token(req.refresh_token)
    if not valid:
        return ReauthResponse(success=False, error=err)

    # 2. Update GCP Secret Manager (if configured)
    secret_updated, secret_err = await _update_gcp_secret(req.refresh_token)
    if secret_err:
        logger.warning("GCP secret update failed (continuing anyway): %s", secret_err)

    # 3. Hot-reload the Gemini client
    from um_agent_coder.daemon.app import reset_gemini_client

    try:
        tier = await reset_gemini_client(req.refresh_token)
    except Exception as e:
        return ReauthResponse(
            success=False,
            secret_updated=secret_updated,
            error=f"Client reload failed: {e}",
        )

    logger.info("Gemini re-auth successful (tier=%s, secret_updated=%s)", tier, secret_updated)
    return ReauthResponse(success=True, tier=tier, secret_updated=secret_updated)


@router.get("/status", response_model=AuthStatusResponse)
async def auth_status(
    _key: Optional[str] = Depends(verify_api_key),
):
    """Check current Gemini auth health."""
    from um_agent_coder.daemon.app import get_gemini_client

    try:
        client = get_gemini_client()
    except Exception as e:
        return AuthStatusResponse(healthy=False, error=str(e))

    try:
        await client.refresh_access_token()
        return AuthStatusResponse(healthy=True, tier=client.tier)
    except Exception as e:
        return AuthStatusResponse(healthy=False, tier=client.tier, error=str(e))
