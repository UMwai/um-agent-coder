#!/usr/bin/env bash
# Extract OAuth credentials from local machine into secrets/ for Docker mounting.
# Usage: ./scripts/extract-oauth-creds.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SECRETS_DIR="$PROJECT_DIR/secrets"

echo "Extracting OAuth credentials to $SECRETS_DIR ..."

# Create secrets directories
mkdir -p "$SECRETS_DIR/codex" "$SECRETS_DIR/gemini"

# --- Codex credentials ---
CODEX_AUTH="$HOME/.codex/auth.json"
CODEX_CONFIG="$HOME/.codex/config.toml"

if [[ -f "$CODEX_AUTH" ]]; then
    cp "$CODEX_AUTH" "$SECRETS_DIR/codex/auth.json"
    # Validate refresh token exists
    if python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d.get('tokens',{}).get('refresh_token')" "$CODEX_AUTH" 2>/dev/null; then
        echo "  [OK] codex/auth.json (refresh_token present)"
    else
        echo "  [WARN] codex/auth.json copied but no refresh_token found"
    fi
else
    echo "  [SKIP] $CODEX_AUTH not found"
fi

if [[ -f "$CODEX_CONFIG" ]]; then
    cp "$CODEX_CONFIG" "$SECRETS_DIR/codex/config.toml"
    echo "  [OK] codex/config.toml"
else
    echo "  [SKIP] $CODEX_CONFIG not found"
fi

# --- Gemini credentials ---
GEMINI_CREDS="$HOME/.gemini/oauth_creds.json"
GEMINI_SETTINGS="$HOME/.gemini/settings.json"

if [[ -f "$GEMINI_CREDS" ]]; then
    cp "$GEMINI_CREDS" "$SECRETS_DIR/gemini/oauth_creds.json"
    if python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d.get('refresh_token')" "$GEMINI_CREDS" 2>/dev/null; then
        echo "  [OK] gemini/oauth_creds.json (refresh_token present)"
    else
        echo "  [WARN] gemini/oauth_creds.json copied but no refresh_token found"
    fi
else
    echo "  [SKIP] $GEMINI_CREDS not found"
fi

if [[ -f "$GEMINI_SETTINGS" ]]; then
    cp "$GEMINI_SETTINGS" "$SECRETS_DIR/gemini/settings.json"
    echo "  [OK] gemini/settings.json"
else
    echo "  [SKIP] $GEMINI_SETTINGS not found"
fi

echo ""
echo "Done. Secrets stored in: $SECRETS_DIR"
echo "These files are gitignored. Do NOT commit them."
