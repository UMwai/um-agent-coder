#!/bin/bash
# Quick test script to verify Codex MCP server setup

echo "=========================================="
echo "Codex MCP Server Test for Gemini CLI"
echo "=========================================="
echo ""

# Check if wrapper script exists
echo "1. Checking wrapper script..."
if [ -x "/Users/waiyang/bin/codex-mcp-wrapper-fixed" ]; then
    echo "   ✅ Wrapper script exists and is executable"
else
    echo "   ❌ Wrapper script not found or not executable"
    exit 1
fi

# Check if codex is installed
echo ""
echo "2. Checking Codex installation..."
if command -v codex &> /dev/null; then
    CODEX_VERSION=$(codex --version 2>&1 | head -1)
    echo "   ✅ Codex found: $CODEX_VERSION"
else
    echo "   ❌ Codex not found in PATH"
    exit 1
fi

# Check API keys
echo ""
echo "3. Checking API keys..."
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "   ✅ ANTHROPIC_API_KEY is set (${ANTHROPIC_API_KEY:0:10}...)"
else
    echo "   ⚠️  ANTHROPIC_API_KEY not set in environment"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo "   ✅ OPENAI_API_KEY is set (${OPENAI_API_KEY:0:10}...)"
else
    echo "   ⚠️  OPENAI_API_KEY not set in environment"
fi

# Test wrapper script syntax
echo ""
echo "4. Testing Codex MCP server startup..."
echo "   Command: /Users/waiyang/bin/codex-mcp-wrapper-fixed -m claude-sonnet-4-5-20250929 mcp-server -c approval_policy=never"
echo ""

# Run the server for 2 seconds and capture output
TEMP_OUTPUT=$(mktemp)
(
    /Users/waiyang/bin/codex-mcp-wrapper-fixed -m claude-sonnet-4-5-20250929 mcp-server \
        -c approval_policy=never \
        -c sandbox=danger-full-access \
        -c features.rollout_recorder=false &
    PID=$!
    sleep 2
    kill $PID 2>/dev/null
    wait $PID 2>/dev/null
) 2>&1 | tee "$TEMP_OUTPUT"

# Check for errors
if grep -q "error:" "$TEMP_OUTPUT"; then
    echo ""
    echo "   ❌ Server startup failed with errors"
    rm "$TEMP_OUTPUT"
    exit 1
else
    echo ""
    echo "   ✅ Server started successfully (no errors detected)"
fi

rm "$TEMP_OUTPUT"

# Check Gemini CLI settings
echo ""
echo "5. Checking Gemini CLI settings..."
SETTINGS_FILE="$HOME/.gemini/settings.json"
if [ -f "$SETTINGS_FILE" ]; then
    echo "   ✅ Gemini CLI settings found at: $SETTINGS_FILE"

    if grep -q "codex-mcp-wrapper-fixed" "$SETTINGS_FILE"; then
        echo "   ✅ Codex MCP server configured in Gemini CLI"

        # Show the configuration
        echo ""
        echo "   Configuration:"
        cat "$SETTINGS_FILE" | jq '.mcpServers.codex' 2>/dev/null || \
            grep -A 15 '"codex"' "$SETTINGS_FILE"
    else
        echo "   ❌ Codex MCP server not found in settings"
    fi
else
    echo "   ❌ Gemini CLI settings not found"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Make sure your API keys are exported in your shell:"
echo "   export ANTHROPIC_API_KEY='your-key'"
echo "   export OPENAI_API_KEY='your-key'"
echo ""
echo "2. Launch Gemini CLI:"
echo "   gemini"
echo ""
echo "3. The Codex MCP server should load automatically"
echo "   Look for: ✓ codex (codex MCP Server)"
echo ""
echo "4. Try using a Codex tool to verify it's working"
echo ""
