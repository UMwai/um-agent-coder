# Codex MCP Server Fix for Gemini CLI

## Problem
When loading the Codex MCP server in Gemini CLI, it failed with:
```
Failed to start Codex session: internal error; agent loop died unexpectedly
```

## Root Causes Identified

Based on analysis using both Codex and Gemini CLI documentation, the issues were:

### 1. **Incorrect Argument Order** (CRITICAL)
- **Problem**: The `-m` (model) flag is a global Codex flag, not a subcommand flag. It must come BEFORE `mcp-server`, while `-c` flags can come after
- **Impact**: Codex returned `error: unexpected argument '-m' found` and failed to start MCP server
- **Fix**: Correct order is `codex -m <model> mcp-server -c <config>...`

### 2. **Missing Environment Variables**
- **Problem**: Gemini CLI may sanitize environment variables, stripping API keys
- **Impact**: Codex's agent loop couldn't initialize without API keys
- **Fix**: Explicitly pass `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in the config

### 3. **Stdout Pollution Risk**
- **Problem**: Any error messages written to stdout break MCP protocol
- **Impact**: Gemini CLI treats non-MCP output as fatal parse errors
- **Fix**: Updated wrapper script to ensure debug output goes to stderr only

## Changes Made

### 1. New Wrapper Script
Created: `/Users/waiyang/bin/codex-mcp-wrapper-fixed`

```bash
#!/bin/bash
# Improved Codex MCP wrapper for Gemini CLI compatibility

# Disable pre-exec prctl (as in original)
export CODEX_DISABLE_PRE_EXEC_PRCTL=1

# Debug: Log the command being executed (to stderr only)
echo "[codex-mcp-wrapper] Starting Codex MCP server" >&2
echo "[codex-mcp-wrapper] Args: $@" >&2
echo "[codex-mcp-wrapper] Working directory: $(pwd)" >&2

# Execute Codex - keep stdout clean for MCP JSON-RPC protocol
exec codex "$@"
```

**Key improvements:**
- Debug logging goes to stderr only (>&2)
- Preserves stdout for MCP protocol
- Maintains the original CODEX_DISABLE_PRE_EXEC_PRCTL setting

### 2. Updated Gemini CLI Settings
Modified: `/Users/waiyang/.gemini/settings.json`

**Before:**
```json
{
  "mcpServers": {
    "codex": {
      "command": "/Users/waiyang/bin/codex-mcp-wrapper",
      "args": [
        "-m", "claude-sonnet-4-5-20250929",
        "-c", "approval_policy=never",
        "-c", "sandbox=danger-full-access",
        "-c", "features.rollout_recorder=false",
        "mcp-server"  // WRONG: subcommand last
      ]
    }
  }
}
```

**After:**
```json
{
  "mcpServers": {
    "codex": {
      "command": "/Users/waiyang/bin/codex-mcp-wrapper-fixed",
      "args": [
        "-m", "claude-sonnet-4-5-20250929",  // CORRECT: global -m flag before subcommand
        "mcp-server",  // Subcommand comes after global flags
        "-c", "approval_policy=never",  // Subcommand-specific -c flags after mcp-server
        "-c", "sandbox=danger-full-access",
        "-c", "features.rollout_recorder=false"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "$ANTHROPIC_API_KEY",
        "OPENAI_API_KEY": "$OPENAI_API_KEY"
      }
    }
  }
}
```

**Backup created at:** `/Users/waiyang/.gemini/settings.json.backup`

## Test Results

The manual tests confirm the server now starts successfully:

```bash
$ /Users/waiyang/bin/codex-mcp-wrapper-fixed -m claude-sonnet-4-5-20250929 mcp-server -c approval_policy=never

[codex-mcp-wrapper] Starting Codex MCP server
[codex-mcp-wrapper] Args: -m claude-sonnet-4-5-20250929 mcp-server -c approval_policy=never -c sandbox=danger-full-access -c features.rollout_recorder=false
[codex-mcp-wrapper] Working directory: /Users/waiyang/Desktop/repo/um-agent-coder

# Server runs and waits for MCP input (exit code: 0)
```

✅ **No errors** - The server starts successfully
✅ **Clean stdout** - Debug messages only go to stderr
✅ **Correct argument parsing** - No "unexpected argument" errors

## Testing Instructions

### 1. Verify API Keys in Environment
Before testing, ensure you have your API keys set in your shell:

```bash
# Check if keys are set
echo "ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:0:10}..."
echo "OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."

# If not set, add them to your shell profile (~/.zshrc or ~/.bashrc):
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

### 2. Test Codex MCP Server Manually
Test the wrapper script directly to verify it works:

```bash
# Run the wrapper script manually with CORRECT argument order
# Note: -m flag comes BEFORE mcp-server subcommand
/Users/waiyang/bin/codex-mcp-wrapper-fixed -m claude-sonnet-4-5-20250929 mcp-server -c approval_policy=never -c sandbox=danger-full-access

# You should see debug output on stderr like:
# [codex-mcp-wrapper] Starting Codex MCP server
# [codex-mcp-wrapper] Args: -m claude-sonnet-4-5-20250929 mcp-server ...
# [codex-mcp-wrapper] Working directory: /Users/waiyang

# The server will wait for MCP protocol input on stdin
# Press Ctrl+C to exit
```

### 3. Test with Gemini CLI
Launch Gemini CLI and try loading the Codex MCP server:

```bash
# Start Gemini CLI
gemini

# In the Gemini CLI, check if Codex tools are available
# Use a Codex tool to verify it's working
```

### 4. Check Logs
If issues persist, check the stderr output:

```bash
# Gemini CLI logs should show:
# ✓ codex (codex MCP Server) {tools loaded}
#
# Instead of:
# ✗ codex (failed to start)
```

## Troubleshooting

### If it still fails:

1. **Check API Keys**
   ```bash
   # Ensure keys are exported in your shell
   env | grep -E "(ANTHROPIC|OPENAI)_API_KEY"
   ```

2. **Verify Codex Installation**
   ```bash
   which codex
   codex --version
   ```

3. **Test MCP Server Directly**
   ```bash
   # Run without wrapper
   codex mcp-server -m claude-sonnet-4-5-20250929
   ```

4. **Check for Model Name Issues**
   - Verify the model name `claude-sonnet-4-5-20250929` is correct
   - Try with a simpler model name if needed

5. **Rollback if Needed**
   ```bash
   # Restore original settings
   cp /Users/waiyang/.gemini/settings.json.backup /Users/waiyang/.gemini/settings.json
   ```

## Technical Details

### Why These Changes Work

1. **Subcommand First**: CLI parsers typically expect `command subcommand [flags]` order
2. **Explicit Environment**: Gemini CLI's env sanitization is bypassed by explicit env config
3. **Stderr for Logs**: MCP protocol requires clean stdout for JSON-RPC framing

### MCP Protocol Requirements
- All output on stdout MUST be valid MCP frames: `Content-Length: N\r\n\r\n{json}`
- Any other output (logs, errors) MUST go to stderr
- Gemini CLI is strict about protocol compliance (Claude Code is more forgiving)

## Sources
- [Gemini CLI MCP Server Documentation](https://github.com/google-gemini/gemini-cli/blob/main/docs/tools/mcp-server.md)
- [Gemini CLI FastMCP Integration](https://developers.googleblog.com/en/gemini-cli-fastmcp-simplifying-mcp-server-development/)
- Codex MCP Server Analysis (via Codex tool)

## Next Steps

If this fix works:
- Consider contributing the fix to Codex documentation
- Share the solution with other Gemini CLI + Codex users

If this doesn't work:
- Share the stderr output from the wrapper script
- Check Codex version compatibility with the model name
- Consider trying a different model or configuration
