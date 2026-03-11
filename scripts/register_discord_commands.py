#!/usr/bin/env python3
"""Register /ask and /iterate slash commands with Discord.

Usage:
    python scripts/register_discord_commands.py

Requires env vars:
    DISCORD_BOT_TOKEN       - Bot token from Discord Developer Portal
    DISCORD_APPLICATION_ID  - Application ID from Discord Developer Portal

Optional:
    DISCORD_GUILD_ID        - Register to a specific guild (instant).
                              Without this, registers globally (takes ~1 hour to propagate).
"""

import os
import sys

import httpx

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN") or os.environ.get("UM_DAEMON_DISCORD_BOT_TOKEN")
APP_ID = os.environ.get("DISCORD_APPLICATION_ID") or os.environ.get("UM_DAEMON_DISCORD_APPLICATION_ID")
GUILD_ID = os.environ.get("DISCORD_GUILD_ID")

COMMANDS = [
    {
        "name": "ask",
        "description": "Ask a question — get a quick Gemini response",
        "options": [
            {
                "name": "prompt",
                "description": "Your question",
                "type": 3,  # STRING
                "required": True,
            }
        ],
    },
    {
        "name": "iterate",
        "description": "Run the full iteration loop (generate → evaluate → retry) until quality threshold met",
        "options": [
            {
                "name": "prompt",
                "description": "Task description",
                "type": 3,  # STRING
                "required": True,
            }
        ],
    },
]


def main():
    if not BOT_TOKEN:
        print("Error: Set DISCORD_BOT_TOKEN or UM_DAEMON_DISCORD_BOT_TOKEN")
        sys.exit(1)
    if not APP_ID:
        print("Error: Set DISCORD_APPLICATION_ID or UM_DAEMON_DISCORD_APPLICATION_ID")
        sys.exit(1)

    if GUILD_ID:
        url = f"https://discord.com/api/v10/applications/{APP_ID}/guilds/{GUILD_ID}/commands"
        scope = f"guild {GUILD_ID}"
    else:
        url = f"https://discord.com/api/v10/applications/{APP_ID}/commands"
        scope = "global"

    headers = {"Authorization": f"Bot {BOT_TOKEN}"}

    print(f"Registering {len(COMMANDS)} commands ({scope})...")

    with httpx.Client(timeout=30) as client:
        # Bulk overwrite all commands
        resp = client.put(url, json=COMMANDS, headers=headers)

        if resp.status_code == 200:
            registered = resp.json()
            for cmd in registered:
                print(f"  /{cmd['name']} — registered (id={cmd['id']})")
            print(f"\nDone. {len(registered)} commands registered.")
            if not GUILD_ID:
                print("Note: Global commands take up to 1 hour to appear. "
                      "Set DISCORD_GUILD_ID for instant registration to a test server.")
        else:
            print(f"Error {resp.status_code}: {resp.text}")
            sys.exit(1)


if __name__ == "__main__":
    main()
