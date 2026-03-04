"""
um-agent-coder daemon: 24/7 service for receiving and processing tasks
via GitHub webhooks, Slack, Discord, and a web dashboard.
"""

from um_agent_coder.daemon.config import DaemonSettings

__all__ = ["DaemonSettings"]
