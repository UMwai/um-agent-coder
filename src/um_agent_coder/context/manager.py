import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ContextType(Enum):
    FILE = "file"
    CODE = "code"
    CONVERSATION = "conversation"
    TOOL_RESULT = "tool_result"
    PROJECT_INFO = "project_info"


@dataclass
class ContextItem:
    content: str
    type: ContextType
    source: str  # file path, tool name, etc.
    tokens: int
    timestamp: float
    priority: int = 5  # 1-10, higher is more important
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ContextManager:
    """Manages context window for LLM interactions."""

    def __init__(self, max_tokens: int = 100000, reserve_tokens: int = 10000):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens  # Reserve for output
        self.available_tokens = max_tokens - reserve_tokens
        self.items: list[ContextItem] = []
        self.current_tokens = 0

    def add(
        self,
        content: str,
        type: ContextType,
        source: str,
        priority: int = 5,
        metadata: dict[str, Any] = None,
    ) -> bool:
        """
        Add a context item.

        Returns:
            True if added successfully, False if would exceed limit
        """
        # Estimate tokens (rough approximation)
        tokens = self._estimate_tokens(content)

        # Check if we need to make room
        if self.current_tokens + tokens > self.available_tokens:
            self._optimize_context(tokens)

        # Check again after optimization
        if self.current_tokens + tokens > self.available_tokens:
            return False

        item = ContextItem(
            content=content,
            type=type,
            source=source,
            tokens=tokens,
            timestamp=time.time(),
            priority=priority,
            metadata=metadata or {},
        )

        self.items.append(item)
        self.current_tokens += tokens
        return True

    def get_context(self) -> str:
        """Get the full context as a string."""
        context_parts = []

        # Group by type for better organization
        grouped = {}
        for item in self.items:
            if item.type not in grouped:
                grouped[item.type] = []
            grouped[item.type].append(item)

        # Build context string
        for context_type, items in grouped.items():
            if items:
                context_parts.append(f"\n## {context_type.value.upper()}\n")
                for item in items:
                    context_parts.append(f"### {item.source}\n{item.content}\n")

        return "\n".join(context_parts)

    def _optimize_context(self, needed_tokens: int):
        """
        Optimize context by removing low-priority or old items.
        """
        # Sort by priority (ascending) and timestamp (ascending)
        # Lower priority and older items will be removed first
        self.items.sort(key=lambda x: (x.priority, x.timestamp))

        while self.current_tokens + needed_tokens > self.available_tokens and self.items:
            removed = self.items.pop(0)
            self.current_tokens -= removed.tokens

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    def summarize_if_needed(self, llm=None) -> Optional[str]:
        """
        Summarize context if approaching limits.

        Args:
            llm: LLM instance to use for summarization

        Returns:
            Summary if created, None otherwise
        """
        usage_ratio = self.current_tokens / self.available_tokens

        if usage_ratio < 0.9 or not llm:
            return None

        # Group conversation items for summarization
        conversation_items = [item for item in self.items if item.type == ContextType.CONVERSATION]

        if not conversation_items:
            return None

        # Create summary prompt
        conversations = "\n".join([item.content for item in conversation_items])
        summary_prompt = f"Summarize the following conversation, preserving key technical details:\n\n{conversations}"

        try:
            summary = llm.chat(summary_prompt)

            # Remove old conversation items
            self.items = [item for item in self.items if item.type != ContextType.CONVERSATION]

            # Add summary as new item
            self.add(content=summary, type=ContextType.CONVERSATION, source="summary", priority=8)

            return summary

        except Exception as e:
            print(f"Failed to summarize: {e}")
            return None

    def get_usage(self) -> dict[str, Any]:
        """Get context usage statistics."""
        return {
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "available_tokens": self.available_tokens,
            "usage_percentage": (self.current_tokens / self.available_tokens) * 100,
            "item_count": len(self.items),
            "items_by_type": {
                context_type.value: len([item for item in self.items if item.type == context_type])
                for context_type in ContextType
            },
        }

    def clear(self):
        """Clear all context items."""
        self.items = []
        self.current_tokens = 0

    def remove_by_source(self, source: str):
        """Remove all items from a specific source."""
        self.items = [item for item in self.items if item.source != source]
        self._recalculate_tokens()

    def _recalculate_tokens(self):
        """Recalculate total token count."""
        self.current_tokens = sum(item.tokens for item in self.items)

    def export_state(self) -> list[dict[str, Any]]:
        """
        Export context state for checkpointing.

        Returns:
            List of serializable context item dictionaries
        """
        items = []
        for item in self.items:
            items.append(
                {
                    "content": item.content,
                    "type": item.type.value,
                    "source": item.source,
                    "tokens": item.tokens,
                    "timestamp": item.timestamp,
                    "priority": item.priority,
                    "metadata": item.metadata,
                }
            )
        return items

    def import_state(self, items: list[dict[str, Any]]):
        """
        Import context state from checkpoint.

        Args:
            items: List of context item dictionaries from export_state()
        """
        self.clear()

        for item_data in items:
            # Convert type string back to enum
            context_type = ContextType(item_data["type"])

            item = ContextItem(
                content=item_data["content"],
                type=context_type,
                source=item_data["source"],
                tokens=item_data["tokens"],
                timestamp=item_data["timestamp"],
                priority=item_data["priority"],
                metadata=item_data.get("metadata", {}),
            )

            self.items.append(item)

        self._recalculate_tokens()
