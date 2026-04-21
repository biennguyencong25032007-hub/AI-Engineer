from collections import deque
from typing import Dict, Any, Optional
from datetime import datetime


class ShortTermMemory:
    """Short-term memory using a sliding window approach"""

    def __init__(self, max_items: int = 10):
        self.max_items = max_items
        self.memory = deque(maxlen=max_items)
        self.conversation_history = []

    def add(self, item: Dict[str, Any]):
        """Add an item to short-term memory"""
        item['timestamp'] = datetime.now().isoformat()
        self.memory.append(item)
        self.conversation_history.append(item)

    def get_recent(self, n: int = None) -> list:
        """Get n most recent items"""
        if n is None:
            return list(self.memory)
        return list(self.memory)[-n:]

    def get_context(self) -> str:
        """Get formatted context for LLM"""
        context = []
        for item in self.get_recent():
            role = item.get('role', 'unknown')
            content = item.get('content', '')
            context.append(f"{role}: {content}")
        return "\n".join(context)

    def clear(self):
        """Clear all memory"""
        self.memory.clear()
        self.conversation_history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary"""
        return {
            'max_items': self.max_items,
            'memory': list(self.memory),
            'conversation_history': self.conversation_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShortTermMemory':
        """Deserialize memory from dictionary"""
        instance = cls(max_items=data.get('max_items', 10))
        instance.memory = deque(data.get('memory', []), maxlen=instance.max_items)
        instance.conversation_history = data.get('conversation_history', [])
        return instance
