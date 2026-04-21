from typing import Dict, Any
from datetime import datetime, timedelta


class LongTermMemory:
    """Long-term memory for storing persistent knowledge and insights"""

    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.memory = []
        self.knowledge_base = {}

    def add(self, item: Dict[str, Any]):
        """Add an item to long-term memory"""
        item['timestamp'] = datetime.now().isoformat()
        item['id'] = len(self.memory)
        self.memory.append(item)

        # Index by type if available
        item_type = item.get('type', 'general')
        if item_type not in self.knowledge_base:
            self.knowledge_base[item_type] = []
        self.knowledge_base[item_type].append(item)

    def get_by_type(self, item_type: str) -> list:
        """Retrieve items by type"""
        return self.knowledge_base.get(item_type, [])

    def get_recent(self, days: int = None) -> list:
        """Get items from last N days"""
        if days is None:
            days = self.retention_days

        cutoff = datetime.now() - timedelta(days=days)
        recent = []
        for item in self.memory:
            item_time = datetime.fromisoformat(item['timestamp'])
            if item_time >= cutoff:
                recent.append(item)
        return recent

    def search(self, query: str) -> list:
        """Simple keyword search in memory"""
        results = []
        query_lower = query.lower()
        for item in self.memory:
            content = str(item.get('content', '')).lower()
            if query_lower in content:
                results.append(item)
        return results

    def clear_old(self):
        """Remove items older than retention period"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        self.memory = [
            item for item in self.memory
            if datetime.fromisoformat(item['timestamp']) >= cutoff
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'retention_days': self.retention_days,
            'memory': self.memory,
            'knowledge_base': self.knowledge_base
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LongTermMemory':
        """Deserialize from dictionary"""
        instance = cls(retention_days=data.get('retention_days', 30))
        instance.memory = data.get('memory', [])
        instance.knowledge_base = data.get('knowledge_base', {})
        return instance
