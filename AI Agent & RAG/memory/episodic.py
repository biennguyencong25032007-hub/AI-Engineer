from typing import Dict, Any, List
from datetime import datetime


class EpisodicMemory:
    """Episodic memory for storing specific events, experiences, and interactions"""

    def __init__(self):
        self.episodes = []
        self.session_episodes = []

    def add_episode(self, episode: Dict[str, Any]):
        """Add an episode to memory"""
        episode['timestamp'] = datetime.now().isoformat()
        episode['id'] = len(self.episodes)
        self.episodes.append(episode)
        self.session_episodes.append(episode)

    def get_episodes_by_type(self, episode_type: str) -> List[Dict[str, Any]]:
        """Get all episodes of a specific type"""
        return [ep for ep in self.episodes if ep.get('type') == episode_type]

    def get_recent_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent episodes"""
        return self.episodes[-n:] if self.episodes else []

    def get_session_episodes(self) -> List[Dict[str, Any]]:
        """Get all episodes from current session"""
        return self.session_episodes

    def clear_session(self):
        """Clear current session episodes"""
        self.session_episodes.clear()

    def search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search episodes by keywords"""
        results = []
        for episode in self.episodes:
            content = str(episode.get('content', '')).lower()
            description = str(episode.get('description', '')).lower()

            # Check if all keywords are present
            if all(kw.lower() in (content + description) for kw in keywords):
                results.append(episode)
        return results

    def get_episodes_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get episodes within a date range"""
        results = []
        for episode in self.episodes:
            ep_time = datetime.fromisoformat(episode['timestamp'])
            if start_date <= ep_time <= end_date:
                results.append(episode)
        return results

    def summarize_episode(self, episode: Dict[str, Any]) -> str:
        """Create a summary of an episode"""
        ep_type = episode.get('type', 'unknown')
        content = episode.get('content', '')
        timestamp = episode.get('timestamp', '')

        if len(content) > 100:
            content = content[:97] + "..."

        return f"[{ep_type}] {content} ({timestamp})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'episodes': self.episodes,
            'session_episodes': self.session_episodes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """Deserialize from dictionary"""
        instance = cls()
        instance.episodes = data.get('episodes', [])
        instance.session_episodes = data.get('session_episodes', [])
        return instance
