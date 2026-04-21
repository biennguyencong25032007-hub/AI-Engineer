import os
from typing import List, Dict, Any


class SearchTool:
    """Tool for searching documents and data"""

    def __init__(self, data_dir: str = None):
        self.name = "search"
        self.description = "Search through documents and data"
        self.data_dir = data_dir or "./data"

    def __call__(self, query: str, search_type: str = "all") -> str:
        """Search for query in available data sources"""
        try:
            results = []

            # Search in documents
            doc_results = self._search_documents(query)
            results.extend(doc_results)

            # Search in memory/logs if available
            if search_type in ["all", "logs"]:
                log_results = self._search_logs(query)
                results.extend(log_results)

            if not results:
                return f"No results found for '{query}'"

            # Format results
            formatted = [f"Found {len(results)} result(s):"]
            for i, result in enumerate(results[:10], 1):
                formatted.append(f"{i}. {result}")

            return "\n".join(formatted)
        except Exception as e:
            return f"Search error: {str(e)}"

    def _search_documents(self, query: str) -> List[str]:
        """Search in document files"""
        results = []
        docs_dir = os.path.join(self.data_dir, "documents")

        if not os.path.exists(docs_dir):
            return results

        query_lower = query.lower()

        for filename in os.listdir(docs_dir):
            filepath = os.path.join(docs_dir, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if query_lower in content.lower():
                            # Extract relevant snippet
                            lines = content.split('\n')
                            for line in lines:
                                if query_lower in line.lower():
                                    results.append(f"{filename}: {line.strip()}")
                except Exception:
                    continue

        return results

    def _search_logs(self, query: str) -> List[str]:
        """Search in log files"""
        results = []
        logs_dir = os.path.join(self.data_dir, "logs")

        if not os.path.exists(logs_dir):
            return results

        query_lower = query.lower()

        for filename in os.listdir(logs_dir):
            if filename.endswith('.log'):
                filepath = os.path.join(logs_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if query_lower in line.lower():
                                results.append(f"{filename}: {line.strip()}")
                except Exception:
                    continue

        return results

    def search_file(self, filepath: str, query: str) -> List[str]:
        """Search in a specific file"""
        results = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if query.lower() in line.lower():
                        results.append(f"Line {i}: {line.strip()}")
        except Exception as e:
            return [f"Error: {str(e)}"]

        return results

    def fuzzy_search(self, query: str, directory: str = None) -> List[Dict[str, Any]]:
        """Perform fuzzy search in directory"""
        results = []
        search_dir = directory or os.path.join(self.data_dir, "documents")

        if not os.path.exists(search_dir):
            return results

        query_lower = query.lower()
        query_words = set(query_lower.split())

        for root, _, files in os.walk(search_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        # Count matching words
                        matches = sum(1 for word in query_words if word in content)
                        if matches > 0:
                            results.append({
                                'file': filepath,
                                'matches': matches,
                                'score': matches / len(query_words) if query_words else 0
                            })
                except Exception:
                    continue

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
