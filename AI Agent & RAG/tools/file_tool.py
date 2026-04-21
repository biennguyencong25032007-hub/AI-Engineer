import os
import json
from datetime import datetime
from typing import Dict, Any, List


class FileTool:
    """Tool for file operations"""

    def __init__(self, base_dir: str = "./data"):
        self.name = "file"
        self.description = "Read, write, and manage files"
        self.base_dir = base_dir

    def __call__(self, operation: str, **kwargs) -> str:
        """Execute file operation"""
        try:
            operation = operation.lower()

            if operation == "read":
                return self.read_file(kwargs.get('path', ''))
            elif operation == "write":
                return self.write_file(
                    kwargs.get('path', ''),
                    kwargs.get('content', '')
                )
            elif operation == "list":
                return self.list_files(kwargs.get('directory', '.'))
            elif operation == "delete":
                return self.delete_file(kwargs.get('path', ''))
            elif operation == "exists":
                return str(self.file_exists(kwargs.get('path', '')))
            elif operation == "info":
                return self.get_file_info(kwargs.get('path', ''))
            else:
                return f"Unknown operation: {operation}"

        except Exception as e:
            return f"File operation error: {str(e)}"

    def read_file(self, filepath: str) -> str:
        """Read file content"""
        full_path = self._resolve_path(filepath)

        if not os.path.exists(full_path):
            return f"Error: File not found - {filepath}"

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return content
        except UnicodeDecodeError:
            # Try reading as binary
            with open(full_path, 'rb') as f:
                content = f.read()
                return f"Binary file, size: {len(content)} bytes"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, filepath: str, content: str) -> str:
        """Write content to file"""
        full_path = self._resolve_path(filepath)

        # Create directory if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {filepath}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def list_files(self, directory: str = '.') -> str:
        """List files in directory"""
        full_path = self._resolve_path(directory)

        if not os.path.exists(full_path):
            return f"Error: Directory not found - {directory}"

        try:
            files = []
            for item in sorted(os.listdir(full_path)):
                item_path = os.path.join(full_path, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    files.append(f"{item} (file, {size} bytes)")
                elif os.path.isdir(item_path):
                    files.append(f"{item}/ (directory)")

            if not files:
                return f"Directory '{directory}' is empty"

            return "\n".join(files)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def delete_file(self, filepath: str) -> str:
        """Delete a file"""
        full_path = self._resolve_path(filepath)

        if not os.path.exists(full_path):
            return f"Error: File not found - {filepath}"

        try:
            os.remove(full_path)
            return f"Successfully deleted {filepath}"
        except Exception as e:
            return f"Error deleting file: {str(e)}"

    def file_exists(self, filepath: str) -> bool:
        """Check if file exists"""
        full_path = self._resolve_path(filepath)
        return os.path.exists(full_path)

    def get_file_info(self, filepath: str) -> str:
        """Get file information"""
        full_path = self._resolve_path(filepath)

        if not os.path.exists(full_path):
            return f"Error: File not found - {filepath}"

        try:
            stat = os.stat(full_path)
            info = {
                'path': filepath,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'is_file': os.path.isfile(full_path),
                'is_dir': os.path.isdir(full_path)
            }

            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error getting file info: {str(e)}"

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to base_dir"""
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)
