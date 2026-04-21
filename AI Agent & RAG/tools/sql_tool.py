import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime


class SQLTool:
    """Tool for executing SQL queries on database"""

    def __init__(self, db_path: str = "./data/database.db"):
        self.name = "sql"
        self.description = "Execute SQL queries on database"
        self.db_path = db_path
        self.connection = None
        self._connect()

    def __call__(self, query: str, params: tuple = None) -> str:
        """Execute SQL query and return results"""
        try:
            cursor = self.connection.cursor()

            # Determine query type
            query_lower = query.strip().lower()

            if query_lower.startswith(('select', 'with')):
                cursor.execute(query, params or ())
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                if not rows:
                    return "No results"

                # Format as table
                result = [f"Columns: {', '.join(columns)}"]
                for row in rows[:50]:  # Limit to 50 rows
                    result.append(" | ".join(str(val) for val in row))

                if len(rows) > 50:
                    result.append(f"... and {len(rows) - 50} more rows")

                return "\n".join(result)
            else:
                # INSERT, UPDATE, DELETE, CREATE, etc.
                cursor.execute(query, params or ())
                self.connection.commit()
                return f"Success. Rows affected: {cursor.rowcount}"

        except sqlite3.Error as e:
            return f"SQL Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _connect(self):
        """Connect to database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
        except Exception as e:
            print(f"Database connection error: {str(e)}")

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

    def get_schema(self) -> str:
        """Get database schema"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            schema = []
            for table in tables:
                table_name = table[0]
                schema.append(f"\nTable: {table_name}")
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()

                for col in columns:
                    schema.append(f"  - {col[1]} ({col[2]})")

            return "\n".join(schema) if schema else "No tables found"
        except Exception as e:
            return f"Error getting schema: {str(e)}"

    def execute_many(self, queries: List[str]) -> List[str]:
        """Execute multiple queries in transaction"""
        results = []
        try:
            cursor = self.connection.cursor()
            self.connection.execute("BEGIN TRANSACTION")

            for query in queries:
                try:
                    cursor.execute(query)
                    if query.strip().lower().startswith(('select', 'with')):
                        rows = cursor.fetchall()
                        results.append(f"Query returned {len(rows)} rows")
                    else:
                        results.append(f"Query affected {cursor.rowcount} rows")
                except Exception as e:
                    results.append(f"Error: {str(e)}")
                    self.connection.rollback()
                    return results

            self.connection.commit()
            return results
        except Exception as e:
            self.connection.rollback()
            return [f"Transaction error: {str(e)}"]

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (table_name,)
            )
            return cursor.fetchone() is not None
        except Exception:
            return False

    def count_rows(self, table_name: str) -> Optional[int]:
        """Count rows in table"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            return cursor.fetchone()[0]
        except Exception:
            return None

    def create_table(self, table_name: str, columns: Dict[str, str]) -> str:
        """Create a new table"""
        try:
            column_defs = [f"{col} {dtype}" for col, dtype in columns.items()]
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_defs)});"
            cursor = self.connection.cursor()
            cursor.execute(query)
            self.connection.commit()
            return f"Table '{table_name}' created successfully"
        except Exception as e:
            return f"Error creating table: {str(e)}"

    def insert(self, table_name: str, data: Dict[str, Any]) -> str:
        """Insert data into table"""
        try:
            columns = list(data.keys())
            placeholders = ['?'] * len(columns)
            values = list(data.values())

            query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)});"
            cursor = self.connection.cursor()
            cursor.execute(query, values)
            self.connection.commit()
            return f"Inserted row with id {cursor.lastrowid}"
        except Exception as e:
            return f"Error inserting data: {str(e)}"
