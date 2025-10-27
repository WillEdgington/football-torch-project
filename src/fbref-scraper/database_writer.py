import sqlite3

from pathlib import Path
from typing import Dict, List, Any

from config import DBDIR, DBNAME

class DatabaseWriter:
    def __init__(self, dbDir: str=DBDIR, dbName: str=DBNAME):
        self.dbPath = Path(dbDir) / dbName
        self.dbPath.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.dbPath)
    
    def createTable(self, tableName: str, schema: Dict[str, Dict[str, Any]]):
        cols = ", ".join([f"{col} {schema[col].get('type', 'TEXT')}" for col in schema.keys()])
        query = f"""
        CREATE TABLE IF NOT EXISTS {tableName} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {cols}
        );
        """
        self.conn.execute(query)
        self.conn.commit()

    def insertMany(self, tableName: str, records: List[Dict[str, Any]]):
        if not records:
            return
        
        cols = records[0].keys()
        placeholders = ", ".join(["?"] * len(cols))
        colStr = ", ".join(cols)
        query = f"""
        INSERT OR IGNORE INTO {tableName} ({colStr})
        VALUES ({placeholders});
        """

        data = [tuple(record.get(c) for c in cols) for record in records]
        self.conn.executemany(query, data)
        self.conn.commit()

    def rowExists(self, tableName: str, colValue: str, col: str) -> bool:
        row = self.conn.execute(f'SELECT 1 FROM "{tableName}" WHERE "{col}" = ? LIMIT 1;', (colValue,))
        return row.fetchone() is not None
    
    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()