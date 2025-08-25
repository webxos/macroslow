from mcp.server import MCPServer
from pydantic import BaseModel
import psycopg2
import os
import json

class DatabaseQuery(BaseModel):
    query: str
    params: list = None

class PostgreSQLServer(MCPServer):
    def __init__(self):
        super().__init__()
        self.conn = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/annot8"))

    async def execute_query(self, query_data: DatabaseQuery):
        try:
            with self.conn.cursor() as cur:
                cur.execute(query_data.query, query_data.params or ())
                results = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return {"results": [dict(zip(columns, row)) for row in results]}
        except Exception as e:
            return {"error": str(e)}

    async def get_schema(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                """)
                return {"schema": [row[0] for row in cur.fetchall()]}
        except Exception as e:
            return {"error": str(e)}

server = PostgreSQLServer()
server.run()
