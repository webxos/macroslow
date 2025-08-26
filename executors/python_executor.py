import asyncio
import subprocess
import os
import tempfile
from typing import Dict, Any

class PythonExecutor:
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    async def execute(self, code: str) -> str:
        script_path = os.path.join(self.temp_dir.name, "script.py")
        with open(script_path, "w") as f:
            f.write("import sys\n")
            f.write("sys.stdout = open('/dev/stdout', 'w')\n")
            f.write("sys.stderr = open('/dev/stderr', 'w')\n")
            f.write(code)

        try:
            result = await asyncio.create_subprocess_exec(
                "python3", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir.name,
                limit=1024 * 1024  # 1MB memory limit
            )
            stdout, stderr = await result.communicate()
            if result.returncode != 0:
                return f"Error: {stderr.decode()}"
            return stdout.decode()
        except Exception as e:
            return f"Execution failed: {str(e)}"
