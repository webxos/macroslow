import sys
import os
import json
import datetime
from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ErrorLog(Base):
    __tablename__ = 'error_logs'
    id = Column(String, primary_key=True)
    commit_hash = Column(String, index=True)
    error_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

def log_error(commit_hash, error_message):
    # Ensure logs directory exists
    log_dir = "server/logs"
    os.makedirs(log_dir, exist_ok=True)

    # Log to file
    log_entry = {
        "commit_hash": commit_hash,
        "error_message": error_message,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    with open(f"{log_dir}/{commit_hash}.json", "a") as f:
        json.dump(log_entry, f, indent=2)
        f.write("\n")

    # Log to SQLAlchemy database
    engine = create_engine('sqlite:///server/logs/error_logs.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    error_log = ErrorLog(
        id=f"{commit_hash}_{datetime.datetime.utcnow().isoformat()}",
        commit_hash=commit_hash,
        error_message=error_message
    )
    session.add(error_log)
    session.commit()
    session.close()

    # Generate MAML-compatible receipt
    maml_receipt = f"""```maml
## Error_Log
```json
{json.dumps(log_entry, indent=2)}
```
"""
    with open(f"{log_dir}/{commit_hash}.mu", "w") as f:
        f.write(maml_receipt)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python error_logger.py <commit_hash> [error_message]")
        sys.exit(1)

    commit_hash = sys.argv[1]
    error_message = sys.argv[2] if len(sys.argv) > 2 else "No error message provided"
    log_error(commit_hash, error_message)
    print(f"Error logged for commit {commit_hash}")