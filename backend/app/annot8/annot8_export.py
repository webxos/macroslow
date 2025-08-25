from fastapi import Depends, HTTPException
from backend.app.database import SessionLocal, Base
from sqlalchemy import Column, Integer, String, DateTime, Text
from backend.app.mcp.sentinel_server import SentinelServer
from datetime import datetime
import csv
import io

sentinel = SentinelServer()
Base = declarative_base()

class Annotation(Base):
    __tablename__ = "annotations"
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    x_percent = Column(Float)
    y_percent = Column(Float)
    user_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/api/annot8/export")
async def export_annotations(format: str = "csv", user=Depends(sentinel.validate_request), db: Session = Depends(get_db)):
    if not user["status"] == "valid":
        raise HTTPException(status_code=403, detail="Unauthorized")
    annotations = db.query(Annotation).filter(Annotation.user_id == "user1").all()
    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "text", "x_percent", "y_percent", "created_at"])
        for a in annotations:
            writer.writerow([a.id, a.text, a.x_percent, a.y_percent, a.created_at])
        return {"data": output.getvalue(), "format": "csv"}
    return {"error": "Unsupported format"}
