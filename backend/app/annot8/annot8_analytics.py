from fastapi import Depends, HTTPException
from backend.app.database import SessionLocal, Base
from sqlalchemy import func
from backend.app.mcp.sentinel_server import SentinelServer
from datetime import datetime

sentinel = SentinelServer()

class AnnotationStats(Base):
    __tablename__ = "annotation_stats"
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/api/annot8/analytics")
async def get_analytics(user=Depends(sentinel.validate_request), db: Session = Depends(get_db)):
    if not user["status"] == "valid":
        raise HTTPException(status_code=403, detail="Unauthorized")
    stats = db.query(AnnotationStats.user_id, func.count(AnnotationStats.id).label("annotation_count"))\
              .group_by(AnnotationStats.user_id).all()
    return {"analytics": [{"user_id": s[0], "count": s[1]} for s in stats]}
