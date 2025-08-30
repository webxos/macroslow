from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class TransformationLog(Base):
    __tablename__ = "transformation_logs"
    id = Column(Integer, primary_key=True)
    input_content = Column(Text, nullable=False)
    output_content = Column(Text, nullable=False)
    errors = Column(Text)

class MarkupDatabase:
    def __init__(self, db_uri: str):
        """Initialize SQLAlchemy database for transformation logs."""
        self.engine = create_engine(db_uri)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def log_transformation(self, input_content: str, output_content: str, errors: List[str]):
        """Log a transformation to the database."""
        session = self.Session()
        log = TransformationLog(
            input_content=input_content,
            output_content=output_content,
            errors=str(errors)
        )
        session.add(log)
        session.commit()
        session.close()
