from sqlalchemy import create_engine, Column, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
import os

# --- CUSTOMIZATION POINT: Configure database connection ---
# Replace with your database URI; supports Dune 3.20.0 % forms
engine = create_engine('postgresql://user:pass@localhost:5432/chimera_hub')

Base = declarative_base()

# --- CUSTOMIZATION POINT: Define custom database models ---
# Supports OCaml Dune 3.20.0 implicit_transitive_deps option
class UserModel(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True)
    data = Column(JSON)
    last_login = Column(DateTime)

class WorkflowLog(Base):
    __tablename__ = 'workflow_logs'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String)
    status = Column(String)
    timestamp = Column(DateTime)

# --- CUSTOMIZATION POINT: Create tables ---
# Integrate with OCaml Dune 3.20.0 describe location
Base.metadata.create_all(engine)