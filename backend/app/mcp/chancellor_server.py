from mcp.server import MCPServer
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
engine = create_engine("sqlite:///wallet.db")

class Wallet(Base):
    __tablename__ = "wallets"
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    balance = Column(Integer)

class WalletRequest(BaseModel):
    user_id: str
    template: str

class ChancellorServer(MCPServer):
    def __init__(self):
        super().__init__()
        Base.metadata.create_all(engine)

    async def create_boilerplate_wallet(self, request: WalletRequest):
        templates = {"Researcher": 100, "Validator": 50, "Developer": 200, "Citizen": 25}
        balance = templates.get(request.template, 0)
        with engine.connect() as conn:
            conn.execute(Wallet.__table__.insert().values(user_id=request.user_id, balance=balance))
        return {"wallet_id": request.user_id, "balance": balance}

    async def get_balance(self, wallet_id: str):
        with engine.connect() as conn:
            result = conn.execute(Wallet.__table__.select().where(Wallet.id == wallet_id)).fetchone()
            return {"balance": result[0]} if result else {"error": "Wallet not found"}

    async def execute_transaction(self, from_wallet: str, to_wallet: str, amount: int):
        with engine.connect() as conn:
            conn.execute(Wallet.__table__.update().where(Wallet.id == from_wallet).values(balance=Wallet.balance - amount))
            conn.execute(Wallet.__table__.update().where(Wallet.id == to_wallet).values(balance=Wallet.balance + amount))
        return {"status": "completed"}

    async def distribute_rewards(self, user_id: str, action: str):
        rewards = {"training_job": 10, "data_contribution": 5}
        amount = rewards.get(action, 0)
        with engine.connect() as conn:
            conn.execute(Wallet.__table__.update().where(Wallet.id == user_id).values(balance=Wallet.balance + amount))
        return {"user_id": user_id, "reward": amount}

    async def propose_dao_governance(self, proposal: str):
        return {"proposal": proposal, "status": "pending"}

server = ChancellorServer()
server.run()
