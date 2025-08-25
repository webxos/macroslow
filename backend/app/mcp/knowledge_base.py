from django_mongo_engine import connect, Document
from django_mongo_engine.fields import StringField
from crewai import Agent

connect('webxos_rag')

class KnowledgeEntry(Document):
    content = StringField(required=True)
    meta = {'collection': 'knowledge'}

class KnowledgeBaseAgent(Agent):
    def __init__(self):
        super().__init__(role="Knowledge Base Agent", goal="Manage Django MongoDB", backstory="Database specialist", llm="openai/gpt-4o")

    def query_or_update(self, query: str, content=None):
        if content:
            entry = KnowledgeEntry(content=content).save()
            return f"New entry created with ID: {entry.id}"
        existing = KnowledgeEntry.objects(content__icontains=query).first()
        return existing.content if existing else "No match, create new entry"

agent = KnowledgeBaseAgent()
