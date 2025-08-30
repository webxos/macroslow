from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from markup_agent import MarkupAgent

app = FastAPI(title="MARKUP Agent API")

class MarkdownInput(BaseModel):
    content: str

class MarkupInput(BaseModel):
    content: str

@app.post("/to_markup")
async def convert_to_markup(input: MarkdownInput):
    """Convert Markdown to Markup via API."""
    agent = MarkupAgent()
    try:
        markup_content, errors = await agent.convert_to_markup(input.content)
        return {"markup": markup_content, "errors": errors}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/to_markdown")
async def convert_to_markdown(input: MarkupInput):
    """Convert Markup to Markdown via API."""
    agent = MarkupAgent()
    try:
        markdown_content, errors = await agent.convert_to_markdown(input.content)
        return {"markdown": markdown_content, "errors": errors}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/visualize")
async def visualize_transformation(markdown: MarkdownInput, markup: MarkupInput):
    """Generate 3D visualization of Markdown-to-Markup transformation."""
    agent = MarkupAgent()
    try:
        await agent.visualize_transformation(markdown.content, markup.content)
        return {"status": "Visualization generated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))