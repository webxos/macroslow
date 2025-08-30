```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from markup_receipts import MarkupReceipts
from markup_config import MarkupConfig

app = FastAPI(title="MARKUP Agent Receipt API")

class MarkdownInput(BaseModel):
    content: str

@app.post("/generate_receipt")
async def generate_receipt(input: MarkdownInput):
    """Generate a .mu receipt from a Markdown file."""
    config = MarkupConfig.load_from_env()
    receipts = MarkupReceipts(config.db_uri)
    try:
        receipt_content, errors = await receipts.generate_receipt(input.content)
        return {"receipt": receipt_content, "errors": errors}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/validate_receipt")
async def validate_receipt(markdown: MarkdownInput, receipt: MarkdownInput):
    """Validate a .mu receipt against its original Markdown file."""
    config = MarkupConfig.load_from_env()
    receipts = MarkupReceipts(config.db_uri)
    try:
        parsed_markdown = receipts.parser.parse_markdown(markdown.content)
        errors = receipts.validate_receipt(parsed_markdown, receipt.content)
        return {"valid": len(errors) == 0, "errors": errors}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))