"""
FastAPI wrapper around the LangChain backend with static file serving.
"""
from __future__ import annotations
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import uvicorn
from fashion_langgraph import get_sku_meta, run_graph  
from pydantic import BaseModel

class SkuGreetRequest(BaseModel):
    session_id: str
    sku: str
    last_user_msg: str | None = ""


class StyleRequest(BaseModel):
    sku_id:  str
    question: str
    language: str = Field("English")
    top_k:    int  = Field(4, ge=1, le=16)

class StyleBlock(BaseModel):
    category: str
    items:    List[Dict[str, Any]]

class StyleResponse(BaseModel):
    blocks: List[StyleBlock]

class SuggestRequest(BaseModel):
    prompt: str
    language: str = Field("English")
    top_k: int = Field(10, ge=1, le=32)

class SuggestResponse(BaseModel):
    skus: List[str]
    items: List[Dict[str, Any]]
    note:  str

class ChatRequest(BaseModel):
    session_id: str          # UUID the UI keeps in localStorage
    message: str
    language: str = "English"
    selected_sku: str | None = None

class ChatResponse(BaseModel):
    assistant_reply: str
    last_items:     list[dict[str, Any]] | None = None
    note:           str | None = None,
    selected_sku:   str | None = None

app = FastAPI(title="Fashion Suggestion API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def serve_index():
    """Serve the main HTML file"""
    return FileResponse("index.html")

@app.get("/health", tags=["internal"])
def health(): 
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest):
    try:
        state = await run_graph(
            session_id=req.session_id,
            user_msg=req.message,
            language=req.language,
            selected_sku=req.selected_sku,
        )
        return {
            "assistant_reply": state["assistant_reply"],
            "last_items":     state.get("last_items"),
            "note":           state.get("note"),
            "selected_sku":   state.get("selected_sku"),
        }
    except Exception as e:
        print('exception:', e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/style", response_model=StyleResponse, tags=["suggestions"])
async def style(req: StyleRequest, request: Request):
    try:
        # This endpoint would need to be implemented based on your needs
        # For now, returning empty response
        return {"blocks": []}
    except Exception as e:
        print('exception:', e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)