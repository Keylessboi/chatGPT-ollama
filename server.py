import argparse
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from langchain_community.llms import Ollama
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.utilities import DuckDuckGoSearchRun
import json
import ollama

import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.head("/")
async def health_check():
    """Return a simple status message so clients can verify the server is up."""
    return {"status": "ok"}


@app.get("/api/tags")
@app.head("/api/tags")
async def list_tags():
    """Proxy the Ollama /api/tags endpoint to list local models."""
    try:
        res = ollama_client.list()
        return res.model_dump()
    except Exception as e:
        logger.exception("/api/tags failed")
        raise HTTPException(
            status_code=500,
            detail=f"Ollama error contacting {OLLAMA_URL}: {e}"
        )

# Data models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7

class ResearchRequest(BaseModel):
    model: str
    query: str

# Initialize memory and vector store
DB_DIR = os.environ.get("CHROMA_DB", "chroma_db")
VECTOR_DB = Chroma(
    persist_directory=DB_DIR,
    embedding_function=OllamaEmbeddings(model="nomic-embed-text", base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
)
memory = VectorStoreRetrieverMemory(
    retriever=VECTOR_DB.as_retriever(search_kwargs={"k": 6})
)

# Underlying Ollama server used for /api/* proxy endpoints
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_URL)


def get_chain(model_name: str):
    llm = Ollama(model=model_name, base_url=OLLAMA_URL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    chain = (
        {
            "history": memory,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    return chain

@app.post("/v1/chat/completions")
async def chat_completions(data: ChatCompletionRequest):
    if not data.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    user_message = data.messages[-1].content
    try:
        chain = get_chain(data.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{data.model}': {e}")

    try:
        response = chain.invoke(user_message)
        memory.save_context({"input": user_message}, {"output": response})
        VECTOR_DB.persist()
        return {"choices": [{"message": {"role": "assistant", "content": response}}]}
    except Exception as e:
        logger.exception("LLM invocation failed")
        raise HTTPException(
            status_code=500,
            detail=f"LLM error or Ollama unreachable at {OLLAMA_URL}: {e}"
        )

@app.post("/v1/vision")
async def vision_endpoint(file: UploadFile = File(...), prompt: str = "Describe the image"):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        llm = Ollama(model="llava", base_url=OLLAMA_URL)
        response = llm.generate([{ 
            "role": "user",
            "content": prompt,
            "images": [image_bytes]
        }])
        return {"text": response.generations[0][0].text}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Vision request failed")
        raise HTTPException(
            status_code=500,
            detail=f"LLM error or Ollama unreachable at {OLLAMA_URL}: {e}"
        )


@app.post("/v1/research")
async def research_endpoint(req: ResearchRequest):
    """Perform a web search and summarize the results with the LLM."""
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(req.query)
        llm = Ollama(model=req.model, base_url=OLLAMA_URL)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research assistant. Provide a concise summary of the following search results."),
            ("human", "{results}")
        ])
        chain = prompt | llm
        summary = chain.invoke({"results": results})
        return {"results": results, "summary": summary}
    except Exception as e:
        logger.exception("Research request failed")
        raise HTTPException(status_code=500, detail=f"Research error: {e}")


@app.post("/api/generate")
async def api_generate(request: Request):
    data = await request.json()
    stream = data.get("stream", True)
    try:
        if stream:
            def gen():
                for chunk in ollama_client.generate(stream=True, **data):
                    yield chunk.model_dump_json() + "\n"

            return StreamingResponse(gen(), media_type="application/json")
        else:
            res = ollama_client.generate(stream=False, **data)
            return res.model_dump()
    except Exception as e:
        logger.exception("/api/generate failed")
        raise HTTPException(status_code=500, detail=f"Ollama error contacting {OLLAMA_URL}: {e}")


@app.post("/api/chat")
async def api_chat(request: Request):
    data = await request.json()
    stream = data.get("stream", True)
    try:
        if stream:
            def gen():
                for chunk in ollama_client.chat(stream=True, **data):
                    yield chunk.model_dump_json() + "\n"

            return StreamingResponse(gen(), media_type="application/json")
        else:
            res = ollama_client.chat(stream=False, **data)
            return res.model_dump()
    except Exception as e:
        logger.exception("/api/chat failed")
        raise HTTPException(status_code=500, detail=f"Ollama error contacting {OLLAMA_URL}: {e}")


@app.get("/api/ps")
@app.head("/api/ps")
async def api_ps():
    """List models currently loaded into memory."""
    try:
        res = ollama_client.ps()
        return res.model_dump()
    except Exception as e:
        logger.exception("/api/ps failed")
        raise HTTPException(status_code=500, detail=f"Ollama error contacting {OLLAMA_URL}: {e}")


def main(host: str = "0.0.0.0", port: int = 8001):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local ChatGPT backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8001, type=int)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
