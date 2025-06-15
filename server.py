import argparse
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from langchain_community.llms import Ollama
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7

# Initialize memory and vector store
DB_DIR = os.environ.get("CHROMA_DB", "chroma_db")
VECTOR_DB = Chroma(
    persist_directory=DB_DIR,
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)
memory = VectorStoreRetrieverMemory(retriever=VECTOR_DB.as_retriever(search_kwargs={"k": 6}))


def get_chain(model_name: str):
    llm = Ollama(model=model_name)
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
        return {"choices": [{"message": {"role": "assistant", "content": response}}]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

@app.post("/v1/vision")
async def vision_endpoint(file: UploadFile = File(...), prompt: str = "Describe the image"):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        llm = Ollama(model="llava")
        response = llm.generate([{ 
            "role": "user",
            "content": prompt,
            "images": [image_bytes]
        }])
        return {"text": response.generations[0][0].text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main(host: str = "0.0.0.0", port: int = 8001):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local ChatGPT backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8001, type=int)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
