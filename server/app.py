"""FastAPI app for Bigram Tensor 1."""

import os

import torch
from fastapi import FastAPI

from bigram.agent import AgentRuntime
from bigram.rag import lexical_search

from .load_model import create_model_bundle
from .schemas import (
    AgentChatRequest,
    AgentChatResponse,
    GenerateRequest,
    GenerateResponse,
    RagSearchRequest,
    RagSearchResponse,
)
from .tools import create_default_tool_registry


app = FastAPI(title="Bigram Tensor 1")
bundle = create_model_bundle()
tool_registry = create_default_tool_registry()


@app.get("/health")
def health():
    cuda = torch.cuda.is_available()
    return {
        "ok": True,
        "cuda": cuda,
        "device": bundle.get("device"),
        "gpu": torch.cuda.get_device_name(0) if cuda else "NO CUDA",
        "torch": torch.__version__,
        "model_loaded": bundle.get("model") is not None,
        "load_error": bundle.get("error"),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model = bundle.get("model")
    tokenizer = bundle.get("tokenizer")
    if model is None or tokenizer is None:
        return GenerateResponse(text="", error="model not loaded")
    tok, tone = tokenizer.encode(req.prompt, add_special=False)
    bos = tokenizer.token_to_id("<bos>")
    tok = [bos] + tok
    tone = [0] + tone
    token_ids = torch.tensor([tok], dtype=torch.long, device=bundle["device"])
    tone_ids = torch.tensor([tone], dtype=torch.long, device=bundle["device"])
    with torch.no_grad():
        out_ids, out_tones, abstained = model.generate(
            token_ids,
            tone_ids,
            max_new_tokens=req.max_new_tokens,
            num_recurrence=req.recurrence,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            abstention_threshold=req.abstention_threshold,
        )
    text = tokenizer.decode(out_ids[0].tolist(), out_tones[0].tolist())
    return GenerateResponse(text=text, abstained=abstained)


@app.post("/agent/chat", response_model=AgentChatResponse)
def agent_chat(req: AgentChatRequest):
    model = bundle.get("model")
    tokenizer = bundle.get("tokenizer")
    if model is None or tokenizer is None:
        return AgentChatResponse(answer="", error="model not loaded")
    runtime = AgentRuntime(model, tokenizer, tool_registry, bundle["device"], req.max_tool_turns)
    messages = [m.model_dump() if hasattr(m, "model_dump") else m.dict() for m in req.messages]
    result = runtime.chat(
        messages,
        {
            "max_tool_turns": req.max_tool_turns,
            "recurrence": req.recurrence,
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "repetition_penalty": req.repetition_penalty,
        },
    )
    return AgentChatResponse(**result)


@app.post("/rag/search", response_model=RagSearchResponse)
def rag_search(req: RagSearchRequest):
    index = os.environ.get("BIGRAM_RAG_INDEX")
    if not index or not os.path.exists(index):
        return RagSearchResponse(results=[], error="RAG index not found")
    return RagSearchResponse(results=lexical_search(index, req.query, req.top_k))
