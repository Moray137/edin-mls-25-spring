import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from queue import Queue, Empty
import threading
import time

app = FastAPI()

# Example documents in memory
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

# Basic Chat LLM
chat_pipeline = pipeline("text-generation", model="facebook/opt-125m")

request_queue = Queue()
global_request_id = 1
response_dict = {}
MAX_BATCH_SIZE = 4       
MAX_WAITING_TIME = 1    

def get_embedding(text: str) -> torch.Tensor:
    """Compute a simple average-pool embedding."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Return tensor

# Precompute document embeddings and store on GPU if available
dev = "cuda" if torch.cuda.is_available() else "cpu"
doc_embeddings = torch.vstack([get_embedding(doc) for doc in documents]).to(dev)

def retrieve_top_k(query_emb: torch.Tensor, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity using PyTorch on GPU."""
    query_emb = query_emb.to(doc_embeddings.device)
    sims = torch.matmul(doc_embeddings, query_emb.T).squeeze(1)
    top_k_indices = torch.argsort(sims, descending=True)[:k]
    return [documents[i] for i in top_k_indices.tolist()]

def rag_pipeline(query: str, k: int = 2) -> str:
    query_emb = get_embedding(query)
    retrieved_docs = retrieve_top_k(query_emb, k)
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    return generated




class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
def predict(payload: QueryRequest):
    global global_request_id
    global_request_id += 1
    current_req_id = global_request_id
    request_queue.put((current_req_id, payload.query, payload.k))
    start_time = time.time()
    while current_req_id not in response_dict:
        time.sleep(0.001)
    return {
        "query": payload.query,
        "result": response_dict.pop(current_req_id),
    }

def process_requests():
    while True:
        batch = []
        try:
            request = request_queue.get(timeout=MAX_WAITING_TIME)
            batch.append(request)
        except Empty:
            continue
        while len(batch) < MAX_BATCH_SIZE:
            try:
                batch.append(request_queue.get_nowait())
            except Empty:
                break
        if batch:
            results = {req_id: rag_pipeline(query, k) for req_id, query, k in batch}
            response_dict.update(results)

worker_thread = threading.Thread(target=process_requests, daemon=True)
worker_thread.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)