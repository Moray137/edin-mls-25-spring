import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import traceback
import sys
import logging

# 设置日志文件
logging.basicConfig(filename="error.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 示例文档
    documents = [
        "Cats are small furry carnivores that are often kept as pets.",
        "Dogs are domesticated mammals, not natural wild animals.",
        "Hummingbirds can hover in mid-air by rapidly flapping their wings."
    ]

    # 加载嵌入模型
    EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)

    # 加载小型聊天模型
    chat_pipeline = pipeline("text-generation", model="facebook/opt-125m", device=0 if torch.cuda.is_available() else -1)

    def get_embedding(text: str) -> torch.Tensor:
        """Compute a simple average-pool embedding."""
        inputs = embed_tokenizer(text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            outputs = embed_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    # 预计算文档嵌入
    doc_embeddings = torch.vstack([get_embedding(doc) for doc in documents]).to(device)

    def retrieve_top_k(query_emb: torch.Tensor, k: int = 2) -> list:
        sims = torch.matmul(doc_embeddings, query_emb.T).squeeze()
        top_k_indices = torch.argsort(sims, descending=True)[:k]
        return [documents[i] for i in top_k_indices.cpu().numpy()]

    def rag_pipeline(query: str, k: int = 2) -> str:
        query_emb = get_embedding(query)
        retrieved_docs = retrieve_top_k(query_emb, k)
        context = "\n".join(retrieved_docs)
        prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
        generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]
        # 校验返回值结构是否正确
        if "generated_text" not in generated:
            raise ValueError(f"Pipeline output malformed: {generated}")
        return generated["generated_text"]

except Exception as e:
    logging.error("Startup failure:\n" + traceback.format_exc())
    sys.exit("Error occurred during app initialization. Exiting.")

# 请求模型
class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
def predict(payload: QueryRequest):
    try:
        result = rag_pipeline(payload.query, payload.k)
        return {"query": payload.query, "result": result}
    except Exception as e:
        logging.error(f"Runtime exception:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Server encountered an error. Check logs for more info.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)