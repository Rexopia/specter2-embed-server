from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

app = FastAPI(title="Specter2 Embedding API")

# Model configuration
BASE_MODEL_ID = "allenai/specter2_base"
ADAPTER_ID = "allenai/specter2"

class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        print(f"Loading tokenizer from {BASE_MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        
        print(f"Loading base model from {BASE_MODEL_ID}...")
        self.model = AutoAdapterModel.from_pretrained(BASE_MODEL_ID)
        
        print(f"Loading adapter from {ADAPTER_ID}...")
        self.model.load_adapter(ADAPTER_ID, source="hf", load_as="specter2", set_active=True)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> tuple[List[List[float]], int]:
        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512, 
                return_token_type_ids=False
            ).to(self.device)

            # Count tokens for this batch
            total_tokens += int(inputs.attention_mask.sum().item())

            with torch.no_grad():
                output = self.model(**inputs)
            
            # Take the first token in the batch as the embedding
            batch_embeddings = output.last_hidden_state[:, 0, :]
            all_embeddings.extend(batch_embeddings.cpu().tolist())
            
            # Optional: Clear CUDA cache if using GPU to prevent fragmentation, though usually not needed for inference
            # if self.device == "cuda":
            #     torch.cuda.empty_cache()

        return all_embeddings, total_tokens

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    model_manager.load_model()

# OpenAI-compatible schemas
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = "specter2"
    encoding_format: Optional[str] = "float" # support float (default) or base64 (not implemented here for simplicity unless asked)
    user: Optional[str] = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    if isinstance(request.input, str):
        input_texts = [request.input]
    else:
        input_texts = request.input
    
    if not input_texts:
        raise HTTPException(status_code=400, detail="Input text list cannot be empty")

    try:
        embeddings, token_count = model_manager.get_embeddings(input_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
    
    data_items = []
    for idx, embedding in enumerate(embeddings):
        data_items.append(EmbeddingData(
            embedding=embedding,
            index=idx
        ))
    
    return EmbeddingResponse(
        data=data_items,
        model=request.model,
        usage=Usage(prompt_tokens=token_count, total_tokens=token_count) 
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_manager.model is not None}

