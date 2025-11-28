import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Model configuration
BASE_MODEL_ID = "allenai/specter2_base"
ADAPTER_ID = "allenai/specter2"

def preload_model():
    print(f"Downloading tokenizer from {BASE_MODEL_ID}...")
    AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    print(f"Downloading base model from {BASE_MODEL_ID}...")
    model = AutoAdapterModel.from_pretrained(BASE_MODEL_ID)
    
    print(f"Downloading adapter from {ADAPTER_ID}...")
    model.load_adapter(ADAPTER_ID, source="hf", load_as="specter2", set_active=True)
    
    print("Model download complete.")

if __name__ == "__main__":
    preload_model()

