
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    from src.config import MODEL_CACHE_DIR
except ImportError:
    MODEL_CACHE_DIR = None

def load_qwen_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Loads the Qwen model with persistent caching.
    """
    print(f"Loading {model_name}...")
    if MODEL_CACHE_DIR:
        print(f"Using persistent cache: {MODEL_CACHE_DIR}")
    
    if torch.cuda.is_available():
        print("GPU detected. Loading in float16.")
        device_map = "auto"
        torch_dtype = torch.float16
    else:
        print("No GPU detected. Loading in full precision on CPU.")
        device_map = "cpu"
        torch_dtype = torch.float32

    # Load tokenizer with cache_dir
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        cache_dir=MODEL_CACHE_DIR
    )

    # Load model with cache_dir
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
