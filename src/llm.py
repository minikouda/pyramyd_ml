
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Try to import cache dir from config, default to None if not found
try:
    from src.config import MODEL_CACHE_DIR
except ImportError:
    MODEL_CACHE_DIR = None

def load_qwen_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Loads the Qwen model from the local cache.
    - L4 GPU: Uses float16 (no quantization) for best performance.
    - T4 GPU: Uses 4-bit quantization.
    - CPU: Uses full precision (warning: slow).
    """
    print(f"Loading {model_name}...")
    if MODEL_CACHE_DIR:
        print(f"Using persistent cache: {MODEL_CACHE_DIR}")

    device_map = "cpu"
    quantization_config = None
    torch_dtype = torch.float32

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Detected: {torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)")
        
        if vram >= 20: 
            print("Mode: Float16 (L4 Optimized)")
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            print("Mode: 4-bit Quantization (T4 Optimized)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            device_map = "auto"
            torch_dtype = None
    else:
        print("Mode: CPU (Full Precision)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        cache_dir=MODEL_CACHE_DIR
    )

    # Load model
    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "cache_dir": MODEL_CACHE_DIR
    }
    if quantization_config:
        load_kwargs["quantization_config"] = quantization_config
    if torch_dtype:
        load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
