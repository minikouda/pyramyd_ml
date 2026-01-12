from __future__ import annotations

from typing import Any, Optional


def _get_cache_dir() -> Optional[str]:
    try:
        from src.config import MODEL_CACHE_DIR

        return MODEL_CACHE_DIR
    except Exception:
        return None


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing optional dependency 'torch'. Install it to use LLM features."
        ) from e

def load_qwen_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Loads the Qwen model from the local cache.
    - L4 GPU: Uses float16 (no quantization) for best performance.
    - T4 GPU: Uses 4-bit quantization.
    - CPU: Uses full precision (warning: slow).
    """
    torch = _require_torch()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency 'transformers'. Install it to use LLM features."
        ) from e

    print(f"Loading {model_name}...")
    cache_dir = _get_cache_dir()
    if cache_dir:
        print(f"Using persistent cache: {cache_dir}")

    device_map: Any = "cpu"
    quantization_config: Any = None
    torch_dtype: Any = torch.float32

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Detected: {torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)")

        if vram >= 20:
            print("Mode: Float16 (GPU)")
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            # Quantization is optional; fall back gracefully if bitsandbytes isn't usable.
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                print("Mode: 4-bit Quantization (GPU)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                device_map = "auto"
                torch_dtype = None
            except Exception:
                print("Quantization unavailable; falling back to float16")
                device_map = "auto"
                torch_dtype = torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Mode: MPS (Apple Silicon)")
        device_map = {"": "mps"}
        torch_dtype = torch.float16
    else:
        print("Mode: CPU (Full Precision)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        cache_dir=cache_dir
    )

    # Load model
    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "cache_dir": cache_dir,
    }
    if quantization_config:
        load_kwargs["quantization_config"] = quantization_config
    if torch_dtype:
        load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model, tokenizer

def generate_response(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
):
    """Generate a completion for a plain-text prompt.

    Returns ONLY newly generated text (prompt is stripped).
    """

    torch = _require_torch()

    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        inputs = inputs.to(model.device)
    except Exception:
        # Some device_map configs don't expose a single `.device` cleanly.
        pass
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    gen_ids = outputs[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_chat(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
):
    """Generate a completion for chat-style messages.

    Works best with Qwen Instruct (uses tokenizer chat template when available).
    """

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\n\nASSISTANT:"
    return generate_response(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
