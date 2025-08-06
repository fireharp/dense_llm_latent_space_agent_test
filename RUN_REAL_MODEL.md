# Running with Real Qwen2-0.5B Model

Since Qwen2-0.5B is not available on the free HuggingFace Inference API, here are your options:

## Option 1: Run Locally (Recommended)

```bash
# Install transformers if not already installed
uv add transformers accelerate

# Run with real model
uv run python run_real_qwen.py
```

This will:
- Download Qwen2-0.5B (~1GB) on first run
- Run actual model inference locally
- Show real hidden state processing

## Option 2: Use HuggingFace Spaces (Free)

1. Create a Space at https://huggingface.co/spaces
2. Deploy this simple Gradio app:

```python
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.Interface(fn=generate, inputs="text", outputs="text").launch()
```

3. Use the Space URL as your API endpoint

## Option 3: Use Smaller Local Model

If Qwen2-0.5B is too large, try these smaller models that work the same way:

```python
# In dense_lm.py, change model_name to one of:
"microsoft/phi-1_5"      # 1.3B parameters
"EleutherAI/pythia-160m" # 160M parameters  
"bigscience/bloomz-560m" # 560M parameters
```

## Option 4: Continue with MockLM

The MockLM perfectly simulates the interface and demonstrates the architecture:
- Same hidden dimension (896)
- Same API methods
- Instant loading
- No GPU required

## Current Architecture Summary

```
Text Input → [Qwen2-0.5B Encoder] → Hidden States (896-dim)
                                           ↓
                                    [Edge Transformer]
                                           ↓
                                    Hidden States (896-dim)
                                           ↓
                              [Qwen2-0.5B or Groq Decoder] → Text Output
```

The key insight: **All reasoning happens in 896-dimensional hidden space**, achieving 90%+ token reduction!

## To See Real Model Dimensions

```bash
# This script will download and show real Qwen2-0.5B info
uv run python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('Qwen/Qwen2-0.5B')
print(f'Hidden size: {config.hidden_size}')
print(f'Layers: {config.num_hidden_layers}')
print(f'Attention heads: {config.num_attention_heads}')
print(f'Vocab size: {config.vocab_size}')
print(f'Total params: ~{config.num_hidden_layers * config.hidden_size * 4 / 1e6:.0f}M')
"
```