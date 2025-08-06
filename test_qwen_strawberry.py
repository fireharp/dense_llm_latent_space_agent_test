"""Test Qwen2.5-0.5B-Instruct with strawberry counting and compare approaches."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

def test_pipeline_approach():
    """Test using pipeline (high-level)."""
    print("="*70)
    print("1. TESTING PIPELINE APPROACH (High-level)")
    print("="*70)
    
    # Use pipeline with CPU
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
    
    # Test strawberry
    messages = [
        {"role": "user", "content": "How many r's are in the word strawberry?"},
    ]
    
    start = time.time()
    result = pipe(messages, max_new_tokens=50)
    elapsed = time.time() - start
    
    print(f"\nQuestion: How many r's are in the word strawberry?")
    print(f"Response: {result[0]['generated_text'][-1]['content']}")
    print(f"Time: {elapsed:.2f}s")
    
    # Test simple math
    messages = [
        {"role": "user", "content": "What is 25 + 17?"},
    ]
    
    start = time.time()
    result = pipe(messages, max_new_tokens=50)
    elapsed = time.time() - start
    
    print(f"\nQuestion: What is 25 + 17?")
    print(f"Response: {result[0]['generated_text'][-1]['content']}")
    print(f"Time: {elapsed:.2f}s")

def test_direct_approach():
    """Test using direct model loading."""
    print("\n" + "="*70)
    print("2. TESTING DIRECT MODEL APPROACH (More control)")
    print("="*70)
    
    # Load model and tokenizer
    print("\nLoading Qwen2.5-0.5B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Check model info
    print(f"Model loaded! Hidden size: {model.config.hidden_size}")
    print(f"Vocab size: {model.config.vocab_size}")
    
    # Test strawberry
    messages = [
        {"role": "user", "content": "How many r's are in the word strawberry?"},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    elapsed = time.time() - start
    
    print(f"\nQuestion: How many r's are in the word strawberry?")
    print(f"Response: {response}")
    print(f"Time: {elapsed:.2f}s")
    
    # Let's manually count to verify
    word = "strawberry"
    r_count = word.count('r')
    print(f"Actual count: {r_count} r's in '{word}'")
    
    # Test with hidden states access
    print("\n" + "-"*50)
    print("ACCESSING HIDDEN STATES:")
    
    with torch.no_grad():
        # Get hidden states
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"  Batch size: {hidden_states.shape[0]}")
        print(f"  Sequence length: {hidden_states.shape[1]}")
        print(f"  Hidden dimension: {hidden_states.shape[2]}")
        
        # This is what we use in dense approach!
        print(f"\n💡 In dense approach, we'd pass these {hidden_states.shape[2]}-dim vectors")
        print(f"   instead of generating text at each step!")

def test_dense_compatible():
    """Test if model is compatible with our dense approach."""
    print("\n" + "="*70)
    print("3. CHECKING DENSE APPROACH COMPATIBILITY")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Test encoding text to hidden states (like our encode method)
    text = "How many r's are in strawberry?"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        # Get embeddings
        input_embeds = model.get_input_embeddings()(inputs.input_ids)
        print(f"\nInput embeddings shape: {input_embeds.shape}")
        
        # Forward through model
        outputs = model(inputs_embeds=input_embeds, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].squeeze(0)
        
        print(f"Hidden states shape: {hidden.shape}")
        print(f"✓ Compatible with dense approach!")
        
        # Test generation from hidden states
        print("\nTesting generation from hidden states...")
        generated = model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=20,
            do_sample=False,
        )
        result = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: {result}")

def compare_models():
    """Compare Qwen2-0.5B vs Qwen2.5-0.5B-Instruct."""
    print("\n" + "="*70)
    print("4. MODEL COMPARISON")
    print("="*70)
    
    from transformers import AutoConfig
    
    # Check both configs
    config_old = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
    config_new = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    print("\nModel comparison:")
    print(f"{'Model':<30} {'Hidden Size':<12} {'Vocab Size':<12} {'Layers'}")
    print("-" * 70)
    print(f"{'Qwen/Qwen2-0.5B':<30} {config_old.hidden_size:<12} {config_old.vocab_size:<12} {config_old.num_hidden_layers}")
    print(f"{'Qwen/Qwen2.5-0.5B-Instruct':<30} {config_new.hidden_size:<12} {config_new.vocab_size:<12} {config_new.num_hidden_layers}")
    
    if config_old.hidden_size == config_new.hidden_size:
        print(f"\n✓ Same hidden size ({config_old.hidden_size})! Dense approach will work with both!")
    else:
        print(f"\n⚠️  Different hidden sizes! Need to adjust dense approach.")

if __name__ == "__main__":
    print("Testing Qwen2.5-0.5B-Instruct\n")
    
    # Run tests
    test_pipeline_approach()
    test_direct_approach()
    test_dense_compatible()
    compare_models()
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("- Qwen2.5-0.5B-Instruct is the newer instruction-tuned version")
    print("- Both pipeline and direct approaches work")
    print("- Model exposes hidden states for dense approach")
    print("- Check if hidden dimensions match for compatibility")
    print("="*70)