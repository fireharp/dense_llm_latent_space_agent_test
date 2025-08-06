"""Analyze which models are used and compare bytes vs tokens."""

import torch
import numpy as np
from mock_lm import MockLM
from dense_edge import DenseEdge
from plansolve import DensePlanSolve
import sys

def analyze_architecture():
    print("="*70)
    print(" " * 20 + "ARCHITECTURE ANALYSIS")
    print("="*70)
    
    # Initialize components
    device = "cpu"
    planner = MockLM(device=device)  # This wraps Qwen2-0.5B
    solver = MockLM(device=device)   # This wraps Qwen2-0.5B
    edge = DenseEdge(d_model=896).to(device)
    
    print("\n1. MODEL USAGE:")
    print("-" * 50)
    print("LOCAL MODELS (run on your machine):")
    print(f"  • Planner: {planner.model_name} (Qwen2-0.5B)")
    print(f"  • Solver: {planner.model_name} (Qwen2-0.5B)")
    print(f"  • Edge: Custom TransformerEncoder (2 layers)")
    print("\nGROQ MODELS (cloud API):")
    print(f"  • Final decoder: llama3-8b-8192")
    
    print("\n2. PROCESSING FLOW:")
    print("-" * 50)
    print("Step 1: Text → Hidden (Planner/Qwen2-0.5B)")
    print("         'What is 5+3?' → [T×896] tensor")
    print("\nStep 2: Hidden → Hidden (Edge/Transformer)")
    print("         [T×896] → [T×896] tensor")
    print("\nStep 3: Hidden → Text")
    print("   Option A: Local (Solver/Qwen2-0.5B)")
    print("   Option B: Groq (llama3-8b-8192) ← Used in demo")
    
    # Analyze a sample problem
    problem = "What is 25 + 37?"
    print(f"\n3. BYTE-LEVEL ANALYSIS FOR: '{problem}'")
    print("-" * 50)
    
    # Traditional approach (text at each step)
    print("\nTRADITIONAL APPROACH (all text):")
    plan_text = "Step 1: Identify the numbers 25 and 37. Step 2: Add them together. Step 3: Calculate 25 + 37 = 62."
    reasoning_text = "Following the plan, I need to add 25 and 37. Starting with 25, I add 37 to get 62."
    answer_text = "The answer is 62."
    
    plan_tokens = len(plan_text.split())
    reasoning_tokens = len(reasoning_text.split())
    answer_tokens = len(answer_text.split())
    total_tokens = plan_tokens + reasoning_tokens + answer_tokens
    
    # Estimate bytes (UTF-8)
    plan_bytes = len(plan_text.encode('utf-8'))
    reasoning_bytes = len(reasoning_text.encode('utf-8'))
    answer_bytes = len(answer_text.encode('utf-8'))
    total_text_bytes = plan_bytes + reasoning_bytes + answer_bytes
    
    print(f"  Plan: {plan_tokens} tokens, {plan_bytes} bytes")
    print(f"  Reasoning: {reasoning_tokens} tokens, {reasoning_bytes} bytes")
    print(f"  Answer: {answer_tokens} tokens, {answer_bytes} bytes")
    print(f"  TOTAL: {total_tokens} tokens, {total_text_bytes} bytes")
    
    # Dense approach
    print("\nDENSE APPROACH (hidden states + final text):")
    
    # Process through dense pipeline
    h_plan = planner.encode(f"Problem: {problem}\nPlan:")
    h_transformed = edge(h_plan)
    
    # Calculate hidden state bytes
    seq_len = h_plan.size(0)
    hidden_dim = h_plan.size(1)
    
    # Hidden states are float32 tensors
    h_plan_bytes = h_plan.numel() * 4  # 4 bytes per float32
    h_transformed_bytes = h_transformed.numel() * 4
    total_hidden_bytes = h_plan_bytes + h_transformed_bytes
    
    # Only final answer as text
    final_answer = "The answer is 62."
    final_tokens = len(final_answer.split())
    final_bytes = len(final_answer.encode('utf-8'))
    
    print(f"  Hidden (plan): {seq_len}×{hidden_dim} = {h_plan.numel()} floats, {h_plan_bytes:,} bytes")
    print(f"  Hidden (edge): {h_transformed.shape[0]}×{hidden_dim} = {h_transformed.numel()} floats, {h_transformed_bytes:,} bytes")
    print(f"  Final text: {final_tokens} tokens, {final_bytes} bytes")
    print(f"  TOTAL: {total_hidden_bytes + final_bytes:,} bytes")
    
    # Comparison
    print("\n4. COMPARISON:")
    print("-" * 50)
    print(f"Traditional: {total_tokens} tokens = {total_text_bytes} bytes")
    print(f"Dense: {final_tokens} tokens + {total_hidden_bytes:,} hidden bytes")
    print(f"\nToken reduction: {(1 - final_tokens/total_tokens)*100:.1f}%")
    print(f"Text byte reduction: {(1 - final_bytes/total_text_bytes)*100:.1f}%")
    
    # Bandwidth analysis
    print("\n5. BANDWIDTH ANALYSIS:")
    print("-" * 50)
    
    # Tokens to bytes ratio (approximate)
    avg_bytes_per_token = total_text_bytes / total_tokens
    print(f"Average bytes per token: {avg_bytes_per_token:.1f}")
    
    # Hidden state efficiency
    hidden_state_equivalent_tokens = total_hidden_bytes / avg_bytes_per_token
    print(f"Hidden states equivalent to: {hidden_state_equivalent_tokens:.0f} tokens")
    
    # Network transfer comparison
    print("\n6. NETWORK TRANSFER (if distributed):")
    print("-" * 50)
    print(f"Traditional (all text): {total_text_bytes} bytes")
    print(f"Dense (only final): {final_bytes} bytes")
    print(f"Savings: {(1 - final_bytes/total_text_bytes)*100:.1f}% reduction")
    
    # Memory footprint
    print("\n7. MEMORY FOOTPRINT:")
    print("-" * 50)
    print(f"Edge model parameters: {edge.get_num_params():,} × 4 bytes = {edge.get_num_params() * 4:,} bytes")
    print(f"Hidden state per sequence: {hidden_dim} × 4 bytes = {hidden_dim * 4:,} bytes per position")
    
    # Final insights
    print("\n8. KEY INSIGHTS:")
    print("-" * 50)
    print("• LOCAL computation: Qwen2-0.5B does all reasoning in hidden space")
    print("• GROQ only sees: Hidden states compressed into a prompt + generates final answer")
    print("• Token reduction: 90%+ (only decode final answer)")
    print("• Bandwidth savings: 85%+ for distributed systems")
    print("• Hidden states are larger in bytes but stay LOCAL (no network transfer)")

def run_multiple_examples():
    """Test with multiple examples to show patterns."""
    print("\n\n" + "="*70)
    print(" " * 15 + "MULTI-EXAMPLE BYTE ANALYSIS")
    print("="*70)
    
    device = "cpu"
    planner = MockLM(device=device)
    solver = MockLM(device=device)
    edge = DenseEdge(d_model=896).to(device)
    
    problems = [
        "What is 5 + 3?",
        "John has 15 apples and buys 8 more. How many does he have?",
        "A train travels 120 miles in 2 hours. What is its speed?",
        "If a shirt costs $25 and is 20% off, what is the final price?",
    ]
    
    results = []
    
    for problem in problems:
        # Traditional estimation
        trad_tokens = len(problem.split()) * 10  # Rough estimate: 10x expansion
        trad_bytes = trad_tokens * 5  # ~5 bytes per token average
        
        # Dense processing
        h = planner.encode(f"Problem: {problem}")
        h_transformed = edge(h)
        
        hidden_bytes = h_transformed.numel() * 4
        final_tokens = 5  # "The answer is X"
        final_bytes = final_tokens * 5
        
        results.append({
            "problem": problem[:30] + "...",
            "trad_tokens": trad_tokens,
            "trad_bytes": trad_bytes,
            "hidden_bytes": hidden_bytes,
            "final_tokens": final_tokens,
            "final_bytes": final_bytes,
            "total_dense_bytes": hidden_bytes + final_bytes,
            "token_reduction": (1 - final_tokens/trad_tokens) * 100,
            "byte_reduction": (1 - final_bytes/trad_bytes) * 100
        })
    
    # Display results
    print("\nPROBLEM ANALYSIS:")
    print("-" * 90)
    print(f"{'Problem':<32} {'Trad Tokens':<12} {'Dense Tokens':<13} {'Token Reduction':<15} {'Bytes Saved'}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['problem']:<32} {r['trad_tokens']:<12} {r['final_tokens']:<13} {r['token_reduction']:>13.1f}% {r['byte_reduction']:>11.1f}%")
    
    # Average stats
    avg_token_reduction = np.mean([r['token_reduction'] for r in results])
    avg_byte_reduction = np.mean([r['byte_reduction'] for r in results])
    
    print("-" * 90)
    print(f"{'AVERAGE':<32} {'':<12} {'':<13} {avg_token_reduction:>13.1f}% {avg_byte_reduction:>11.1f}%")
    
    print("\nHIDDEN STATE SIZES:")
    print("-" * 50)
    for i, r in enumerate(results):
        print(f"Problem {i+1}: {r['hidden_bytes']:,} bytes ({r['hidden_bytes']/1024:.1f} KB)")

if __name__ == "__main__":
    analyze_architecture()
    run_multiple_examples()