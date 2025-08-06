"""Simple test of dense communication without DSPy signatures."""

import torch
import time
from mock_lm import MockLM
from dense_edge import DenseEdge
import json

print("=" * 60)
print("Mock LM Simple Test - Direct Dense Communication")
print("=" * 60)

# Test problems
test_problems = [
    ("What is 5 + 3?", 8),
    ("John has 8 apples and gives 3 to Mary. How many apples does John have?", 5),
    ("Calculate 12 - 7", 5),
    ("What is 4 * 6?", 24),
    ("Sarah had 15 cookies and ate 6. How many are left?", 9),
]

# Initialize components
print("\n1. Initializing components...")
device = "cpu"

planner = MockLM(device=device)
solver = MockLM(device=device)
edge = DenseEdge(d_model=896).to(device)

print("✓ Components initialized")

# Test dense communication
print("\n2. Testing dense communication flow...")
print("-" * 60)

results = []

for problem, expected in test_problems:
    print(f"\nProblem: {problem}")
    print(f"Expected: {expected}")
    
    # Dense pipeline
    start_time = time.time()
    
    # Step 1: Planner encodes
    planning_prompt = f"Problem: {problem}\nLet me create a step-by-step plan:"
    h_plan = planner.encode(planning_prompt)
    print(f"  1. Planner encoded → {h_plan.shape}")
    
    # Step 2: Edge transforms
    h_transformed = edge(h_plan)
    print(f"  2. Edge transformed → {h_transformed.shape}")
    
    # Step 3: Solver decodes
    solution = solver.decode(h_transformed)
    dense_time = time.time() - start_time
    
    print(f"  3. Solver decoded → {solution[:60]}...")
    
    # Count tokens
    dense_tokens = len(solution.split())
    
    # Compare with traditional approach (simulate)
    traditional_plan = "Step 1: Identify the numbers. Step 2: Apply the operation. Step 3: Calculate result."
    traditional_solution = f"Following the plan, the answer is {expected}."
    traditional_tokens = len(traditional_plan.split()) + len(traditional_solution.split())
    
    token_reduction = 1 - (dense_tokens / traditional_tokens)
    
    print(f"\nMetrics:")
    print(f"  Dense tokens: {dense_tokens}")
    print(f"  Traditional tokens: {traditional_tokens}")
    print(f"  Token reduction: {token_reduction:.1%}")
    print(f"  Time: {dense_time*1000:.1f}ms")
    
    results.append({
        "problem": problem,
        "expected": expected,
        "solution": solution,
        "dense_tokens": dense_tokens,
        "traditional_tokens": traditional_tokens,
        "token_reduction": token_reduction,
        "time_ms": dense_time * 1000
    })

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

avg_dense_tokens = sum(r["dense_tokens"] for r in results) / len(results)
avg_trad_tokens = sum(r["traditional_tokens"] for r in results) / len(results)
avg_reduction = sum(r["token_reduction"] for r in results) / len(results)
avg_time = sum(r["time_ms"] for r in results) / len(results)

print(f"\nAverage Metrics:")
print(f"  Dense tokens: {avg_dense_tokens:.1f}")
print(f"  Traditional tokens: {avg_trad_tokens:.1f}")
print(f"  Token reduction: {avg_reduction:.1%}")
print(f"  Average time: {avg_time:.1f}ms")
print(f"  Dense uses {(avg_dense_tokens/avg_trad_tokens):.1%} of baseline tokens")

# Test batch processing
print("\n" + "=" * 60)
print("BATCH PROCESSING TEST")
print("=" * 60)

# Encode all problems (they have different lengths, so process separately)
batch_prompts = [f"Problem: {p}\nPlan:" for p, _ in test_problems]
batch_hidden = [planner.encode(prompt) for prompt in batch_prompts]
print(f"\nBatch encoded: {len(batch_hidden)} sequences")
print(f"  Sequence lengths: {[h.shape[0] for h in batch_hidden]}")

# Process through edge
batch_transformed = [edge(h) for h in batch_hidden]
print(f"Batch transformed: {len(batch_transformed)} sequences")

# Decode all
batch_solutions = [solver.decode(h) for h in batch_transformed]
print(f"Batch decoded: {len(batch_solutions)} solutions")
print(f"  First solution: {batch_solutions[0][:50]}...")

# Show hidden state statistics
print("\n" + "=" * 60)
print("HIDDEN STATE ANALYSIS")
print("=" * 60)

example_hidden = planner.encode("What is 10 + 5?")
print(f"\nHidden state shape: {example_hidden.shape}")
print(f"Hidden state stats:")
print(f"  Mean: {example_hidden.mean().item():.4f}")
print(f"  Std: {example_hidden.std().item():.4f}")
print(f"  Min: {example_hidden.min().item():.4f}")
print(f"  Max: {example_hidden.max().item():.4f}")

# Save results
with open("mock_lm_simple_results.json", "w") as f:
    json.dump({
        "results": results,
        "summary": {
            "avg_dense_tokens": avg_dense_tokens,
            "avg_traditional_tokens": avg_trad_tokens,
            "avg_token_reduction": avg_reduction,
            "avg_time_ms": avg_time,
            "token_ratio": avg_dense_tokens / avg_trad_tokens
        }
    }, f, indent=2)

print("\n✓ Results saved to mock_lm_simple_results.json")
print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)