"""Run with REAL Qwen2-0.5B model (not mock)."""

import torch
from dense_lm import DenseLM  # Real implementation
from dense_edge import DenseEdge
from plansolve import DensePlanSolve
import time

print("Loading REAL Qwen2-0.5B model...")
print("This will download ~1GB on first run and requires ~2GB RAM/VRAM")
print("-" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Check if we have GPU
if device == "cuda":
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Initialize REAL Qwen2-0.5B models
print("\nLoading Planner (Qwen2-0.5B)...")
start = time.time()
planner = DenseLM(model_name="Qwen/Qwen2-0.5B", device=device)
print(f"✓ Planner loaded in {time.time()-start:.1f}s")

print("\nLoading Solver (Qwen2-0.5B)...")
start = time.time()
solver = DenseLM(model_name="Qwen/Qwen2-0.5B", device=device)
print(f"✓ Solver loaded in {time.time()-start:.1f}s")

# Edge is always custom (not pretrained)
edge = DenseEdge(d_model=896).to(device)
print(f"✓ Edge initialized ({edge.get_num_params():,} parameters)")

# Create dense module
dense_module = DensePlanSolve(
    planner_lm=planner,
    solver_lm=solver,
    edge=edge,
    device=device,
    share_lm=False  # Using separate instances
)

# Test it
print("\n" + "="*60)
print("Testing REAL Qwen2-0.5B Dense Pipeline")
print("="*60)

problems = [
    "What is 15 + 27?",
    "John has 8 apples and buys 5 more. How many does he have?",
]

for problem in problems:
    print(f"\nProblem: {problem}")
    print("Processing with real Qwen2-0.5B...")
    
    start = time.time()
    
    # Step 1: Encode with real model
    h_plan = planner.encode(f"Problem: {problem}\nPlan:")
    print(f"  • Encoded to hidden states: {h_plan.shape}")
    
    # Step 2: Transform
    h_transformed = edge(h_plan)
    print(f"  • Transformed by edge: {h_transformed.shape}")
    
    # Step 3: Decode with real model
    solution = solver.decode(h_transformed, max_new_tokens=50)
    
    elapsed = time.time() - start
    
    print(f"  • Processing time: {elapsed:.1f}s")
    print(f"  • Solution: {solution}")
    
print("\n" + "="*60)
print("Real Qwen2-0.5B Characteristics:")
print("  • Model size: ~500M parameters")
print("  • Hidden dimension: 896")
print("  • Vocabulary size: 151,936")
print("  • Memory usage: ~2GB")
print("="*60)