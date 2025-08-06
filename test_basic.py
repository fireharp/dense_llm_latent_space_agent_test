"""Basic component tests for Dense-Vector DSPy Agent."""

import sys
import time

print("=" * 50)
print("Stage 1: Basic Component Testing")
print("=" * 50)

# Test 1: Imports
print("\n1. Testing imports...")
try:
    import torch
    print("✓ torch imported")
    
    import transformers
    print("✓ transformers imported")
    
    import dspy
    print("✓ dspy imported")
    
    from dense_lm import DenseLM
    print("✓ DenseLM imported")
    
    from dense_edge import DenseEdge
    print("✓ DenseEdge imported")
    
    from plansolve import DensePlanSolve, PromptPlanSolve
    print("✓ PlanSolve modules imported")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: DenseEdge initialization
print("\n2. Testing DenseEdge...")
try:
    edge = DenseEdge(d_model=896)
    params = edge.get_num_params()
    print(f"✓ DenseEdge initialized with {params:,} parameters")
    
    # Test forward pass with dummy tensor
    dummy_input = torch.randn(10, 896)  # [T=10, d=896]
    output = edge(dummy_input)
    assert output.shape == dummy_input.shape
    print(f"✓ Forward pass successful: {dummy_input.shape} -> {output.shape}")
    
except Exception as e:
    print(f"✗ DenseEdge error: {e}")

# Test 3: Check for trained weights
print("\n3. Checking for trained edge weights...")
import os
if os.path.exists("edge_state_dict.pt"):
    print("✓ Trained weights found: edge_state_dict.pt")
else:
    print("⚠ No trained weights found (will use random initialization)")

# Test 4: Environment variables
print("\n4. Checking environment...")
groq_key = os.environ.get("GROQ_API_KEY")
if groq_key:
    print(f"✓ GROQ_API_KEY set (length: {len(groq_key)})")
else:
    print("⚠ GROQ_API_KEY not found in environment")

# Test 5: Device availability
print("\n5. Checking compute devices...")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ CUDA not available, will use CPU")

print("\n" + "=" * 50)
print("Basic component tests completed!")
print("=" * 50)