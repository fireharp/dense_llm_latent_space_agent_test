"""Stage 2: Integration testing without full model loading."""

import torch
import sys
import os

print("=" * 50)
print("Stage 2: Integration Testing")
print("=" * 50)

# Test module interactions without loading the full LLM
print("\n1. Testing DenseEdge integration...")
try:
    from dense_edge import DenseEdge
    from dense_lm import DenseLM
    
    # Create edge module
    edge = DenseEdge(d_model=896)
    
    # Simulate hidden states from planner
    mock_hidden_states = torch.randn(20, 896)  # [T=20, d=896]
    print(f"✓ Input hidden states shape: {mock_hidden_states.shape}")
    
    # Transform through edge
    transformed = edge(mock_hidden_states)
    print(f"✓ Transformed shape: {transformed.shape}")
    
    # Verify shape preservation
    assert transformed.shape == mock_hidden_states.shape
    print("✓ Shape preserved through edge transformation")
    
except Exception as e:
    print(f"✗ Edge integration error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: PlanSolve module structure
print("\n2. Testing PlanSolve module structure...")
try:
    from plansolve import DensePlanSolve, PromptPlanSolve
    
    # Check class attributes
    print("✓ DensePlanSolve has methods:", [m for m in dir(DensePlanSolve) if not m.startswith('_')])
    print("✓ PromptPlanSolve has methods:", [m for m in dir(PromptPlanSolve) if not m.startswith('_')])
    
except Exception as e:
    print(f"✗ Module structure error: {e}")

# Test 3: Groq availability
print("\n3. Testing Groq integration setup...")
try:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        print(f"✓ Groq available with API key (length: {len(api_key)})")
    else:
        print("⚠ Groq available but no API key set")
except ImportError:
    print("⚠ Groq not installed")

# Test 4: DSPy configuration
print("\n4. Testing DSPy setup...")
try:
    import dspy
    print(f"✓ DSPy version: {dspy.__version__ if hasattr(dspy, '__version__') else 'unknown'}")
    
    # Test signature creation
    sig = dspy.Signature("input -> output")
    print("✓ DSPy signature creation works")
    
except Exception as e:
    print(f"✗ DSPy error: {e}")

print("\n" + "=" * 50)
print("Integration tests completed!")
print("=" * 50)