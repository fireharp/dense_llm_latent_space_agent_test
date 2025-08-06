"""Simple ReACT evaluation to show token differences."""

from dense_react import create_dense_react_system
from baseline_react import BaselineReACT
from mock_lm import MockLM
import time

print("=" * 70)
print("Simple ReACT Token Comparison")
print("=" * 70)

# Test problem
problem = "What is 25 + 17?"
print(f"\nProblem: {problem}")
print("-" * 70)

# Test Dense ReACT
print("\n1. Dense ReACT:")
dense_react = create_dense_react_system()
start = time.time()
dense_result = dense_react(problem)
dense_time = time.time() - start

print(f"   Answer: {dense_result['answer']}")
print(f"   Iterations: {dense_result['iterations']}")
print(f"   Total tokens: {dense_result['total_tokens']} (only final decode)")
print(f"   Time: {dense_time*1000:.1f}ms")

# Test Baseline ReACT
print("\n2. Baseline ReACT:")
baseline = BaselineReACT(lm=MockLM(), verbose=False)
start = time.time()
baseline_result = baseline.forward(problem)
baseline_time = time.time() - start

print(f"   Answer: {baseline_result['answer']}")
print(f"   Iterations: {baseline_result['iterations']}")
print(f"   Total tokens: {baseline_result['total_tokens']} (all intermediate steps)")
print(f"   Time: {baseline_time*1000:.1f}ms")

# Show breakdown
if 'breakdown' in baseline_result:
    print("\n   Token breakdown:")
    for category, count in baseline_result['breakdown'].items():
        print(f"     - {category}: {count} tokens")

# Calculate reduction
reduction = 1 - (dense_result['total_tokens'] / baseline_result['total_tokens'])
print(f"\n3. Comparison:")
print(f"   Token reduction: {reduction:.1%}")
print(f"   Dense uses only {(dense_result['total_tokens']/baseline_result['total_tokens']):.1%} of baseline tokens")
print(f"   Speed: {baseline_time/dense_time:.1f}x")

# Show what baseline generated
print("\n4. What Baseline Generated (sample):")
for i, text in enumerate(baseline_result['trace']['all_generated_text'][:3]):
    print(f"   Step {i+1}: {text[:60]}...")

print("\n5. Key Insight:")
print("   - Dense: Only decodes final answer (hidden state reasoning)")
print("   - Baseline: Generates text for EVERY reasoning step")
print("=" * 70)