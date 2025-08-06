"""Option 1: End-to-end test with MockLM - demonstrates dense communication without model loading."""

import torch
import time
from mock_lm import MockLM
from dense_edge import DenseEdge
from plansolve import DensePlanSolve, PromptPlanSolve
import json

print("=" * 60)
print("Option 1: Mock LM End-to-End Test")
print("=" * 60)

# Test problems
test_problems = [
    "What is 5 + 3?",
    "John has 8 apples and gives 3 to Mary. How many apples does John have?",
    "Calculate 12 - 7",
    "If I have 4 groups of 6 items each, how many items total?",
    "Sarah had 15 cookies and ate 6. How many are left?",
    "What is 9 * 3?",
    "Divide 20 by 4",
    "Tom has 10 marbles and finds 7 more. How many does he have now?",
    "What is 25 - 13?",
    "Calculate 6 + 9",
]

print(f"\nTesting {len(test_problems)} math problems...")
print("-" * 60)

# Initialize modules with MockLM
print("\n1. Initializing modules with MockLM...")
device = "cpu"  # Use CPU for consistency

# Create mock LMs
mock_planner = MockLM(device=device)
mock_solver = MockLM(device=device)

# Create modules
dense_module = DensePlanSolve(
    planner_lm=mock_planner,
    solver_lm=mock_solver,
    device=device,
    share_lm=False  # Use separate instances
)

prompt_module = PromptPlanSolve(lm=MockLM(device=device))

print("✓ Modules initialized")

# Test both approaches
results = {
    "dense": {"correct": 0, "total_tokens": 0, "times": []},
    "prompt": {"correct": 0, "total_tokens": 0, "times": []}
}

print("\n2. Running inference tests...")
print("-" * 60)

for i, problem in enumerate(test_problems):
    print(f"\nProblem {i+1}: {problem}")
    
    # Test Dense approach
    start_time = time.time()
    try:
        dense_output = dense_module(goal=problem, use_dense=True)
        dense_time = time.time() - start_time
        
        # Count tokens in dense mode (only final decode)
        dense_tokens = len(dense_output["solution"].split())
        results["dense"]["total_tokens"] += dense_tokens
        results["dense"]["times"].append(dense_time)
        
        print(f"  Dense solution: {dense_output['solution'][:50]}...")
        print(f"  Dense tokens: {dense_tokens}")
        
        # Show hidden state info
        h_plan = dense_output["hidden_states"]["h_plan"]
        h_transformed = dense_output["hidden_states"]["h_transformed"]
        print(f"  Hidden states: plan {h_plan.shape} → transformed {h_transformed.shape}")
        
    except Exception as e:
        print(f"  Dense error: {e}")
        dense_time = 0
        dense_tokens = 0
    
    # Test Prompt approach
    start_time = time.time()
    try:
        prompt_output = prompt_module(goal=problem)
        prompt_time = time.time() - start_time
        
        # Count tokens in prompt mode (plan + solution)
        plan_tokens = len(prompt_output.get("plan", "").split())
        solution_tokens = len(prompt_output["solution"].split())
        prompt_tokens = plan_tokens + solution_tokens
        results["prompt"]["total_tokens"] += prompt_tokens
        results["prompt"]["times"].append(prompt_time)
        
        print(f"  Prompt solution: {prompt_output['solution'][:50]}...")
        print(f"  Prompt tokens: {prompt_tokens} (plan: {plan_tokens}, solution: {solution_tokens})")
        
    except Exception as e:
        print(f"  Prompt error: {e}")
        prompt_time = 0
        prompt_tokens = 0

# Calculate statistics
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

# Token reduction
dense_avg_tokens = results["dense"]["total_tokens"] / len(test_problems)
prompt_avg_tokens = results["prompt"]["total_tokens"] / len(test_problems)
token_reduction = 1 - (dense_avg_tokens / prompt_avg_tokens) if prompt_avg_tokens > 0 else 0

print(f"\nToken Usage:")
print(f"  Dense average: {dense_avg_tokens:.1f} tokens/problem")
print(f"  Prompt average: {prompt_avg_tokens:.1f} tokens/problem")
print(f"  Token reduction: {token_reduction:.1%}")
print(f"  Dense uses {dense_avg_tokens/prompt_avg_tokens:.1%} of baseline tokens")

# Speed comparison
dense_avg_time = sum(results["dense"]["times"]) / len(results["dense"]["times"])
prompt_avg_time = sum(results["prompt"]["times"]) / len(results["prompt"]["times"])

print(f"\nInference Speed:")
print(f"  Dense average: {dense_avg_time*1000:.1f}ms/problem")
print(f"  Prompt average: {prompt_avg_time*1000:.1f}ms/problem")
print(f"  Speed ratio: {prompt_avg_time/dense_avg_time:.2f}x")

# Success check
success_criteria = {
    "token_reduction": token_reduction >= 0.9,  # ≥90% reduction (≤10% of baseline)
    "inference_works": len(results["dense"]["times"]) == len(test_problems)
}

print(f"\nSuccess Criteria:")
print(f"  ✓ Token reduction ≥90%: {'✓' if success_criteria['token_reduction'] else '✗'}")
print(f"  ✓ All inferences complete: {'✓' if success_criteria['inference_works'] else '✗'}")

# Save results
results_data = {
    "test_problems": test_problems,
    "dense": results["dense"],
    "prompt": results["prompt"],
    "statistics": {
        "dense_avg_tokens": dense_avg_tokens,
        "prompt_avg_tokens": prompt_avg_tokens,
        "token_reduction": token_reduction,
        "dense_avg_time_ms": dense_avg_time * 1000,
        "prompt_avg_time_ms": prompt_avg_time * 1000,
    },
    "success_criteria": success_criteria
}

with open("mock_lm_test_results.json", "w") as f:
    json.dump(results_data, f, indent=2)

print("\nResults saved to: mock_lm_test_results.json")

# Demonstrate hidden state flow
print("\n" + "=" * 60)
print("HIDDEN STATE FLOW DEMONSTRATION")
print("=" * 60)

example_problem = "What is 7 + 5?"
print(f"\nExample: {example_problem}")

# Show the flow
h_plan = mock_planner.encode(f"Problem: {example_problem}\nLet me create a step-by-step plan:")
print(f"1. Planner encodes → Hidden states: {h_plan.shape}")

h_transformed = dense_module.edge(h_plan)
print(f"2. Edge transforms → Hidden states: {h_transformed.shape}")

solution = mock_solver.decode(h_transformed)
print(f"3. Solver decodes → Solution: {solution[:80]}...")

print("\n" + "=" * 60)
print("Mock LM test completed successfully!")
print("=" * 60)