"""Option 2: Performance comparison test - quantify benefits of dense approach."""

import torch
import time
import json
import psutil
import os
from mock_lm import MockLM
from dense_edge import DenseEdge
from plansolve import DensePlanSolve, PromptPlanSolve
import numpy as np
# import matplotlib.pyplot as plt  # Optional for visualization

print("=" * 60)
print("Option 2: Performance Comparison Test")
print("=" * 60)

# Extended test set
test_problems = [
    # Simple arithmetic
    "What is 5 + 3?",
    "Calculate 12 - 7",
    "What is 6 * 4?",
    "Divide 20 by 5",
    
    # Word problems
    "John has 8 apples and gives 3 to Mary. How many apples does John have?",
    "Sarah had 15 cookies and ate 6. How many are left?",
    "Tom has 10 marbles and finds 7 more. How many does he have now?",
    "A box contains 24 items. If 8 are removed, how many remain?",
    
    # Multi-step problems
    "If I buy 3 packs of 5 pencils each, how many pencils do I have?",
    "Jane has 20 dollars. She spends 8 on lunch and 5 on a book. How much is left?",
    "A farmer has 50 chickens. He sells 15 and buys 8 more. How many chickens now?",
    "Calculate: (10 + 5) - (3 * 2)",
    
    # Comparison problems
    "Which is larger: 25 or 19?",
    "Is 7 + 8 greater than 14?",
    "How much more is 30 than 22?",
]

print(f"\nTesting {len(test_problems)} problems across different categories")

# Initialize modules
print("\n1. Initializing modules...")
device = "cpu"

# Create modules with mock LMs
dense_planner = MockLM(device=device)
dense_solver = MockLM(device=device)
dense_module = DensePlanSolve(
    planner_lm=dense_planner,
    solver_lm=dense_solver,
    device=device,
    share_lm=False
)

# For fair comparison, create a "traditional" module that generates full text
class TraditionalPlanSolve:
    def __init__(self, lm):
        self.lm = lm
        
    def forward(self, goal):
        # Simulate traditional approach with explicit planning and solving
        plan = f"To solve '{goal}', I will: 1) Identify the key information, 2) Determine the operation needed, 3) Perform the calculation, 4) Verify the result"
        solution = self.lm.basic_request(f"Problem: {goal}\nPlan: {plan}\nNow solve it step by step.")
        
        return {
            "plan": plan,
            "solution": solution,
            "total_tokens": len(plan.split()) + len(solution.split())
        }

traditional_module = TraditionalPlanSolve(MockLM(device=device))

print("✓ Modules initialized")

# Performance metrics storage
metrics = {
    "dense": {
        "tokens": [],
        "times": [],
        "memory": []
    },
    "traditional": {
        "tokens": [],
        "times": [],
        "memory": []
    }
}

# Get process for memory monitoring
process = psutil.Process(os.getpid())

print("\n2. Running performance comparison...")
print("-" * 60)

for i, problem in enumerate(test_problems):
    print(f"\n[{i+1}/{len(test_problems)}] {problem}")
    
    # Measure dense approach
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    # Dense pipeline
    h_plan = dense_planner.encode(f"Problem: {problem}\nPlan:")
    h_transformed = dense_module.edge(h_plan)
    dense_solution = dense_solver.decode(h_transformed)
    
    dense_time = time.time() - start_time
    dense_memory = (process.memory_info().rss / 1024 / 1024) - initial_memory
    dense_tokens = len(dense_solution.split())
    
    metrics["dense"]["tokens"].append(dense_tokens)
    metrics["dense"]["times"].append(dense_time * 1000)  # Convert to ms
    metrics["dense"]["memory"].append(max(0, dense_memory))  # Avoid negative
    
    # Measure traditional approach
    initial_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()
    
    trad_output = traditional_module.forward(problem)
    
    trad_time = time.time() - start_time
    trad_memory = (process.memory_info().rss / 1024 / 1024) - initial_memory
    trad_tokens = trad_output["total_tokens"]
    
    metrics["traditional"]["tokens"].append(trad_tokens)
    metrics["traditional"]["times"].append(trad_time * 1000)
    metrics["traditional"]["memory"].append(max(0, trad_memory))
    
    # Show comparison
    token_reduction = 1 - (dense_tokens / trad_tokens)
    speed_ratio = trad_time / dense_time if dense_time > 0 else 1
    
    print(f"  Tokens: Dense={dense_tokens}, Trad={trad_tokens} (↓{token_reduction:.0%})")
    print(f"  Time: Dense={dense_time*1000:.1f}ms, Trad={trad_time*1000:.1f}ms ({speed_ratio:.1f}x)")

# Calculate statistics
print("\n" + "=" * 60)
print("PERFORMANCE ANALYSIS")
print("=" * 60)

# Token statistics
dense_avg_tokens = np.mean(metrics["dense"]["tokens"])
trad_avg_tokens = np.mean(metrics["traditional"]["tokens"])
token_reduction = 1 - (dense_avg_tokens / trad_avg_tokens)

print(f"\n📊 Token Usage:")
print(f"  Dense: {dense_avg_tokens:.1f} tokens/problem (std: {np.std(metrics['dense']['tokens']):.1f})")
print(f"  Traditional: {trad_avg_tokens:.1f} tokens/problem (std: {np.std(metrics['traditional']['tokens']):.1f})")
print(f"  Reduction: {token_reduction:.1%}")
print(f"  Dense uses {(dense_avg_tokens/trad_avg_tokens):.1%} of baseline tokens")

# Time statistics
dense_avg_time = np.mean(metrics["dense"]["times"])
trad_avg_time = np.mean(metrics["traditional"]["times"])
speed_ratio = trad_avg_time / dense_avg_time

print(f"\n⏱️  Inference Speed:")
print(f"  Dense: {dense_avg_time:.1f}ms/problem (std: {np.std(metrics['dense']['times']):.1f})")
print(f"  Traditional: {trad_avg_time:.1f}ms/problem (std: {np.std(metrics['traditional']['times']):.1f})")
print(f"  Speed ratio: {speed_ratio:.2f}x")

# Memory statistics
dense_avg_memory = np.mean(metrics["dense"]["memory"])
trad_avg_memory = np.mean(metrics["traditional"]["memory"])

print(f"\n💾 Memory Usage:")
print(f"  Dense: {dense_avg_memory:.2f}MB/problem")
print(f"  Traditional: {trad_avg_memory:.2f}MB/problem")

# Problem complexity analysis
print("\n📈 Complexity Analysis:")
problem_lengths = [len(p.split()) for p in test_problems]
for i in range(0, len(test_problems), 5):
    subset = test_problems[i:i+5]
    subset_dense_tokens = metrics["dense"]["tokens"][i:i+5]
    subset_trad_tokens = metrics["traditional"]["tokens"][i:i+5]
    
    avg_dense = np.mean(subset_dense_tokens)
    avg_trad = np.mean(subset_trad_tokens)
    reduction = 1 - (avg_dense / avg_trad)
    
    category = ["Simple arithmetic", "Word problems", "Multi-step", "Comparison"][i//5] if i//5 < 4 else "Other"
    print(f"  {category}: {reduction:.0%} token reduction")

# Success criteria check
print("\n" + "=" * 60)
print("SUCCESS CRITERIA")
print("=" * 60)

criteria = {
    "token_reduction_90": token_reduction >= 0.9,
    "token_reduction_target": dense_avg_tokens <= 0.1 * trad_avg_tokens,
    "faster_inference": dense_avg_time < trad_avg_time,
    "consistent_reduction": all(d < t for d, t in zip(metrics["dense"]["tokens"], metrics["traditional"]["tokens"]))
}

for criterion, passed in criteria.items():
    print(f"  {'✓' if passed else '✗'} {criterion.replace('_', ' ').title()}: {'PASS' if passed else 'FAIL'}")

overall_success = all(criteria.values())
print(f"\nOVERALL: {'SUCCESS' if overall_success else 'PARTIAL SUCCESS'}")

# Save detailed results
results_data = {
    "test_problems": test_problems,
    "metrics": metrics,
    "statistics": {
        "token": {
            "dense_avg": dense_avg_tokens,
            "traditional_avg": trad_avg_tokens,
            "reduction": token_reduction,
            "ratio": dense_avg_tokens / trad_avg_tokens
        },
        "time": {
            "dense_avg_ms": dense_avg_time,
            "traditional_avg_ms": trad_avg_time,
            "speed_ratio": speed_ratio
        },
        "memory": {
            "dense_avg_mb": dense_avg_memory,
            "traditional_avg_mb": trad_avg_memory
        }
    },
    "criteria": {k: bool(v) for k, v in criteria.items()},  # Convert numpy bool to Python bool
    "overall_success": bool(overall_success)
}

with open("performance_test_results.json", "w") as f:
    json.dump(results_data, f, indent=2)

print("\n✓ Detailed results saved to performance_test_results.json")

# Create simple text visualization
print("\n" + "=" * 60)
print("VISUAL COMPARISON")
print("=" * 60)

# Token usage bar chart (text-based)
print("\nToken Usage Comparison:")
max_tokens = max(max(metrics["traditional"]["tokens"]), max(metrics["dense"]["tokens"]))
scale = 50 / max_tokens  # Scale to 50 chars width

for i in range(min(5, len(test_problems))):  # Show first 5 problems
    trad_bar = "█" * int(metrics["traditional"]["tokens"][i] * scale)
    dense_bar = "█" * int(metrics["dense"]["tokens"][i] * scale)
    print(f"P{i+1} Trad: {trad_bar} ({metrics['traditional']['tokens'][i]})")
    print(f"    Dense: {dense_bar} ({metrics['dense']['tokens'][i]})")
    print()

# Summary bars
print("\nAverage Token Usage:")
trad_avg_bar = "█" * int(trad_avg_tokens * scale)
dense_avg_bar = "█" * int(dense_avg_tokens * scale)
print(f"Traditional: {trad_avg_bar} ({trad_avg_tokens:.1f})")
print(f"Dense:       {dense_avg_bar} ({dense_avg_tokens:.1f})")
print(f"Reduction:   {token_reduction:.0%}")

print("\n" + "=" * 60)
print("Performance test completed!")
print("=" * 60)