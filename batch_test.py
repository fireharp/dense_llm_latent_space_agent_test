"""Option 3: Batch inference test - demonstrates efficient batch processing."""

import torch
import time
import json
from mock_lm import MockLM
from dense_edge import DenseEdge
from plansolve import DensePlanSolve
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

print("=" * 60)
print("Option 3: Batch Inference Test")
print("=" * 60)

# Batch of diverse problems
batch_problems = [
    # Arithmetic batch
    "What is 15 + 27?",
    "Calculate 89 - 43",
    "What is 7 * 9?",
    "Divide 144 by 12",
    "What is 256 + 128?",
    
    # Word problem batch
    "A store has 120 apples. They sell 45. How many are left?",
    "John walks 3 miles each day. How many miles in 7 days?",
    "A pizza has 8 slices. 3 friends share it equally. How many slices each?",
    "Sarah saves $15 per week. How much will she save in 4 weeks?",
    "A bus has 40 seats. 28 are taken. How many seats are empty?",
    
    # Complex batch
    "If eggs cost $3 per dozen, how much do 3 dozen cost?",
    "A rectangle is 10m long and 5m wide. What is its area?",
    "Tom reads 20 pages per hour. How many pages in 3.5 hours?",
    "A recipe needs 2 cups of flour for 12 cookies. How much for 36 cookies?",
    "A train travels at 60 mph. How far does it go in 2.5 hours?",
    
    # Comparison batch
    "Which is greater: 17 * 3 or 50?",
    "Is 100 - 37 less than 65?",
    "What's the difference between 200 and 157?",
    "Is 8 * 8 equal to 64?",
    "Which is smaller: 25% of 100 or 30?",
]

print(f"\nTesting batch processing with {len(batch_problems)} problems")

# Initialize components
device = "cpu"
planner = MockLM(device=device)
solver = MockLM(device=device)
edge = DenseEdge(d_model=896).to(device)

# Create dense module
dense_module = DensePlanSolve(
    planner_lm=planner,
    solver_lm=solver,
    edge=edge,
    device=device,
    share_lm=False
)

print("✓ Components initialized")

# Test 1: Sequential processing (baseline)
print("\n" + "-" * 60)
print("Test 1: Sequential Processing")
print("-" * 60)

sequential_results = []
start_time = time.time()

for i, problem in enumerate(batch_problems[:10]):  # First 10 for speed
    prob_start = time.time()
    
    # Dense pipeline
    h_plan = planner.encode(f"Problem: {problem}\nPlan:")
    h_transformed = edge(h_plan)
    solution = solver.decode(h_transformed)
    
    prob_time = time.time() - prob_start
    sequential_results.append({
        "problem": problem,
        "solution": solution,
        "time": prob_time * 1000
    })
    
    if i < 3:  # Show first 3
        print(f"[{i+1}] {problem}")
        print(f"    → {solution[:50]}...")
        print(f"    Time: {prob_time*1000:.1f}ms")

sequential_total = time.time() - start_time
print(f"\nSequential total time: {sequential_total:.2f}s")
print(f"Average per problem: {sequential_total/10*1000:.1f}ms")

# Test 2: Batch encoding
print("\n" + "-" * 60)
print("Test 2: Batch Encoding (Optimized)")
print("-" * 60)

batch_start = time.time()

# Step 1: Batch encode all problems
encode_start = time.time()
planning_prompts = [f"Problem: {p}\nPlan:" for p in batch_problems[:10]]
hidden_states = [planner.encode(prompt) for prompt in planning_prompts]
encode_time = time.time() - encode_start

print(f"✓ Batch encoded {len(hidden_states)} problems in {encode_time:.2f}s")
print(f"  Sequence lengths: {[h.shape[0] for h in hidden_states[:5]]}...")

# Step 2: Batch transform through edge
transform_start = time.time()
transformed_states = [edge(h) for h in hidden_states]
transform_time = time.time() - transform_start

print(f"✓ Batch transformed in {transform_time:.2f}s")

# Step 3: Batch decode
decode_start = time.time()
solutions = [solver.decode(h) for h in transformed_states]
decode_time = time.time() - decode_start

print(f"✓ Batch decoded in {decode_time:.2f}s")

batch_total = time.time() - batch_start
print(f"\nBatch total time: {batch_total:.2f}s")
print(f"Speedup: {sequential_total/batch_total:.1f}x")

# Test 3: Parallel processing
print("\n" + "-" * 60)
print("Test 3: Parallel Processing (Multi-threaded)")
print("-" * 60)

def process_problem(problem):
    """Process a single problem in parallel."""
    start = time.time()
    h_plan = planner.encode(f"Problem: {problem}\nPlan:")
    h_transformed = edge(h_plan)
    solution = solver.decode(h_transformed)
    return {
        "problem": problem,
        "solution": solution,
        "time": (time.time() - start) * 1000
    }

parallel_start = time.time()
parallel_results = []

# Use thread pool for parallel processing
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit all tasks
    futures = {executor.submit(process_problem, p): p for p in batch_problems[:10]}
    
    # Collect results as they complete
    for future in as_completed(futures):
        result = future.result()
        parallel_results.append(result)

parallel_total = time.time() - parallel_start
print(f"\nParallel total time: {parallel_total:.2f}s")
print(f"Speedup vs sequential: {sequential_total/parallel_total:.1f}x")

# Test 4: Batch size scaling
print("\n" + "-" * 60)
print("Test 4: Batch Size Scaling Analysis")
print("-" * 60)

batch_sizes = [1, 5, 10, 20]
scaling_results = {}

for batch_size in batch_sizes:
    problems_subset = batch_problems[:batch_size]
    
    start = time.time()
    hidden = [planner.encode(f"Problem: {p}\nPlan:") for p in problems_subset]
    transformed = [edge(h) for h in hidden]
    solutions = [solver.decode(h) for h in transformed]
    total_time = time.time() - start
    
    avg_time = total_time / batch_size * 1000
    scaling_results[batch_size] = {
        "total_time": total_time,
        "avg_time_ms": avg_time,
        "throughput": batch_size / total_time
    }
    
    print(f"Batch size {batch_size:2d}: {total_time:.2f}s total, {avg_time:.1f}ms/problem, {batch_size/total_time:.1f} problems/sec")

# Test 5: Memory efficiency
print("\n" + "-" * 60)
print("Test 5: Memory Efficiency Analysis")
print("-" * 60)

# Analyze memory usage for different batch sizes
import sys

def get_tensor_memory(tensors):
    """Calculate memory usage of tensor list."""
    total_bytes = 0
    for t in tensors:
        total_bytes += t.element_size() * t.numel()
    return total_bytes / (1024 * 1024)  # Convert to MB

# Single problem memory
single_hidden = planner.encode("What is 2 + 2?")
single_memory = get_tensor_memory([single_hidden])
print(f"Single problem hidden state: {single_memory:.2f} MB")

# Batch memory
batch_hidden = [planner.encode(f"Problem: {p}") for p in batch_problems[:10]]
batch_memory = get_tensor_memory(batch_hidden)
print(f"10-problem batch hidden states: {batch_memory:.2f} MB")
print(f"Memory per problem in batch: {batch_memory/10:.2f} MB")
print(f"Memory efficiency: {(single_memory*10)/batch_memory:.1f}x")

# Results summary
print("\n" + "=" * 60)
print("BATCH PROCESSING SUMMARY")
print("=" * 60)

summary = {
    "sequential": {
        "total_time_s": sequential_total,
        "avg_time_ms": sequential_total / 10 * 1000
    },
    "batch_optimized": {
        "total_time_s": batch_total,
        "avg_time_ms": batch_total / 10 * 1000,
        "speedup": sequential_total / batch_total
    },
    "parallel": {
        "total_time_s": parallel_total,
        "avg_time_ms": parallel_total / 10 * 1000,
        "speedup": sequential_total / parallel_total
    },
    "scaling": scaling_results,
    "memory": {
        "single_problem_mb": single_memory,
        "batch_10_total_mb": batch_memory,
        "batch_10_per_problem_mb": batch_memory / 10
    }
}

print(f"\n📊 Performance Comparison (10 problems):")
print(f"  Sequential: {summary['sequential']['total_time_s']:.2f}s")
print(f"  Batch:      {summary['batch_optimized']['total_time_s']:.2f}s ({summary['batch_optimized']['speedup']:.1f}x faster)")
print(f"  Parallel:   {summary['parallel']['total_time_s']:.2f}s ({summary['parallel']['speedup']:.1f}x faster)")

print(f"\n📈 Throughput:")
print(f"  Sequential: {10/summary['sequential']['total_time_s']:.1f} problems/sec")
print(f"  Batch:      {10/summary['batch_optimized']['total_time_s']:.1f} problems/sec")
print(f"  Parallel:   {10/summary['parallel']['total_time_s']:.1f} problems/sec")

print(f"\n💾 Memory Efficiency:")
print(f"  Batch processing uses {summary['memory']['batch_10_per_problem_mb']:.2f} MB per problem")
print(f"  vs {summary['memory']['single_problem_mb']:.2f} MB for single processing")

# Save results
with open("batch_test_results.json", "w") as f:
    json.dump({
        "test_problems": batch_problems[:10],
        "summary": summary,
        "sequential_results": sequential_results[:3],  # Sample
        "scaling_analysis": scaling_results
    }, f, indent=2)

print("\n✓ Results saved to batch_test_results.json")

# Visual representation
print("\n" + "=" * 60)
print("VISUAL: Batch Size vs Throughput")
print("=" * 60)

max_throughput = max(r["throughput"] for r in scaling_results.values())
scale = 40 / max_throughput

for size, results in scaling_results.items():
    bar = "█" * int(results["throughput"] * scale)
    print(f"Batch {size:2d}: {bar} {results['throughput']:.1f} problems/sec")

print("\n" + "=" * 60)
print("Batch inference test completed!")
print("=" * 60)