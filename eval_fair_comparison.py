"""Fair evaluation comparing Dense vs Baseline with same base model and hidden state measurement."""

import torch
import time
import json
import numpy as np
from typing import Dict, Any, List

# Use real DenseLM for both approaches
from dense_lm import DenseLM
from dense_edge import DenseEdge


class FairDenseReACT:
    """Dense ReACT using real DenseLM with hidden state tracking."""
    
    def __init__(self, device: str = "cpu"):
        # Use real DenseLM
        self.reasoner = DenseLM(device=device)
        self.actor = DenseLM(device=device)
        self.observer = DenseLM(device=device)
        
        # Edge modules
        hidden_dim = self.reasoner.hidden_size
        self.thought_edge = DenseEdge(d_model=hidden_dim).to(device)
        self.action_edge = DenseEdge(d_model=hidden_dim).to(device)
        self.obs_edge = DenseEdge(d_model=hidden_dim).to(device)
        
        self.device = device
        self.max_iterations = 3
        
    def forward(self, problem: str) -> Dict[str, Any]:
        """Execute with hidden state tracking."""
        trace = {
            "iterations": [],
            "hidden_state_info": [],
            "total_hidden_elements": 0,
            "final_tokens": 0
        }
        
        # Initial encoding
        h_thought = self.reasoner.encode(f"Problem: {problem}\nThought:")
        trace["hidden_state_info"].append({
            "stage": "initial_thought",
            "shape": list(h_thought.shape),
            "elements": h_thought.numel()
        })
        trace["total_hidden_elements"] += h_thought.numel()
        
        for i in range(self.max_iterations):
            # Transform through edges (all in hidden space)
            h_action_space = self.thought_edge(h_thought)
            trace["total_hidden_elements"] += h_action_space.numel()
            
            h_action = self.action_edge(h_action_space)
            trace["total_hidden_elements"] += h_action.numel()
            
            # Simulate observation
            h_observation = self.observer.encode("Observation: Calculating...")
            trace["total_hidden_elements"] += h_observation.numel()
            
            h_new_thought = self.obs_edge(h_observation)
            trace["total_hidden_elements"] += h_new_thought.numel()
            
            # Update thought
            min_len = min(h_thought.size(0), h_new_thought.size(0))
            h_thought = h_thought[:min_len] + 0.5 * h_new_thought[:min_len]
            
            trace["iterations"].append({
                "iteration": i + 1,
                "hidden_shapes": {
                    "thought": list(h_thought.shape),
                    "action": list(h_action.shape),
                    "observation": list(h_observation.shape)
                },
                "hidden_elements": h_thought.numel() + h_action.numel() + h_observation.numel()
            })
            
            # Check if we can answer (simplified)
            if i == self.max_iterations - 1:
                # Only decode at the very end
                answer = self.reasoner.decode(h_thought)
                trace["final_tokens"] = len(answer.split())
                
                return {
                    "answer": answer,
                    "trace": trace,
                    "total_tokens": trace["final_tokens"],  # Only final decode
                    "total_hidden_elements": trace["total_hidden_elements"],
                    "hidden_to_token_ratio": trace["total_hidden_elements"] / max(trace["final_tokens"], 1)
                }


class FairBaselineReACT:
    """Baseline ReACT using same DenseLM but generating text at each step."""
    
    def __init__(self, device: str = "cpu"):
        # Use same DenseLM as dense version
        self.lm = DenseLM(device=device)
        self.device = device
        self.max_iterations = 3
        
    def forward(self, problem: str) -> Dict[str, Any]:
        """Execute with text generation at each step."""
        total_tokens = 0
        trace = {
            "iterations": [],
            "all_text": []
        }
        
        context = f"Problem: {problem}"
        
        for i in range(self.max_iterations):
            # Generate thought (text)
            thought = self.lm.basic_request(f"{context}\nThought: I need to")
            thought_tokens = len(thought.split())
            total_tokens += thought_tokens
            trace["all_text"].append(f"Thought: {thought}")
            
            # Generate action (text)
            action = self.lm.basic_request(f"{context}\n{thought}\nAction: I will")
            action_tokens = len(action.split())
            total_tokens += action_tokens
            trace["all_text"].append(f"Action: {action}")
            
            # Generate observation (text)
            observation = "Observation: Calculated the result"
            obs_tokens = len(observation.split())
            total_tokens += obs_tokens
            trace["all_text"].append(observation)
            
            trace["iterations"].append({
                "iteration": i + 1,
                "tokens": {
                    "thought": thought_tokens,
                    "action": action_tokens,
                    "observation": obs_tokens,
                    "iteration_total": thought_tokens + action_tokens + obs_tokens
                }
            })
            
            context = f"{context}\n{thought}\n{action}\n{observation}"
        
        # Final answer
        answer = self.lm.basic_request(f"{context}\nFinal answer:")
        answer_tokens = len(answer.split())
        total_tokens += answer_tokens
        
        return {
            "answer": answer,
            "trace": trace,
            "total_tokens": total_tokens,
            "token_breakdown": {
                "intermediate": total_tokens - answer_tokens,
                "final": answer_tokens
            }
        }


def compare_approaches(problem: str, device: str = "cpu"):
    """Fair comparison with hidden state measurement."""
    print("=" * 70)
    print(f"Problem: {problem}")
    print("=" * 70)
    
    # Initialize both with same base model
    print("\nInitializing with same base model (DenseLM/Qwen)...")
    
    # Test Dense
    print("\n1. DENSE APPROACH:")
    dense_system = FairDenseReACT(device=device)
    start = time.time()
    dense_result = dense_system.forward(problem)
    dense_time = time.time() - start
    
    print(f"   Final answer: {dense_result['answer'][:50]}...")
    print(f"   Total tokens generated: {dense_result['total_tokens']} (only final decode)")
    print(f"   Total hidden elements processed: {dense_result['total_hidden_elements']:,}")
    print(f"   Time: {dense_time*1000:.1f}ms")
    
    # Show hidden state flow
    print("\n   Hidden state flow:")
    for iter_info in dense_result['trace']['iterations']:
        print(f"   - Iteration {iter_info['iteration']}: {iter_info['hidden_elements']:,} elements")
        for stage, shape in iter_info['hidden_shapes'].items():
            print(f"     {stage}: {shape}")
    
    # Test Baseline
    print("\n2. BASELINE APPROACH:")
    baseline_system = FairBaselineReACT(device=device)
    start = time.time()
    baseline_result = baseline_system.forward(problem)
    baseline_time = time.time() - start
    
    print(f"   Final answer: {baseline_result['answer'][:50]}...")
    print(f"   Total tokens generated: {baseline_result['total_tokens']}")
    print(f"   - Intermediate steps: {baseline_result['token_breakdown']['intermediate']} tokens")
    print(f"   - Final answer: {baseline_result['token_breakdown']['final']} tokens")
    print(f"   Time: {baseline_time*1000:.1f}ms")
    
    # Show token generation at each step
    print("\n   Token generation per iteration:")
    for iter_info in baseline_result['trace']['iterations']:
        tokens = iter_info['tokens']
        print(f"   - Iteration {iter_info['iteration']}: {tokens['iteration_total']} tokens")
        print(f"     (thought: {tokens['thought']}, action: {tokens['action']}, obs: {tokens['observation']})")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    token_reduction = 1 - (dense_result['total_tokens'] / baseline_result['total_tokens'])
    
    print(f"\nToken Generation:")
    print(f"  Dense: {dense_result['total_tokens']} tokens (final only)")
    print(f"  Baseline: {baseline_result['total_tokens']} tokens (all steps)")
    print(f"  Reduction: {token_reduction:.1%}")
    
    print(f"\nInformation Flow:")
    print(f"  Dense: {dense_result['total_hidden_elements']:,} hidden elements")
    print(f"  Baseline: {baseline_result['total_tokens']} tokens × ~{dense_system.reasoner.hidden_size} dims = "
          f"~{baseline_result['total_tokens'] * dense_system.reasoner.hidden_size:,} equivalent elements")
    
    # Hidden state to token ratio
    hidden_per_token = dense_result['total_hidden_elements'] / max(baseline_result['total_tokens'], 1)
    print(f"\nEfficiency:")
    print(f"  Dense processes ~{hidden_per_token:.0f} hidden elements per baseline token")
    print(f"  But only generates {dense_result['total_tokens']} actual tokens!")
    
    print(f"\nSpeed:")
    print(f"  Dense: {dense_time*1000:.1f}ms")
    print(f"  Baseline: {baseline_time*1000:.1f}ms")
    if baseline_time > 0:
        print(f"  Ratio: {dense_time/baseline_time:.2f}x")
    
    return {
        "dense": dense_result,
        "baseline": baseline_result,
        "token_reduction": token_reduction,
        "timing": {"dense": dense_time, "baseline": baseline_time}
    }


def run_cot_comparison(problem: str, device: str = "cpu"):
    """Compare Chain-of-Thought approaches."""
    print("\n" + "=" * 70)
    print("Chain-of-Thought Comparison")
    print("=" * 70)
    
    # Dense CoT
    from dense_cot import create_dense_cot_system
    dense_cot = create_dense_cot_system(num_steps=3, device=device)
    
    # For baseline, we'll simulate with DenseLM
    baseline_lm = DenseLM(device=device)
    
    print(f"\nProblem: {problem}")
    
    # Dense CoT
    start = time.time()
    h = baseline_lm.encode(f"Problem: {problem}\nStep 1:")
    hidden_elements = h.numel()
    
    for step in range(3):
        h = dense_cot.step_edges[step](h)
        hidden_elements += h.numel()
    
    answer = baseline_lm.decode(h)
    dense_time = time.time() - start
    dense_tokens = len(answer.split())
    
    print(f"\nDense CoT:")
    print(f"  Hidden elements: {hidden_elements:,}")
    print(f"  Final tokens: {dense_tokens}")
    
    # Baseline CoT
    start = time.time()
    total_tokens = 0
    context = f"Problem: {problem}"
    
    for step in range(3):
        step_text = baseline_lm.basic_request(f"{context}\nStep {step+1}:")
        total_tokens += len(step_text.split())
        context += f"\nStep {step+1}: {step_text}"
    
    final = baseline_lm.basic_request(f"{context}\nTherefore:")
    total_tokens += len(final.split())
    baseline_time = time.time() - start
    
    print(f"\nBaseline CoT:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Token reduction: {1 - (dense_tokens/total_tokens):.1%}")


if __name__ == "__main__":
    # Test problems
    test_problems = [
        "What is 15 + 27?",
        "John has 8 apples, buys 5 more, and eats 2. How many does he have?",
    ]
    
    print("FAIR COMPARISON: Same Base Model (DenseLM/Qwen)")
    print("=" * 70)
    print("Both approaches use the same model, but:")
    print("- Dense: Passes hidden states, only decodes final answer")  
    print("- Baseline: Generates text at every step")
    print("=" * 70)
    
    device = "cpu"  # Change to "cuda" if available
    
    # ReACT comparison
    for problem in test_problems:
        results = compare_approaches(problem, device)
        
    # CoT comparison
    print("\n\n")
    for problem in test_problems[:1]:
        run_cot_comparison(problem, device)