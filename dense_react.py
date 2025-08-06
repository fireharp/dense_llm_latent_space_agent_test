"""Dense ReACT: Reasoning and Acting with hidden state communication."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import re
from mock_lm import MockLM
from dense_edge import DenseEdge
import json


class DenseReACT(nn.Module):
    """ReACT implementation with dense hidden state communication.
    
    Reasoning → Action → Observation loop entirely in hidden space.
    Only decodes at the final answer.
    """
    
    def __init__(
        self,
        reasoner: MockLM,
        actor: MockLM,
        observer: MockLM,
        thought_edge: DenseEdge,
        action_edge: DenseEdge,
        obs_edge: DenseEdge,
        max_iterations: int = 5,
        device: str = "cpu"
    ):
        """Initialize Dense ReACT.
        
        Args:
            reasoner: LM for reasoning/thinking
            actor: LM for action generation
            observer: LM for observation encoding
            thought_edge: Transforms thoughts to action space
            action_edge: Transforms actions to observation space
            obs_edge: Transforms observations back to thought space
            max_iterations: Maximum reasoning iterations
            device: Device to run on
        """
        super().__init__()
        
        self.reasoner = reasoner
        self.actor = actor
        self.observer = observer
        self.thought_edge = thought_edge
        self.action_edge = action_edge
        self.obs_edge = obs_edge
        self.max_iterations = max_iterations
        self.device = device
        
        # Move edges to device
        self.thought_edge.to(device)
        self.action_edge.to(device)
        self.obs_edge.to(device)
        
    def forward(self, problem: str) -> Dict[str, Any]:
        """Execute ReACT loop with dense communication.
        
        Args:
            problem: The problem to solve
            
        Returns:
            Dictionary with solution and trace information
        """
        # Track hidden states for analysis
        trace = {
            "iterations": [],
            "hidden_states": [],
            "final_tokens": 0
        }
        
        # Initial thought encoding
        h_thought = self.reasoner.encode(f"Problem: {problem}\nThought:")
        
        for i in range(self.max_iterations):
            iteration_info = {"iteration": i + 1}
            
            # Step 1: Transform thought to action space
            h_action_space = self.thought_edge(h_thought)
            
            # Step 2: Determine if we have enough info to answer
            # (In practice, this would be a learned stopping criterion)
            # For demo, we check hidden state properties
            confidence = torch.sigmoid(h_action_space.mean()).item()
            
            if confidence > 0.7:  # High confidence - provide answer
                # Final decode only happens here
                answer = self.actor.decode(h_action_space)
                trace["final_tokens"] = len(answer.split())
                
                return {
                    "answer": answer,
                    "iterations": i + 1,
                    "trace": trace,
                    "confident": True,
                    "total_tokens": trace["final_tokens"]  # Only final decode
                }
            
            # Step 3: Need more info - generate action in hidden space
            h_action = self.action_edge(h_action_space)
            
            # Step 4: Execute action (simulated for demo)
            action_type = self._infer_action_type(h_action)
            observation = self._execute_action(action_type, problem)
            
            # Step 5: Encode observation
            h_observation = self.observer.encode(f"Observation: {observation}")
            
            # Step 6: Transform observation back to thought space
            h_new_thought = self.obs_edge(h_observation)
            
            # Step 7: Combine with previous thought (residual connection)
            # Handle size mismatch by aligning dimensions
            min_len = min(h_thought.size(0), h_new_thought.size(0))
            if h_thought.size(0) > min_len:
                h_thought = h_thought[:min_len] + 0.5 * h_new_thought
            else:
                h_thought = h_thought + 0.5 * h_new_thought[:min_len]
            
            # Record iteration info
            iteration_info.update({
                "confidence": confidence,
                "action_type": action_type,
                "observation": observation,
                "hidden_dims": {
                    "thought": h_thought.shape,
                    "action": h_action.shape,
                    "observation": h_observation.shape
                }
            })
            trace["iterations"].append(iteration_info)
            trace["hidden_states"].append({
                "thought_norm": h_thought.norm().item(),
                "action_norm": h_action.norm().item()
            })
        
        # Max iterations reached - decode current state
        final_answer = self.reasoner.decode(h_thought)
        trace["final_tokens"] = len(final_answer.split())
        
        return {
            "answer": final_answer,
            "iterations": self.max_iterations,
            "trace": trace,
            "confident": False,
            "total_tokens": trace["final_tokens"]  # Only final decode
        }
    
    def _infer_action_type(self, h_action: torch.Tensor) -> str:
        """Infer action type from hidden state (simplified)."""
        # In practice, this would be learned
        # For demo, use hidden state statistics
        mean_activation = h_action.mean().item()
        
        if mean_activation > 0.5:
            return "calculate"
        elif mean_activation > 0:
            return "extract_number"
        else:
            return "reason"
    
    def _execute_action(self, action_type: str, problem: str) -> str:
        """Execute action and return observation (simulated)."""
        if action_type == "calculate":
            # Look for math expressions
            numbers = re.findall(r'\d+', problem)
            if len(numbers) >= 2:
                a, b = int(numbers[0]), int(numbers[1])
                if "+" in problem or "sum" in problem or "total" in problem:
                    return f"Calculation result: {a + b}"
                elif "-" in problem or "subtract" in problem or "left" in problem:
                    return f"Calculation result: {a - b}"
                elif "*" in problem or "multiply" in problem or "product" in problem:
                    return f"Calculation result: {a * b}"
                elif "/" in problem or "divide" in problem:
                    return f"Calculation result: {a // b}"
            return "No calculation needed"
            
        elif action_type == "extract_number":
            numbers = re.findall(r'\d+', problem)
            if numbers:
                return f"Found numbers: {', '.join(numbers)}"
            return "No numbers found"
            
        else:  # reason
            if "step" in problem.lower():
                return "This is a multi-step problem"
            else:
                return "This is a single-step problem"


class SimpleCalculator:
    """Simple calculator tool for ReACT."""
    
    def calculate(self, expression: str) -> float:
        """Safely evaluate mathematical expression."""
        # Remove any non-math characters
        safe_expr = re.sub(r'[^0-9+\-*/().]', '', expression)
        try:
            return eval(safe_expr)
        except:
            return 0.0


def create_dense_react_system(device: str = "cpu") -> DenseReACT:
    """Create a complete Dense ReACT system."""
    # Initialize components
    reasoner = MockLM(device=device)
    actor = MockLM(device=device)
    observer = MockLM(device=device)
    
    # Create edge modules
    hidden_dim = 896
    thought_edge = DenseEdge(d_model=hidden_dim, n_heads=8, n_layers=2)
    action_edge = DenseEdge(d_model=hidden_dim, n_heads=8, n_layers=2)
    obs_edge = DenseEdge(d_model=hidden_dim, n_heads=8, n_layers=2)
    
    return DenseReACT(
        reasoner=reasoner,
        actor=actor,
        observer=observer,
        thought_edge=thought_edge,
        action_edge=action_edge,
        obs_edge=obs_edge,
        max_iterations=5,
        device=device
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Dense ReACT Test")
    print("=" * 60)
    
    # Create system
    dense_react = create_dense_react_system()
    
    # Test problems
    test_problems = [
        "What is 15 + 27?",
        "John has 8 apples and buys 5 more. How many apples does he have?",
        "Calculate 45 - 18",
    ]
    
    for problem in test_problems:
        print(f"\nProblem: {problem}")
        result = dense_react(problem)
        
        print(f"Answer: {result['answer']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Total tokens (dense): {result['total_tokens']}")
        print(f"Confident: {result['confident']}")
        
        # Show trace
        for iter_info in result['trace']['iterations']:
            print(f"  Iteration {iter_info['iteration']}: "
                  f"action={iter_info['action_type']}, "
                  f"confidence={iter_info['confidence']:.2f}")
    
    print("\n" + "=" * 60)
    print("Dense ReACT: Only final answer is decoded!")
    print("=" * 60)