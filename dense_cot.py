"""Dense Chain-of-Thought: Multi-step reasoning in hidden space."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from mock_lm import MockLM
from dense_edge import DenseEdge


class DenseCoT(nn.Module):
    """Chain-of-Thought with dense hidden state communication.
    
    Each reasoning step happens in hidden space, with only the final
    answer decoded to text.
    """
    
    def __init__(
        self,
        reasoner: MockLM,
        step_edges: List[DenseEdge],
        num_steps: int = 3,
        device: str = "cpu"
    ):
        """Initialize Dense CoT.
        
        Args:
            reasoner: LM for reasoning
            step_edges: List of edge modules for each reasoning step
            num_steps: Number of reasoning steps
            device: Device to run on
        """
        super().__init__()
        
        self.reasoner = reasoner
        self.num_steps = num_steps
        self.device = device
        
        # Create step edges if not provided
        if len(step_edges) < num_steps:
            hidden_dim = 896
            self.step_edges = nn.ModuleList([
                DenseEdge(d_model=hidden_dim, n_heads=8, n_layers=1)
                for _ in range(num_steps)
            ])
        else:
            self.step_edges = nn.ModuleList(step_edges[:num_steps])
            
        # Move to device
        self.step_edges.to(device)
        
        # Memory accumulator
        self.memory_projection = nn.Linear(896, 896).to(device)
        
    def forward(self, problem: str) -> Dict[str, Any]:
        """Execute chain-of-thought reasoning in hidden space.
        
        Args:
            problem: The problem to solve
            
        Returns:
            Dictionary with solution and trace
        """
        trace = {
            "steps": [],
            "hidden_states": [],
            "final_tokens": 0
        }
        
        # Initial encoding
        h_current = self.reasoner.encode(f"Problem: {problem}\nLet me think step by step:")
        h_memory = torch.zeros_like(h_current)
        
        # Execute reasoning steps in hidden space
        for step in range(self.num_steps):
            step_info = {"step": step + 1}
            
            # Apply step-specific transformation
            h_transformed = self.step_edges[step](h_current)
            
            # Update memory (accumulate insights)
            h_memory = h_memory + 0.3 * self.memory_projection(h_transformed)
            
            # Combine current reasoning with memory
            h_current = h_transformed + 0.5 * h_memory
            
            # Record step information
            step_info.update({
                "hidden_norm": h_current.norm().item(),
                "memory_norm": h_memory.norm().item(),
                "hidden_shape": list(h_current.shape)
            })
            trace["steps"].append(step_info)
            trace["hidden_states"].append(h_current.mean(dim=0).detach())  # Store summary
        
        # Final answer - only decode here
        answer = self.reasoner.decode(h_current)
        trace["final_tokens"] = len(answer.split())
        
        return {
            "answer": answer,
            "num_steps": self.num_steps,
            "trace": trace,
            "total_tokens": trace["final_tokens"]  # Only final decode
        }
    
    def get_step_representations(self, problem: str) -> List[torch.Tensor]:
        """Get hidden representations at each step (for analysis)."""
        representations = []
        
        h_current = self.reasoner.encode(f"Problem: {problem}\nLet me think step by step:")
        h_memory = torch.zeros_like(h_current)
        
        for step in range(self.num_steps):
            h_transformed = self.step_edges[step](h_current)
            h_memory = h_memory + 0.3 * self.memory_projection(h_transformed)
            h_current = h_transformed + 0.5 * h_memory
            representations.append(h_current.clone())
            
        return representations


class DenseCoTWithBranching(DenseCoT):
    """CoT with parallel reasoning branches that merge."""
    
    def __init__(
        self,
        reasoner: MockLM,
        num_branches: int = 3,
        device: str = "cpu"
    ):
        """Initialize branching CoT."""
        # Create separate edges for each branch
        step_edges = [
            DenseEdge(d_model=896, n_heads=8, n_layers=1)
            for _ in range(num_branches)
        ]
        super().__init__(reasoner, step_edges, num_branches, device)
        
        # Aggregation layer for merging branches
        self.aggregator = nn.Linear(896 * num_branches, 896).to(device)
        self.num_branches = num_branches
        
    def forward(self, problem: str) -> Dict[str, Any]:
        """Execute parallel branches of reasoning."""
        trace = {
            "branches": [],
            "final_tokens": 0
        }
        
        # Initial encoding
        h_init = self.reasoner.encode(f"Problem: {problem}\nLet me consider multiple approaches:")
        
        # Process each branch in parallel
        branch_outputs = []
        for branch in range(self.num_branches):
            h_branch = self.step_edges[branch](h_init)
            branch_outputs.append(h_branch)
            
            trace["branches"].append({
                "branch": branch + 1,
                "hidden_norm": h_branch.norm().item()
            })
        
        # Aggregate branches
        # Stack and flatten for aggregation
        h_combined = torch.cat(branch_outputs, dim=-1)  # Concatenate along feature dim
        
        # Handle dimension mismatch
        if h_combined.size(-1) != 896 * self.num_branches:
            # Truncate or pad as needed
            target_size = 896 * self.num_branches
            if h_combined.size(-1) > target_size:
                h_combined = h_combined[..., :target_size]
            else:
                padding = torch.zeros(*h_combined.shape[:-1], target_size - h_combined.size(-1))
                h_combined = torch.cat([h_combined, padding], dim=-1)
        
        h_final = self.aggregator(h_combined)
        
        # Decode final answer
        answer = self.reasoner.decode(h_final)
        trace["final_tokens"] = len(answer.split())
        
        return {
            "answer": answer,
            "num_branches": self.num_branches,
            "trace": trace,
            "total_tokens": trace["final_tokens"]
        }


def create_dense_cot_system(num_steps: int = 3, device: str = "cpu") -> DenseCoT:
    """Create a Dense CoT system."""
    reasoner = MockLM(device=device)
    step_edges = [
        DenseEdge(d_model=896, n_heads=8, n_layers=1).to(device)
        for _ in range(num_steps)
    ]
    return DenseCoT(reasoner, step_edges, num_steps, device)


if __name__ == "__main__":
    print("=" * 60)
    print("Dense Chain-of-Thought Test")
    print("=" * 60)
    
    # Test problems
    test_problems = [
        "What is 25 + 17?",
        "If a train travels 60 miles in 2 hours, how fast is it going?",
        "John has 15 apples. He gives 3 to Mary and 4 to Tom. How many does he have left?",
    ]
    
    # Test standard CoT
    print("\n1. Standard Dense CoT (3 steps):")
    cot = create_dense_cot_system(num_steps=3)
    
    for problem in test_problems:
        print(f"\nProblem: {problem}")
        result = cot(problem)
        print(f"Answer: {result['answer']}")
        print(f"Steps: {result['num_steps']}")
        print(f"Total tokens: {result['total_tokens']} (only final decode)")
        
        # Show step progression
        print("Step progression:")
        for step_info in result['trace']['steps']:
            print(f"  Step {step_info['step']}: hidden_norm={step_info['hidden_norm']:.2f}")
    
    # Test branching CoT
    print("\n\n2. Branching Dense CoT (3 parallel branches):")
    branching_cot = DenseCoTWithBranching(MockLM(), num_branches=3)
    
    for problem in test_problems[:1]:  # Just one example
        print(f"\nProblem: {problem}")
        result = branching_cot(problem)
        print(f"Answer: {result['answer']}")
        print(f"Branches: {result['num_branches']}")
        print(f"Total tokens: {result['total_tokens']} (only final decode)")
        
        # Show branch info
        print("Branch processing:")
        for branch_info in result['trace']['branches']:
            print(f"  Branch {branch_info['branch']}: norm={branch_info['hidden_norm']:.2f}")
    
    print("\n" + "=" * 60)
    print("Dense CoT: All reasoning in hidden space!")
    print("=" * 60)