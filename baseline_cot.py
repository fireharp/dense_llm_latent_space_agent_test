"""Baseline Chain-of-Thought: Traditional text-based step-by-step reasoning."""

from typing import Dict, Any, List
from mock_lm import MockLM


class BaselineCoT:
    """Traditional CoT implementation with text generation at each step."""
    
    def __init__(
        self,
        lm: MockLM,
        num_steps: int = 3,
        verbose: bool = False
    ):
        """Initialize Baseline CoT.
        
        Args:
            lm: Language model
            num_steps: Number of reasoning steps
            verbose: Whether to print intermediate steps
        """
        self.lm = lm
        self.num_steps = num_steps
        self.verbose = verbose
        
    def forward(self, problem: str) -> Dict[str, Any]:
        """Execute chain-of-thought with text at each step.
        
        Args:
            problem: The problem to solve
            
        Returns:
            Dictionary with solution and token counts
        """
        total_tokens = 0
        trace = {
            "steps": [],
            "all_generated_text": []
        }
        
        # Initial prompt
        context = f"Problem: {problem}\nLet's think step by step."
        
        # Generate each reasoning step
        for step in range(self.num_steps):
            step_info = {"step": step + 1}
            
            # Generate reasoning for this step
            step_prompt = f"{context}\nStep {step + 1}: Let me think about this carefully."
            step_reasoning = self.lm.basic_request(step_prompt)
            step_tokens = len(step_reasoning.split())
            total_tokens += step_tokens
            
            trace["all_generated_text"].append(f"Step {step + 1}: {step_reasoning}")
            
            if self.verbose:
                print(f"\nStep {step + 1}: {step_reasoning[:100]}...")
            
            # Update context with this step's reasoning
            context = f"{context}\nStep {step + 1}: {step_reasoning}"
            
            step_info.update({
                "reasoning": step_reasoning,
                "tokens": step_tokens
            })
            trace["steps"].append(step_info)
        
        # Generate final answer based on all steps
        answer_prompt = f"{context}\nTherefore, based on my step-by-step reasoning, the answer is:"
        final_answer = self.lm.basic_request(answer_prompt)
        answer_tokens = len(final_answer.split())
        total_tokens += answer_tokens
        
        trace["all_generated_text"].append(f"Answer: {final_answer}")
        
        return {
            "answer": final_answer,
            "num_steps": self.num_steps,
            "total_tokens": total_tokens,
            "trace": trace,
            "breakdown": {
                "step_tokens": [s["tokens"] for s in trace["steps"]],
                "total_step_tokens": sum(s["tokens"] for s in trace["steps"]),
                "final_answer_tokens": answer_tokens
            }
        }


class BaselineCoTWithBranching:
    """CoT with parallel reasoning branches (text-based)."""
    
    def __init__(
        self,
        lm: MockLM,
        num_branches: int = 3,
        verbose: bool = False
    ):
        """Initialize branching CoT."""
        self.lm = lm
        self.num_branches = num_branches
        self.verbose = verbose
        
    def forward(self, problem: str) -> Dict[str, Any]:
        """Execute parallel branches of reasoning with text."""
        total_tokens = 0
        trace = {
            "branches": [],
            "all_generated_text": []
        }
        
        # Initial prompt
        context = f"Problem: {problem}\nLet me consider multiple approaches to solve this."
        
        # Generate reasoning for each branch
        branch_reasonings = []
        for branch in range(self.num_branches):
            branch_info = {"branch": branch + 1}
            
            # Generate branch-specific reasoning
            branch_prompt = f"{context}\nApproach {branch + 1}: Let me try a different way of thinking about this."
            branch_reasoning = self.lm.basic_request(branch_prompt)
            branch_tokens = len(branch_reasoning.split())
            total_tokens += branch_tokens
            
            branch_reasonings.append(branch_reasoning)
            trace["all_generated_text"].append(f"Branch {branch + 1}: {branch_reasoning}")
            
            if self.verbose:
                print(f"\nBranch {branch + 1}: {branch_reasoning[:100]}...")
            
            branch_info.update({
                "reasoning": branch_reasoning,
                "tokens": branch_tokens
            })
            trace["branches"].append(branch_info)
        
        # Synthesize branches into final answer
        synthesis_prompt = f"{context}\n"
        for i, reasoning in enumerate(branch_reasonings):
            synthesis_prompt += f"\nApproach {i+1}: {reasoning}"
        synthesis_prompt += "\n\nConsidering all approaches, the best answer is:"
        
        final_answer = self.lm.basic_request(synthesis_prompt)
        answer_tokens = len(final_answer.split())
        total_tokens += answer_tokens
        
        trace["all_generated_text"].append(f"Final synthesis: {final_answer}")
        
        return {
            "answer": final_answer,
            "num_branches": self.num_branches,
            "total_tokens": total_tokens,
            "trace": trace,
            "breakdown": {
                "branch_tokens": [b["tokens"] for b in trace["branches"]],
                "total_branch_tokens": sum(b["tokens"] for b in trace["branches"]),
                "synthesis_tokens": answer_tokens
            }
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Baseline Chain-of-Thought Test")
    print("=" * 60)
    
    # Test problems
    test_problems = [
        "What is 25 + 17?",
        "If a train travels 60 miles in 2 hours, how fast is it going?",
        "John has 15 apples. He gives 3 to Mary and 4 to Tom. How many does he have left?",
    ]
    
    # Test standard CoT
    print("\n1. Standard Baseline CoT (3 steps):")
    cot = BaselineCoT(lm=MockLM(), num_steps=3, verbose=True)
    
    for problem in test_problems:
        print(f"\n{'='*60}")
        print(f"Problem: {problem}")
        print(f"{'='*60}")
        
        result = cot.forward(problem)
        print(f"\nFinal Answer: {result['answer']}")
        print(f"Total tokens: {result['total_tokens']}")
        print(f"Breakdown: {result['breakdown']['total_step_tokens']} step tokens + "
              f"{result['breakdown']['final_answer_tokens']} answer tokens")
    
    # Test branching CoT
    print("\n\n2. Branching Baseline CoT (3 branches):")
    branching_cot = BaselineCoTWithBranching(MockLM(), num_branches=3, verbose=False)
    
    for problem in test_problems[:1]:  # Just one example
        print(f"\nProblem: {problem}")
        result = branching_cot.forward(problem)
        print(f"Final Answer: {result['answer']}")
        print(f"Total tokens: {result['total_tokens']}")
        print(f"Breakdown: {result['breakdown']['total_branch_tokens']} branch tokens + "
              f"{result['breakdown']['synthesis_tokens']} synthesis tokens")
    
    print("\n" + "=" * 60)
    print("Baseline CoT: Every step generates text tokens!")
    print("=" * 60)