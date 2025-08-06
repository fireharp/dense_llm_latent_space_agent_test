"""Baseline ReACT: Traditional text-based Reasoning and Acting."""

from typing import Dict, Any, List
import re
from mock_lm import MockLM


class BaselineReACT:
    """Traditional ReACT implementation with full text generation at each step."""
    
    def __init__(
        self,
        lm: MockLM,
        max_iterations: int = 5,
        verbose: bool = True
    ):
        """Initialize Baseline ReACT.
        
        Args:
            lm: Language model for all components
            max_iterations: Maximum reasoning iterations
            verbose: Whether to print intermediate steps
        """
        self.lm = lm
        self.max_iterations = max_iterations
        self.verbose = verbose
        
    def forward(self, problem: str) -> Dict[str, Any]:
        """Execute ReACT loop with text generation at each step.
        
        Args:
            problem: The problem to solve
            
        Returns:
            Dictionary with solution and token counts
        """
        total_tokens = 0
        trace = {
            "iterations": [],
            "all_generated_text": []
        }
        
        # Initial state
        context = f"Problem: {problem}"
        
        for i in range(self.max_iterations):
            iteration_info = {"iteration": i + 1}
            
            # Step 1: Generate thought
            thought_prompt = f"{context}\nThought: Let me think about this problem step by step."
            thought = self.lm.basic_request(thought_prompt)
            thought_tokens = len(thought.split())
            total_tokens += thought_tokens
            trace["all_generated_text"].append(f"Thought: {thought}")
            
            if self.verbose:
                print(f"\n[Iteration {i+1}] Thought: {thought[:100]}...")
            
            # Step 2: Decide on action
            action_prompt = f"{thought_prompt}\n{thought}\nAction: Based on my thinking, I need to"
            action = self.lm.basic_request(action_prompt)
            action_tokens = len(action.split())
            total_tokens += action_tokens
            trace["all_generated_text"].append(f"Action: {action}")
            
            if self.verbose:
                print(f"Action: {action[:100]}...")
            
            # Step 3: Execute action and get observation
            observation = self._execute_action(action, problem)
            obs_prompt = f"Observation: {observation}"
            # Even observation formatting counts as tokens in traditional approach
            obs_tokens = len(observation.split())
            total_tokens += obs_tokens
            trace["all_generated_text"].append(f"Observation: {observation}")
            
            if self.verbose:
                print(f"Observation: {observation}")
            
            # Step 4: Check if we can answer
            answer_check_prompt = f"{context}\n{thought}\n{action}\n{obs_prompt}\nCan I answer now? Let me check if I have enough information."
            answer_check = self.lm.basic_request(answer_check_prompt)
            check_tokens = len(answer_check.split())
            total_tokens += check_tokens
            trace["all_generated_text"].append(f"Check: {answer_check}")
            
            # Step 5: Generate answer if ready
            if "yes" in answer_check.lower() or "enough" in answer_check.lower() or i == self.max_iterations - 1:
                answer_prompt = f"{context}\nBased on my reasoning and observations, the answer is:"
                answer = self.lm.basic_request(answer_prompt)
                answer_tokens = len(answer.split())
                total_tokens += answer_tokens
                trace["all_generated_text"].append(f"Answer: {answer}")
                
                iteration_info.update({
                    "thought_tokens": thought_tokens,
                    "action_tokens": action_tokens,
                    "obs_tokens": obs_tokens,
                    "check_tokens": check_tokens,
                    "answer_tokens": answer_tokens,
                    "iteration_total": thought_tokens + action_tokens + obs_tokens + check_tokens + answer_tokens
                })
                trace["iterations"].append(iteration_info)
                
                return {
                    "answer": answer,
                    "iterations": i + 1,
                    "total_tokens": total_tokens,
                    "trace": trace,
                    "breakdown": {
                        "thoughts": sum(it.get("thought_tokens", 0) for it in trace["iterations"]),
                        "actions": sum(it.get("action_tokens", 0) for it in trace["iterations"]),
                        "observations": sum(it.get("obs_tokens", 0) for it in trace["iterations"]),
                        "checks": sum(it.get("check_tokens", 0) for it in trace["iterations"]),
                        "final_answer": answer_tokens
                    }
                }
            
            # Update context for next iteration
            context = f"{context}\n{thought}\n{action}\n{obs_prompt}"
            
            iteration_info.update({
                "thought_tokens": thought_tokens,
                "action_tokens": action_tokens,
                "obs_tokens": obs_tokens,
                "check_tokens": check_tokens,
                "iteration_total": thought_tokens + action_tokens + obs_tokens + check_tokens
            })
            trace["iterations"].append(iteration_info)
        
        # Shouldn't reach here, but just in case
        final_answer = "Unable to determine answer within iteration limit."
        final_tokens = len(final_answer.split())
        total_tokens += final_tokens
        
        return {
            "answer": final_answer,
            "iterations": self.max_iterations,
            "total_tokens": total_tokens,
            "trace": trace
        }
    
    def _execute_action(self, action: str, problem: str) -> str:
        """Execute action and return observation.
        
        Same logic as dense version for fair comparison.
        """
        action_lower = action.lower()
        
        if "calculat" in action_lower or "comput" in action_lower or "add" in action_lower or "subtract" in action_lower:
            # Look for math expressions
            numbers = re.findall(r'\d+', problem)
            if len(numbers) >= 2:
                a, b = int(numbers[0]), int(numbers[1])
                if "+" in problem or "sum" in problem or "total" in problem or "add" in action_lower:
                    return f"I calculated {a} + {b} = {a + b}"
                elif "-" in problem or "subtract" in problem or "left" in problem:
                    return f"I calculated {a} - {b} = {a - b}"
                elif "*" in problem or "multiply" in problem or "product" in problem:
                    return f"I calculated {a} * {b} = {a * b}"
                elif "/" in problem or "divide" in problem:
                    return f"I calculated {a} / {b} = {a // b}"
            return "I need to identify what calculation to perform"
            
        elif "extract" in action_lower or "find" in action_lower or "identify" in action_lower:
            numbers = re.findall(r'\d+', problem)
            if numbers:
                return f"I found these numbers in the problem: {', '.join(numbers)}"
            return "I couldn't find any numbers in the problem"
            
        elif "analyz" in action_lower or "understand" in action_lower:
            if "step" in problem.lower():
                return "This appears to be a multi-step problem that requires sequential reasoning"
            else:
                return "This appears to be a single-step problem that can be solved directly"
                
        else:
            return "I need to think more about what action to take"


if __name__ == "__main__":
    print("=" * 60)
    print("Baseline ReACT Test")
    print("=" * 60)
    
    # Create baseline ReACT
    baseline = BaselineReACT(lm=MockLM(), verbose=True)
    
    # Test problems
    test_problems = [
        "What is 15 + 27?",
        "John has 8 apples and buys 5 more. How many apples does he have?",
        "Calculate 45 - 18",
    ]
    
    for problem in test_problems:
        print(f"\n{'='*60}")
        print(f"Problem: {problem}")
        print(f"{'='*60}")
        
        result = baseline(problem)
        
        print(f"\nFinal Answer: {result['answer']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Total tokens (baseline): {result['total_tokens']}")
        
        # Show token breakdown
        if 'breakdown' in result:
            print("\nToken Breakdown:")
            for category, count in result['breakdown'].items():
                print(f"  {category}: {count} tokens")
        
        # Show per-iteration tokens
        print("\nPer-iteration tokens:")
        for iter_info in result['trace']['iterations']:
            print(f"  Iteration {iter_info['iteration']}: {iter_info.get('iteration_total', 0)} tokens")
    
    print("\n" + "=" * 60)
    print("Baseline ReACT: Every intermediate step generates tokens!")
    print("=" * 60)