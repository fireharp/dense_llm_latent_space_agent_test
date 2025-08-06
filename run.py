"""CLI runner for interactive inference with optional Groq integration."""

import argparse
import os
import sys
import torch
from typing import Optional
import time

from dense_lm import DenseLM
from dense_edge import DenseEdge
from plansolve import DensePlanSolve, PromptPlanSolve

# Optional Groq import
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq not available. Install with: pip install groq")


class GroqDecoder:
    """Wrapper for Groq API decoding."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3-8b-8192"):
        """Initialize Groq decoder.
        
        Args:
            api_key: Groq API key (will use GROQ_API_KEY env var if not provided)
            model: Groq model to use
        """
        if not GROQ_AVAILABLE:
            raise ImportError("Groq package not installed")
            
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable.")
            
        self.client = Groq(api_key=api_key)
        self.model = model
        
    def decode(self, text: str, max_tokens: int = 256) -> str:
        """Decode/refine text using Groq.
        
        Args:
            text: Input text to decode/refine
            max_tokens: Maximum tokens to generate
            
        Returns:
            Decoded/refined text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides clear, concise answers."},
                    {"role": "user", "content": text}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            return text  # Fallback to original text


def run_interactive(
    module,
    use_dense: bool = True,
    use_groq: bool = False,
    groq_decoder: Optional[GroqDecoder] = None
):
    """Run interactive inference loop.
    
    Args:
        module: PlanSolve module to use
        use_dense: Whether to use dense mode
        use_groq: Whether to use Groq for final decoding
        groq_decoder: GroqDecoder instance
    """
    print("\n" + "="*60)
    print("Interactive DSPy Dense-Vector Agent")
    print("="*60)
    print(f"Mode: {'Dense (hidden-state)' if use_dense else 'Prompt-based'}")
    print(f"Groq decoding: {'Enabled' if use_groq and groq_decoder else 'Disabled'}")
    print("\nEnter math problems or type 'quit' to exit.")
    print("Example: 'John has 5 apples and gives 2 to Mary. How many apples does John have?'")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get user input
            problem = input("Problem: ").strip()
            
            if problem.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not problem:
                continue
                
            # Process the problem
            print("\nThinking...", end='', flush=True)
            start_time = time.time()
            
            if isinstance(module, DensePlanSolve):
                output = module(goal=problem, use_dense=use_dense)
            else:
                output = module(goal=problem)
                
            elapsed = time.time() - start_time
            print(f" (took {elapsed:.1f}s)")
            
            # Display results
            print("\n" + "-"*40)
            
            if "plan" in output and output["plan"]:
                print("Plan:")
                print(output["plan"])
                print()
                
            solution = output["solution"]
            
            # Apply Groq decoding if enabled
            if use_groq and groq_decoder and use_dense:
                print("Raw solution (pre-Groq):")
                print(solution)
                print("\nRefining with Groq...")
                solution = groq_decoder.decode(
                    f"Based on this solution, provide a clear final answer: {solution}"
                )
                print()
                
            print("Solution:")
            print(solution)
            
            # Show token stats if available
            if use_dense and isinstance(module, DensePlanSolve):
                # Estimate tokens
                tokens = len(solution.split())
                print(f"\nTokens emitted: ~{tokens}")
                
            print("-"*40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
            continue
        except Exception as e:
            print(f"\nError: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Run interactive DSPy agent")
    parser.add_argument("--mode", type=str, choices=["dense", "prompt"], default="dense",
                       help="Inference mode")
    parser.add_argument("--use-groq", action="store_true",
                       help="Use Groq for final decoding (requires GROQ_API_KEY)")
    parser.add_argument("--groq-model", type=str, default="llama3-8b-8192",
                       help="Groq model to use")
    parser.add_argument("--edge-weights", type=str, default="edge_state_dict.pt",
                       help="Path to trained edge weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--problem", type=str, help="Single problem to solve (non-interactive)")
    
    args = parser.parse_args()
    
    print(f"Loading model on {args.device}...")
    
    # Initialize module
    if args.mode == "dense":
        module = DensePlanSolve(device=args.device)
        
        # Load trained edge weights if available
        if os.path.exists(args.edge_weights):
            print(f"Loading edge weights from {args.edge_weights}")
            module.load_edge_weights(args.edge_weights)
        else:
            print(f"Warning: Edge weights not found at {args.edge_weights}, using random initialization")
            
        use_dense = True
    else:
        module = PromptPlanSolve()
        use_dense = False
        
    # Initialize Groq decoder if requested
    groq_decoder = None
    if args.use_groq:
        if not GROQ_AVAILABLE:
            print("Error: Groq package not installed. Install with: pip install groq")
            sys.exit(1)
            
        try:
            groq_decoder = GroqDecoder(model=args.groq_model)
            print(f"Groq decoder initialized with model: {args.groq_model}")
        except Exception as e:
            print(f"Failed to initialize Groq: {e}")
            print("Continuing without Groq decoding...")
            args.use_groq = False
            
    # Run single problem or interactive mode
    if args.problem:
        # Single problem mode
        print(f"\nProblem: {args.problem}")
        print("\nProcessing...")
        
        if isinstance(module, DensePlanSolve):
            output = module(goal=args.problem, use_dense=use_dense)
        else:
            output = module(goal=args.problem)
            
        print("\nPlan:")
        print(output.get("plan", "N/A"))
        
        solution = output["solution"]
        if args.use_groq and groq_decoder and use_dense:
            solution = groq_decoder.decode(
                f"Based on this solution, provide a clear final answer: {solution}"
            )
            
        print("\nSolution:")
        print(solution)
    else:
        # Interactive mode
        run_interactive(
            module,
            use_dense=use_dense,
            use_groq=args.use_groq,
            groq_decoder=groq_decoder
        )


if __name__ == "__main__":
    main()