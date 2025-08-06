"""Evaluation script for comparing dense vs prompt-based approaches."""

import argparse
import os
import json
import time
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple
import re

import dspy
from dspy.datasets import GSM8K

from dense_lm import DenseLM
from dense_edge import DenseEdge
from plansolve import DensePlanSolve, PromptPlanSolve


def extract_number(text: str) -> str:
    """Extract the final numeric answer from text.
    
    Args:
        text: Text containing a number
        
    Returns:
        Extracted number as string, or None if not found
    """
    # Remove commas from numbers
    text = text.replace(',', '')
    
    # Look for patterns like "answer is X" or "= X" at the end
    patterns = [
        r'(?:answer is|equals?|=)\s*([+-]?\d+\.?\d*)\s*(?:\.|$)',
        r'([+-]?\d+\.?\d*)\s*(?:is the answer|$)',
        r'(?:^|\s)([+-]?\d+\.?\d*)\s*$',  # Number at the end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback: find all numbers and return the last one
    numbers = re.findall(r'[+-]?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
        
    return None


def evaluate_example(module, example, use_dense: bool = True) -> Tuple[bool, int, str]:
    """Evaluate a single example.
    
    Args:
        module: The module to evaluate
        example: GSM8K example
        use_dense: Whether to use dense mode
        
    Returns:
        Tuple of (is_correct, token_count, prediction)
    """
    try:
        # Get prediction
        if isinstance(module, DensePlanSolve):
            output = module(goal=example.question, use_dense=use_dense)
        else:
            output = module(goal=example.question)
            
        prediction = output["solution"]
        
        # Extract numbers
        pred_num = extract_number(prediction)
        true_num = extract_number(example.answer)
        
        # Check correctness
        is_correct = False
        if pred_num and true_num:
            try:
                is_correct = abs(float(pred_num) - float(true_num)) < 1e-5
            except ValueError:
                pass
                
        # Estimate token count
        if use_dense and isinstance(module, DensePlanSolve):
            # Dense mode: only count tokens in final decode
            token_count = len(prediction.split())
        else:
            # Text mode: count all generated tokens
            plan_tokens = len(output.get("plan", "").split())
            solution_tokens = len(prediction.split())
            token_count = plan_tokens + solution_tokens
            
        return is_correct, token_count, prediction
        
    except Exception as e:
        print(f"Error evaluating example: {e}")
        return False, 0, ""


def evaluate_dataset(
    module,
    dataset: List,
    use_dense: bool = True,
    desc: str = "Evaluating",
    limit: int = None
) -> Dict:
    """Evaluate module on a dataset.
    
    Args:
        module: Module to evaluate
        dataset: List of examples
        use_dense: Whether to use dense mode
        desc: Description for progress bar
        limit: Maximum number of examples to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    correct = 0
    total = 0
    total_tokens = 0
    predictions = []
    
    # Limit dataset if specified
    eval_dataset = dataset[:limit] if limit else dataset
    
    for example in tqdm(eval_dataset, desc=desc):
        is_correct, token_count, prediction = evaluate_example(module, example, use_dense)
        
        if is_correct:
            correct += 1
        total += 1
        total_tokens += token_count
        
        predictions.append({
            "question": example.question,
            "true_answer": example.answer,
            "prediction": prediction,
            "correct": is_correct,
            "tokens": token_count
        })
        
    accuracy = correct / total if total > 0 else 0
    avg_tokens = total_tokens / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "predictions": predictions
    }


def run_comparison(args):
    """Run comparison between dense and prompt approaches."""
    print(f"Device: {args.device}")
    print(f"Evaluating on {args.eval_size} examples from {args.split} split\n")
    
    # Load dataset
    print("Loading GSM8K dataset...")
    gsm8k = GSM8K()
    
    if args.split == "train":
        dataset = gsm8k.train[:args.eval_size]
    else:
        dataset = gsm8k.dev[:args.eval_size]
        
    print(f"Loaded {len(dataset)} examples")
    
    # Initialize modules
    print("\nInitializing modules...")
    
    # Dense module
    dense_module = DensePlanSolve(device=args.device)
    if args.edge_weights and os.path.exists(args.edge_weights):
        print(f"Loading edge weights from {args.edge_weights}")
        dense_module.load_edge_weights(args.edge_weights)
    else:
        print("Using untrained edge weights")
        
    # Baseline module
    baseline_module = PromptPlanSolve()
    
    # Evaluate baseline
    print("\n" + "="*50)
    print("Evaluating BASELINE (Prompt-based) approach...")
    print("="*50)
    
    start_time = time.time()
    baseline_results = evaluate_dataset(
        baseline_module,
        dataset,
        use_dense=False,
        desc="Baseline evaluation",
        limit=args.eval_size
    )
    baseline_time = time.time() - start_time
    
    print(f"\nBaseline Results:")
    print(f"  Accuracy: {baseline_results['accuracy']:.3f} ({baseline_results['correct']}/{baseline_results['total']})")
    print(f"  Avg tokens: {baseline_results['avg_tokens']:.1f}")
    print(f"  Total tokens: {baseline_results['total_tokens']}")
    print(f"  Time: {baseline_time:.1f}s")
    
    # Evaluate dense
    print("\n" + "="*50)
    print("Evaluating DENSE (Hidden-state) approach...")
    print("="*50)
    
    start_time = time.time()
    dense_results = evaluate_dataset(
        dense_module,
        dataset,
        use_dense=True,
        desc="Dense evaluation",
        limit=args.eval_size
    )
    dense_time = time.time() - start_time
    
    print(f"\nDense Results:")
    print(f"  Accuracy: {dense_results['accuracy']:.3f} ({dense_results['correct']}/{dense_results['total']})")
    print(f"  Avg tokens: {dense_results['avg_tokens']:.1f}")
    print(f"  Total tokens: {dense_results['total_tokens']}")
    print(f"  Time: {dense_time:.1f}s")
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    
    accuracy_diff = dense_results['accuracy'] - baseline_results['accuracy']
    token_reduction = 1 - (dense_results['avg_tokens'] / baseline_results['avg_tokens'])
    
    print(f"Accuracy difference: {accuracy_diff:+.3f}")
    print(f"Token reduction: {token_reduction:.1%}")
    print(f"Speed ratio: {baseline_time/dense_time:.2f}x")
    
    # Check success criteria
    print("\n" + "="*50)
    print("SUCCESS CRITERIA")
    print("="*50)
    
    criteria_met = []
    criteria_failed = []
    
    # Criterion 1: Accuracy within 1% of baseline
    if dense_results['accuracy'] >= baseline_results['accuracy'] - 0.01:
        criteria_met.append(f"✓ Accuracy: {dense_results['accuracy']:.3f} >= {baseline_results['accuracy'] - 0.01:.3f}")
    else:
        criteria_failed.append(f"✗ Accuracy: {dense_results['accuracy']:.3f} < {baseline_results['accuracy'] - 0.01:.3f}")
        
    # Criterion 2: ≤10% of baseline tokens
    if dense_results['avg_tokens'] <= 0.1 * baseline_results['avg_tokens']:
        criteria_met.append(f"✓ Token budget: {dense_results['avg_tokens']:.1f} <= {0.1 * baseline_results['avg_tokens']:.1f}")
    else:
        criteria_failed.append(f"✗ Token budget: {dense_results['avg_tokens']:.1f} > {0.1 * baseline_results['avg_tokens']:.1f}")
        
    for criterion in criteria_met:
        print(criterion)
    for criterion in criteria_failed:
        print(criterion)
        
    overall_success = len(criteria_failed) == 0
    print(f"\nOVERALL: {'SUCCESS' if overall_success else 'FAILURE'}")
    
    # Save results
    if args.save_results:
        results = {
            "config": vars(args),
            "baseline": baseline_results,
            "dense": dense_results,
            "comparison": {
                "accuracy_diff": accuracy_diff,
                "token_reduction": token_reduction,
                "speed_ratio": baseline_time/dense_time,
                "success": overall_success
            }
        }
        
        # Remove predictions for compact output
        results["baseline"].pop("predictions", None)
        results["dense"].pop("predictions", None)
        
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate dense vs prompt approaches")
    parser.add_argument("--module", type=str, choices=["PlanSolve", "PromptPlanSolve", "both"], 
                       default="both", help="Which module to evaluate")
    parser.add_argument("--split", type=str, choices=["train", "dev"], default="dev",
                       help="Dataset split to evaluate on")
    parser.add_argument("--eval-size", type=int, default=500,
                       help="Number of examples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--edge-weights", type=str, default="edge_state_dict.pt",
                       help="Path to trained edge weights")
    parser.add_argument("--save-results", type=str, default="evaluation_results.json",
                       help="Path to save results JSON")
    
    args = parser.parse_args()
    
    if args.module == "both":
        run_comparison(args)
    else:
        # Single module evaluation
        print(f"Evaluating {args.module} on {args.eval_size} examples...")
        
        gsm8k = GSM8K()
        dataset = gsm8k.dev[:args.eval_size] if args.split == "dev" else gsm8k.train[:args.eval_size]
        
        if args.module == "PlanSolve":
            module = DensePlanSolve(device=args.device)
            if args.edge_weights and os.path.exists(args.edge_weights):
                module.load_edge_weights(args.edge_weights)
            use_dense = True
        else:
            module = PromptPlanSolve()
            use_dense = False
            
        results = evaluate_dataset(module, dataset, use_dense=use_dense)
        
        print(f"\nResults:")
        print(f"  Accuracy: {results['accuracy']:.3f}")
        print(f"  Avg tokens: {results['avg_tokens']:.1f}")


if __name__ == "__main__":
    main()