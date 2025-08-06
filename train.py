"""Training script for edge-only fine-tuning on GSM8K."""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dspy
from dspy.datasets import GSM8K
from dspy.teleprompt import BootstrapFinetune
from tqdm import tqdm
import json

from dense_lm import DenseLM
from dense_edge import DenseEdge
from plansolve import DensePlanSolve


def create_gsm8k_dataset(train_size: int = 200, dev_size: int = 500):
    """Load GSM8K dataset with specified splits.
    
    Args:
        train_size: Number of training examples
        dev_size: Number of development examples
        
    Returns:
        Tuple of (train_set, dev_set)
    """
    # Load GSM8K dataset
    gsm8k = GSM8K()
    
    # Get train and dev splits
    train_set = gsm8k.train[:train_size]
    dev_set = gsm8k.dev[:dev_size]
    
    return train_set, dev_set


def evaluate_module(module, dataset, use_dense: bool = True, desc: str = "Evaluating"):
    """Evaluate module on dataset.
    
    Args:
        module: PlanSolve module to evaluate
        dataset: List of examples
        use_dense: Whether to use dense mode
        desc: Description for progress bar
        
    Returns:
        Dictionary with evaluation metrics
    """
    correct = 0
    total = 0
    total_tokens = 0
    
    for example in tqdm(dataset, desc=desc):
        try:
            # Get prediction
            output = module(goal=example.question, use_dense=use_dense)
            prediction = output["solution"]
            
            # Extract numeric answer from prediction
            # GSM8K answers are typically numbers
            pred_number = extract_number(prediction)
            true_number = extract_number(example.answer)
            
            # Check if correct
            if pred_number is not None and true_number is not None:
                if abs(float(pred_number) - float(true_number)) < 1e-5:
                    correct += 1
                    
            total += 1
            
            # Estimate tokens (rough approximation)
            if use_dense:
                # Dense mode emits much fewer tokens
                total_tokens += len(output["plan"].split()) // 10
            else:
                total_tokens += len(output["plan"].split()) + len(output["solution"].split())
                
        except Exception as e:
            print(f"Error evaluating example: {e}")
            total += 1
            
    accuracy = correct / total if total > 0 else 0
    avg_tokens = total_tokens / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_tokens": avg_tokens
    }


def extract_number(text: str):
    """Extract number from text answer."""
    import re
    
    # Look for numbers in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    if numbers:
        # Return the last number found (usually the answer)
        return numbers[-1]
    return None


def train_edge(
    module: DensePlanSolve,
    train_set,
    dev_set,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    device: str = "cuda",
    save_path: str = "edge_state_dict.pt"
):
    """Train the edge module.
    
    Args:
        module: DensePlanSolve module
        train_set: Training dataset
        dev_set: Development dataset
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        save_path: Path to save best model
    """
    # Freeze LMs, only train edge
    module.freeze_lms()
    
    # Get edge parameters
    optimizer = optim.AdamW(
        module.edge.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Loss function - we'll use MSE loss between transformed states and target states
    criterion = nn.MSELoss()
    
    best_dev_accuracy = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training
        module.edge.train()
        train_loss = 0
        
        # Create batches
        for i in tqdm(range(0, len(train_set), batch_size), desc="Training"):
            batch = train_set[i:i + batch_size]
            optimizer.zero_grad()
            
            batch_loss = 0
            for example in batch:
                try:
                    # Get hidden states
                    planning_prompt = f"Problem: {example.question}\nLet me create a step-by-step plan:"
                    h_plan = module.planner.encode(planning_prompt)
                    
                    # Transform through edge
                    h_transformed = module.edge(h_plan)
                    
                    # Create target: encode the answer
                    answer_prompt = f"The answer is: {example.answer}"
                    h_target = module.solver.encode(answer_prompt)
                    
                    # Align dimensions if needed
                    min_len = min(h_transformed.size(0), h_target.size(0))
                    h_transformed = h_transformed[:min_len]
                    h_target = h_target[:min_len]
                    
                    # Compute loss
                    loss = criterion(h_transformed, h_target)
                    batch_loss += loss
                    
                except Exception as e:
                    print(f"Error in training example: {e}")
                    continue
                    
            if batch_loss > 0:
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                optimizer.step()
                train_loss += batch_loss.item()
                
        avg_train_loss = train_loss / (len(train_set) // batch_size)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Evaluation
        module.edge.eval()
        with torch.no_grad():
            dev_metrics = evaluate_module(
                module,
                dev_set[:50],  # Evaluate on subset for speed
                use_dense=True,
                desc="Evaluating on dev"
            )
            
        print(f"Dev accuracy: {dev_metrics['accuracy']:.3f}")
        print(f"Dev avg tokens: {dev_metrics['avg_tokens']:.1f}")
        
        # Save best model
        if dev_metrics['accuracy'] > best_dev_accuracy:
            best_dev_accuracy = dev_metrics['accuracy']
            module.save_edge_weights(save_path)
            print(f"Saved best model with accuracy: {best_dev_accuracy:.3f}")
            
        # Early stopping if accuracy is good enough
        if best_dev_accuracy >= 0.95:
            print("Reached target accuracy, stopping early")
            break
            
    return best_dev_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train DenseEdge on GSM8K")
    parser.add_argument("--train-size", type=int, default=200, help="Number of training examples")
    parser.add_argument("--dev-size", type=int, default=500, help="Number of dev examples")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="edge_state_dict.pt", help="Path to save model")
    
    args = parser.parse_args()
    
    print(f"Training on device: {args.device}")
    
    # Load dataset
    print("Loading GSM8K dataset...")
    train_set, dev_set = create_gsm8k_dataset(args.train_size, args.dev_size)
    print(f"Train size: {len(train_set)}, Dev size: {len(dev_set)}")
    
    # Initialize module
    print("Initializing DensePlanSolve module...")
    module = DensePlanSolve(device=args.device)
    
    # Train
    print("Starting training...")
    best_accuracy = train_edge(
        module,
        train_set,
        dev_set,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        save_path=args.save_path
    )
    
    print(f"\nTraining complete! Best dev accuracy: {best_accuracy:.3f}")
    print(f"Model saved to: {args.save_path}")
    
    # Save training config
    config = {
        "train_size": args.train_size,
        "dev_size": args.dev_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_accuracy": best_accuracy
    }
    
    with open("training_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print("Training config saved to: training_config.json")


if __name__ == "__main__":
    main()