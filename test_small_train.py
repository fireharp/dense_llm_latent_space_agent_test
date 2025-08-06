"""Stage 3: Small training test with mock LM."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os

print("=" * 50)
print("Stage 3: Small Training Test")
print("=" * 50)

# Mock training without loading full model
print("\n1. Setting up mock training...")

try:
    from dense_edge import DenseEdge
    
    # Create edge module
    edge = DenseEdge(d_model=896)
    print(f"✓ Edge initialized with {edge.get_num_params():,} parameters")
    
    # Create optimizer
    optimizer = optim.AdamW(edge.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # Mock training data
    num_train = 10
    num_eval = 20
    
    print(f"\n2. Running mock training on {num_train} examples...")
    
    edge.train()
    total_loss = 0
    
    for i in tqdm(range(num_train), desc="Training"):
        # Simulate hidden states
        h_plan = torch.randn(20, 896)  # Mock planner output
        h_target = torch.randn(20, 896)  # Mock target
        
        # Forward pass
        h_transformed = edge(h_plan)
        
        # Compute loss
        loss = criterion(h_transformed, h_target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / num_train
    print(f"✓ Training completed. Average loss: {avg_loss:.4f}")
    
    # Save mock weights
    mock_weights_path = "mock_edge_state_dict.pt"
    torch.save(edge.state_dict(), mock_weights_path)
    print(f"✓ Saved mock weights to {mock_weights_path}")
    
    # Mock evaluation
    print(f"\n3. Running mock evaluation on {num_eval} examples...")
    
    edge.eval()
    eval_loss = 0
    
    with torch.no_grad():
        for i in tqdm(range(num_eval), desc="Evaluating"):
            h_plan = torch.randn(20, 896)
            h_target = torch.randn(20, 896)
            h_transformed = edge(h_plan)
            loss = criterion(h_transformed, h_target)
            eval_loss += loss.item()
    
    avg_eval_loss = eval_loss / num_eval
    print(f"✓ Evaluation completed. Average loss: {avg_eval_loss:.4f}")
    
    # Save training config
    config = {
        "train_size": num_train,
        "eval_size": num_eval,
        "train_loss": avg_loss,
        "eval_loss": avg_eval_loss,
        "edge_params": edge.get_num_params()
    }
    
    with open("mock_training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ Saved training config to mock_training_config.json")
    
except Exception as e:
    print(f"✗ Training error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Small training test completed!")
print("Note: This was a mock test without the full LLM.")
print("For real training, use: python train.py --train-size 10")
print("=" * 50)