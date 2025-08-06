"""Option 4: Mock training demonstration - shows edge learning with mock LM."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import numpy as np
from mock_lm import MockLM
from dense_edge import DenseEdge
from plansolve import DensePlanSolve
# import matplotlib.pyplot as plt  # Optional

print("=" * 60)
print("Option 4: Mock Training Demonstration")
print("=" * 60)

# Training dataset (simple math problems with ground truth)
training_data = [
    {"question": "What is 5 + 3?", "answer": "8"},
    {"question": "Calculate 12 - 7", "answer": "5"},
    {"question": "What is 6 * 4?", "answer": "24"},
    {"question": "Divide 20 by 5", "answer": "4"},
    {"question": "What is 15 + 8?", "answer": "23"},
    {"question": "Calculate 30 - 12", "answer": "18"},
    {"question": "What is 7 * 9?", "answer": "63"},
    {"question": "Divide 48 by 6", "answer": "8"},
    {"question": "What is 25 + 17?", "answer": "42"},
    {"question": "Calculate 100 - 37", "answer": "63"},
    # Word problems
    {"question": "John has 8 apples and gets 5 more. How many total?", "answer": "13"},
    {"question": "Sarah had 15 cookies and ate 6. How many are left?", "answer": "9"},
    {"question": "Tom has 10 marbles and finds 7 more. Total?", "answer": "17"},
    {"question": "A box has 24 items. Remove 8. How many remain?", "answer": "16"},
    {"question": "Mary walks 3 miles for 5 days. Total miles?", "answer": "15"},
]

# Validation dataset
validation_data = [
    {"question": "What is 9 + 6?", "answer": "15"},
    {"question": "Calculate 20 - 8", "answer": "12"},
    {"question": "What is 5 * 5?", "answer": "25"},
    {"question": "Jane has 12 apples and gives away 4. How many left?", "answer": "8"},
    {"question": "A train travels 60 miles in 2 hours. Speed?", "answer": "30"},
]

print(f"\nTraining dataset: {len(training_data)} examples")
print(f"Validation dataset: {len(validation_data)} examples")

# Initialize components
device = "cpu"
print("\n1. Initializing components...")

# Create modules
planner = MockLM(device=device)
solver = MockLM(device=device)
edge = DenseEdge(d_model=896).to(device)

print(f"✓ Edge initialized with {edge.get_num_params():,} trainable parameters")

# Training configuration
config = {
    "epochs": 5,
    "batch_size": 3,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
}

print(f"\n2. Training configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")

# Create optimizer
optimizer = optim.AdamW(edge.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
criterion = nn.MSELoss()

# Training metrics
training_history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": []
}

print("\n3. Starting training...")
print("-" * 60)

# Training loop
for epoch in range(config["epochs"]):
    print(f"\nEpoch {epoch + 1}/{config['epochs']}")
    
    # Training phase
    edge.train()
    train_losses = []
    
    # Shuffle training data
    indices = torch.randperm(len(training_data))
    
    # Progress bar
    pbar = tqdm(range(0, len(training_data), config["batch_size"]), desc="Training")
    
    for i in pbar:
        batch_indices = indices[i:i + config["batch_size"]]
        batch_loss = 0
        
        optimizer.zero_grad()
        
        for idx in batch_indices:
            example = training_data[idx]
            
            # Encode problem
            h_plan = planner.encode(f"Problem: {example['question']}\nPlan:")
            
            # Transform through edge
            h_transformed = edge(h_plan)
            
            # Create target: encode the answer
            h_target = solver.encode(f"The answer is: {example['answer']}")
            
            # Align dimensions
            min_len = min(h_transformed.size(0), h_target.size(0))
            h_transformed = h_transformed[:min_len]
            h_target = h_target[:min_len]
            
            # Compute loss
            loss = criterion(h_transformed, h_target)
            batch_loss += loss
        
        # Average loss for batch
        if len(batch_indices) > 0:
            batch_loss = batch_loss / len(batch_indices)
            batch_loss.backward()
            optimizer.step()
            
            train_losses.append(batch_loss.item())
            pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})
    
    avg_train_loss = np.mean(train_losses)
    
    # Validation phase
    edge.eval()
    val_losses = []
    correct = 0
    
    with torch.no_grad():
        for example in validation_data:
            # Encode and transform
            h_plan = planner.encode(f"Problem: {example['question']}\nPlan:")
            h_transformed = edge(h_plan)
            
            # Decode solution
            solution = solver.decode(h_transformed)
            
            # Create target for loss
            h_target = solver.encode(f"The answer is: {example['answer']}")
            min_len = min(h_transformed.size(0), h_target.size(0))
            loss = criterion(h_transformed[:min_len], h_target[:min_len])
            val_losses.append(loss.item())
            
            # Check accuracy (simple string matching for demo)
            if example['answer'] in solution:
                correct += 1
    
    avg_val_loss = np.mean(val_losses)
    val_accuracy = correct / len(validation_data)
    
    # Record metrics
    training_history["epoch"].append(epoch + 1)
    training_history["train_loss"].append(avg_train_loss)
    training_history["val_loss"].append(avg_val_loss)
    training_history["val_accuracy"].append(val_accuracy)
    
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Accuracy: {val_accuracy:.1%} ({correct}/{len(validation_data)})")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

# Save trained model
torch.save(edge.state_dict(), "mock_trained_edge.pt")
print("\n✓ Trained edge saved to mock_trained_edge.pt")

# Test on new problems
print("\n4. Testing on new problems...")
print("-" * 60)

test_problems = [
    "What is 11 + 9?",
    "Calculate 50 - 23",
    "A shop has 30 items and sells 12. How many left?",
    "What is 8 * 7?",
]

# Compare untrained vs trained edge
untrained_edge = DenseEdge(d_model=896).to(device)

print("\nComparing untrained vs trained edge:")
for problem in test_problems:
    print(f"\nProblem: {problem}")
    
    # Encode
    h_plan = planner.encode(f"Problem: {problem}\nPlan:")
    
    # Untrained
    h_untrained = untrained_edge(h_plan)
    solution_untrained = solver.decode(h_untrained)
    
    # Trained
    h_trained = edge(h_plan)
    solution_trained = solver.decode(h_trained)
    
    print(f"  Untrained: {solution_untrained[:50]}...")
    print(f"  Trained:   {solution_trained[:50]}...")

# Visualize training progress
print("\n5. Training visualization...")

# Text-based loss plot
print("\nTraining Progress (Loss):")
max_loss = max(max(training_history["train_loss"]), max(training_history["val_loss"]))
scale = 40 / max_loss

for epoch in range(config["epochs"]):
    train_bar = "█" * int(training_history["train_loss"][epoch] * scale)
    val_bar = "░" * int(training_history["val_loss"][epoch] * scale)
    print(f"Epoch {epoch+1}: Train {train_bar} {training_history['train_loss'][epoch]:.4f}")
    print(f"        Val   {val_bar} {training_history['val_loss'][epoch]:.4f}")

# Summary statistics
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

print(f"\n📊 Final Metrics:")
print(f"  Final train loss: {training_history['train_loss'][-1]:.4f}")
print(f"  Final val loss: {training_history['val_loss'][-1]:.4f}")
print(f"  Final val accuracy: {training_history['val_accuracy'][-1]:.1%}")

print(f"\n📈 Improvement:")
print(f"  Train loss reduction: {(1 - training_history['train_loss'][-1]/training_history['train_loss'][0]):.1%}")
print(f"  Val loss reduction: {(1 - training_history['val_loss'][-1]/training_history['val_loss'][0]):.1%}")

# Hidden state analysis
print(f"\n🔍 Hidden State Analysis:")
example_h = planner.encode("What is 10 + 10?")
h_before = untrained_edge(example_h)
h_after = edge(example_h)

print(f"  Untrained edge output variance: {h_before.var().item():.4f}")
print(f"  Trained edge output variance: {h_after.var().item():.4f}")
print(f"  Cosine similarity: {torch.cosine_similarity(h_before.flatten(), h_after.flatten(), dim=0).item():.4f}")

# Save full results
results = {
    "config": config,
    "training_history": training_history,
    "final_metrics": {
        "train_loss": training_history["train_loss"][-1],
        "val_loss": training_history["val_loss"][-1],
        "val_accuracy": training_history["val_accuracy"][-1],
        "parameters": edge.get_num_params()
    },
    "test_examples": [
        {"problem": p, "trained_solution": solver.decode(edge(planner.encode(f"Problem: {p}\nPlan:")))[:50]}
        for p in test_problems[:2]
    ]
}

with open("mock_train_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to mock_train_results.json")

print("\n" + "=" * 60)
print("Mock training demonstration completed!")
print("=" * 60)