"""Visual step-by-step comparison of Text vs Dense processing."""

import torch
import numpy as np
from mock_lm import MockLM
from dense_edge import DenseEdge
import time

def visualize_tensor(tensor, name, max_positions=3, max_dims=5):
    """Visualize a tensor with ASCII art."""
    print(f"\n{name} [{tensor.shape[0]}×{tensor.shape[1]}]:")
    print("┌" + "─" * 50 + "┐")
    
    for i in range(min(tensor.shape[0], max_positions)):
        values = tensor[i, :max_dims].tolist()
        formatted = [f"{v:6.3f}" for v in values]
        print(f"│ Pos {i:2d}: [{', '.join(formatted)}, ...] │")
    
    if tensor.shape[0] > max_positions:
        print(f"│  ... ({tensor.shape[0] - max_positions} more positions) ...{' '*28}│")
    
    print("└" + "─" * 50 + "┘")
    print(f"Total values: {tensor.numel():,} floats = {tensor.numel() * 4:,} bytes")

def main():
    # Setup
    device = "cpu"
    planner = MockLM(device=device)
    solver = MockLM(device=device)
    edge = DenseEdge(d_model=896).to(device)
    
    problem = "What is 25 + 17?"
    
    print("="*70)
    print(" "*20 + "TEXT vs DENSE PROCESSING")
    print("="*70)
    print(f"\n🎯 Problem: '{problem}'")
    
    # ==== TEXT-BASED PROCESSING ====
    print("\n" + "┏" + "━"*68 + "┓")
    print("┃" + " "*20 + "📝 TEXT-BASED APPROACH" + " "*26 + "┃")
    print("┗" + "━"*68 + "┛")
    
    print("\n[STEP 1] Problem → Planner → Plan Text")
    print("─" * 40)
    
    # Simulate text generation
    plan_prompt = f"Problem: {problem}\nGenerate a plan:"
    print(f"Input: '{plan_prompt}'")
    
    time.sleep(0.1)  # Simulate processing
    plan_text = "1. Identify the numbers: 25 and 17\n2. Add them together\n3. Return the result"
    
    print(f"\n🤖 Planner generates TEXT:")
    print(f"   '{plan_text}'")
    print(f"   Tokens: {len(plan_text.split())} | Bytes: {len(plan_text.encode('utf-8'))}")
    
    print("\n[STEP 2] Plan Text → Solver → Solution Text")
    print("─" * 40)
    
    solve_prompt = f"Plan: {plan_text}\nProblem: {problem}\nSolve:"
    print(f"Input: '{solve_prompt[:60]}...'")
    
    time.sleep(0.1)  # Simulate processing
    solution_text = "Following the plan:\n1. Numbers are 25 and 17\n2. 25 + 17 = 42\n3. The answer is 42"
    
    print(f"\n🤖 Solver generates TEXT:")
    print(f"   '{solution_text}'")
    print(f"   Tokens: {len(solution_text.split())} | Bytes: {len(solution_text.encode('utf-8'))}")
    
    total_text_tokens = len(plan_text.split()) + len(solution_text.split())
    total_text_bytes = len(plan_text.encode('utf-8')) + len(solution_text.encode('utf-8'))
    
    print(f"\n📊 TOTAL TEXT: {total_text_tokens} tokens, {total_text_bytes} bytes")
    
    # ==== DENSE PROCESSING ====
    print("\n\n" + "┏" + "━"*68 + "┓")
    print("┃" + " "*20 + "🧠 DENSE VECTOR APPROACH" + " "*24 + "┃")
    print("┗" + "━"*68 + "┛")
    
    print("\n[STEP 1] Problem → Planner → Hidden States")
    print("─" * 40)
    
    print(f"Input: '{problem}'")
    
    # Encode to hidden states
    h_plan = planner.encode(f"Problem: {problem}\nPlan:")
    
    print(f"\n🧠 Planner generates HIDDEN STATES (no text!):")
    visualize_tensor(h_plan, "Planning Hidden States")
    
    print("\n💡 These 896 numbers per position encode:")
    print("   • Understanding that 25 and 17 are numbers")
    print("   • Knowledge that '+' means addition")
    print("   • Planning steps (all implicit in vectors)")
    
    print("\n[STEP 2] Hidden States → Edge → Transformed Hidden States")
    print("─" * 40)
    
    # Transform with edge
    h_transformed = edge(h_plan)
    
    print(f"\n⚡ Edge transforms hidden states (still no text!):")
    visualize_tensor(h_transformed, "Transformed Hidden States")
    
    print("\n💡 The Edge learned to:")
    print("   • Apply attention between positions")
    print("   • Transform planning → solving representations")
    print("   • All computation in vector space!")
    
    print("\n[STEP 3] Hidden States → Solver → Final Answer Only")
    print("─" * 40)
    
    # Decode only final answer
    final_answer = solver.decode(h_transformed, max_new_tokens=10)
    
    print(f"\n🎯 Solver decodes ONLY the final answer:")
    print(f"   '{final_answer}'")
    print(f"   Tokens: {len(final_answer.split())} | Bytes: {len(final_answer.encode('utf-8'))}")
    
    # ==== COMPARISON ====
    print("\n\n" + "="*70)
    print(" "*25 + "📊 COMPARISON")
    print("="*70)
    
    print("\n┌─────────────────────┬─────────────────┬─────────────────┐")
    print("│     Metric          │   Text-Based    │   Dense Vector  │")
    print("├─────────────────────┼─────────────────┼─────────────────┤")
    print(f"│ Tokens Generated    │      {total_text_tokens:3d}         │       {len(final_answer.split()):3d}        │")
    print(f"│ Bytes (Network)     │      {total_text_bytes:3d}         │       {len(final_answer.encode('utf-8')):3d}        │")
    print(f"│ Reduction           │       -         │      {(1 - len(final_answer.split())/total_text_tokens)*100:.0f}%        │")
    print("├─────────────────────┼─────────────────┼─────────────────┤")
    print("│ Intermediate Steps  │   All as text   │  Hidden vectors │")
    print("│ Network Transfer    │   All text      │  Final only     │")
    print("│ Information Flow    │   ~100 dims     │   896 dims      │")
    print("└─────────────────────┴─────────────────┴─────────────────┘")
    
    print("\n🔑 KEY INSIGHT:")
    print("   Text approach: Every module must GENERATE then PARSE text")
    print("   Dense approach: Modules pass RICH VECTORS, decode once at end")
    
    print("\n💰 COST SAVINGS:")
    print(f"   If each token costs $0.01:")
    print(f"   • Text approach: ${total_text_tokens * 0.01:.2f}")
    print(f"   • Dense approach: ${len(final_answer.split()) * 0.01:.2f}")
    print(f"   • Savings: ${(total_text_tokens - len(final_answer.split())) * 0.01:.2f} ({(1 - len(final_answer.split())/total_text_tokens)*100:.0f}%)")
    
    print("\n🚀 PERFORMANCE:")
    print("   • Dense keeps all computation LOCAL")
    print("   • Only final answer needs API/network")
    print("   • Hidden states are larger but stay on device")
    print("   • Perfect for distributed systems!")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()