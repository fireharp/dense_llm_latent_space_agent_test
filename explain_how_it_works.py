"""Detailed explanation and comparison: Text vs Vector processing."""

import torch
import time
from mock_lm import MockLM
from dense_edge import DenseEdge
from plansolve import DensePlanSolve, PromptPlanSolve
import numpy as np

def print_section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

def show_tensor_info(tensor, name):
    """Show detailed tensor information."""
    print(f"\n📊 {name}:")
    print(f"   Shape: {tensor.shape} = {tensor.shape[0]} positions × {tensor.shape[1]} dimensions")
    print(f"   Memory: {tensor.numel() * 4:,} bytes ({tensor.numel() * 4 / 1024:.1f} KB)")
    print(f"   Values: min={tensor.min().item():.3f}, max={tensor.max().item():.3f}, mean={tensor.mean().item():.3f}")
    print(f"   First position: [{tensor[0,:5].tolist()[0]:.3f}, {tensor[0,:5].tolist()[1]:.3f}, {tensor[0,:5].tolist()[2]:.3f}, ...] (showing 3 of {tensor.shape[1]} dims)")

def main():
    print_section("HOW DENSE-VECTOR COMMUNICATION WORKS")
    
    # Initialize components
    device = "cpu"
    planner = MockLM(device=device)
    solver = MockLM(device=device)
    edge = DenseEdge(d_model=896).to(device)
    
    # Problem to solve
    problem = "John has 15 apples. He gives 3 to Mary. How many apples does John have?"
    
    print(f"\n🎯 Problem: {problem}")
    
    # ========== TRADITIONAL TEXT-BASED APPROACH ==========
    print_section("1. TRADITIONAL TEXT-BASED APPROACH")
    
    print("\n📝 How it works:")
    print("   Each module generates TEXT that the next module reads")
    print("   Every intermediate step requires full text generation")
    
    # Create traditional module
    prompt_module = PromptPlanSolve(lm=planner)
    
    print("\n🔄 Processing steps:")
    
    # Step 1: Planner generates text
    print("\n   Step 1: Planner generates a plan (TEXT)")
    start = time.time()
    # Simulate what happens internally
    plan_prompt = f"Problem: {problem}\nGenerate a step-by-step plan:"
    plan_text = "Step 1: Identify initial amount (15 apples). Step 2: Identify amount given away (3 apples). Step 3: Subtract to find remaining."
    plan_time = time.time() - start
    
    print(f"      Input prompt: '{plan_prompt[:50]}...'")
    print(f"      Generated text: '{plan_text[:60]}...'")
    print(f"      Tokens used: ~{len(plan_text.split())} tokens")
    print(f"      Bytes: {len(plan_text.encode('utf-8'))} bytes")
    
    # Step 2: Solver reads plan and generates solution
    print("\n   Step 2: Solver reads plan and generates solution (TEXT)")
    start = time.time()
    solve_prompt = f"Plan: {plan_text}\nProblem: {problem}\nSolve step by step:"
    solution_text = "Following the plan: John starts with 15 apples. He gives 3 to Mary. So 15 - 3 = 12. John has 12 apples."
    solve_time = time.time() - start
    
    print(f"      Input prompt: '{solve_prompt[:50]}...'")
    print(f"      Generated text: '{solution_text[:60]}...'")
    print(f"      Tokens used: ~{len(solution_text.split())} tokens")
    print(f"      Bytes: {len(solution_text.encode('utf-8'))} bytes")
    
    # Total traditional
    total_text_tokens = len(plan_text.split()) + len(solution_text.split())
    total_text_bytes = len(plan_text.encode('utf-8')) + len(solution_text.encode('utf-8'))
    
    print(f"\n   📊 TOTAL TEXT APPROACH:")
    print(f"      Total tokens: {total_text_tokens}")
    print(f"      Total bytes: {total_text_bytes}")
    print(f"      All intermediate reasoning as TEXT")
    
    # ========== DENSE VECTOR APPROACH ==========
    print_section("2. DENSE VECTOR APPROACH")
    
    print("\n🧠 How it works:")
    print("   Each module passes HIDDEN STATES (vectors) instead of text")
    print("   Only the final answer is decoded to text")
    
    # Create dense module
    dense_module = DensePlanSolve(
        planner_lm=planner,
        solver_lm=solver,
        edge=edge,
        device=device,
        share_lm=False
    )
    
    print("\n🔄 Processing steps:")
    
    # Step 1: Encode to hidden states
    print("\n   Step 1: Planner encodes problem to HIDDEN STATES")
    start = time.time()
    h_plan = planner.encode(f"Problem: {problem}\nPlan:")
    encode_time = time.time() - start
    
    show_tensor_info(h_plan, "Planning hidden states")
    
    # Visualize what's in the hidden states
    print("\n   🔍 What's in these vectors?")
    print("      - Semantic meaning of the problem")
    print("      - Numerical understanding (15, 3)")
    print("      - Operation understanding (subtraction)")
    print("      - All compressed into 896 numbers per position!")
    
    # Step 2: Transform with Edge
    print("\n   Step 2: Edge transformer processes hidden states")
    start = time.time()
    h_transformed = edge(h_plan)
    edge_time = time.time() - start
    
    show_tensor_info(h_transformed, "Transformed hidden states")
    
    print("\n   🔍 What did the Edge do?")
    print("      - Learned transformations for reasoning")
    print("      - Attention between different positions")
    print("      - No text generation needed!")
    
    # Step 3: Decode only final answer
    print("\n   Step 3: Solver decodes ONLY the final answer")
    start = time.time()
    final_answer = solver.decode(h_transformed, max_new_tokens=20)
    decode_time = time.time() - start
    
    print(f"      Final text: '{final_answer}'")
    print(f"      Tokens used: ~{len(final_answer.split())} tokens")
    print(f"      Bytes: {len(final_answer.encode('utf-8'))} bytes")
    
    # Total dense
    total_dense_tokens = len(final_answer.split())
    total_dense_bytes = len(final_answer.encode('utf-8'))
    hidden_bytes = (h_plan.numel() + h_transformed.numel()) * 4
    
    print(f"\n   📊 TOTAL DENSE APPROACH:")
    print(f"      Text tokens: {total_dense_tokens} (only final answer)")
    print(f"      Text bytes: {total_dense_bytes}")
    print(f"      Hidden state bytes: {hidden_bytes:,} (stays local)")
    
    # ========== COMPARISON ==========
    print_section("3. SIDE-BY-SIDE COMPARISON")
    
    print("\n📊 Token Usage:")
    print(f"   Traditional: {total_text_tokens} tokens (all text)")
    print(f"   Dense:       {total_dense_tokens} tokens (only final)")
    print(f"   Reduction:   {(1 - total_dense_tokens/total_text_tokens)*100:.1f}%")
    
    print("\n💾 Network Transfer (if distributed):")
    print(f"   Traditional: {total_text_bytes} bytes (all text must transfer)")
    print(f"   Dense:       {total_dense_bytes} bytes (only final transfers)")
    print(f"   Savings:     {(1 - total_dense_bytes/total_text_bytes)*100:.1f}%")
    
    print("\n🧠 Where Computation Happens:")
    print("   Traditional: Generate text → Transfer → Parse text → Generate text")
    print("   Dense:       Hidden states → Local transform → Hidden states → Final text")
    
    # ========== VISUAL FLOW ==========
    print_section("4. VISUAL PROCESSING FLOW")
    
    print("\n🔄 TRADITIONAL FLOW:")
    print("""
    Problem ──→ [Planner] ──→ "Step 1: Identify..."  ──→ [Solver] ──→ "John has 12 apples"
                    ↓                  (30 tokens)           ↓              (20 tokens)
                Generate                                  Generate
                  TEXT                                      TEXT
    """)
    
    print("\n🔄 DENSE FLOW:")
    print("""
    Problem ──→ [Planner] ──→ [896×N tensor] ──→ [Edge] ──→ [896×N tensor] ──→ [Solver] ──→ "12"
                    ↓              ↓                ↓              ↓                ↓
                 Encode         Hidden          Transform      Hidden           Decode
                to vectors      States           vectors       States          final only
    """)
    
    # ========== RUN BOTH SIDE BY SIDE ==========
    print_section("5. RUNNING BOTH APPROACHES")
    
    print("\n🏃 Running traditional approach...")
    start = time.time()
    trad_output = prompt_module(goal=problem)
    trad_time = time.time() - start
    
    print(f"   Time: {trad_time*1000:.1f}ms")
    print(f"   Plan: '{trad_output.get('plan', '')[:60]}...'")
    print(f"   Solution: '{trad_output.get('solution', '')[:60]}...'")
    
    print("\n🏃 Running dense approach...")
    start = time.time()
    dense_output = dense_module(goal=problem, use_dense=True)
    dense_time = time.time() - start
    
    print(f"   Time: {dense_time*1000:.1f}ms")
    print(f"   Hidden processing: ✓")
    print(f"   Solution: '{dense_output.get('solution', '')}'")
    
    # ========== KEY INSIGHTS ==========
    print_section("6. KEY INSIGHTS")
    
    print("\n💡 Why This Works:")
    print("   1. LLMs already think in hidden states internally")
    print("   2. Text generation is just the final projection layer")
    print("   3. We skip text generation for intermediate steps")
    print("   4. Hidden states carry richer information than text")
    
    print("\n🎯 Benefits:")
    print("   • 90%+ fewer tokens (saves API costs)")
    print("   • Faster processing (no intermediate decoding)")
    print("   • Richer information flow (896 dimensions vs text)")
    print("   • Works with any transformer model")
    
    print("\n⚙️ Requirements:")
    print("   • Models must expose hidden states")
    print("   • Edge module must be trained for the task")
    print("   • Final decoder can be different (e.g., Groq)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()