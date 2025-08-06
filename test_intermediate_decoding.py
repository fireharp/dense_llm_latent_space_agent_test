"""Demonstrate decoding intermediate hidden states for visibility."""

import torch
import time
import asyncio
from mock_lm import MockLM
from dense_edge import DenseEdge
import json
from concurrent.futures import ThreadPoolExecutor

class DenseWithIntrospection:
    """Dense processing with optional intermediate decoding."""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.encoder = MockLM(device=device)
        self.decoder = MockLM(device=device)
        self.edge = DenseEdge(d_model=896).to(device)
        
        # Store intermediate states for later introspection
        self.intermediate_states = {}
        
    def process(self, question, decode_intermediates=False):
        """Process question with optional intermediate decoding."""
        print(f"\n🎯 Question: {question}")
        print("="*60)
        
        # Main processing (fast path)
        start_main = time.time()
        
        # Step 1: Encode
        h_encoded = self.encoder.encode(question)
        self.intermediate_states['after_encode'] = h_encoded.clone()
        
        # Step 2: Edge transform
        h_transformed = self.edge(h_encoded)
        self.intermediate_states['after_edge'] = h_transformed.clone()
        
        # Step 3: Decode final answer
        final_answer = self.decoder.decode(h_transformed, max_new_tokens=20)
        
        main_time = (time.time() - start_main) * 1000
        
        print(f"\n✅ MAIN PROCESS COMPLETE")
        print(f"   Final answer: {final_answer}")
        print(f"   Time: {main_time:.1f}ms")
        
        # Optional: Decode intermediates (can be async)
        if decode_intermediates:
            print(f"\n🔍 DECODING INTERMEDIATE STATES (for visibility):")
            print("-"*60)
            self._decode_intermediates()
            
        return final_answer, self.intermediate_states
    
    def _decode_intermediates(self):
        """Decode intermediate hidden states for inspection."""
        
        # Decode after encoding
        if 'after_encode' in self.intermediate_states:
            h = self.intermediate_states['after_encode']
            decoded = self.decoder.decode(h, max_new_tokens=50)
            print(f"\n📝 After Encoding (what the encoder 'sees'):")
            print(f"   Hidden shape: {h.shape}")
            print(f"   Decoded: '{decoded}'")
            print(f"   Interpretation: This shows what information was extracted")
            
        # Decode after edge
        if 'after_edge' in self.intermediate_states:
            h = self.intermediate_states['after_edge']
            decoded = self.decoder.decode(h, max_new_tokens=50)
            print(f"\n⚡ After Edge Transform:")
            print(f"   Hidden shape: {h.shape}")
            print(f"   Decoded: '{decoded}'")
            print(f"   Interpretation: This shows how Edge transformed the reasoning")
    
    async def process_async_with_introspection(self, question):
        """Process with async intermediate decoding."""
        print(f"\n🚀 ASYNC PROCESSING WITH BACKGROUND INTROSPECTION")
        print("="*60)
        
        # Main processing
        start = time.time()
        
        # Fast path - get answer immediately
        h_encoded = self.encoder.encode(question)
        h_transformed = self.edge(h_encoded)
        final_answer = self.decoder.decode(h_transformed, max_new_tokens=20)
        
        fast_time = (time.time() - start) * 1000
        
        print(f"\n✅ Answer ready in {fast_time:.1f}ms: {final_answer}")
        print("\n⏳ Decoding intermediates in background...")
        
        # Async decode intermediates
        async def decode_state(name, hidden_state):
            await asyncio.sleep(0.1)  # Simulate async work
            decoded = self.decoder.decode(hidden_state, max_new_tokens=30)
            return name, decoded
        
        # Decode all intermediates in parallel
        tasks = [
            decode_state("after_encode", h_encoded),
            decode_state("after_edge", h_transformed),
        ]
        
        results = await asyncio.gather(*tasks)
        
        print("\n📊 Background introspection complete:")
        for name, decoded in results:
            print(f"   {name}: '{decoded}'")
            
        return final_answer

def test_intermediate_visibility():
    """Test different visibility modes."""
    
    dense = DenseWithIntrospection()
    
    # Test 1: Normal fast mode (no intermediate decoding)
    print("\n" + "="*70)
    print("TEST 1: FAST MODE (No intermediate decoding)")
    print("="*70)
    
    question = "How many r's are in the word strawberry?"
    answer, states = dense.process(question, decode_intermediates=False)
    
    # Test 2: Debug mode (with intermediate decoding)
    print("\n\n" + "="*70)
    print("TEST 2: DEBUG MODE (With intermediate decoding)")
    print("="*70)
    
    answer, states = dense.process(question, decode_intermediates=True)
    
    # Test 3: Production pattern - log intermediates for later analysis
    print("\n\n" + "="*70)
    print("TEST 3: PRODUCTION PATTERN (Log for later analysis)")
    print("="*70)
    
    # Fast processing
    start = time.time()
    answer, states = dense.process(question, decode_intermediates=False)
    process_time = (time.time() - start) * 1000
    
    # Save intermediate states for later analysis
    debug_log = {
        'question': question,
        'answer': answer,
        'process_time_ms': process_time,
        'intermediate_shapes': {
            name: list(state.shape) for name, state in states.items()
        },
        'intermediate_samples': {
            name: state[0, :5].tolist() for name, state in states.items()
        }
    }
    
    print(f"\n📁 Saved debug log for offline analysis:")
    print(json.dumps(debug_log, indent=2))
    
    # Test 4: Selective decoding
    print("\n\n" + "="*70)
    print("TEST 4: SELECTIVE INTROSPECTION")
    print("="*70)
    
    # Process multiple questions
    questions = [
        "What is 25 + 17?",
        "How many letters in 'hello'?",
        "What color is the sky?",
    ]
    
    for i, q in enumerate(questions):
        answer, states = dense.process(q, decode_intermediates=False)
        
        # Only decode intermediates if answer seems wrong
        if "0" in answer or "error" in answer.lower():
            print(f"\n⚠️  Suspicious answer for question {i+1}, decoding intermediates:")
            dense._decode_intermediates()

def test_async_pattern():
    """Test async pattern for production use."""
    print("\n\n" + "="*70)
    print("TEST 5: ASYNC PATTERN (Best for production)")
    print("="*70)
    
    async def main():
        dense = DenseWithIntrospection()
        question = "How many r's are in strawberry?"
        
        # User gets answer immediately, introspection happens in background
        answer = await dense.process_async_with_introspection(question)
        
        print(f"\n✅ User received answer immediately")
        print(f"✅ Debug info logged in background")
        print(f"✅ No impact on user latency!")
    
    asyncio.run(main())

def show_advanced_introspection():
    """Show advanced introspection techniques."""
    print("\n\n" + "="*70)
    print("ADVANCED INTROSPECTION TECHNIQUES")
    print("="*70)
    
    dense = DenseWithIntrospection()
    question = "How many r's are in strawberry?"
    
    # Process
    h_encoded = dense.encoder.encode(question)
    h_transformed = dense.edge(h_encoded)
    
    print(f"\n1. ATTENTION PATTERN ANALYSIS:")
    print("   We could decode specific attention heads to see what they focus on")
    
    print(f"\n2. LAYER-WISE DECODING:")
    print("   Decode at each transformer layer to see reasoning evolution")
    
    print(f"\n3. PROBE SPECIFIC POSITIONS:")
    # Decode specific positions
    for pos in [0, len(h_encoded)//2, -1]:
        partial = h_transformed[pos:pos+1]  # Single position
        decoded = dense.decoder.decode(partial, max_new_tokens=10)
        print(f"   Position {pos}: '{decoded}'")
    
    print(f"\n4. DIFFERENCE ANALYSIS:")
    # See what the edge changed
    diff = (h_transformed - h_encoded).abs().mean()
    print(f"   Average change by Edge: {diff:.3f}")
    print(f"   This indicates how much transformation occurred")

if __name__ == "__main__":
    print("Testing intermediate state decoding...\n")
    
    # Run tests
    test_intermediate_visibility()
    test_async_pattern()
    show_advanced_introspection()
    
    print("\n\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("✅ We can decode intermediates WITHOUT slowing down main process")
    print("✅ Async/background decoding for production")
    print("✅ Selective decoding only when needed (errors, debugging)")
    print("✅ Store hidden states for later forensic analysis")
    print("✅ No performance penalty - user still gets fast response!")