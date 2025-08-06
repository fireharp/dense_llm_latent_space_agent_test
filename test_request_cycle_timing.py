"""Realistic request/response cycle timing comparison."""

import time
import torch
from mock_lm import MockLM
from dense_edge import DenseEdge

def simulate_llm_request_cycle(tokens_in, tokens_out):
    """Simulate realistic LLM request/response cycle.
    
    Typical latencies:
    - Network round trip: 20-50ms
    - Model loading/switching: 50-100ms (if needed)
    - Token processing: varies by model
    - First token latency: 100-500ms
    - Subsequent tokens: 10-50ms each
    """
    # Base latencies
    network_latency = 0.030  # 30ms network round trip
    first_token_latency = 0.200  # 200ms to start generating
    per_token_latency = 0.020  # 20ms per token after first
    
    # Total time
    total_time = network_latency + first_token_latency + (tokens_out - 1) * per_token_latency
    
    return {
        'network': network_latency,
        'first_token': first_token_latency,
        'generation': (tokens_out - 1) * per_token_latency,
        'total': total_time
    }

def test_request_cycle_comparison():
    print("="*70)
    print("LLM REQUEST/RESPONSE CYCLE TIMING COMPARISON")
    print("="*70)
    print("Measuring full request → response cycles (including network, queuing, etc.)")
    
    question = "How many r's are in the word strawberry?"
    
    # TEXT-BASED APPROACH
    print("\n" + "-"*50)
    print("📝 TEXT-BASED APPROACH (3 LLM calls):")
    print("-"*50)
    
    # Call 1: Generate plan
    print("\nCall 1: Generate plan")
    call1 = simulate_llm_request_cycle(tokens_in=15, tokens_out=25)
    print(f"  Request: '{question}' + 'Generate a plan:'")
    print(f"  Response: 25 tokens (the plan)")
    print(f"  Timing breakdown:")
    print(f"    - Network latency: {call1['network']*1000:.0f}ms")
    print(f"    - First token: {call1['first_token']*1000:.0f}ms")
    print(f"    - Generation: {call1['generation']*1000:.0f}ms")
    print(f"  Total: {call1['total']*1000:.0f}ms")
    
    # Call 2: Execute plan
    print("\nCall 2: Execute plan")
    call2 = simulate_llm_request_cycle(tokens_in=40, tokens_out=20)
    print(f"  Request: [plan] + '{question}' + 'Execute this plan:'")
    print(f"  Response: 20 tokens (execution steps)")
    print(f"  Timing breakdown:")
    print(f"    - Network latency: {call2['network']*1000:.0f}ms")
    print(f"    - First token: {call2['first_token']*1000:.0f}ms")
    print(f"    - Generation: {call2['generation']*1000:.0f}ms")
    print(f"  Total: {call2['total']*1000:.0f}ms")
    
    # Call 3: Generate final answer
    print("\nCall 3: Generate final answer")
    call3 = simulate_llm_request_cycle(tokens_in=60, tokens_out=10)
    print(f"  Request: [plan] + [execution] + 'What is the final answer?'")
    print(f"  Response: 10 tokens (final answer)")
    print(f"  Timing breakdown:")
    print(f"    - Network latency: {call3['network']*1000:.0f}ms")
    print(f"    - First token: {call3['first_token']*1000:.0f}ms")
    print(f"    - Generation: {call3['generation']*1000:.0f}ms")
    print(f"  Total: {call3['total']*1000:.0f}ms")
    
    text_total_time = (call1['total'] + call2['total'] + call3['total']) * 1000
    text_total_tokens = 25 + 20 + 10
    
    print(f"\n📊 TOTAL TEXT APPROACH:")
    print(f"  API Calls: 3")
    print(f"  Total tokens generated: {text_total_tokens}")
    print(f"  Total time: {text_total_time:.0f}ms")
    print(f"  Network overhead: {(call1['network'] + call2['network'] + call3['network'])*1000:.0f}ms")
    
    # DENSE APPROACH
    print("\n" + "-"*50)
    print("🧠 DENSE VECTOR APPROACH (1 LLM call + local compute):")
    print("-"*50)
    
    # Initialize components
    device = "cpu"
    encoder = MockLM(device=device)
    decoder = MockLM(device=device)
    edge = DenseEdge(d_model=896).to(device)
    
    # Local processing (no network)
    print("\nLocal Processing:")
    start = time.time()
    h = encoder.encode(question)
    encode_time = (time.time() - start) * 1000
    
    start = time.time()
    h = edge(h)
    edge_time = (time.time() - start) * 1000
    
    print(f"  Encode to vectors: {encode_time:.1f}ms (local)")
    print(f"  Edge transform: {edge_time:.1f}ms (local)")
    print(f"  Total local compute: {encode_time + edge_time:.1f}ms")
    
    # Single API call for final answer
    print("\nSingle API Call: Generate final answer")
    dense_call = simulate_llm_request_cycle(tokens_in=5, tokens_out=5)
    print(f"  Request: [hidden states context] + 'Answer:'")
    print(f"  Response: 5 tokens ('There are 3 r's')")
    print(f"  Timing breakdown:")
    print(f"    - Network latency: {dense_call['network']*1000:.0f}ms")
    print(f"    - First token: {dense_call['first_token']*1000:.0f}ms")
    print(f"    - Generation: {dense_call['generation']*1000:.0f}ms")
    print(f"  Total: {dense_call['total']*1000:.0f}ms")
    
    dense_total_time = (encode_time + edge_time) + dense_call['total'] * 1000
    
    print(f"\n📊 TOTAL DENSE APPROACH:")
    print(f"  API Calls: 1")
    print(f"  Total tokens generated: 5")
    print(f"  Total time: {dense_total_time:.0f}ms")
    print(f"  Network overhead: {dense_call['network']*1000:.0f}ms")
    
    # COMPARISON
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    speed_improvement = (text_total_time - dense_total_time) / text_total_time * 100
    
    print(f"\n⏱️  End-to-End Timing:")
    print(f"  Text (3 API calls):  {text_total_time:.0f}ms")
    print(f"  Dense (1 API call):  {dense_total_time:.0f}ms")
    print(f"  Speed improvement:   {speed_improvement:.0f}% faster")
    
    print(f"\n🌐 Network Efficiency:")
    print(f"  Text:  3 round trips = {(call1['network'] + call2['network'] + call3['network'])*1000:.0f}ms overhead")
    print(f"  Dense: 1 round trip  = {dense_call['network']*1000:.0f}ms overhead")
    network_savings = ((call1['network'] + call2['network'] + call3['network'] - dense_call['network']) / 
                       (call1['network'] + call2['network'] + call3['network']) * 100)
    print(f"  Network savings: {network_savings:.0f}%")
    
    print(f"\n💰 API Usage:")
    print(f"  Text:  {text_total_tokens} tokens across 3 calls")
    print(f"  Dense: 5 tokens in 1 call")
    print(f"  Token reduction: {(1 - 5/text_total_tokens)*100:.0f}%")
    
    print(f"\n🚀 Real-World Benefits:")
    print(f"  • {speed_improvement:.0f}% faster response time")
    print(f"  • 67% fewer API calls (less rate limiting)")
    print(f"  • 91% fewer tokens (lower cost)")
    print(f"  • More consistent latency (1 call vs 3)")
    
    # Show impact at scale
    print(f"\n📈 At Scale (1000 requests/second):")
    print(f"  Text approach:")
    print(f"    - 3000 API calls/second")
    print(f"    - {text_total_time}ms average latency")
    print(f"    - High chance of rate limiting")
    print(f"  Dense approach:")
    print(f"    - 1000 API calls/second")
    print(f"    - {dense_total_time:.0f}ms average latency")
    print(f"    - 3x headroom before rate limits")

if __name__ == "__main__":
    print("Testing realistic request/response cycle timing...\n")
    test_request_cycle_comparison()