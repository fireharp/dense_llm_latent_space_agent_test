"""MockLM: Lightweight LM replacement for testing without model loading delays."""

import torch
import torch.nn as nn
import dspy
from typing import Optional, List
import numpy as np
import re


class MockLM(dspy.LM):
    """Mock language model that simulates Qwen2-0.5B interface without actual model loading."""
    
    def __init__(
        self,
        model_name: str = "mock/qwen2-0.5b",
        device: Optional[str] = None,
        hidden_size: int = 896,  # Match Qwen2-0.5B
        vocab_size: int = 32000,
        **kwargs
    ):
        """Initialize MockLM with instant loading."""
        super().__init__(model=model_name)
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Model dimensions
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.model_name = model_name
        
        # Simple embedding layers instead of full model
        self.embeddings = nn.Embedding(vocab_size, hidden_size).to(device)
        self.output_projection = nn.Linear(hidden_size, vocab_size).to(device)
        
        # Initialize with small random weights
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        
        print(f"MockLM initialized instantly! (hidden_size={hidden_size})")
        
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to hidden states using mock embeddings."""
        # Simple tokenization (split by spaces and punctuation)
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        
        # Convert to mock token IDs (hash-based)
        token_ids = [hash(token) % self.vocab_size for token in tokens]
        token_ids = torch.tensor(token_ids, device=self.device)
        
        # Get embeddings
        hidden_states = self.embeddings(token_ids)
        
        # Add some deterministic "processing" based on text
        if "plan" in text.lower():
            hidden_states = hidden_states * 1.1
        if "solve" in text.lower():
            hidden_states = hidden_states * 0.9
            
        return hidden_states  # [T, d]
    
    def forward(self, h: torch.Tensor, num_layers: Optional[int] = None) -> torch.Tensor:
        """Process hidden states (mock transformation)."""
        # Simple transformation to simulate model processing
        h_transformed = h * 1.05 + 0.1
        
        # Add some noise for realism
        noise = torch.randn_like(h) * 0.01
        h_transformed = h_transformed + noise
        
        return h_transformed
    
    def decode(self, h: torch.Tensor, max_new_tokens: int = 256) -> str:
        """Decode hidden states to text (mock generation)."""
        # Project to vocabulary
        logits = self.output_projection(h)  # [T, vocab_size]
        
        # Get average activation
        avg_activation = h.mean().item()
        
        # Generate mock response based on hidden state properties
        if avg_activation > 1.0:
            response_templates = [
                "Let me solve this step by step. First, I'll analyze the problem. Then, I'll calculate the answer. The final answer is {}.",
                "To solve this problem: Step 1: Understand what we have. Step 2: Apply the operation. Step 3: The result is {}.",
                "Breaking down the problem: We start with the given values. After performing the calculation, we get {}."
            ]
        else:
            response_templates = [
                "The answer is {}.",
                "After calculating, the result is {}.",
                "The solution is {}."
            ]
            
        # Extract a "number" from the hidden states (mock)
        # Use sum of first few hidden values as seed
        seed = int(abs(h[:5].sum().item() * 100)) % 100
        
        # Try to be somewhat consistent for math problems
        if seed < 20:
            answer = seed
        else:
            answer = seed % 10
            
        template = response_templates[seed % len(response_templates)]
        return template.format(answer)
    
    def basic_request(self, prompt: str, **kwargs) -> str:
        """DSPy-compatible text generation (mock)."""
        import json
        
        # Simple rule-based responses for testing
        prompt_lower = prompt.lower()
        
        # Check if DSPy is asking for JSON output
        if "json" in prompt_lower or "plan" in prompt_lower or "solution" in prompt_lower:
            # Math problem detection and solving
            result = None
            math_patterns = [
                (r'(\d+)\s*\+\s*(\d+)', lambda m: int(m.group(1)) + int(m.group(2))),
                (r'(\d+)\s*-\s*(\d+)', lambda m: int(m.group(1)) - int(m.group(2))),
                (r'(\d+)\s*\*\s*(\d+)', lambda m: int(m.group(1)) * int(m.group(2))),
                (r'(\d+)\s*/\s*(\d+)', lambda m: int(m.group(1)) // int(m.group(2))),
            ]
            
            for pattern, solver in math_patterns:
                match = re.search(pattern, prompt)
                if match:
                    result = solver(match)
                    break
            
            # Word problem handling
            if result is None and ("apples" in prompt_lower or "has" in prompt_lower):
                numbers = re.findall(r'\d+', prompt)
                if len(numbers) >= 2:
                    num1, num2 = int(numbers[0]), int(numbers[1])
                    if "gives" in prompt_lower or "eats" in prompt_lower:
                        result = num1 - num2
                    else:
                        result = num1 + num2
            
            if result is None:
                result = 42
            
            # Return JSON formatted response for DSPy
            response_obj = {}
            
            if "plan" in prompt_lower and "solution" in prompt_lower:
                # ChainOfThought expects both
                response_obj["reasoning"] = "Let me think about this problem step by step."
                response_obj["plan"] = f"1. Identify the numbers\\n2. Perform the calculation\\n3. Verify the result"
                response_obj["solution"] = f"The answer is {result}"
            elif "plan" in prompt_lower:
                response_obj["plan"] = f"To solve this, I'll calculate the result step by step"
            elif "solution" in prompt_lower:
                response_obj["solution"] = f"The answer is {result}"
            else:
                response_obj["answer"] = str(result)
                
            return json.dumps(response_obj)
        
        # Non-JSON response for direct use
        return f"The answer is 42."
    
    def __call__(self, prompt: str = None, messages: list = None, **kwargs) -> list:
        """DSPy-compatible call method."""
        if messages is not None:
            # Convert messages to prompt
            prompt = ""
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    prompt += f"{content}\n"
            prompt = prompt.strip()
            
        if prompt is None:
            raise ValueError("Either prompt or messages must be provided")
            
        response = self.basic_request(prompt, **kwargs)
        return [response]
    
    def get_num_params(self) -> int:
        """Get number of parameters (mock)."""
        # Approximate Qwen2-0.5B size
        return 500_000_000  # 0.5B parameters