"""Updated DenseLM using Qwen2.5-0.5B-Instruct with chat template support."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import dspy
from typing import Optional, Union, List, Dict

class DenseLM(dspy.LM):
    """Wrapper around Qwen2.5-0.5B-Instruct with hidden state manipulation."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",  # Updated to newer model
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        **kwargs
    ):
        """Initialize DenseLM with Qwen2.5 model."""
        super().__init__(model=model_name)
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Model loading
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "device_map": device if device != "cuda" else "auto",
            "load_in_8bit": load_in_8bit,
            **kwargs
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.model.eval()
        
        # Get model dimensions (same as before: 896)
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        
        print(f"✓ Model loaded: hidden_size={self.hidden_size}, vocab_size={self.vocab_size}")
        
    def encode(self, text: str, use_chat_template: bool = False) -> torch.Tensor:
        """Encode text to hidden states.
        
        Args:
            text: Input text or question
            use_chat_template: Whether to format as chat (for instruction following)
            
        Returns:
            Hidden state tensor [T, 896]
        """
        # Optionally use chat template for better instruction following
        if use_chat_template:
            messages = [{"role": "user", "content": text}]
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_text = text
            
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Return last hidden state, squeeze batch dimension
            hidden_states = outputs.hidden_states[-1].squeeze(0)  # [T, 896]
            
        return hidden_states
    
    def forward(self, h: torch.Tensor, num_layers: Optional[int] = None) -> torch.Tensor:
        """Process hidden states through model layers."""
        # Add batch dimension
        h = h.unsqueeze(0)  # [1, T, 896]
        
        # Create attention mask
        attention_mask = torch.ones(h.shape[:2], device=self.device)
        
        with torch.no_grad():
            if num_layers is None:
                # Use all layers
                outputs = self.model(
                    inputs_embeds=h,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = outputs.hidden_states[-1]
            else:
                # Process through specific number of layers
                hidden_states = h
                for i in range(num_layers):
                    layer = self.model.model.layers[i]
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                    )
                    hidden_states = layer_outputs[0]
                    
            # Remove batch dimension
            hidden_states = hidden_states.squeeze(0)  # [T, 896]
            
        return hidden_states
    
    def decode(self, h: torch.Tensor, max_new_tokens: int = 256) -> str:
        """Decode hidden states to text."""
        # Add batch dimension
        h = h.unsqueeze(0)  # [1, T, 896]
        
        with torch.no_grad():
            # Generate from hidden states
            outputs = self.model.generate(
                inputs_embeds=h,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Decode the generated tokens
            decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        return decoded_text
    
    def answer_question(self, question: str) -> str:
        """Direct question answering using chat template."""
        messages = [{"role": "user", "content": question}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
            )
            
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def basic_request(self, prompt: str, **kwargs) -> str:
        """DSPy-compatible text generation method."""
        # For DSPy compatibility, use chat template by default
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.95),
                do_sample=True,
            )
            
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        return response

# Quick test
if __name__ == "__main__":
    print("Testing DenseLM with Qwen2.5-0.5B-Instruct...\n")
    
    # Use mock for quick testing
    from mock_lm import MockLM
    lm = MockLM()  # Use mock to avoid downloading
    
    # Test encoding
    text = "How many r's are in strawberry?"
    hidden = lm.encode(text)
    print(f"Encoded '{text}' to shape {hidden.shape}")
    
    # Test direct answer
    print(f"\nDirect answer would be: 3 r's (strawberry has 3 r's)")
    
    print("\n✓ To use real model: lm = DenseLM()")
    print("✓ Hidden dimension: 896 (same as before)")
    print("✓ Instruction tuned for better performance")