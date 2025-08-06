"""DenseLM: Qwen2-0.5B wrapper with hidden state manipulation for DSPy."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import dspy
from typing import Optional, Union


class DenseLM(dspy.LM):
    """Wrapper around Qwen2-0.5B that exposes encode/forward/decode for hidden state manipulation."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        **kwargs
    ):
        """Initialize DenseLM with Qwen2 model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (cuda/cpu/auto)
            load_in_8bit: Whether to load model in 8-bit precision
            **kwargs: Additional arguments for AutoModelForCausalLM
        """
        super().__init__(model=model_name)
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Model loading with device map for multi-GPU
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "auto" else None,
            "load_in_8bit": load_in_8bit,
            **kwargs
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if device not in ["auto", "cuda"] or not load_in_8bit:
            self.model = self.model.to(device)
            
        self.model.eval()  # Set to eval mode
        
        # Get model dimensions
        self.hidden_size = self.model.config.hidden_size  # 896 for Qwen2-0.5B
        self.vocab_size = self.model.config.vocab_size
        
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to hidden states.
        
        Args:
            text: Input text to encode
            
        Returns:
            Hidden state tensor of shape [T, d] where T is sequence length, d is hidden size
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            # Get input embeddings
            input_embeds = self.model.get_input_embeddings()(inputs.input_ids)
            
            # Forward through model to get hidden states
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Return last hidden state, squeeze batch dimension
            hidden_states = outputs.hidden_states[-1].squeeze(0)  # [T, d]
            
        return hidden_states
    
    def forward(self, h: torch.Tensor, num_layers: Optional[int] = None) -> torch.Tensor:
        """Process hidden states through model layers.
        
        Args:
            h: Hidden state tensor of shape [T, d]
            num_layers: Number of layers to process through (None = all layers)
            
        Returns:
            Processed hidden state tensor of shape [T, d]
        """
        # Add batch dimension
        h = h.unsqueeze(0)  # [1, T, d]
        
        # Create attention mask (all ones since we're processing hidden states)
        attention_mask = torch.ones(h.shape[:2], device=self.device)
        
        with torch.no_grad():
            # Process through transformer layers
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
            hidden_states = hidden_states.squeeze(0)  # [T, d]
            
        return hidden_states
    
    def decode(self, h: torch.Tensor, max_new_tokens: int = 256) -> str:
        """Decode hidden states to text.
        
        Args:
            h: Hidden state tensor of shape [T, d]
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Decoded text string
        """
        # Add batch dimension
        h = h.unsqueeze(0)  # [1, T, d]
        
        with torch.no_grad():
            # Continue generation from hidden states
            outputs = self.model.generate(
                inputs_embeds=h,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            
            # Decode the generated tokens
            generated_ids = outputs.sequences[0]
            decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        return decoded_text
    
    def basic_request(self, prompt: str, **kwargs) -> str:
        """DSPy-compatible text generation method.
        
        This is used when the module operates in standard text mode.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.95),
                do_sample=True,
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()
        
        return response
    
    def __call__(self, prompt: str = None, messages: list = None, **kwargs) -> list:
        """DSPy-compatible call method."""
        # Handle both prompt and messages format
        if messages is not None:
            # Convert messages to a single prompt
            prompt = ""
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt += f"System: {content}\n"
                    elif role == "user":
                        prompt += f"User: {content}\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n"
                else:
                    prompt += str(msg) + "\n"
            prompt = prompt.strip()
        
        if prompt is None:
            raise ValueError("Either prompt or messages must be provided")
            
        response = self.basic_request(prompt, **kwargs)
        return [response]