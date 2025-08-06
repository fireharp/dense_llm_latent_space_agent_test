# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements DSPy agents that communicate via dense hidden-state tensors (896-dimensional vectors) instead of text prompts, achieving 90%+ token reduction while maintaining reasoning quality. The system uses Qwen2.5-0.5B-Instruct as the base model with optional Groq API integration for fast final decoding.

## Development Commands

```bash
# Environment setup (ALWAYS run first)
source .venv/bin/activate
source .env  # Contains GROQ_API_KEY and HUGGINGFACE_API_KEY

# Alternative: Use uv run (automatically handles venv and .env)
uv run python <script.py>

# Package management - ALWAYS use uv
uv add <package-name>  # DO NOT use pip install

# Run tests
./run_tests.sh  # Complete test suite
python test_basic.py  # Component tests
python test_integration.py  # Integration tests
python performance_test.py  # Performance analysis

# Interactive demos
python run.py --use-groq  # Dense mode with Groq
python run.py --mode prompt  # Traditional text mode
python demo_pipeline.py  # Full system showcase

# Evaluation comparisons
python eval_react_simple.py  # Dense vs baseline ReACT
python test_strawberry_dense.py  # Detailed token comparison

# Training (edge module only)
python mock_train.py --epochs 5  # Quick test with mock
python train.py --train-size 200 --epochs 3  # Real training
```

## Architecture & Key Concepts

### Core Innovation
The system keeps all intermediate reasoning in 896-dimensional hidden space instead of generating text at each step. Only the final answer is decoded to text.

### Processing Flow
```
Traditional: Text → LLM → Text → LLM → Text → LLM → Answer (50+ tokens)
Dense:       Text → Hidden → Edge → Hidden → Edge → Answer (4 tokens)
```

### Key Components

1. **DenseLM** (`dense_lm.py`, `dense_lm_v2.py`)
   - Wraps Qwen2.5-0.5B-Instruct with `encode()`, `forward()`, `decode()` methods
   - Hidden dimension: 896
   - Handles text ↔ hidden state conversion

2. **DenseEdge** (`dense_edge.py`)
   - 2-layer TransformerEncoder (19.3M parameters)
   - Transforms hidden states between reasoning steps
   - The only trainable component (LMs stay frozen)

3. **Communication Patterns**
   - **PlanSolve** (`plansolve.py`): Linear pipeline - Planner → Edge → Solver
   - **ReACT** (`dense_react.py`): Iterative reasoning loop in hidden space
   - **CoT** (`dense_cot.py`): Multi-step chain-of-thought reasoning

### MockLM for Testing
- `mock_lm.py` provides instant testing without model downloads
- Same interface as DenseLM but lightweight
- Use for rapid prototyping and CI/CD

## Implementation Guidelines

### When adding new dense patterns:
1. Encode text input once at the beginning
2. Keep all intermediate processing in hidden space (896-dim tensors)
3. Only decode to text for the final output
4. Compare against a baseline text version for evaluation

### Model selection:
- Use `Qwen/Qwen2.5-0.5B-Instruct` (not older Qwen2-0.5B)
- Hidden size must be 896 for compatibility
- Chat template improves instruction following

### Performance expectations:
- Dense approaches should achieve 85-95% token reduction
- Processing time dominated by final text generation (~20ms/token)
- Hidden state operations are fast (<10ms total)

### Debugging hidden states:
- Use `test_intermediate_decoding.py` patterns for visibility
- Decode intermediates asynchronously to avoid latency impact
- Store hidden states for offline analysis when needed

## Common Pitfalls

1. **Model loading**: Use `device="cpu"` if GPU memory issues
2. **Import errors**: Ensure virtual environment is activated
3. **Qwen version**: Must use Qwen2.5-0.5B-Instruct (not Qwen2-0.5B)
4. **Token counting**: Only count decoded text, not hidden state sizes

## API Keys

Environment variables in `.env`:
- `GROQ_API_KEY`: For fast final decoding with Llama3-8B
- `HUGGINGFACE_API_KEY`: For model downloads (not needed for inference)