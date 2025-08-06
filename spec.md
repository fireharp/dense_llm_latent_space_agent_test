# SPEC ‖ Dense-Vector DSPy Agent Prototype  
*Last updated : 2025-08-01*

---

## 1 ✔ Purpose
Create a reference implementation of **DSPy** agents that communicate via **hidden-state tensors** instead of prompt strings, using **Qwen 2-0.5 B** vertices.  
The prototype must:

1. Match or exceed a prompt-string baseline on **GSM8K** (exact-match).  
2. Emit **≤ 10 %** of the tokens the baseline emits.  
3. Keep the LLM vertices frozen; only fine-tune the inter-module “edge”.  
4. Support optional final decoding through **Groq Cloud** for ≥ 300 req/s.

---

## 2 📐 Scope
| In scope | Out of scope |
|----------|--------------|
| DenseLM / DenseEdge wrappers | Full LMNet fine-tuning |
| Edge-only training on a small split (≤ 200 train, 500 dev) | Multi-GPU distributed training |
| Baseline prompt-string pipeline | UI / orchestration tooling |
| CLI scripts for eval & run | Production infra, auth, monitoring |

---

## 3 🏗 Architecture

```
[user] → Planner(DenseLM) → Edge(DenseEdge) → Solver(DenseLM) → .decode() → [answer]
                     (hidden-state tensors, no tokens in flight)
```

* **DenseLM** – Qwen 0.5 B with `encode`, `forward`, `decode`.  
* **DenseEdge** – 2-layer `TransformerEncoder` (`d_model = 4096`).  
* **PlanSolve module** – chains Planner → Edge → Solver.  

---

## 4 🔧 Key Interfaces (pseudo-code)

```python
class DenseLM(dspy.LM):
    encode(text:str)->Tensor[T,d]
    forward(h:Tensor[T,d])->Tensor[T,d]
    decode(h:Tensor[T,d])->str

class DenseEdge(nn.Module):
    forward(h)->h′           # maintains (T,d)

class PlanSolve(dspy.Module):
    forward(goal:str)->str   # planner▸edge▸solver
```

---

## 5 📚 Dependencies

* **Python ≥ 3.10**  
* `torch`, `transformers`, `dspy`, `accelerate`  
* Model: `Qwen/Qwen2-0.5B` (HF Hub)  
* Optional: Groq API key for `POST /chat/completions`

---

## 6 🗄 Datasets & Splits

| Dataset | Train | Dev | Metric |
|---------|-------|-----|--------|
| GSM8K (DSPy built-in) | 200 | 500 | Exact-match |

---

## 7 ⚙ Training Procedure

1. **Load** PlanSolve; freeze both DenseLM vertices.  
2. **Optimiser**: AdamW, lr = 2e-4, weight_decay = 0.01.  
3. **Epochs**: 3 (early-stop on dev accuracy).  
4. **Saved artefact**: `edge_state_dict.pt` (≈ 17 MB).

---

## 8 📊 Evaluation Plan

| Run ID | Planner-Edge-Solver | Dev accuracy | Avg emitted tokens |
|--------|--------------------|--------------|--------------------|
| **B0** | Prompt-string (baseline) | target ≥ XX % | baseline N |
| **D1** | DenseEdge (edge only) | must be ≥ B0 – 1 % | **≤ 0.1 × N** |

*Scripts*:
```bash
python eval.py --module PlanSolve    # dense
python eval.py --module PromptPlanSolve  # baseline
```

Success = D1 meets both accuracy & token budget.

---

## 9 🚀 Inference / Deployment

* **Hidden-state hops** require a backend that permits `inputs_embeds`; run on a single A100 40 GB or HF Endpoint.  
* **Final decode** can be proxied to **Groq Cloud**:  
  1. `txt = DenseLM.decode(h_solver)`  
  2. `groq.chat(model="llama3-8b-8192", messages=[{"role":"user","content":txt}])`  

Optional flag `--use-groq` in `run.py`.

---

## 10 📦 Deliverables

| File / Dir | Description |
|------------|-------------|
| `dense_lm.py` | `DenseLM` wrapper |
| `dense_edge.py` | `DenseEdge` module |
| `plansolve.py` | DSPy chain |
| `train.py` | edge-only fine-tune script |
| `eval.py` | baseline vs. dense evaluator |
| `run.py` | CLI runner with `--use-groq` |
| `edge_state_dict.pt` | trained weights |
| `README.md` | this spec + quickstart |

---

## 11 ⚠ Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Hidden-state tensors too large for RPC | Add mean-pool or 2-× linear projection |
| DenseEdge underfits | Allow 4 layers or unfreeze last LM block |
| Groq returns inconsistent completions | Fallback to local `.decode()` |

---

## 12 🔗 References

1. Wu et al., “**Dense Communication between Language Models**”, ICML 2025.  
2. DSPy repo – <https://github.com/stanfordnlp/dspy>  
3. Qwen 2-0.5 B – <https://huggingface.co/Qwen/Qwen2-0.5B>  
4. Groq Cloud docs – <https://console.groq.com/docs/api-reference>

---