 NeuralGPT

<div align="center">


<br/>

> **A GPT-style Large Language Model built in pure PyTorch**


<br/>


</div>

---

##  What is NeuralGPT?

Everyone uses LLMs. Few understand what is actually happening inside them.

I studied Sebastian Raschka's *Build a Large Language Model From Scratch* and re-implemented its GPT architecture and training pipeline to understand transformers at the code level.

NeuralGPT is an attempt to close that gap: a GPT-style transformer language model implemented using pure PyTorch, covering every core component — the tokenizer, the token embeddings, positional encodings, multi-head self-attention with causal masking, feed-forward layers, layer normalization, and the full autoregressive text generation loop.

Most importantly: no HuggingFace Transformers, no pre-built attention modules.

The goal was not to build something that competes with GPT-4. The goal was to build something that made me understand exactly what GPT-4 is doing - mathematically, architecturally, and computationally. After building NeuralGPT, I can explain from first principles why attention works, what residual connections solve, and how a model generates the next token one step at a time.

It ships with a Chainlit-powered web UI so you can chat with your trained model directly in the browser.

---

##  Architecture

```
  Raw Text Input
       │
       ▼
┌──────────────────────────────┐
│ |  GPT-2 BPE vocabulary      
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│     Token Embedding Table    │  shape: (vocab_size, d_model)
│   +                          │
│     Positional Encodings     │  learned position vectors
└──────────────────────────────┘
       │
       ▼  ──────────────── repeated N times ────────────────
┌──────────────────────────────────────────────────────────┐
│                   Transformer Block                      │
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │         Multi-Head Self-Attention                │   │
│   │                                                  │   │
│   │  Input ──▶  Q, K, V linear projections           │   │
│   │             Scaled dot-product attention         │   │
│   │             QK^T / √d_k  →  softmax  →  × V     │   │
│   │             Causal mask (upper-triangular -∞)    │   │
│   │             Concat heads  →  output projection   │   │
│   └──────────────────────────────────────────────────┘   │
│                      + Residual connection               │
│                        LayerNorm                         │
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │         Position-wise Feed-Forward Network       │   │
│   │                                                  │   │
│   │  Linear(d_model → 4*d_model) → ReLU             │   │
│   │  Linear(4*d_model → d_model)                     │   │
│   └──────────────────────────────────────────────────┘   │
│                      + Residual connection               │
│                        LayerNorm                         │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│   Final Linear Projection    │  d_model → vocab_size
│   Logits → Softmax           │
└──────────────────────────────┘
       │
       ▼
  Next Token (greedy / sampling)
```

---

##  Features

| Feature | Description |
|---|---|
|  Pure PyTorch implementation | Every layer coded from scratch — no transformer library abstractions |
|  Multi-head self-attention | Full Q/K/V projection with causal masking for autoregressive generation |
|  Positional encodings | Learned position vectors injected into the embedding space |
|  Residual connections | Skip connections throughout every transformer block for stable deep training |
|  Layer normalization | Pre-norm applied before each sub-layer for training stability |
|  GPT-2 BPE tokenizer | Uses tiktoken's GPT-2 byte-pair encoding for subword tokenization - text to integer ids and back |
|  Autoregressive generation | Temperature + top-k sampling loop (greedy argmax also available via `generate_text_simple`) - model generates one token at a time |
|  Chainlit web UI | Chat with your trained model in a browser ; no CLI required |
|  Model persistence | Save trained weights to disk, reload and resume without retraining |
|  Environment checker | `python_environment_check.py` verifies PyTorch and CUDA before training |

---

##  Tech Stack

| Layer | Technology | Why This Choice |
|---|---|---|
| Language | Python 3.10+ | Standard for deep learning - extensive PyTorch ecosystem |
| Deep Learning | PyTorch | Dynamic computation graphs make implementing custom architectures natural |
| Chat Interface | Chainlit | Builds a production-quality chat UI around any Python function in minutes |
| Package Manager | uv | Dramatically faster than pip - virtual environment and installs in seconds |

---

##  Project Structure

```
NeuralGPT/
│
├── app.py                       # Chainlit chat UI - loads model, handles message loop
├── model.py                     # GPT architecture + inference helpers (imported by both app.py and save_model.py)
├── save_model.py                # Training entry point - downloads GPT-2 weights, fine-tunes, saves weights (only runs via `python save_model.py`)
├── python_environment_check.py  # Pre-flight check - verifies Python, PyTorch, CUDA
├── instruction-data.json        # Training data for instruction fine-tuning
├── chainlit.md                  # Chainlit welcome screen markdown
├── requirements.txt             # All Python dependencies with minimum versions pinned
├── .gitignore                   # Excludes model weights and data directories
└── README.md

# Generated locally after running save_model.py - not committed to the repo
# gpt2/                          # Downloaded pretrained GPT-2 checkpoint files
# <model-name>-sft.pth           # Fine-tuned model weights, saved to project root

> `gpt2/` and the final `.pth` weights file are excluded via `.gitignore` - the checkpoint is downloaded and the fine-tuned weights are saved locally when you run the training script. This keeps the repo lightweight while the actual weights live where they belong: on your hardware.
---

##  Getting Started

### Prerequisites

- Python 3.10+
- pip (to install uv)
- A GPU is helpful but not required - the model will train on CPU, just slower

### 1. Clone the repository

```bash
git clone https://github.com/ayushhh-here/NeuralGPT.git
cd NeuralGPT
```

### 2. Install uv

```bash
pip install uv
```

### 3. Create a virtual environment

```bash
uv venv --python=python3.10
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 4. Install all dependencies

```bash
uv pip install packaging
uv pip install -r requirements.txt
```

### 5. Verify your environment

```bash
python python_environment_check.py
```

This script checks Python version, PyTorch installation, and CUDA availability. Resolve any issues it flags before proceeding to training.

---

##  Running the Project

### Step 1 - Train the model and save weights

```bash
python save_model.py
```

This runs the full training loop on `instruction-data.json` and saves the resulting model weights as a .pth file in the project root, and downloads GPT-2 checkpoints into gpt2/.
### Step 2 - Launch the chat interface

```bash
chainlit run app.py
```

Chainlit starts a local web server and opens the chat UI in your browser automatically. Type any prompt and watch your model generate a response token by token.

---

## What I Learned Building This

**Why attention actually works - mathematically**
I get the scaled dot-product attention not as a magic black box but as a learned similarity function. The query-key dot product measures how relevant each token is to every other token. Scaling by √d_k prevents the dot products from growing too large and pushing softmax into regions with near-zero gradients. 

**What positional encodings solve and why**
Transformers process all tokens in parallel - they have no inherent notion of sequence order. Without positional encodings, the model would treat "the dog bit the man" and "the man bit the dog" identically. Addition of position vectors to the embeddings injects order information directly into the representation space.

**Residual connections and the vanishing gradient problem**
Without skip connections, gradients in deep networks vanish exponentially as they propagate back through many layers. Residual connections provide a direct gradient highway back to early layers. I observed this directly by training shallow vs. deep configurations without residuals and watching training diverge.

**Autoregressive generation loop**
Generation is simply the repeated forward passes. At each step, pass the full context so far through the model, take the logits for the final position, sample or take the argmax, append the new token, and repeat. Implementing this loop — and then adding temperature scaling and top-k filtering on top — made the entire concept of "text generation" concrete and demystified.

**PyTorch internals — tensors, autograd, training loops**
I'm now genuinely comfortable with tensor shapes and broadcasting, writing custom training loops with manual gradient zeroing, gradient clipping to prevent exploding gradients, and understanding what `.backward()` actually computes. Libraries like HuggingFace take all of this away.  
---

##  Roadmap

- [ ] Train on a larger and more diverse text corpus
- [x] Add temperature and top-k sampling to the generation loop
- [ ] Deploy a lightweight model version to HuggingFace Spaces
- [ ] Experiment with different depth and width configurations
- [ ] Align inference prompt format exactly with training format for better response quality
- [ ] Increase max_new_tokens for longer, more complete responses

---

## 📄 License

MIT — use it, fork it, break it, learn from it.

---


(resource used : "Build a Large Language Model From Scratch"-Sebastian Raschka)

(This repo is MIT licensed for my own code. python_environment_check.py is adapted from Sebastian Raschka's LLMs-from-scratch and remains under its original Apache-2.0 license)

<div align="center">

Built by [ayushhh-here](https://github.com/ayushhh-here) · 3rd year B.Tech · NIT Agartala

*cheers !!!*

</div>
