# Learning LLM: From Zero to Building Your Own

A hands-on, day-by-day journey from Python basics to building and running a small LLM.

45 days. Each day is a folder with a Jupyter notebook, notes, and working code.

Each day folder contains:
- `notebook.ipynb` — main hands-on notebook (concepts + code together)
- `notes.md` — summary of key concepts for quick revision
- `code/` — standalone scripts if needed

---

## Phase 1: Python & Math Foundations (Day 01-05)

| Day | Topic | What You Build |
|-----|-------|----------------|
| 01 | Python for ML — NumPy, matrix ops, vectorization | Matrix calculator, dot product visualizer |
| 02 | Calculus intuition — derivatives, chain rule, gradients | Gradient descent from scratch to find minimum |
| 03 | PyTorch basics — tensors, autograd, GPU | Rewrite Day 02 in PyTorch, see autograd magic |
| 04 | Data handling — datasets, batching, tokenization basics | Text file reader that batches data for training |
| 05 | Plotting & debugging — matplotlib, loss curves, common errors | Training dashboard utility |

## Phase 2: Neural Networks from Scratch (Day 06-12)

| Day | Topic | What You Build |
|-----|-------|----------------|
| 06 | Single neuron — weights, bias, activation | A neuron that learns AND/OR gates |
| 07 | Multi-layer network — forward pass, backprop by hand | 2-layer network from scratch (no PyTorch) |
| 08 | Training loop — loss functions, optimizers, learning rate | Train on a toy dataset, plot the loss curve |
| 09 | Same network in PyTorch — nn.Module, nn.Linear | Compare your scratch version vs PyTorch |
| 10 | Classifying text — bag of words, simple text classifier | Spam/not-spam classifier |
| 11 | Word embeddings — what are they, Word2Vec intuition | Train tiny embeddings on a small corpus |
| 12 | Project: Sentiment classifier | End-to-end: data → tokenize → embed → classify → evaluate |

## Phase 3: Sequence Models & Attention (Day 13-18)

| Day | Topic | What You Build |
|-----|-------|----------------|
| 13 | Why sequences matter — RNN intuition, vanishing gradients | Character-level name generator with RNN |
| 14 | Bigram language model — predict next character | Bigram model that generates text |
| 15 | Self-attention — the core idea, scaled dot-product | Single attention head from scratch |
| 16 | Multi-head attention — why multiple heads, concatenation | Multi-head attention module |
| 17 | Positional encoding — why position matters, sinusoidal | Add position info, see the difference |
| 18 | Project: Attention-based text generator | Combine Day 14-17 into a working generator |

## Phase 4: Build a Transformer (Day 19-25)

| Day | Topic | What You Build |
|-----|-------|----------------|
| 19 | Transformer block — LayerNorm, residual connections, FFN | One transformer block |
| 20 | Stacking blocks — full decoder-only transformer | Mini GPT architecture (< 200 lines) |
| 21 | Tokenization deep dive — BPE, SentencePiece | Train your own tokenizer on a corpus |
| 22 | Training your mini GPT — on Shakespeare dataset | Train and generate Shakespeare-like text |
| 23 | Sampling strategies — temperature, top-k, top-p | Interactive text generator with knobs |
| 24 | Evaluation — perplexity, loss analysis, overfitting | Evaluate and improve your model |
| 25 | Project: Your own GPT trained on custom data | Pick a dataset, train, generate, document |

## Phase 5: Hugging Face & Pretrained Models (Day 26-28)

| Day | Topic | What You Build |
|-----|-------|----------------|
| 26 | Hugging Face ecosystem — transformers, datasets, pipeline | Load and use GPT-2 for text generation |
| 27 | Exploring pretrained models — model hub, tokenizers, configs | Compare outputs of different models (GPT-2, Phi, Qwen) |
| 28 | Text generation pipeline — batching, padding, inference | Build a reusable generation script with HF |

## Phase 6: Fine-Tuning (Day 29-33)

| Day | Topic | What You Build |
|-----|-------|----------------|
| 29 | Full fine-tuning — what it is, why it's expensive | Fine-tune GPT-2 small on a custom dataset |
| 30 | LoRA — low-rank adaptation, why it works | Implement LoRA from scratch on a small model |
| 31 | QLoRA & PEFT library — 4-bit fine-tuning | Fine-tune a 1-3B model on your Mac using QLoRA |
| 32 | Dataset preparation — formatting, chat templates, data quality | Build a training dataset from raw text/conversations |
| 33 | Project: Fine-tune your own model | End-to-end: pick a base model → prepare data → LoRA train → evaluate |

## Phase 7: Alignment & Instruction Tuning (Day 34-36)

| Day | Topic | What You Build |
|-----|-------|----------------|
| 34 | SFT — supervised fine-tuning, instruction-response format | Turn a base model into an instruction follower |
| 35 | Reward models & DPO — preference learning, why RLHF matters | Train a simple DPO on chosen/rejected pairs |
| 36 | Evaluation & comparison — benchmarks, human eval, vibe checks | Compare base vs SFT vs DPO outputs side by side |

## Phase 8: Quantization & Local Inference (Day 37-38)

| Day | Topic | What You Build |
|-----|-------|----------------|
| 37 | Quantization — INT8, INT4, GGUF format, why it works | Quantize your fine-tuned model to GGUF |
| 38 | Local inference — llama.cpp, ollama, MLX | Run your own model locally with ollama |

## Phase 9: RAG & Applications (Day 39-42)

| Day | Topic | What You Build |
|-----|-------|----------------|
| 39 | Embeddings & vector search — what embeddings capture, similarity | Build a semantic search over your own documents |
| 40 | RAG pipeline — retrieval + generation, chunking strategies | RAG chatbot that answers from your docs |
| 41 | Serving your model — FastAPI, streaming responses | REST API that serves your local model |
| 42 | Tool use & structured output — function calling, JSON mode | Agent that can call tools and return structured data |

## Phase 10: Final Project (Day 43-45)

| Day | Topic | What You Build |
|-----|-------|----------------|
| 43 | Architecture & planning | Design your final app, pick components |
| 44 | Build | Full local AI app — chat UI + your fine-tuned model + RAG + tools |
| 45 | Polish & document | Clean up, write docs, push to GitHub |

---

## How to Use This Repo

```bash
# Each day
cd day-XX-topic-name/
cat notes.md          # Read the concepts
cd code/
python main.py        # Run the code
```

## Setup

> **Note:** This project uses `pyenv local` so Python 3.11.13 is only active inside this folder. Your system Python stays unchanged.

### Prerequisites

- [pyenv](https://github.com/pyenv/pyenv) installed
- Python 3.11.13 installed via pyenv (`pyenv install 3.11.13`)

### First Time Setup

```bash
cd learning-llm

# pyenv local is already set via .python-version file
# Verify correct Python is active:
python --version  # Should show 3.11.13

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch numpy matplotlib jupyterlab ipykernel

# Register Jupyter kernel
python -m ipykernel install --user --name learning-llm --display-name "Learning LLM (3.11)"
```

### Daily Usage

```bash
cd learning-llm
source venv/bin/activate

# Start Jupyter
jupyter lab

# Or run a script directly
python day-01-python-for-ml/code/main.py
```

### Later Phases (Day 26+)

```bash
pip install transformers datasets sentencepiece
```
