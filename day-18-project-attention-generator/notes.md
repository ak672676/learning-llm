# Day 18: Project — Attention-Based Text Generator

## Why This Matters

You've built every piece separately. Today: **assemble them into a complete, polished text generator**.

This is the Phase 3 capstone — the model you ship as v1 before adding the rest of transformer plumbing (layer norm, residual, MLP) on Day 19.

```
Day 14: Bigram LM — predicts from 1 token
Day 18: Attention LM — predicts from a full window of tokens
        (and uses position info, multiple heads, sampling controls)
Day 25: Full GPT — same code structure, more transformer blocks stacked
```

Today is the bridge. After this, the only thing left for a "real" model is more layers.

---

## What We're Building

A complete attention-based language model with:

1. **Clean, reusable code** (`Tokenizer`, `Config`, `AttentionLM`, `Trainer`)
2. **Train/val split** with proper held-out loss
3. **Position-aware multi-head attention** (everything from Days 15-17)
4. **Multiple sampling strategies** (temperature, top-k)
5. **Model save/load** for production-style use
6. **Inference function** that prints generated text
7. **Side-by-side comparison** with the Day 14 bigram

This is the difference between "I built a thing in a notebook" and "I built a thing I could put on GitHub."

---

## The Architecture

```
Token IDs                              (B, T)
     │
     ▼ token embedding
embeddings                             (B, T, D)
     │
     ▼ + position embedding             ← Day 17
positioned embeddings                  (B, T, D)
     │
     ▼ multi-head causal attention      ← Day 15-16
attended                               (B, T, D)
     │
     ▼ output linear head
logits over vocab                      (B, T, V)
     │
     ▼ cross-entropy loss (training)
     │  or softmax + sample (generation)
     ▼
prediction or next token
```

Five lines of architecture. That's a complete, working language model. The same skeleton scales up to GPT-3 — just deeper, wider, more data.

---

## Sampling Strategies

When the model produces logits over the vocabulary, we have to pick a token. Several options:

### 1. Argmax (greedy)
Take the most likely token every time.
```python
next_token = logits.argmax()
```
- Deterministic, fast
- Often gets stuck in loops ("the the the the...")
- Used when you want consistent output

### 2. Multinomial sampling
Sample proportional to probability.
```python
probs = softmax(logits)
next_token = torch.multinomial(probs, num_samples=1)
```
- Diverse output
- What we used in Days 13, 14
- Sometimes picks weird low-prob tokens

### 3. Temperature scaling
Divide logits by temperature before softmax.
```python
probs = softmax(logits / T)
```
- T = 1.0 → normal
- T < 1.0 → more confident, less diverse ("sharper")
- T > 1.0 → less confident, more diverse ("flatter")
- Real LLMs expose this as a parameter

### 4. Top-K sampling
Only sample from the K most likely tokens.
```python
top_k_logits, top_k_indices = logits.topk(k)
# Sample from the top K
```
- Prevents weird low-probability picks
- Standard in modern LLMs (Llama, GPT)

### 5. Top-P (nucleus) sampling
Sample from the smallest set of tokens whose probabilities sum to ≥ P.
- Adaptive: in confident moments only takes a few; in uncertain moments takes more
- Used in production LLMs

We'll implement temperature + top-K today. Top-P is straightforward (left as exercise).

---

## Train / Val Split for Language Models

A bit different from classifier train/val:
- Take the FIRST 90% of tokens as train
- Use the LAST 10% as validation
- Sample random windows from each

Why not a fully random split? Because **language is sequential** — splitting randomly would let the model peek at the future and "remember" sentences.

```
Corpus: [t0, t1, t2, ..., t999]
                 ↓
   train: t0..t899 (sample random windows)
   val:   t900..t999 (sample random windows)
```

Both losses should drop as the model learns. If train loss drops but val loss rises → overfitting.

---

## What Makes a Project Different From a Notebook

We've been writing notebooks. Today we'll write **as if it were a real project** — code that someone else could pick up and extend.

| Notebook style | Project style |
|----------------|---------------|
| Hardcoded magic numbers | `Config` class with named parameters |
| All globals | Functions and classes with clear contracts |
| Each cell does one thing in isolation | Layers compose, modules can be reused |
| `# TODO: figure out later` | Tested, documented behavior |

Even with our small dataset, this pays off. You can change ONE config value and re-run; you can extract the model class and use it elsewhere; you can save artifacts and load them later.

---

## The 6-Step Production Workflow

Every real ML project does these things. Today we'll do all of them:

1. **Configure** — single `Config` object holding all hyperparameters
2. **Tokenize** — convert text to/from IDs
3. **Split & batch** — train/val, then sample windows
4. **Define model** — `nn.Module` subclass
5. **Train** — loop with val-loss tracking, save best
6. **Generate / inference** — load the model and use it on demand

---

## What's Still Missing (For Day 19+)

After today, your generator works! But compared to a real transformer, you're missing:

| Missing | What it adds | Day |
|---------|-------------|-----|
| **Residual connections** | Stable training of many layers | 19 |
| **LayerNorm** | Stable training of deep nets | 19 |
| **Feed-forward MLP** | More capacity per block | 19 |
| **Multiple stacked blocks** | Deeper model = more capability | 20 |
| **BPE tokenization** | Better than char-level | 21 |
| **Real corpus** | Bigger, more diverse data | 22 |

We've got the HARD ideas (Q/K/V, multi-head, causal mask, position encoding). The remaining additions are mostly plumbing.

---

## Mental Model

```
Day 14 (bigram):
   "see one previous token → predict next"
   "see 'e' → predict probabilities over all chars"

Day 18 (attention generator):
   "see WINDOW of previous tokens → predict next"
   "see 'to be or not to b' → predict probabilities over all chars"
   "use position info + multi-head attention to pick which words matter"

Day 25 (full GPT):
   Same skeleton as Day 18, but with 12+ transformer blocks
   instead of one attention layer. And trained on more data.
```

Day 18 IS the architecture of a transformer. It's just that the "block" is a single attention layer; tomorrow's transformer block adds normalization and an MLP to make stacking many of them feasible.

---

## Summary

| Skill | Where it came from |
|-------|---------------------|
| Tokenizer | Day 4, 10 |
| nn.Module | Day 9 |
| Embeddings | Day 11 |
| RNN (for contrast) | Day 13 |
| Bigram task framework | Day 14 |
| Self-attention | Day 15 |
| Multi-head attention | Day 16 |
| Position encoding | Day 17 |
| Train/val tracking | Day 5, 8 |
| Sampling + temperature | Day 13, 14 |
| Top-K sampling | NEW today |
| Save/load artifacts | Day 8, 9, 12 |

### Where we are

```
Phase 3 (sequence models)
├── Day 13: RNN                              ✓
├── Day 14: Bigram LM                        ✓
├── Day 15: Self-attention                   ✓
├── Day 16: Multi-head attention             ✓
├── Day 17: Positional encoding              ✓
└── Day 18: PROJECT — attention generator    ← YOU ARE HERE

Phase 4 (transformers)
├── Day 19: Transformer block (LayerNorm + MLP + residual)
├── Day 20: Mini GPT (stack of blocks)
├── Day 21: BPE tokenization
├── Day 22: Train mini GPT for real
├── Day 23: Sampling strategies (top-K, top-P)
├── Day 24: Evaluation (perplexity, examples)
└── Day 25: PROJECT — your own GPT on custom data
```

We're past the hard ideas. Day 18 packages them. Day 19 starts assembling them into the canonical transformer.

**Tomorrow:** Day 19 — the transformer block. Layer norm, residual connections, and the MLP layer that makes stacking work. Once we have that block, "transformer" = "stack N of these blocks."
