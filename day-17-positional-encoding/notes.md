# Day 17: Positional Encoding — Telling the Model About Order

## Why This Matters

Here's a strange thing about self-attention: **it doesn't know about position**.

```
"the cat sat on the mat"   and   "mat the on sat cat the"
```

To the attention mechanism, these two sequences look IDENTICAL after the Q/K/V projections — same tokens, just reordered. Attention is **permutation invariant** — it sees a *set* of tokens, not a *sequence*.

Obviously that's a disaster for language. "Dog bites man" and "Man bites dog" mean completely different things.

The fix: **add positional information to the embeddings**. After Day 17, your model finally knows what "position 0" vs "position 5" means.

This is the LAST missing piece. After today, Day 18 builds a complete attention generator, and Day 19+ assembles the transformer.

---

## 1. Why Attention is Position-Blind

Let's see it concretely. Self-attention computes:

```
output_i = sum over j of (softmax(Q_i · K_j) · V_j)
```

Notice: nothing in there depends on the *position* of i or j. If you reorder the input tokens, you get the same outputs in a different order — but each token's output is computed the same way.

```
Input  X:  [t0, t1, t2]    →  Output: [o0, o1, o2]
Input  X': [t2, t0, t1]    →  Output: [o2, o0, o1]    (just reordered)
```

This is **mathematically convenient** (we can parallelize) but **language-stupid** (order matters).

---

## 2. The Solution: Inject Position Info

The standard trick: **add a position-specific vector to each token embedding**, BEFORE attention.

```
Original embeddings:                position embeddings:
token[0] = [0.5, -0.2, 0.8, ...]    pos[0] = [0.0, 1.0, 0.0, ...]
token[1] = [0.1,  0.7, 0.3, ...]    pos[1] = [0.1, 0.9, 0.2, ...]
token[2] = [0.6,  0.2, 0.4, ...]    pos[2] = [0.2, 0.8, 0.4, ...]

Combined (just add!):
input[0] = token[0] + pos[0]
input[1] = token[1] + pos[1]
input[2] = token[2] + pos[2]
```

Now each token's representation depends on **what it is** AND **where it is**. Attention can use both.

### Why adding instead of concatenating?

You could concatenate `[token_emb, pos_emb]` instead. But:
- Adding keeps the dimension fixed (input/output dims still match — stackable)
- Adding works empirically just as well or better
- It's what Vaswani et al. did in "Attention Is All You Need" and what everyone follows

---

## 3. Two Main Approaches

### Approach A: Learned Position Embeddings

Just another `nn.Embedding` — but instead of mapping `word_id → vector`, it maps `position → vector`:

```python
self.tok_emb = nn.Embedding(vocab_size, embed_dim)
self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

# In forward:
positions = torch.arange(T)              # [0, 1, 2, ..., T-1]
x = tok_emb(idx) + pos_emb(positions)    # broadcast
```

Pros:
- Simple
- Used by GPT-2 and many modern models
- Lets the model learn whatever position representation works best

Cons:
- Fixed maximum sequence length (limited by the embedding size)
- Can't extrapolate to sequences longer than seen in training

### Approach B: Sinusoidal Position Encoding

The clever trick from the original "Attention Is All You Need":

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

For each position, use a vector of alternating sines and cosines at different frequencies.

Pros:
- No learned parameters
- Can handle ANY sequence length (just compute for that position)
- Has nice mathematical property: `PE(pos+k)` is a linear transform of `PE(pos)` — making "relative position" learnable

Cons:
- More complex to implement
- In practice, modern models often use learned embeddings or newer alternatives

---

## 4. The Sinusoidal Magic

Why sines and cosines at different frequencies? Here's the intuition:

```
For each position, we get a unique "barcode":
  pos=0:   [sin(0/1), cos(0/1), sin(0/100), cos(0/100), ...]
  pos=1:   [sin(1/1), cos(1/1), sin(1/100), cos(1/100), ...]
  pos=2:   [sin(2/1), cos(2/1), sin(2/100), cos(2/100), ...]
```

Different frequencies capture different scales:
- High frequency dims: distinguish nearby positions (1 vs 2)
- Low frequency dims: distinguish far-apart positions (10 vs 1000)

This is exactly how a **clock** works:
- Seconds hand: fast (high frequency) — distinguishes nearby times
- Hour hand: slow (low frequency) — distinguishes far-apart times

Together, they give every position a unique "time stamp."

---

## 5. The Beautiful Property

Sinusoidal encoding has a magical property: `PE(pos + k)` can be expressed as a **linear combination** of `PE(pos)`.

That means the model can learn to compute relative offsets:
- "Look 3 positions back"
- "Look at the next position"

without having to memorize each specific position pair.

(Don't worry about the math proof — just know this is why sinusoidal works so well.)

---

## 6. Newer Alternatives (Brief Tour)

After GPT-2, researchers found even better ways:

### RoPE (Rotary Position Embeddings)
Used by Llama, PaLM, GPT-NeoX. Instead of adding a position vector, **rotate** the Q and K vectors by an angle proportional to position. Mathematically elegant, extrapolates well.

### ALiBi (Attention Linear Biases)
Used by Bloom. Don't encode position at all — just add a linear bias to attention scores favoring nearby tokens.

### Relative Position
Encode the *distance* between tokens rather than their absolute position.

For this course, we'll use **learned positional embeddings** (like GPT-2) and **sinusoidal** (like the original paper). Modern alternatives are an optimization on top.

---

## 7. How Position Encoding Fits in a Transformer

```
input token IDs                                     (B, T)
       ↓
nn.Embedding(vocab, embed_dim)                      (B, T, embed_dim)
       │
       ├── + position encoding (B, T, embed_dim)    ← NEW today
       ▼
multi-head attention layer                          (B, T, embed_dim)
       ↓
... (more layers)
       ↓
nn.Linear(embed_dim, vocab_size)                    (B, T, vocab_size)
```

That's the only change. Add position info **once**, right after token embedding. Every attention layer benefits.

---

## 8. A Tiny Demo of "No Position = Disaster"

Train two identical small models — one with position encoding, one without. The position-aware model will dramatically out-perform on any task where order matters.

```
Without position:  "cat sat" and "sat cat" → same predictions
With position:     "cat sat" and "sat cat" → different predictions
```

For language modeling, the difference is enormous. For some tasks (image patches, set processing) you don't need position — but for text, you always do.

---

## 9. Implementation Patterns

### Learned (GPT-2 style)
```python
class GPTBlock(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
    
    def forward(self, idx):
        T = idx.size(1)
        pos = torch.arange(T, device=idx.device)
        return self.tok_emb(idx) + self.pos_emb(pos)
```

### Sinusoidal (no learnable params)
```python
def get_sinusoidal_pe(max_len, dim):
    pe = torch.zeros(max_len, dim)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
```

---

## Mental Model

```
Without position encoding:
   "cat sat" → attention sees: { cat, sat }
   "sat cat" → attention sees: { sat, cat }
   To attention's math, these are THE SAME SET.

With position encoding:
   "cat sat" → attention sees: { (cat, pos=0), (sat, pos=1) }
   "sat cat" → attention sees: { (sat, pos=0), (cat, pos=1) }
   Now they're clearly different.

Position encoding = stamping each token with its address.
```

---

## Summary

| Concept | What It Does |
|---------|-------------|
| **Position encoding** | Adds info about WHERE each token is |
| **Why needed** | Attention itself is permutation-invariant |
| **How** | Add a position-specific vector to each token embedding |
| **Learned position** | `nn.Embedding(max_seq_len, dim)` — simple, GPT-2 style |
| **Sinusoidal** | Fixed sin/cos at multiple frequencies — extrapolates, original Transformer |
| **RoPE / ALiBi** | Newer alternatives (Llama, Bloom) |
| **Where added** | Once, right after token embedding |

### Where we are

```
Day 13: RNN                              ✓
Day 14: Bigram LM                        ✓
Day 15: Self-attention                   ✓
Day 16: Multi-head attention             ✓
Day 17: POSITIONAL ENCODING              ← YOU ARE HERE
Day 18: Project — attention generator
Day 19: Transformer block
Day 20: Mini GPT — stack of blocks
Day 21+: Tokenization, training, sampling, evaluation
```

We have ALL the ingredients now. Day 18 is the project that combines them. Day 19 wraps it all into the canonical "transformer block."

**Tomorrow:** Day 18 — a complete attention-based text generator. We'll combine token embeddings, positional encoding, multi-head attention, and a final classifier head into a working language model that generates text.
