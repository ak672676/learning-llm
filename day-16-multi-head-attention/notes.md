# Day 16: Multi-Head Attention — Many Eyes on the Same Sequence

## Why This Matters

Yesterday's self-attention had ONE set of Q/K/V weights, producing ONE attention pattern. That's like processing a sentence with one fixed "perspective."

But a sentence has many KINDS of relationships:
- Who's the subject?
- Who's the object?
- Which adjective modifies which noun?
- What's the time relationship?
- What's the spatial relationship?

A single attention head has to learn all of these mixed together — and it's not great at any one of them.

**Multi-head attention** runs SEVERAL attention computations in parallel, each focusing on something different. Then it combines their outputs. This is the actual building block used by every transformer.

---

## 1. The Core Idea

Instead of one big attention, we use `h` smaller attention heads:

```
single-head:  embed_dim=128, head_dim=128
              (one big attention computation)

multi-head:   embed_dim=128, num_heads=4, head_dim=32
              (4 smaller attention computations in parallel)
              outputs of 4 heads concatenated → embed_dim=128
```

Each head has its OWN W_Q, W_K, W_V weights. So each head can specialize.

After all heads compute their outputs, we **concatenate them** and pass through a final linear layer (`W_O`) to mix them.

---

## 2. The Architecture

```
Input X: (batch, seq_len, embed_dim)
         │
         ├── Head 1: small Q/K/V → attention → out_1  (batch, seq_len, head_dim)
         ├── Head 2: small Q/K/V → attention → out_2  (batch, seq_len, head_dim)
         ├── Head 3: ...
         └── Head h: ...
              ↓
         Concatenate all head outputs along the last dim
              ↓
         (batch, seq_len, num_heads × head_dim)
              ↓
         Output projection W_O (mix the heads)
              ↓
Output: (batch, seq_len, embed_dim)
```

Notice the output dim equals the input dim. This is intentional — it lets us **stack** multi-head attention layers (transformer = many such layers stacked).

---

## 3. Why Multiple Heads Help

In practice, different heads learn different patterns:

```
Head 1: "previous word" attention      (closely related to bigrams)
Head 2: "subject-verb" attention       (longer range)
Head 3: "matching brackets" attention  (very specific)
Head 4: "same noun phrase" attention   (semantic)
```

This is observed empirically in trained transformers. We can't tell each head what to learn — they specialize on their own based on what's useful.

**It's like a committee of specialists** instead of one generalist.

---

## 4. The Magic of "Single Big Matrix" Implementation

In theory, you could implement multi-head attention with a Python loop:

```python
# Naive — slow!
outputs = []
for head in range(num_heads):
    Q = self.heads[head].W_q(x)
    K = self.heads[head].W_k(x)
    V = self.heads[head].W_v(x)
    out = attention(Q, K, V)
    outputs.append(out)
combined = torch.cat(outputs, dim=-1)
```

In practice, we do all heads at once with **ONE big linear layer and reshape**:

```python
# Fast — actual implementation
Q = self.W_q(x)                          # (B, T, embed_dim)
Q = Q.view(B, T, num_heads, head_dim)    # split into heads
Q = Q.transpose(1, 2)                    # (B, num_heads, T, head_dim)

# Same for K, V, then attention happens in parallel across heads
```

This is much faster — one matrix multiply instead of `num_heads` separate ones.

The trick: `W_q` has shape `(embed_dim, embed_dim)` but we INTERPRET it as `num_heads` separate `(embed_dim, head_dim)` matrices stacked.

---

## 5. The Math: One Head vs Many Heads

### One head with embed_dim=128:

```
Parameters: 3 × (128 × 128) = ~49,000 (for Q, K, V)
            + 128 × 128 = ~16,000 (output projection)
            ≈ 65,000 params
```

### Four heads with head_dim=32 each:

```
Parameters: 3 × (128 × 32 × 4 heads) = ~49,000
            + 128 × 128 (output projection) = ~16,000
            ≈ 65,000 params
```

**Same parameter count!** The total is identical. We just SPLIT the projections so each head sees a smaller subspace.

The clever bit: same params, but the model can learn `num_heads` different specialized attention patterns instead of forcing one to do everything.

---

## 6. Concatenation Then Projection — Why Both?

After heads compute their outputs:

```
out_1: (B, T, head_dim)
out_2: (B, T, head_dim)
out_3: (B, T, head_dim)
out_4: (B, T, head_dim)
        ↓ concatenate along last dim
combined: (B, T, num_heads * head_dim) = (B, T, embed_dim)
        ↓ W_O
final:    (B, T, embed_dim)
```

Why the extra `W_O`? Two reasons:

1. **Mixes the heads.** Concatenation just stacks them side by side. `W_O` lets each output dimension combine info from all heads.

2. **Keeps dimensions consistent.** Output shape = input shape, so we can stack layers.

---

## 7. Multi-Head Attention in PyTorch

PyTorch has a built-in: `nn.MultiheadAttention`. Same interface, but their argument names are slightly different. For learning, we'll build it ourselves.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # All Q, K, V projections combined into one
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Causal mask
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))
    
    def forward(self, x):
        B, T, C = x.shape   # batch, seq, embed_dim
        
        # Project and split into heads
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape now: (B, num_heads, T, head_dim)
        
        # Attention for all heads in parallel
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        # Shape: (B, num_heads, T, head_dim)
        
        # Recombine: (B, num_heads, T, head_dim) → (B, T, num_heads × head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out), weights
```

The whole thing is what GPT uses internally. Add positional encoding (Day 17) and you have a transformer block.

---

## 8. Heads Are Cheap, Use Many

The cost of multi-head attention is **the same** as single-head with the same total dimension. So why not use many heads?

Empirical guidelines:
- BERT-base: 12 heads
- BERT-large: 16 heads
- GPT-2 medium: 16 heads
- GPT-3: 96 heads
- Llama-2 7B: 32 heads

`head_dim = embed_dim / num_heads`. With embed_dim=768 and 12 heads → head_dim=64.

More heads = more specialized patterns, but each head's "subspace" gets smaller. There's a sweet spot.

---

## 9. What Heads Actually Learn

When researchers visualize attention heads in trained models, they see:

- **Some heads attend to the previous token** (like a bigram)
- **Some attend to the next-occurrence of the same word**
- **Some attend to verbs from nouns** (subject-verb agreement)
- **Some attend "globally"** to a few specific positions (sentence-level info)
- **Some heads are nearly useless** (the model could prune them)

A famous paper, "What Does BERT Look At?", catalogs many such patterns.

---

## Mental Model

```
Single-head: ONE pair of glasses → you see one focus area clearly

Multi-head:  MANY pairs of glasses (each tuned differently)
             → red glasses see one thing, blue see another, etc.
             → combine all views to understand the full picture

Each head: "I'll handle ___ relationships."
Concat:    "Here are all the views together."
W_O:       "Let me blend them into one coherent output."
```

---

## Summary

| Concept | What It Does |
|---------|-------------|
| **Multi-head attention** | Run h attention heads in parallel, concatenate |
| **head_dim** | embed_dim / num_heads (must divide evenly) |
| **W_O (output projection)** | Linear layer that mixes heads' outputs |
| **Reshape trick** | Use one big matrix; reshape to (batch, heads, seq, head_dim) |
| **Specialized heads** | Different heads learn different attention patterns |
| **Stackable** | Output dim = input dim → can stack many layers |

### The pattern that makes transformers

```
input → Multi-Head Attention → output (same shape)
input → MultiHeadAttention → MultiHeadAttention → ... → output
                                    (stackable!)
```

One transformer = a stack of multi-head attentions + a few extras (layer norm, MLP, residual).

### Where we are

```
Day 13: RNN — first sequence model              ✓
Day 14: Bigram LM — next-token task             ✓
Day 15: Single-head attention                    ✓
Day 16: MULTI-HEAD attention                     ← YOU ARE HERE
Day 17: Positional encoding — telling model about ORDER
Day 18: Project — attention-based generator
Day 19: Transformer block (puts it all together)
Day 20+: Full GPT, training, sampling
```

**Tomorrow:** Positional encoding. Right now, the attention model has NO sense of order — it sees the same set of tokens regardless of arrangement. We'll fix that with sinusoidal position embeddings.
