# Day 15: Self-Attention — The Idea Behind Every Transformer

## Why This Matters

If you take ONE thing from this entire 45-day journey, it should be self-attention. This is the trick that:

- Made GPT, Claude, BERT, Llama, ChatGPT possible
- Replaced RNNs as the dominant sequence model
- Won the 2017 paper "Attention Is All You Need" its place in history

Once self-attention clicks, transformers stop being mysterious. They become **stacks of self-attention layers**, plus some standard plumbing.

---

## 1. The Problem We're Solving

Yesterday's bigram model:

```
"the cat sat on the m___"   ← bigram sees only "m"
```

The bigram model can't tell whether the previous word was "ate," "saw," or "swam." It's blind to everything except the immediately previous token.

What we WANT:

```
"the cat sat on the m___"  → model uses "cat" + "sat" + "on" + "the" to predict
```

Each position should be able to "look at" all earlier positions and use them.

That's exactly what self-attention does.

---

## 2. The Big Intuition

Imagine each word in a sentence wants to **gather information** from other words:

```
"The cat that I saw yesterday sat on the mat"
                                ↑
                              When processing "sat":
                              - look at "cat"        (HIGH attention — that's the subject)
                              - look at "I saw"      (LOW attention — irrelevant clause)
                              - look at "yesterday"  (LOW attention — just a time)
```

Each word **chooses how much to attend** to each other word, then combines them weighted by attention.

That's the algorithm: **weighted average of all words, with weights learned from the data.**

---

## 3. The Three Roles: Query, Key, Value

For each token, we compute THREE things from its embedding:

```
Query (Q):  "What am I looking for?"
Key (K):    "What information do I have?"
Value (V):  "What information do I actually carry?"
```

Each is a vector produced by multiplying the token's embedding by a learnable weight matrix.

### How attention scores work:

For position `i` to attend to position `j`:
- Take `Q_i` (what i is looking for)
- Take `K_j` (what j offers)
- Compute their dot product → `score(i, j)`
- High dot product = "i is interested in j"

For ALL positions at once, this is a single matrix multiply: `scores = Q @ K.T`

### The full formula

```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
                            │             │           │
                            │             │           └─ weighted sum of values
                            │             └─ normalize to keep scale small
                            └─ scores: each row sums to 1 after softmax
```

Don't memorize this. Understand the steps:

1. **Q @ K.T** → matrix of attention scores (how much each token attends to each other)
2. **divide by sqrt(d_k)** → keeps numbers from getting huge (stability trick)
3. **softmax** → convert scores to weights summing to 1
4. **@ V** → weighted average of value vectors

---

## 4. Self-Attention vs Cross-Attention

- **Self-attention:** Q, K, V all come from the SAME sequence. Tokens in a sentence attend to each other.
- **Cross-attention:** Q comes from one sequence, K and V from another. (Used in encoder-decoder transformers for translation.)

For LLMs like GPT, we use **self-attention only**.

---

## 5. Causal Masking — "Don't Cheat by Looking Ahead"

When predicting the next token, you can ONLY look at previous tokens. The model can't see the future during training.

```
For position 4 in "the cat sat on _":
  Attend to:  the, cat, sat, on        ← OK
  Attend to:  future tokens             ← NO! That's cheating.
```

We enforce this by adding `-infinity` to attention scores for "future" positions BEFORE the softmax. After softmax, those positions get probability 0.

The "mask" looks like this (1 = visible, 0 = blocked):

```
Position 0 can see:  [1, 0, 0, 0, 0]   (only itself)
Position 1 can see:  [1, 1, 0, 0, 0]   (itself + position 0)
Position 2 can see:  [1, 1, 1, 0, 0]
Position 3 can see:  [1, 1, 1, 1, 0]
Position 4 can see:  [1, 1, 1, 1, 1]
```

This is a **lower-triangular mask**. Standard in GPT-style models.

---

## 6. Why "Self-Attention" is a Big Deal

Three properties make it superior to RNNs:

### 1. Parallelism

RNNs process one token at a time (slow). Self-attention processes ALL tokens at once in matrix multiplies → GPU-friendly, **much faster training**.

### 2. Long-range dependencies

RNNs forget through long sequences (vanishing gradients). Self-attention can attend to any position equally — **distance 1 or distance 1000 is the same for the math**.

### 3. Interpretable

Attention weights show what the model is "looking at." This is harder with RNN hidden states.

---

## 7. The Shape Story

For input of shape `(batch, seq_len, embed_dim)`:

```
Embeddings  X:           (batch, seq_len, embed_dim)
              ↓ multiply by W_Q, W_K, W_V (each shape (embed_dim, head_dim))
Q, K, V:                 (batch, seq_len, head_dim)
              ↓ Q @ K.T
Scores:                  (batch, seq_len, seq_len)    ← attention matrix
              ↓ scale + mask + softmax
Weights:                 (batch, seq_len, seq_len)    ← rows sum to 1
              ↓ @ V
Output:                  (batch, seq_len, head_dim)   ← weighted-sum values per token
```

The middle is the magical `(seq_len × seq_len)` matrix. Each row = "how much attention does token i pay to each other token?"

---

## 8. The Tiny Worked Example

Take a sequence of 3 tokens, embed_dim=2 (so we can SEE everything):

```
Embeddings:
  token 0: [1, 0]
  token 1: [0, 1]
  token 2: [1, 1]

Assume Q = K = V = embeddings (skip the weight matrices for now).

Scores = Q @ K.T:
       to t0    to t1    to t2
  t0:  [ 1,      0,       1 ]      (t0's query against all 3 keys)
  t1:  [ 0,      1,       1 ]
  t2:  [ 1,      1,       2 ]

After softmax (row-wise):
  t0:  [0.42, 0.16, 0.42]    (t0 attends 42% to itself, 16% to t1, 42% to t2)
  t1:  [0.21, 0.42, 0.42]
  t2:  [0.21, 0.21, 0.58]    (t2 mostly attends to itself)

Output = weights @ V:
  t0:  0.42*[1,0] + 0.16*[0,1] + 0.42*[1,1] = [0.84, 0.58]
  t1:  0.21*[1,0] + 0.42*[0,1] + 0.42*[1,1] = [0.63, 0.84]
  t2:  0.21*[1,0] + 0.21*[0,1] + 0.58*[1,1] = [0.79, 0.79]
```

The outputs are weighted averages of values — but the weights came from how similar Q and K were.

In a trained model, the Q/K/V projections aren't identity. They're LEARNED to extract the most useful info for the task at hand.

---

## 9. Single-Head Self-Attention — The Building Block

That's everything for ONE attention "head":

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, head_dim, bias=False)
    
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        Q = self.W_q(x)                              # (B, T, head_dim)
        K = self.W_k(x)
        V = self.W_v(x)
        
        scores = Q @ K.transpose(-2, -1)             # (B, T, T)
        scores = scores / (Q.size(-1) ** 0.5)        # scale
        
        # Causal mask: -inf for future positions
        T = x.size(1)
        mask = torch.tril(torch.ones(T, T))           # lower-triangular
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        out = weights @ V                             # (B, T, head_dim)
        return out
```

That's a complete attention layer! Tomorrow we'll add **multiple heads** (run several of these in parallel and concatenate).

---

## 10. What This Layer Replaces

| Concept | RNN approach | Self-attention approach |
|---------|-------------|-------------------------|
| Remember past | Hidden state, passed step by step | Direct lookup of all past tokens |
| Long range | Vanishing gradients | Same cost regardless of distance |
| Parallelism | Sequential (one step at a time) | Fully parallel (one matmul) |
| Interpretability | Opaque hidden state | Attention weights visible |

---

## Mental Model

```
For each token, self-attention says:
  "Hey, all of you other tokens... let me look at you.
   How RELATED am I to each of you? (that's Q @ K)
   Now I'll combine your info, weighted by how related you are."

That's it. Every token does this. Everyone in the sentence
looks at everyone else, weighted by learned relevance.
```

---

## Summary

| Concept | What It Does |
|---------|-------------|
| **Query (Q)** | "What am I looking for?" — one vector per token |
| **Key (K)** | "What info do I provide?" — one per token |
| **Value (V)** | "What info do I actually carry?" — one per token |
| **Q @ K.T** | Score matrix — how much token i pays attention to token j |
| **Scale by √d** | Stability — prevents huge scores when d is large |
| **Causal mask** | Zero out "future" positions before softmax |
| **Softmax** | Turn scores into weights (each row sums to 1) |
| **weights @ V** | Weighted average of values — the output |

### The journey:

```
Day 13: RNN — sequence model #1 (with limits)         ✓
Day 14: Bigram LM — the next-token task               ✓
Day 15: SELF-ATTENTION                                ← YOU ARE HERE
Day 16: Multi-head — many attention heads in parallel
Day 17: Positional encoding — telling the model about ORDER
Day 18: Project — character generator with attention
Day 19+: Transformer (stack everything together)
```

**Tomorrow:** Multi-head attention. Multiple attention layers running in parallel, each learning different patterns. The actual building block of every transformer.
