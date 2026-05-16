# Day 14: Bigram Language Model — GPT's Great-Grandparent

## Why This Matters

Today we build the **simplest possible language model**. It only looks at ONE previous token to predict the next. It's almost embarrassingly basic — but it introduces the exact framework that GPT-4 uses.

The training task itself, "predict the next token," is the most important idea in modern AI:
- GPT, Claude, Llama all train on this single task
- The task is unsupervised — just predict what comes next from raw text
- Scale it up with attention + transformers + trillions of tokens = ChatGPT

A bigram model is GPT-4's ancestor. Same task, much simpler architecture.

---

## 1. What is a Language Model?

A **language model** assigns a probability to a sequence of tokens. In practice, we usually train it to do one thing:

> "Given the tokens so far, predict the probability distribution over the next token."

```
Input:  "The cat sat on the ___"
Output: P(mat) = 0.4
        P(floor) = 0.2
        P(roof) = 0.1
        ... (probabilities over the entire vocabulary)
```

If the model assigns high probability to the actual next word, it's good. If not, it's bad.

That's it. Train this task on enough data, and astonishing things emerge.

---

## 2. The "Bigram" Restriction

A **bigram model** is the most limited language model possible:

> "To predict the next token, look ONLY at the immediately previous token. Ignore everything else."

```
Bigram input:  ___ ___ ___ ___ "the" → predict next
                              ↑
                       only this one matters
```

It's a Markov chain of order 1. The model has no memory beyond one step.

This is obviously a terrible language model. "the" can be followed by literally anything. But it's the simplest case, and it sets up all the patterns we'll keep using:
- Tokenize text
- Build (input, target) pairs
- Train a model to predict the target
- Sample text by repeatedly predicting one token at a time

---

## 3. The "Lookup Table" Formulation

For a bigram model, the entire model is just a `(vocab_size, vocab_size)` table:

```
Row "the" → probabilities over all words that can follow "the"
Row "cat" → probabilities over all words that can follow "cat"
Row "sat" → probabilities over all words that can follow "sat"
...
```

To predict the next token after "the": grab row "the" → that's your distribution.

### Two ways to compute this table

**A) Counting (no learning needed)**

Walk through the text. For every adjacent pair `(a, b)`, increment `count[a, b]`. Then normalize each row to get probabilities. Done.

**B) Neural network (1-layer, 1 parameter matrix)**

Use `nn.Embedding(vocab_size, vocab_size)`. Each row IS a logit vector for "what comes next after this token." Train with Cross-Entropy.

Both produce essentially the same result. **The neural version is the prototype for all bigger language models**, so we'll focus on it.

---

## 4. The Neural Bigram Model

The architecture is shockingly simple:

```
input token ID  →  nn.Embedding(vocab_size, vocab_size)  →  logits over vocab
```

That's the whole model.

### Wait, that's just one matrix?

Yes. The embedding matrix has shape `(vocab_size, vocab_size)`:
- Look up row `i` to get the prediction logits for "what comes after token i"
- Softmax those logits to get probabilities
- Train with Cross-Entropy: each (input, target) pair pushes the right cell up

It's literally a learnable lookup table.

### Comparison to Day 11

| Day 11 | Day 14 |
|--------|--------|
| `nn.Embedding(vocab, embed_dim)` → small dense vector | `nn.Embedding(vocab, vocab)` → distribution over next token |
| Used embeddings as FEATURES | Embeddings ARE the predictions |
| Feed into another layer | Direct output, no extra layers |

---

## 5. The Training Objective

We use **Cross-Entropy loss** (the standard for predicting one of many classes):

```
loss = -log(P[correct_next_token])

Where P comes from softmax(model_logits).
```

Intuition: if the model predicts the right next-token with 99% probability, `-log(0.99) ≈ 0.01` (tiny loss). If it predicts only 1%, `-log(0.01) ≈ 4.6` (big loss).

### Initial loss = "what you'd get by guessing"

If your vocab has V tokens, completely random guessing gives `log(V)` loss:

```
V = 30 chars  →  log(30) ≈ 3.4
V = 50,000 tokens (GPT) → log(50000) ≈ 10.8
```

A trained model should beat this baseline. If your loss is stuck at log(V), the model isn't learning.

---

## 6. Sampling — Generating Text

Once trained, generating text is a loop:

```
1. Start with a seed token (e.g., '.' or 'a')
2. Look it up → get logits → softmax → get probabilities
3. Sample the next token from this distribution
4. Make that the new "current token"
5. Repeat for N steps
```

The exact same loop will work for GPT-4 — just with a vastly bigger and smarter model.

---

## 7. Why Bigrams Fail (and What Comes Next)

The bigram model has one fatal flaw: **it only sees one previous token**.

```
"the boy sat on the chair, but the dog jumped over the ___"

Bigram sees only:           "the ___"  → could be anything
GPT sees the whole context: "the boy ... over the ___" → "fence"
```

To get better, you need to look at MORE context.

This is the path forward:
- **Day 15+:** Attention — let the model look at multiple previous tokens, weighted
- **Day 19+:** Transformers — stack attention layers to process huge contexts (GPT-4: 128k tokens)

---

## 8. Counts vs Neural: a Beautiful Equivalence

Two views of the same thing:

**Counting:**
```
P(b | a) = count(a, b) / count(a)
```

**Neural (after training):**
```
P(b | a) = softmax(embedding[a])[b]
```

If trained perfectly, the neural version converges to the counting answer. We use the neural version because:
1. It generalizes to bigger contexts (trigrams, 4-grams, ... full transformers)
2. It's differentiable end-to-end with the rest of a model
3. It scales

The counting version doesn't extend to context lengths of 1000+ — you'd need astronomical memory.

---

## 9. What This Sets Up

The bigram model establishes EVERY pattern that GPT uses:

| Day 14 (bigram) | GPT |
|----------------|-----|
| Predict next token | Predict next token |
| Cross-Entropy loss | Cross-Entropy loss |
| Sample one token at a time | Sample one token at a time |
| Temperature for sampling | Temperature for sampling |
| Look up token → predict | Look up token + position → many attention layers → predict |
| Character vocab | BPE subword vocab |
| 1 layer | 96+ layers |

We've changed the **architecture**. The **task** is identical.

---

## Mental Model

```
Bigram model = a tiny brain that only remembers the LAST word
              "After 'the', I usually expect... a noun?"
              "After 'sat', I usually expect... 'on' or 'down'?"

GPT-4 = a giant brain that remembers thousands of previous words
              "After this entire paragraph, the next word is probably..."

Both are doing the same task. GPT is just way more capable at it.
```

---

## Summary

| Concept | What It Does |
|---------|-------------|
| **Language model** | Predicts probability of the next token |
| **Bigram** | Uses only the immediately previous token |
| **Next-token prediction** | The unifying training task for all LLMs |
| **`nn.Embedding(V, V)`** | A learnable lookup of next-token logits |
| **Cross-Entropy loss** | Penalizes wrong predictions |
| **Sampling** | Generate text by repeatedly predicting + picking a next token |
| **Temperature** | Knob for diversity in sampling |
| **Perplexity** | exp(loss) — "how surprised is the model?" — common LM metric |

### The path through Phase 3:

```
Day 13: RNN — sequence + memory                  ✓
Day 14: Bigram LM — the next-token task           ← YOU ARE HERE
Day 15: Self-attention — look at multiple tokens, weighted
Day 16: Multi-head attention
Day 17: Positional encoding
Day 18: Attention-based generator
Day 19+: Transformers (it all comes together)
```

**Tomorrow:** Day 15 — Self-attention. The single most important concept in modern AI. The trick that lets one model look at many tokens simultaneously, weighted by relevance.
