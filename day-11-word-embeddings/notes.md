# Day 11: Word Embeddings — Words as Dense Vectors

## Why This Matters

Yesterday's BoW classifier had a fundamental flaw: it treats every word as completely unrelated to every other word. To the model, "good" and "great" are as different as "good" and "banana." That's silly — they mean almost the same thing!

**Word embeddings** fix this. Instead of representing each word as a sparse one-hot vector (mostly zeros, one 1), we represent it as a **dense vector** — a list of numbers like `[0.2, -0.5, 0.8, 0.1, -0.3]`.

The magic: similar words get **similar vectors**. "good" and "great" end up close in this vector space.

This is THE foundational idea behind modern NLP. Every LLM (GPT, Claude, Llama) starts by converting tokens to embeddings.

---

## 1. The Core Idea

### Sparse one-hot (yesterday)

If your vocab has 5000 words, each word is a vector of length 5000:

```
"good"  = [0, 0, 0, ..., 1, 0, 0, ...]    (one 1 at position 3, rest are 0)
"great" = [0, 0, ..., 1, 0, 0, 0, ...]    (one 1 at position 4187, rest are 0)
```

Problems:
- **Wasteful** — 99.98% of every vector is zero
- **No meaning** — "good" and "great" are completely orthogonal
- **No similarity** — every pair of words is equally distant

### Dense embedding (today)

Each word is a vector of, say, length 100:

```
"good"  = [0.21, -0.45, 0.83, ...,  0.12]    (100 numbers, all meaningful)
"great" = [0.19, -0.42, 0.85, ...,  0.10]    (very close to "good"!)
"banana"= [-0.5, 0.83, -0.21, ..., -0.92]    (very different)
```

Properties:
- **Compact** — 100 numbers vs 5000
- **Meaningful** — each dimension captures some semantic property
- **Similar words → similar vectors**

---

## 2. The `nn.Embedding` Layer

PyTorch makes this easy. `nn.Embedding` is essentially a **learnable lookup table**:

```python
embedding = nn.Embedding(num_embeddings=5000, embedding_dim=100)
# This creates a (5000, 100) weight matrix.
# Row i = the embedding for word index i.

# Look up word #42:
word_id = torch.tensor([42])
vector = embedding(word_id)   # shape (1, 100)
```

Internally, `nn.Embedding` is just an `nn.Linear` applied to a one-hot vector — but optimized to skip the zero multiplications. It's **just a lookup**.

### Trainable!

The embedding matrix has `requires_grad=True`, so during training:
- Each weight in the matrix gets a gradient
- The optimizer updates them
- Over many examples, similar-context words end up with similar vectors

This is HOW the model learns meaning.

---

## 3. How Embeddings Get Trained

In a real LLM, embeddings are learned through the prediction task itself:

```
Task: predict next word given previous words
Forward:
   "the cat sat on the ___"
       ↓ embeddings
   [vec_the, vec_cat, vec_sat, ...]
       ↓ neural network
   probability over vocabulary
       ↓ loss
   loss tells us we should adjust embeddings to make the right answer more likely
       ↓ backprop
   embeddings update
```

After billions of examples:
- Words that appear in similar contexts → similar vectors
- "king" and "queen" both follow "the" and precede "of England"
- → They get similar embeddings

This idea is from the famous Word2Vec paper (2013). Modern models use the same principle, just embedded inside larger networks.

---

## 4. The Famous Word Analogies

The most famous demonstration of embeddings:

```
king - man + woman ≈ queen
```

Vector arithmetic on embeddings actually produces meaningful results!

```python
v_king  = embedding("king")
v_man   = embedding("man")
v_woman = embedding("woman")

target = v_king - v_man + v_woman
# Find the closest word in vocab → it's "queen"!
```

This works because well-trained embeddings capture **semantic dimensions**. There's a "gender" direction, a "royalty" direction, etc., and the math respects them.

Other examples:
- `paris - france + germany ≈ berlin` (capitals)
- `walking - walked + ran ≈ running` (verb tenses)
- `bigger - big + small ≈ smaller` (comparative form)

---

## 5. Cosine Similarity — Measuring "How Close"

To find similar words, we need to measure distance between vectors. The standard tool: **cosine similarity**.

```
cosine(a, b) = (a · b) / (|a| × |b|)

Range: -1 to 1
  +1.0 = same direction (very similar)
   0.0 = unrelated
  -1.0 = opposite
```

Why cosine and not Euclidean distance? Because we care about DIRECTION, not magnitude. Two embeddings can be similar even if one has a bigger magnitude.

```python
def cosine_sim(a, b):
    return (a @ b) / (a.norm() * b.norm())
```

To find the most similar word to a query:

```python
query = embedding("good")
similarities = []
for word, idx in vocab.items():
    other = embedding(idx)
    similarities.append((word, cosine_sim(query, other).item()))

# Sort by similarity, take top
top = sorted(similarities, key=lambda x: -x[1])[:10]
```

---

## 6. Embedding Dim — How Big?

| Dim | Use case |
|-----|----------|
| 8-16 | Toy / educational models |
| 50-100 | Small models, simple tasks |
| 300 | Word2Vec, GloVe (classics) |
| 768 | BERT base, GPT-2 small |
| 1024-1280 | BERT large, GPT-2 medium |
| 4096-12288 | GPT-3, GPT-4, Llama |

Bigger dim = more expressive, but more parameters and slower.

For small datasets (like our movie reviews), 50-100 is plenty. For training real LLMs, 768+.

---

## 7. From One-Hot to Embedding — The Pipeline

Yesterday:
```
"good"  →  [0, 0, ..., 1, 0, ...]   (length 5000, sparse)  →  Linear  →  output
```

Today:
```
"good"  →  3 (token index)  →  nn.Embedding  →  [0.2, -0.5, ...]  (length 100, dense)  →  Linear  →  output
                                ↑
                             this is the new layer
```

For a sentence, we look up each word and combine the vectors:

```
"the cat sat"
   ↓ tokenize + lookup indices
[0, 14, 23]
   ↓ nn.Embedding (dim=100)
[[0.2, ...], [0.5, ...], [0.1, ...]]   shape (3, 100)
   ↓ aggregate (e.g., mean)
[0.27, ...]                              shape (100,)
   ↓ classifier
prediction
```

This is the **average word embedding** approach — simple but effective.

---

## 8. Why Mean of Embeddings Works

Taking the mean of word embeddings might seem too simple, but it has nice properties:

- Captures the "overall direction" of the sentence
- Order-invariant (same as BoW — but with semantic info!)
- Works well for short documents and simple tasks
- Used in early sentence embedding methods

For better order-aware models, we'll use sequence models (RNN, transformer) — but mean embedding is a great baseline.

---

## 9. Pre-trained Embeddings (a Preview)

Training embeddings from scratch needs a LOT of data. Lucky for us, the community has trained them on huge corpora and released them:

- **Word2Vec** (Google, 2013) — 300-dim, trained on 100B words
- **GloVe** (Stanford, 2014) — 50/100/200/300-dim, trained on Wikipedia
- **fastText** (Facebook, 2016) — handles subwords

You can download these and use them directly. We'll do this in a later day.

For modern LLMs, embeddings are learned ALONGSIDE the rest of the network. We won't use pre-trained word embeddings for our LLM — we'll learn them ourselves.

---

## Mental Model

```
One-hot encoding   = giving each word a unique badge number, no info beyond identity
                       "good" = #3, "great" = #4187 → no relationship visible

Embedding         = giving each word a profile of features
                       "good" = [warmth=0.8, size=0.1, age=0.3, ...]
                       "great" = [warmth=0.9, size=0.2, age=0.3, ...]
                       Now you can SEE they're similar

Each dimension = some learned semantic property
                  (no human can label them, but the model uses them)
```

---

## Summary

| Concept | What It Does |
|---------|-------------|
| **One-hot** | Sparse, identity-only encoding |
| **Embedding** | Dense, learned vector that captures meaning |
| **`nn.Embedding`** | PyTorch's lookup table (vocab × dim matrix) |
| **Cosine similarity** | Measures "how close" two vectors are |
| **Word analogies** | king - man + woman ≈ queen |
| **Mean embedding** | Average word vectors → sentence vector |
| **Embedding dim** | Bigger = more expressive (50 for toys, 4096 for GPT-4) |

### What changes from Day 10:

```
Day 10:  text → BoW (sparse, 5000-dim) → Linear → prediction
Day 11:  text → tokens → Embedding lookup → mean → Linear → prediction
                           ↑
                  THE difference — dense, learned, captures meaning
```

### The journey:

```
Day 1-9:   Build any neural network                  ✓
Day 10:    BoW text classifier                        ✓
Day 11:    Word embeddings — words have meaning       ← YOU ARE HERE
Day 12:    Project — full sentiment classifier
Day 13+:   Sequence models that respect word order
Day 19+:   Transformers (where embeddings really shine)
```

**Tomorrow:** We'll build a complete sentiment classifier project using everything we've learned, including embeddings.
