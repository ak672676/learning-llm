# Day 04: Data Handling — Datasets, Batching, Tokenization Basics

## Why This Matters

Days 1-3 we worked with tiny data (5 students, 50 numbers). Real LLMs train on **trillions of words**. Before we can train anything on text, we need to answer three questions:

1. **How do you turn words into numbers?** (Tokenization)
2. **How do you feed data to a model?** (Datasets & DataLoaders)
3. **Why not feed all data at once?** (Batching)

---

## 1. The Problem: Models Only Understand Numbers

Our model from Day 3 took numbers in and produced numbers out:

```
input: [hours_studied=3, hours_slept=8] → output: score=75
```

But LLMs work with text:

```
input: "The cat sat on the" → output: "mat"
```

How do you turn "The cat sat on the" into numbers that a model can process?

---

## 2. Tokenization — Turning Text Into Numbers

### Step 1: Split text into pieces (tokens)

```
"The cat sat on the mat"
    ↓ split
["The", "cat", "sat", "on", "the", "mat"]
```

These pieces are called **tokens**. They could be:
- **Words:** "The", "cat", "sat" (simple, but vocabulary is huge)
- **Characters:** "T", "h", "e", " ", "c", "a", "t" (small vocabulary, but sequences are very long)
- **Subwords:** "The", "cat", "s", "at", "on", "the", "mat" (the sweet spot — used by all modern LLMs)

### Step 2: Map each token to a number (token ID)

Build a vocabulary — a lookup table:

```
Vocabulary:
  "the"  → 0
  "cat"  → 1
  "sat"  → 2
  "on"   → 3
  "mat"  → 4
  "dog"  → 5
  ...
```

Then convert:

```
["The", "cat", "sat", "on", "the", "mat"]
    ↓ lookup
[0, 1, 2, 3, 0, 4]
```

Now the model sees `[0, 1, 2, 3, 0, 4]` — just numbers! It can do matrix math on these.

### Step 3: The model works with these numbers

```
[0, 1, 2, 3, 0, ?]  → model predicts → 4 (which maps to "mat")
```

This is what "next token prediction" means — the model sees a sequence of token IDs and predicts the next one.

---

## 3. Character-Level vs Word-Level vs Subword Tokenization

### Character-level (what we'll build today)

```
"hello" → ["h", "e", "l", "l", "o"] → [7, 4, 11, 11, 14]
```

- Vocabulary: just ~100 characters (a-z, A-Z, 0-9, punctuation)
- Pros: tiny vocabulary, handles any word
- Cons: sequences are very long (every character is a token)
- Used for: learning/toy models (like what we'll build)

### Word-level

```
"hello world" → ["hello", "world"] → [3421, 892]
```

- Vocabulary: 50,000-100,000+ words
- Pros: compact sequences
- Cons: can't handle new/rare words ("bruh" → unknown token)
- Used for: older models

### Subword (BPE) — what GPT/Llama actually use

```
"unhappiness" → ["un", "happiness"] → [348, 12045]
"ChatGPT" → ["Chat", "G", "PT"] → [7126, 38, 2898]
```

- Vocabulary: ~32,000-100,000 subwords
- Pros: handles any word, reasonable sequence length
- Cons: more complex to implement
- Used for: all modern LLMs (GPT, Llama, Claude)

We'll build subword tokenization on Day 21. Today we start simple with character-level.

---

## 4. Why Batching? Why Not Feed All Data at Once?

### Problem 1: Memory

If your text is 1 million characters and each token embedding is 768 numbers:
```
1,000,000 × 768 × 4 bytes = 3 GB just for the input!
```
GPU memory is limited (8-80 GB). You can't fit everything at once.

### Problem 2: Training dynamics

Gradient descent works BETTER with smaller, noisier updates (stochastic gradient descent) than with one giant update per epoch. The noise helps escape local minima.

### Solution: Batches

Split the data into small chunks and process one chunk at a time:

```
Full dataset: "The cat sat on the mat the dog ran in the park..."
                                    ↓ split into batches
Batch 1: "The cat sat on"    → train → update weights
Batch 2: "the mat the dog"   → train → update weights
Batch 3: "ran in the park"   → train → update weights
```

Each batch is small enough to fit in memory, and you still go through all the data.

### Key terms

| Term | Meaning |
|------|---------|
| **Batch size** | How many samples in one batch (e.g., 32, 64, 128) |
| **Epoch** | One full pass through ALL the data |
| **Iteration/Step** | Processing one batch |
| **Steps per epoch** | total_samples / batch_size |

Example: 1000 samples, batch_size=100 → 10 steps per epoch.

---

## 5. PyTorch DataLoader — Automatic Batching

PyTorch provides `Dataset` and `DataLoader` to handle all of this:

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __len__(self):
        return len(self.data)         # how many samples
    
    def __getitem__(self, idx):
        return self.data[idx]         # return one sample

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # batch is automatically 32 samples
    prediction = model(batch)
    loss = ...
    loss.backward()
    optimizer.step()
```

The DataLoader handles:
- Splitting data into batches
- Shuffling (random order each epoch — helps training)
- Loading data in parallel (faster)

---

## 6. Sequence Data for Language Models

For a language model, one "sample" is:

```
Input:  [T, h, e, , c, a, t]     (the sequence so far)
Target: [h, e, , c, a, t, _]     (the next character at each position)
```

Notice: the target is just the input **shifted by one position**. The model learns to predict the next character at every position simultaneously.

This is called **teacher forcing** — at every position, we tell the model the correct answer and ask "what would you have predicted?"

### Context length (block size)

We can't feed the entire text at once (too long). So we use fixed-length windows:

```
Text: "The cat sat on the mat"
Block size: 8

Sample 1: input="The cat " → target="he cat s"
Sample 2: "sat on t"       → "at on th"
Sample 3: "the mat"        → "he mat "
```

The block size is the model's "memory" — how far back it can look. GPT-4 has a context length of ~128,000 tokens.

---

## Mental Model

```
Raw text:  "The cat sat on the mat"
              ↓ tokenize (split into characters)
Tokens:    ['T','h','e',' ','c','a','t',' ','s','a','t',...]
              ↓ map to numbers
Token IDs: [20, 8, 5, 0, 3, 1, 20, 0, 19, 1, 20,...]
              ↓ split into fixed-length chunks
Batches:   [[20,8,5,0,3,1,20,0], [19,1,20,0,...], ...]
              ↓ feed to model, one batch at a time
Training:  optimizer.zero_grad() → loss.backward() → optimizer.step()
```

**Tomorrow:** We'll visualize training progress, plot loss curves, and learn common debugging patterns (Day 05).
