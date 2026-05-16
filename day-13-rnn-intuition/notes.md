# Day 13: RNN — Models That Read Sequences

## Why This Matters

Every model you've built so far had a critical flaw: **it doesn't care about word order**.

```
"not amazing"     → averaged embeddings → looks like "amazing not"
"the cat sat"     → same BoW vector as → "sat the cat"
```

For real text understanding — translation, sentiment, generation — order matters everywhere. Today we learn the **first** sequence model: the **Recurrent Neural Network (RNN)**.

This isn't the final answer (transformers eventually replace RNNs), but RNNs introduce concepts that carry forward:
- **Hidden state** — memory that flows through the sequence
- **Step-by-step processing** — one token at a time
- **Variable-length inputs** — sentences can be any length

---

## 1. The Core Idea

An RNN processes a sequence **one token at a time**, carrying a "memory" forward:

```
Input:   "the"    "cat"    "sat"
            ↓        ↓        ↓
Hidden:   h0  →   h1   →   h2   →   h3  (the "memory")
            ↑        ↑        ↑
         "fresh"  "knows  "knows
          start"  about    about
                  the"     the cat"
```

At each step:
1. Take the current word + the previous hidden state
2. Combine them → new hidden state
3. Pass the new hidden state forward

The **final hidden state** contains a summary of the whole sequence.

---

## 2. The RNN Equation

For each step `t`:

```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
```

Where:
- `x_t` = the input word at time t (an embedding vector)
- `h_{t-1}` = the previous hidden state (the memory so far)
- `W_xh` = weights that transform the input
- `W_hh` = weights that transform the previous memory
- `b` = bias
- `tanh` = activation (keeps values bounded -1 to 1)

The output at each step is just `h_t`. To predict something at the end, you take the FINAL hidden state and feed it to a classifier.

### What gets learned?

PyTorch updates `W_xh`, `W_hh`, and `b` through backprop. Over training:
- `W_xh` learns to extract useful info from each word
- `W_hh` learns how to "remember" or "forget" past info
- The hidden state learns to summarize the sequence

---

## 3. Why "Recurrent"?

The same weights are used at EVERY time step. It's the same operation, looped over the sequence:

```python
h = torch.zeros(hidden_dim)         # start with empty memory
for word in sentence:
    h = rnn_step(word, h)            # same function applied repeatedly
# h now contains the summary of the whole sentence
```

This is the recurrence — the function feeds back into itself.

---

## 4. The Vanishing Gradient Problem

RNNs in their basic form have a SERIOUS problem with long sequences.

### What happens

When we backpropagate through 100 time steps:
- The gradient gets multiplied by `W_hh` 100 times
- If values in W_hh are < 1 → gradients shrink to ~0 (**vanishing**)
- If values are > 1 → gradients explode to infinity (**exploding**)

Result: **the model can't learn long-range dependencies**.

```
"I grew up in France ... [50 words later] ... I speak ___"
                                                       ↑
                          To predict "French" the model needs to remember 
                          something from 50 steps ago. RNNs lose this signal.
```

### How researchers fixed it

- **LSTM** (1997) — adds "gates" that control what to remember/forget
- **GRU** (2014) — simpler version of LSTM
- **Transformer** (2017) — abandons recurrence entirely, uses attention instead

For most modern NLP, transformers won. But understanding RNNs makes you understand WHY transformers are better.

---

## 5. PyTorch's Built-in RNN

You don't have to implement the loop yourself. PyTorch has `nn.RNN`, `nn.LSTM`, `nn.GRU`:

```python
rnn = nn.RNN(input_size=embed_dim, hidden_size=64, batch_first=True)

# Input shape: (batch, seq_len, embed_dim)
output, final_hidden = rnn(embedded)
# output: (batch, seq_len, hidden) — h_t at each step
# final_hidden: (1, batch, hidden) — just the last h_t
```

For classification, you usually use the **final hidden state** and feed it to a Linear layer.

---

## 6. Character-Level Generation: a Classic RNN Task

A famous RNN demo: train it on a corpus of names (or Shakespeare, etc.) one character at a time:

```
Input:  "amit"
Target: "mit"     ← shifted by 1 (predict next char)
```

After training, you can generate new names by:
1. Start with a random character
2. Feed it through the RNN
3. Sample the next character from the output distribution
4. Feed that back in
5. Repeat

This is how Karpathy's famous "char-rnn" worked — generated Shakespeare-like text from scratch.

---

## 7. The Three Roles of an RNN's Output

An RNN has two outputs per step. Which you use depends on your task:

| Task | Use |
|------|-----|
| Classification (sentiment, topic) | **Final hidden state** → one prediction |
| Sequence labeling (NER, POS) | **All outputs** → prediction at each step |
| Sequence generation (text gen) | **All outputs** → predict next token at each step |

---

## 8. Vocabulary You'll See

Some terms that come up around sequence models:

- **Time step / sequence step** — one position in the sequence
- **Hidden state / hidden** — the "memory" vector
- **Cell state** — extra memory in LSTMs (not in basic RNNs)
- **Unroll** — visualize the recurrence as a linear chain of operations
- **Teacher forcing** — during training, feed the TRUE previous token instead of the predicted one
- **Many-to-one** — RNN output is single prediction (classification)
- **Many-to-many** — RNN output is sequence (translation, generation)

---

## 9. Why We Still Learn RNNs

If transformers are better, why bother with RNNs?

1. **History** — transformers were a reaction to RNN limitations. Understanding RNNs makes transformers click.
2. **Sequence concepts** — "hidden state," "step-by-step," "teacher forcing" → all introduced by RNNs.
3. **Still used** — time series forecasting, edge devices, simpler problems.
4. **Foundation for LSTM/GRU** — which ARE still widely used.

RNN is the rite of passage before attention.

---

## Mental Model

Think of an RNN as a **note-taker reading a book**:

```
Reads "The"        →  writes a note about "The"
Reads "cat"        →  updates note: "subject is 'the cat'"
Reads "sat"        →  updates note: "the cat is sitting"
Reads "on"         →  updates note: "the cat is sitting on..."
Reads "the"        →  updates note: still expecting an object
Reads "mat"        →  updates note: "the cat sits on the mat"
END                →  gives you the final summary note
```

That "note" is the hidden state. It's a fixed-size vector that has to capture EVERYTHING relevant from the sentence.

The fundamental weakness: it's just ONE note, no matter how long the book. Too much info → forgetting.

---

## Summary

| Concept | What It Does |
|---------|-------------|
| **RNN** | Reads sequence one step at a time |
| **Hidden state** | Fixed-size "memory" updated at each step |
| **`nn.RNN`** | PyTorch's basic RNN — `h_t = tanh(Wx + Uh + b)` |
| **Final hidden** | The summary of the whole sequence |
| **Vanishing gradient** | Long sequences lose signal during backprop |
| **LSTM/GRU** | Improved RNNs with "gates" to manage memory |
| **Transformer** | What replaced RNNs (we'll get to it in Day 19+) |

### Today's evolution:

```
Day 10-11:  Mean of embeddings → ignore order  (BoW thinking)
Day 13:     RNN with hidden state              ← YOU ARE HERE
Day 14:     Bigram language model (RNN's simpler cousin)
Day 15-18:  Attention — the key idea that beats RNNs
Day 19+:    Transformer — attention done at scale
```

**Tomorrow:** Day 14 — bigram language models, the simplest "predict the next word" task. This is the foundation of GPT.
