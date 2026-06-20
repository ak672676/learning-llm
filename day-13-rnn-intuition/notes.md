# Day 13: RNN — In Simple Words

## The Mental Model

Imagine you're reading a book. You can't hold the WHOLE book in your head — but you can carry a small **notebook** that you update after every word.

```
Before reading anything:  notebook is blank
After reading word 1:     notebook has some notes about word 1
After reading word 2:     notebook has notes about words 1 AND 2 (mixed together)
After reading word 3:     notebook has notes about words 1, 2, AND 3
...and so on
```

That's an RNN. The "notebook" is called the **hidden state**. It's just a small vector of numbers (say, 64 numbers).

The cool thing: the SAME notebook size, no matter how long the book. RNNs can read sentences of any length.

---

## Why We Need This

Before today (Day 10-12), all our models had a fatal flaw: they ignored word ORDER.

```
"not bad"  ←→  "bad not"     ← these meant the SAME thing to our old models!
```

An RNN reads left to right and carries context, so it can finally tell these apart.

---

## The RNN Formula — Plain English

The "scary" math is:

```
new_notebook = tanh(weights × [word + old_notebook] + bias)
```

But really, it's just three steps:

```
STEP 1: Glue together (concatenate)
        new_input = [word ; old_notebook]

STEP 2: Multiply by a weight matrix + add bias
        z = W × new_input + b

STEP 3: Squish through tanh
        new_notebook = tanh(z)        ← keeps values between -1 and +1
```

That's literally it. Three operations. Done.

The SAME `W` and `b` are used at every time step. That's why we can handle any sequence length — we only learn ONE set of weights.

---

## What Each Piece of Code Does

### `self.embedding = nn.Embedding(vocab_size, embed_dim)`

A **lookup table** that turns each character ID into a small vector.

Think: a dictionary where:
- "a" → `[0.5, -0.2, 0.8, ...]`
- "b" → `[-0.1, 0.7, 0.3, ...]`
- "c" → `[0.0, 0.4, -0.6, ...]`

These vectors are LEARNED — the model figures out which letters should have similar vectors based on how they're used.

### `self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)`

The "notebook updater." It:
- Takes the current character vector
- Mixes it with the current notebook
- Outputs a new notebook

`batch_first=True` is just convention — it makes the input shape `(batch, time, features)` which feels natural.

### `self.fc = nn.Linear(hidden_dim, vocab_size)`

The "decision maker." At each step, it looks at the notebook and votes for what the NEXT character should be. The output is logits — a score for every possible character.

To convert logits → probabilities → an actual choice, we use softmax + sampling.

### `forward(self, x, h=None)`

The function that runs everything. Three lines:

```python
embedded = self.embedding(x)        # IDs → vectors
output, h = self.rnn(embedded, h)   # vectors → updated notebook at each step
logits   = self.fc(output)           # notebook → next-char scores
```

That's the whole model. Three layers stacked.

---

## The "Shift By 1" Training Trick

For each name, we train the model to PREDICT THE NEXT CHARACTER at every position.

Example: name = "amit", wrapped with markers → ".amit."

```
position:     0   1   2   3   4   5
chars:        .   a   m   i   t   .

INPUT (X):    .   a   m   i   t           (positions 0-4)
TARGET (Y):   a   m   i   t   .           (positions 1-5)
```

The model learns:
- "See '.', predict 'a'"
- "See '.a', predict 'm'" (using the notebook)
- "See '.am', predict 'i'"
- "See '.ami', predict 't'"
- "See '.amit', predict '.'"  (stop!)

ONE name = FIVE training signals. Efficient!

---

## How Generation Works

After training, we ask the model to invent new names. The loop:

```
1. Start with '.' (the start marker).
2. Run it through the model → get probabilities for next character.
3. Sample a random character based on those probabilities.
4. Append it to the output.
5. Feed it back as the next input. The model's notebook carries forward!
6. Stop when we sample '.' (the end marker).
```

This is **autoregressive generation**. The exact same loop ChatGPT uses, just at character level instead of token level.

---

## Temperature — The Creativity Knob

When sampling, we divide logits by a "temperature" `T`:

```
T = 0.1:   Very confident. Almost always picks the most likely letter.
           → Output: predictable, often gets stuck in loops.
           
T = 1.0:   Normal. Sample from the original distribution.
           → Output: balanced.
           
T = 2.0:   Spread out. Less likely letters get picked too.
           → Output: creative, sometimes nonsense.
```

The math: `probs = softmax(logits / T)`. Dividing by a small T makes some logits MUCH bigger (after exp), so softmax concentrates on the winner. Dividing by a big T flattens everything.

Try it: the same trained model can generate "anita" (low T) or "xqzfwt" (high T).

---

## Hidden State Heatmap — What You're Seeing

When we plot the hidden state over time:

```
Columns = each character of the name (in order)
Rows    = the 64 numbers in the notebook
Color   = the value
            red = positive (+1)
            blue = negative (-1)
            white = zero
```

Each column is a snapshot of the notebook AT THAT POINT in reading.

What to look for:
- **Some rows change a lot, others stay stable.** The stable ones might be tracking "is this a name?" The changing ones track local patterns.
- **Big shifts when crossing certain letters** = the model noticed something important.

The exact patterns aren't human-readable — the model invents its own concepts. But the heatmap PROVES information is flowing forward.

---

## Vanishing Gradient — The RNN's Achilles Heel

When training, gradients flow BACKWARDS through every step. Each step multiplies the gradient by some number (often less than 1).

Multiply many times → number shrinks to almost zero:

```
0.7 × 0.7 × 0.7 × ... (50 times) ≈ 0.0000018
```

By the time the gradient reaches step 1, there's nothing left. The model **can't learn long-range dependencies**.

```
"I grew up in France ... [70 words later] ... I speak ___"
                                                       ↑
                Should be "French" — but the RNN's gradient died
                way before reaching "France" during training.
                So it never learned the connection.
```

This is why we move on from RNNs to **transformers** (Day 15+). Transformers don't have this problem because every position can directly look at every other position — no chains of multiplication.

---

## LSTM — A Quick Fix (Not Our Final Answer)

LSTM is "RNN with extra plumbing" to fight vanishing gradient.

Instead of just overwriting the notebook every step, LSTM has THREE "gates":

```
Forget gate:  "What old info should I erase?"
Input gate:   "What new info should I save?"
Output gate:  "What part of memory should I expose to the next layer?"
```

Each gate is a tiny neural network outputting 0 (block) to 1 (let through).

Result: LSTMs can keep info for HUNDREDS of steps. Much better than plain RNNs.

But — we still move past them to transformers, because attention is even better.

---

## When to Use What (Cheat Sheet)

| Task | Best tool |
|------|-----------|
| Learning the basics of sequences | Plain RNN (this day) |
| Time series, simple tasks | LSTM / GRU |
| Modern NLP, long context | Transformer (Day 15+) |
| Anything serious today | Transformer |

---

## What Carries Forward to Transformers

You'll use these CONCEPTS again — even though we abandon the RNN architecture:

| Concept | Where it appears in transformers |
|---------|----------------------------------|
| **Embedding** layer | Same — `nn.Embedding` is the first layer of GPT |
| **Hidden state** | "Token representations" between layers |
| **Per-step output** | Same — predict at every position |
| **Cross-entropy loss** | Same — exactly |
| **Autoregressive generation** | Same — sample one token at a time |
| **Temperature** | Same — knob in every LLM API |

The only thing we throw away is the RECURRENCE (the step-by-step loop). Attention replaces it with "look at all past positions at once."

---

## TL;DR

```
RNN = a tiny brain with a small notebook.
It reads ONE word/letter at a time.
After each word, it updates the notebook.
The notebook carries forward all the context.

Strengths:  handles any sequence length, captures order
Weaknesses: vanishing gradient → can't remember far-back stuff

What it teaches us: the concepts of sequence modeling
What replaces it:    transformers (Day 15+)
```

You won't build GPT with RNNs. But every concept you learned today (hidden state as memory, predict-next-token, autoregressive generation, temperature sampling) carries forward.
