# Day 10: Text Classification — Bag of Words + Neural Network

## Why This Matters

You've built networks for numbers (Day 1-9). Today we tackle **text** for the first time.

The challenge: a model only understands numbers. So we need to:
1. Turn text into numbers (Day 4 review + new tricks)
2. Build a model that classifies the text
3. Train it end-to-end

Today's approach is simple but powerful: **Bag of Words**. We'll use it to build a sentiment classifier (positive/negative reviews). This is the foundation for understanding why modern approaches (embeddings, transformers) are better.

---

## 1. The Big Picture: Text Classification

A text classifier:

```
Input:  "This movie was amazing! Best film of the year."
        → some processing
Output: positive (1) or negative (0)
```

We'll do this in 3 stages:
1. **Tokenize** the text (split into words)
2. **Vectorize** it (turn the list of words into a number vector)
3. **Classify** with a neural network

---

## 2. Tokenization (Words This Time)

In Day 4 we did character-level tokenization. For text classification, **word-level** is better.

```python
text = "This movie was amazing"
tokens = text.lower().split()    # ['this', 'movie', 'was', 'amazing']
```

Real tokenizers do more (handle punctuation, handle "don't" → ["do", "n't"]), but `lower().split()` is enough for now.

---

## 3. Bag of Words (BoW)

The **simplest** way to turn text into a vector: count how many times each word appears, ignoring order.

### Step 1: Build a vocabulary

Look at all training data, find the unique words:

```python
vocab = ['the', 'movie', 'was', 'great', 'bad', 'amazing', ...]
# Each word gets an INDEX:
#   'the'    → 0
#   'movie'  → 1
#   'was'    → 2
#   'great'  → 3
#   ...
```

### Step 2: Count words in each document

For "the movie was great", the BoW vector is:

```
position:  0    1    2    3    4    5    ...
word:    the movie was great bad amazing
count:    1    1    1    1   0    0     ...
```

So the document becomes a vector of length `vocab_size`.

### Step 3: Feed into a classifier

Now you have a numerical input → use any model.

### Pros and cons

| Pros | Cons |
|------|------|
| Simple to implement | Ignores word order ("not good" looks same as "good") |
| Fast | Vocabulary can be huge (50k+ words) |
| Surprisingly effective | Can't generalize to new words |

This is why we'll move to **embeddings** in Day 11 — they fix all these issues.

---

## 4. Vocabulary Tricks

### Limit vocab size — keep only the K most common words

A real dataset might have 100,000+ unique words. Most are rare and noisy. Common practice: keep top 5,000-20,000.

```python
from collections import Counter

word_counts = Counter()
for doc in training_docs:
    word_counts.update(doc.split())

# Keep top 5000 most common
vocab = [word for word, count in word_counts.most_common(5000)]
```

### Special tokens

You'll always want at least one special token:

- `<unk>` (unknown) — for words not in your vocab
- `<pad>` (padding) — for making sequences the same length (we don't need this for BoW, but will for sequence models)

```python
vocab = ['<unk>'] + word_counts.most_common(4999)
# Now vocab[0] = '<unk>' for any unseen word
```

---

## 5. Stop Words and Preprocessing

Common words like "the", "a", "is" appear in EVERY document. They don't help distinguish positive from negative reviews.

**Stop words** = the most common words you might want to remove.

```python
stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', ...}
tokens = [t for t in tokens if t not in stop_words]
```

For sentiment analysis, this can help (or hurt — sometimes "not", "very" matter). Always test both ways.

---

## 6. The Model: Just a Linear Layer

For BoW, surprisingly, a single linear layer works very well:

```python
class BoWClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(vocab_size, num_classes)
    
    def forward(self, x):
        # x is the BoW vector (shape: batch × vocab_size)
        return self.fc(x)            # logits
```

That's the whole model. Each weight learns something like "if word X appears, lean toward class Y."

For binary classification: 1 output (BCE loss).
For multi-class: K outputs (cross-entropy loss).

---

## 7. The Loss: Cross-Entropy

For multi-class classification, **Cross-Entropy** is the right loss (Day 8):

```python
loss_fn = nn.CrossEntropyLoss()
# Targets are integer class indices (not one-hot!)
# Logits are raw scores (no softmax — CrossEntropyLoss applies it internally)
```

For binary classification, you have 2 options:
1. **2 outputs + cross-entropy** — same as multi-class
2. **1 output + BCE-with-logits** — the binary version

Both work. The community often uses option 1 for consistency.

---

## 8. Evaluation Metrics for Classification

Loss alone doesn't tell you everything. For classification, also track:

### Accuracy
```python
correct = (predictions.argmax(dim=1) == labels).sum()
accuracy = correct / total
```

### Confusion matrix (for binary)
```
                    Predicted
                    Pos    Neg
              Pos  [TP]   [FN]      ← True positives, False negatives
   Actual
              Neg  [FP]   [TN]      ← False positives, True negatives
```

- **Precision** = TP / (TP + FP) — "of those I predicted positive, how many were?"
- **Recall** = TP / (TP + FN) — "of all actual positives, how many did I catch?"
- **F1** = harmonic mean of precision and recall

For balanced datasets, accuracy is fine. For imbalanced (e.g., 99% negative), look at precision/recall too.

---

## 9. Inspecting What the Model Learned

For BoW + Linear classifier, you can interpret the weights directly:

```python
weights = model.fc.weight                   # shape (num_classes, vocab_size)

# For binary classification: weights[1] - weights[0] tells which words push toward "positive"
sentiment = weights[1] - weights[0]
top_positive_words = sentiment.topk(20)
top_negative_words = (-sentiment).topk(20)
```

This is one of the few cases where you can EXPLAIN exactly what the model learned. Modern models (BERT, GPT) are far less interpretable.

---

## 10. Why BoW Has Limits (Setting Up Day 11)

BoW treats each word as totally separate:
- "good" and "great" → completely unrelated to the model
- "not good" and "good" → almost identical (just one extra word in the count)
- "amazing" not in training? → unknown, treated as random noise

The fix: **word embeddings** (Day 11). Instead of one-hot vectors, learn a dense vector per word. Similar words end up with similar vectors.

```
BoW:        each word = one-hot vector of length 5000  (very sparse!)
Embedding:  each word = dense vector of length 100      (compact, learned)
```

Embeddings are the foundation of every modern NLP model.

---

## Mental Model

```
"This movie was amazing!"
    ↓ tokenize
["this", "movie", "was", "amazing"]
    ↓ count words against vocabulary
[0, 1, 1, 0, 0, 1, 0, 0, ..., 1]    (length = vocab_size)
    ↓ Linear layer
[logit_negative, logit_positive]
    ↓ argmax
1 → "positive"
```

Simple, but it actually works for many tasks.

---

## Summary

| Concept | What It Does |
|---------|-------------|
| **Tokenize** | Split text into words |
| **Vocabulary** | Map word → integer index |
| **Bag of Words** | Vector of word counts (ignores order) |
| **Stop words** | Common words that often get removed |
| **Linear classifier** | One layer is enough for BoW |
| **Cross-Entropy** | The right loss for multi-class classification |
| **Accuracy** | Most intuitive metric |
| **Confusion matrix** | Detailed per-class performance |
| **Limitations** | No semantic similarity, ignores word order |

### The journey so far:

```
Day 1-7:  Numbers in, numbers out
Day 8-9:  Production training infrastructure
Day 10:   Text in, classification out (BoW) ← YOU ARE HERE
Day 11:   Word embeddings (replace one-hot with dense vectors)
Day 12:   Project — full sentiment classifier
Day 13+:  Sequence models (RNN, then transformers)
```

**Tomorrow:** We'll build word embeddings — the foundation of every modern NLP model.
