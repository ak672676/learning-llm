# Day 01: Python for ML — NumPy, Matrix Ops, Vectorization

## Why This Matters for LLMs

When ChatGPT generates a response, it's not "thinking" in words. Under the hood, it's doing **millions of math operations on numbers**. Specifically:

- Every word is converted into a list of numbers (called a **vector** or **embedding**)
- The model processes these numbers through layers of **matrix multiplications**
- The output is another list of numbers, which gets converted back into a word

So before we build anything, we need to be comfortable with the math operations that make it all work. The good news: NumPy does the heavy lifting for us.

---

## 1. NumPy — Why Not Just Use Python Lists?

Python lists are general-purpose containers. They can hold anything: strings, numbers, other lists, mixed types. This flexibility makes them **slow** for math.

NumPy arrays are specialized for numbers. They're stored as a continuous block of memory (like C arrays), and operations run in compiled C code instead of interpreted Python.

```python
# Python list — loops through each element one by one
a = [1, 2, 3]
b = [4, 5, 6]
result = [a[i] + b[i] for i in range(3)]  # [5, 7, 9]

# NumPy — processes the entire array in one C operation
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = a + b  # array([5, 7, 9])
```

For 1 million elements, NumPy is typically **50-200x faster** than Python lists. When you're doing billions of operations per second (like in an LLM), this matters enormously.

### When to Use What

| Use Python lists when... | Use NumPy arrays when... |
|--------------------------|--------------------------|
| Mixed types (strings + numbers) | All numbers of the same type |
| Building up data dynamically | Fixed-size numerical data |
| Non-math operations | Math operations (add, multiply, etc.) |

---

## 2. Shapes and Dimensions

Every NumPy array has a **shape** — a tuple that tells you its dimensions. This is the **single most important concept** for ML programming. Most bugs you'll encounter will be shape mismatches.

### The Different Dimensions

```
Scalar:  42                              → shape: ()       → 0 dimensions
  Just a single number. Like a temperature reading.

Vector:  [0.2, 0.8, 0.1, 0.5]          → shape: (4,)     → 1 dimension
  A list of numbers. Like ONE word's embedding.

Matrix:  [[1, 2, 3],                    → shape: (2, 3)   → 2 dimensions
           [4, 5, 6]]                       2 rows, 3 columns
  A grid of numbers. Like a weight matrix in a neural network layer.

3D Tensor: [[[1,2],[3,4]],              → shape: (2, 2, 2) → 3 dimensions
             [[5,6],[7,8]]]
  A batch of matrices. Like processing 2 sentences at once,
  where each sentence is a matrix of word embeddings.
```

### Reading Shape Tuples

The shape tuple is read from left to right, outer to inner:

```
shape (32, 10, 768) means:
  32  → batch size (processing 32 sentences at once)
  10  → sequence length (each sentence has 10 words)
  768 → embedding size (each word is represented by 768 numbers)
```

This exact shape shows up constantly in transformer models.

### Reshaping

You can change the shape without changing the data:

```python
a = np.arange(12)        # [0, 1, 2, ..., 11], shape (12,)
b = a.reshape(3, 4)      # same 12 numbers, now in 3 rows × 4 columns
c = a.reshape(2, 2, 3)   # same 12 numbers, now in 2 batches of 2×3

# Use -1 to auto-calculate one dimension
d = a.reshape(4, -1)     # 4 rows, NumPy figures out columns = 3
```

---

## 3. Dot Product — "How Similar Are Two Vectors?"

The dot product is one of the most important operations in ML. It takes two vectors and produces a single number.

### How It Works

Multiply corresponding elements and add them up:

```
a = [1, 2, 3]
b = [4, 5, 6]

dot product = 1×4 + 2×5 + 3×6
            = 4   + 10  + 18
            = 32
```

### What It Means

The dot product tells you **how similar two vectors are**:
- **Large positive number** → vectors point in the same direction (similar)
- **Near zero** → vectors are unrelated (perpendicular)
- **Large negative number** → vectors point in opposite directions

### Why LLMs Care

In transformer models, the **attention mechanism** uses dot products to decide which words should "pay attention" to which other words:

```
"The cat sat on the mat"

To process "sat", the model computes:
  dot(sat, the)  = 2.1  → some relevance
  dot(sat, cat)  = 8.5  → high relevance! (cat is the one sitting)
  dot(sat, on)   = 3.2  → moderate relevance
  dot(sat, mat)  = 5.1  → relevant (where the sitting happens)
```

The higher the dot product, the more "attention" the model pays to that word. This is the core mechanism that makes LLMs work.

### Cosine Similarity

Raw dot products are affected by vector magnitude (longer vectors = bigger dot products). **Cosine similarity** normalizes for this:

```
cosine_similarity(a, b) = dot(a, b) / (|a| × |b|)
```

This gives a value between -1 and 1, regardless of vector length:
- 1.0 = identical direction
- 0.0 = completely unrelated
- -1.0 = opposite directions

---

## 4. Matrix Multiplication — The Core of Neural Networks

Every layer in a neural network does one main thing: **matrix multiplication**.

### How It Works

Each element in the result is a dot product of a row from the first matrix and a column from the second matrix:

```
A = [[1, 2, 3],      B = [[7, 10],
     [4, 5, 6]]           [8, 11],
                           [9, 12]]

A is (2, 3)           B is (3, 2)

Result C = A @ B is (2, 2):

C[0,0] = row 0 of A · col 0 of B = 1×7 + 2×8 + 3×9 = 50
C[0,1] = row 0 of A · col 1 of B = 1×10 + 2×11 + 3×12 = 68
C[1,0] = row 1 of A · col 0 of B = 4×7 + 5×8 + 6×9 = 122
C[1,1] = row 1 of A · col 1 of B = 4×10 + 5×11 + 6×12 = 167
```

### The Shape Rule

For `A @ B` to work, the **inner dimensions must match**:

```
A shape: (m, n)   @   B shape: (n, p)   →   Result shape: (m, p)
              ^^^              ^^^
              these must be equal

Examples:
  (2, 3) @ (3, 4) = (2, 4)  ✓
  (5, 10) @ (10, 1) = (5, 1) ✓
  (2, 3) @ (4, 5) = ERROR    ✗  (3 ≠ 4)
```

### Why This IS a Neural Network Layer

A single neural network layer is literally just:

```python
output = input @ weights + bias
```

Where:
- `input` = your data, shape (batch_size, input_features)
- `weights` = learnable parameters, shape (input_features, output_features)
- `bias` = learnable offset, shape (output_features,)
- `output` = the result, shape (batch_size, output_features)

For example:
```
input: 3 features (like a 3-number word embedding)
weights: transforms 3 features → 4 features
output: 4 features

(1, 3) @ (3, 4) + (4,) = (1, 4)
```

Stack many of these layers → you get a neural network. Stack ~96 layers with special attention mechanisms → you get GPT-4.

---

## 5. Broadcasting — NumPy's Auto-Expansion

When you operate on arrays with different shapes, NumPy automatically "stretches" the smaller one to match. This is called **broadcasting**.

### How It Works

```python
matrix = np.array([[1, 2, 3],     # shape (3, 3)
                    [4, 5, 6],
                    [7, 8, 9]])

bias = np.array([10, 20, 30])     # shape (3,)

result = matrix + bias
# [[11, 22, 33],     ← [1,2,3] + [10,20,30]
#  [14, 25, 36],     ← [4,5,6] + [10,20,30]
#  [17, 28, 39]]     ← [7,8,9] + [10,20,30]
```

The bias vector (3,) was automatically added to **every row** of the matrix.

### Why This Matters

This is exactly what happens when a neural network adds a bias:
- You have a batch of 32 inputs (shape 32×768)
- The bias is a single vector (shape 768)
- Broadcasting adds that same bias to all 32 inputs at once

Without broadcasting, you'd need explicit loops — which would be much slower.

### Broadcasting Rules

NumPy compares shapes from right to left. Dimensions are compatible if:
1. They're equal, OR
2. One of them is 1 (it gets stretched)

```
(3, 4) + (4,)   → (3, 4)  ✓  (4 matches 4)
(3, 4) + (3, 1) → (3, 4)  ✓  (1 stretches to 4)
(3, 4) + (3,)   → ERROR   ✗  (4 ≠ 3)
```

---

## 6. Vectorization = Don't Write Loops

Whenever you find yourself writing a `for` loop over array elements, there's almost certainly a NumPy function that does it faster.

```python
# SLOW — Python loop
result = 0
for i in range(len(a)):
    result += a[i] * b[i]

# FAST — vectorized
result = a @ b  # or np.dot(a, b)
```

### Common Patterns

```python
# Instead of looping to sum:
total = np.sum(array)           # sum all elements
col_sums = np.sum(array, axis=0)  # sum each column
row_sums = np.sum(array, axis=1)  # sum each row

# Instead of looping to find max:
np.max(array)              # maximum value
np.argmax(array, axis=1)   # INDEX of max per row (used in classification)

# Instead of looping to apply math:
np.exp(array)   # e^x for each element
np.log(array)   # log of each element
np.sqrt(array)  # square root of each element
```

---

## 7. Useful NumPy for ML — Quick Reference

### Creating Arrays
```python
np.zeros((3, 4))           # 3×4 matrix of zeros
np.ones((3, 4))            # 3×4 matrix of ones
np.random.randn(3, 4)      # 3×4 matrix, random from normal distribution
np.arange(10)              # [0, 1, 2, ..., 9]
np.linspace(0, 1, 5)       # [0, 0.25, 0.5, 0.75, 1.0]
```

### Inspecting Arrays
```python
array.shape     # (rows, columns, ...)
array.ndim      # number of dimensions
array.dtype     # data type (float64, int32, etc.)
array.size      # total number of elements
```

### Reshaping
```python
array.reshape(3, 4)    # change shape
array.T                # transpose (swap rows and columns)
array.flatten()        # convert to 1D
array.squeeze()        # remove dimensions of size 1
```

### Math
```python
a + b, a - b, a * b, a / b   # element-wise operations
a @ b                          # matrix multiplication
np.dot(a, b)                   # dot product
np.sum(a), np.mean(a)          # aggregations
np.linalg.norm(a)              # vector length (magnitude)
```

---

## How This Connects to LLMs

| NumPy Concept | LLM Usage |
|---------------|-----------|
| Vector (1D array) | A single word's embedding (e.g., 768 numbers for "cat") |
| Matrix (2D array) | A sentence (stack of word vectors), or a weight matrix |
| 3D tensor | A batch of sentences being processed at once |
| Dot product | Attention scores — how much should word A attend to word B? |
| Matrix multiplication | Every layer: output = input @ weights + bias |
| Broadcasting | Adding bias to every sample in a batch |
| Argmax | Picking the most likely next word from model output |
| Random initialization | Starting weights before training begins |

---

## Mental Model

```
Text:    "The cat sat"
           ↓ tokenize
Tokens:  [The] [cat] [sat]
           ↓ embed (lookup table: word → vector)
Vectors: [[0.2, 0.8, ...], [0.5, 0.1, ...], [0.3, 0.6, ...]]
           ↓ stack into matrix
Matrix:  shape (3, 768) — 3 words, 768 dimensions each
           ↓ multiply by weight matrices (many layers)
Output:  shape (3, 50257) — probability for each possible next word
           ↓ argmax
Next word: "on" (highest probability)
```

Everything in this pipeline is what we learned today: arrays, shapes, matrix multiplication, dot products.

**Tomorrow:** How does the model learn the right weight matrices? → Derivatives & Gradient Descent.
