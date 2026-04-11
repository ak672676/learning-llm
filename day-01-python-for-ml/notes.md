# Day 01: Python for ML — NumPy, Matrix Ops, Vectorization

## Why This Matters for LLMs

Everything in an LLM is matrix math. When a model "thinks," it's really just multiplying huge matrices together. Before we touch any ML library, we need to be comfortable with:

- **Vectors** — a list of numbers (e.g., a word represented as [0.2, 0.8, 0.1])
- **Matrices** — a grid of numbers (e.g., model weights)
- **Matrix multiplication** — the core operation in every neural network
- **Vectorization** — doing math on entire arrays at once instead of looping (100x faster)

## Key Concepts

### 1. NumPy Arrays vs Python Lists

```python
# Python list — slow, loops needed
a = [1, 2, 3]
b = [4, 5, 6]
result = [a[i] + b[i] for i in range(3)]  # [5, 7, 9]

# NumPy — fast, no loops
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = a + b  # array([5, 7, 9])
```

NumPy is written in C under the hood — it operates on entire arrays at once.

### 2. Shapes and Dimensions

```
Scalar:  42                  → shape: ()
Vector:  [1, 2, 3]          → shape: (3,)
Matrix:  [[1, 2], [3, 4]]   → shape: (2, 2)
3D:      [[[1,2],[3,4]]]    → shape: (1, 2, 2)
```

**Shape matters everywhere in ML.** Most bugs are shape mismatches.

### 3. Dot Product

Two vectors → single number. Measures "how similar" they are.

```
a = [1, 2, 3]
b = [4, 5, 6]
dot = 1×4 + 2×5 + 3×6 = 32
```

In LLMs, dot products determine how much "attention" one word pays to another.

### 4. Matrix Multiplication

Row × Column → single number. Repeat for every row-column pair.

```
A (2×3) @ B (3×2) = C (2×2)
```

Rule: inner dimensions must match. (2×**3**) @ (**3**×2) ✓

### 5. Broadcasting

NumPy auto-expands smaller arrays to match larger ones:

```python
matrix = np.array([[1, 2], [3, 4]])  # shape (2, 2)
vector = np.array([10, 20])          # shape (2,)
result = matrix + vector              # vector broadcasts to each row
# [[11, 22], [13, 24]]
```

### 6. Vectorization = No Loops

```python
# BAD — slow loop
result = np.zeros(1000000)
for i in range(1000000):
    result[i] = a[i] * b[i]

# GOOD — vectorized
result = a * b  # same result, 100x faster
```

## Mental Model

Think of it this way:
- A **word** is a vector of numbers (embedding)
- A **sentence** is a matrix (stack of word vectors)
- A **neural network layer** is a matrix multiplication: output = input @ weights
- **Training** is adjusting those weight matrices so the output gets better

Tomorrow we'll see how "getting better" works (gradients & calculus).
