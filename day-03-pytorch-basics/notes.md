# Day 03: PyTorch Basics — Tensors, Autograd, and GPU

## Why This Matters

In Day 02, we wrote gradient descent from scratch — computing derivatives by hand, writing our own update loop. That worked for `y = wx + b` with 2 parameters.

But what about a model with 125 million parameters? You can't hand-derive 125 million gradient formulas. You need a tool that:

1. **Automatically computes gradients** for any computation (no matter how complex)
2. **Runs on GPU** for speed (matrix math is 10-100x faster on GPU)
3. **Provides building blocks** (layers, loss functions, optimizers) so you don't reinvent the wheel

That tool is **PyTorch**. It's what OpenAI, Meta, Google, and most AI researchers use.

---

## 1. What is a Tensor?

A tensor is just NumPy's `ndarray` but with two superpowers:

| Feature | NumPy Array | PyTorch Tensor |
|---------|-------------|----------------|
| Math operations | ✓ | ✓ |
| GPU support | ✗ | ✓ |
| Auto-gradient computation | ✗ | ✓ |

The word "tensor" sounds fancy, but it just means "multi-dimensional array":

```
Scalar (0D tensor):  42
Vector (1D tensor):  [1, 2, 3]
Matrix (2D tensor):  [[1, 2], [3, 4]]
3D tensor:           a batch of matrices
```

Same shapes as NumPy — if you understood Day 01, you already understand tensors.

### Creating Tensors

```python
import torch

# From a list
a = torch.tensor([1.0, 2.0, 3.0])

# From NumPy (they share memory — changing one changes the other!)
import numpy as np
np_array = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(np_array)

# Common initializations
zeros = torch.zeros(3, 4)        # 3×4 matrix of zeros
ones = torch.ones(3, 4)          # 3×4 matrix of ones
random = torch.randn(3, 4)       # 3×4 random normal values (mean=0, std=1)
```

### Tensor Operations — Same as NumPy

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

a + b           # element-wise add → [5, 7, 9]
a * b           # element-wise multiply → [4, 10, 18]
a @ b           # dot product → 32.0
A @ B           # matrix multiplication (same as NumPy)
a.shape         # shape tuple
a.reshape(1, 3) # reshape
```

If you know NumPy, you basically know PyTorch tensors. The API is almost identical on purpose.

---

## 2. Autograd — Automatic Gradient Computation

This is PyTorch's killer feature. Remember in Day 02, we had to manually derive:

```python
# Day 02: we figured these out by hand
dw = np.mean(2 * (y_pred - y_true) * X)
db = np.mean(2 * (y_pred - y_true))
```

With PyTorch, you just say `loss.backward()` and it computes ALL gradients automatically, no matter how complex the computation.

### How It Works

```python
# 1. Create a tensor and tell PyTorch to track its gradients
w = torch.tensor(2.0, requires_grad=True)

# 2. Do any computation you want
y = w ** 2 + 3 * w + 1

# 3. Call backward() — PyTorch computes dy/dw automatically
y.backward()

# 4. The gradient is stored in w.grad
print(w.grad)  # 2*w + 3 = 2*2 + 3 = 7.0
```

### What happens behind the scenes?

When `requires_grad=True`, PyTorch builds a **computation graph** as you do math:

```
w = 2.0
  ↓
w² → 4.0     (remembers: derivative of w² is 2w)
  ↓
+ 3w → 10.0  (remembers: derivative of 3w is 3)
  ↓
+ 1 → 11.0   (remembers: derivative of constant is 0)
  ↓
y = 11.0
```

When you call `y.backward()`, PyTorch walks this graph **backwards** (backpropagation!) and applies the chain rule at each step. This is exactly what we did by hand in Day 02 — but PyTorch does it automatically for ANY computation.

### The Key Insight

You write the forward pass (the computation). PyTorch figures out the backward pass (the gradients) for free. This is why deep learning is possible — you can build arbitrarily complex models and still get exact gradients.

---

## 3. Gradient Descent in PyTorch

### The Manual Way (to see what's happening)

```python
w = torch.tensor(2.0, requires_grad=True)
learning_rate = 0.1

for step in range(20):
    # Forward pass
    loss = (w - 5) ** 2  # minimum at w=5
    
    # Backward pass — compute gradients
    loss.backward()
    
    # Update weights (must disable gradient tracking for the update)
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # Zero the gradient (PyTorch accumulates gradients by default!)
    w.grad.zero_()
```

### Important: Why `w.grad.zero_()`?

PyTorch **adds** new gradients to existing ones (accumulates). This is useful for some advanced techniques, but for basic training you need to reset gradients to zero before each step. Forgetting this is a very common bug.

### The PyTorch Way (using built-in optimizer)

```python
w = torch.tensor(2.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

for step in range(20):
    loss = (w - 5) ** 2
    
    optimizer.zero_grad()  # zero gradients
    loss.backward()         # compute gradients
    optimizer.step()        # update weights
```

This 3-line pattern — `zero_grad()`, `backward()`, `step()` — is the training loop for every PyTorch model, from a single weight to GPT-4.

---

## 4. `torch.no_grad()` — When NOT to Track Gradients

Tracking gradients uses memory and compute. You only need it during training. Turn it off for:

- **Inference** (making predictions with a trained model)
- **Evaluation** (measuring performance on test data)
- **Manually updating weights** (the update itself shouldn't be part of the graph)

```python
# During training: gradients ON
loss.backward()

# During inference: gradients OFF
with torch.no_grad():
    predictions = model(test_data)
```

---

## 5. GPU Acceleration

Matrix math is "embarrassingly parallel" — you can multiply thousands of rows at the same time. GPUs have thousands of cores designed for exactly this.

### Moving Tensors to GPU

```python
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# On Mac with Apple Silicon:
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Move tensors to GPU
x = torch.randn(1000, 1000).to(device)
w = torch.randn(1000, 1000).to(device)

# Operations happen on GPU automatically
result = x @ w  # this runs on GPU
```

### Important Rule: All tensors in a computation must be on the same device

```python
# This will ERROR:
cpu_tensor = torch.randn(3, 3)
gpu_tensor = torch.randn(3, 3).to('cuda')
result = cpu_tensor + gpu_tensor  # ERROR! different devices
```

---

## 6. Common PyTorch Patterns You'll See Everywhere

### Pattern 1: The Training Loop
```python
for epoch in range(num_epochs):
    predictions = model(inputs)        # forward pass
    loss = loss_fn(predictions, targets)  # compute loss
    optimizer.zero_grad()               # clear old gradients
    loss.backward()                     # compute new gradients
    optimizer.step()                    # update weights
```

### Pattern 2: NumPy ↔ PyTorch Conversion
```python
# NumPy → PyTorch
tensor = torch.from_numpy(numpy_array)

# PyTorch → NumPy (must be on CPU and detached from grad)
numpy_array = tensor.detach().cpu().numpy()
```

### Pattern 3: Check Shapes Obsessively
```python
print(x.shape)      # always check shapes when debugging
print(w.shape)
print((x @ w).shape)
```

---

## Comparison: Day 02 (Manual) vs Day 03 (PyTorch)

| What | Day 02 (NumPy) | Day 03 (PyTorch) |
|------|----------------|-------------------|
| Create weights | `w = np.random.randn()` | `w = torch.randn(1, requires_grad=True)` |
| Forward pass | `y_pred = w * X + b` | `y_pred = w * X + b` (same!) |
| Compute loss | `loss = np.mean(...)` | `loss = torch.mean(...)` (same!) |
| Compute gradients | Derive by hand, code the formula | `loss.backward()` (automatic!) |
| Update weights | `w = w - lr * dw` | `optimizer.step()` |
| Zero gradients | Not needed (we recompute) | `optimizer.zero_grad()` (required!) |

The forward pass is identical. The magic is all in the backward pass — PyTorch handles it automatically.

---

## Mental Model

```
Day 01: "Neural networks are just matrix multiplication"
Day 02: "Training is just gradient descent — measure slope, take a step"
Day 03: "PyTorch automates the gradient computation so we can build anything"
```

Think of PyTorch as a calculator that:
1. Lets you type any math formula (forward pass)
2. Automatically tells you the derivative with respect to any variable (backward pass)
3. Runs on GPU for speed

**Tomorrow:** We'll handle real data — loading text, splitting into batches, and tokenizing (converting words to numbers).
