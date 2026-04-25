# Day 09: PyTorch `nn.Module` — Building Clean, Modular Networks

## Why This Matters

So far, we've been writing models like this:

```python
w1 = torch.randn(2, 4, requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)
w2 = torch.randn(4, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

def forward(x):
    h = torch.relu(x @ w1 + b1)
    return torch.sigmoid(h @ w2 + b2)
```

Tedious. For a model with 96 layers (like GPT-3), this would be unmanageable.

PyTorch gives us `nn.Module` and ready-made building blocks (`nn.Linear`, `nn.ReLU`, etc.). Today we'll learn to use them properly. This is how every real PyTorch project is written.

---

## 1. `nn.Module` — The Base Class for All Models

Every model in PyTorch inherits from `nn.Module`. This gives you:

- Automatic parameter tracking (`model.parameters()`)
- Easy device transfer (`model.to('mps')` moves everything)
- State management (`model.train()`, `model.eval()`)
- Saving/loading (`model.state_dict()`)

### Two methods you must implement:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()           # always call this first!
        # define your layers here
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        # define the computation here
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

That's it. PyTorch handles the rest.

### How to use it:

```python
model = MyModel()
output = model(input)        # NOT model.forward(input) — just call it
```

When you do `model(x)`, PyTorch:
1. Calls `forward(x)` internally
2. Tracks the computation for autograd
3. Returns the output

---

## 2. Pre-built Layers — `nn.Linear`, `nn.ReLU`, etc.

PyTorch ships with all the layers you need. Some essential ones:

### Linear layer (fully-connected)
```python
layer = nn.Linear(in_features=10, out_features=5)
# Equivalent to: y = x @ W + b
# Where W has shape (5, 10) and b has shape (5,)
```

### Activation layers
```python
nn.ReLU()       # max(0, x)
nn.Sigmoid()    # 1 / (1 + e^-x)
nn.Tanh()       # tanh(x)
nn.GELU()       # smoother ReLU — used in transformers
```

### Other useful layers (we'll use later)
```python
nn.Dropout(p=0.5)         # randomly zero some values (regularization)
nn.LayerNorm(dim)          # normalize across features (used in transformers)
nn.Embedding(vocab, dim)   # token ID → vector lookup
nn.MultiheadAttention(...) # self-attention (the heart of transformers)
```

---

## 3. `nn.Sequential` — Stack Layers in One Line

For simple networks, `nn.Sequential` lets you skip the class definition:

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

# Use it the same way
output = model(input)
```

Pros: super clean for simple feed-forward networks.
Cons: can't do anything complex (skip connections, conditional logic). For real models, you'll define a class.

---

## 4. Inspecting a Model

Once you have a model, you can inspect it:

```python
print(model)                          # see the architecture
list(model.parameters())              # list of all weight tensors
sum(p.numel() for p in model.parameters())  # total parameter count
model.named_parameters()              # parameters with names
```

Always check the parameter count when you build a new model. A typo can give you 100x more or fewer parameters than you expected.

---

## 5. Building Block: a Reusable "MLP Block"

A common pattern is `Linear → activation → maybe dropout`. Make it reusable:

```python
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
```

Now you can stack them:

```python
class DeepMLP(nn.Module):
    def __init__(self, dims=[10, 32, 32, 32, 1]):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLPBlock(dims[i], dims[i+1])
            for i in range(len(dims) - 1)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

Notice `nn.ModuleList` — when you have a list of layers, use this (NOT a regular Python list). It tells PyTorch "track these for parameter management."

---

## 6. Important: `nn.ModuleList` vs `nn.Sequential` vs Python List

Three ways to store multiple layers:

### Python list — DON'T DO THIS
```python
self.layers = [nn.Linear(10, 5), nn.Linear(5, 1)]  # ✗ broken!
```
PyTorch won't see these as part of the model. `model.parameters()` will return nothing.

### `nn.ModuleList` — for arbitrary layer sequences
```python
self.layers = nn.ModuleList([nn.Linear(10, 5), nn.Linear(5, 1)])
# Iterate manually in forward():
def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    return x
```

### `nn.Sequential` — for "just run them in order"
```python
self.layers = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
)
# Just call it:
def forward(self, x):
    return self.layers(x)
```

Use `Sequential` when you just want to chain layers.
Use `ModuleList` when you need custom logic (loops, conditionals) in `forward`.

---

## 7. Initialization

PyTorch initializes weights automatically (using a sensible default). But you can override:

```python
class CustomInitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        
        # Custom initialization
        nn.init.kaiming_normal_(self.fc.weight)  # good for ReLU
        nn.init.zeros_(self.fc.bias)
```

Common initialization functions:
- `nn.init.kaiming_normal_` — for ReLU networks
- `nn.init.xavier_normal_` — for tanh/sigmoid networks
- `nn.init.normal_(tensor, mean=0, std=0.02)` — what GPT uses

You usually don't need to change defaults. But if your model isn't training, this is a place to look.

---

## 8. Putting It All Together

Here's the canonical PyTorch model template:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Use it:
model = MyModel(10, 32, 1)
print(model)                              # show architecture
print(f"Params: {sum(p.numel() for p in model.parameters())}")

# Standard training loop:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(num_epochs):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

This template works for any feed-forward model.

---

## Mental Model

```
nn.Module = the LEGO baseplate
          ↳ everything you build snaps onto it

Pre-built layers = LEGO bricks
          ↳ nn.Linear, nn.ReLU, nn.Dropout, nn.LayerNorm...

You compose them in:
  nn.Sequential = pre-glued stack of bricks (simple, fast)
  Custom class = build whatever you want (flexible, powerful)
```

For a transformer, you'll need the flexible approach. But you'll still use `nn.Linear` and `nn.LayerNorm` as building blocks.

---

## Summary

| Concept | What It Does |
|---------|-------------|
| `nn.Module` | Base class — tracks parameters automatically |
| `super().__init__()` | Always call first — initializes the parent class |
| `__init__` | Define layers (state) |
| `forward` | Define computation |
| `model(x)` | Calls forward (NOT `model.forward(x)`) |
| `nn.Linear` | Fully-connected layer |
| `nn.Sequential` | Stack of layers run in order |
| `nn.ModuleList` | List of layers (not Python list!) |
| `model.parameters()` | All learnable tensors |
| `model.train() / eval()` | Toggle mode (dropout/batchnorm) |
| `model.to(device)` | Move to GPU |
| `model.state_dict()` | Get all weights (for saving) |

### Why we care

```
Day 1-7:  Built models from raw tensors (educational, but tedious)
Day 8:    Wrote a Trainer class with proper training infrastructure
Day 9:    Now we use nn.Module — the standard way
Day 10+:  We'll build CNNs, RNNs, transformers using these patterns
```

From now on, every model we build will inherit from `nn.Module`. The tedious manual tensor management is over.

**Tomorrow:** Day 10 — we use everything we've learned to build a real text classifier (sentiment analysis on movie reviews).
