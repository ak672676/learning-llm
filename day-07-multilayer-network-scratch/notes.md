# Day 07: Multi-Layer Network From Scratch — Cracking XOR

## Why This Matters

Yesterday, a single neuron FAILED at XOR. Today we solve it by stacking neurons into layers.

This is a HUGE moment in AI history. In 1969, Marvin Minsky published a book proving single neurons can't learn XOR. This discouraged neural net research for ~20 years ("AI winter"). Then in the 1980s, researchers figured out how to train MULTI-LAYER networks with backpropagation — and AI took off.

Today you'll:
1. Build a 2-layer network (hidden layer + output layer) from scratch
2. Derive backpropagation BY HAND through 2 layers
3. Train it to solve XOR
4. Understand what each hidden neuron "learned"

---

## 1. The Architecture: Hidden Layer + Output Layer

```
Input           Hidden Layer          Output
              (2 neurons)
                                           
x1 ────┬────▶  [neuron h1]   ──┐
       │        (ReLU)          │
       │                         ├──▶  [neuron out] ──▶ y
       │                         │      (sigmoid)
x2 ────┴────▶  [neuron h2]   ──┘
                (ReLU)
```

### The math:

**Layer 1 (hidden layer — 2 neurons):**
```
h1 = ReLU(w11*x1 + w12*x2 + b1)
h2 = ReLU(w21*x1 + w22*x2 + b2)
```

**Layer 2 (output — 1 neuron):**
```
y = sigmoid(v1*h1 + v2*h2 + c)
```

### How many parameters?

- Hidden layer: 2 neurons × (2 weights + 1 bias) = **6 params**
- Output layer: 1 neuron × (2 weights + 1 bias) = **3 params**
- **Total: 9 parameters**

Small but enough to solve XOR.

---

## 2. Why 2 Layers Can Solve XOR

A single neuron draws ONE line. Two neurons draw TWO lines. When you combine them, you can create non-linear regions.

### Visual intuition:

```
XOR has 1s at opposite corners:

  x2
   |
 1 | 1 ← red        0 ← blue
   |
 0 | 0 ← blue       1 ← red
   +-------------→ x1
     0              1

Neuron h1 might learn: "output 1 if x1+x2 > 0.5"  (catches (0,1), (1,0), (1,1))
Neuron h2 might learn: "output 1 if x1+x2 > 1.5"  (catches only (1,1))

Output neuron learns: "output 1 if h1 is on but h2 is off"
→ This is exactly XOR!
```

The HIDDEN LAYER creates useful intermediate features. The OUTPUT LAYER combines them to solve the problem.

---

## 3. Backpropagation Through Two Layers

Remember Day 02's chain rule? Now we apply it THROUGH 2 layers.

### The full forward pass:

```
z1 = W1 @ x + b1        (layer 1 pre-activation)
h  = ReLU(z1)           (layer 1 output)
z2 = W2 @ h + b2        (layer 2 pre-activation)
y  = sigmoid(z2)        (final output)
loss = BCE(y, target)
```

### Backprop (chain rule applied backwards):

Starting from the loss and moving backwards:

```
∂loss/∂y  = (y - target) / [y*(1-y)]          (BCE derivative)
∂y/∂z2    = y * (1 - y)                        (sigmoid derivative)
∂loss/∂z2 = y - target                          (simplifies nicely!)

∂loss/∂W2 = ∂loss/∂z2 × h                       (chain rule)
∂loss/∂b2 = ∂loss/∂z2

∂loss/∂h  = W2 × ∂loss/∂z2                      (push gradient back)
∂h/∂z1    = 1 if z1 > 0 else 0                  (ReLU derivative)
∂loss/∂z1 = ∂loss/∂h × ∂h/∂z1

∂loss/∂W1 = ∂loss/∂z1 × x
∂loss/∂b1 = ∂loss/∂z1
```

### What to notice:

1. We go from the END backwards (backprop = reverse order)
2. Each layer's gradient depends on the NEXT layer's gradient (hence the name)
3. ReLU's derivative is super simple: 1 if input was positive, 0 otherwise
4. The chain multiplies gradients through each step

**We could derive all this by hand... but PyTorch does it for us.** Today we'll show both: the raw math AND the PyTorch version.

---

## 4. The Universal Approximation Theorem

**Claim:** A neural network with ONE hidden layer of enough neurons can approximate ANY continuous function.

This is one of the most important theorems in ML. It means:
- Neural networks aren't limited to specific function shapes
- With enough neurons, you can fit any pattern
- You just need enough hidden neurons AND enough training

In practice, DEEPER networks (more layers) work better than WIDER (one huge layer), even though both have similar theoretical power. That's why we have "deep" learning — not "wide" learning.

---

## 5. Initialization Matters

For a single neuron, random initialization is fine. For multi-layer networks, initialization matters A LOT.

### Why?

If all weights start at the same value (like all zeros):
- Both hidden neurons compute the same thing
- Both get the same gradient
- Both update identically
- You effectively have ONE neuron (not two)

**Solution: random initialization breaks symmetry.**

```python
# Bad: all same
W = torch.zeros(2, 2)  # all neurons start identical → stay identical!

# Good: random
W = torch.randn(2, 2) * 0.1  # each neuron learns something different
```

Modern networks use smart initializations (Xavier, He), but random small values work fine for our 2-layer net.

---

## 6. The Training Loop (Same as Before!)

Here's the beautiful part: the training loop doesn't change AT ALL.

```python
for epoch in range(epochs):
    y_pred = model(x)                    # forward
    loss = loss_fn(y_pred, target)       # measure
    optimizer.zero_grad()                 # clear
    loss.backward()                       # all gradients, automatically
    optimizer.step()                      # update
```

This is the core pattern you'll use for GPT-4. Networks get bigger, but the loop stays the same.

---

## Mental Model

```
Layer 1 neurons: extract useful features from raw input
Layer 2 neurons: combine features to make the final decision

Example for XOR:
  h1 might learn: "input has at least one 1"
  h2 might learn: "input has two 1s"
  Output learns: "answer = h1 AND NOT h2"
                 = "at least one 1, but not two 1s"
                 = XOR!
```

Each layer is a **learned transformation** of the previous layer's output.

---

## Summary

| Concept | What It Does |
|---------|-------------|
| **Multi-layer network** | Stack of linear layers + activation functions |
| **Hidden layer** | Layer between input and output; learns features |
| **Backpropagation** | Apply chain rule through all layers (backwards) |
| **Parameter count** | (inputs × hidden) + hidden + (hidden × output) + output |
| **Universal approximator** | Enough neurons can learn ANY function |
| **Random init** | Break symmetry so neurons learn different things |

### The progression:

```
Day 06: 1 neuron   → linear boundary → AND, OR ✓, XOR ✗
Day 07: 2 layers   → non-linear boundary → AND, OR, XOR all ✓
Day 08+: Even more layers → more complex patterns → eventually GPT
```

**Tomorrow:** We'll build a proper training loop with validation, checkpointing, and learning rate scheduling (Day 08).
