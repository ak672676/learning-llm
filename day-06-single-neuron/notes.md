# Day 06: The Single Neuron — The Building Block of Every Neural Network

## Why This Matters

You've been doing linear regression (`y = w*x + b`). That's actually ALREADY a single neuron — almost. To make it a true neuron, we add one thing: an **activation function**.

A neural network is just thousands of these neurons stacked together. Understand ONE neuron and you understand the whole thing.

Today we'll:
1. Build a neuron from scratch
2. Learn what activation functions do and why they matter
3. Train a single neuron to learn logic gates (AND, OR)
4. Discover why ONE neuron can't learn XOR — setting up Day 07

---

## 1. What is a Neuron?

A neuron takes some inputs, does a weighted sum, and applies an activation function:

```
         ┌─────────────┐
x1 ──w1─▶│             │
         │  Σ (sum) →  │── activation ──▶ output
x2 ──w2─▶│             │
         └─────────────┘
         +b (bias)
```

### The math:

```
z = w1*x1 + w2*x2 + b       (weighted sum — this is what we've done before)
y = activation(z)            (new! apply a function to z)
```

### Biological inspiration (not crucial but interesting):

Your brain's neurons work similarly:
- Dendrites receive signals (inputs)
- Each connection has a strength (weight)
- The neuron sums them up
- If the total is strong enough, it FIRES (activation)

AI neurons are a simplified math version of this.

---

## 2. Why Activation Functions?

### Without activation: just a line

If you only do `y = w*x + b`, no matter how many neurons you stack:

```
Neuron 1: y1 = w1*x + b1
Neuron 2: y2 = w2*y1 + b2 = w2*(w1*x + b1) + b2 = (w2*w1)*x + (w2*b1 + b2)
          ↑
          Still just a line! Stacking doesn't help.
```

**Without activation, stacking neurons = still just a line.** Useless for anything non-linear.

### With activation: you can learn curves

Adding a non-linear activation function breaks this:

```
y = activation(w*x + b)
```

Now stacking creates genuinely complex functions. This is why neural networks can learn anything — language, images, speech. The activation function is what gives them power.

---

## 3. Common Activation Functions

### Sigmoid: squishes everything to (0, 1)

```
sigmoid(z) = 1 / (1 + e^(-z))

sigmoid(-10) ≈ 0.00
sigmoid( 0)  = 0.50
sigmoid(+10) ≈ 1.00
```

Shape: S-curve. Good for binary classification (output is "probability").

### ReLU: the most popular one

```
relu(z) = max(0, z)

relu(-5) = 0
relu( 0) = 0
relu( 5) = 5
```

Shape: hockey stick. Super simple, works well. Used in almost every modern deep network.

### Tanh: like sigmoid but (-1, 1)

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

tanh(-5) ≈ -1
tanh( 0) =  0
tanh( 5) ≈  1
```

Centered around 0 (unlike sigmoid). Used in RNNs, older networks.

### When to use which?

| Activation | Output range | Best for |
|-----------|-------------|----------|
| **ReLU** | [0, ∞) | Hidden layers (default choice) |
| **Sigmoid** | (0, 1) | Binary classification output |
| **Tanh** | (-1, 1) | Old RNNs, when you need negative values |
| **Softmax** | (0, 1), sums to 1 | Multi-class classification output |

---

## 4. Binary Classification & Binary Cross-Entropy Loss

For linear regression (Day 2), we used **Mean Squared Error** (MSE). But for classifying (is this spam? yes/no), MSE is suboptimal.

### Binary Cross-Entropy (BCE)

When the target is 0 or 1, and the prediction is a probability (from sigmoid):

```
BCE = -[y * log(ŷ) + (1 - y) * log(1 - ŷ)]

Where:
  y = actual label (0 or 1)
  ŷ = predicted probability
```

### Intuition:

- If actual = 1 and predicted = 0.99 → loss ≈ 0 (good!)
- If actual = 1 and predicted = 0.01 → loss huge (bad!)
- If actual = 0 and predicted = 0.99 → loss huge (bad!)
- If actual = 0 and predicted = 0.01 → loss ≈ 0 (good!)

BCE punishes the model MORE for being confident but wrong, which is what we want.

---

## 5. Logic Gates — A Classic Teaching Example

Logic gates are simple "if X and Y, output Z" rules. They're perfect for learning because:
- Only 4 training examples each
- Clear right/wrong answers
- Easy to visualize

### AND Gate

```
Input 1 | Input 2 | Output
--------|---------|-------
   0    |    0    |   0
   0    |    1    |   0
   1    |    0    |   0
   1    |    1    |   1
```

Rule: output = 1 ONLY if both inputs are 1.

### OR Gate

```
Input 1 | Input 2 | Output
--------|---------|-------
   0    |    0    |   0
   0    |    1    |   1
   1    |    0    |   1
   1    |    1    |   1
```

Rule: output = 1 if EITHER input is 1.

### XOR (Exclusive OR)

```
Input 1 | Input 2 | Output
--------|---------|-------
   0    |    0    |   0
   0    |    1    |   1
   1    |    0    |   1
   1    |    1    |   0
```

Rule: output = 1 if inputs are DIFFERENT.

**XOR is special** — one neuron can't learn it. Keep reading...

---

## 6. Linear Separability — Why XOR Is Hard

A single neuron can only draw a **straight line** to separate classes.

### AND is linearly separable:

```
  x2                Can draw a line that puts
   |                the single 1 on one side
 1 | 0    1         and the three 0s on the other ✓
   |
 0 | 0    0
   +---------→ x1
     0    1
```

### OR is linearly separable:

```
  x2                Same — one line separates
   |                the three 1s from the single 0 ✓
 1 | 1    1
   |
 0 | 0    1
   +---------→ x1
     0    1
```

### XOR is NOT linearly separable:

```
  x2                NO straight line can put
   |                the 1s together and the 0s together
 1 | 1    0
   |
 0 | 0    1
   +---------→ x1
     0    1
```

This is the fundamental limitation that sparked the need for **multi-layer networks** (Day 07).

---

## 7. The Single Neuron Training Loop

Same pattern as before, just with a neuron instead of linear regression:

```python
for epoch in range(epochs):
    # Forward: weighted sum + activation
    z = w1*x1 + w2*x2 + b
    y_pred = sigmoid(z)
    
    # Loss: Binary Cross-Entropy
    loss = bce(y_pred, y_true)
    
    # Backward: PyTorch autograd
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The only change from Day 2-3: added `sigmoid` and swapped MSE for BCE.

---

## Mental Model

Think of a neuron as a simple decision-maker:

```
"I look at my inputs, weigh their importance (w1, w2),
add them up with my bias (b), and then the activation
function decides: 'Do I fire? How strongly?'"
```

A neural network is millions of these tiny decision-makers, each learning a different pattern, stacked into layers that combine their decisions.

---

## Summary

| Concept | What It Does |
|---------|-------------|
| **Neuron** | `activation(w1*x1 + w2*x2 + b)` |
| **Weight** | Importance of each input |
| **Bias** | Baseline shift (like intercept in linear) |
| **Activation** | Non-linear function that gives networks power |
| **Sigmoid** | Squishes to (0,1) — good for probabilities |
| **ReLU** | max(0, z) — the default for hidden layers |
| **BCE** | Loss for binary classification (0/1 labels) |
| **Linearly separable** | Classes can be split by a straight line |

### Today's evolution:

```
Day 2-3: Linear regression → y = w*x + b
Day 6:   Single neuron    → y = activation(w*x + b)   ← just added activation!
Day 7:   Multi-layer      → y = f(f(f(x)))            ← stack many neurons
```

**Tomorrow:** We'll stack neurons into layers to solve XOR (which one neuron can't do).
