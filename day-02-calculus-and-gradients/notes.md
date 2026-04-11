# Day 02: Calculus Intuition — Derivatives, Chain Rule, Gradients

## Why This Matters for LLMs

Yesterday we learned that a neural network layer does `output = input @ weights + bias`. But those weights start as random numbers. How does the model figure out the **right** weights?

The answer is a simple loop called **training**:

```
1. Feed some data in                          (forward pass)
2. Compare output to the correct answer        (compute loss)
3. Ask: "which way should I nudge each weight   (compute gradients)
   to make the answer a little less wrong?"
4. Nudge the weights slightly                  (update step)
5. Go back to step 1, repeat millions of times
```

Step 3 is where calculus comes in. But don't worry — you don't need to be a math expert. You just need to understand one idea: **a derivative tells you which direction is "downhill."**

---

## 1. What is a Derivative?

### The Simple Version

Imagine you're standing on a hill. The derivative tells you **how steep the ground is under your feet** and **which direction is downhill**.

- **Positive derivative** → the ground is sloping UP to the right → go LEFT to go downhill
- **Negative derivative** → the ground is sloping DOWN to the right → go RIGHT to go downhill
- **Zero derivative** → the ground is FLAT → you might be at the bottom (or a hilltop!)

### The Math Version

The derivative of a function `f(x)` at a specific point tells you: **"if I increase x by a tiny bit, how much does f(x) change?"**

```
f(x) = x²

The derivative is: f'(x) = 2x

Let's check specific points:
  At x = 3:  f'(3) = 6   → f is increasing steeply
  At x = 0:  f'(0) = 0   → f is flat (this is the minimum!)
  At x = -2: f'(-2) = -4  → f is decreasing
```

### Why Do We Care?

In ML, `f` is our **loss function** (how wrong the model is), and `x` is a **weight**. The derivative tells us: "if I increase this weight a tiny bit, does the model get more wrong or less wrong?"

Then we adjust the weight in the direction that makes it **less wrong**.

---

## 2. Computing Derivatives — Two Ways

### Way 1: Numerical (the "just measure it" approach)

You don't need any calculus formulas. Just:
1. Evaluate f at `x`
2. Nudge x by a tiny amount `h` (like 0.00001)
3. Evaluate f again
4. The slope = change in f / change in x

```python
def numerical_derivative(f, x, h=0.00001):
    return (f(x + h) - f(x - h)) / (2 * h)
```

This is like measuring the steepness of a hill by taking one step forward and one step back.

**Pros:** Works for ANY function, no math required
**Cons:** Slow (need to evaluate f twice per weight), slightly approximate

### Way 2: Analytical (the "use calculus rules" approach)

There are simple rules to compute derivatives:

```
Function        Derivative        Example
--------        ----------        -------
x²         →    2x               at x=3: derivative = 6
x³         →    3x²              at x=2: derivative = 12
5x         →    5                constant slope
7 (constant) →   0                flat line
sin(x)     →    cos(x)
eˣ         →    eˣ               (it's its own derivative!)
```

**Pros:** Exact and fast (computed once, reused)
**Cons:** Need to know the rules

**PyTorch uses analytical derivatives internally** (via a system called autograd). You won't need to derive them by hand — but understanding what's happening helps you debug.

---

## 3. What is a Loss Function?

Before we get to gradient descent, we need to understand what we're minimizing.

The **loss function** measures "how wrong is the model?" Lower loss = better model.

### Common Loss: Mean Squared Error (MSE)

```
loss = average of (prediction - actual)²

Example:
  Model predicts: [2.5, 4.0, 6.1]
  Actual values:  [3.0, 4.0, 5.0]
  Errors:         [-0.5, 0.0, 1.1]
  Squared errors: [0.25, 0.0, 1.21]
  MSE = (0.25 + 0.0 + 1.21) / 3 = 0.487
```

Why squared?
- Makes all errors positive (a prediction of +2 off is as bad as -2 off)
- Punishes big errors more than small ones (squared means a 10x error is 100x worse)

### The Goal of Training

Find weights that make the loss as small as possible. That's it. The entire training process is just **minimizing a number**.

---

## 4. Gradient Descent — The Learning Algorithm

This is THE algorithm behind every neural network, every LLM. It's surprisingly simple.

### The Analogy: Hiking in Dense Fog

Imagine you're lost on a hilly landscape in thick fog. You want to reach the lowest valley. You can't see anything, but you **can feel the slope under your feet**.

Strategy:
1. Feel which direction is downhill (compute gradient)
2. Take a step in that direction (update weights)
3. Repeat until you reach the bottom (loss stops decreasing)

### The Algorithm

```python
# Start with random weights
weight = random_number()

# Repeat many times:
for step in range(1000):
    # 1. Compute how wrong we are
    loss = compute_loss(weight)
    
    # 2. Compute the slope (derivative of loss w.r.t. weight)
    gradient = compute_derivative(loss, weight)
    
    # 3. Take a step downhill
    #    Subtract because we want to go OPPOSITE to the slope
    weight = weight - learning_rate * gradient
```

### Why Subtract the Gradient?

- If gradient is **positive** → loss increases when weight increases → **decrease** weight
- If gradient is **negative** → loss decreases when weight increases → **increase** weight
- Subtracting always moves us toward lower loss

### What is an Epoch?

One **epoch** = one pass through all the training data. Training typically runs for many epochs (10, 100, sometimes thousands). Each epoch, the loss should get a bit lower.

---

## 5. Learning Rate — The Most Important Knob

The **learning rate** controls how big each step is.

```
weight = weight - learning_rate × gradient
                  ^^^^^^^^^^^^^^^
                  this controls step size
```

### Too Small (e.g., 0.0001)
- Takes tiny baby steps
- Will eventually reach the minimum, but takes forever
- Like an ant trying to walk down a mountain

### Too Large (e.g., 1.0)
- Takes huge leaps
- Overshoots the minimum, bounces back and forth
- Can actually make the loss INCREASE (diverge)
- Like trying to step down a hill in giant boots — you leap right over the valley

### Just Right (e.g., 0.01 to 0.1, depends on the problem)
- Makes steady progress toward the minimum
- Loss decreases smoothly
- Finding the right learning rate is one of the most important parts of training

### How to Know if Your Learning Rate is Right

Look at the **loss curve** (loss plotted over training steps):
- Smooth decrease → good
- Flat line → too small (or model can't learn this)
- Jagged/increasing → too large
- Fast decrease then flat → just right, already converged

---

## 6. Partial Derivatives — When You Have Multiple Weights

Real models have millions of weights. We need to know the derivative of the loss with respect to **each weight separately**.

A **partial derivative** means: "how does the loss change when I nudge THIS ONE weight, keeping all others fixed?"

### Example

```
f(w1, w2) = w1² + 3·w1·w2 + w2²

Partial derivative with respect to w1 (treat w2 as a constant):
  ∂f/∂w1 = 2·w1 + 3·w2

Partial derivative with respect to w2 (treat w1 as a constant):
  ∂f/∂w2 = 3·w1 + 2·w2
```

### The Gradient = All Partial Derivatives Together

The **gradient** is just a vector containing all the partial derivatives:

```
gradient = [∂f/∂w1, ∂f/∂w2, ∂f/∂w3, ...]
```

It points in the direction of steepest INCREASE. So we go the opposite way (steepest DECREASE).

For each weight:
```
w1 = w1 - learning_rate × ∂loss/∂w1
w2 = w2 - learning_rate × ∂loss/∂w2
... (same for every weight in the model)
```

A GPT model might have 125 million weights. Each training step computes 125 million partial derivatives and updates all of them. That's why GPUs exist.

---

## 7. Chain Rule — Why Backpropagation Works

A neural network is layers stacked on top of each other:

```
input → [layer 1] → [layer 2] → [layer 3] → prediction → loss
         weights1     weights2     weights3
```

To update weights1, we need to know: **"how does changing weights1 affect the final loss?"**

But weights1 doesn't directly touch the loss — it affects layer 1's output, which affects layer 2's output, which affects layer 3's output, which affects the loss.

The **chain rule** handles this by multiplying the derivatives through each step:

```
How loss changes w.r.t. weights1 =

  (how loss changes w.r.t. layer3 output)
  × (how layer3 output changes w.r.t. layer2 output)
  × (how layer2 output changes w.r.t. layer1 output)
  × (how layer1 output changes w.r.t. weights1)
```

### Simple Example

```
y = (3x + 2)²

Break it into steps:
  Step 1: a = 3x + 2        (da/dx = 3)
  Step 2: y = a²             (dy/da = 2a)

Chain rule: dy/dx = dy/da × da/dx = 2a × 3 = 2(3x+2) × 3 = 6(3x+2)
```

### This is Backpropagation

**Backpropagation** = applying the chain rule backwards through the network.

1. Compute the loss (forward: left to right)
2. Compute derivative of loss w.r.t. last layer's output
3. Multiply by derivative of last layer → get gradient for second-to-last layer
4. Keep multiplying backwards until you reach the first layer
5. Now you have gradients for ALL weights → update them all

This is why it's called **back**propagation — we propagate the gradient **backwards** through the network.

---

## 8. Putting It All Together — How an LLM Trains

Here's the full picture of what happens when training a model like GPT:

```
1. Take a batch of text:     "The cat sat on the ___"
2. Model predicts next word:  "dog" (wrong! should be "mat")
3. Loss function measures:    loss = 4.2 (high = bad)
4. Backpropagation:           compute gradient for every weight
5. Update:                    nudge every weight slightly
6. Repeat with next batch:    "I like to eat ___"
7. After millions of batches: model gets good at predicting next words
```

The loss goes from ~10 (random guessing) down to ~2-3 (coherent text generation) over the course of training.

---

## Quick Reference: Derivative Rules You'll See

| Function | Derivative | Used For |
|----------|-----------|----------|
| x² | 2x | Loss functions (MSE) |
| wx + b | w (w.r.t. x), x (w.r.t. w) | Linear layers |
| eˣ | eˣ | Softmax, attention |
| log(x) | 1/x | Cross-entropy loss |
| max(0, x) | 1 if x > 0, else 0 | ReLU activation |

You don't need to memorize these — PyTorch computes them automatically. But recognizing them helps you understand what's happening.

---

## Mental Model Summary

| Concept | Plain English |
|---------|--------------|
| Derivative | "Which direction is downhill?" |
| Gradient | "Which direction is downhill, for ALL weights at once?" |
| Loss | "How wrong is the model right now?" |
| Gradient Descent | "Take a small step downhill. Repeat." |
| Learning Rate | "How big is each step?" |
| Chain Rule | "How to trace the effect of early weights through many layers" |
| Backpropagation | "Compute gradients by applying chain rule backwards" |
| Epoch | "One full pass through all the training data" |

**Tomorrow:** PyTorch does all of this automatically. You just define the forward pass, and it handles gradients, chain rule, and updates for you. That's what `autograd` is.
