Absolutely. I’ll treat this as **Lecture 3 — Chunk 1** and build a compact, exam-focused compression. When you send chunks 2 and 3, I’ll keep the same structure and then merge all three into one final Lecture 3 exam sheet.

# Lecture 3 — Exam Compression: Chunk 1

## Self-Supervised Learning Foundations

### 1\. Learning paradigms — know the distinction

| Paradigm | Data | Main idea |
| --- | --- | --- |
| **Transfer learning** | Labelled source → labelled target | Exploit labels learned from another dataset |
| **Semi-supervised learning** | Labelled + unlabelled | Exploit similarity between labelled/unlabelled examples |
| **Self-supervised learning (SSL)** | Unlabelled | Exploit the **structure of the data itself** |
| **Unsupervised learning** | Unlabelled | Discover structure, e.g. clustering/dimensionality reduction |

**Exam definition:**

> **Self-supervised learning = unsupervised pre-training where labels are generated automatically from the data to learn a general-purpose feature extractor.**

---

## 2\. Why learn from unlabelled data?

Unlabelled data still contains **statistical structure**, such as:

- feature correspondences
- patterns
- spatial relationships
- semantic structure
- visual regularities

The goal is to learn these patterns as **representations/features** that can later be useful for another task.

### Core pipeline

```
Large unlabelled dataset
        ↓
Self-supervised pre-training
        ↓
General feature extractor / backbone
        ↓
Fine-tune
        ↓
Different downstream task
```

### Key idea

> We don't necessarily care about the pre-training task itself. We care about the **useful representation learned while solving it**.

---

# 3\. Hybrid learning approaches

In practice, learning paradigms can be combined.

Typical pipeline:

```
Unlabelled data
      ↓
Self-supervised pre-training
      ↓
Labelled + unlabelled data
      ↓
Semi-supervised fine-tuning
      ↓
Transfer to other tasks/domains
```

### Remember

**SSL → Semi-supervised → Transfer learning** can be used sequentially.

---

# 4\. Consistency regularisation recap

The consistency objective encourages a model to produce similar predictions for an example and a perturbed version of the same example.

General idea:

```
x ───────────────→ f(x)
│
└─ perturbation → f(x + η)

Require:
f(x) ≈ f(x + η)
```

The loss combines:

### Labelled data — hard loss

\[ CE\[y_i,f(x_i)\] \]

### Unlabelled data — soft consistency loss

\[ KL\[f(x_j),f(x_j+\\eta)\] \]

Therefore:

\[ \\boxed{ L = CE\[y_i,f(x_i)\] + KL\[f(x_j),f(x_j+\\eta)\] } \]

### Exam interpretation

- **CE** = model must predict the correct ground-truth label.
- **KL** = model should be consistent under perturbation.
- Labelled data anchors the model to the **actual task**.
- Unlabelled data provides consistency information.

---

# 5\. Why consistency alone fails

Suppose there are **no labels at all**.

We use only:

\[ KL\[f(x),f(x+\\eta)\] \]

What happens?

The model can choose a trivial solution:

\[ \\boxed{f(x)=0\\quad\\forall x} \]

Then:

\[ f(x)=f(x+\\eta) \]

so the consistency loss is minimized.

But the representation contains **no useful information**.

### Critical exam point

> **Consistency alone does not guarantee meaningful representations because of trivial/collapse solutions.**

In semi-supervised learning, the supervised CE loss prevents this by anchoring the model to the desired task.

---

# 6\. Classical unsupervised learning

Traditional unsupervised learning includes:

- **clustering**
- **dimensionality reduction**

These methods describe relationships between data points.

### Dimensionality reduction

Compression can learn useful features:

```
High-dimensional data
        ↓
Compact representation
        ↓
Important structure preserved
```

### Intuition

> Learning to represent information compactly forces the model to capture something about the underlying data structure.

---

# 7\. Denoising autoencoders

A denoising autoencoder can learn representations by reconstructing an input from a corrupted version.

```
Clean image
    ↓
add noise
    ↓
corrupted image
    ↓
encoder → representation → decoder
    ↓
reconstructed image
```

But there is an important distinction.

### Easy reconstruction task

Pixel-level denoising may require only **low-level features**.

It can sometimes be solved using a simple filter, such as a median filter.

Therefore:

> A self-supervised task is useful only if solving it forces the model to learn meaningful structure.

---

# 8\. What makes a good self-supervised task?

A good self-supervised task should:

1. Generate labels automatically.
2. Require the model to understand meaningful structure.
3. Encourage useful high-level representations.
4. Be transferable to downstream tasks.

### Bad example

```
Remove simple pixel noise
        ↓
reconstruct pixels
```

Could be solved using simple low-level statistics.

### Better example

```
Hide part of an image
        ↓
predict missing content
```

Image inpainting can require understanding:

- shapes
- context
- object structure
- semantic information

---

# 9\. The core SSL trick

This is probably one of the **most important definitions from this chunk**:

> **Self-supervised learning generates labels automatically from unlabelled data, creating a supervised-like learning problem whose solution requires useful feature representations.**

Think:

```
Unlabelled image
      ↓
Artificial transformation
      ↓
Automatically known "label"
      ↓
Supervised training objective
      ↓
Useful representation
```

No human annotation is required.

---

# 10\. Proxy / pretext objectives

A **pretext task** is an artificial task used to learn useful representations.

The task itself is usually **not the final task**.

Example:

```
Image
 ↓
Rotate it
 ↓
Predict rotation
```

The model learns visual representations while solving the rotation problem.

Then:

```
Throw away pretext head
          ↓
Keep learned backbone
          ↓
Attach downstream classifier
```

### Key distinction

| Term | Meaning |
| --- | --- |
| **Pretext task** | Artificial task used for representation learning |
| **Downstream task** | Actual task we ultimately care about |
| **Backbone / feature extractor** | Network component that learns reusable representations |
| **Pretext head** | Task-specific output layer for the artificial task |

---

# 11\. Rotation prediction — important example

The lecture uses rotation prediction as a classic SSL example.

Given an image:

```
Original
   ↓
randomly rotate
   ↓
0°, 90°, 180°, 270°
```

The model predicts:

\[ \\boxed{{0^\\circ,90^\\circ,180^\\circ,270^\\circ}} \]

### Why isn't this trivial?

To correctly identify the rotation, the model needs to understand the image's **correct orientation**.

Therefore it can learn:

- low-level visual features
- object shapes
- semantic information
- contextual cues
- spatial structure
- even some world knowledge

### Exam answer

**Why can rotation prediction be useful for SSL?**

> Although rotation prediction appears simple, correctly solving it requires understanding the semantic and spatial structure of the image. Thus, training on the artificial rotation task can force the network to learn transferable visual representations.

---

# 12\. Pretext task vs final task

This distinction is extremely exam-worthy.

### Pretext task

```
"What transformation happened to this image?"
```

Example:

> Predict whether the image was rotated by 0°, 90°, 180°, or 270°.

### Downstream task

```
"What do I actually want the model to predict?"
```

Example:

> Classify an image as cat, dog, car, etc.

### Therefore

```
Pretext task
     ↓
learn representation
     ↓
discard/replace pretext head
     ↓
downstream task
```

---

# 13\. High-yield comparison

| Concept | Key point |
| --- | --- |
| **Self-supervised learning** | Automatically generate supervision from unlabelled data |
| **Representation learning** | Learn useful features rather than directly solving a target task |
| **Pretext task** | Artificial task used to learn representations |
| **Downstream task** | Actual task of interest |
| **Consistency regularisation** | Make predictions invariant to perturbations |
| **Consistency-only training** | Can collapse to trivial solution |
| **CE loss** | Uses ground-truth labels |
| **KL consistency loss** | Encourages similar predictions |
| **Denoising** | May learn only low-level features |
| **Image inpainting** | Can require higher-level image structure |
| **Rotation prediction** | Example of a useful pretext task |

---

# 14\. Must-memorise formulas

### Semi-supervised consistency objective

\[ \\boxed{ L = CE\[y_i,f(x_i)\] + KL\[f(x_j),f(x_j+\\eta)\] } \]

where:

- (x_i,y_i): labelled example
- (x\_j): unlabelled example
- (x\_j+\\eta): perturbed unlabelled example
- (CE): supervised/hard loss
- (KL): consistency/soft loss

### Consistency-only objective

\[ \\boxed{ L=KL\[f(x),f(x+\\eta)\] } \]

Problem:

\[ \\boxed{f(x)=0;\\forall x} \]

is a trivial solution.

---

# 15\. Likely exam questions

### Q1. Define self-supervised learning.

**Answer:** Self-supervised learning is a form of unsupervised pre-training where supervision/labels are automatically generated from the structure of unlabelled data. The objective is to learn useful general-purpose representations that can be transferred to downstream tasks.

### Q2. What is a pretext task?

**Answer:** An artificially constructed task whose automatically generated labels provide a learning signal for learning useful representations. It is usually not the final task.

### Q3. Why is consistency regularisation alone insufficient?

**Answer:** Because it permits trivial collapsed solutions such as (f(x)=0) for every input. All inputs then have identical predictions, satisfying consistency without learning meaningful features.

### Q4. Why does consistency regularisation work in semi-supervised learning?

**Answer:** The supervised CE loss anchors the model to the actual target task, while the consistency loss exploits unlabelled data.

### Q5. Why might simple denoising be a poor SSL task?

**Answer:** Because low-level pixel noise can sometimes be removed without learning meaningful semantic representations; simple filters may solve the task.

### Q6. Why can image inpainting be a better pretext task?

**Answer:** Predicting missing image regions can require understanding context, shapes, objects and higher-level image structure.

### Q7. Why is rotation prediction useful?

**Answer:** Correctly predicting image orientation requires understanding spatial and semantic properties of the image, encouraging transferable feature learning.

---

# 16\. Ultra-compressed memory version

If you only have **30 seconds before the exam**, remember:

```
SSL = create labels from unlabelled data
      ↓
train on artificial/pretext task
      ↓
learn reusable representation
      ↓
transfer to downstream task

Consistency:
CE(labelled) + KL(unlabelled perturbations)

Consistency alone:
      ↓
collapse / f(x)=0
      ↓
not useful

Good SSL task:
automatically labelled + requires meaningful structure

Bad task:
can be solved using trivial low-level statistics

Pretext ≠ final task
Pretext = tool for learning the backbone

Rotation prediction:
0°, 90°, 180°, 270°
→ requires spatial/semantic understanding
→ learns transferable features
```

**Chunk 1's central message:**

> **The challenge in self-supervised learning is not merely creating an artificial label; it is designing a pretext task whose solution forces the network to learn representations that are useful beyond the pretext task itself.**
