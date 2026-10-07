# Lecture 3 Chunk 3 — Exam Compression

## 1\. Masked Word Prediction

**Core idea:** Self-supervised language learning can treat text like **linguistic inpainting**.

- Hide/mask a word → model predicts the missing word from context.
- Examples:
- “I put my hat on my \_” → **head**
- “I opened the door and got in my \_” → **truck**
- “The Polish word for brotherhood is \_” → **braterstwo**
- **BERT** uses this type of masked-word prediction for pre-training.
- Key principle: **learn useful representations from context without human labels.**

**Exam phrase:**

> Masked word prediction is a self-supervised pretext task where words are hidden and the model reconstructs them using linguistic context.

---

# 2\. Pretext Generalisation

Different pretext tasks teach different kinds of representations:

- Colourisation → visual/semantic information
- Inpainting → spatial + semantic context
- Jigsaw/context prediction → spatial relationships
- Temporal prediction → dynamics/physics
- Masked word prediction → linguistic/contextual knowledge

### Problem

We don't know beforehand which pretext task will produce the features most useful for our downstream task.

### Solution

**Contrastive learning** provides a more general framework:

> Learn representations where things that should be similar are close, and things that should be different are far apart.

---

# 3\. Contrastive Learning — CORE EXAM TOPIC ⭐⭐⭐

### Fundamental idea

Create different **views** of the same underlying data.

For images:

\[ x \\rightarrow \\text{augmentation}_1(x),\\quad x \\rightarrow \\text{augmentation}_2(x) \]

These are a **positive pair**.

The model learns:

\[ \\text{same underlying example} \\Rightarrow \\text{similar representations} \]

while different examples become **negative pairs**:

\[ \\text{different examples} \\Rightarrow \\text{dissimilar representations} \]

### One-line exam answer

> Contrastive learning learns representations by attracting positive pairs and repelling negative pairs.

---

# 4\. SimCLR

**SimCLR = Simple Framework for Contrastive Learning of Visual Representations**

Key intuition:

> **Different augmented views of the same image should have similar representations.**

### Pipeline

\[ x \\rightarrow \\text{two random augmentations} \\rightarrow \\text{encoder} \\rightarrow h \\rightarrow \\text{projection head} \\rightarrow z \]

Where:

- (x) = original image
- (h) = richer feature representation
- (z) = projection/bottleneck representation used for contrastive loss

### Critical distinction ⭐

**Contrastive loss is applied to****(****z****)****, but downstream tasks use****(****h****)****.**

Why?

- (z): learns augmentation-invariant information specifically useful for contrastive training.
- (h): retains richer information useful for downstream tasks.

**Exam trap:** Don't say the downstream classifier necessarily uses (z).

---

# 5\. Why Positive Pairs Alone Fail

Suppose we only require:

\[ z_i \\approx z_j \]

for two augmented views of the same image.

The trivial solution is:

\[ z_1=z_2=\\cdots=z\_N \]

i.e. **every image gets the same representation.**

This is called:

## Mode Collapse ⭐

Therefore, consistency/agreement alone is insufficient.

We need:

- **Positive pairs:** representations should be similar.
- **Negative pairs:** representations should be different.

---

# 6\. Positive vs Negative Pairs

For an anchor (z\_i):

### Positive

Another augmentation of the **same original sample**:

\[ (z_i,z_j) \]

→ **attract**

### Negative

Representation from a **different sample**:

\[ (z_i,z_k) \]

→ **repel**

Thus:

\[ \\boxed{\\text{Contrastive learning = attraction + repulsion}} \]

A batch supplies both positive and negative examples.

---

# 7\. InfoNCE Loss ⭐⭐⭐

This is probably one of the most important equations from this section.

For positive pair ((z_i,z_j)):

\[ L_i = -\\log \\frac{ \\exp(\\operatorname{sim}(z_i,z_j)/\\tau) }{ \\sum_{k} \\exp(\\operatorname{sim}(z_i,z_k)/\\tau) } \]

where:

- (\\operatorname{sim}(z_i,z_j)) = similarity between representations
- (j) = positive example
- (k) = candidate examples, including negatives
- (\\tau) = **temperature**

### What the loss wants

For positive pair:

\[ \\operatorname{sim}(z_i,z_j) \\uparrow \]

For negative pairs:

\[ \\operatorname{sim}(z_i,z_k) \\downarrow \]

So the positive gets a high probability in the softmax.

---

# 8\. InfoNCE = Cross-Entropy + Temperature

Very important conceptual connection:

> **InfoNCE is essentially cross-entropy applied to a softmax over similarities, with temperature****(****\\tau****)****.**

Think of it as:

\[ \\text{similarities} \\rightarrow \\text{divide by }\\tau \\rightarrow \\text{softmax} \\rightarrow \\text{cross-entropy} \]

The "correct class" is the **positive pair**.

### Temperature (\\tau)

Controls the sharpness/difficulty of the softmax.

\[ \\text{logits} = \\frac{\\operatorname{sim}(z_i,z_k)}{\\tau} \]

Smaller (\\tau) → sharper distribution → stronger emphasis on similarity differences.

**Exam definition:**

> Temperature controls how strongly the loss distinguishes between positive and negative similarities.

---

# 9\. SimCLR Augmentations ⭐⭐⭐

Augmentations are **critical** to SimCLR performance.

Two particularly important findings:

### Crop + resize

Teaches:

> **spatial/scale invariance**

The model learns that an object remains the same despite changes in crop and scale.

### Colour distortion

Teaches:

> **shape/texture-oriented features**

It prevents the model from relying too heavily on colour.

### Major takeaway

> The **composition of multiple augmentations** is crucial; using two complementary augmentations can dramatically improve representation quality.

---

# 10\. Important Augmentation Principle

For self-supervised learning, augmentations should preserve the **semantics** we want the representation to capture.

Examples:

- Brighten image → still same object.
- Darken image → still same object.
- Crop → usually same object.
- Flip → often same semantic class.

But be careful:

> An augmentation can only be useful if it removes irrelevant information while preserving information relevant to the task.

---

# 11\. MoCo and DINO

Contrastive learning developed beyond basic SimCLR.

## MoCo

**MoCo = Momentum Contrast**

- Uses a **momentum encoder**.
- Provides a more stable moving representation/target.
- Helps maintain useful negative representations efficiently.

## DINO ⭐

**DINO = self-distillation with no labels**

Key difference:

> DINO does **not need explicit negative-pair repulsion** like SimCLR.

Instead:

\[ \\text{student} \\rightarrow \\text{teacher} \]

The teacher is a **moving-average version** of the network.

### DINO takeaway

- Self-distillation
- No labels
- Moving-average teacher
- Particularly effective with **Vision Transformers (ViTs)**
- More memory-efficient than approaches requiring many explicit negatives

### Surprising result

DINO + ViT can produce useful object/semantic structure and even **zero-shot segmentation**, despite having **no labels**.

---

# 12\. CLIP — Cross-Modal Contrastive Learning ⭐⭐⭐

SimCLR:

\[ \\text{image view}_1 \\leftrightarrow \\text{image view}_2 \]

CLIP extends this idea across **modalities**:

\[ \\boxed{\\text{image} \\leftrightarrow \\text{text}} \]

Instead of generating two augmented views of the same image, CLIP uses:

> **image-caption pairs representing the same concept.**

For example:

\[ \\text{image of dog} \\leftrightarrow \\text{"a dog running through a field"} \]

The model learns to align image and language representations.

---

# 13\. Why CLIP Is Powerful

CLIP enables **zero-shot image reasoning/classification**.

Instead of training a classifier specifically for classes, you can compare an image embedding against text embeddings such as:

- “a photo of a cat”
- “a photo of a dog”
- “a photo of a car”

and choose the most similar text representation.

### Core idea

\[ \\boxed{\\text{image} \\leftrightarrow \\text{text concept}} \]

This allows transfer across tasks without conventional task-specific labelled training.

---

# 14\. Big Picture: Evolution of Learning

Memorise this progression:

\[ \\boxed{ \\text{Supervised learning} \\rightarrow \\text{Self-supervised learning} \\rightarrow \\text{Contrastive learning} \\rightarrow \\text{Foundation models} } \]

More specifically:

### Traditional supervised pre-training

Huge labelled datasets → powerful image representations.

### Self-supervised pre-training

Huge unlabelled datasets → useful representations without manual labels.

### Contrastive learning

A general self-supervised framework:

\[ \\text{positive pairs attract} \\quad+\\quad \\text{negative pairs repel} \]

### Multimodal contrastive learning

CLIP:

\[ \\text{image} \\leftrightarrow \\text{text} \]

This became especially important for modern **language and multimodal models**.

---

# 15\. High-Yield Comparison Table

| Method | Positive signal | Negative signal | Key idea |
| --- | --- | --- | --- |
| **SimCLR** | Augmented views of same image | Other images | Contrastive attraction + repulsion |
| **MoCo** | Matching views | Explicit negatives | Momentum encoder |
| **DINO** | Student/teacher agreement | No explicit negatives | Self-distillation |
| **CLIP** | Matching image-text pair | Other image-text pairs | Cross-modal contrast |

---

# 16\. Most Likely Exam Questions

### Q1. Why is contrastive learning better than simply matching two augmented views?

Because matching alone permits **mode collapse**, where every sample receives the same representation. Negative examples force representations of different samples apart.

---

### Q2. What is a positive pair in SimCLR?

Two different augmentations/views of the **same original input**.

---

### Q3. What is a negative pair?

Two representations originating from **different input samples**.

---

### Q4. What does InfoNCE do?

It treats the positive example as the correct class in a softmax over similarities and uses cross-entropy to make the positive more similar than negatives.

---

### Q5. What does temperature (\\tau) do?

It controls the sharpness of the softmax over similarities and therefore how strongly the model distinguishes positives from negatives.

---

### Q6. Why are augmentations important in SimCLR?

They define what the representation should be **invariant** to. Different augmentations teach different invariances, and their composition strongly affects representation quality.

---

### Q7. What is the difference between (h) and (z) in SimCLR?

\[ h=\\text{rich feature representation} \]

\[ z=\\text{projection/bottleneck representation used for contrastive loss} \]

Downstream tasks generally use **(****h****)**.

---

### Q8. How does DINO differ from SimCLR?

SimCLR explicitly uses **positive and negative pairs**. DINO instead uses **self-distillation with a moving-average teacher**, avoiding explicit negative-pair repulsion.

---

### Q9. How does CLIP extend contrastive learning?

It replaces two augmented views of the same image with two modalities describing the same concept:

\[ \\text{image} \\leftrightarrow \\text{text} \]

---

### Q10. Why is self-supervised learning useful?

Because **unlabelled data is abundant**, while manually labelled data is expensive. Self-supervised pre-training learns representations that can later be transferred to supervised downstream tasks.

---

# 17\. Absolute Minimum to Memorise 🚨

If you're extremely short on revision time, know these **10 statements**:

1. **Self-supervised learning creates its own supervision from the data.**
2. Pretext tasks learn representations for later **downstream tasks**.
3. A bad pretext task can exploit **spurious/shortcut signals**.
4. **Contrastive learning:** similar things → close; different things → far.
5. **SimCLR:** two augmentations of the same image form a positive pair.
6. Positive-only training causes **mode collapse**.
7. **InfoNCE = softmax over similarities + cross-entropy + temperature.**
8. SimCLR uses (z) for contrastive loss but (h) for downstream representation.
9. **DINO:** self-distillation + moving-average teacher + no explicit negatives.
10. **CLIP:** contrastive learning between **images and text**, enabling powerful zero-shot transfer.

### The single mental model

\[ \\boxed{ \\text{Augment} \\rightarrow \\text{encode} \\rightarrow \\text{positive = same concept} \\rightarrow \\text{negative = different concept} \\rightarrow \\text{InfoNCE} \\rightarrow \\text{use learned representation downstream} } \]

This is the core of the lecture chunk.
