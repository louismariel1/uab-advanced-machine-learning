# Lecture 3 — Exam Compression: Self-Supervised & Contrastive Learning

## 1\. Core idea: Self-Supervised Learning

**Self-supervised learning (SSL)** learns representations from **unlabelled data** by creating supervision automatically from the data itself.

```
Unlabelled data
      ↓
Pretext / proxy task
      ↓
Learn representation
      ↓
Discard/replace pretext head
      ↓
Downstream task
```

### Must know

> **The pretext task is not the final goal. Its purpose is to force the model to learn transferable representations.**

A good SSL task must be difficult to solve using trivial shortcuts and should require learning useful structure.

---

# 2\. Learning paradigms

| Paradigm | Key idea |
| --- | --- |
| **Supervised** | Human-provided labels |
| **Semi-supervised** | Labelled + unlabelled data |
| **Unsupervised** | Discover structure without labels |
| **Self-supervised** | Generate supervision automatically from data |
| **Transfer learning** | Reuse learned representation on another task |

### SSL as transfer learning

```
Large unlabelled dataset
        ↓
SSL pre-training
        ↓
general feature extractor
        ↓
supervised fine-tuning
        ↓
downstream task
```

---

# 3\. Consistency regularisation

Core idea:

> Different perturbations/views of the same input should produce similar predictions/representations.

\[ x \\rightarrow f(x) \]

\[ x+\\eta \\rightarrow f(x+\\eta) \]

Require:

\[ f(x)\\approx f(x+\\eta) \]

Typical semi-supervised objective:

\[ \\boxed{ L=CE\[y_i,f(x_i)\]+KL\[f(x_j),f(x_j+\\eta)\] } \]

- **CE** → labelled data / ground truth.
- **KL** → consistency on unlabelled data.

### Why consistency alone fails

If we only minimise:

\[ KL\[f(x),f(x+\\eta)\] \]

the model can collapse:

\[ \\boxed{f(x)=0\\quad\\forall x} \]

Every input receives the same output, so consistency is perfect but the representation is useless.

### Key lesson

> **Consistency encourages invariance, but invariance alone does not prevent collapse.**

This becomes important later in contrastive learning.

---

# 4\. Pretext tasks

A **pretext task** is an artificially constructed task with automatically available targets.

Examples:

- rotation prediction
- image colourisation
- image inpainting
- jigsaw/spatial arrangement
- temporal prediction
- masked word prediction

### Good vs bad pretext task

**Good:**

> Solving the task requires meaningful semantic/spatial/temporal structure.

**Bad:**

> The task can be solved using simple low-level statistics or shortcuts.

### Example: denoising

Denoising may only require low-level filtering.

Therefore:

\[ \\text{good pretext task} \\neq \\text{task that is merely easy to supervise} \]

It should force useful representation learning.

---

# 5\. Shortcut / spurious-signal problem ⭐

A model may solve the pretext task without learning what we intended.

### Example

For shuffled image patches, we want:

> Understand visual content → recover spatial arrangement.

But the model may detect tiny **chromatic aberration/camera artefacts** indicating patch position.

So:

```
Intended:
image semantics → solve task

Shortcut:
tiny statistical artefact → solve task
```

### Exam definition

> **A spurious signal is an unintended statistical cue that allows the model to solve the pretext task without learning the desired representation.**

### Central SSL design principle

\[ \\boxed{ \\text{Prevent shortcuts → force useful representation learning} } \]

---

# 6\. How to evaluate SSL

High pretext-task accuracy does **not necessarily** mean useful representations.

Better evaluation:

```
SSL pre-training
      ↓
freeze/fine-tune representation
      ↓
downstream supervised task
      ↓
measure performance
```

### Exam point

> **The ultimate test of an SSL representation is transfer to downstream tasks, not merely success on the pretext task.**

---

# 7\. Data augmentation

Augmentation modifies the input while ideally preserving the relevant semantics.

Examples:

- crop
- resize
- flip
- rotation
- colour distortion
- grayscale
- noise
- occlusion
- random erasing

### Why augmentation?

It teaches the model:

> **Which changes should not affect the representation.**

Thus augmentation encourages **invariance** and robustness.

---

# 8\. Label-preserving vs semantic-preserving

### Supervised learning

If:

\[ (x,y) \]

then ideally:

\[ (T(x),y) \]

where (T) is a label-preserving transformation.

### SSL

There may be no human label.

Instead, think:

\[ \\boxed{\\text{preserve useful semantics}} \]

The transformation can then help construct the self-supervised objective.

---

# 9\. Cutout / Random Erasing

Randomly remove a region of an image.

### Why useful?

Without erasing:

> Model may rely heavily on one easy feature.

With erasing:

> That feature can disappear → model must use other features.

So:

\[ \\boxed{\\text{Cutout encourages diverse/robust feature use}} \]

---

# 10\. MixUp

Take two examples and linearly interpolate **both inputs and labels**.

\[ \\boxed{x'=\\lambda x_1+(1-\\lambda)x_2} \]

\[ \\boxed{y'=\\lambda y_1+(1-\\lambda)y_2} \]

Example:

\[ y'=0.7(\\text{cat})+0.3(\\text{dog}) \]

### Why MixUp?

- smooths representations
- reduces overconfidence
- encourages local linearity
- acts like label smoothing
- encourages feature sharing/reuse

### Memorise

> **MixUp = mix the whole images + mix the labels.**

---

# 11\. CutMix

Combines ideas from Cutout and MixUp.

Instead of blending entire images:

```
Image A
   +
patch from Image B
   ↓
mixed image
```

Label is mixed according to the area contributed by each image.

If 75% is A and 25% is B:

\[ \\boxed{y'=0.75y_A+0.25y_B} \]

### Memorise

> **CutMix = cut a patch from one image + paste into another + area-weight labels.**

---

# 12\. AutoAugment vs RandAugment

### AutoAugment

Learns augmentation policies automatically, using **reinforcement learning**.

\[ \\text{search policies}\\rightarrow\\text{good augmentation policy} \]

Problem:

> Computationally expensive.

### RandAugment

Simplifies this:

- randomly choose (N) transformations
- control their strength with magnitude (M)

\[ \\boxed{\\text{RandAugment}=(N,M)} \]

Typical intuition:

- (N) = number of transformations
- (M) = magnitude/strength

### Key comparison

| AutoAugment | RandAugment |
| --- | --- |
| Learns policy | Random policy |
| RL search | No expensive RL search |
| Dataset-specific search | Simple/general |
| Expensive | Much cheaper |

---

# 13\. Weak vs strong augmentation

### Weak augmentation

Mainly:

\[ \\boxed{\\text{regularisation}} \]

Helps prevent overfitting / provides relatively stable targets.

### Strong augmentation

Mainly:

\[ \\boxed{\\text{learn invariance}} \]

Forces the model to ignore irrelevant variation.

---

# 14\. Temporal context

SSL can exploit **time**, not just spatial structure.

Example:

\[ \\text{current frames}\\rightarrow\\text{predict future frame} \]

This can encourage learning:

- temporal relationships
- dynamics
- physics
- world knowledge

### Key idea

> Predicting what happens next requires understanding what is happening now.

---

# 15\. Masked Word Prediction / Linguistic Inpainting

Hide part of a sentence and predict it.

Example:

> I put my hat on my **\[MASK\]**.

→ **head**

Another:

> The Polish word for brotherhood is **\[MASK\]**.

→ **braterstwo**

### Core idea

The text itself provides the supervision.

```
Original text
     ↓
mask word
     ↓
context
     ↓
predict missing word
```

This is essentially:

\[ \\boxed{\\text{linguistic inpainting}} \]

### Key example

**BERT** uses masked language modelling as a pre-training objective.

---

# 16\. Pretext generalisation → Contrastive Learning ⭐⭐⭐

Different pretext tasks learn different features:

```
rotation
colourisation
inpainting
jigsaw
...
   ↓
different representations
```

Problem:

> Which pretext task is best for our downstream task?

Instead of manually choosing one specific pretext task, contrastive learning provides a more general framework.

---

# 17\. Contrastive learning — central idea ⭐⭐⭐

> **Similar/positive samples should have similar representations; dissimilar/negative samples should have different representations.**

For images:

\[ \\boxed{\\text{different views of same data} \\rightarrow \\text{similar representations}} \]

This is the key intuition behind **SimCLR**.

---

# 18\. SimCLR

**SimCLR = Simple Framework for Contrastive Learning of Visual Representations**

Core process:

```
Image x
 ├── augmentation 1 → view 1 → encoder → z_i
 └── augmentation 2 → view 2 → encoder → z_j

same image → positive pair
other images → negative pairs
```

### Positive pair

Two different augmentations/views of the **same input**.

\[ (z_i,z_j) \]

### Negative pair

Representations from **different inputs**.

### Objective

\[ \\boxed{ \\text{positive pairs attract} \\quad+\\quad \\text{negative pairs repel} } \]

---

# 19\. Why negatives are necessary ⭐⭐⭐

If we only optimise:

\[ \\text{similarity}(z_i,z_j) \]

for positive pairs, the model can map everything to the same point:

\[ z_1=z_2=\\cdots=z\_N \]

This is **mode collapse**.

Therefore:

\[ \\boxed{ \\text{positive agreement alone is insufficient} } \]

Contrastive learning adds negative examples:

```
Positive → agree
Negative → disagree
```

This prevents the trivial all-identical representation.

---

# 20\. SimCLR batch structure

A batch contains:

- positive pairs
- negative pairs

For each anchor (z\_i):

```
             positive
                ↓
             z_j  ✓
                |
anchor z_i -----+
                |
       negatives ↓
       z_k, z_l, ... ✗
```

The model must identify the positive among the candidate representations.

---

# 21\. InfoNCE loss ⭐⭐⭐

For positive pair ((z_i,z_j)):

\[ \\boxed{ L_i= -\\log \\frac{ \\exp(\\operatorname{sim}(z_i,z_j)/\\tau) }{ \\sum_{k=0}^{N} \\exp(\\operatorname{sim}(z_i,z_k)/\\tau) } } \]

where:

- (z\_i) = anchor
- (z\_j) = positive
- (z\_k) = candidates/negatives
- (\\operatorname{sim}(\\cdot,\\cdot)) = similarity, typically cosine similarity
- (\\tau) = **temperature**

### What should happen?

For positive:

\[ \\operatorname{sim}(z_i,z_j)\\uparrow \]

For negatives:

\[ \\operatorname{sim}(z_i,z_k)\\downarrow \]

---

# 22\. Temperature (\\tau) ⭐⭐

Temperature controls the **sharpness/difficulty of the contrastive distribution**.

In:

\[ \\exp(\\operatorname{sim}/\\tau) \]

smaller (\\tau) makes differences in similarity more pronounced.

### Memorise simply

> **Temperature controls how strongly the model focuses on differences between similarities, particularly making negatives easier/harder to distinguish.**

---

# 23\. InfoNCE = Cross-Entropy with Softmax ⭐⭐⭐

This is an important conceptual connection.

InfoNCE can be understood as:

\[ \\boxed{\\text{Cross-entropy over similarity scores}} \]

The similarity scores act like logits:

\[ \\hat y_k = \\frac{\\exp(\\operatorname{sim}(z_i,z_k)/\\tau)} {\\sum_j\\exp(\\operatorname{sim}(z_i,z_j)/\\tau)} \]

The correct class is the **positive example**.

Therefore:

```
Anchor
  ↓
similarity with candidates
  ↓
softmax
  ↓
positive should receive probability ≈ 1
  ↓
cross-entropy
```

### Exam answer

> **InfoNCE is essentially cross-entropy applied to a softmax over pairwise similarity scores, with the positive pair treated as the correct class and temperature scaling the logits.**

---

# 24\. SimCLR: (h) vs (z) ⭐⭐⭐

This is a very testable implementation detail.

SimCLR has:

```
input
  ↓
encoder
  ↓
h
  ↓
projection head
  ↓
z
  ↓
contrastive loss
```

### (h)

Higher-dimensional representation from the encoder.

Used for:

\[ \\boxed{\\text{downstream tasks}} \]

Contains richer information.

### (z)

Projection/bottleneck representation.

Used for:

\[ \\boxed{\\text{contrastive loss}} \]

Learns an augmentation-invariant compressed representation.

### Critical distinction

> **Contrastive loss is applied to****(****z****)****, but downstream tasks use****(****h****)****.**

Why?

Because forcing the representation itself to discard all augmentation-sensitive information can remove useful information. The projection head allows the contrastive objective to impose invariance while preserving richer features in (h).

---

# 25\. SimCLR augmentations ⭐⭐⭐

Augmentation choice is **critical**.

Two particularly important findings:

### Crop + resize

Encourages:

\[ \\boxed{\\text{spatial/scale invariance}} \]

### Colour distortion

Encourages:

\[ \\boxed{\\text{shape/texture features}} \]

### Most important

> **The composition of augmentations matters, not merely applying one augmentation.**

---

# 26\. Why augmentations are central to contrastive learning

The positive pair is created by applying different augmentations to the same input:

\[ x_i^1=T_1(x) \]

\[ x_i^2=T_2(x) \]

Then:

\[ \\boxed{(T_1(x),T_2(x))=\\text{positive pair}} \]

Therefore the augmentation defines:

> **What information the representation should become invariant to.**

This is one of the deepest ideas in SimCLR.

---

# 27\. Beyond SimCLR

Contrastive learning has evolved beyond basic negative-pair methods.

### MoCo

**Momentum Contrast**

Uses a **momentum/stable encoder** to create a moving/stable target.

### DINO

**Self-Distillation with No Labels**

Instead of explicitly repelling negative pairs:

\[ \\boxed{\\text{student learns from moving-average teacher}} \]

The teacher is a stable moving target.

This is related to the same general idea as **Mean Teacher**.

---

# 28\. DINO vs SimCLR ⭐⭐

| SimCLR | DINO |
| --- | --- |
| Contrastive | Self-distillation |
| Positive + negative pairs | No explicit negative-pair repulsion |
| InfoNCE | Teacher/student objective |
| Needs negative comparisons | Uses moving-average teacher |
| Augmentation-based views | Multiple views + teacher/student |
| Contrastive representation learning | Distillation-based SSL |

### DINO advantages highlighted in lecture

- More memory efficient.
- Particularly effective with **Vision Transformers (ViTs)**.
- Can produce surprisingly strong emergent visual features.

---

# 29\. DINO and zero-shot segmentation

A striking result:

> DINO + ViT self-attention can produce **zero-shot segmentation-like behaviour without labels**.

Meaning:

\[ \\boxed{\\text{self-supervised training + no labels} \\rightarrow \\text{semantic visual structure}} \]

This demonstrates that self-supervised representations can contain surprisingly rich information.

---

# 30\. Cross-modal contrast: CLIP ⭐⭐⭐

Contrastive learning does not have to operate within one modality.

**CLIP = Contrastive Language-Image Pre-Training**

Instead of:

\[ \\text{two augmented views of same image} \]

CLIP uses:

\[ \\boxed{\\text{image + caption}} \]

These are two views of the **same concept**.

```
Image ─────→ image encoder ──→ image representation
                                      ↕
                                 contrastive
                                      ↕
Caption ───→ text encoder ────→ text representation
```

### Key difference

SimCLR:

\[ \\boxed{\\text{same image, different augmentations}} \]

CLIP:

\[ \\boxed{\\text{same concept, different modalities}} \]

---

# 31\. Why CLIP is powerful

The internet contains huge amounts of naturally paired:

\[ \\boxed{\\text{image + text}} \]

No manual class labels are required.

Contrastive training aligns:

\[ \\text{visual concepts} \\leftrightarrow \\text{language concepts} \]

This enables powerful **zero-shot image reasoning/classification**.

For example:

```
Image
 ↓
compare with text:
"a dog"
"a cat"
"a car"
 ↓
highest image-text similarity
 ↓
prediction
```

---

# 32\. Big historical picture

The lecture's broad argument:

### Image models

\[ \\boxed{\\text{supervised pre-training}} \]

produced a huge leap in performance.

### Language + multimodal models

\[ \\boxed{\\text{self-supervised pre-training}} \]

and especially contrastive/self-supervised objectives enabled enormous advances.

---

# 33\. One unified mental model

The entire lecture can be understood as progressively solving the **collapse + task-design problem**.

### Stage 1 — Consistency

```
same input under perturbations
            ↓
       agree
```

Problem:

\[ \\boxed{\\text{collapse}} \]

---

### Stage 2 — Pretext tasks

```
create artificial task
       ↓
learn useful features
```

Problem:

\[ \\boxed{\\text{shortcut / spurious signals}} \]

---

### Stage 3 — Contrastive learning

```
positive pairs → attract
negative pairs → repel
```

Problem of trivial collapse is addressed through negative examples.

---

### Stage 4 — SimCLR

```
same image
   ↓
different augmentations
   ↓
positive pair

different images
   ↓
negative pairs
```

Use InfoNCE.

---

### Stage 5 — DINO / MoCo

Move beyond explicit negative-pair contrast:

```
stable moving target
       ↓
self-distillation
```

---

### Stage 6 — CLIP

Generalise contrast beyond one modality:

```
image ↔ text
```

---

# 34\. High-yield comparison table

| Concept | **ONE thing to remember** |
| --- | --- |
| **SSL** | Generate supervision from unlabelled data |
| **Pretext task** | Artificial task used to learn representations |
| **Downstream task** | Actual task of interest |
| **Consistency** | Same input under perturbation → same output |
| **Consistency alone** | Can collapse |
| **Spurious signal** | Shortcut solving task without intended features |
| **Augmentation** | Defines useful invariances |
| **Cutout** | Erase region → diverse features |
| **MixUp** | Mix images + labels |
| **CutMix** | Paste patch + area-weight labels |
| **AutoAugment** | RL searches augmentation policy |
| **RandAugment** | Random (N) transforms with magnitude (M) |
| **SimCLR** | Different views of same image should agree |
| **Positive pair** | Two views of same underlying sample |
| **Negative pair** | Different samples |
| **Mode collapse** | Everything maps to same representation |
| **InfoNCE** | Cross-entropy over similarity softmax |
| **Temperature****(****\\tau****)** | Controls sharpness of similarity distribution |
| **(****z****)** | Projection vector used for contrastive loss |
| **(****h****)** | Rich encoder representation used downstream |
| **MoCo** | Momentum/stable encoder |
| **DINO** | Self-distillation with moving-average teacher |
| **DINO advantage** | Efficient + strong ViT representations |
| **CLIP** | Contrast image and text representations |
| **CLIP positive pair** | Matching image-caption |
| **Zero-shot** | Use learned representation without task-specific labels |

---

# 35\. Must-memorise equations

### Consistency

\[ \\boxed{ L=CE\[y,f(x)\]+KL\[f(x),f(x+\\eta)\] } \]

### Collapse

\[ \\boxed{ f(x)=c\\quad\\forall x } \]

Any constant output can satisfy consistency.

### SimCLR / InfoNCE

\[ \\boxed{ L_i= -\\log \\frac{ e^{sim(z_i,z_j)/\\tau} }{ \\sum_k e^{sim(z_i,z_k)/\\tau} } } \]

### MixUp

\[ \\boxed{x'=\\lambda x_1+(1-\\lambda)x_2} \]

\[ \\boxed{y'=\\lambda y_1+(1-\\lambda)y_2} \]

### CutMix

\[ \\boxed{ y'=\\lambda y_A+(1-\\lambda)y_B } \]

where (\\lambda) corresponds to the image-area contribution.

---

# 36\. Likely exam questions + compressed answers

### Q1. Why is consistency regularisation alone insufficient?

> Because a constant/collapsed representation gives identical predictions for every input and therefore satisfies consistency without learning useful information.

### Q2. What is a pretext task?

> An automatically supervised artificial task designed to force learning of transferable representations.

### Q3. What makes a good pretext task?

> It should be difficult to solve using shortcuts and should require learning structure useful for downstream tasks.

### Q4. What is a spurious signal?

> An unintended statistical cue that allows the model to solve the pretext task without learning the intended semantics.

### Q5. Explain SimCLR.

> Generate two augmented views of each image, treat them as a positive pair, treat representations from other images as negatives, and use InfoNCE to attract the positive and repel negatives.

### Q6. Why are negative pairs needed?

> Agreement between positive pairs alone permits mode collapse, where all samples map to the same representation.

### Q7. What is InfoNCE?

> Cross-entropy applied to a softmax over similarity scores, with the positive pair as the correct class.

### Q8. What does temperature do?

> It scales similarity logits and controls the sharpness/difficulty of the contrastive classification problem.

### Q9. What is the difference between (h) and (z) in SimCLR?

> (z) is the projection representation used by the contrastive loss; (h) is the richer encoder representation used for downstream tasks.

### Q10. Why are augmentations important in SimCLR?

> They define which variations the representation should become invariant to. Crop/resize promotes spatial-scale invariance; colour distortion encourages reliance on shape/texture rather than exact colour.

### Q11. How does DINO differ from SimCLR?

> DINO uses self-distillation with a moving-average teacher instead of explicitly relying on negative-pair repulsion.

### Q12. What is CLIP?

> A multimodal contrastive model that aligns image and text representations using naturally paired image-caption data.

### Q13. SimCLR vs CLIP?

> SimCLR uses two augmented views of the same image; CLIP uses two modalities—an image and its matching caption—as views of the same concept.

---

# 37\. 60-second final cram sheet

```
SSL
│
├── Generate supervision from unlabelled data
│
├── Pretext task
│     └── learn transferable representation
│
├── Danger 1: trivial collapse
│     └── consistency alone → everything identical
│
├── Danger 2: shortcut learning
│     └── spurious signal solves pretext task
│
├── Augmentation
│     └── defines invariances
│
└── Contrastive learning
      │
      ├── positive = same underlying data
      ├── negative = different data
      │
      └── SimCLR
            ├── augmented views
            ├── InfoNCE
            ├── temperature τ
            ├── z → contrastive loss
            └── h → downstream task

Beyond SimCLR:
MoCo → momentum/stable encoder
DINO → moving-average teacher + self-distillation
CLIP → image ↔ text contrast

Key equation:

L_i = -log[
 exp(sim(z_i,z_j)/τ)
 /
 Σ_k exp(sim(z_i,z_k)/τ)
]

InfoNCE = cross-entropy over similarity softmax.
```

## The single most important conceptual chain

> **SSL creates artificial supervision → pretext tasks learn representations → bad pretext tasks permit shortcuts → consistency alone can collapse → contrastive learning prevents collapse using positives + negatives → SimCLR formalises this with InfoNCE → DINO removes explicit negative-pair dependence via self-distillation → CLIP generalises contrastive learning across image and language.**

That is the **Lecture 3 story** to reconstruct under exam pressure.
