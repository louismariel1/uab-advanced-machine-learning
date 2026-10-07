# Practical 3 — Exam Compression: Self-Supervised Learning

## 1\. Core problem

**Self-supervised learning (SSL)** learns useful representations from **unlabelled data** by creating an artificial **pretext task** whose labels can be generated automatically.

Core pipeline:

```
Unlabelled images
       ↓
Create artificial pretext task
       ↓
Pre-train CNN backbone
       ↓
Discard pretext head
       ↓
Attach target classification head
       ↓
Fine-tune on small labelled dataset
       ↓
Compare against baseline
```

### Key exam idea

> **Self-supervised learning creates supervision from the data itself, rather than requiring human-provided labels.**

---

# 2\. Data setup

The practical uses **two datasets/tasks**.

### Source / pre-training data

**CIFAR-10**

- 50,000 training images.
- Original labels are deliberately ignored/destroyed.
- Model initially sees only images.
- Used for self-supervised pre-training.

Conceptually:

```
CIFAR-10 image → no human label
```

### Target / downstream data

A small labelled subset of **CIFAR-100**.

- Select one superclass.
- Each superclass contains 5 classes.
- Target task = **5-class classification**.
- Labels are available.
- Used for final fine-tuning/evaluation.

Example:

```
vehicles
├── bicycle
├── bus
├── motorcycle
├── pickup truck
└── train
```

### Important distinction

```
CIFAR-10 → unlabelled → self-supervised pre-training
CIFAR-100 → labelled   → downstream classification
```

The source and target classes do **not** need to overlap.

---

# 3\. What is the baseline?

The baseline answers:

> **How well can we solve the target task without self-supervised pre-training?**

Pipeline:

```
Small labelled CIFAR-100 subset
             ↓
        CNN from scratch
             ↓
       5-class classifier
             ↓
       validation accuracy
```

The entire model is randomly initialised and trained directly on the target data.

Record:

```
baseline_acc = baseline_metrics.best_val_acc
```

### Why is the baseline important?

Because the final question is not:

> "Did the pretext model learn?"

It is:

> **"Did self-supervised pre-training improve downstream classification compared with training from scratch?"**

---

# 4\. Backbone vs head — HIGH-YIELD

The model consists of two conceptual parts.

## Backbone

The **backbone** extracts reusable visual features.

```
image
  ↓
ConvBackbone
  ↓
64-dimensional feature vector
```

The backbone learns things such as:

- edges
- textures
- shapes
- object parts
- spatial structure

## Head

The **head** converts features into predictions for a particular task.

```
64 features
    ↓
projection
    ↓
class logits
```

### Key principle

> **Backbone = reusable representation; head = task-specific prediction.**

Therefore:

```
Pretext head → discard
Backbone     → keep
Target head  → create
```

---

# 5\. What is a pretext task?

A **pretext task** is an artificial prediction problem created from unlabelled data.

Example:

```
Original image
      ↓
rotate image
      ↓
predict rotation
```

The model must predict:

```
0°   → class 0
90°  → class 1
180° → class 2
270° → class 3
```

The labels are generated automatically.

No human annotation is required.

### Definition to memorise

> **A pretext task is an automatically constructed task used to train a model on unlabelled data so that it learns useful representations for a downstream task.**

---

# 6\. Recommended pretext task: rotation prediction

The simplest practical implementation is **four-way rotation prediction**.

For each unlabelled image:

```
randomly choose:
0°, 90°, 180°, 270°
```

Then create:

```
(rotated_image, rotation_label)
```

Example:

```
rotation = 90°
label = 1
```

The pretext model becomes:

```
CIFAR-10 image
      ↓
random rotation
      ↓
CNN backbone
      ↓
4-class rotation head
      ↓
0 / 90 / 180 / 270
```

---

# 7\. Why does rotation prediction help?

To predict the rotation correctly, the model needs to learn useful visual information such as:

- orientation
- shapes
- edges
- object structure
- spatial relationships

The goal is therefore not really to build a good rotation classifier.

The goal is:

> **Use the rotation task to force the backbone to learn useful representations.**

Those representations can then transfer to the CIFAR-100 task.

---

# 8\. Why is this self-supervised?

Normally:

```
image → human label
```

With self-supervision:

```
image
  ↓
apply transformation
  ↓
transformation itself provides label
```

For example:

```
image → rotate 90° → label = 90°
```

The human never provided the rotation label.

Therefore:

> **The supervision is generated automatically from the input data.**

---

# 9\. Pretext DataLoader

The original unlabelled loader gives something like:

```
image
```

But the pretext model requires:

```
image, pretext_label
```

Therefore you need a new dataset/collate function/loader that performs:

```
unlabelled image
       ↓
choose transformation
       ↓
apply transformation
       ↓
generate corresponding label
       ↓
return (transformed_image, label)
```

For rotation:

```
0°   → label 0
90°  → label 1
180° → label 2
270° → label 3
```

### Exam point

> The DataLoader converts unlabelled examples into supervised-looking `(x, y)` pairs using automatically generated pretext labels.

---

# 10\. Pretext model

For rotation prediction:

```
pretext_head = ClassifierHead(
    input_size=64,
    projection_size=32,
    num_classes=4
)
```

Then:

```
pretext_model = nn.Sequential(
    pretext_backbone,
    pretext_head
)
```

Architecture:

```
Image
  ↓
ConvBackbone
  ↓
64 features
  ↓
ClassifierHead
  ↓
4 logits
```

Why 4 outputs?

Because there are four possible rotations:

```
0°, 90°, 180°, 270°
```

---

# 11\. Pretext training

Once artificial labels exist, the pretext task is just a normal classification problem.

### Loss

```
nn.CrossEntropyLoss()
```

Objective:

\[ \\boxed{ L_{pretext}=CE(y_{rotation},f_\\theta(x_{rotated})) } \]

Training:

```
rotated image
      ↓
    model
      ↓
rotation prediction
      ↓
CrossEntropyLoss
      ↓
backpropagation
```

---

# 12\. What is actually being learned?

After pre-training:

```
Random backbone
       ↓
Pretext training
       ↓
Useful visual representation
```

The important object is:

```
pretext_backbone
```

not the rotation classifier.

The backbone may now contain filters/features useful for:

- edges
- textures
- shapes
- spatial structure
- object parts

---

# 13\. Transfer to the target task

After pre-training:

```
pretext_model
├── backbone
└── rotation head
```

Keep:

```
backbone
```

Discard:

```
rotation head
```

Then create a new target head:

```
target_head = ClassifierHead(
    input_size=64,
    projection_size=32,
    num_classes=5
)
```

Combine:

```
finetune_model = nn.Sequential(
    pretext_backbone,
    target_head
)
```

Now:

```
CIFAR-100 image
      ↓
pre-trained backbone
      ↓
64 features
      ↓
new target head
      ↓
5 class predictions
```

---

# 14\. Why replace the head?

The pretext task and target task are different.

### Pretext head

Answers:

> "Which rotation was applied?"

Therefore:

```
4 outputs
```

### Target head

Answers:

> "Which CIFAR-100 class is this?"

Therefore:

```
5 outputs
```

So:

\[ \\boxed{\\text{pretext head} \\neq \\text{target head}} \]

The head is task-specific, while the backbone is reusable.

---

# 15\. Fine-tuning

The assignment asks you to **fine-tune** the transferred model.

Fine-tuning means:

> Start from pretrained weights and continue training on the labelled target task.

Typically both can update:

```
pretrained backbone ← updated
new target head     ← updated
```

So:

```
Pretrained representation
          ↓
Target labelled data
          ↓
Fine-tuning
          ↓
Adapt representation to target classes
```

### Freezing is different

If:

```
param.requires_grad = False
```

for the backbone, only the new head learns.

That is **feature extraction/frozen transfer**, not full fine-tuning.

---

# 16\. Why fine-tuning helps

Training from scratch:

```
random features
      ↓
small labelled dataset
      ↓
learn everything from limited data
```

Self-supervised transfer:

```
useful pretrained features
      ↓
small labelled dataset
      ↓
adapt features to target task
```

Therefore the model does not have to learn all visual features from scratch using a small amount of labelled data.

---

# 17\. Complete pipeline — MUST KNOW

```
                 SELF-SUPERVISED LEARNING

                UNLABELLED CIFAR-10
                       │
                       ▼
              Create pretext labels
                       │
                       ▼
                Rotation task
              0 / 90 / 180 / 270
                       │
                       ▼
              ┌─────────────────┐
              │    Backbone     │
              │       ↓         │
              │  Rotation Head  │
              └────────┬────────┘
                       │
                  pre-training
                       │
                       ▼
              PRETRAINED BACKBONE
                       │
                discard head
                       │
                       ▼
              attach new target head
                       │
                       ▼
             LABELLED CIFAR-100
                       │
                   fine-tune
                       │
                       ▼
                FINAL MODEL
                       │
                       ▼
              validation accuracy
                       │
                       ▼
             compare with baseline
```

---

# 18\. Pretext accuracy vs downstream accuracy

This is an important conceptual trap.

A high pretext accuracy does **not automatically mean** the representation is useful.

For example:

```
Pretext accuracy = 99%
```

does not prove:

```
excellent downstream features
```

The actual objective is:

\[ \\boxed{ \\text{Does SSL improve target validation accuracy?} } \]

Therefore compare:

\[ \\boxed{ Acc_{SSL} - Acc_{baseline} } \]

### Example

```
Baseline:       42%
SSL + FT:       49%

Improvement:    +7 percentage points
```

That is evidence that the learned representation transferred successfully.

---

# 19\. Augmentation — important trap

Augmentations should make the pretext task useful without destroying its label.

For example, if the task is rotation prediction:

```
original
   ↓
rotate 90°
   ↓
label = 90°
```

Do **not** then randomly rotate it again.

Otherwise:

```
intended: 90°
extra augmentation: +180°
actual image: 270°
```

but the label still says:

```
90°
```

This creates a contradictory training example.

### General rule

> **Augmentations must preserve the information needed to solve the pretext task.**

---

# 20\. Pretext task quality

A good pretext task should:

- be solvable from the image;
- require useful visual reasoning;
- produce transferable representations;
- avoid relying on trivial shortcuts;
- not be so difficult that learning fails.

### Important insight

The best pretext task is **not necessarily the one with the highest pretext accuracy**.

What matters is:

\[ \\boxed{\\text{quality of transferred representation}} \]

measured by downstream performance.

---

# 21\. Baseline vs self-supervised model

The required comparison is:

```
                 Target CIFAR-100
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   Train from scratch       Self-supervised transfer
          │                         │
          ▼                         ▼
    baseline accuracy       SSL + fine-tuning accuracy
```

The baseline should be as fair as possible.

If the SSL model uses a particular augmentation/training setup, be careful when comparing it against a baseline with completely different training conditions.

### Main metric

> **Best target validation accuracy.**

---

# 22\. Practical implementation issue

The notebook contains:

```
v2.RandomResizedCrop(size=img_dim, scale=(0.8, 1.0))
```

while the variable used earlier may be:

```
img_size
```

If `img_dim` is not defined, this produces:

```
NameError
```

Likely correction:

```
v2.RandomResizedCrop(
    size=img_size,
    scale=(0.8, 1.0)
)
```

Also check the expected input type/range when using `torchvision.transforms.v2`.

---

# 23\. Choosing the target category

The target is one CIFAR-100 superclass containing five classes.

Possible categories include:

- vehicles
- flowers
- fish
- aquatic mammals
- food containers
- fruit and vegetables
- electrical devices
- furniture
- insects
- large carnivores
- large buildings
- natural scenes
- large herbivores
- medium mammals
- invertebrates
- people
- reptiles
- small mammals
- trees
- weird vehicles

For example:

```
TARGET_CATEGORY = "vehicles"
```

gives:

```
bicycle
bus
motorcycle
pickup truck
train
```

---

# 24\. What the provided notebook already handles

Usually the notebook already provides:

- datasets;
- train/validation splitting;
- preprocessing;
- augmentation infrastructure;
- DataLoaders;
- CNN backbone;
- classifier head;
- supervised training loop;
- evaluation;
- accuracy calculation;
- metric tracking;
- plotting;
- baseline training.

### Main implementation responsibility

You primarily need to:

1. design the pretext task;
2. generate artificial labels;
3. create the pretext DataLoader;
4. construct the pretext model;
5. pre-train the backbone;
6. transfer the backbone;
7. replace the pretext head;
8. fine-tune on the target task;
9. compare against baseline.

---

# 25\. Common exam traps

### Trap 1 — Using CIFAR-10 labels

❌

> Train the pretext model using the original CIFAR-10 class labels.

That defeats the purpose of the self-supervised task.

✅

> Ignore human labels and generate labels from the transformation.

---

### Trap 2 — Keeping the pretext head

❌

```
pretrained backbone + rotation head
       ↓
CIFAR-100 classification
```

The rotation head predicts 4 rotation classes.

✅

```
pretrained backbone
       ↓
new 5-class target head
```

---

### Trap 3 — Thinking SSL means no labels anywhere

Not exactly.

The **pre-training stage** uses no human labels.

The **downstream fine-tuning stage** uses labelled target data.

```
Pre-training → unlabelled
Fine-tuning  → labelled
```

---

### Trap 4 — Freezing automatically means fine-tuning

❌

Freezing the backbone means the backbone does not adapt.

✅

Fine-tuning generally allows pretrained parameters to continue updating.

---

### Trap 5 — High pretext accuracy proves SSL worked

False.

The final test is downstream performance.

\[ \\boxed{ Acc_{SSL+FT} \> Acc_{baseline} } \]

is the relevant evidence.

---

### Trap 6 — Corrupting the pretext label with augmentation

If the task is rotation prediction, an additional random rotation can change the correct label.

Therefore the augmentation pipeline must preserve the pretext-task semantics.

---

### Trap 7 — Source and target classes must be identical

False.

The purpose is to learn **general representations**, not memorise source class identities.

---

# 26\. Self-supervised learning vs Practical 1 transfer learning

This distinction is highly useful for the exam.

### Practical 1 — Transfer learning

```
Labelled source data
       ↓
supervised pre-training
       ↓
transfer backbone
       ↓
target fine-tuning
```

### Practical 3 — Self-supervised learning

```
Unlabelled source data
       ↓
create artificial pretext labels
       ↓
self-supervised pre-training
       ↓
transfer backbone
       ↓
target fine-tuning
```

### Key difference

> **Transfer learning can use labelled source supervision; self-supervised learning creates its own supervision from unlabelled data.**

---

# 27\. Self-supervised learning vs Practical 2

This distinction is also important.

### Practical 2 — Semi-supervised learning

Uses:

```
small labelled dataset
+
large unlabelled dataset
```

**during the same target training problem.**

The unlabelled data contributes through consistency regularisation:

\[ L=L_{sup}+\\lambda_uL\_{unsup} \]

### Practical 3 — Self-supervised learning

Uses unlabelled source data to learn a representation first:

```
unlabelled source
       ↓
pretext task
       ↓
pretrained backbone
       ↓
labelled target task
```

### One-line distinction

> **P2 uses unlabelled target-domain data as an additional training signal; P3 uses unlabelled data to pre-train a reusable representation through an artificial task.**

---

# 28\. Likely exam questions

1. **What is self-supervised learning?**
2. What is a **pretext task**?
3. Explain how rotation prediction can create labels without human annotation.
4. Why can a rotation-prediction task learn useful visual representations?
5. Explain the difference between a **backbone and a head**.
6. Why is the pretext head discarded?
7. Why is a new target head required?
8. What happens if you keep the 4-class rotation head for the 5-class target task?
9. Explain the complete SSL pipeline.
10. Why can source and target classes be different?
11. Why is downstream accuracy more important than pretext accuracy?
12. Explain the difference between **self-supervised learning and transfer learning**.
13. Explain the difference between **self-supervised and semi-supervised learning**.
14. Explain **fine-tuning vs freezing**.
15. Design a pretext task for an unlabelled image dataset.
16. Explain why an augmentation can be harmful to a pretext task.
17. Given baseline and SSL validation accuracies, calculate:

\[ \\boxed{ \\Delta Acc=Acc_{SSL}-Acc_{baseline} } \]

18. Given a network with a 4-class pretext head, modify it for a 5-class target task.

---

# 29\. Essential equations

### Pretext classification

\[ \\boxed{ L_{pretext}=CE(y_{pretext},f_\\theta(x_{transformed})) } \]

### Downstream supervised learning

\[ \\boxed{ L_{target}=CE(y_{target},f\_\\theta(x)) } \]

### SSL improvement

\[ \\boxed{ \\Delta Acc=Acc_{SSL+FT}-Acc_{baseline} } \]

### Core representation-learning idea

\[ \\boxed{ \\text{unlabelled data} \\rightarrow \\text{pretext task} \\rightarrow \\text{representation} \\rightarrow \\text{downstream task} } \]

---

# 30\. One-minute revision summary

If you only remember a few things:

### Self-supervised learning

> **Create supervision automatically from unlabelled data.**

### Pretext task

> **Artificial task used to force the model to learn useful representations.**

### Rotation example

```
image
 ↓
0/90/180/270° rotation
 ↓
automatic label
 ↓
4-class prediction
```

### Backbone

> **Learns reusable visual features.**

### Pretext head

> **Solves the artificial pretext task.**

### Target head

> **Solves the actual downstream classification task.**

### Transfer

```
pretext backbone
      ↓
discard pretext head
      ↓
attach target head
      ↓
fine-tune
```

### Final evaluation

> **Compare SSL + fine-tuning against training from scratch.**

Not:

> "Did the pretext classifier achieve high accuracy?"

---

# 31\. Final conceptual picture

The whole practical can be remembered as:

\[ \\boxed{ \\text{Unlabelled images} \\rightarrow \\text{Artificial labels} \\rightarrow \\text{Pretext training} \\rightarrow \\text{Useful backbone} \\rightarrow \\text{New target head} \\rightarrow \\text{Fine-tuning} \\rightarrow \\text{Better target accuracy} } \]

Or, in one sentence:

> **Practical 3 = use unlabelled CIFAR-10 images to create a pretext task such as rotation prediction, train a CNN backbone to learn reusable visual representations, discard the pretext head, attach a new 5-class CIFAR-100 head, fine-tune on the small labelled target dataset, and compare its validation accuracy with a from-scratch baseline.**

## 32\. Absolute must-memorise

1. **SSL creates supervision from unlabelled data.**
2. **Pretext task = artificial task used to learn representations.**
3. **Rotation prediction is a simple example: 0°, 90°, 180°, 270°.**
4. **Backbone learns reusable features; head is task-specific.**
5. **Pretext head is discarded after pre-training.**
6. **A new target head is attached.**
7. **Fine-tuning normally updates the pretrained backbone and new head.**
8. **Pretext accuracy is not the final objective — downstream target accuracy is.**
9. **Source and target classes can be different.**
10. **Do not use augmentations that destroy the information needed to solve the pretext task.**
11. **P1 = supervised source pre-training → transfer.**
12. **P2 = labelled + unlabelled target data → consistency regularisation.**
13. **P3 = unlabelled source data → artificial pretext task → reusable representation → labelled target fine-tuning.**

### Likely exam model answer

> **Self-supervised learning learns representations from unlabelled data by constructing an artificial pretext task whose labels are generated automatically. For example, an image can be randomly rotated by 0°, 90°, 180° or 270°, with the applied rotation used as the label. A CNN is trained to solve this task, causing its backbone to learn useful visual features. After pre-training, the task-specific rotation head is discarded and replaced with a new classification head for the downstream target task. The pretrained backbone is then fine-tuned using the small labelled target dataset. The effectiveness of the self-supervised representation is evaluated by comparing downstream validation accuracy against a model trained from scratch.**
