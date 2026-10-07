# Practice 2 — Exam Compression (Chunk 1)

## 1\. Core setting: Semi-supervised learning

**Semi-supervised learning (SSL):**

- Small **labelled** dataset + large **unlabelled** dataset.
- Goal: use both to improve performance.

### Transfer learning vs SSL

| Transfer learning | Semi-supervised learning |
| --- | --- |
| Source dataset/task provides knowledge | Unlabelled data provides extra training signal |
| Source may differ in domain/task | Labelled + unlabelled data come from the **same domain** |
| Source is typically labelled | Large dataset has **no labels** |
| E.g. pretrained model → target task | E.g. 10% labelled CIFAR-10 + 90% unlabelled |

**Key exam distinction:**

> Transfer learning transfers knowledge from a source domain/task; SSL exploits unlabelled examples from the target domain.

---

## 2\. Problem setup used in practical

CIFAR-10:

- 10 classes.
- 90% of training labels destroyed → label `-1`.
- Remaining 10% = labelled training set.
- Validation set remains labelled.

Three subsets:

- **TrainL:** labelled training data.
- **TrainU:** unlabelled training data.
- **Validation:** labelled data used for evaluation.

### Baselines

**Lower bound:** train normally using TrainL only.

**Upper-bound reference:** hypothetical training using all labels.

Expected partial-label baseline:

> \~50–55% validation accuracy.

Full-label training gives substantially higher performance (\~75% in this practical).

### Why establish baselines?

You need to show that consistency regularisation actually improves over:

> **supervised training on labelled data alone.**

---

# 3\. Consistency Regularisation — central idea

The key assumption:

> **Small semantic-preserving changes to an input should not change the model's prediction.**

For an unlabelled image (x):

\[ f(x) \\approx f(\\text{augment}(x)) \]

So instead of needing the true label, use the model's prediction on one version as a **pseudo-label** for another version.

### Training uses two signals

**Labelled data:**

\[ L\_{\\text{sup}} = CE(y,f(x)) \]

**Unlabelled data:**

\[ L\_{\\text{unsup}} = KL(f(x)\\parallel f(x+\\eta)) \]

Total:

\[ \\boxed{ L=L_{\\text{sup}}+\\lambda_u L\_{\\text{unsup}} } \]

where (\\lambda\_u) controls the strength of the unsupervised signal.

---

# 4\. UDA — Unsupervised Data Augmentation

UDA = **Unsupervised Data Augmentation**.

Core procedure:

```
             labelled x,y
                  │
                  ▼
              model f
                  │
              prediction
                  │
                  ▼
             CE(y, f(x))
```

and simultaneously:

```
       unlabelled x
          │
       ┌──┴──┐
       ▼     ▼
   clean x  augmented x
       │       │
       ▼       ▼
     f(x)    f(x_aug)
       │       │
       └───┬───┘
           ▼
       KL divergence
```

Then:

\[ \\boxed{ L=CE+\\lambda\_u KL } \]

---

# 5\. Data augmentation

The augmentation must:

> **Change the appearance while preserving the semantic class.**

Recommended combination:

### Geometric transformation

Examples:

- `RandomHorizontalFlip`
- `RandomCrop`
- `RandomResizedCrop`
- `RandomRotation`

### Photometric transformation

Examples:

- `ColorJitter`
- brightness/contrast changes
- channel permutation

The practical implementation uses:

\[ \\boxed{\\text{RandomHorizontalFlip + ColorJitter}} \]

### Avoid

For CIFAR-10, avoid excessive:

- blur
- pixel-level noise

because images are already very low-resolution and can become unrecognisable.

---

# 6\. Critical implementation detail: augmentation order

The dataloader initially normalises images.

But `torchvision.transforms.v2` expects image values approximately in:

\[ \[0,1\] \]

Therefore:

\[ \\boxed{ \\text{Image} \\rightarrow \\text{Augmentation} \\rightarrow \\text{Normalisation} } \]

**NOT**

\[ \\text{Image} \\rightarrow \\text{Normalisation} \\rightarrow \\text{Augmentation} \]

The augmented loader returns:

\[ \\boxed{(x_{\\text{clean}},x_{\\text{aug}})} \]

for each unlabelled batch.

---

# 7\. KL divergence implementation — HIGH-YIELD

This is one of the most likely practical/exam traps.

Suppose:

- clean prediction = pseudo-label (P)
- augmented prediction = model prediction (Q)

We want:

\[ \\boxed{KL(P\\parallel Q)} \]

### Correct conceptual direction

The **clean prediction is the target**:

\[ P = \\operatorname{softmax}(\\text{clean logits}) \]

The augmented prediction is what we optimise:

\[ Q = \\operatorname{softmax}(\\text{aug logits}) \]

Therefore:

```
clean logits
     ↓
softmax
     ↓
P = pseudo-label
     ↓
detach()
```

and:

```
aug logits
     ↓
log_softmax
     ↓
log Q
```

Then:

```
F.kl_div(log_Q, P, reduction='batchmean')
```

### PyTorch quirk

Mathematically:

\[ KL(P\\parallel Q) \]

but PyTorch's:

```
F.kl_div(input, target)
```

expects:

\[ input=\\log Q,\\qquad target=P \]

Therefore:

\[ \\boxed{ F.kl\_div(\\log Q,P) = KL(P\\parallel Q) } \]

**Do not accidentally reverse the arguments.**

---

# 8\. Why `.detach()` is essential

Clean prediction is being used as a **pseudo-label**.

Therefore:

```
clean_probs = F.softmax(clean_logits.detach(), dim=1)
```

### Why detach?

Without `.detach()`:

\[ KL(P\\parallel Q) \]

could update **both** (P) and (Q).

That means the model could change its pseudo-label as well as its prediction.

This makes the training target unstable.

With detach:

\[ \\boxed{ P=\\text{fixed pseudo-label} } \]

while (Q) is pushed toward (P).

### Exam wording

> **Detach the clean prediction so gradients do not flow through the pseudo-label branch.**

---

# 9\. `cross_entropy` vs `kl_div`

KL divergence and cross-entropy have closely related optimisation behaviour.

You can potentially use:

```
F.cross_entropy(...)
```

instead of:

```
F.kl_div(...)
```

but their input conventions differ.

### `F.cross_entropy`

Expects:

- input = **raw logits**
- target = probabilities/labels depending on usage

### `F.kl_div`

Expects:

- input = **log-probabilities**
- target = **probabilities**

So with KL:

\[ \\boxed{ \\text{clean logits} \\xrightarrow{\\text{softmax}} P } \]

\[ \\boxed{ \\text{aug logits} \\xrightarrow{\\text{log-softmax}} \\log Q } \]

---

# 10\. Why `batchmean`?

Use:

```
reduction='batchmean'
```

for KL divergence.

Reason:

> Avoid unwanted dependence of the loss magnitude on batch size.

---

# 11\. Pseudo-label interpretation

The clean prediction is treated as a **soft pseudo-label**.

Unlike a hard label such as:

\[ \[0,0,1,0,\\ldots\] \]

a soft prediction might be:

\[ \[0.02,0.05,0.80,0.10,\\ldots\] \]

This contains uncertainty information.

The model is trained so that:

\[ \\boxed{ f(x_{\\text{aug}})\\approx f(x_{\\text{clean}}) } \]

---

# 12\. Full UDA training loop

Each iteration does:

### Step 1 — labelled batch

Get:

\[ (x,y) \]

Compute:

\[ L\_{\\text{sup}}=CE(f(x),y) \]

### Step 2 — unlabelled batch

Get:

\[ (x_u,x_{u,\\text{aug}}) \]

### Step 3 — two predictions

\[ z_u=f(x_u) \]

\[ z_{u,\\text{aug}}=f(x_{u,\\text{aug}}) \]

### Step 4 — consistency loss

\[ L_{\\text{unsup}} = KL( \\operatorname{softmax}(z_u) \\parallel \\operatorname{softmax}(z\_{u,\\text{aug}}) ) \]

with clean branch detached.

### Step 5 — combine

\[ \\boxed{ L_{\\text{total}} = L_{\\text{sup}} + \\lambda_uL_{\\text{unsup}} } \]

### Step 6

```
zero_grad
→ backward
→ optimizer.step
```

### Step 7

Evaluate on labelled validation data.

---

# 13\. Why `itertools.cycle(trainU_loader)`?

The labelled and unlabelled loaders can contain different numbers of batches.

For example:

```
TrainL:  8 batches
TrainU: 72 batches
```

If you simply zip them, the loop stops after the shorter loader.

Using:

```
itertools.cycle(trainU_loader)
```

creates an effectively infinite stream of unlabelled batches.

Then:

```
for batch in trainL_loader:
    ux_clean, ux_aug = next(trainU_iter)
```

means:

> One unlabelled batch is sampled for every labelled batch, recycling the unlabelled loader as necessary.

---

# 14\. Warm-up

Early in training, model predictions are poor.

Therefore pseudo-labels are unreliable.

If you immediately apply a strong consistency loss:

\[ \\lambda_u L_{\\text{unsup}} \]

you may reinforce incorrect predictions.

### Solution: warm-up

Initially:

\[ \\lambda\_u=0 \]

Train using supervised data first.

After (N) warm-up epochs:

\[ \\lambda\_u\>0 \]

Conceptually:

```
Early:
supervised learning
       ↓
model learns useful predictions
       ↓
consistency loss activated
       ↓
better pseudo-labels
```

---

# 15\. Alternative: ramp-up (\\lambda\_u)

Instead of suddenly switching from:

\[ 0\\rightarrow\\lambda\_u \]

you can gradually increase the weight:

\[ \\lambda\_u(t)\\uparrow \]

This reduces the influence of noisy unsupervised predictions early in training.

### Key principle

> **Trust the unsupervised signal more as the model becomes better.**

---

# 16\. Confidence masking

Another way to reduce noisy pseudo-labels:

Only use consistency loss when the clean prediction is sufficiently confident.

For example:

```
mask = (confidence > 0.8)
```

Then calculate the KL loss only for high-confidence examples.

### Principle

\[ \\boxed{ \\text{high confidence} \\Rightarrow \\text{more trustworthy pseudo-label} } \]

Low-confidence predictions are ignored.

---

# 17\. Why larger unlabelled batches?

The unsupervised signal is noisier than the supervised signal.

A larger unlabelled batch can help:

> **average/smooth the noisy consistency signal.**

So it can be useful to use a larger batch size for TrainU than TrainL.

---

# 18\. Model architecture — know only the essentials

Basic CNN:

```
Input RGB
  ↓
Conv 3→32
  ↓
MaxPool
  ↓
Conv 32→64
  ↓
MaxPool
  ↓
Conv 64→64
  ↓
Conv 64→64
  ↓
FC 4096→256
  ↓
Dropout
  ↓
FC 256→128
  ↓
Dropout
  ↓
Classifier → 10 classes
```

Important practical point:

> Dropout is included to reduce overfitting because the labelled training set is tiny.

---

# 19\. Training details worth remembering

Baseline:

- Optimiser: **AdamW**
- Learning rate: (10^{-3})
- L2/weight decay: (10^{-3})
- Up to 40 epochs
- Early stopping
- Cross-entropy loss

UDA:

- Same basic supervised objective
- Add KL consistency loss
- Tune (\\lambda\_u)
- Potentially use warm-up
- Potentially confidence-mask pseudo-labels
- Usually needs more epochs

---

# 20\. Exam traps

### Trap 1 — augmenting after normalisation

❌

\[ \\text{normalise}\\rightarrow\\text{augmentation} \]

✅

\[ \\boxed{\\text{augmentation}\\rightarrow\\text{normalise}} \]

---

### Trap 2 — wrong KL direction

Want:

\[ KL(P\\parallel Q) \]

where:

- (P) = clean pseudo-label
- (Q) = augmented prediction

PyTorch:

```
F.kl_div(
    F.log_softmax(aug_logits, dim=1),
    F.softmax(clean_logits.detach(), dim=1),
    reduction='batchmean'
)
```

---

### Trap 3 — forgetting detach

❌

```
clean_probs = F.softmax(clean_logits, dim=1)
```

✅

```
clean_probs = F.softmax(clean_logits.detach(), dim=1)
```

---

### Trap 4 — treating unlabelled data as if it has labels

The unlabelled images have no ground-truth (y).

Their training signal comes from:

\[ \\boxed{\\text{prediction consistency}} \]

not ordinary CE against true labels.

---

### Trap 5 — blindly trusting pseudo-labels

Early predictions can be bad.

Solutions:

- warm-up
- ramp-up (\\lambda\_u)
- confidence threshold/masking

---

# 21\. One-page mental model

```
SEMI-SUPERVISED LEARNING
│
├── Small labelled set
│      │
│      └── Cross-entropy
│
└── Large unlabelled set
       │
       ├── Clean image ──→ model ──→ pseudo-label P
       │                                  │
       │                               detach
       │                                  │
       └── Augmented image → model ──→ prediction Q
                                          │
                                          ▼
                                     KL(P || Q)
                                          │
                                          ▼
                               unsupervised loss
                                          │
                    ┌─────────────────────┘
                    ▼
        L = L_supervised + λu L_unsupervised
```

## 22\. Absolute must-memorise

If you're short on revision time, memorise these **8 points**:

1. **SSL = labelled + unlabelled data.**
2. **Consistency regularisation:** semantic-preserving input changes should produce similar predictions.
3. **UDA:** clean prediction acts as a soft pseudo-label for an augmented version.
4. \[

\\boxed{L=CE+\\lambda\_u KL} \]

5. Clean branch = **pseudo-label → detach**.
6. For PyTorch:

\[ \\boxed{F.kl\_div(\\log Q,P)} \] implements mathematical (KL(P|Q)).

7. Augment **before normalisation**.
8. Because early pseudo-labels are noisy, use **warm-up/ramp-up/confidence masking**.

### Likely exam question

> **Explain how consistency regularisation allows unlabelled data to contribute to training.**

**Model answer structure:**

> Apply two semantic-preserving views of each unlabelled example: a clean version and an augmented version. Pass both through the model. Treat the clean prediction as a detached soft pseudo-label and minimise the KL divergence between it and the augmented prediction. Combine this consistency loss with the normal cross-entropy loss on labelled examples, weighted by (\\lambda\_u). This encourages the model to learn predictions that are invariant to the chosen augmentation, allowing unlabelled data to provide a training signal without ground-truth labels.
