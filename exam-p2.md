# Practical 2 — Final Exam Compression

This merges **Chunk 1 + Chunk 2** into one exam-focused revision sheet, with the Chunk 2 additions on **confidence filtering, hyperparameter sweeps, early stopping, experimentation, and assessment/report requirements**.

---

## 1\. Core problem: Semi-supervised learning

**Semi-supervised learning (SSL)** uses:

- A small **labelled** dataset.
- A large **unlabelled** dataset.
- Both typically come from the same target domain.

Goal:

> Use the unlabelled data to improve performance beyond supervised training on the labelled subset alone.

### Transfer learning vs SSL

| Transfer learning | Semi-supervised learning |
| --- | --- |
| Transfers knowledge from another source task/domain | Exploits unlabelled target-domain examples |
| Usually starts from a pretrained model | Uses labelled + unlabelled target data |
| Source data is generally labelled | Unlabelled data has no ground-truth labels |

**Exam phrase:**

> Transfer learning transfers knowledge from a source domain/task, whereas SSL exploits additional unlabelled examples from the target domain.

---

# 2\. Practical setup

Dataset: **CIFAR-10**

- 10 classes.
- 90% of training labels removed → label `-1`.
- 10% remains labelled.
- Validation data remains labelled.

Three important datasets:

- **TrainL** → labelled training examples.
- **TrainU** → unlabelled training examples.
- **Validation** → labelled evaluation data.

### Baselines

**Lower baseline:**

> Train only using TrainL.

This tells us how well ordinary supervised learning performs with limited labels.

**Upper reference:**

> Train using all available labels.

This shows the approximate performance achievable if labels were available.

### Why baselines matter

The purpose of SSL is not merely to train a model.

It should ideally:

\[ \\boxed{\\text{SSL performance} \> \\text{partial-label baseline}} \]

---

# 3\. Consistency regularisation — THE CENTRAL IDEA

Main assumption:

> A small semantic-preserving transformation should not change the model's prediction.

Therefore:

\[ \\boxed{f(x)\\approx f(\\operatorname{augment}(x))} \]

For labelled data, use ordinary supervised learning:

\[ L\_{\\text{sup}}=CE(y,f(x)) \]

For unlabelled data, enforce prediction consistency:

\[ L\_{\\text{unsup}} = KL(P\\parallel Q) \]

Total loss:

\[ \\boxed{ L_{\\text{total}} = L_{\\text{sup}} + \\lambda_u L_{\\text{unsup}} } \]

where:

- (L\_{\\text{sup}}) = supervised cross-entropy.
- (L\_{\\text{unsup}}) = consistency/KL loss.
- (\\lambda\_u) = strength of unlabelled signal.

---

# 4\. UDA — Unsupervised Data Augmentation

**UDA = Unsupervised Data Augmentation.**

For labelled data:

```
labelled x,y
     ↓
   model
     ↓
 prediction
     ↓
 CE(prediction,y)
```

For unlabelled data:

```
             unlabelled x
                 │
            ┌────┴────┐
            ↓         ↓
         clean      augmented
            ↓         ↓
          model     model
            ↓         ↓
       prediction P  prediction Q
            │         │
            └────┬────┘
                 ↓
             KL(P || Q)
```

Final objective:

\[ \\boxed{L=CE+\\lambda\_u KL} \]

---

# 5\. Data augmentation

The augmentation must alter appearance while preserving the **class/semantics**.

Examples:

### Geometric

- RandomHorizontalFlip
- RandomCrop
- RandomResizedCrop
- RandomRotation

### Photometric

- ColorJitter
- Brightness changes
- Contrast changes
- Channel changes

Practical combination:

\[ \\boxed{\\text{RandomHorizontalFlip + ColorJitter}} \]

### Avoid excessive transformations

CIFAR-10 is low resolution, so excessive:

- blur
- pixel noise
- destructive transformations

can make images unrecognisable.

**Core principle:**

> Augmentation should create a different view of the same class, not a different semantic object.

---

# 6\. CRITICAL: augmentation order

`torchvision.transforms.v2` augmentation expects approximately `[0,1]` image values.

Therefore:

\[ \\boxed{ \\text{Image} \\rightarrow \\text{Augmentation} \\rightarrow \\text{Normalisation} } \]

NOT:

\[ \\text{Image} \\rightarrow \\text{Normalisation} \\rightarrow \\text{Augmentation} \]

For TrainU, the loader provides:

\[ \\boxed{(x_{\\text{clean}},x_{\\text{aug}})} \]

---

# 7\. KL divergence — HIGH-YIELD EXAM POINT

This is one of the most important implementation details.

Let:

- (P) = clean prediction = **pseudo-label**.
- (Q) = augmented prediction = prediction being trained toward (P).

We want:

\[ \\boxed{KL(P\\parallel Q)} \]

### Clean branch

```
clean_probs = F.softmax(
    clean_logits.detach(),
    dim=1
)
```

So:

\[ P=\\operatorname{softmax}(\\text{clean logits}) \]

### Augmented branch

```
aug_log_probs = F.log_softmax(
    aug_logits,
    dim=1
)
```

So:

\[ \\log Q=\\operatorname{logsoftmax}(\\text{aug logits}) \]

Then:

```
F.kl_div(
    aug_log_probs,
    clean_probs,
    reduction='batchmean'
)
```

Mathematically:

\[ \\boxed{ F.kl\_div(\\log Q,P)=KL(P\\parallel Q) } \]

### Memorise this

> **PyTorch****`kl_div(input, target)`****takes****`log(Q)`****first and****`P`****second.**

---

# 8\. Why `.detach()`?

The clean prediction is being used as the pseudo-label.

We want:

\[ P=\\text{fixed target} \]

rather than allowing the model to modify both sides of the KL objective.

Correct:

```
clean_probs = F.softmax(
    clean_logits.detach(),
    dim=1
)
```

Without detach:

> Gradients can flow through the pseudo-label branch, making the target unstable.

**Exam answer:**

> Detach the clean prediction so that gradients do not flow through the pseudo-label branch.

---

# 9\. Soft pseudo-labels

The clean prediction is a **soft pseudo-label**.

Hard label:

\[ \[0,0,1,0,\\ldots\] \]

Soft pseudo-label:

\[ \[0.02,0.05,0.80,0.10,\\ldots\] \]

The soft version contains uncertainty information.

The model therefore learns:

\[ \\boxed{ f(x_{\\text{clean}}) \\approx f(x_{\\text{aug}}) } \]

---

# 10\. `cross_entropy` vs `kl_div`

Important difference:

### Cross-entropy

Normally:

```
F.cross_entropy(logits, target)
```

uses raw logits.

### KL divergence

Use:

```
F.kl_div(
    log_probs,
    probs,
    reduction='batchmean'
)
```

Therefore:

\[ \\boxed{ P=\\operatorname{softmax}(\\text{clean logits}) } \]

\[ \\boxed{ \\log Q=\\operatorname{logsoftmax}(\\text{aug logits}) } \]

Then:

\[ KL(P\\parallel Q) \]

---

# 11\. Why `batchmean`?

Use:

```
reduction='batchmean'
```

because it gives a sensible batch-level KL loss and avoids unwanted dependence of the loss scale on the number of classes/elements.

**Exam keyword:**

> `batchmean` gives an appropriate batch-averaged KL divergence.

---

# 12\. Complete UDA training algorithm

For every labelled batch:

### 1\. Get labelled data

\[ (x,y) \]

### 2\. Get unlabelled pair

\[ (x_u,x_{u,\\text{aug}}) \]

### 3\. Supervised prediction

\[ z=f(x) \]

\[ L\_{\\text{sup}}=CE(z,y) \]

### 4\. Unlabelled predictions

\[ z_u=f(x_u) \]

\[ z_{u,\\text{aug}}=f(x_{u,\\text{aug}}) \]

### 5\. Create pseudo-label

\[ P=\\operatorname{softmax}(z\_u^{\\text{detach}}) \]

### 6\. Calculate consistency loss

\[ Q=\\operatorname{softmax}(z\_{u,\\text{aug}}) \]

\[ L\_{\\text{unsup}}=KL(P\\parallel Q) \]

### 7\. Combine

\[ \\boxed{ L_{\\text{total}} = L_{\\text{sup}} + \\lambda_uL_{\\text{unsup}} } \]

### 8\. Optimisation

```
zero_grad()
↓
backward()
↓
optimizer.step()
```

### 9\. Validation

Evaluate on the **labelled validation set**.

---

# 13\. Why `itertools.cycle(trainU_loader)`?

TrainL and TrainU may contain different numbers of batches.

Example:

```
TrainL = 8 batches
TrainU = 72 batches
```

If you use `zip()`, iteration ends when the shorter loader ends.

Instead:

```
trainU_iter = iter(itertools.cycle(trainU_loader))
```

then:

```
for batch in trainL_loader:
    ux_clean, ux_aug = next(trainU_iter)
```

This means:

> One unlabelled batch is obtained for every labelled batch, with TrainU recycled when necessary.

---

# 14\. Warm-up

Early in training:

> Model predictions are poor → pseudo-labels are unreliable.

Immediately applying a strong consistency loss can reinforce incorrect predictions.

Therefore use **warm-up**.

During warm-up:

\[ \\boxed{\\lambda\_u=0} \]

After warm-up:

\[ \\boxed{\\lambda\_u\>0} \]

Conceptually:

```
poor model
   ↓
supervised training only
   ↓
better predictions
   ↓
more reliable pseudo-labels
   ↓
activate consistency loss
```

---

# 15\. Ramp-up

Instead of suddenly switching:

\[ 0\\rightarrow\\lambda\_u \]

you can gradually increase:

\[ \\lambda\_u(t)\\uparrow \]

This reduces the effect of noisy pseudo-labels at the beginning.

**Key idea:**

> Trust the unsupervised signal more as the model becomes better.

---

# 16\. Confidence thresholding

Chunk 2 adds an important extension.

Instead of using every unlabelled example, only use examples where the clean prediction is sufficiently confident.

For example:

\[ \\boxed{\\text{confidence}\\geq0.8} \]

Code:

```
clean_probs = F.softmax(
    clean_logits.detach(),
    dim=1
)

confidence = clean_probs.max(dim=1).values

mask = confidence >= 0.8
```

Then:

```
unsupervised_loss = unsup_loss_fn(
    clean_logits[mask],
    aug_logits[mask]
)
```

### Why?

Low-confidence pseudo-label:

> Probably unreliable.

High-confidence pseudo-label:

> More likely to represent the correct class.

Therefore:

\[ \\boxed{ \\text{confidence filtering} \\Rightarrow \\text{reduce noisy pseudo-labels} } \]

### If no examples pass the threshold

Use:

```
unsupervised_loss = torch.tensor(
    0.0,
    device=device
)
```

This avoids attempting to calculate a loss on an empty selection.

---

# 17\. Standard UDA vs confidence-filtered UDA

### Standard UDA

Every unlabelled example contributes:

\[ L\_{\\text{unsup}} = KL(P\\parallel Q) \]

### Confidence UDA

Only high-confidence examples contribute:

\[ \\boxed{ L_{\\text{unsup}} = KL(P_{\\text{confident}}\\parallel Q\_{\\text{confident}}) } \]

where:

\[ P(\\max P\_i\\geq0.8) \]

---

# 18\. Hyperparameters tested in Chunk 2

The practical experiments sweep:

| (\\lambda\_u) | warm-up |
| --- | --- |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 3 | 3 |
| 2 | 5 |
| 3 | 5 |

And compare two conditions:

### Run 1

```
USE_CONFIDENCE_THRESHOLD = False
```

→ Standard UDA.

### Run 2

```
USE_CONFIDENCE_THRESHOLD = True
CONFIDENCE_THRESHOLD = 0.8
```

→ UDA + confidence filtering.

---

# 19\. VERY IMPORTANT: fresh model per experiment

Each experiment must start with:

```
experiment_model = BasicCNN(num_classes=10)
```

Why?

Because otherwise one experiment could inherit learned weights from a previous experiment.

That makes the comparison unfair.

**Exam principle:**

> Hyperparameter experiments should start from equivalent initial conditions.

---

# 20\. Experimental variables

Main hyperparameters:

### (\\lambda\_u)

Controls how strongly the unlabelled consistency loss affects training.

\[ L=L_{\\text{sup}}+\\lambda_uL\_{\\text{unsup}} \]

- Too small → unlabelled data has little influence.
- Too large → noisy pseudo-labels may dominate.

### `warmup_epochs`

Controls how long training uses only supervised learning before activating consistency regularisation.

### `confidence_threshold`

Controls how selective pseudo-labeling is.

Higher threshold:

- fewer examples used;
- potentially more reliable pseudo-labels.

Lower threshold:

- more examples used;
- potentially noisier pseudo-labels.

---

# 21\. Early stopping

Training can stop if validation accuracy stops improving.

Given:

```
stopping_patience = 8
```

the code checks recent validation performance.

Concept:

```
train
 ↓
validation accuracy improves
 ↓
continue
 ↓
validation accuracy stops improving
 ↓
early stopping
```

### Why?

- Prevent overfitting.
- Save computation.
- Select the best validation performance.

---

# 22\. Important bug/trap: what is being compared?

The practical tracks:

```
metrics.best_val_acc
```

So the reported performance is the **best validation accuracy achieved**, not necessarily the accuracy of the final epoch.

Therefore:

\[ \\boxed{\\text{best validation accuracy} \\neq \\text{necessarily final accuracy}} \]

---

# 23\. Model architecture — know the essentials

Basic CNN:

```
RGB input
   ↓
Conv 3 → 32
   ↓
MaxPool
   ↓
Conv 32 → 64
   ↓
MaxPool
   ↓
Conv 64 → 64
   ↓
Conv 64 → 64
   ↓
FC 4096 → 256
   ↓
Dropout
   ↓
FC 256 → 128
   ↓
Dropout
   ↓
10-class classifier
```

Important point:

> Dropout helps reduce overfitting because only a small fraction of labels are available.

---

# 24\. Training configuration

Typical practical settings:

- Optimiser: **AdamW**
- Learning rate: (10^{-3})
- Weight decay/L2: (10^{-3})
- Maximum epochs: 40
- Early stopping: enabled
- Supervised loss: cross-entropy
- Unsupervised loss: KL consistency
- (\\lambda\_u): experimentally tuned
- Warm-up: experimentally tuned

---

# 25\. Why larger unlabelled batches?

The unsupervised signal is noisier than the supervised signal.

Larger unlabelled batches can help:

> Average/smooth the noisy consistency signal.

Therefore TrainU can reasonably use a larger batch size than TrainL.

---

# 26\. What happens if UDA doesn't improve?

The practical guidance highlights three major checks:

### 1\. Detach pseudo-label

```
clean_logits.detach()
```

### 2\. Correct KL direction

Want:

\[ KL(P\\parallel Q) \]

implemented as:

```
F.kl_div(
    log_Q,
    P,
    reduction='batchmean'
)
```

### 3\. Tune (\\lambda\_u)

The optimum may be higher than expected.

Also consider:

- warm-up;
- confidence filtering;
- augmentation strength.

---

# 27\. Exam traps — MUST KNOW

## Trap 1: Wrong augmentation order

❌

```
normalisation → augmentation
```

✅

```
augmentation → normalisation
```

---

## Trap 2: Wrong KL direction

Want:

\[ KL(P\\parallel Q) \]

where:

- (P) = clean pseudo-label.
- (Q) = augmented prediction.

Correct:

```
F.kl_div(
    F.log_softmax(aug_logits, dim=1),
    F.softmax(clean_logits.detach(), dim=1),
    reduction='batchmean'
)
```

---

## Trap 3: Forgetting `.detach()`

❌

```
F.softmax(clean_logits, dim=1)
```

✅

```
F.softmax(clean_logits.detach(), dim=1)
```

---

## Trap 4: Using ground-truth labels for TrainU

TrainU has **no ground-truth labels**.

Its training signal comes from:

\[ \\boxed{\\text{prediction consistency}} \]

---

## Trap 5: Applying consistency loss too strongly too early

Bad pseudo-labels can reinforce errors.

Solutions:

- warm-up;
- ramp-up;
- confidence thresholding.

---

## Trap 6: Reusing the same model between experiments

❌ Train experiment A → continue training same model for experiment B.

✅ Create a fresh model for every experiment.

---

## Trap 7: Confidence mask with zero selected samples

If:

```
mask.any() == False
```

there are no confident examples.

Set:

```
unsupervised_loss = torch.tensor(
    0.0,
    device=device
)
```

---

# 28\. Core comparison to memorise

```
PARTIAL SUPERVISED
       │
       └── TrainL only
              ↓
             CE

UDA
       │
       ├── TrainL → CE
       │
       └── TrainU
             │
             ├── clean → P
             │          ↓
             │        detach
             │
             └── augmented → Q
                            ↓
                         KL(P||Q)

       ↓

L = CE + λu KL
```

Confidence UDA adds:

```
P
↓
confidence
↓
confidence ≥ 0.8 ?
↓
YES → use KL
NO  → ignore example
```

---

# 29\. Practical 2 — experimental logic

The entire practical can be understood as:

```
1. Train using only labelled data
             ↓
       baseline accuracy
             ↓
2. Add UDA consistency
             ↓
   tune λu / warm-up
             ↓
3. Compare against baseline
             ↓
4. Optionally add confidence filtering
             ↓
5. Sweep hyperparameters
             ↓
6. Use fresh model for every experiment
             ↓
7. Record best validation accuracy
             ↓
8. Report improvement over baseline
```

The key comparison is:

\[ \\boxed{ \\text{UDA accuracy} - \\text{partial-label baseline accuracy} } \]

---

# 30\. Expected result / success criterion

The practical states that a correctly implemented consistency objective should be capable of achieving approximately:

\[ \\boxed{\\geq 5%} \]

improvement in validation accuracy over the baseline.

So, conceptually:

\[ \\boxed{ \\text{Best UDA validation accuracy}

>

\\text{partial-label baseline} } \]

and ideally by at least about 5 percentage points.

---

# 31\. Assessment requirements

You need to submit **both**:

### Text report

A brief report, preferably PDF, containing:

- Baseline performance with supervised training on partial labels.
- Plot(s) showing the augmentation applied to unlabelled data.
- Semi-supervised/UDA performance using unlabelled data.
- A few sentences explaining:
- your solution;
- problems encountered;
- how you solved them;
- insights about consistency regularisation.

One page is sufficient.

### Notebook

Submit the completed `.ipynb` containing the implementation.

**Important:**

> Both the report and notebook must be submitted for credit.

---

# 32\. One-page mental model

```
                 SEMI-SUPERVISED LEARNING
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
         LABELLED DATA             UNLABELLED DATA
              │                         │
              ↓                    ┌────┴────┐
        Cross-entropy            clean     augmented
              │                    │           │
              │                    ↓           ↓
              │                   f(x)       f(xaug)
              │                    │           │
              │                 pseudo P       Q
              │                    │
              │                 detach
              │                    │
              │                    └────┬──────┘
              │                         ↓
              │                     KL(P || Q)
              │                         │
              └────────────┬────────────┘
                           ↓
                 L = CE + λu KL
                           │
                    validation set
                           ↓
                  compare with baseline
                           │
                  ┌────────┴────────┐
                  ↓                 ↓
              UDA works       UDA insufficient
                  │                 │
                  ↓                 ↓
             tune λu          warm-up / ramp-up
                              confidence ≥ 0.8
```

---

# 33\. Absolute must-memorise — final 12

If you only have a few minutes before the exam, memorise these:

1. **SSL = labelled + unlabelled data from the target domain.**
2. **UDA = Unsupervised Data Augmentation.**
3. Consistency assumption:

\[ \\boxed{f(x)\\approx f(\\operatorname{augment}(x))} \]

4. Total loss:

\[ \\boxed{L=CE+\\lambda\_u KL} \]

5. Clean prediction = **soft pseudo-label**.
6. **Detach the clean pseudo-label.**
7. Want:

\[ \\boxed{KL(P\\parallel Q)} \]

8. PyTorch implementation:

```
F.kl_div(
    F.log_softmax(aug_logits, dim=1),
    F.softmax(clean_logits.detach(), dim=1),
    reduction='batchmean'
)
```

9. **Augment before normalisation.**
10. Early pseudo-labels are unreliable → use **warm-up/ramp-up**.
11. Confidence filtering:

\[ \\boxed{\\max(P)\\geq0.8} \]

keeps only sufficiently confident examples.

12. Every hyperparameter experiment should use a **fresh model**, and compare **best validation accuracy against the partial-label baseline**.

---

# 34\. Likely exam questions + model answers

### Q1. How does consistency regularisation use unlabelled data?

> Two semantic-preserving versions of an unlabelled example are passed through the model. The prediction from the clean version is treated as a detached soft pseudo-label, and the model minimises the KL divergence between this prediction and the prediction for the augmented version. This consistency loss is combined with the supervised cross-entropy loss from labelled data, allowing unlabelled examples to provide a training signal without ground-truth labels.

### Q2. Why detach the clean prediction?

> The clean prediction is being used as a pseudo-label, so it should be treated as a fixed target. Detaching prevents gradients from flowing through the pseudo-label branch and prevents the model from changing the target to reduce the loss.

### Q3. Why is KL divergence written with apparently reversed arguments in PyTorch?

> Mathematically we want (KL(P\\parallel Q)), where (P) is the clean pseudo-label and (Q) is the augmented prediction. PyTorch's `F.kl_div` expects log-probabilities as its first argument and probabilities as its target, so we use `F.kl_div(log_Q, P)`. Thus the apparent argument order is due to PyTorch's API convention.

### Q4. Why use confidence thresholding?

> Pseudo-labels from low-confidence predictions are more likely to be incorrect. Confidence thresholding only applies the consistency loss to sufficiently confident predictions, reducing the influence of noisy pseudo-labels.

### Q5. Why use warm-up?

> At the beginning of training the model's predictions are poor, so its pseudo-labels are unreliable. Warm-up first trains using labelled data, allowing the model to learn useful representations before introducing the unsupervised consistency loss.

### Q6. Why does the practical compare against a partial-label baseline?

> The purpose of semi-supervised learning is to exploit unlabelled data to improve performance when labels are scarce. The partial-label supervised model provides the appropriate baseline, so improvement can be attributed to exploiting the unlabelled data.

### Q7. Why must every experiment use a fresh model?

> To ensure a fair comparison. If a model trained in one experiment is reused in another, the second experiment inherits information from the first, so the hyperparameter comparison is no longer controlled.

---

## Final 30-second recall

> **Small labelled set → CE. Large unlabelled set → clean + augmented views. Clean prediction → detach → soft pseudo-label. Augmented prediction → match pseudo-label using KL. Total = CE + λu KL. Augment before normalisation. Warm-up because early pseudo-labels are bad. Confidence thresholding can remove unreliable pseudo-labels. Sweep λu/warm-up, use a fresh model each time, and compare best validation accuracy with the partial-label baseline.**

That is the **core of Practical 2**.
