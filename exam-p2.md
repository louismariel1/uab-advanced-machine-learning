## Practical 2 — Exam Representation: Semi-Supervised Learning / Consistency Regularisation

### 1\. Core problem

Improve classification performance when only a **small fraction of training data is labelled** by exploiting a much larger **unlabelled dataset**.

**Key idea:**

`Small labelled set + large unlabelled set → supervised loss + consistency loss → improved classifier`

Unlike Practical 1:

- **P1:** transfer knowledge from a separate **source dataset**.
- **P2:** exploit **unlabelled data from the target dataset/domain**.

---

### 2\. Data setup

- Dataset: **CIFAR-10**
- 10 classes.
- 90% of training labels are removed → represented as `-1`.
- Remaining 10% = **labelled training data**.
- 90% = **unlabelled training data**.
- Validation set remains fully labelled.

Three sets:

```
Training data
     │
     ├── TrainL → labelled examples
     │
     └── TrainU → unlabelled examples

Validation → labelled → evaluation
```

The key comparison is:

```
TrainL only
   ↓
supervised baseline

TrainL + TrainU
   ↓
semi-supervised UDA
```

---

### 3\. Baseline model

`BasicCNN` is used as the classifier.

Conceptually:

```
Input RGB
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

For CIFAR-10:

```
BasicCNN(num_classes=10)
```

---

### 4\. Baseline training

Train the CNN using **only TrainL**.

Typical settings:

- Loss: **CrossEntropyLoss**
- Optimiser: **AdamW**
- Learning rate: `1e-3`
- Weight decay / L2: `1e-3`
- Up to `40` epochs
- Early stopping
- Metric: **best validation accuracy**

Expected problem:

> Only 10% of the labels are available → supervised baseline is limited.

The baseline provides the reference that UDA must beat.

---

### 5\. Consistency regularisation

Main assumption:

> A semantic-preserving transformation of an image should not change the model's prediction.

Therefore:

\[ \\boxed{f(x)\\approx f(\\text{augment}(x))} \]

For labelled data:

\[ L\_{sup}=CE(y,f(x)) \]

For unlabelled data:

\[ L\_{unsup}=KL(P\\parallel Q) \]

where:

- (P) = prediction for the clean image.
- (Q) = prediction for the augmented image.

Total objective:

\[ \\boxed{ L=L_{sup}+\\lambda_uL\_{unsup} } \]

where (\\lambda\_u) controls the strength of the unlabelled signal.

---

### 6\. UDA — Unsupervised Data Augmentation

The practical implements **UDA**.

For labelled examples:

```
labelled x,y
     ↓
   model
     ↓
 prediction
     ↓
 CrossEntropy(prediction,y)
```

For unlabelled examples:

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

Final:

\[ \\boxed{ L_{total}=CE+\\lambda_uKL } \]

---

### 7\. Data augmentation

The augmentation must change appearance while preserving the class.

Examples:

- `RandomHorizontalFlip`
- `RandomCrop`
- `RandomResizedCrop`
- `RandomRotation`
- `ColorJitter`

Practical combination:

\[ \\boxed{\\text{RandomHorizontalFlip + ColorJitter}} \]

For CIFAR-10, avoid excessive transformations such as strong blur/noise because the images are already low-resolution.

---

### 8\. Critical augmentation order

The correct order is:

\[ \\boxed{ \\text{Image} \\rightarrow \\text{Augmentation} \\rightarrow \\text{Normalisation} } \]

NOT:

\[ \\text{Normalisation} \\rightarrow \\text{Augmentation} \]

Reason:

> `torchvision.transforms.v2` augmentation expects image values approximately in `[0,1]`.

The unlabelled loader provides:

```
(x_clean, x_aug)
```

---

### 9\. KL divergence — HIGH-YIELD

The clean prediction is the **pseudo-label**:

\[ P=\\operatorname{softmax}(\\text{clean logits}) \]

The augmented prediction is:

\[ Q=\\operatorname{softmax}(\\text{augmented logits}) \]

We want:

\[ \\boxed{KL(P\\parallel Q)} \]

PyTorch implementation:

```
F.kl_div(
    F.log_softmax(aug_logits, dim=1),
    F.softmax(clean_logits.detach(), dim=1),
    reduction='batchmean'
)
```

Remember:

> `F.kl_div(input, target)` expects **log Q first, P second**.

Therefore:

\[ \\boxed{ F.kl\_div(\\log Q,P)=KL(P\\parallel Q) } \]

---

### 10\. Why `.detach()`?

The clean prediction is the pseudo-label.

It should act as a fixed target:

```
clean_probs = F.softmax(
    clean_logits.detach(),
    dim=1
)
```

Without `.detach()`:

- gradients can flow through the pseudo-label;
- both predictions can move;
- the training target becomes unstable.

**Exam answer:**

> Detach the clean prediction so gradients do not flow through the pseudo-label branch.

---

### 11\. Soft pseudo-label

The clean prediction is a **soft pseudo-label**, not a hard class.

Example:

\[ \[0.02,0.05,0.80,0.10,\\ldots\] \]

instead of:

\[ \[0,0,1,0,\\ldots\] \]

This retains uncertainty information.

---

### 12\. Full UDA training loop

Each iteration:

```
1. Get labelled (x,y)
        ↓
2. Get unlabelled (x_clean,x_aug)
        ↓
3. Compute supervised prediction
        ↓
4. CE loss
        ↓
5. Compute clean + augmented predictions
        ↓
6. Detach clean prediction
        ↓
7. Calculate KL consistency loss
        ↓
8. L = CE + λu KL
        ↓
9. backward()
        ↓
10. optimizer.step()
        ↓
11. Validate on labelled validation set
```

---

### 13\. Why `itertools.cycle(trainU_loader)`?

TrainL and TrainU can contain different numbers of batches.

For example:

```
TrainL = 8 batches
TrainU = 72 batches
```

Using `zip()` would stop after 8 batches.

Instead:

```
trainU_iter = iter(itertools.cycle(trainU_loader))
```

Then:

```
for batch in trainL_loader:
    ux_clean, ux_aug = next(trainU_iter)
```

This gives approximately **one unlabelled batch per labelled batch**, recycling TrainU when necessary.

---

### 14\. Warm-up

Early predictions are poor, so early pseudo-labels are unreliable.

During warm-up:

\[ \\boxed{\\lambda\_u=0} \]

After warm-up:

\[ \\boxed{\\lambda\_u\>0} \]

Concept:

```
Supervised training
       ↓
model improves
       ↓
pseudo-labels become more reliable
       ↓
activate consistency loss
```

Purpose:

> Prevent the model from reinforcing its own early mistakes.

---

### 15\. Ramp-up

Instead of suddenly changing:

\[ 0\\rightarrow\\lambda\_u \]

the consistency weight can gradually increase:

\[ \\lambda\_u(t)\\uparrow \]

**Principle:**

> Trust the unsupervised signal more as the model becomes more reliable.

---

### 16\. Confidence thresholding

Chunk 2 introduces an optional extension:

> Only use unlabelled examples whose clean prediction is sufficiently confident.

Example:

\[ \\boxed{\\text{confidence}\\geq0.8} \]

Implementation:

```
clean_probs = F.softmax(
    clean_logits.detach(),
    dim=1
)

confidence = clean_probs.max(dim=1).values

mask = confidence >= 0.8
```

Only masked examples contribute to KL:

```
unsupervised_loss = unsup_loss_fn(
    clean_logits[mask],
    aug_logits[mask]
)
```

If no examples pass the threshold:

```
unsupervised_loss = torch.tensor(
    0.0,
    device=device
)
```

Main idea:

\[ \\boxed{ \\text{High confidence} \\Rightarrow \\text{more trustworthy pseudo-label} } \]

---

### 17\. Standard UDA vs confidence UDA

**Standard UDA:**

```
all unlabelled examples
        ↓
consistency loss
```

**Confidence-filtered UDA:**

```
all unlabelled examples
        ↓
confidence check
        ↓
 ┌──────┴──────┐
 ↓             ↓
≥ 0.8         < 0.8
 ↓             ↓
use KL       ignore
```

Confidence filtering trades:

- **fewer examples**
- for **potentially cleaner pseudo-labels**.

---

### 18\. Hyperparameter experimentation

The practical sweeps:

| (\\lambda\_u) | Warm-up |
| --- | --- |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 3 | 3 |
| 2 | 5 |
| 3 | 5 |

Two experiment modes:

```
USE_CONFIDENCE_THRESHOLD = False
```

→ Standard UDA.

and:

```
USE_CONFIDENCE_THRESHOLD = True
CONFIDENCE_THRESHOLD = 0.8
```

→ UDA + confidence filtering.

---

### 19\. Fresh model for every experiment

Every experiment should start with:

```
experiment_model = BasicCNN(num_classes=10)
```

Why?

> Each hyperparameter combination must start from equivalent conditions.

Otherwise, experiment B could inherit knowledge learned during experiment A, making the comparison unfair.

---

### 20\. Important hyperparameters

#### (\\lambda\_u)

Controls the strength of the unsupervised loss:

\[ L=CE+\\lambda\_uKL \]

- Too low → unlabelled data has little influence.
- Too high → noisy pseudo-labels can dominate.

#### `warmup_epochs`

How long to train without the consistency loss.

#### `confidence_threshold`

How confident a prediction must be before being used as a pseudo-label.

Higher threshold:

- fewer examples;
- potentially cleaner labels.

Lower threshold:

- more examples;
- potentially noisier labels.

---

### 21\. Early stopping

The practical can use:

```
early_stopping=True
stopping_patience=8
```

Training stops when validation performance has stopped improving.

Benefits:

- reduces overfitting;
- saves computation;
- selects the best-performing model.

Important:

> Report `best_val_acc`, not necessarily the accuracy at the final epoch.

---

### 22\. Baseline vs UDA comparison

The central experiment is:

```
                 CIFAR-10
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
     10% labelled       90% unlabelled
          │                   │
          ↓                   ↓
       TrainL              TrainU
          │                   │
          ↓                   ↓
       CE loss         consistency/KL
          │                   │
          └─────────┬─────────┘
                    ↓
             UDA model
                    ↓
            validation accuracy
                    ↓
       compare with TrainL baseline
```

Improvement:

\[ \\boxed{ \\Delta Acc = Acc_{UDA} - Acc_{baseline} } \]

---

### 23\. Expected success criterion

The practical states that a correctly implemented consistency objective should achieve approximately:

\[ \\boxed{\\geq5\\text{ percentage-point improvement}} \]

over the partial-label baseline.

So the key question is:

> Did exploiting the unlabelled data improve validation accuracy compared with supervised training on the labelled subset?

---

### 24\. Common problems / debugging

If UDA is not improving:

1. Check that the **clean logits are detached**.
2. Check the **KL direction**.
3. Check that `log_softmax` is used for the prediction being optimised.
4. Check that `softmax` is used for the pseudo-label.
5. Check augmentation occurs **before normalisation**.
6. Tune (\\lambda\_u).
7. Try warm-up.
8. Try confidence thresholding.
9. Make sure every experiment starts with a **fresh model**.

---

### 25\. Likely exam tasks/questions

1. **Explain semi-supervised learning.**
2. Explain how **consistency regularisation** uses unlabelled data.
3. Explain **UDA**.
4. Write the total loss:

\[ L=CE+\\lambda\_uKL \]

5. Explain why the clean prediction must be **detached**.
6. Complete the correct PyTorch `F.kl_div()` implementation.
7. Explain why the KL arguments appear reversed.
8. Explain why augmentation must happen **before normalisation**.
9. Explain why `itertools.cycle(trainU_loader)` is used.
10. Explain **warm-up/ramp-up**.
11. Explain **confidence thresholding**.
12. Given results for different (\\lambda\_u), warm-up and confidence thresholds, identify the best configuration.
13. Calculate improvement over the baseline:

\[ \\Delta Acc=Acc_{UDA}-Acc_{baseline} \]

14. Explain why each experiment should use a **fresh model**.
15. Given training/validation curves, identify **overfitting**.
16. Explain why low-confidence pseudo-labels can harm training.
17. Distinguish:
18. supervised learning;
19. semi-supervised learning;
20. consistency regularisation;
21. pseudo-labeling;
22. UDA.

---

### 26\. Important code pattern to memorise

```
# labelled prediction
pred = model(x)

supervised_loss = loss_func(pred, y)

# unlabelled predictions
clean_logits = model(ux_clean)
aug_logits = model(ux_aug)

# clean = detached pseudo-label
clean_probs = F.softmax(
    clean_logits.detach(),
    dim=1
)

# augmented prediction
aug_log_probs = F.log_softmax(
    aug_logits,
    dim=1
)

# consistency loss
unsupervised_loss = F.kl_div(
    aug_log_probs,
    clean_probs,
    reduction='batchmean'
)

# total
total_loss = (
    supervised_loss
    + lambda_u * unsupervised_loss
)
```

Confidence version:

```
confidence = clean_probs.max(dim=1).values
mask = confidence >= 0.8
```

---

### 27\. One-line memory version

> **Practical 2 = small labelled set + large unlabelled set → train with CE on labelled data + KL consistency between clean and augmented unlabelled predictions → detach clean pseudo-label → tune λu/warm-up → optionally confidence-filter at 0.8 → compare best validation accuracy against the partial-label baseline.**

---

### 28\. Practical 1 vs Practical 2 — essential distinction

|  | Practical 1 | Practical 2 |
| --- | --- | --- |
| Main technique | **Transfer learning** | **Semi-supervised learning** |
| Extra information | Labelled source dataset | Unlabelled target-domain data |
| Main idea | Pretrain → fine-tune | Consistency regularisation |
| Dataset relationship | Source → target | Same target domain |
| Main loss | Cross-entropy | CE + KL |
| Key trick | Replace classifier | Detach pseudo-label |
| Main risk | Overfitting target data | Noisy pseudo-labels |
| Main solution | Pretraining | Warm-up/confidence filtering |
| Key comparison | Scratch vs transfer | Partial-label baseline vs UDA |

---

## Final 30-second recall

> **P2 = SSL. Only 10% of CIFAR-10 labels are available. TrainL uses CE. TrainU has clean + augmented images. Clean prediction becomes a detached soft pseudo-label. Augmented prediction is trained to match it using****(****KL(P****|****Q)****)****. Total loss is****(****CE+\\lambda****_u KL_****_)_****_. Augment before normalisation. Warm-up prevents unreliable early pseudo-labels. Confidence filtering keeps predictions with confidence ≥ 0.8. Tune_****_(_****_\\lambda_****u****)****/warm-up, use a fresh model for each experiment, and compare best validation accuracy with the partial-label baseline.**
