Absolutely. I can compress **Chunk 2** into exam-focused notes, then merge it with **Chunk 1** into one Practical 2 master sheet. I only have the Chunk 2 material in the current visible context, so I’ll do Chunk 2 now; send/paste Chunk 1 afterward and I’ll merge them.

## Practical 2 — Exam Compression: Chunk 2

:::writing{variant="document" id="58321" title="Practical 2 — Exam Compression: Chunk 2"}

# Practical 2 — Chunk 2: Semi-Supervised Learning / Consistency Regularisation

## 1\. Core idea: Consistency Regularisation

Use both:

- **Labelled data** → standard supervised classification loss.
- **Unlabelled data** → force the model to give consistent predictions for the same image under different augmentations.

For an unlabelled example:

\[ x_{\\text{clean}} \\rightarrow \\text{model} \\rightarrow z_{\\text{clean}} \]

\[ x_{\\text{aug}} \\rightarrow \\text{model} \\rightarrow z_{\\text{aug}} \]

The model should produce similar predictions:

\[ z_{\\text{clean}} \\approx z_{\\text{aug}} \]

Total objective:

\[ \\boxed{ L = L_{\\text{sup}}+\\lambda_u L\_{\\text{unsup}} } \]

where:

- (L\_{\\text{sup}}) = supervised classification loss.
- (L\_{\\text{unsup}}) = consistency loss on unlabelled examples.
- (\\lambda\_u) = strength/weight of the unsupervised objective.

---

## 2\. Supervised loss

For labelled examples:

```
pred = model(x)
supervised_loss = loss_func(pred, y)
```

with:

```
loss_func = nn.CrossEntropyLoss()
```

So:

\[ L\_{\\text{sup}}=\\text{CrossEntropy}(\\text{prediction},\\text{true label}) \]

This is the normal supervised-learning component.

---

## 3\. Unsupervised consistency loss

Obtain a clean and augmented version of the same unlabelled data:

```
ux_clean, ux_aug = next(trainU_iter)
```

Then:

```
clean_logits = model(ux_clean)
aug_logits = model(ux_aug)
```

The consistency loss compares the two predictions:

```
unsupervised_loss = unsup_loss_fn(
    clean_logits,
    aug_logits
)
```

### Important exam point

The clean prediction acts as the target/pseudo-target.

It should be **detached** when used to generate the target:

```
clean_logits.detach()
```

This prevents gradients from flowing through the target branch.

---

## 4\. Why augmentation is important

The model should ideally predict the same class despite small changes to the input.

Example:

```
Original image → CAT
Augmented image → CAT
```

Therefore augmentation creates a useful training signal even when no label is available.

The underlying assumption is:

> A valid augmentation should not change the semantic class.

---

## 5\. Warm-up epochs

The unsupervised loss does not necessarily need to be used immediately.

During warm-up:

```
if e < warmup_epochs:
    current_lambda_u = 0.0
else:
    current_lambda_u = lambda_u
```

Therefore:

\[ L=L\_{\\text{sup}} \]

during warm-up.

After warm-up:

\[ L=L_{\\text{sup}}+\\lambda_uL\_{\\text{unsup}} \]

### Why?

Initially the model is poorly trained, so its predictions on unlabelled data may be unreliable.

Warm-up allows the model to learn useful supervised features first.

---

# 6\. UDA training loop

The important sequence is:

1. Get labelled batch.
2. Get unlabelled clean/augmented pair.
3. Clear gradients.
4. Calculate supervised prediction/loss.
5. Calculate clean and augmented predictions.
6. Calculate consistency loss.
7. Combine losses.
8. Backpropagate.
9. Update parameters.
10. Track metrics.
11. Evaluate on validation data.
12. Optionally apply early stopping.

Core update:

```
total_loss = (
    supervised_loss
    + current_lambda_u * unsupervised_loss
)

total_loss.backward()
opt.step()
```

---

# 7\. UDA implementation details

Model:

```
uda_model = BasicCNN(num_classes=10)
```

Typical settings used:

```
num_epochs = 40
lr = 1e-3
l2_reg = 1e-3
lambda_u = 1.0
warmup_epochs = 3
```

Optimizer:

```
torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=l2_reg
)
```

Validation accuracy is measured on the labelled validation set.

The important comparison is:

\[ \\text{UDA validation accuracy} \\quad \\text{vs} \\quad \\text{partial-label baseline} \]

---

# 8\. Early stopping

Training can stop when validation accuracy has converged.

The code examines recent validation results:

```
recent_best_acc = np.max(
    metrics.val_acc[-stopping_patience:]
)
```

If the recent accuracy does not improve on the best validation accuracy:

```
break
```

Purpose:

- avoid unnecessary training;
- reduce overfitting;
- save computation.

---

# 9\. Confidence threshold experiment

Extension: only use unlabelled examples where the model is sufficiently confident.

First calculate probabilities:

```
clean_probs = F.softmax(
    clean_logits.detach(),
    dim=1
)
```

Then confidence:

```
confidence = clean_probs.max(dim=1).values
```

This gives the probability of the most likely class.

Create a mask:

```
mask = confidence >= confidence_threshold
```

For threshold (0.8):

\[ \\text{use example if confidence} \\geq 0.8 \]

Then calculate consistency loss only on selected examples:

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

### Why confidence filtering?

Low-confidence pseudo-targets may be wrong.

Therefore:

\[ \\boxed{ \\text{Only trust sufficiently confident predictions} } \]

This can make the consistency signal more reliable, although it also reduces the number of unlabelled examples contributing to training.

---

# 10\. Standard UDA vs confidence-filtered UDA

### Standard UDA

Every unlabelled example contributes:

\[ L_{\\text{unsup}} = \\text{Consistency}( z_{\\text{clean}}, z\_{\\text{aug}} ) \]

### Confidence-filtered UDA

Only examples satisfying:

\[ \\max_k p(y=k|x_{\\text{clean}})\\geq0.8 \]

contribute.

### Key trade-off

Higher threshold:

- ✅ more reliable pseudo-targets;
- ❌ fewer examples used.

Lower/no threshold:

- ✅ more unlabelled data used;
- ❌ potentially more noisy/unreliable targets.

---

# 11\. Experimental sweep

The practical tests different values of:

\[ \\lambda\_u \]

and:

\[ \\text{warmup epochs} \]

Experiments:

| (\\lambda\_u) | Warm-up |
| --- | --- |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 3 | 3 |
| 2 | 5 |
| 3 | 5 |

Two runs are performed.

### Run 1

```
USE_CONFIDENCE_THRESHOLD = False
```

This is standard UDA.

### Run 2

```
USE_CONFIDENCE_THRESHOLD = True
CONFIDENCE_THRESHOLD = 0.8
```

This is UDA + confidence filtering.

---

# 12\. Fresh model for every experiment

Very important:

```
experiment_model = BasicCNN(num_classes=10)
```

A completely fresh model is created for each experiment.

### Why?

If experiments reused the same trained model, later experiments would be affected by earlier experiments.

That would make the comparison unfair.

Therefore:

\[ \\boxed{\\text{One fresh model per experiment}} \]

---

# 13\. Measuring improvement

Baseline:

```
partial_label_val_acc
```

Experiment:

```
val_acc = experiment_metrics.best_val_acc
```

Improvement:

```
improvement = val_acc - partial_label_val_acc
```

Mathematically:

\[ \\boxed{ \\text{improvement} = \\text{new accuracy} - \\text{baseline accuracy} } \]

Example:

Baseline = 70%

UDA = 76%

\[ 76%-70%=6% \]

So the improvement is **6 percentage points**.

---

# 14\. What result is expected?

The practical states that a correctly implemented consistency-regularisation objective should achieve approximately:

\[ \\boxed{\\geq 5%} \]

improvement in validation accuracy over the baseline.

If training works but does not beat the baseline, check:

1. Is the unsupervised target detached?
2. Is the KL divergence/consistency-loss direction correct?
3. Is (\\lambda\_u) large enough?
4. Is the warm-up appropriate?
5. Is the augmentation reasonable?

---

# 15\. Most important implementation pitfalls

### Pitfall 1 — Not detaching the target

The target prediction should generally be detached:

```
clean_logits.detach()
```

Otherwise gradients can flow through the target branch.

### Pitfall 2 — Wrong KL-divergence direction

Consistency losses involving KL divergence are sensitive to which prediction is treated as the target/distribution.

Remember:

> Check carefully which branch provides the target and which branch receives the gradient.

### Pitfall 3 — Bad (\\lambda\_u)

If:

\[ \\lambda\_u \\text{ too small} \]

the unlabelled data has little influence.

If:

\[ \\lambda\_u \\text{ too large} \]

the noisy unsupervised objective can dominate supervised learning.

Therefore (\\lambda\_u) should be experimented with.

### Pitfall 4 — Using the same model between experiments

Always initialise a fresh model.

### Pitfall 5 — Confidence threshold with no selected samples

If:

```
mask.any() == False
```

the unsupervised loss must safely become zero.

---

# 16\. Practical assessment requirements

To receive credit, submit **both**:

### Text report

A short report containing:

- baseline supervised performance using partial labels;
- augmentation applied to unlabelled data;
- semi-supervised performance;
- a few plots comparing results;
- brief explanation of the method;
- problems encountered and how they were solved;
- insights about consistency regularisation.

The report can be approximately **one page**.

### Notebook

Submit the completed `.ipynb` notebook containing the implementation.

The assessment is binary:

- completed task + explanation → **1**
- otherwise → **0**

---

# 17\. Exam memory sheet

## Know these equations

### Total loss

\[ \\boxed{ L=L_{\\text{sup}}+\\lambda_uL\_{\\text{unsup}} } \]

### Confidence

\[ \\boxed{ c=\\max\_k p(y=k|x) } \]

### Confidence filtering

\[ \\boxed{ \\text{use example if }c\\geq0.8 } \]

### Improvement

\[ \\boxed{ \\Delta Acc=Acc_{\\text{experiment}}-Acc_{\\text{baseline}} } \]

---

## Know these code concepts

```
pred = model(x)
supervised_loss = loss_func(pred, y)
```

```
clean_logits = model(ux_clean)
aug_logits = model(ux_aug)
```

```
clean_probs = F.softmax(
    clean_logits.detach(), dim=1
)
```

```
confidence = clean_probs.max(dim=1).values
mask = confidence >= 0.8
```

```
total_loss = (
    supervised_loss
    + lambda_u * unsupervised_loss
)
```

```
total_loss.backward()
opt.step()
```

---

# 18\. One-minute exam explanation

**Consistency regularisation is a semi-supervised learning method that uses labelled and unlabelled data. The labelled data produces the normal cross-entropy loss, while the unlabelled data is passed through the model in clean and augmented forms. A consistency loss encourages the two predictions to be similar. The losses are combined as****(****L=L****_{sup}+\\lambda_****uL\_{unsup}****)****. A warm-up period can initially train using only labelled data because early predictions on unlabelled data may be unreliable. Confidence filtering is an extension where only unlabelled examples whose clean prediction has confidence above a threshold such as 0.8 contribute to the consistency loss. Experiments should use a fresh model for each hyperparameter combination and compare validation accuracy against the partial-label baseline.** :::

Send me **Chunk 1** and I’ll turn both into a single **“Practical 2 — Ultimate Exam Compression”** with duplicates removed and the highest-yield exam points consolidated.
