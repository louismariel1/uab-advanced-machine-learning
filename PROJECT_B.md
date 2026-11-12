# Project B — Mitigating Catastrophic Forgetting in Continual Image Classification

## Transfer Learning and Knowledge Distillation in Class-Incremental Learning

**Course:** Advanced Machine Learning  
**Status:** Academic project proposal — pending professor review  
**Methods:** Transfer Learning, Continual Learning, Knowledge Distillation (KD)

## 1. Abstract

This project studies whether knowledge distillation can reduce **catastrophic forgetting** in a transfer-learning-based class-incremental image classifier. A model initialized from a pretrained backbone learns successive groups of image classes. Two controlled pipelines are compared: ordinary sequential fine-tuning and sequential learning with knowledge distillation from a frozen copy of the previous model. Both pipelines use the same task order, pretrained initialization, and architecture. The study measures final average accuracy across all learned tasks, forgetting of earlier tasks, and the ability to acquire new classes. The expected contribution is an empirical analysis of whether distillation improves the stability–plasticity trade-off when earlier training examples are unavailable.

## 2. Research Problem and Motivation

Traditional supervised fine-tuning assumes access to a stationary training dataset. In continual learning, new tasks or classes arrive sequentially. Updating a model using only new-task data can degrade performance on earlier classes, a phenomenon known as **catastrophic forgetting**.

Transfer learning provides an effective initial representation, but does not eliminate forgetting. KD offers a possible mitigation: when a new task arrives, freeze the previous model and use its predictions as soft targets to constrain the updated model's behavior on previously learned classes.

The central research question concerns **the full sequential pipeline**, not merely the independent performance of transfer learning or KD.

## 3. Research Question

> Does incorporating previous-model knowledge distillation into a transfer-learning-based class-incremental learning pipeline improve final classification accuracy across all learned classes and reduce catastrophic forgetting relative to conventional sequential fine-tuning?

## 4. Compared Pipelines

### B1 — Reference pipeline: Transfer Learning → Sequential Fine-Tuning

```text
Pretrained image backbone
      ↓
Task 1: Learn classes 1–5
      ↓
Task 2: Fine-tune on classes 6–10 only
      ↓
Task 3: Fine-tune on classes 11–15 only
      ↓
Task 4: Fine-tune on classes 16–20 only
      ↓
Final model M1: evaluate on all 20 classes
```

### B2 — Proposed pipeline: Transfer Learning → Continual Learning with KD

```text
Same pretrained image backbone
      ↓
Task 1: Learn classes 1–5
      ↓
Freeze copy of previous model as teacher
      ↓
Task 2: Learn classes 6–10 + distill old-class predictions
      ↓
Freeze updated model as next teacher
      ↓
Task 3: Learn classes 11–15 + distill old-class predictions
      ↓
Freeze updated model as next teacher
      ↓
Task 4: Learn classes 16–20 + distill old-class predictions
      ↓
Final model M2: evaluate on all 20 classes
```

**Key point:** KD occurs **during every incremental update**, not just after all tasks. B1 is also a sequential/class-incremental training protocol; its baseline algorithm is naive fine-tuning without an explicit forgetting-mitigation mechanism.

## 5. Hypotheses

- **H1 — Final accuracy (primary):** B2 achieves higher final average accuracy across the four learned tasks than B1.
- **H2 — Forgetting:** B2 exhibits lower average forgetting of previously learned classes.
- **H3 — Plasticity:** B2 maintains useful accuracy on newly introduced classes rather than preserving old knowledge at the expense of learning new classes.
- **H4 — Efficiency (secondary):** Distillation's computational and memory overhead is quantified to assess the cost of reduced forgetting.

These are testable hypotheses, not promised outcomes. Without old-example replay, distillation may not fully prevent forgetting.

## 6. Dataset and Incremental Task Protocol

| Element | Proposed specification |
|---|---|
| Dataset | CIFAR-100 |
| Selected classes | 20 distinct classes, with a documented reproducible selection |
| Task sequence | Four successive tasks of five new classes each |
| Evaluation protocol | **Class-incremental**: predict among all classes seen so far, without a task ID at inference |
| Task 1 | Classes 1–5 (5 cumulative) |
| Task 2 | Classes 6–10 (10 cumulative) |
| Task 3 | Classes 11–15 (15 cumulative) |
| Task 4 | Classes 16–20 (20 cumulative) |
| Old-task training data | Unavailable at later stages in the primary protocol (no replay) |
| Validation/test | Fixed per-class validation split; untouched official test images for final reporting |
| Task order | Fixed and shared within paired runs; vary across replications if feasible |

The expanding classification head adds outputs for new classes. During evaluation, all currently seen classes compete in a **single shared prediction space**.

## 7. Model and Training Design

**Proposed backbone:** ImageNet-pretrained ResNet-18, subject to course approval and resource availability. Both pipelines start from the same weights and use the same expanding classification-head design.

At incremental task `t`:

1. Copy the model trained through task `t−1` and freeze it as the teacher.
2. Expand the current model's classification head for the new classes.
3. Train on labeled images from task `t`.
4. For B2 only, add a distillation loss that preserves the previous model's predictions over previously seen classes.
5. Validate on all seen classes; save the updated model for the next task.

A representative B2 objective is:

\[
\mathcal{L}_{B2} = \mathcal{L}_{CE,new} + \lambda T^2 D_{KL}(p_{old}^{(T)}\parallel p_{current,old}^{(T)})
\]

The distillation probabilities are restricted and normalized consistently over the **old classes**. `T` is the temperature and `λ` the loss weight. Their values will be selected on validation data.

**Important limitation:** In the no-replay setting, old-model predictions are obtained on *current-task images*. They may not adequately represent earlier classes, so KD can reduce—but does not guarantee elimination of—forgetting. This is a meaningful subject of the study.

## 8. Baseline and Evaluation Metrics

**Primary baseline:** B1, transfer initialization followed by naive sequential fine-tuning on each new task without KD.

Let `a[t,k]` be the held-out test accuracy on task `k` after completing task `t`, with all seen classes eligible as predictions.

**Final average accuracy:**

\[
A_{final}=\frac{1}{K}\sum_{k=1}^{K} a[K,k]
\]

**Average forgetting:**

\[
F=\frac{1}{K-1}\sum_{k=1}^{K-1}\left(\max_{t\in\{k,\ldots,K-1\}}a[t,k]-a[K,k]\right)
\]

where `K=4`. This operationalizes the loss of performance on earlier tasks; the final task is excluded because there is no later task in which to forget it.

| Metric | Purpose |
|---|---|
| **Final average accuracy (primary)** | Evaluate performance across all learned tasks |
| **Average forgetting (co-primary or key secondary)** | Quantify degradation of earlier-task knowledge |
| Accuracy after each task | Show learning/forgetting trajectories |
| Accuracy on newest classes | Measure plasticity and acquisition of new knowledge |
| Backward transfer (optional) | Measure how later learning affects earlier tasks |
| Training time and peak GPU memory | Quantify KD overhead |

Accuracy should be reported for the full class-incremental prediction space, not with an oracle task label.

## 9. Experimental Methodology

1. **Prepare data:** Fix 20 classes, their four-task order, training/validation splits, and class mappings.
2. **Initialize models:** Load identical pretrained weights and train the first task under the same conditions.
3. **Run B1:** Sequentially fine-tune on new-task examples only; evaluate all seen classes after each task.
4. **Run B2:** Repeat the identical schedule with previous-model KD at each transition.
5. **Select hyperparameters:** Use validation data to choose learning rates, temperature, and KD weight; never tune using test images.
6. **Replicate:** Use at least three independent seeds, preferably five; vary task orders if compute permits.
7. **Analyze:** Report mean and variability of final average accuracy, forgetting, and newest-task accuracy, plus paired uncertainty intervals.

A useful exploratory analysis is the effect of KD strength on the **stability–plasticity trade-off**. A stronger distillation penalty may preserve old knowledge but inhibit new-class learning.

## 10. Expected Results and Scientific Contribution

The proposed B2 pipeline is expected to reduce forgetting and potentially improve final average accuracy relative to B1, because it uses previous-model predictions to constrain updates. However, this benefit is not guaranteed; the absence of replay may limit retention, and excessive regularization can reduce new-task accuracy.

The academic contribution is a **controlled empirical study of distillation-based forgetting mitigation within a transferred class-incremental classifier**, with explicit analysis of final accuracy, forgetting, plasticity, and computational overhead.

## 11. Threats to Validity and Limitations

- A 20-class, four-task CIFAR-100 benchmark is deliberately compact and may not generalize to large-scale continual learning.
- Task order and class similarity can strongly affect forgetting.
- ImageNet pretraining may improve both pipelines but introduces external pretraining data and assumptions.
- Without old-example replay, teacher predictions on new-task images may be insufficient to preserve old-class performance.
- Expansion of the classification head changes output dimension over time; distillation must be implemented carefully over old classes.
- Final average accuracy alone can conceal poor performance on particular earlier tasks; report per-task results and forgetting.

## 12. Proposed Deliverables

- Reproducible PyTorch class-incremental data pipeline and expanding-head model.
- Reference sequential fine-tuning and distillation-regularized continual learning implementations.
- Per-task accuracy matrices and forgetting curves.
- Final comparative results with multiple seeds and uncertainty analysis.
- Academic report covering the stability–plasticity trade-off, limitations, and resource costs.

## 13. Feasibility and Course Alignment

The project uses the course's Transfer Learning, Continual Learning, and Knowledge Distillation method groups. The KD practical provides relevant loss and teacher–student training concepts, but **class-incremental task management and forgetting evaluation require additional implementation**. Pretrained ResNet-18 use and the precise class-incremental benchmark should be approved against the course rules.

**Approval requested:** Research question, no-replay protocol, pretrained-model policy, four-task benchmark, and acceptable computational scope.

---

**Project status:** Proposal only. No accuracy or forgetting results have yet been measured.
