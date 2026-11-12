# Project A — Efficient Image Classification through Self-Supervised Learning and Knowledge Distillation

## A Comparative Study of Dense Fine-Tuning and Low-Rank Adaptation Pipelines

**Course:** Advanced Machine Learning  
**Status:** Academic project proposal — pending professor review  
**Methods:** Self-Supervised Learning (SSL), Low-Rank Adaptation (LoRA; parameter-efficient adaptation), Knowledge Distillation (KD)

## 1. Abstract

This project investigates whether **Low-Rank Adaptation (LoRA)** can replace dense fine-tuning within a multi-stage **Self-Supervised Learning → Teacher Adaptation → Knowledge Distillation** image-classification pipeline without materially reducing the accuracy of the final compact student. Two controlled pipelines share the same SSL-pretrained teacher, labeled target data, student architecture, and distillation procedure. They differ only in how the teacher is adapted: dense fine-tuning versus LoRA. The study will evaluate final student classification accuracy and the adaptation-stage trade-offs in trainable parameters, memory, and training time. The expected contribution is an empirical assessment of whether parameter-efficient teacher adaptation offers a favorable accuracy–efficiency trade-off in an end-to-end learning pipeline.

## 2. Research Problem and Motivation

SSL can extract useful image representations without source labels. For downstream classification, the resulting model must be adapted using target labels. Dense fine-tuning offers flexible adaptation but can require substantial trainable parameters and optimizer memory. LoRA instead learns low-rank updates to selected frozen weights. After adaptation, KD can transfer the teacher's predictions to a smaller deployable student.

The open question is whether the constrained teacher adaptation performed by LoRA affects the predictive knowledge ultimately transferred to the student. **The object of study is the full pipeline**, not merely isolated LoRA parameter counts.

## 3. Research Question

> Can an SSL → LoRA → KD pipeline produce a compact image classifier whose accuracy is non-inferior to that of an SSL → Dense Fine-Tuning → KD pipeline, while reducing the computational resources required for teacher adaptation?

An additional exploratory question is whether LoRA adaptation can sometimes *improve* final student generalization.

## 4. Compared Pipelines

### A1 — Reference pipeline

```text
Unlabeled CIFAR-10
      ↓
Self-Supervised Learning (rotation-prediction pretext task)
      ↓
SSL-pretrained teacher backbone
      ↓
Dense fine-tuning on labeled five-class CIFAR-100 target
      ↓
Adapted teacher A1
      ↓
Knowledge Distillation
      ↓
Compact student S1
```

### A2 — Proposed pipeline

```text
Same unlabeled CIFAR-10 and same SSL teacher checkpoint
      ↓
Low-Rank Adaptation on labeled five-class CIFAR-100 target
      ↓
Adapted teacher A2
      ↓
Same Knowledge Distillation procedure
      ↓
Compact student S2 (same architecture as S1)
```

**Controlled difference:** The teacher adaptation method (dense versus LoRA). SSL initialization and KD protocol are held constant.

## 5. Hypotheses

- **H1 — Accuracy non-inferiority (primary):** The final student S2 is not more than **2 percentage points** worse in test accuracy than S1. The margin is a proposed practical threshold and must be approved before experimentation.
- **H2 — Parameter efficiency:** LoRA reduces the number of trainable parameters during teacher adaptation relative to the dense reference.
- **H3 — Resource efficiency:** LoRA reduces adaptation-stage optimizer memory and potentially peak GPU memory or training time; these outcomes must be measured rather than assumed.
- **H4 — Exploratory accuracy gain:** S2 may outperform S1 in generalization accuracy under some labeled-data regimes, but this is not guaranteed.

For formal non-inferiority, the lower confidence bound of the paired accuracy difference `Accuracy(S2) − Accuracy(S1)` should exceed `−2 percentage points`.

## 6. Data and Model Design

| Element | Proposed specification |
|---|---|
| SSL source | CIFAR-10 training images with labels hidden during SSL |
| Downstream target | Five-class CIFAR-100 subset; proposed superclass: `fish` |
| Target classes | `aquarium_fish`, `flatfish`, `ray`, `shark`, `trout` |
| Target split | Fixed train/validation split of official training data; untouched official test images for final reporting |
| SSL pretext | Predict image rotations: 0°, 90°, 180°, 270° |
| Teacher | `ConvMLP`-style CNN adapted from the LoRA practical |
| Dense adaptation | Freeze early convolutional layers; train selected fully connected layers and new five-class head |
| LoRA adaptation | Freeze original FC weights; add trainable low-rank updates to `fc1`–`fc4`; train five-class head |
| LoRA rank study | `r ∈ {4, 8, 16}`; select rank using validation data |
| Student | Smaller CNN adapted from the KD practical, with five-class output |
| Distillation | Frozen teacher, hard-label cross entropy plus temperature-scaled soft-target loss |

The final architecture, training schedule, and target split will be fixed before final evaluation. A limited-label setting may be included to test whether pretraining and adaptation are especially useful when labels are scarce.

## 7. Methodology

1. **Data preparation:** Create fixed target class mapping and reproducible train/validation splits; reserve the test set.
2. **SSL pretraining:** Train the teacher backbone on unlabeled CIFAR-10 with the rotation-prediction pretext task; save one common checkpoint.
3. **Teacher adaptation:** Branch from that identical checkpoint. Fine-tune one teacher densely (A1) and one using LoRA (A2), with matched target data and comparable optimization budgets.
4. **Knowledge distillation:** Train identically structured students S1 and S2 using their respective frozen teachers, the same labeled data, and the same KD settings.
5. **Evaluation:** Select models using validation data; evaluate once on held-out test data and measure adaptation efficiency.
6. **Replication:** Repeat with at least three seeds, preferably five, and report uncertainty for key differences.

**Distillation objective:**

\[
\mathcal{L} = (1-\lambda)\mathcal{L}_{CE} + \lambda T^2 D_{KL}(p_{teacher}^{(T)}\parallel p_{student}^{(T)})
\]

The teacher is frozen during KD. The temperature `T` and weight `λ` are tuned on validation data and kept identical across A1 and A2. The exact KL orientation and implementation will be documented consistently.

## 8. Baseline and Evaluation Metrics

**Primary baseline:** A1, the complete **SSL → Dense Fine-Tuning → KD** pipeline. A2 is compared directly against it; this study does **not** by itself establish superiority over a supervised-only classifier.

| Metric | Purpose |
|---|---|
| **Final student test accuracy (primary)** | Determine whether S2 retains S1's predictive performance |
| Accuracy difference with confidence interval | Assess non-inferiority and uncertainty |
| Teacher validation/test accuracy | Explain differences in transferred knowledge |
| Trainable parameters during adaptation | Quantify LoRA parameter efficiency |
| Peak GPU memory and optimizer-state size | Measure adaptation resource costs |
| Adaptation training time | Evaluate practical training efficiency |
| Student inference parameters, checkpoint size, latency | Confirm comparable deployment footprint |
| Total pipeline compute | Account for SSL, adaptation, and KD rather than reporting LoRA savings alone |

**Interpretation:** Because S1 and S2 have the same architecture, LoRA does not automatically reduce the deployed student's model size. The efficiency claim concerns the **teacher adaptation stage**. KD supplies the compact student in both pipelines.

## 9. Expected Results and Scientific Contribution

The expected—not guaranteed—outcome is that A2 approaches A1's final student accuracy while training fewer teacher parameters. A2 may or may not save wall-clock time or GPU memory; implementation overhead and hardware characteristics matter. The study will identify whether any accuracy gap is justified by adaptation savings.

The contribution is a **controlled, pipeline-level evaluation** of the effect of dense versus low-rank teacher adaptation on downstream knowledge transfer to a compact student.

## 10. Threats to Validity and Limitations

- CIFAR-10 and the selected CIFAR-100 classes are small-scale benchmarks; results may not generalize to larger images or models.
- Rotation prediction may not learn the most transferable representations for the chosen target classes.
- The LoRA rank and adaptation budget can influence outcomes; hyperparameter selection must not use test data.
- Different random seeds and student initialization can influence KD results.
- Fewer trainable parameters do not imply lower end-to-end compute or a smaller deployed student.
- Pretraining on source images introduces an additional training-data and compute budget common to both pipelines.

## 11. Proposed Deliverables

- Reproducible PyTorch code for SSL, dense adaptation, LoRA, and KD.
- Configuration files, seed management, dataset/split documentation, and experiment scripts.
- Training and validation curves; final test accuracy and confidence intervals.
- Tables comparing trainable parameters, memory, time, and checkpoint sizes.
- Final academic report explaining results, limitations, and reproducibility.

## 12. Feasibility and Course Alignment

This proposal builds directly on three reviewed Advanced Machine Learning practicals: SSL pretext learning, LoRA adaptation, and KD. Their code and architectures require harmonization, particularly the shared teacher and five-class output space, but the main components are already represented in the coursework.

**Approval requested:** Suitability of the research question, two-pipeline design, non-inferiority margin, experimental scope, and expected deliverables.

---

**Project status:** Proposal only. No accuracy, resource-saving, or statistical results have yet been measured.
