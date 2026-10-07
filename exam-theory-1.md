## Lecture 1 — Exam Compression: Transfer Learning Theory

### 1\. Core definition

**Transfer learning:** use knowledge learned from a **source** domain/task to improve learning on a **target** domain/task.

Notation:

- Source: (D_S, T_S)
- Target: (D_T, T_T)
- Goal: improve learning of (f\_T).

At least one differs:

\[ D_S\\neq D_T \\quad \\text{or} \\quad T_S\\neq T_T \]

### 2\. Why transfer learning?

Main benefits:

- **Data efficiency:** less labelled target data needed.
- **Compute efficiency:** avoid expensive training from scratch.
- **Faster convergence:** start from useful representations.
- Particularly useful when the target dataset is small.

Mental model:

> **Don't learn from scratch if useful knowledge already exists.**

---

### 3\. Domain vs task

**Domain** = characteristics/statistical distribution of the data.

**Task** = what the model is trying to predict/solve.

Examples:

- Domain: photographs vs medical scans.
- Task: classification vs segmentation.

This distinction is essential for classifying transfer-learning scenarios.

---

### 4\. Inductive vs transductive transfer

| Type | Domain | Task |
| --- | --- | --- |
| **Inductive** | Same | Different |
| **Transductive** | Different | Same/similar |

\[ \\boxed{\\text{Inductive: }D_S=D_T,;T_S\\neq T_T} \]

\[ \\boxed{\\text{Transductive: }D_S\\neq D_T,;T_S\\approx T_T} \]

**Transductive transfer is closely associated with domain adaptation.**

---

### 5\. Fine-tuning / model-centric transfer

Basic procedure:

```
Source data
    ↓
Pre-trained model
    ↓
Modify output layer if necessary
    ↓
Train on target data
    ↓
Target model
```

Why it works:

- Early neural-network layers often learn **generic features**.
- Later layers tend to become more **task-specific**.

Vision example:

```
Early layers → edges / textures / simple shapes
Later layers → task-specific representations
```

---

### 6\. Freezing layers / feature extraction

Instead of retraining the whole network:

- **Freeze** selected pretrained layers.
- Their weights do not change.
- Train only the remaining layers.

Typical intuition:

\[ \\text{Early layers} \\rightarrow \\text{generic} \]

\[ \\text{Later layers} \\rightarrow \\text{task-specific} \]

The optimal freezing strategy depends on:

- source/target similarity,
- amount of target data,
- task similarity,
- model architecture.

There is **no universally optimal number of frozen layers**.

---

### 7\. Domain shift

Domain shift:

\[ \\boxed{P_S(X)\\neq P_T(X)} \]

The source and target input distributions differ.

Example:

> Model trained on bright images → deployed on dark images.

The model may know the relevant concept, but the input statistics have changed.

---

### 8\. Domain adaptation

**Goal:** compensate for source-target domain differences.

Two major approaches:

```
Domain adaptation
├── Input-space alignment
└── Feature-space alignment
```

---

### 9\. Input-space alignment

Change the **data** so source and target distributions become more compatible.

Important examples:

#### CORAL

**Correlation Alignment**

- Aligns statistical properties of source and target.
- Particularly concerned with first- and second-order statistics.

#### Domain translation

Learn:

\[ g:X_S\\rightarrow X_T \]

Transform source samples into target-like samples.

Can potentially operate without target labels.

#### Generative domain translation

Use generative models for more complex source → target transformations.

---

### 10\. Sim2Real

Transfer from:

\[ \\boxed{\\text{Simulation}\\rightarrow\\text{Reality}} \]

Common in:

- robotics,
- autonomous driving.

Problem:

\[ P_{\\text{simulation}}(X)\\neq P_{\\text{real}}(X) \]

Therefore simulation-trained models may struggle on real data.

---

### 11\. Domain randomisation

Alternative to explicitly matching simulation to reality.

Instead:

> **Make the source domain sufficiently diverse that the target domain is covered.**

Randomise things such as:

- lighting,
- colours,
- textures,
- camera properties,
- object appearance,
- environmental conditions.

Key contrast:

\[ \\text{Domain alignment: Source}\\rightarrow\\text{Target} \]

versus

\[ \\text{Domain randomisation: Large diverse Source}\\supseteq\\text{Target} \]

**Exam takeaway:** don't make the source look exactly like the target; make the source broad enough that the target looks like another possible source example.

---

### 12\. Feature-space alignment

Instead of changing raw inputs, align source and target **internal representations**.

Goal:

\[ \\boxed{\\text{Source features}\\approx\\text{Target features}} \]

This encourages **domain-invariant representations**.

---

### 13\. Domain confusion

Train using:

- normal task/classification loss (L\_{\\text{cls}})
- domain discrepancy loss (L\_{\\text{dom}})

Combined objective:

\[ \\boxed{L=L_{\\text{cls}}+L_{\\text{dom}}} \]

Goal:

> Features should be useful for the task while being less dependent on which domain produced them.

---

### 14\. DANN

**Domain-Adversarial Neural Network**

Instead of directly measuring feature-distribution distance:

- Add a **domain classifier**.
- It tries to determine whether a feature came from source or target.
- The feature extractor is trained to make this difficult.

Desired representation:

```
Useful for task ✓
Identifies source vs target ✗
```

Thus DANN learns **domain-invariant features through adversarial training**.

---

### 15\. Input-space vs feature-space — HIGH PRIORITY

| Method | What changes? | Key idea |
| --- | --- | --- |
| Input-space alignment | Raw data | Transform data |
| CORAL | Data statistics | Align correlations/statistics |
| Domain translation | Raw data | Source → target mapping |
| Domain randomisation | Source distribution | Make source very diverse |
| Feature-space alignment | Internal representations | Align learned features |
| Domain confusion | Feature distributions | Reduce domain discrepancy |
| DANN | Feature representations | Adversarially prevent domain identification |

**Best exam sentence:**

> **Input-space methods change the data; feature-space methods change how the model represents the data.**

---

### 16\. Pre-trained models

In practice, you usually don't pre-train the model yourself.

Workflow:

```
Choose architecture
      ↓
Load pretrained checkpoint
      ↓
Modify required layers
      ↓
Fine-tune on target data
```

Examples:

- Torchvision vision models.
- Hugging Face language models such as BERT.

Important conceptual point:

> **Pre-training can be viewed as an excellent model initialisation.**

Instead of:

\[ \\text{Random weights}\\rightarrow\\text{training} \]

use:

\[ \\boxed{\\text{Pretrained weights}\\rightarrow\\text{fine-tuning}} \]

---

### 17\. Foundation models

Large models trained on enormous datasets can be reused for many downstream tasks.

```
Massive pre-training
        ↓
Foundation model
   ↙    ↓    ↘
Task A Task B Task C
```

Main idea:

> Expensive pre-training is amortised across many downstream applications.

---

### 18\. Multimodal transfer

Different pretrained components can be combined:

\[ \\text{Vision backbone}+\\text{Language backbone} \\rightarrow \\text{Multimodal model} \]

Again, the principle is **reuse learned representations rather than learn everything from scratch**.

---

## 19\. Highest-priority exam knowledge

If you have limited revision time, memorise these:

1. **Transfer learning** = reuse source knowledge for a target task/domain.
2. **Domain** = data distribution/characteristics.
3. **Task** = prediction objective.
4. **Inductive** = same domain, different task.
5. **Transductive** = different domain, same/similar task.
6. **Fine-tuning** = pretrained model → adapt → train on target.
7. **Freezing** = keep selected pretrained weights fixed.
8. **Domain shift**:

\[ P_S(X)\\neq P_T(X) \]

9. **Domain adaptation** = explicitly deal with domain shift.
10. **Input-space alignment** = modify data.
11. **Feature-space alignment** = modify/align representations.
12. **Domain randomisation** = make source distribution broad enough to contain target.
13. **Domain confusion** = classification loss + domain-alignment loss.
14. **DANN** = adversarial domain classifier encourages domain-invariant features.
15. **Pretrained model** = high-quality initialisation.
16. **Foundation model** = massive reusable pretrained model.

---

## 20\. Likely exam questions

Be prepared to:

- Define **transfer learning**, **domain**, and **task**.
- Distinguish **inductive vs transductive transfer**.
- Explain why **fine-tuning** works.
- Explain **freezing layers** and when it might help.
- Identify **domain shift** from a scenario.
- Distinguish **domain adaptation** from ordinary fine-tuning.
- Compare **input-space** and **feature-space** alignment.
- Explain **CORAL**, domain translation, and domain randomisation.
- Explain **sim2real**.
- Explain **domain confusion** and:

\[ L=L_{\\text{cls}}+L_{\\text{dom}} \]

- Explain the basic mechanism of **DANN**.
- Explain why pretrained models are useful.
- Explain the relationship between **pretraining, fine-tuning, and foundation models**.

### One-line master summary

> **Transfer learning reuses knowledge from a source; model-centric transfer uses pretrained representations/fine-tuning, while domain adaptation addresses source-target distribution differences through input-space or feature-space alignment.**
