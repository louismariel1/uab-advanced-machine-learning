Yes. I’ll turn each practical into a **compact exam representation** that preserves what can realistically be assessed, while dropping implementation boilerplate, plotting helpers, download code, and other context-heavy details.

## Practical 1 — Exam Representation: Task Transfer / Transfer Learning

### 1\. Core problem

Improve classification performance on a **small target dataset** by first learning useful representations from a **larger source dataset**, then fine-tuning on the target task.

**Key idea:** `Source pre-training → adapt classifier → target fine-tuning → compare with baseline`

### 2\. Data setup

- **Target:** subset of CIFAR-100.
- Choose one superclass.
- Each superclass contains **5 classes**.
- 32×32 RGB images.
- 400 training + 100 validation examples per class.
- Target task = 5-class classification.
- **Source:** CIFAR-10.
- 10 classes.
- 50,000 training examples.
- Used to learn general visual features.
- Target and source classes are **different**, but the visual information can transfer.

### 3\. Baseline model

`BasicCNN`:

```
Input: 3 × 32 × 32

Conv2d(3 → 32, kernel=3, padding=1)
ReLU
MaxPool(2)

Conv2d(32 → 64, kernel=3, padding=1)
ReLU
MaxPool(2)

Conv2d(64 → 64, kernel=3, padding=1)
ReLU

Flatten: 64 × 8 × 8

Linear(4096 → 256)
ReLU
Linear(256 → 128)
ReLU
Linear(128 → num_classes)
```

For the target baseline:

```
num_classes = 5
```

### 4\. Baseline training

- Loss: **CrossEntropyLoss**
- Optimiser: **Adam**
- Learning rate: `1e-3`
- Weight decay / L2 regularisation: `1e-3`
- Batch size: `128`
- Epochs: `20`
- Objective: maximise **target validation accuracy**.

Expected issue:

- Small target dataset → **overfitting**
- Baseline validation accuracy is relatively poor.

### 5\. Transfer-learning procedure

#### Stage A — Pre-training

Train the same CNN architecture on **CIFAR-10**.

```
BasicCNN(num_classes=10)
        ↓
Train on CIFAR-10
        ↓
Learn general visual representations
```

Important:

- Source classifier has **10 outputs**.
- Save the pretrained model/checkpoint after training.

Example hyperparameters:

- CrossEntropyLoss
- Adam
- `lr = 1e-3`
- `weight_decay = 1e-3`
- \~10 epochs

#### Stage B — Adapt model to target

Load pretrained weights, then replace the source classifier:

```
ft_model.classifier = nn.Linear(128, 5)
```

Why?

The pretrained classifier predicts **10 CIFAR-10 classes**, whereas the target task has **5 different classes**.

The earlier layers retain the learned visual features; the final classifier is changed for the new task.

#### Stage C — Fine-tuning

Train the adapted model on the target dataset.

Example:

- Epochs: `20`
- Learning rate: `1e-4`
- Weight decay: `1e-3`
- CrossEntropyLoss
- Adam

The lower learning rate helps avoid destroying useful pretrained representations.

### 6\. Required comparison

Compare:

```
Target data
    │
    ├── Baseline CNN trained from scratch
    │
    └── CNN pretrained on CIFAR-10
            ↓
       classifier replaced
            ↓
       fine-tuned on target
```

Measure:

- Best validation accuracy of baseline.
- Best validation accuracy after transfer learning.
- Improvement / transfer boost.

Expected result:

- Approximately **5–10 percentage-point improvement** is possible, depending on the selected target superclass.

### 7\. Important implementation concepts

An exam could test:

**Transfer learning**

- Why pre-training can help when target data is scarce.
- Why source and target classes do not need to be identical.
- Why generic visual features learned on CIFAR-10 can help CIFAR-100.

**Classifier replacement**

- Source: `Linear(128, 10)`
- Target: `Linear(128, 5)`
- Why the final layer must change.

**Fine-tuning**

- Load pretrained weights.
- Replace task-specific output layer.
- Train on target data.
- Often use a smaller learning rate.

**Overfitting**

- Small target dataset → high risk.
- Training accuracy can continue increasing while validation accuracy stops improving/decreases.

**Evaluation**

- Validation accuracy is the main metric.
- Compare the **best validation accuracy**, not merely the final epoch.

### 8\. Likely exam tasks/questions

1. **Explain the purpose of pre-training and fine-tuning.**
2. Given a pretrained CIFAR-10 CNN, **modify it for a 5-class target task**.
3. Explain why the final classifier must be replaced.
4. Write/complete the code for:
5. pre-training,
6. loading a checkpoint,
7. replacing the classifier,
8. fine-tuning.
9. Explain why a **smaller learning rate** may be appropriate during fine-tuning.
10. Given training/validation curves, identify **overfitting**.
11. Calculate the transfer-learning improvement:

\[ \\Delta Acc = Acc_{FT} - Acc_{baseline} \]

8. Explain why transfer learning can work even when the source and target classes are different.
9. Distinguish:
10. training from scratch,
11. pre-training,
12. fine-tuning,
13. freezing layers.
14. Explain what would happen if the 10-class CIFAR-10 classifier were **not replaced**.

### 9\. Optional extension concepts

Not graded, but potentially useful conceptually:

- Freeze different numbers of layers.
- Compare architectures/capacity.
- Change source:target data ratio.
- Allocate a fixed compute budget differently between pre-training and fine-tuning.

### 10\. One-line memory version

> **Practical 1 = small target + large source → pretrain CNN on source → replace output layer → fine-tune on target with smaller LR → compare validation accuracy against training-from-scratch baseline.**

This is the representation I’ll use for **P1** when you later send the other practicals and eventually the proposed mock exam.
