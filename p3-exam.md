# Advanced Machine Learning — Practice Exam

Based on Practical 3: Self-Supervised Learning · Autumn 2026 exam style

Exam conditions: 90 minutes · 50 points · Pen and paper

Question format: Multiple choice and free-written answers.

Instructions

- Attempt all questions.
- Multiple-choice questions may have more than one correct answer. Select all that apply.
- Explain your reasoning in written questions. Concise answers are acceptable if they contain the essential details.
- You do not need to write executable code. Use pseudocode, equations, diagrams, or natural language.
- Assume the practical uses CIFAR-10 as the unlabelled source dataset and a five-class CIFAR-100 subset as the labelled target dataset, unless a question states otherwise.
- The exam focuses on conceptual understanding, algorithm interpretation, and explaining why methods work.

## Section A — Multiple-choice questions (12 points)

### Question 1 — Learning paradigms and data distributions (2 points)

Which of the following statements about the notebook's learning setup are correct? Mark all that apply.

[ ]A. Self-supervised learning generates training targets from the unlabelled data itself.

[ ]B. Transfer learning requires the source and target datasets to have identical class labels.

[ ]C. Semi-supervised learning typically combines labelled and unlabelled examples to improve learning for a target task.

[ ]D. Self-supervised pre-training guarantees higher downstream validation accuracy than training from scratch.

[ ]E. Self-supervised learning and transfer learning can be used sequentially in the same pipeline.

### Question 2 — Transfer learning and model architecture (2 points)

After rotation-prediction pre-training, the notebook transfers a backbone to a five-class classification task. Which statements are correct?

[ ]A. The pre-trained backbone can contain useful visual features even though it was not trained on the target class labels.

[ ]B. The four-class rotation head must be retained unchanged for five-class classification.

[ ]C. A new target classification head can be initialized while retaining the pre-trained backbone weights.

[ ]D. The backbone must always be frozen during fine-tuning.

[ ]E. Fine-tuning can update the backbone as well as the new classification head.

### Question 3 — Pretext tasks and data augmentation (2 points)

Which statements about rotation prediction and augmentation are correct?

[ ]A. The rotation label can be generated automatically from the rotation applied to an image.

[ ]B. The rotation-prediction head requires four output classes for rotations of 0°, 90°, 180°, and 270°.

[ ]C. Every augmentation improves the usefulness of the learned representation, regardless of the pretext task.

[ ]D. Randomly changing the rotation after assigning its label can create incorrect training targets.

[ ]E. High rotation-prediction accuracy necessarily proves that downstream classification will improve.

### Question 4 — Contrastive learning (2 points)

Suppose you replace rotation prediction with a contrastive pretext task. You create two augmented views of the same image and views of other images. Which statements are correct?

[ ]A. The two views of the same image are usually treated as a positive pair.

[ ]B. Views from different images are often treated as negative pairs in standard contrastive learning.

[ ]C. The objective is generally to bring positive-pair representations closer and separate negative-pair representations.

[ ]D. Contrastive learning always requires human-provided class labels.

[ ]E. The exact loss and pair construction influence which representations are learned.

### Question 5 — Training and evaluation (2 points)

Which practices make the comparison between the supervised baseline and the SSL-pre-trained model more reliable?

[ ]A. Evaluate both models on the same target validation split.

[ ]B. Include the target validation labels in gradient updates during training.

[ ]C. Compare their best validation accuracies and inspect their learning curves.

[ ]D. Report an improvement only if the measured results support it.

[ ]E. Treat pretext validation accuracy as interchangeable with target classification accuracy.

### Question 6 — Model implementation (2 points)

The backbone outputs 64 features, the target head projects to 32 dimensions, and there are five target classes. Which statements are correct?

[ ]A. For a batch of size B, the backbone output has shape (B, 64).

[ ]B. The target head's final output has shape (B, 4).

[ ]C. Cross-entropy loss can be used with five-class logits and integer labels from 0 to 4.

[ ]D. The rotation head and target head must have the same number of output neurons because they share a backbone.

[ ]E. A saved state dictionary contains the model weights but does not by itself define the complete model architecture.

## Section B — Short-answer and applied questions (38 points)

### Question 7 — Explain the complete learning pipeline (4 points)

The notebook trains a supervised baseline, pre-trains a backbone using an automatically generated pretext task, and then fine-tunes the backbone on the labelled target dataset.

(a) (2 points) Describe the three stages in the correct order. For each stage, identify the data used and the main learning objective.

(b) (2 points) Explain why the notebook replaces the pretext head with a new target classification head instead of reusing the original head.

### Question 8 — Rotation prediction and algorithm interpretation (5 points)

Consider the following algorithm, which uses unlabelled source images.

For each training batch:

1. Sample a batch of unlabelled images.
2. Randomly choose a rotation $r_i \in \{0,1,2,3\}$ for each image.
3. Rotate each image by $90r_i$ degrees.
4. Pass the rotated images through a backbone and a four-class prediction head.
5. Calculate cross-entropy loss between the predicted rotation classes and the generated labels.
6. Backpropagate the loss and update the model parameters.

(a) (2 points) What learning paradigm and training objective does this algorithm represent? Explain why it does not require human-provided class labels.

(b) (2 points) Explain why training the backbone on this task might help classify the five target classes later. Identify two kinds of visual features it might learn.

(c) (1 point) The rotation-prediction model achieves very high validation accuracy, but the fine-tuned model performs no better than the supervised baseline. Give one plausible explanation.

### Question 9 — Transfer learning and fine-tuning (5 points)

A student has completed rotation-prediction pre-training and saved the backbone weights. They now want to classify five target classes from a small labelled dataset.

(a) (2 points) Describe the steps required to transfer the pre-trained backbone to the target task. State which parameters are retained and which are newly initialized.

(b) (2 points) Compare these two strategies:

- Strategy A: freeze the backbone and train only the new classification head.
- Strategy B: update both the backbone and the new classification head.

Explain one potential advantage and one potential disadvantage of fine-tuning the entire model.

(c) (1 point) Why can features learned from CIFAR-10 be useful for classifying a CIFAR-100 subset even though their class labels differ?

### Question 10 — Contrastive learning and data augmentation (5 points)

A student decides to use contrastive learning instead of rotation prediction.

For each source image $x$, they create two randomly augmented views, $x_i$ and $x_j$. The model maps both views to feature embeddings $z_i$ and $z_j$.

(a) (2 points) Explain the role of positive and negative pairs in a standard contrastive-learning objective. Describe how the loss encourages useful representations.

(b) (1 point) Why might two different augmentations of the same image be useful for representation learning?

(c) (2 points) The student applies an aggressive random crop that frequently removes the main object from one view. Explain why this may harm learning and describe one way to improve the augmentation strategy.

### Question 11 — Diagnose an implementation error (5 points)

A student implements the notebook's rotation-prediction task but encounters the following problems:

- The pretext head produces five logits instead of four.
- The target classification head produces four logits instead of five.
- The code uses `size=img_dim`, but the notebook defines `img_size`.
- During pre-training, random rotation is applied after the rotation label is assigned, so the image's final rotation may not match the label.

(a) (2 points) Identify the correct output dimensions for the pretext head and target head, and explain why they differ.

(b) (1 point) Explain the likely problem caused by using `img_dim` when that variable is undefined.

(c) (2 points) Explain why inconsistent rotation labels are harmful to training. Describe how you would correct the procedure.

### Question 12 — Evaluate the experimental results (5 points)

A student obtains the following results:

| Metric | Supervised baseline | SSL pre-trained and fine-tuned |
| --- | --- | --- |
| Best target validation accuracy | 42% | 49% |
| Final training accuracy | 78% | 83% |
| Pretext validation accuracy | — | 91% |

(a) (2 points) Calculate the absolute improvement in target validation accuracy in percentage points. Calculate the relative improvement as a percentage of the baseline accuracy.

(b) (2 points) Interpret these results. What do they suggest about the usefulness of pre-training, and what can you conclude from the pretext validation accuracy?

(c) (1 point) Name one additional experiment or control that would make the conclusion more convincing.

### Question 13 — Overfitting, validation, and experimental design (5 points)

During training, a student observes that training accuracy continues to rise, while target validation accuracy peaks early and subsequently decreases.

(a) (2 points) What phenomenon does this pattern suggest? Explain why it can occur when training on a small labelled dataset.

(b) (2 points) Give two appropriate ways to address the problem. Explain briefly how each could help.

(c) (1 point) Explain why the target validation set should not be used to calculate gradients during training.

### Question 14 — Learning paradigms and choosing a method (4 points)

For each scenario, identify the most appropriate learning paradigm from the following options:

- Supervised learning
- Semi-supervised learning
- Self-supervised learning
- Transfer learning

A paradigm may be used more than once.

(a) (1 point) A model learns to predict image rotations using automatically generated labels from unlabelled images.

(b) (1 point) A model is trained using a small collection of labelled images and a larger collection of unlabelled images to improve the same target classification task.

(c) (1 point) A model trained on a source task has its learned backbone weights reused for a different target task.

(d) (1 point) A model learns to classify images using only images paired with manually provided class labels, with no pre-training stage.

## Exam summary

| Section | Questions | Marks |
| --- | --- | --- |
| A — Multiple choice | 1–6 | 12 |
| B — Short-answer and applied questions | 7–14 | 38 |
| Total | 14 questions | 50 points |

Suggested time allocation:

- Section A: 15 minutes
- Section B: 65 minutes
- Review: 10 minutes

Coverage: Self-supervised learning, transfer learning, contrastive learning, data augmentation, pretext tasks, model architecture, fine-tuning, supervised and semi-supervised learning, loss functions, debugging, overfitting, and experimental evaluation.

The questions follow the official exam's emphasis on understanding what algorithms optimise, why they work, and how to reason about their behaviour. The answer key is intentionally separate so you can attempt the exam under timed conditions first.
