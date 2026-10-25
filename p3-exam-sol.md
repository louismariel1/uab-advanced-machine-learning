# Advanced Machine Learning — Comprehensive Practice Exam Solutions

Practical 3: Self-Supervised Learning · Complete answer key and marking guide

These solutions explain not only the correct answers, but also the reasoning behind them, common exam traps, and the key technical details needed to earn full marks.

The practice exam is worth 50 points. For multiple-choice questions, select every correct statement. The number of correct options is not determined by the question's mark allocation.

# Section A — Multiple-choice questions (12 points)

## Question 1 — Learning paradigms and data distributions (2 points)

Correct answers: A, C, E.

- A — True. Self-supervised learning (SSL) generates training targets automatically from the data. In rotation prediction, the rotation applied to an image supplies the target label. No human annotation of the image's semantic class is needed.
- B — False. Transfer learning does not require identical source and target class labels. The source task might teach general visual features that are useful for an unrelated target classification task.
- C — True. Semi-supervised learning typically combines labelled and unlabelled data to improve learning on a target task. For example, a model may use a small labelled image set and a larger unlabelled set from the same task or domain.
- D — False. SSL is not guaranteed to improve downstream performance. The pretext task may encourage irrelevant features, the source data may differ substantially from the target data, or fine-tuning may be poorly configured.
- E — True. SSL and transfer learning are complementary. In this notebook, SSL creates the pre-trained representation, and transfer learning reuses it for downstream classification.

Key distinction to remember:

- Supervised learning learns from manually supplied target labels.
- Semi-supervised learning combines labelled and unlabelled data for a target task.
- Self-supervised learning generates a learning signal from unlabelled data.
- Transfer learning reuses knowledge learned from a source task or dataset.

## Question 2 — Transfer learning and model architecture (2 points)

Correct answers: A, C, E.

- A — True. The backbone may learn reusable features such as edges, textures, shapes, and object parts without ever seeing the target class labels.
- B — False. The rotation head predicts four rotation categories, whereas the target head predicts five target classes. They solve different tasks, so the pretext head is replaced.
- C — True. The pre-trained backbone weights can be retained while a new classification head is initialized for the target task.
- D — False. Freezing is optional. It is one transfer-learning strategy, not a requirement.
- E — True. In full fine-tuning, both the backbone and the new target head are updated using labelled target examples.

Why this matters: The backbone learns a representation; the head maps that representation to the outputs required by a particular task. Separating these components makes it possible to reuse the representation with a different prediction objective.

## Question 3 — Pretext tasks and data augmentation (2 points)

Correct answers: A, B, D.

- A — True. The transformation itself provides the label. If an image is rotated by two quarter-turns, its rotation label is 2.
- B — True. Four possible rotations require four output classes.
- C — False. An augmentation can remove useful information or make the pretext task inconsistent. Augmentation must suit both the data and the learning objective.
- D — True. If the image's final rotation differs from the assigned label, the model receives an incorrect target. This introduces label noise into the pretext task.
- E — False. High pretext accuracy indicates that the model has learned to solve that particular task. It does not establish that its representations are useful for the downstream task.

For example, a network may learn simple orientation cues that make rotation prediction easy without learning the semantic distinctions required to classify vehicles.

## Question 4 — Contrastive learning (2 points)

Correct answers: A, B, C, E.

- A — True. Two augmented views of the same underlying image are usually treated as a positive pair.
- B — True, with a qualification. In standard contrastive approaches such as SimCLR, views of different images in the batch are often treated as negative pairs. Other methods use different mechanisms and may not explicitly require negative pairs.
- C — True. The objective encourages positive pairs to have similar representations and, in standard contrastive formulations, discourages negative pairs from having overly similar representations.
- D — False. The identity of the source image can provide the positive-pair relationship without a human-provided semantic label.
- E — True. The choice of augmentations, loss, temperature, projection head, and pair construction can affect the representations learned.

Core intuition: The model is trained to preserve useful similarities across different views of the same image while distinguishing those views from other images. This can encourage invariance to certain changes in appearance.

## Question 5 — Training and evaluation (2 points)

Correct answers: A, C, D.

- A — True. Using the same target validation split makes the comparison more meaningful because both models are evaluated on the same examples.
- B — False. Target validation labels must not be used for gradient updates. Doing so leaks evaluation information into training.
- C — True. Best validation accuracy measures peak observed validation performance, while learning curves help diagnose convergence and overfitting.
- D — True. An improvement must be supported by measured results. It should not be assumed simply because SSL was used.
- E — False. Pretext accuracy and target classification accuracy measure different tasks. A model can solve the pretext task well without learning features that improve target classification.

A useful experimental comparison includes both the baseline and fine-tuned model's validation results, with their data splits and evaluation procedure held consistent.

## Question 6 — Model implementation (2 points)

Correct answers: A, C, E.

- A — True. For a batch of $B$ images, the backbone returns one 64-dimensional feature vector per image, producing shape $(B,64)$.
- B — False. The target head has five outputs, so its final shape is $(B,5)$, not $(B,4)$.
- C — True. Cross-entropy loss accepts five-class logits and integer class indices from 0 to 4.
- D — False. The heads can have different output dimensions because they solve different tasks.
- E — True. A state dictionary stores parameter tensors. To load them, you must recreate a compatible model architecture.

The expected shapes are:

$$
(B,64)\rightarrow(B,32)\rightarrow(B,5)
$$

For rotation prediction, the final mapping instead ends in four logits:

$$
(B,64)\rightarrow(B,32)\rightarrow(B,4)
$$

# Section B — Short-answer and applied questions (38 points)

## Question 7 — Explain the complete learning pipeline (4 points)

### (a) Describe the three stages in the correct order. (2 points)

Model answer:

1. Supervised baseline: Initialize a backbone and target classification head, then train both on the small labelled target training set using the true target labels and cross-entropy loss. Evaluate the model on the target validation set to establish a reference performance.
2. Self-supervised pre-training: Train a backbone on the larger unlabelled source dataset using a pretext task, such as predicting which rotation was applied to each image. The labels are generated automatically from the transformations.
3. Supervised fine-tuning: Transfer the pre-trained backbone weights to a downstream model, attach a new target classification head, and train on the labelled target dataset. Evaluate on the same target validation split as the baseline.

Marking guide:

- 1 point for correctly describing the baseline and SSL pre-training stages.
- 1 point for correctly describing fine-tuning and the downstream evaluation.

The order matters: the baseline establishes a benchmark, SSL learns representations from unlabelled images, and fine-tuning adapts those representations to the target task.

### (b) Why replace the pretext head? (2 points)

Model answer:

The pretext head predicts the rotation category, so it has four output classes. The downstream classification head must predict five target classes instead.

The backbone is retained because its features may be useful for the new task, but the pretext head is replaced because its outputs have a different meaning and dimension. A newly initialized target head learns to map the transferred features to the target labels.

Marking guide:

- 1 point for identifying the mismatch between the pretext and target objectives or output dimensions.
- 1 point for explaining that the backbone is reused while a new target head is initialized.

Common mistake: saying that the entire pretext model must be discarded. That would throw away the very representation the transfer-learning stage is designed to reuse.

## Question 8 — Rotation prediction and algorithm interpretation (5 points)

### (a) Identify the learning paradigm and objective. (2 points)

Model answer:

This is self-supervised learning, specifically a rotation-prediction pretext task.

For each unlabelled image $x_i$, a rotation $r_i$ is sampled from $\{0,1,2,3\}$, and the image is transformed accordingly. The rotation index becomes the automatically generated target. The model learns to predict the applied rotation using a four-class classification head and cross-entropy loss.

The objective is to minimize the prediction loss:

$$
\mathcal{L}_{\text{pretext}}
=-\frac{1}{B}\sum_{i=1}^{B}
\log p_\theta(r_i\mid T_{r_i}(x_i))
$$

Here, $T_{r_i}$ denotes the rotation transformation and $p_\theta$ is the model's predicted probability for each rotation class.

No human-provided class labels are required because the training target is known from the transformation applied to each image.

Marking guide:

- 1 point for identifying SSL and the rotation-prediction objective.
- 1 point for explaining how the transformation generates the labels automatically.

### (b) Why might this help downstream classification? (2 points)

Model answer:

To predict rotations, the network may learn useful visual representations that describe the structure of images. These representations can include:

- Low-level features: edges, corners, textures, and boundaries.
- Higher-level features: shapes, object parts, and spatial relationships.

These features may be useful for distinguishing target classes, even though the network was not explicitly trained on the target labels during pre-training.

For example, learning to recognize object shapes in CIFAR-10 images might help distinguish bicycles, buses, and trains in a CIFAR-100 target subset.

Marking guide:

- 1 point for explaining why the pretext task can teach transferable features.
- 1 point for identifying two plausible types of visual features.

### (c) High pretext accuracy but no downstream improvement. (1 point)

Model answer:

The pretext task may have encouraged the model to learn features that are useful for detecting rotation but not sufficiently useful for distinguishing the target classes. For example, it might exploit simple orientation cues rather than learning strong semantic representations.

Other valid explanations include a source-target distribution mismatch, poor fine-tuning, or a pretext task that is too easy.

Exam insight: Pretext-task success is not the final goal. The key question is whether the learned representation improves downstream performance.

## Question 9 — Transfer learning and fine-tuning (5 points)

### (a) Describe the transfer-learning procedure. (2 points)

Model answer:

1. Recreate the same backbone architecture used during pre-training.
2. Load the saved pre-trained backbone weights into that architecture.
3. Initialize a new target classification head with five output neurons.
4. Combine the backbone and target head into the downstream model.
5. Train the model on the labelled target training set and evaluate it on the target validation set.

The backbone's learned weights are retained as the initialization. The pretext head is discarded, and the target head is newly initialized.

In full fine-tuning, both the transferred backbone and new head are updated using the target classification loss.

Marking guide:

- 1 point for correctly transferring the backbone weights.
- 1 point for replacing the pretext head, training on target labels, and evaluating downstream performance.

### (b) Compare freezing and full fine-tuning. (2 points)

Model answer:

| Strategy | Advantage | Disadvantage |
| --- | --- | --- |
| A: Freeze the backbone | Fewer trainable parameters, lower computational cost, and reduced risk of destroying useful pre-trained features | The fixed features may not be sufficiently suitable for the target task |
| B: Fine-tune the entire model | The backbone can adapt its representations to the target data | More computation is required, and a small dataset may lead to overfitting or loss of useful pre-trained representations |

With a frozen backbone, only the target head learns. With full fine-tuning, gradients can update both the head and the backbone.

Marking guide:

- 1 point for correctly describing the two strategies.
- 1 point for a valid advantage and disadvantage of full fine-tuning.

### (c) Why can CIFAR-10 features transfer to CIFAR-100? (1 point)

Model answer:

Although the datasets have different class labels, they share general visual structure. Features such as edges, textures, shapes, and object parts are useful across many image-classification tasks. Therefore, the backbone can provide a useful initialization even when the source and target classes differ.

Important qualification: Transfer is possible, not guaranteed. Its success depends on the relevance of the learned features and the similarity between the source and target data distributions.

## Question 10 — Contrastive learning and data augmentation (5 points)

### (a) Explain positive and negative pairs. (2 points)

Model answer:

A positive pair consists of two different augmented views of the same underlying image. A negative pair typically consists of views from different images.

The contrastive objective encourages the embeddings of positive pairs to become similar while separating embeddings of negative pairs. This encourages the model to learn representations that remain consistent across suitable augmentations while distinguishing different examples.

For example, if two augmented views of the same bicycle image are given to the model, their embeddings should be close in feature space even if the views differ in colour or cropping.

A conceptual objective is:

$$
\mathcal{L}_{\text{contrastive}}
\approx
\underbrace{\text{positive-pair distance}}_{\text{minimize}}
+
\underbrace{\text{negative-pair similarity}}_{\text{discourage}}
$$

This is an intuition rather than the exact formula for a particular contrastive loss.

Marking guide:

- 1 point for defining positive and negative pairs.
- 1 point for explaining the objective's effect on the embeddings.

### (b) Why create two augmented views? (1 point)

Model answer:

Different augmentations create different appearances of the same underlying image. Training the model to produce similar representations for these views encourages it to ignore irrelevant changes, such as moderate colour or crop variations, while preserving useful information about the image.

This is a form of learned invariance.

### (c) Why can an aggressive random crop harm learning? (2 points)

Model answer:

An aggressive crop may remove the main object or important visual features from one view. The two views may then contain substantially different semantic information, making it harder for the model to learn a meaningful shared representation.

For example, one view might contain a bus while another contains only a patch of sky or road. Treating these as equivalent views can provide a misleading training signal.

One improvement is to restrict the crop scale or aspect ratio so that the main object is more likely to remain visible. The student could also combine moderate cropping with colour jitter or horizontal flipping, provided these augmentations are appropriate for the dataset.

Marking guide:

- 1 point for explaining how the crop damages the learning signal.
- 1 point for proposing a sensible, task-compatible augmentation change.

Exam insight: More augmentation is not automatically better. The objective is to create useful variation without destroying the information that the representation should preserve.

## Question 11 — Diagnose an implementation error (5 points)

### (a) Correct output dimensions. (2 points)

Model answer:

- Rotation-prediction head: 4 output logits, one for each rotation class.
- Target classification head: 5 output logits, one for each target class.

The pretext task predicts which of four rotations was applied. The downstream task predicts one of five target categories.

For a batch of $B$ examples, the corresponding final output shapes are:

$$
\text{Pretext output}: (B,4)
$$

$$
\text{Target output}: (B,5)
$$

The backbone can remain the same because both tasks use the same type of image representation. The task-specific heads differ because their prediction targets differ.

Marking guide:

- 1 point for both correct dimensions.
- 1 point for explaining the difference in task objectives.

### (b) Undefined variable `img_dim`. (1 point)

Model answer:

If `img_dim` has not been defined, Python raises a `NameError` when that variable is evaluated. The transformation pipeline cannot be constructed or executed as intended.

The fix is to use the existing variable `img_size`, or define `img_dim` correctly if that is the intended variable name.

### (c) Inconsistent rotation labels. (2 points)

Model answer:

The model is trained using input-target pairs that disagree. For example, an image may be labelled as a 90° rotation but then randomly rotated to 180°. The correct target no longer matches the input.

This introduces incorrect supervision, making it harder for the model to minimize the loss consistently.

To correct the procedure, first apply augmentations that do not change the rotation label, then sample the rotation and apply it, and finally return the corresponding rotation index. No later transformation should change the rotation in a way that invalidates that label.

Marking guide:

- 1 point for identifying the mismatch as incorrect or noisy supervision.
- 1 point for describing a procedure that preserves label consistency.

## Question 12 — Evaluate the experimental results (5 points)

The results are:

| Metric | Supervised baseline | SSL + fine-tuning |
| --- | --- | --- |
| Best target validation accuracy | 42% | 49% |
| Final training accuracy | 78% | 83% |
| Pretext validation accuracy | — | 91% |

### (a) Calculate the absolute and relative improvements. (2 points)

Absolute improvement:

$$
49\%-42\%=\boxed{7\text{ percentage points}}
$$

Relative improvement:

Relative improvement compares the increase to the baseline:

$$
\frac{49-42}{42}\times100\%
=
\frac{7}{42}\times100\%
$$

$$
\boxed{16.67\%\text{ approximately}}
$$

The correct interpretation is a gain of 7 percentage points, or a relative accuracy increase of approximately 16.7% compared with the baseline.

These two quantities are different. A change from 42% to 49% is not a 7% relative improvement.

Marking guide:

- 1 point for the absolute improvement.
- 1 point for the relative improvement with the correct denominator.

### (b) Interpret the results. (2 points)

Model answer:

The SSL-pre-trained and fine-tuned model achieves higher target validation accuracy than the supervised baseline. In this experiment, the increase from 42% to 49% suggests that pre-training provided useful information for the downstream classification task.

The pretext validation accuracy of 91% indicates that the model learned to predict rotations successfully on the pretext validation data. However, it does not directly measure target classification performance and does not, by itself, prove that the learned representations are useful for the downstream task.

The target validation comparison provides the more direct evidence of downstream benefit.

The training accuracies are higher than the corresponding best validation accuracies, suggesting a possible generalization gap. However, the final training accuracy should not be compared too precisely with a best validation accuracy from a different epoch; the metrics are not necessarily measured at the same training point.

Marking guide:

- 1 point for interpreting the downstream improvement correctly.
- 1 point for distinguishing pretext accuracy from target-task performance.

### (c) One additional experiment or control. (1 point)

Model answer:

Repeat the baseline and SSL experiment with multiple random seeds, keeping the target data split and evaluation procedure consistent, and compare the mean validation accuracy and variation across runs.

Other valid answers include:

- Compare frozen-backbone transfer with full fine-tuning.
- Use a randomly initialized backbone with the same target training procedure as a control.
- Compare different pretext tasks using the same downstream evaluation.
- Check whether the improvement persists across different target train/validation splits.

Repeated experiments help determine whether the observed improvement is robust rather than an effect of random initialization or a particular training run.

## Question 13 — Overfitting, validation, and experimental design (5 points)

### (a) Identify and explain the phenomenon. (2 points)

Model answer:

This pattern suggests overfitting. The model continues to improve on the training data but begins to perform worse on unseen validation examples.

A small labelled dataset may not contain enough diverse examples to constrain a high-capacity neural network. The model can memorize training-specific details instead of learning features that generalize to new images.

The typical pattern is increasing training accuracy accompanied by validation accuracy that plateaus or decreases. Validation loss may also begin to rise.

Marking guide:

- 1 point for identifying overfitting.
- 1 point for explaining the relationship between limited training data, model capacity, and generalization.

### (b) Give two appropriate solutions. (2 points)

Model answer:

1. Data augmentation: Apply suitable random crops, horizontal flips, or moderate colour jitter to create varied training examples. This can reduce reliance on incidental details and encourage more robust features.
2. Early stopping: Stop training when validation performance no longer improves, and retain the best-performing checkpoint. This prevents continued optimization of the training set from unnecessarily degrading generalization.

Other valid solutions include weight decay, dropout, freezing some pre-trained layers, collecting more labelled data, reducing model capacity, or tuning the learning rate.

Marking guide: Award 1 point per valid approach with a reasonable explanation.

### (c) Why not use validation labels for gradient updates? (1 point)

Model answer:

The validation set is intended to estimate generalization to data not used to fit the model. Using its labels to calculate gradients leaks validation information into training, making the validation results less independent and potentially overoptimistic.

Validation results can guide model selection and early stopping, but the validation examples should not be used directly for gradient updates.

## Question 14 — Learning paradigms and choosing a method (4 points)

Each part is worth 1 point.

### (a) Predict rotations from automatically generated labels

Answer: Self-supervised learning.

The rotation transformation supplies the training target without requiring a human to label the image's semantic class.

### (b) Combine a small labelled dataset and a larger unlabelled dataset for the same target task

Answer: Semi-supervised learning.

The model uses both labelled and unlabelled examples to improve the target-task learning process.

### (c) Reuse a backbone trained on a source task for a different target task

Answer: Transfer learning.

The model transfers learned parameters or representations from the source task to the target task.

### (d) Train a classifier only on images with manually provided class labels

Answer: Supervised learning.

The model learns a mapping from input images to their supplied target classes using labelled examples.

Important distinction: A practical can use more than one learning paradigm. The notebook combines self-supervised pre-training with transfer learning and supervised fine-tuning. These labels describe different aspects of the learning procedure, so they are not mutually exclusive.

# Final revision sheet: concepts most likely to be tested

## 1. Compare the learning paradigms

| Method | Training signal | Main purpose |
| --- | --- | --- |
| Supervised learning | Human-provided labels | Learn the target prediction task |
| Semi-supervised learning | Labelled and unlabelled examples | Improve target-task learning with limited labels |
| Self-supervised learning | Automatically generated targets | Learn representations from unlabelled data |
| Transfer learning | Knowledge from a source model or task | Reuse learned features for a new task |

## 2. Remember the notebook's architecture

Unlabelled source images

CIFAR-10 images · No semantic labels needed

Self-supervised pre-training

Backbone + 4-class rotation head

Transfer the backbone weights

Discard the rotation head

Supervised fine-tuning

Transferred backbone + new 5-class target head

Compare target validation accuracy

SSL + fine-tuning versus supervised baseline

## 3. Key equations and facts

Normalization

$$
x_{\text{norm}}=\frac{x-0.5}{0.2}
$$

An input value of $0.5$ becomes zero.

Cross-entropy for rotation prediction

$$
\mathcal L=-\frac{1}{B}\sum_{i=1}^{B}
\log p_\theta(r_i\mid x_i)
$$

The model learns to assign high probability to the correct rotation label.

Absolute improvement

$$
\Delta_{\text{pp}}=A_{\text{SSL}}-A_{\text{baseline}}
$$

Relative improvement

$$
\Delta_{\text{rel}}=
\frac{A_{\text{SSL}}-A_{\text{baseline}}}
{A_{\text{baseline}}}\times100\%
$$

Model output dimensions

- Backbone: $(B,64)$
- Projection layer: $(B,32)$
- Rotation head: $(B,4)$
- Target classification head: $(B,5)$

## 4. Common exam traps

- High pretext accuracy does not guarantee high downstream accuracy.
- Self-supervised learning does not require human-provided semantic labels, but it still needs a learning signal.
- Source and target datasets do not need identical classes for transfer learning.
- The pretext head is task-specific; the backbone contains the transferable representation.
- Freezing the backbone is optional. Full fine-tuning updates both backbone and target head.
- Augmentations must preserve the information required by the learning objective.
- A 7-percentage-point gain from 42% to 49% is a 16.67% relative improvement.
- Validation data should guide evaluation and model selection, not gradient updates.
- A model's training performance can improve while its generalization worsens.

## 5. How to maximize marks on written questions

For a 2–5 point question, structure your answer around three things:

1. Identify the concept or objective. State which method is being used and what it optimizes.
2. Explain the mechanism. Describe the data, representation, loss, parameter update, or transfer procedure that matters.
3. Explain the consequence. State why the method helps, when it might fail, or what the observed results imply.

For example, rather than writing “SSL helps because it learns features,” explain that SSL uses automatically generated targets to train a backbone on unlabelled images; the resulting representations may capture transferable visual patterns, which can improve classification after supervised fine-tuning, although improvement depends on the pretext task and data.

That level of explanation matches the official exam's emphasis on understanding algorithms rather than memorizing their names.
