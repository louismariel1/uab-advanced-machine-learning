# Advanced Machine Learning — Practice Exam

## Backbones, Heads, and Neural Network Computation

Original practice exam · Autumn 2026 · Modeled on the structure and reasoning style of the supplied past exam

I've identified the essential structure of your professor's exam:

- Multiple-choice questions worth 2 points, with potentially multiple correct answers.
- Short written explanations worth 2–5 points, testing conceptual understanding rather than memorization.
- Algorithm interpretation: explain what a training procedure optimizes and why.
- Tensor and feature-space reasoning: follow representations through the model and reason about the desired outputs.
- Code and implementation analysis: identify a problem or explain the behavior without writing code.
- Theoretical questions: justify why a technique works, including its limitations.

The practice exam below follows that structure, but all questions focus on our topic: backbone/head architecture, tensor transformations, feature extraction, gradients, transfer learning, and advanced prediction heads.

The supplied exam's examples total 35 points, so this practice exam also totals 35 points, with a mixture of multiple-choice and written-answer questions.

## Question 1 — Component identification and deployment (2 points)

Which of the following statements about the backbone and head of a neural network are true? Mark all that apply.

□ A. A backbone can contain convolutional layers, attention blocks, normalization layers, and activation functions.

□ B. A classification head must always contain a softmax layer as its final operation.

□ C. Replacing a classification head can allow a pretrained backbone to be reused for a new task with a different number of classes.

□ D. Every fully connected layer in a neural network necessarily belongs to the head.

□ E. A frozen backbone can still compute feature representations during inference.

## Question 2 — Fine-tuning and parameter updates (2 points)

Which of the following statements are true when fine-tuning a pretrained backbone and adapting its head to a new target task? Mark all that apply.

□ A. Freezing the backbone prevents its parameters from being updated, but it can still produce feature tensors.

□ B. Fine-tuning requires every parameter in the backbone and head to be updated.

□ C. A smaller learning rate for the backbone than for the newly initialized head can be a reasonable strategy.

□ D. Setting `model.eval()` automatically freezes all parameters and disables gradient computation.

□ E. A new head may need training even if the pretrained backbone is already effective at extracting features.

## Question 3 — Tensor dimensions and feature extraction (2 points)

A CNN backbone produces a feature tensor with shape

$$
X\in\mathbb{R}^{16\times256\times8\times8}.
$$

The tensor is passed through global average pooling and then through a linear classification head with 12 output classes.

Which statements are true? Mark all that apply.

□ A. Global average pooling produces a tensor with shape $16\times256$.

□ B. The linear head can use a weight matrix of shape $12\times256$, assuming the convention $y=Wx+b$.

□ C. The final output has shape $16\times12$.

□ D. Global average pooling learns 256 independent convolution kernels.

□ E. The number of input images in the batch changes the number of classes predicted by the head.

## Question 4 — Changing the output head (3 points)

A researcher loads a pretrained CNN that was trained to classify 1,000 classes. The researcher replaces its final linear layer with a new layer intended to classify 20 classes.

However, during evaluation, the model still produces an output tensor with shape $B\times1000$, where $B$ is the batch size.

Explain two possible reasons for this behavior and describe how the researcher should investigate and resolve the problem.

Your answer should distinguish between the backbone's feature dimension and the head's output dimension.

## Question 5 — Feature embeddings and contrastive learning (3 points)

A researcher wants to train a backbone using contrastive learning. Two images, $a$ and $b$, are sampled from the dataset. Each image is augmented independently, producing $a^*$ and $b^*$.

The four images are passed through the backbone and a projection head to produce embeddings. The goal is to make two augmented views of the same original image similar while making views of different images dissimilar.

Fill each off-diagonal cell with HIGH or LOW, according to the desired embedding distance.

|  | $a$ | $a^*$ | $b$ | $b^*$ |
| --- | --- | --- | --- | --- |
| $a$ | — | ? | ? | ? |
| $a^*$ | ? | — | ? | ? |
| $b$ | ? | ? | — | ? |
| $b^*$ | ? | ? | ? | — |

Then answer the following:

1. Why should the distances between $a$ and $a^*$, and between $b$ and $b^*$, be low?
2. What role does the projection head play in this training scheme?
3. Why might the learned backbone be useful for a later classification task even if the contrastive pretraining used no class labels?

## Question 6 — Freezing a backbone and training a head (4 points)

Consider the following training procedure:

1. Load a pretrained backbone.
2. Freeze all backbone parameters.
3. Replace the original classification head with a randomly initialized head for a new task.
4. For each training batch, compute the backbone features, calculate the head's predictions, evaluate the loss against the ground-truth labels, and update the trainable parameters.
5. Repeat until the training objective converges or the training budget is exhausted.

Answer the following questions.

(a) (2 points) Describe the algorithm in pseudocode or natural English, identifying which parameters are updated and which remain fixed.

(b) (1 point) Explain why the head can learn useful predictions even though the backbone is frozen.

(c) (1 point) Give one situation in which subsequently fine-tuning some backbone layers could improve performance.

## Question 7 — Interpreting a model architecture (4 points)

A computer vision model consists of three components:

- Backbone: a CNN that outputs feature maps.
- Neck: a feature pyramid network that combines features at multiple spatial resolutions.
- Head: a detection module that predicts object classes and bounding-box coordinates.

A researcher removes the neck and connects the detection head directly to the backbone's final feature map. The model still runs, but its detection performance deteriorates, especially for small objects.

Answer the following.

(a) (2 points) Explain the functional roles of the backbone, neck, and head in this model.

(b) (1 point) Explain why removing the neck could reduce performance on small objects.

(c) (1 point) Is the neck necessarily part of the backbone or the head? Explain why the answer depends on the architecture's component definitions.

## Question 8 — Reading a training procedure (5 points)

A researcher uses the following procedure to train a neural network:

1. Compute the backbone features for every training example.
2. Use the head to calculate a prediction and a loss for each example.
3. Select only examples whose loss is below a threshold $\lambda$.
4. Calculate the mean loss over the selected examples.
5. Backpropagate this loss and update the trainable parameters.
6. Repeat the procedure over multiple epochs.

Answer the following.

(a) (2 points) Explain what this algorithm does and why selecting examples according to their losses might be useful.

(b) (1 point) What is the likely effect of choosing a very small threshold $\lambda$? What might happen if it is too large?

(c) (1 point) If the backbone is frozen, which parameters can be updated in this procedure, assuming the head is trainable?

(d) (1 point) Identify one potential failure mode of this training strategy.

## Question 9 — Diagnosing a model implementation (5 points)

A researcher has a pretrained model with a backbone that produces 512-dimensional feature vectors. The researcher replaces the original classification head with a linear layer intended to classify 10 classes.

The following conceptual implementation is used:

python
```
features = backbone(images)
logits = head(features)

loss = cross_entropy(logits, labels)
loss.backward()
optimizer.step()
```

The researcher reports one of two problems:

- Problem A: The head produces 1,000 logits instead of 10.
- Problem B: The backbone's parameters do not change, even though the researcher expects to fine-tune the entire model.

Answer the following.

(a) (2 points) For Problem A, identify the likely architectural mistake and describe the expected dimensions of the head's weight matrix and output tensor.

(b) (2 points) For Problem B, give two possible reasons why the backbone's parameters might not change and explain how the researcher could investigate them.

(c) (1 point) Explain why calling `loss.backward()` and `optimizer.step()` is not, by itself, a guarantee that every model parameter will be updated.

## Question 10 — Why can a pretrained backbone outperform training from scratch? (5 points)

Two researchers want to train an image classifier using a small labelled dataset.

- Researcher A initializes both the backbone and head randomly and trains the entire network from scratch.
- Researcher B loads a backbone pretrained on a large image dataset, replaces its original head, and trains the new head on the small labelled dataset.

Researcher B achieves better validation accuracy.

Answer the following.

(a) (2 points) Give a theoretical explanation for why Researcher B might achieve better performance.

(b) (1 point) Explain the role of the new head in adapting the pretrained representation to the target task.

(c) (1 point) Identify one situation in which the pretrained backbone might be less effective for the new task.

(d) (1 point) Explain the difference between freezing the pretrained backbone and fine-tuning it, and describe one trade-off between these approaches.

## End of practice exam

## Exam summary

35 points

Suggested duration: 60–75 minutes for this focused practice set.

| Questions | Assessment style | Points |
| --- | --- | --- |
| 1–3 | Multiple choice, multiple answers possible | 6 |
| 4–5 | Short written answers and feature-space reasoning | 6 |
| 6–7 | Algorithm and architecture explanations | 8 |
| 8–9 | Training procedure and implementation analysis | 10 |
| 10 | Theoretical explanation | 5 |
| Total |  | 35 |
