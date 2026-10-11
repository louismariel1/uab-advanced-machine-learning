# Advanced Machine Learning — Comprehensive Solutions and Explanations

## Backbones, Heads, and Neural Network Computation

This solution guide follows the 10-question, 35-point practice exam. The aim is not just to memorize the correct answers, but to understand the reasoning well enough to handle variations of the questions in a pen-and-paper exam.

For each question, we'll cover:

- Correct answer or model solution
- Reasoning: why the answer is correct
- Common traps: why other answers may appear plausible but are incorrect
- Exam takeaway: what you should remember and be able to explain

## The foundation: how the components fit together

## 1. Backbone

Input data → feature extraction

Learned weights, convolutions, attention, normalization, activations

2. Optional neck or intermediate processing

Feature fusion, reshaping, pooling, multiscale processing

## 3. Head

Features → task-specific outputs

Class scores, bounding boxes, masks, regression values, or token scores

The backbone/head distinction is functional, not an absolute rule about layer types. For example, convolutional layers can appear in a backbone or a prediction head. A pooling operation can be considered part of the backbone or the head depending on how the architecture is defined.

A second distinction is equally important:

- Architecture: which components and operations the model contains.
- Parameters: learned weights and biases.
- Activations: intermediate values produced during the forward pass.
- Training procedure: how predictions, losses, gradients, and parameter updates are computed.

These concepts interact, but they are not interchangeable.

# Question 1 — Component identification and deployment (2 points)

Correct answers: A, C, E.

### A. A backbone can contain convolutional layers, attention blocks, normalization layers, and activation functions.

True.

A CNN backbone may contain convolutions, nonlinear activations such as ReLU, normalization layers such as BatchNorm, and residual connections.

A Transformer backbone may contain self-attention, feed-forward networks, normalization, and residual connections.

The architecture depends on the data and task. The defining feature of a backbone is its role in constructing useful representations.

### B. A classification head must always contain a softmax layer as its final operation.

False.

A classification head often produces logits, which are unnormalized class scores. Softmax may then convert those scores into probabilities.

However, the model may return logits directly. During training, cross-entropy implementations often combine log-softmax and the loss internally.

Furthermore, multilabel classification typically uses independent sigmoid outputs, and regression heads often produce numerical values without softmax.

### C. Replacing a classification head can allow a pretrained backbone to be reused for a new task with a different number of classes.

True.

Suppose a pretrained backbone produces 512 features. Its original head predicts 1,000 classes, but the new task requires 20 classes.

The original head might use:

$$
W_{\text{old}}\in\mathbb{R}^{1000\times512}
$$

The new head can use:

$$
W_{\text{new}}\in\mathbb{R}^{20\times512}
$$

The backbone can remain unchanged because it still produces the same 512-dimensional feature representation.

### D. Every fully connected layer in a neural network necessarily belongs to the head.

False.

A fully connected layer can be used inside a backbone. For example, Transformer feed-forward blocks use linear layers to transform hidden representations.

A final linear classification layer is typically part of the head, but the operation alone does not determine the component's role.

### E. A frozen backbone can still compute feature representations during inference.

True.

Freezing prevents its parameters from being updated during training. It does not remove the backbone or stop its forward pass.

The frozen backbone still transforms input data into features, which the head can use to make predictions.

### Exam takeaway

Remember the distinction between what a component computes and whether its parameters are being trained. A frozen backbone is still computationally active.

# Question 2 — Fine-tuning and parameter updates (2 points)

Correct answers: A, C, E.

### A. Freezing the backbone prevents its parameters from being updated, but it can still produce feature tensors.

True.

The backbone computes features normally. Its parameters are simply excluded from gradient-based learning, commonly by setting their `requires_grad` attributes to `False`.

### B. Fine-tuning requires every parameter in the backbone and head to be updated.

False.

Fine-tuning can update all parameters, but it does not have to.

Common strategies include:

- Training only the new head.
- Training the head and the last few backbone layers.
- Fine-tuning the entire model.
- Using different learning rates for different parts of the model.

### C. A smaller learning rate for the backbone than for the newly initialized head can be a reasonable strategy.

True.

The pretrained backbone already contains useful features. A large update might destroy some of that useful knowledge, especially when the target dataset is small.

A newly initialized head, on the other hand, may need larger parameter updates to learn the target task.

For example:

$$
\eta_{\text{backbone}}=10^{-5},
\qquad
\eta_{\text{head}}=10^{-3}
$$

These are illustrative values, not universal recommendations.

This approach is often called discriminative learning rates.

### D. Setting `model.eval()` automatically freezes all parameters and disables gradient computation.

False.

In PyTorch, these operations serve different purposes:

- `model.eval()` changes the behavior of modules such as Dropout and BatchNorm.
- `requires_grad = False` prevents gradients from being computed for selected parameters.
- `torch.no_grad()` disables gradient recording for operations executed within its context.

Calling `eval()` alone does not freeze parameters.

### E. A new head may need training even if the pretrained backbone is already effective at extracting features.

True.

The backbone may extract useful features, but the head must learn how those features relate to the target labels.

For example, visual features learned from a general image dataset may be useful for flower classification, but the original head's classes and weights may not be appropriate.

### Exam takeaway

Freezing, evaluation mode, gradient tracking, and optimizer updates are four related but distinct concepts. Be prepared to explain each separately.

# Question 3 — Tensor dimensions and feature extraction (2 points)

Correct answers: A, B, C.

The backbone produces:

$$
X\in\mathbb{R}^{16\times256\times8\times8}
$$

The dimensions represent:

- $16$: batch size.
- $256$: number of feature channels.
- $8\times8$: spatial resolution of each feature map.

### A. Global average pooling produces a tensor with shape $16\times256$.

True.

For each example and each channel, global average pooling averages the $8\times8$ spatial values:

$$
y_{n,c}=
\frac{1}{64}
\sum_{h=1}^{8}\sum_{w=1}^{8}X_{n,c,h,w}
$$

There are 256 channels, so each example is reduced to 256 numbers.

Therefore:

$$
(16,256,8,8)\rightarrow(16,256)
$$

### B. The linear head can use a weight matrix of shape $12\times256$.

True.

Using the convention $y=Wx+b$, there are 12 output classes and 256 input features:

$$
W\in\mathbb{R}^{12\times256},
\qquad
b\in\mathbb{R}^{12}
$$

For one example:

$$
(12\times256)(256\times1)
=
12\times1
$$

The result contains 12 class logits.

### C. The final output has shape $16\times12$.

True.

The model processes 16 examples, and the head produces 12 logits per example:

$$
(16,256)\rightarrow(16,12)
$$

### D. Global average pooling learns 256 independent convolution kernels.

False.

Global average pooling averages existing feature values. Standard global average pooling has no trainable convolution kernels.

The preceding convolutional layers may contain learned kernels, but pooling does not learn those kernels.

### E. The number of input images in the batch changes the number of classes predicted by the head.

False.

The batch size determines the number of rows in the output tensor. The output dimension of the head determines the number of classes.

For a fixed 12-class head:

$$
\text{Output shape}=(B,12)
$$

Changing $B$ does not change the 12 classes.

### Exam takeaway

Always distinguish the batch dimension, feature dimension, and output dimension. This is one of the most common sources of errors when analyzing neural network architectures.

# Question 4 — Changing the output head (3 points)

## Model answer

There are two plausible explanations.

Reason 1: The original classification head was not actually replaced.

The researcher may have loaded the pretrained model but continued using its original 1,000-class output layer. They should inspect the model architecture and confirm that the head used in the forward pass is the newly initialized 20-class layer.

Reason 2: The new head was created but is not used in the forward pass.

The researcher might have defined a new layer but accidentally continued calling the original classifier. They should trace the forward pass and inspect the shape of the tensor returned by the final layer.

To diagnose the problem, inspect the feature shape immediately before the head and the logits shape immediately after it.

If the backbone produces 512-dimensional features, the new head should have:

$$
W\in\mathbb{R}^{20\times512}
$$

The output should have shape:

$$
(B,512)\rightarrow(B,20)
$$

where $B$ is the batch size.

If the model still outputs $B\times1000$, the original output layer or another 1,000-output component is still being used.

## Explanation

The key distinction is between the backbone's feature dimension and the head's output dimension.

The backbone determines the size of the representation that the head receives. The head determines the size and meaning of the final prediction.

Changing the number of classes does not ordinarily require changing the backbone's output dimension.

## Marking guide

| Criterion | Points |
| --- | --- |
| Identifies the possibility that the old head remains in use | 1 |
| Identifies a second plausible cause or explains how to inspect the forward pass | 1 |
| Correctly explains the dimensions of the new head and its output | 1 |
| Total | 3 |

### Common trap

Saying only "change the output layer to 20 classes" is incomplete. A strong answer explains why the output dimension remains 1,000 and how to trace the actual computation.

# Question 5 — Feature embeddings and contrastive learning (3 points)

## 1. Completed distance matrix

The intended distances are:

|  | $a$ | $a^*$ | $b$ | $b^*$ |
| --- | --- | --- | --- | --- |
| $a$ | — | LOW | HIGH | HIGH |
| $a^*$ | LOW | — | HIGH | HIGH |
| $b$ | HIGH | HIGH | — | LOW |
| $b^*$ | HIGH | HIGH | LOW | — |

This assumes that $a$ and $a^*$ form one positive pair, $b$ and $b^*$ form another, and different original images serve as negative pairs.

## 2. Why should the positive-pair distances be low? (1 point)

The two views of the same original image should have similar representations despite the random augmentations.

For example, changing an image's crop, brightness, or color should not necessarily change the semantic identity of the object.

Contrastive learning encourages the model to learn features that are relatively invariant to such transformations.

## 3. What is the role of the projection head? (1 point)

The projection head transforms the backbone's representation into an embedding space where the contrastive objective is applied.

The projection head might consist of one or more fully connected layers and nonlinear activations. The contrastive loss trains the backbone and projection head jointly, assuming both are trainable.

The projection head is distinct from a downstream classification head: it is designed for the pretraining objective rather than directly predicting the target classes.

## 4. Why can the backbone be useful for classification without class labels? (1 point)

Contrastive learning encourages the backbone to capture useful structure in the data by distinguishing related views from unrelated examples.

The resulting representations may encode shapes, textures, object identity, and other semantic information. A new classification head can then learn to map those representations to class labels using a smaller labelled dataset.

Important qualification: this benefit is not guaranteed. Poor augmentations, an unsuitable objective, or weak training can produce unhelpful representations. Also, in a full contrastive-learning setup, negatives are often other examples or views in the batch; their relationship to the positive pair matters.

## Marking guide

| Criterion | Points |
| --- | --- |
| Correct matrix entries | 1 |
| Explains positive-pair similarity and the projection head's role | 1 |
| Explains the transfer of learned representations to a downstream task | 1 |
| Total | 3 |

### Common trap

The goal is not to make every pair of embeddings close. The model should distinguish different examples while making the designated positive pairs similar. The exact objective and its behavior depend on the contrastive-learning method.

# Question 6 — Freezing a backbone and training a head (4 points)

## (a) Describe the algorithm (2 points)

A suitable pseudocode answer is:

text
```
Load a pretrained backbone
Freeze its parameters
Initialize a new task-specific head

Repeat for each training batch:
    Compute features using the backbone
    Compute predictions using the head
    Calculate the loss against the ground-truth labels
    Backpropagate the loss through the trainable computation
    Update the head parameters

Stop when the stopping criterion is met
```

The backbone produces fixed representations because its parameters are not updated. The head learns how to map those representations to the new task's outputs.

Strictly speaking, the backbone still performs a forward computation for every batch. The parameters are fixed, but the feature values vary with the input.

## (b) Why can the head learn useful predictions? (1 point)

The pretrained backbone may already encode useful visual information, such as shapes, textures, and object parts.

The new head learns the relationship between those existing features and the target labels.

For example, a feature vector may encode fur texture, ear shape, and other visual patterns. The head learns how those features help distinguish the target classes.

## (c) When could fine-tuning improve performance? (1 point)

Fine-tuning may help when the target dataset differs substantially from the pretraining dataset.

For example, a backbone pretrained on ordinary photographs may not extract sufficiently useful features for medical microscopy images. Updating some backbone layers can adapt the representations to the new domain.

Fine-tuning can also help when the target task requires subtle features that the pretrained model does not represent well.

### Important implementation nuance

Freezing backbone parameters is not the same as disabling all gradient computation throughout the model. If no backbone parameters need gradients and its input does not require gradients, the backbone forward pass can often run under `torch.no_grad()` to save memory.

However, the backbone should remain in the computational path so that its output features reach the head. In addition, BatchNorm running statistics may still change in training mode even when its learnable parameters are frozen, unless the module's behavior is controlled separately.

### Marking guide

| Criterion | Points |
| --- | --- |
| Describes feature extraction, prediction, loss, and updating the head | 2 |
| Explains the reuse of pretrained representations | 1 |
| Gives a valid reason for subsequent fine-tuning | 1 |
| Total | 4 |

# Question 7 — Interpreting a model architecture (4 points)

## (a) Functional roles of the backbone, neck, and head (2 points)

The backbone extracts hierarchical features from the input image. Earlier stages often preserve more spatial detail, while deeper stages typically provide more semantically meaningful features.

The neck combines or transforms backbone features. A feature pyramid network, for example, integrates information from multiple resolutions to produce feature maps useful for detecting objects at different scales.

The head uses these features to predict task-specific outputs, such as object classes and bounding-box coordinates.

The overall computation can be summarized as:

$$
\text{Image}
\rightarrow
\text{Backbone}
\rightarrow
\text{Neck}
\rightarrow
\text{Detection head}
\rightarrow
\text{Predictions}
$$

## (b) Why might removing the neck hurt small-object detection? (1 point)

Small objects occupy relatively few pixels, so detecting them often requires high-resolution features that preserve fine spatial detail.

A neck can combine high-resolution features from earlier backbone stages with semantically strong features from deeper stages. Removing it may deprive the detection head of this multiscale information.

Consequently, small objects can become harder to localize or distinguish from background regions.

## (c) Is the neck necessarily part of the backbone or head? (1 point)

No. A neck is commonly defined as a separate architectural component between the backbone and the head.

However, the boundary depends on the model's design and terminology. Some implementations group the neck with the feature extractor, while others group certain feature-processing layers with the prediction module.

The correct interpretation depends on the architecture's explicit component definitions and the functional role of the layers.

### Common trap

Do not assume that every operation before the final prediction layer belongs to the backbone. Modern vision models often have a separate neck, decoder, or other intermediate module.

# Question 8 — Reading a training procedure (5 points)

## (a) What does the algorithm do, and why select examples by loss? (2 points)

This procedure is a form of self-paced or loss-based curriculum learning.

At each training step, the model calculates the loss for each example. It selects examples whose loss falls below a threshold $\lambda$, then updates its parameters using the selected examples.

The intended intuition is that examples with low loss are currently easier for the model to handle. Training on these examples first may help the model learn relatively simple patterns before attempting more difficult examples.

A clear algorithm description is:

text
```
For each epoch:
    Calculate the prediction and loss for every example
    Select examples whose loss is below λ

    If at least one example is selected:
        Calculate the mean loss over selected examples
        Backpropagate the selected loss
        Update trainable parameters

Repeat
```

This can make training more manageable when examples vary in difficulty or label quality.

However, this is not automatically a better training strategy. The algorithm's behavior depends on the loss threshold, the quality of the labels, and how example difficulty relates to the target task.

## (b) What happens if $\lambda$ is too small or too large? (1 point)

If $\lambda$ is too small, very few examples may be selected. Training can become slow, unstable, or fail to make meaningful progress if no examples satisfy the condition.

If $\lambda$ is too large, nearly all examples may be selected. The method then behaves more like ordinary training and loses much of its selective-training effect.

A threshold that changes over time can gradually admit harder examples.

## (c) If the backbone is frozen, which parameters can be updated? (1 point)

Only the trainable parameters included in the optimization process can be updated. If the head is trainable and the backbone is frozen, the head's parameters are updated while the backbone's parameters remain fixed.

The selection procedure determines which examples contribute to the loss. It does not, by itself, determine which parameters are trainable.

## (d) Identify one potential failure mode (1 point)

One failure mode is confirmation bias toward easy examples.

If low-loss examples are disproportionately simple, redundant, or unrepresentative, the model may learn a narrow decision boundary and fail to generalize to difficult examples.

Another failure mode is excluding mislabeled examples in a useful way but also excluding valid, informative hard examples. A loss threshold cannot reliably distinguish label noise from legitimate task difficulty.

### Common trap

Loss is a measure of how well the current model fits an example; it is not a perfect measure of the example's intrinsic difficulty. A difficult but informative example can have a high loss, while a misleading or mislabeled example can sometimes have a low loss.

# Question 9 — Diagnosing a model implementation (5 points)

## (a) Problem A: The head produces 1,000 logits instead of 10 (2 points)

The likely mistake is that the original 1,000-class classification head remains in the model or is still being called during the forward pass.

The researcher should check the actual head module, its output dimension, and the forward pass to confirm that the replacement layer is being used.

If the backbone produces 512-dimensional features, the new linear head should have:

$$
W\in\mathbb{R}^{10\times512}
$$

and, for a batch of $B$ images, its output should be:

$$
\text{logits}\in\mathbb{R}^{B\times10}
$$

The output contains ten class scores for every image.

The backbone can still output 512 features; it is the head that must map those features to ten classes.

## (b) Problem B: The backbone parameters do not change (2 points)

There are several possible explanations.

Reason 1: The backbone parameters are frozen.

If its parameters have `requires_grad=False`, gradients will not be accumulated for those parameters. The researcher should inspect the `requires_grad` attribute for the backbone's parameters.

Reason 2: The optimizer does not include the backbone parameters.

The optimizer might have been initialized using only the head's parameters:

python
```
optimizer = torch.optim.Adam(head.parameters())
```

In this case, the optimizer will not update the backbone, even if the backbone's parameters have gradients.

For full-model fine-tuning, the optimizer must include the intended trainable parameters, for example:

python
```
optimizer = torch.optim.Adam(model.parameters())
```

This works as intended only if the relevant parameters are eligible for updates and receive suitable gradients.

Other possible causes include a missing backward pass, gradients that are zero or `None`, an incorrect computation graph, or an update that is too small to notice.

A good diagnosis checks both the gradients and whether the optimizer includes the relevant parameters.

## (c) Why are `backward()` and `step()` insufficient to guarantee that every parameter changes? (1 point)

`loss.backward()` computes gradients for eligible tensors connected to the loss through the recorded computation graph. It does not guarantee that every parameter receives a nonzero gradient.

`optimizer.step()` updates parameters managed by the optimizer according to their gradients and the optimizer's update rules.

A parameter may remain unchanged if it is frozen, absent from the optimizer, disconnected from the loss, or has a zero gradient. Some optimizer configurations may also leave a parameter unchanged in a particular step.

### Marking guide

| Criterion | Points |
| --- | --- |
| Diagnoses the wrong head and specifies the correct dimensions | 2 |
| Gives two plausible explanations for the missing backbone updates | 2 |
| Explains why gradient computation and optimizer updates do not guarantee universal parameter changes | 1 |
| Total | 5 |

### Common trap

Do not treat `backward()` as the operation that changes the weights. It computes gradients; the optimizer's `step()` applies parameter updates.

# Question 10 — Why can a pretrained backbone outperform training from scratch? (5 points)

## (a) Theoretical explanation (2 points)

Researcher B may perform better because the pretrained backbone has already learned useful representations from a large dataset.

Instead of learning all visual features from a small labelled dataset, Researcher B starts with features that may already represent edges, textures, shapes, and higher-level semantic patterns.

This offers two major advantages:

1. Better data efficiency: fewer labelled examples may be required to learn the target task.
2. A better initialization: optimization starts from a representation that is already useful, rather than from random weights.

Researcher A must learn both low-level and high-level representations from the limited target dataset. This can be harder and can increase the risk of overfitting.

Pretraining is not a guarantee of better results, but it often provides a strong starting point when the source and target tasks are sufficiently related.

## (b) Role of the new head (1 point)

The new head maps the pretrained feature representation to the target task's outputs.

For example, a backbone pretrained on general images may produce a feature vector of dimension 512. A new head can map this representation to the 20 classes of a flower-classification task.

The backbone provides reusable features, while the head learns how to interpret them for the new labels.

## (c) When might the pretrained backbone be less effective? (1 point)

It may be less effective when the target domain differs substantially from the source domain.

For example, a backbone pretrained on everyday photographs may not contain sufficiently useful representations for specialized medical scans, satellite imagery, or unusual industrial sensor data.

The source model may also lack the fine-grained features needed for the target task.

In such cases, fine-tuning, domain adaptation, or additional pretraining may help.

## (d) Freezing versus fine-tuning (1 point)

Freezing keeps the backbone's parameters fixed and trains the new head. It is computationally efficient and can reduce overfitting on small datasets, but it cannot adapt the backbone's features to the new task.

Fine-tuning updates some or all of the backbone parameters along with the head. It can improve performance when the target task needs different representations, but it requires more computation and may overfit or damage useful pretrained features if the updates are poorly controlled.

A practical compromise is to freeze the backbone initially, train the head, and then fine-tune selected backbone layers with a smaller learning rate.

### Marking guide

| Criterion | Points |
| --- | --- |
| Explains the benefits of learned representations and initialization | 2 |
| Explains the head's task-specific role | 1 |
| Gives a valid domain-shift or representation-mismatch scenario | 1 |
| Distinguishes freezing from fine-tuning and explains a trade-off | 1 |
| Total | 5 |

# Final revision sheet — What you should remember for the exam

## 1. The essential concepts

| Concept | Exam-ready explanation |
| --- | --- |
| Backbone | Extracts useful representations from raw inputs. |
| Head | Maps representations to task-specific predictions. |
| Neck | Combines or transforms features between backbone and head, when present. |
| Parameters | Learnable weights and biases. |
| Activations | Intermediate values computed during a forward pass. |
| Tensor shape | Describes the dimensions of a tensor, such as batch, channels, height, and width. |
| Forward pass | Computes predictions from input data. |
| Loss | Measures disagreement between predictions and targets or another training objective. |
| Backpropagation | Computes gradients of the loss with respect to trainable quantities. |
| Optimizer | Uses gradients to update eligible parameters. |
| Freezing | Prevents selected parameters from being updated through ordinary gradient-based training. |
| Fine-tuning | Continues training a pretrained model on a target task. |
| Transfer learning | Reuses knowledge learned on one task or dataset for another. |

## 2. The four equations worth understanding

Linear head

$$
z=Wx+b
$$

Maps a feature vector to output logits.

Global average pooling

$$
y_c=\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}X_{c,h,w}
$$

Reduces each channel's spatial feature map to one value.

Softmax

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Converts logits into normalized class probabilities.

Gradient-based update

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L
$$

Illustrates a basic gradient descent update. More sophisticated optimizers, such as Adam, use additional state and different update rules.

## 3. Five common exam traps

1. A layer type does not determine component ownership. A linear layer can appear in the backbone or head.
2. Frozen does not mean inactive. A frozen backbone still computes features.
3. `eval()` does not mean frozen. Evaluation mode and gradient tracking are different.
4. Gradients are not parameter updates. Backpropagation computes gradients; the optimizer applies updates.
5. High loss does not necessarily mean a bad example. It can indicate a difficult but informative example, label noise, or a mismatch between the model and the data.

## 4. A reliable approach to written questions

When you see a question about a new neural network, work through these steps:

1. Identify the input and the required output.
2. Trace the backbone's feature representation and its tensor shape.
3. Identify any intermediate processing, such as pooling, reshaping, or a neck.
4. Identify the head's output dimension and the meaning of its predictions.
5. Determine which parameters are trainable and which are frozen.
6. If training is involved, trace the loss, gradients, and optimizer updates.
7. Explain why the design helps the task and identify a limitation or failure mode.

This approach works across CNNs, Transformers, classification models, object detectors, and segmentation systems.

My recommendation for your next study session: practise Questions 4, 5, 8, and 9 without looking at the solutions. They combine architecture, mathematical reasoning, and implementation details—the kinds of connections that are harder to answer through memorization alone.
