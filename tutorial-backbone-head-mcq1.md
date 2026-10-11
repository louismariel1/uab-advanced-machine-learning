# Mastering Neural Network Backbones and Heads: Comprehensive MCQ

The goal of this exercise is to move beyond simply knowing that a backbone extracts features and a head makes predictions. You should be able to inspect an actual neural network, follow the tensors through its layers, understand the mathematical computations, identify the trainable objects, and determine which components belong to the backbone, the head, or neither.

We'll use image classification as our main example, then extend the concepts to object detection, segmentation, and Transformers.

## The mental model to keep throughout the quiz

## BACKBONE

Feature extraction and representation learning

Layers · kernels · attention · activations · normalization · feature tensors

Feature representation

A tensor passed from one component to the next

## HEAD

Task-specific prediction

Prediction layers · output scores · probabilities · boxes · masks

One crucial qualification: backbone and head are functional roles, not universal names for particular mathematical operations. A convolution, linear layer, activation function, or attention mechanism can belong to either, depending on the architecture and where it is used.

## How to use this MCQ

- 30 questions, organized from foundational concepts to advanced implementation details.
- Each question has four options, the correct answer, and an explanation.
- Try answering each question before revealing its answer.
- The final section includes integrated scenarios that test whether you can classify real components and follow the computations.

## Part 1 — Fundamental concepts and component ownership

Questions 1–6 · Beginner to intermediate

### Question 1. What fundamentally defines a backbone?

A. It is always the first five layers of a neural network.

B. It extracts or constructs useful representations from input data.

C. It converts all neural network outputs into probabilities.

D. It contains every trainable parameter in the model.

Reveal answer and explanation

### Question 2. What fundamentally defines a head?

A. It must be a single fully connected layer.

B. It always performs softmax.

C. It converts representations into outputs for a particular task.

D. It is responsible only for storing model parameters.

Reveal answer and explanation

### Question 3. Which component normally contains the convolutional layers in a standard ResNet image classifier?

A. Only the head.

B. The backbone.

C. The loss function.

D. The optimizer.

Reveal answer and explanation

### Question 4. Which statement about trainable parameters is correct?

A. Only the head has trainable parameters.

B. Only the backbone has trainable parameters.

C. Both can have trainable parameters, depending on the training strategy.

D. Neither contains trainable parameters because they are computational structures.

Reveal answer and explanation

### Question 5. What is the most reliable way to decide whether a layer belongs to the backbone or the head?

A. Check whether it uses matrix multiplication.

B. Check whether it is implemented in PyTorch or TensorFlow.

C. Examine its role in the architecture and the model's component boundaries.

D. Assume all layers after the first linear layer belong to the head.

Reveal answer and explanation

### Question 6. Which statement best distinguishes a layer, a parameter, and a tensor?

A. They all mean the same thing.

B. A layer performs a transformation, a parameter is a learnable value, and a tensor is a multidimensional data structure.

C. A tensor is always a learnable parameter.

D. A parameter is always the output of softmax.

Reveal answer and explanation

## Part 2 — Computations inside the backbone and head

Questions 7–12 · Mathematical understanding

### Question 7. What does a convolutional layer do inside a CNN backbone?

A. Directly chooses the final class label without learning features.

B. Applies learned filters to local regions of an input feature map.

C. Calculates the cross-entropy loss automatically.

D. Converts every feature into a probability.

Reveal answer and explanation

### Question 8. What does a typical linear classification head compute?

A. $y=Wx+b$

B. $y=x/W$

C. $y=\operatorname{argmax}(x)$ in every layer

D. $y=\text{cross-entropy}(x)$ during inference

Reveal answer and explanation

### Question 9. A backbone produces a feature vector with 512 values. The head must classify images into 10 classes. Which weight shape is appropriate for a standard linear head?

A. $512\times512$

B. $10\times10$

C. $10\times512$

D. $512\times10\times10$

Reveal answer and explanation Correct answer: C

Using the convention $y=Wx+b$, the weight matrix has one row per output class and one column per input feature.

$$
(10\times512)(512\times1)=10\times1
$$

For a batch of $N$ examples, the input can have shape $N\times512$, and the output will have shape $N\times10$.

### Question 10. Where does ReLU belong?

A. Exclusively in the backbone.

B. Exclusively in the head.

C. It can appear in either component.

D. ReLU is not a neural network operation.

Reveal answer and explanation Correct answer: C

ReLU computes $\max(0,x)$ elementwise. It is commonly used throughout a CNN backbone, and hidden layers in a prediction head may also use it.

The final output layer often uses a different transformation, or none, depending on the task.

### Question 11. What is the purpose of global average pooling in a typical CNN classifier?

A. It averages each channel's spatial feature map into a single value.

B. It averages the training loss across the dataset.

C. It updates convolution kernels.

D. It converts all logits to probabilities.

Reveal answer and explanation

### Question 12. Which statement about softmax is correct?

A. Softmax learns convolution kernels.

B. Softmax converts a vector of logits into normalized probabilities.

C. Softmax calculates gradients for every parameter.

D. Softmax is required in every neural network head.

Reveal answer and explanation Correct answer: B

For logits $z_i$, softmax computes

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

The probabilities sum to one across the classes. Softmax has no learned parameters. In many classification implementations, it is combined mathematically with the loss and not explicitly applied during training.

Regression heads and many binary or multilabel heads do not use multiclass softmax.

## Part 3 — Tensors, shapes, and object manipulation

Questions 13–18 · Implementation literacy

### Question 13. A CNN backbone outputs a tensor of shape $32\times256\times7\times7$. What does this normally represent in NCHW format?

A. 32 classes, 256 images, and a 7-by-7 kernel.

B. A batch of 32 examples, each with 256 channels and a 7-by-7 feature map.

C. 32 parameters, 256 layers, and 49 classes.

D. A batch of 32 examples with 256 classes and 7 output probabilities.

Reveal answer and explanation Correct answer: B

The dimensions are:

- $N=32$: batch size
- $C=256$: feature channels
- $H=7$: feature-map height
- $W=7$: feature-map width

These are intermediate features, not necessarily the final predictions.

### Question 14. Given the tensor in Question 13, what is the output shape after global average pooling over height and width?

A. $32\times256$

B. $32\times7\times7$

C. $256\times7\times7$

D. $32\times256\times7\times7$

Reveal answer and explanation Correct answer: A

Each of the 256 channels is reduced from 49 spatial values to one average. The batch dimension remains unchanged.

$$
(32,256,7,7)\rightarrow(32,256)
$$

This pooled representation is often passed to a classification head.

### Question 15. After pooling, a tensor has shape $32\times256$. A linear head produces 5 class logits. What is the output shape?

A. $32\times5$

B. $5\times256$

C. $32\times256\times5$

D. $256\times5$

Reveal answer and explanation Correct answer: A

The batch contains 32 examples and each example receives five scores:

$$
(32,256)\rightarrow(32,5)
$$

The head maps the feature dimension from 256 to 5, while preserving the batch dimension.

### Question 16. Which statement about `reshape` or `view` is correct?

A. Reshaping automatically learns new features.

B. Reshaping always creates a new trainable layer.

C. Reshaping changes how tensor dimensions are organized, not the underlying learned representation by itself.

D. Reshaping converts feature values into probabilities.

Reveal answer and explanation Correct answer: C

Reshape operations rearrange tensor dimensions while preserving the elements and their order. Depending on the framework, a reshape may return a view or require a copy.

For example, flattening a $256\times7\times7$ feature tensor yields a vector with $12{,}544$ elements. It does not itself learn anything.

### Question 17. Why might `flatten` appear between a CNN backbone and a linear head?

A. To turn multidimensional features into a vector compatible with a linear layer.

B. To freeze the backbone automatically.

C. To compute cross-entropy.

D. To replace the model's learned parameters.

Reveal answer and explanation Correct answer: A

Flattening commonly converts a feature tensor such as $N\times C\times H\times W$ into $N\times(CHW)$.

However, flattening is not always the best choice. Global average pooling can reduce the spatial dimensions while avoiding a very large fully connected layer.

### Question 18. In PyTorch, what is `model.parameters()` used for?

A. To return the model's predicted labels.

B. To retrieve parameter objects, typically for passing to an optimizer.

C. To compute all feature maps.

D. To change a model's output classes automatically.

Reveal answer and explanation Correct answer: B

`model.parameters()` yields the model's registered parameter objects, including weights and biases. An optimizer can update eligible parameters using gradients.

The method does not return intermediate activations, and it does not by itself indicate whether a parameter is frozen. Inspect `requires_grad` and the optimizer configuration as well.

## Part 4 — Training, gradients, losses, and optimization

Questions 19–23 · Understanding what actually learns

### Question 19. What is the main purpose of a loss function during training?

A. To extract spatial features from an image.

B. To measure how poorly predictions match the training targets.

C. To store the backbone weights.

D. To convert a tensor into a Python list.

Reveal answer and explanation Correct answer: B

The loss compares predictions with target labels or values. For classification, cross-entropy is common; for regression, mean squared error is one option.

The loss is part of the training objective, not inherently part of the backbone or the head. It evaluates the model's output and drives learning through backpropagation.

### Question 20. During backpropagation, what happens in an end-to-end trainable backbone and head?

A. Only the head receives gradients because it produces the output.

B. Gradients propagate backward through the head and into the backbone, allowing both to be updated if trainable.

C. The backbone is always updated before the head in a separate forward pass.

D. The optimizer calculates the loss from scratch without gradients.

Reveal answer and explanation

### Question 21. You freeze a pretrained backbone and train a new head. What is the intended effect?

A. The backbone stops producing features.

B. The backbone's parameters are not updated, but its forward computation still supplies features to the head.

C. The head becomes part of the loss function.

D. The backbone's output becomes a constant zero tensor.

Reveal answer and explanation

### Question 22. Which statement about `model.eval()` is correct in PyTorch?

A. It automatically freezes all model parameters.

B. It disables all gradient computations.

C. It changes the behavior of modules such as Dropout and BatchNorm, but does not itself disable gradients or freeze parameters.

D. It deletes the classification head.

Reveal answer and explanation Correct answer: C

Evaluation mode changes the behavior of certain modules. Dropout typically stops randomly dropping activations, while BatchNorm typically uses its running statistics.

It is separate from parameter freezing and gradient tracking. `eval()`, `requires_grad`, and `no_grad()` serve different purposes.

### Question 23. You replace a pretrained 1000-class head with a 10-class head. Which statement is most accurate?

A. All backbone parameters must be randomly reinitialized.

B. The new head must have an output dimension of 1000.

C. The new head must produce 10 class scores, and its parameters generally need training for the new task.

D. The loss function becomes unnecessary.

Reveal answer and explanation

## Part 5 — Architecture boundaries and advanced applications

Questions 24–27 · Beyond basic image classification

### Question 24. In an object-detection model, which statement is correct?

A. The backbone always predicts final bounding boxes directly.

B. The backbone extracts features, while task-specific detection components predict classes and box coordinates.

C. The head is always a single softmax layer.

D. Object detection does not require intermediate feature maps.

Reveal answer and explanation Correct answer: B

A detection system may contain a backbone, a feature pyramid or neck, and one or more detection heads. The head can output class scores and box coordinates, often for multiple candidate locations or proposals.

A neck is an additional feature-processing component. It is important not to force every architecture into a strict two-component design.

### Question 25. In semantic segmentation, what does a typical segmentation head produce?

A. One class label for the entire image only.

B. A class prediction for each pixel, usually represented by a spatial tensor of logits.

C. Only the backbone's convolution kernels.

D. A scalar learning rate.

Reveal answer and explanation Correct answer: B

A segmentation head commonly outputs logits shaped like $N\times K\times H\times W$, where $K$ is the number of segmentation classes.

The system may also use a neck or decoder to combine multiscale features and recover spatial detail. Softmax or another output transformation can then convert logits into predictions.

### Question 26. In a Transformer-based language model, which description is most appropriate?

A. The backbone must consist of CNN kernels.

B. Transformer blocks construct contextual token representations, while an output head can map hidden states to vocabulary logits.

C. The output head always produces image bounding boxes.

D. The backbone never contains attention operations.

Reveal answer and explanation

### Question 27. A model includes a backbone, a feature pyramid network, and detection heads. How should the feature pyramid network be understood?

A. It must be the loss function.

B. It is often called a neck: a component that transforms or combines backbone features before the heads use them.

C. It is always the final classification layer.

D. It cannot have trainable parameters.

Reveal answer and explanation Correct answer: B

A neck combines or refines features, often at different spatial resolutions. A feature pyramid network, for example, helps detection heads access both high-resolution spatial information and stronger semantic features.

The backbone-neck-head decomposition is common in modern computer vision. Some architectures merge these boundaries or use different names.

## Part 6 — Integrated scenarios and code reasoning

Questions 28–30 · Advanced mastery

### Question 28. Inspect this PyTorch model. Which components belong to the backbone and which belong to the head?

python
```
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.head = nn.Linear(32, 4)

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(1)
        logits = self.head(features)
        return logits
```

A. The convolutions are the head, and the linear layer is the backbone.

B. The convolutional layers are in the backbone, the linear layer is in the head, and flattening is a shape transformation between them.

C. The ReLU operations are trainable heads, and flattening is the optimizer.

D. Everything except `forward()` belongs to the loss function.

Reveal answer and explanation Correct answer: B

The explicitly named `backbone` module contains convolution, activation, and adaptive pooling operations. It outputs a tensor with 32 channels and spatial size $1\times1$.

`flatten(1)` changes its shape from $N\times32\times1\times1$ to $N\times32$. The head then maps 32 features to four class logits.

Notice that the code's names define the intended module boundaries. Another architecture could reasonably place pooling in a separate component.

### Question 29. Consider the following training setup:

python
```
for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(
    model.head.parameters(),
    lr=1e-3
)
```

What does this configuration do?

A. It freezes the backbone parameters and configures the optimizer to update the head parameters.

B. It freezes the head and trains the backbone.

C. It makes all model outputs constant.

D. It disables the forward pass through the backbone.

Reveal answer and explanation Correct answer: A

The backbone parameters have gradient tracking disabled, and the optimizer receives only the head's parameters.

During a normal training step, the backbone still computes features. The head can learn how to use those fixed features for the new task.

This is a common transfer-learning setup. In a real training loop, you would still need to calculate the predictions, evaluate the loss, call `loss.backward()`, and call `optimizer.step()`.

### Question 30. A researcher changes the number of classes from 1000 to 20, but the model still outputs 1000 logits. What is the most likely problem?

A. The backbone has too many convolutional filters.

B. The optimizer is using Adam instead of SGD.

C. The output layer or head has not been correctly changed to produce 20 outputs.

D. The input images are too small.

Reveal answer and explanation Correct answer: C

For ordinary single-label classification, the final classification layer should output 20 logits per example. If it still produces 1000, the old head likely remains in place, or another component is producing the original output dimension.

The backbone can often remain unchanged because its job is to generate features, not to determine the number of target classes.

## Final mastery checklist

Use this checklist when inspecting any neural network architecture.

## Self-assessment

0 of 8

[ ]I can distinguish layers, parameters, tensors, and intermediate activations.Know what is learned, what is computed, and what is stored.

[ ]I can trace tensor shapes from input through backbone to head.Follow batch, channel, spatial, feature, and output dimensions.

[ ]I understand convolution, attention, pooling, flattening, and linear projections.Identify the mathematical operation and its purpose.

[ ]I know that a computation's role—not its name alone—determines its component.The same type of layer can appear in either backbone or head.

[ ]I can explain logits, probabilities, loss, gradients, and optimizer updates.Separate forward computation from the training procedure.

[ ]I understand freezing, fine-tuning, and replacing a pretrained head.Know which parameters change and which features remain reusable.

[ ]I recognize backbone, neck, and head in detection and segmentation models.Understand that modern architectures may have multiple heads or extra feature-processing components.

[ ]I can read a model implementation and identify its component boundaries.Relate module definitions and forward-pass operations to their roles.

## The most important distinction to remember

| Backbone | Head |
| --- | --- |
| Learns or computes reusable representations | Converts representations into task-specific outputs |
| Often contains most of the feature-extraction layers | Often contains the task-specific prediction layers |
| Outputs intermediate features | Outputs logits, coordinates, masks, or other predictions |
| Can be frozen or fine-tuned | Can be replaced or retrained for a new task |
| May use convolutions, attention, normalization, pooling, and activations | May use linear layers, convolutions, attention, activations, and output transformations |

Neither column is a universal list of operations. The decisive question is: what role does this component play in the architecture?

If you can answer that question, trace its tensors, explain its parameters, and describe how gradients update it, you understand the backbone/head distinction at a practical engineering level.
