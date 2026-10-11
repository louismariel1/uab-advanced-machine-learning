# Mastering Neural Network Backbones and Heads — Complete Answer Key

Below are all 30 questions with the correct answer and explanation placed directly beneath each question. This version is ready to copy into a GitHub Markdown file, for example, `backbones_and_heads_mcq_solutions.md`.

# Part 1 — Fundamental Concepts and Component Ownership

## Question 1. What fundamentally defines a backbone?

A. It is always the first five layers of a neural network.

B. It extracts or constructs useful representations from input data.

C. It converts all neural network outputs into probabilities.

D. It contains every trainable parameter in the model.

Correct answer: B — It extracts or constructs useful representations from input data.

Explanation:

A backbone transforms raw inputs into useful intermediate representations, often called features or embeddings. In a vision model, it might transform an image into feature maps that encode edges, textures, shapes, and higher-level semantic information.

A backbone can contain convolutional layers, Transformer blocks, attention mechanisms, normalization layers, and activation functions.

The other options are incorrect because:

- A: A backbone is defined by its architectural role, not by a fixed number of layers.
- C: Converting logits into probabilities is an output transformation, commonly associated with a prediction head.
- D: A backbone may contain many trainable parameters, but it does not necessarily contain every trainable parameter in the model.

Key takeaway: The backbone's primary role is feature extraction and representation learning.

## Question 2. What fundamentally defines a head?

A. It must be a single fully connected layer.

B. It always performs softmax.

C. It converts representations into outputs for a particular task.

D. It is responsible only for storing model parameters.

Correct answer: C — It converts representations into outputs for a particular task.

Explanation:

A head maps the features produced by the backbone into task-specific predictions.

Examples include:

- Classification head: produces class logits.
- Object-detection head: predicts object classes and bounding-box coordinates.
- Segmentation head: produces pixel-level class scores.
- Regression head: predicts continuous values.
- Language-model output head: maps hidden states to vocabulary logits.

A head can contain one layer or many layers. It may use linear transformations, convolutions, attention, activation functions, or output transformations.

The other options are incorrect because a head does not have to be a single fully connected layer, does not always use softmax, and is not merely a parameter-storage object.

Key takeaway: A head is defined by its task-specific prediction role, not by a particular mathematical operation.

## Question 3. Which component normally contains the convolutional layers in a standard ResNet image classifier?

A. Only the head.

B. The backbone.

C. The loss function.

D. The optimizer.

Correct answer: B — The backbone.

Explanation:

In a typical ResNet image classifier, the backbone consists of convolutional layers and residual blocks that extract increasingly useful visual features.

The resulting features are pooled and passed to a final classification layer, which is typically the head.

The loss function evaluates predictions against targets, while the optimizer updates trainable parameters using gradients.

Key takeaway: Convolutional layers are commonly found in CNN backbones, although convolutions can also appear in prediction heads.

## Question 4. Which statement about trainable parameters is correct?

A. Only the head has trainable parameters.

B. Only the backbone has trainable parameters.

C. Both can have trainable parameters, depending on the training strategy.

D. Neither contains trainable parameters because they are computational structures.

Correct answer: C — Both can have trainable parameters, depending on the training strategy.

Explanation:

A backbone may contain learned convolution kernels, attention projection matrices, normalization parameters, and other trainable values. A head may contain learned weights and biases for mapping features to predictions.

The training strategy determines which parameters are updated:

- Feature extraction: the backbone is frozen, and the head is trained.
- Partial fine-tuning: selected backbone layers and the head are trained.
- End-to-end fine-tuning: both the backbone and head are trained.

A parameter can exist without being trainable during a particular phase of training.

Key takeaway: Parameter ownership and parameter trainability are different concepts.

## Question 5. What is the most reliable way to decide whether a layer belongs to the backbone or the head?

A. Check whether it uses matrix multiplication.

B. Check whether it is implemented in PyTorch or TensorFlow.

C. Examine its role in the architecture and the model's component boundaries.

D. Assume all layers after the first linear layer belong to the head.

Correct answer: C — Examine its role in the architecture and the model's component boundaries.

Explanation:

Backbone and head are functional architectural concepts, not universal mathematical categories.

For example, a linear layer may belong to a Transformer backbone because it processes hidden representations inside a Transformer block. A linear layer at the end of a classifier may belong to the head because it produces class logits.

Similarly, convolutions and activation functions can appear in either component.

The framework used to implement the model does not determine the component's role.

Key takeaway: Identify what a layer contributes to the model's computation rather than classifying it by operation type alone.

## Question 6. Which statement best distinguishes a layer, a parameter, and a tensor?

A. They all mean the same thing.

B. A layer performs a transformation, a parameter is a learnable value, and a tensor is a multidimensional data structure.

C. A tensor is always a learnable parameter.

D. A parameter is always the output of softmax.

Correct answer: B — A layer performs a transformation, a parameter is a learnable value, and a tensor is a multidimensional data structure.

Explanation:

Consider a convolutional layer.

- Layer: the computational module that applies convolution.
- Parameters: the learned convolution kernels and possibly biases.
- Tensor: the multidimensional input or output containing feature values.

A tensor can contain input data, activations, gradients, or parameters. Not every tensor is trainable.

For example, a convolutional layer might receive an input tensor with shape $N\times3\times224\times224$ and produce a feature tensor with shape $N\times64\times112\times112$.

Key takeaway: Layers perform operations, parameters store learnable values, and tensors hold multidimensional data.

# Part 2 — Computations Inside the Backbone and Head

## Question 7. What does a convolutional layer do inside a CNN backbone?

A. Directly chooses the final class label without learning features.

B. Applies learned filters to local regions of an input feature map.

C. Calculates the cross-entropy loss automatically.

D. Converts every feature into a probability.

Correct answer: B — Applies learned filters to local regions of an input feature map.

Explanation:

A convolutional layer applies learned kernels to local regions of its input. Each kernel produces a feature map that responds to particular patterns in the input.

Early layers may respond to edges and simple textures, while deeper layers can combine information into more complex patterns.

For a two-dimensional convolution, the operation can be represented schematically as

$$
Y_{n,o,h,w}
=
b_o+
\sum_{c,i,j}
W_{o,c,i,j}X_{n,c,h+i,w+j},
$$

with the precise input positions depending on stride, padding, and dilation.

The convolution does not automatically calculate cross-entropy or convert features into probabilities.

Key takeaway: Convolutions transform input feature maps using learned filters.

## Question 8. What does a typical linear classification head compute?

A. $y=Wx+b$

B. $y=x/W$

C. $y=\operatorname{argmax}(x)$ in every layer

D. $y=\text{cross-entropy}(x)$ during inference

Correct answer: A — $y=Wx+b$.

Explanation:

A linear layer computes a weighted combination of its input features and adds a bias.

For a feature vector $x\in\mathbb{R}^{d}$ and $K$ output classes,

$$
z=Wx+b,
$$

where

$$
W\in\mathbb{R}^{K\times d},
\qquad
b\in\mathbb{R}^{K}.
$$

The resulting vector $z\in\mathbb{R}^{K}$ contains the class logits.

These logits can be used directly by a classification loss or transformed into probabilities using softmax.

The other options describe neither the general computation of a linear layer nor its normal inference behavior.

Key takeaway: A linear classification head maps a feature vector into a vector of class scores.

## Question 9. A backbone produces a feature vector with 512 values. The head must classify images into 10 classes. Which weight shape is appropriate for a standard linear head?

A. $512\times512$

B. $10\times10$

C. $10\times512$

D. $512\times10\times10$

Correct answer: C — $10\times512$.

Explanation:

The input contains 512 features, and the output must contain 10 class logits.

Using the convention $y=Wx+b$,

$$
W\in\mathbb{R}^{10\times512}.
$$

For a single example, the matrix multiplication is

$$
(10\times512)(512\times1)=10\times1.
$$

For a batch of $N$ examples, the input has shape $N\times512$, and the output has shape $N\times10$.

The head therefore contains $10\times512=5{,}120$ weight parameters and 10 bias parameters, for a total of 5,130 trainable parameters if both weights and biases are enabled.

Key takeaway: The weight matrix's input dimension must match the backbone's feature dimension, and its output dimension must match the required number of logits.

## Question 10. Where does ReLU belong?

A. Exclusively in the backbone.

B. Exclusively in the head.

C. It can appear in either component.

D. ReLU is not a neural network operation.

Correct answer: C — It can appear in either component.

Explanation:

ReLU is an elementwise activation function:

$$
\operatorname{ReLU}(x)=\max(0,x).
$$

It is commonly used in CNN backbones to introduce nonlinearity. It can also appear in hidden layers of a multilayer prediction head.

A final classification layer often produces logits without applying ReLU, because logits should generally be allowed to take positive or negative values.

ReLU itself has no learned parameters.

Key takeaway: An activation function's location depends on the architecture, not on the activation function's identity.

## Question 11. What is the purpose of global average pooling in a typical CNN classifier?

A. It averages each channel's spatial feature map into a single value.

B. It averages the training loss across the dataset.

C. It updates convolution kernels.

D. It converts all logits to probabilities.

Correct answer: A — It averages each channel's spatial feature map into a single value.

Explanation:

Suppose the backbone produces a feature tensor with shape

$$
X\in\mathbb{R}^{N\times C\times H\times W}.
$$

Global average pooling computes one spatial average for each channel of each example:

$$
Y_{n,c}
=
\frac{1}{HW}
\sum_{h=1}^{H}\sum_{w=1}^{W}X_{n,c,h,w}.
$$

The resulting shape is

$$
(N,C,H,W)\rightarrow(N,C).
$$

For example,

$$
(32,256,7,7)\rightarrow(32,256).
$$

Standard global average pooling has no learned convolution kernels and does not calculate the training loss.

Key takeaway: Global average pooling reduces spatial dimensions while preserving the batch and channel dimensions.

## Question 12. Which statement about softmax is correct?

A. Softmax learns convolution kernels.

B. Softmax converts a vector of logits into normalized probabilities.

C. Softmax calculates gradients for every parameter.

D. Softmax is required in every neural network head.

Correct answer: B — Softmax converts a vector of logits into normalized probabilities.

Explanation:

Given logits $z_1,\ldots,z_K$, softmax computes

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}.
$$

Each probability is between zero and one, and the probabilities sum to one.

Softmax has no learned parameters. It is commonly used when classes are mutually exclusive, although many training implementations combine the softmax-related computation with the cross-entropy loss for numerical stability.

Softmax is not appropriate for every task. Regression heads may output numerical values directly, and multilabel classifiers commonly use independent sigmoid outputs.

Key takeaway: Softmax is an output transformation, not a learned feature-extraction operation.

# Part 3 — Tensors, Shapes, and Object Manipulation

## Question 13. A CNN backbone outputs a tensor of shape $32\times256\times7\times7$. What does this normally represent in NCHW format?

A. 32 classes, 256 images, and a 7-by-7 kernel.

B. A batch of 32 examples, each with 256 channels and a 7-by-7 feature map.

C. 32 parameters, 256 layers, and 49 classes.

D. A batch of 32 examples with 256 classes and 7 output probabilities.

Correct answer: B — A batch of 32 examples, each with 256 channels and a 7-by-7 feature map.

Explanation:

In NCHW format, the dimensions represent:

- $N=32$: batch size.
- $C=256$: number of channels.
- $H=7$: feature-map height.
- $W=7$: feature-map width.

Each example therefore has 256 feature maps, each containing $7\times7=49$ spatial values.

The tensor contains

$$
32\times256\times7\times7=401{,}408
$$

values.

These are intermediate feature activations, not necessarily class predictions.

Key takeaway: Always interpret tensor dimensions according to the model's tensor layout.

## Question 14. Given the tensor in Question 13, what is the output shape after global average pooling over height and width?

A. $32\times256$

B. $32\times7\times7$

C. $256\times7\times7$

D. $32\times256\times7\times7$

Correct answer: A — $32\times256$.

Explanation:

Global average pooling averages the 49 spatial values in each channel, producing one value per channel per example.

Therefore,

$$
(32,256,7,7)\rightarrow(32,256).
$$

The batch dimension remains 32, and the channel dimension remains 256. Only the spatial dimensions are reduced.

This representation is often passed to a linear classification head.

Key takeaway: Global average pooling removes the spatial dimensions, not the batch or channel dimensions.

## Question 15. After pooling, a tensor has shape $32\times256$. A linear head produces 5 class logits. What is the output shape?

A. $32\times5$

B. $5\times256$

C. $32\times256\times5$

D. $256\times5$

Correct answer: A — $32\times5$.

Explanation:

The input contains 32 feature vectors, each with 256 features. The head maps each feature vector to five class logits.

The weight matrix has shape

$$
W\in\mathbb{R}^{5\times256}.
$$

For the batch, the transformation is

$$
(32,256)(256,5)=(32,5).
$$

The batch dimension is preserved, and the output dimension becomes five.

Key takeaway: A classification head typically transforms the feature dimension into the number of output classes.

## Question 16. Which statement about `reshape` or `view` is correct?

A. Reshaping automatically learns new features.

B. Reshaping always creates a new trainable layer.

C. Reshaping changes how tensor dimensions are organized, not the underlying learned representation by itself.

D. Reshaping converts feature values into probabilities.

Correct answer: C — Reshaping changes how tensor dimensions are organized, not the underlying learned representation by itself.

Explanation:

Reshape operations change the dimensions used to organize tensor elements without performing a learned feature transformation.

For example, a tensor with shape

$$
(N,256,7,7)
$$

can be flattened to

$$
(N,12{,}544),
$$

because $256\times7\times7=12{,}544$.

Reshaping does not introduce learned weights. In PyTorch, `view()` generally requires compatible memory layout, whereas `reshape()` may return a view or create a copy when necessary.

Both preserve the tensor's values and element ordering.

Key takeaway: Reshaping changes tensor organization, not the learned feature values themselves.

## Question 17. Why might `flatten` appear between a CNN backbone and a linear head?

A. To turn multidimensional features into a vector compatible with a linear layer.

B. To freeze the backbone automatically.

C. To compute cross-entropy.

D. To replace the model's learned parameters.

Correct answer: A — To turn multidimensional features into a vector compatible with a linear layer.

Explanation:

A CNN often outputs a tensor with shape

$$
(N,C,H,W).
$$

A flatten operation can convert this to

$$
(N,CHW),
$$

allowing a standard linear layer to process the resulting feature vectors.

For example,

$$
(N,256,7,7)\rightarrow(N,12{,}544).
$$

An alternative is global average pooling, which produces only 256 features per example. This can substantially reduce the size of the subsequent linear layer.

Key takeaway: Flattening prepares multidimensional features for a vector-based operation; it does not itself learn new features.

## Question 18. In PyTorch, what is `model.parameters()` used for?

A. To return the model's predicted labels.

B. To retrieve parameter objects, typically for passing to an optimizer.

C. To compute all feature maps.

D. To change a model's output classes automatically.

Correct answer: B — To retrieve parameter objects, typically for passing to an optimizer.

Explanation:

In PyTorch, `model.parameters()` yields the registered parameter objects associated with the model and its submodules.

These commonly include layer weights and biases. An optimizer can use them to update eligible parameters during training.

For example:

python
```
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
```

This configures the optimizer with the model's registered parameters. However, a parameter's presence in this collection does not guarantee that it will change: it may be frozen, lack a gradient, or receive a zero update.

To inspect trainability, check `parameter.requires_grad`.

Key takeaway: `model.parameters()` exposes parameter objects; it does not return predictions or intermediate feature tensors.

# Part 4 — Training, Gradients, Losses, and Optimization

## Question 19. What is the main purpose of a loss function during training?

A. To extract spatial features from an image.

B. To measure how poorly predictions match the training targets.

C. To store the backbone weights.

D. To convert a tensor into a Python list.

Correct answer: B — To measure how poorly predictions match the training targets.

Explanation:

A loss function quantifies the discrepancy between a model's predictions and its targets, or more generally evaluates a training objective.

For classification, cross-entropy is commonly used. For regression, mean squared error or other regression losses may be appropriate.

The training procedure uses the loss to calculate gradients that guide parameter updates.

The loss function is conceptually separate from the backbone and head, although it operates on their outputs and influences how their parameters are trained.

Key takeaway: The loss defines the training objective; it is not a feature-extraction layer.

## Question 20. During backpropagation, what happens in an end-to-end trainable backbone and head?

A. Only the head receives gradients because it produces the output.

B. Gradients propagate backward through the head and into the backbone, allowing both to be updated if trainable.

C. The backbone is always updated before the head in a separate forward pass.

D. The optimizer calculates the loss from scratch without gradients.

Correct answer: B — Gradients propagate backward through the head and into the backbone, allowing both to be updated if trainable.

Explanation:

During a normal training step, the model performs a forward pass:

$$
x\rightarrow\text{backbone}\rightarrow
\text{head}\rightarrow\hat y.
$$

The loss is calculated from the predictions $\hat y$ and the targets.

Backpropagation then applies the chain rule to calculate gradients. The loss gradient propagates through the head and, when the computation graph permits it, into the backbone.

For a backbone parameter $\theta_b$ and head parameter $\theta_h$, the gradients can be written as

$$
\frac{\partial L}{\partial\theta_b},
\qquad
\frac{\partial L}{\partial\theta_h}.
$$

An optimizer can update both sets of parameters if they are trainable, included in the optimizer, and receive suitable gradients.

Key takeaway: Backpropagation computes gradients through the connected computation graph; the optimizer uses those gradients to update parameters.

## Question 21. You freeze a pretrained backbone and train a new head. What is the intended effect?

A. The backbone stops producing features.

B. The backbone's parameters are not updated, but its forward computation still supplies features to the head.

C. The head becomes part of the loss function.

D. The backbone's output becomes a constant zero tensor.

Correct answer: B — The backbone's parameters are not updated, but its forward computation still supplies features to the head.

Explanation:

Freezing a backbone means keeping its parameters fixed during the relevant training phase.

The backbone still processes each input image and generates features. The head learns how to map those features to the target outputs.

This is useful in transfer learning, particularly when labelled data is limited or the pretrained representation is already effective.

Freezing does not mean the backbone's output is identical for every input. Different inputs can produce different feature representations even when the parameters remain unchanged.

Key takeaway: Freezing prevents parameter updates; it does not disable forward computation.

## Question 22. Which statement about `model.eval()` is correct in PyTorch?

A. It automatically freezes all model parameters.

B. It disables all gradient computations.

C. It changes the behavior of modules such as Dropout and BatchNorm, but does not itself disable gradients or freeze parameters.

D. It deletes the classification head.

Correct answer: C — It changes the behavior of modules such as Dropout and BatchNorm, but does not itself disable gradients or freeze parameters.

Explanation:

PyTorch distinguishes several operations:

- `model.eval()` switches modules into evaluation mode.
- `parameter.requires_grad = False` prevents gradients from being calculated for that parameter through ordinary autograd.
- `torch.no_grad()` prevents autograd from recording operations executed within its context.
- `optimizer.step()` applies optimizer updates to eligible parameters.

For example, Dropout typically stops randomly dropping activations in evaluation mode. BatchNorm typically uses its stored running statistics instead of updating them from the current batch.

None of this means that `eval()` automatically freezes all parameters.

Key takeaway: Evaluation mode, gradient tracking, parameter freezing, and optimizer updates are separate mechanisms.

## Question 23. You replace a pretrained 1000-class head with a 10-class head. Which statement is most accurate?

A. All backbone parameters must be randomly reinitialized.

B. The new head must have an output dimension of 1000.

C. The new head must produce 10 class scores, and its parameters generally need training for the new task.

D. The loss function becomes unnecessary.

Correct answer: C — The new head must produce 10 class scores, and its parameters generally need training for the new task.

Explanation:

The backbone can often be reused because its role is to extract representations rather than determine the target class count.

If the backbone produces 512-dimensional features, the new head can use a weight matrix

$$
W\in\mathbb{R}^{10\times512}.
$$

The output for a batch of $N$ examples is

$$
(N,512)\rightarrow(N,10).
$$

The new head's parameters are often randomly initialized and trained on the target dataset. The backbone can initially remain frozen or be fine-tuned later.

Key takeaway: Changing the task's number of classes usually requires adapting the head, not reinitializing the entire backbone.

# Part 5 — Architecture Boundaries and Advanced Applications

## Question 24. In an object-detection model, which statement is correct?

A. The backbone always predicts final bounding boxes directly.

B. The backbone extracts features, while task-specific detection components predict classes and box coordinates.

C. The head is always a single softmax layer.

D. Object detection does not require intermediate feature maps.

Correct answer: B — The backbone extracts features, while task-specific detection components predict classes and box coordinates.

Explanation:

A typical object-detection architecture may contain:

1. Backbone: extracts visual representations from the input image.
2. Neck: combines features from different stages or spatial resolutions.
3. Detection head: predicts class scores and bounding-box coordinates.

A detection head may produce predictions at multiple spatial positions or process object proposals. Its output can be considerably more complex than a single classification vector.

The backbone's primary role is feature extraction, although the precise architecture and component boundaries vary.

Key takeaway: Object detection commonly requires task-specific prediction components that operate on backbone features, often through a neck.

## Question 25. In semantic segmentation, what does a typical segmentation head produce?

A. One class label for the entire image only.

B. A class prediction for each pixel, usually represented by a spatial tensor of logits.

C. Only the backbone's convolution kernels.

D. A scalar learning rate.

Correct answer: B — A class prediction for each pixel, usually represented by a spatial tensor of logits.

Explanation:

Semantic segmentation assigns a semantic class to each pixel.

A segmentation head may produce a logits tensor with shape

$$
(N,K,H,W),
$$

where:

- $N$ is the batch size.
- $K$ is the number of segmentation classes.
- $H,W$ are the output spatial dimensions.

For example, a tensor with shape $(4,6,224,224)$ represents four images and six class logits at each of $224\times224$ spatial locations.

The logits may be converted into pixel-level class predictions. Some architectures use a decoder or neck to combine features and recover spatial detail.

Key takeaway: A segmentation head generally produces spatially organized predictions rather than one class label per image.

## Question 26. In a Transformer-based language model, which description is most appropriate?

A. The backbone must consist of CNN kernels.

B. Transformer blocks construct contextual token representations, while an output head can map hidden states to vocabulary logits.

C. The output head always produces image bounding boxes.

D. The backbone never contains attention operations.

Correct answer: B — Transformer blocks construct contextual token representations, while an output head can map hidden states to vocabulary logits.

Explanation:

A Transformer backbone commonly contains self-attention blocks, feed-forward networks, normalization layers, and residual connections.

These operations transform token representations into contextual hidden states.

For a language model, an output head can map each hidden state to scores over the vocabulary:

$$
z_t=Wh_t+b,
$$

where $h_t$ is the hidden representation for token position $t$, and $z_t$ contains vocabulary logits.

If the vocabulary contains $V$ tokens and the hidden dimension is $d$, the output projection commonly has weight shape $V\times d$.

Some language models share weights between their input token embeddings and output projection, so the head does not necessarily introduce an entirely separate set of weights.

Key takeaway: Backbones and heads apply across architectures; the operations they contain depend on the task and model design.

## Question 27. A model includes a backbone, a feature pyramid network, and detection heads. How should the feature pyramid network be understood?

A. It must be the loss function.

B. It is often called a neck: a component that transforms or combines backbone features before the heads use them.

C. It is always the final classification layer.

D. It cannot have trainable parameters.

Correct answer: B — It is often called a neck: a component that transforms or combines backbone features before the heads use them.

Explanation:

A feature pyramid network (FPN) combines features from multiple backbone stages, often at different spatial resolutions.

Shallow features may retain detailed spatial information, while deeper features may contain stronger semantic information. The FPN combines these complementary properties into feature maps that are useful for detection.

The FPN may contain trainable convolutional layers and other operations. It is not a loss function, and it is not necessarily a final prediction layer.

Key takeaway: The neck is an intermediate feature-processing component, commonly used to improve the features available to task-specific heads.

# Part 6 — Integrated Scenarios and Code Reasoning

## Question 28. Inspect this PyTorch model. Which components belong to the backbone and which belong to the head?

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

Correct answer: B — The convolutional layers are in the backbone, the linear layer is in the head, and flattening is a shape transformation between them.

Explanation:

The code explicitly defines the backbone as a sequence containing two convolutional layers, two ReLU activations, and adaptive average pooling.

For an input image batch with shape

$$
(N,3,H,W),
$$

the convolutions preserve the spatial dimensions because their kernel size is 3 and their padding is 1, while the number of channels changes from 3 to 16 and then from 16 to 32.

Adaptive average pooling reduces the spatial dimensions to $1\times1$:

$$
(N,3,H,W)
\rightarrow(N,16,H,W)
\rightarrow(N,32,H,W)
\rightarrow(N,32,1,1).
$$

The `flatten(1)` operation then produces

$$
(N,32,1,1)\rightarrow(N,32).
$$

Finally, the linear head maps 32 features to four logits:

$$
(N,32)\rightarrow(N,4).
$$

The flattening operation does not contain trainable parameters. It changes the tensor's shape to make it compatible with the linear layer.

Key takeaway: Trace the forward pass to identify how the backbone's feature tensor becomes the head's prediction tensor.

## Question 29. Consider the following training setup:

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

Correct answer: A — It freezes the backbone parameters and configures the optimizer to update the head parameters.

Explanation:

The first block iterates over the backbone's registered parameters and sets `requires_grad=False`. These parameters will not receive gradients through ordinary autograd.

The optimizer is initialized with the head's parameters, so its intended updates are restricted to the head.

During training, the backbone still performs a forward pass and generates features. Those features are used by the head to make predictions.

A typical training step would still require:

python
```
optimizer.zero_grad()

logits = model(images)
loss = criterion(logits, labels)

loss.backward()
optimizer.step()
```

This assumes the model's forward pass connects the head to the loss and that the head has trainable parameters.

One additional nuance is that freezing the backbone's parameters does not automatically put BatchNorm layers into evaluation mode. If their running statistics must remain fixed, the relevant module behavior needs to be managed separately.

Key takeaway: Freezing controls parameter gradients; optimizer configuration controls which parameters are eligible for updates.

## Question 30. A researcher changes the number of classes from 1000 to 20, but the model still outputs 1000 logits. What is the most likely problem?

A. The backbone has too many convolutional filters.

B. The optimizer is using Adam instead of SGD.

C. The output layer or head has not been correctly changed to produce 20 outputs.

D. The input images are too small.

Correct answer: C — The output layer or head has not been correctly changed to produce 20 outputs.

Explanation:

For ordinary single-label classification, a head that predicts 20 classes must output 20 logits per example.

If the model still outputs 1,000 logits, the original classification layer may still be present, the new head may not be used in the forward pass, or another component may be producing the original output dimension.

Suppose the backbone produces 512-dimensional feature vectors. The new linear head should use

$$
W\in\mathbb{R}^{20\times512},
\qquad
b\in\mathbb{R}^{20}.
$$

The output shape for a batch of $N$ examples should be

$$
(N,512)\rightarrow(N,20).
$$

The optimizer choice and image size do not ordinarily determine the output layer's number of classes.

Key takeaway: The head's output dimension must match the target task's required output dimension.

# Final Mastery Checklist

Use this checklist to assess whether you understand the concepts beyond memorizing the correct options.

- [ ]I can distinguish layers, parameters, tensors, and intermediate activations.
- [ ]I can trace tensor shapes from the input through the backbone and head.
- [ ]I understand convolution, attention, pooling, flattening, and linear projections.
- [ ]I know that a layer's function—not just its operation type—determines its architectural role.
- [ ]I can explain logits, probabilities, losses, gradients, and optimizer updates.
- [ ]I understand freezing, fine-tuning, and replacing a pretrained head.
- [ ]I can identify backbones, necks, and heads in detection and segmentation models.
- [ ]I can inspect a PyTorch model and determine which parameters are trainable.

## The most important distinction

| Backbone | Head |
| --- | --- |
| Extracts or constructs useful representations | Maps representations to task-specific outputs |
| Often contains feature-extraction layers | Often contains prediction layers |
| Produces intermediate features | Produces logits, coordinates, masks, or other predictions |
| Can be frozen or fine-tuned | Can be replaced or retrained for a new task |
| May use convolutions, attention, normalization, pooling, and activations | May also use convolutions, attention, normalization, pooling, and activations |

The decisive question is not whether a component is a convolution, linear layer, or activation function. It is what role that component plays in the architecture.

Once you can identify that role, follow the tensor shapes, explain the computations, and determine how the parameters are updated, you have developed a practical understanding of neural network backbones and heads.
