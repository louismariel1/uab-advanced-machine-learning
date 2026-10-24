# Advanced Machine Learning — Practical 4: Low-Rank Adaptation (LoRA)

Comprehensive multiple-choice question bank covering the entire notebook, from dataset preparation and transfer learning to LoRA mathematics, implementation, training, evaluation, performance measurement, and the optional extension exercises.

The questions below are based on the notebook you supplied. They include conceptual questions, code interpretation, mathematical derivations, debugging scenarios, and exam-style applications. Correct answers and explanations are provided for every question.

Important notebook-specific details:

- The target superclass is `fish`, containing five CIFAR-100 classes.
- The convolutional layers are frozen in both the dense fine-tuning baseline and the LoRA model.
- The LoRA implementation is incomplete in the supplied notebook, so implementation questions use the mathematically correct intended solution.
- Actual training accuracies, runtimes, memory usage, and checkpoint sizes must be measured; they are not assumed here.

## Section 1 — Assignment objectives and transfer learning

### Q1. What is the primary objective of Practical 4?

A. Train a neural network from scratch using CIFAR-100.

B. Replace convolutional layers with transformer attention.

C. Reduce the computation and number of trainable parameters required for fine-tuning a pretrained model.

D. Eliminate the need for labelled target data.

Correct answer: C

Explanation: The practical investigates parameter-efficient fine-tuning. It compares dense fine-tuning with Low-Rank Adaptation (LoRA), aiming to adapt a pretrained model while updating substantially fewer parameters.

### Q2. What is transfer learning in this practical?

A. Transferring model weights from a pretrained source-domain task to a target-domain task.

B. Transferring images between CIFAR-10 and CIFAR-100.

C. Copying validation labels into the training set.

D. Replacing the optimizer after every epoch.

Correct answer: A

Explanation: The model is pretrained on CIFAR-10 and then adapted to classify a subset of CIFAR-100. The pretrained weights provide useful representations that can be reused for the new task.

### Q3. Which dataset is used for pretraining?

A. MNIST

B. CIFAR-100

C. ImageNet

D. CIFAR-10

Correct answer: D

Explanation: CIFAR-10 is the source dataset, containing 10 classes. The model learns an initial representation from this dataset before being adapted to the target task.

### Q4. Which target superclass is selected by the supplied notebook?

A. Insects

B. Fish

C. Vehicles

D. Flowers

Correct answer: B

Explanation: The notebook defines:

python

```
TARGET_CATEGORY = 'fish'
```

The selected superclass contains five classes: `aquarium_fish`, `flatfish`, `ray`, `shark`, and `trout`.

### Q5. How many target classes does the selected superclass contain?

A. 2

B. 10

C. 5

D. 100

Correct answer: C

Explanation: Each of CIFAR-100's 20 coarse superclasses contains five fine-grained classes. The `fish` superclass therefore creates a five-class classification task.

### Q6. Why is the classifier head reinitialized for target-domain fine-tuning?

A. The pretrained classifier contains no trainable parameters.

B. CIFAR-10 and CIFAR-100 use different image dimensions.

C. The convolutional layers cannot process CIFAR-100 images.

D. The source classifier outputs 10 classes, while the target classifier must output 5.

Correct answer: D

Explanation: The output dimension must match the number of target classes. The pretrained feature-extraction layers can be reused, but the final classifier needs a different output size and is initialized with new weights.

### Q7. Which sequence best describes the notebook's main experimental workflow?

A. Pretrain on CIFAR-10, perform dense target fine-tuning, then perform LoRA target fine-tuning and compare results.

B. Train LoRA on CIFAR-100, then pretrain on CIFAR-10.

C. Train both models from scratch and compare their architectures.

D. Train only the classifier and discard the pretrained model.

Correct answer: A

Explanation: The notebook first trains a source-domain model, saves its checkpoint, establishes a dense fine-tuning baseline, and finally evaluates LoRA as a parameter-efficient alternative.

## Section 2 — Dataset preparation and preprocessing

### Q8. What is the purpose of `download_from_mirror()`?

A. Download pretrained model weights from PyTorch.

B. Download a dataset archive if necessary and extract it into the data directory.

C. Convert all images to grayscale.

D. Create the train-validation split.

Correct answer: B

Explanation: The function checks whether the archive and extracted directory already exist. It downloads the archive using `gdown` if needed and extracts it if the dataset directory is absent.

### Q9. Why does the notebook use `torch.utils.data.Subset` to create the target dataset?

A. To resize images to different dimensions.

B. To freeze selected model layers.

C. To select only examples belonging to the chosen superclass.

D. To convert the dataset into a NumPy array.

Correct answer: C

Explanation: `subset_idxs` contains the original CIFAR-100 indices whose labels belong to the selected superclass. `Subset` restricts the target dataset to those examples without copying the underlying image data.

### Q10. What is the purpose of `target_data.remap`?

A. To map the original CIFAR-100 class indices to consecutive target labels from 0 to 4.

B. To map image pixels to normalized values.

C. To map the model's layers to CUDA.

D. To convert class names into RGB values.

Correct answer: A

Explanation: The original CIFAR-100 class indices are not necessarily consecutive within the selected superclass. Remapping ensures that the five target classes use labels compatible with a five-output classifier.

For example, the target labels must be valid integers in the range $0,\ldots,4$.

### Q11. Which transformation is applied to each image?

A. Random horizontal flipping only.

B. Conversion to grayscale followed by resizing.

C. Standardization using the ImageNet mean and standard deviation.

D. Conversion to a tensor followed by channel-wise normalization using mean 0.5 and standard deviation 0.2.

Correct answer: D

Explanation: The notebook defines:

python

```
pre_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.2, 0.2, 0.2)
    )
])
```

`ToTensor()` converts image data to a tensor, typically scaling pixel values to $[0,1]$. Normalization then applies

$$
x_{\text{norm}}=\frac{x-0.5}{0.2}
$$

independently to each RGB channel.

### Q12. What fraction of each dataset is allocated to validation?

A. 10%

B. 20%

C. 25%

D. 50%

Correct answer: B

Explanation: The notebook sets `val_frac = 0.2`. Thus, 80% is allocated to training and 20% to validation. The split is performed separately for the source and target datasets.

### Q13. What is the purpose of the custom `collate_target()` function?

A. To remove target images from the dataset.

B. To generate random target predictions.

C. To batch images, apply preprocessing, remap labels, and move tensors to the selected device.

D. To calculate the validation accuracy before training.

Correct answer: C

Explanation: The collate function prepares a batch of images and labels. Crucially, it applies the target label remapping before constructing the label tensor, ensuring that the labels match the five-class output head.

### Q14. Why can `shuffle=True` be useful for a training DataLoader?

A. It randomizes the order in which training examples are presented, reducing dependence on a fixed ordering.

B. It automatically balances all classes perfectly.

C. It increases the number of training images.

D. It guarantees better validation accuracy.

Correct answer: A

Explanation: Shuffling changes the order of training examples between epochs. It can improve optimization behavior, but it does not guarantee class balance or better validation performance.

The notebook also sets `shuffle=True` for validation. That is unnecessary because evaluation does not require shuffled examples, although it does not inherently invalidate aggregate evaluation metrics.

## Section 3 — Model architecture and parameter counting

### Q15. Which description correctly summarizes `ConvMLP`?

A. Four convolutional layers followed by a transformer.

B. An embedding layer followed by recurrent layers.

C. A fully connected network with no image-processing layers.

D. Two convolutional layers, pooling, four hidden fully connected layers, and a classifier.

Correct answer: D

Explanation: The architecture first extracts features using convolutional layers, then applies a multilayer perceptron (MLP). The MLP contains `fc1`, `fc2`, `fc3`, and `fc4`, followed by the classification head.

### Q16. What is the output shape of `conv1` before max pooling for a batch of 256 CIFAR images?

A. $(256, 3, 32, 32)$

B. $(256, 64, 32, 32)$

C. $(256, 64, 16, 16)$

D. $(256, 256, 8, 8)$

Correct answer: B

Explanation: `conv1` uses 64 output channels, a $3\times3$ kernel, and padding of 1. These settings preserve the spatial dimensions of the $32\times32$ input.

The max-pooling layer subsequently halves the spatial dimensions to $16\times16$.

### Q17. What does `nn.AdaptiveAvgPool2d((1,1))` accomplish?

A. It reduces the number of output classes to one.

B. It converts the feature map into a $1\times1$ spatial representation per channel.

C. It averages the model's predictions across the batch.

D. It freezes the convolutional weights.

Correct answer: B

Explanation: After the second convolution, adaptive average pooling produces a tensor of shape

$$
(B,256,1,1)
$$

where $B$ is the batch size. Reshaping then gives a 256-dimensional feature vector per image, which is passed to the MLP.

### Q18. How many trainable parameters are in `fc1` before freezing or LoRA adaptation?

The layer maps 256 inputs to 1024 outputs and includes a bias.

A. 262,144

B. 263,168

C. 264,192

D. 1,049,600

Correct answer: B

Explanation: A fully connected layer with input dimension $d_{\text{in}}$ and output dimension $d_{\text{out}}$ has

$$
N=d_{\text{in}}d_{\text{out}}+d_{\text{out}}.
$$

For `fc1`:

$$
N=256(1024)+1024=263,168.
$$

The additional 1,024 parameters are the bias vector.

### Q19. Why is the classifier head excluded when loading the pretrained checkpoint into the target model?

A. The target classifier has a different output dimension, so its weights and biases cannot be loaded directly.

B. The classifier is a convolutional layer.

C. The classifier has no bias parameters.

D. PyTorch cannot load fully connected layers from checkpoints.

Correct answer: A

Explanation: The source classifier maps 256 features to 10 classes, whereas the target classifier maps 256 features to 5 classes. Their weight shapes differ, so the source classifier parameters are deleted from the checkpoint before loading.

### Q20. What does `requires_grad_(False)` do to a model layer?

A. It deletes the layer's parameters.

B. It sets all weights to zero.

C. It prevents gradients from being calculated for the layer's parameters during ordinary autograd backpropagation.

D. It disables the layer's forward pass.

Correct answer: C

Explanation: Freezing a layer prevents its parameters from being updated by gradient-based optimization. The layer still participates in the forward computation. Its output can still be used by downstream trainable layers.

### Q21. Which statement about the dense fine-tuning baseline is correct?

A. Every model parameter is frozen.

B. Only the convolutional layers are trainable.

C. The entire model, including the convolutional layers, is retrained.

D. The convolutional layers are frozen, while the MLP layers and the new classifier are trainable.

Correct answer: D

Explanation: Both `conv1` and `conv2` are frozen. The fully connected layers `fc1`–`fc4` and the target classifier remain trainable. This is a partial fine-tuning baseline rather than completely unrestricted full-model fine-tuning.

## Section 4 — Training, optimization, and evaluation

### Q22. Which optimizer and loss function are used in the notebook?

A. SGD and mean squared error.

B. AdamW and cross-entropy loss.

C. RMSprop and binary cross-entropy.

D. Adam and triplet loss.

Correct answer: B

Explanation: The training function uses:

python

```
loss_func = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=l2_reg
)
```

Cross-entropy is appropriate for multiclass classification with integer class labels and raw class logits.

### Q23. Why should the model output raw logits rather than applying softmax before `CrossEntropyLoss`?

A. Cross-entropy loss expects raw logits and internally performs the equivalent of log-softmax followed by negative log-likelihood.

B. Softmax is incompatible with classification.

C. Raw logits must always be integers.

D. Softmax can only be used during validation.

Correct answer: A

Explanation: `nn.CrossEntropyLoss()` expects unnormalized scores. Applying softmax first is unnecessary and changes the numerical behavior of the loss computation.

The predicted class is obtained using `argmax`, which can be applied directly to logits.

### Q24. What does `opt.zero_grad()` do?

A. Resets all model weights to zero.

B. Deletes the optimizer.

C. Clears previously accumulated parameter gradients before the next backward pass.

D. Disables gradient computation.

Correct answer: C

Explanation: PyTorch accumulates gradients by default. Clearing them before each training step ensures that the optimizer normally uses the gradients from the current batch rather than their unintended accumulation across batches.

### Q25. What is the purpose of `batch_loss.backward()`?

A. It updates model parameters immediately without an optimizer.

B. It computes the validation accuracy.

C. It freezes parameters that have large gradients.

D. It calculates gradients of the loss with respect to the trainable parameters.

Correct answer: D

Explanation: Backpropagation computes derivatives using autograd. The subsequent `opt.step()` applies the optimizer's parameter updates using the gradients that were calculated.

### Q26. Why does `evaluate_model()` use both `model.eval()` and `torch.no_grad()`?

A. To make all parameters trainable during validation.

B. To use evaluation behavior for layers that distinguish training from evaluation and avoid recording gradients.

C. To apply a new optimizer during validation.

D. To reset the model's learned weights.

Correct answer: B

Explanation: `model.eval()` switches modules such as dropout and batch normalization into evaluation mode. `torch.no_grad()` disables gradient tracking, reducing unnecessary memory and computation during validation.

The supplied architecture does not contain dropout or batch normalization, but the evaluation pattern remains appropriate.

### Q27. How is top-1 accuracy calculated by `top1_acc()`?

A. It compares the highest-scoring predicted class with the ground-truth class for each example and averages the matches.

B. It averages the raw logits.

C. It calculates the percentage of positive weights.

D. It measures the difference between training and validation loss.

Correct answer: A

Explanation: The function computes:

python

```
(pred.argmax(axis=1) == y).float().mean().item()
```

The result is a scalar between 0 and 1. For example, an accuracy of `0.80` corresponds to 80%.

### Q28. Which statement about the learning rates in the notebook is correct?

A. Pretraining and fine-tuning both use $2\times10^{-3}$.

B. Pretraining uses $10^{-4}$, while fine-tuning uses $2\times10^{-3}$.

C. Pretraining uses $2\times10^{-3}$, while dense fine-tuning and LoRA use $10^{-4}$.

D. Neither stage uses a learning rate.

Correct answer: C

Explanation: The notebook uses:

- Pretraining: `lr = 2e-3`.
- Dense fine-tuning: `lr = 1e-4`.
- LoRA fine-tuning: the same `lr = 1e-4`.

The lower fine-tuning learning rate helps adapt the pretrained representations more conservatively.

### Q29. What is a key limitation of averaging batch accuracies equally when batch sizes differ?

A. It makes accuracy impossible to calculate.

B. It automatically increases accuracy.

C. It changes the model's architecture.

D. It can produce a slightly inaccurate dataset-wide estimate because smaller and larger batches receive equal weight.

Correct answer: D

Explanation: The notebook calculates the mean of batch-level accuracies and losses. If the last batch is smaller, its metrics receive the same weight as those of a full batch.

A more precise aggregation weights each batch by its number of examples, or computes total correct predictions divided by the total number of examples.

### Q30. What is the main purpose of comparing dense fine-tuning and LoRA using the same target dataset and training hyperparameters?

A. To guarantee that LoRA wins.

B. To make the comparison more controlled by keeping important experimental conditions consistent.

C. To eliminate the need for validation.

D. To ensure both models have exactly the same number of trainable parameters.

Correct answer: B

Explanation: Keeping the dataset, pretrained initialization, target head setup, training duration, learning rate, and weight decay consistent helps isolate the effect of changing the fine-tuning method.

The models do not have the same number of trainable parameters, and LoRA is not guaranteed to achieve higher accuracy.

## Section 5 — LoRA mathematics and conceptual foundations

### Q31. What is the fundamental idea behind LoRA?

A. Replace every pretrained weight with a random matrix.

B. Fine-tune a pretrained model by updating only its biases.

C. Freeze the original weight matrix and learn a low-rank update represented by smaller trainable matrices.

D. Remove the pretrained weights and train only the convolutional layers.

Correct answer: C

Explanation: LoRA keeps the original weight matrix $W$ frozen and learns a low-rank update. Instead of training a full-sized residual matrix, it factorizes the update into two smaller matrices.

This reduces the number of trainable parameters when the rank is sufficiently small.

### Q32. Which equation correctly describes the LoRA weight update using the notebook's notation?

A. $W_{\text{effective}}=W+BA$

B. $W_{\text{effective}}=W-ABW$

C. $W_{\text{effective}}=AB$

D. $W_{\text{effective}}=W/A+B$

Correct answer: A

Explanation: The original weights remain unchanged, and the effective weights are

$$
W_{\text{effective}}=W+\Delta W,
\qquad
\Delta W=BA.
$$

The original matrix supplies the pretrained behavior, while $BA$ represents the learned task-specific update.

### Q33. For a linear layer with weight matrix $W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$, what are the dimensions of $A$ and $B$ in the standard LoRA factorization?

A. $A\in\mathbb{R}^{d_{\text{out}}\times d_{\text{out}}}$, $B\in\mathbb{R}^{d_{\text{in}}\times d_{\text{in}}}$

B. $A\in\mathbb{R}^{r\times d_{\text{in}}}$, $B\in\mathbb{R}^{d_{\text{out}}\times r}$

C. $A\in\mathbb{R}^{d_{\text{in}}\times r}$, $B\in\mathbb{R}^{r\times d_{\text{out}}}$

D. Both matrices must have shape $d_{\text{out}}\times d_{\text{in}}$

Correct answer: B

Explanation: The product must have the same shape as the original weight matrix:

$$
BA:
(d_{\text{out}}\times r)(r\times d_{\text{in}})
=d_{\text{out}}\times d_{\text{in}}.
$$

Here, $r$ is the LoRA rank and is normally much smaller than the input and output dimensions.

### Q34. What does the rank $r$ control?

A. The number of target classes.

B. The number of convolutional channels.

C. The number of epochs.

D. The inner dimension of the low-rank factorization, affecting adapter capacity and trainable parameter count.

Correct answer: D

Explanation: Increasing $r$ increases the representational capacity of the update and the number of trainable adapter parameters.

A larger rank may improve adaptation, but it also increases computation and memory requirements. The best value depends on the task and must be tested experimentally.

### Q35. Why should the initial LoRA update normally be zero?

A. To ensure the adapted layer initially behaves like the original pretrained layer.

B. To ensure the output of every layer is zero.

C. To prevent the classifier from learning.

D. To eliminate the need for backpropagation.

Correct answer: A

Explanation: If the initial update satisfies $BA=0$, then

$$
Wx+BAx+b=Wx+b.
$$

The adapted layer initially produces the same output as the original layer. This makes the adaptation start from the pretrained model's behavior rather than introducing a random perturbation.

### Q36. Which initialization is recommended in the notebook's LoRA implementation tips?

A. Initialize both $A$ and $B$ with ones.

B. Initialize both matrices to zero.

C. Initialize $A$ randomly, typically using a normal distribution, and initialize $B$ to zero.

D. Initialize $A$ to zero and $B$ to the original full weight matrix.

Correct answer: C

Explanation: Randomly initializing $A$ and zero-initializing $B$ makes $BA=0$ initially while preserving a useful random projection for learning.

If both matrices are initialized to zero, then the gradients of their product with respect to both factors can initially vanish, preventing ordinary gradient descent from starting the intended learning process.

### Q37. In the notebook's notation, which forward-pass expression represents LoRA for a fully connected layer with a bias?

A. $z=\operatorname{ReLU}(BAx)$

B. $z=\operatorname{ReLU}(Wx+BAx+b)$

C. $z=\operatorname{ReLU}(WBAx)$

D. $z=\operatorname{ReLU}(Wx)+BAx+b$

Correct answer: B

Explanation: The original linear transformation and the adapter update are added before the bias and activation are applied:

$$
z=\operatorname{ReLU}(Wx+BAx+b).
$$

The `LoraLayer` wrapper itself generally returns the linear output; the ReLU is applied later by `ConvMLP.forward()`.

### Q38. Why can the LoRA update be represented with fewer parameters than a full update matrix?

Suppose the original matrix has $d_{\text{out}}d_{\text{in}}$ weights.

A. The update matrix is discarded after each forward pass.

B. The model does not calculate gradients.

C. The original matrix is converted into a scalar.

D. The update requires only $r(d_{\text{in}}+d_{\text{out}})$ parameters instead of $d_{\text{in}}d_{\text{out}}$.

Correct answer: D

Explanation: The adapter matrices contain

$$
N_{\text{LoRA}}=r d_{\text{in}}+d_{\text{out}}r
=r(d_{\text{in}}+d_{\text{out}})
$$

trainable weights.

When $r$ is sufficiently small, this can be far fewer than the parameters of a full-sized update matrix.

### Q39. The notebook mentions scaling the update by $1/r$. Which statement is correct?

A. Scaling is intended to make the effective update magnitude less dependent on the chosen rank, although practical LoRA implementations often use $\alpha/r$.

B. Scaling makes all adapter weights non-trainable.

C. Scaling increases the number of parameters in the original matrix.

D. Scaling changes the number of classes predicted by the classifier.

Correct answer: A

Explanation: A common LoRA formulation is

$$
W_{\text{effective}}=W+\frac{\alpha}{r}BA,
$$

where $\alpha$ is a scaling hyperparameter.

The notebook describes a simpler $1/r$ scaling convention and notes that it did not make much difference in the author's experiments. Scaling conventions should be reported accurately when comparing implementations.

### Q40. Is LoRA mathematically identical to dense fine-tuning in every possible case?

A. Yes, for every possible rank and training configuration.

B. No, because LoRA always modifies convolutional layers instead.

C. No. A low-rank update restricts the update space, so exact equivalence requires sufficient rank and compatible optimization and parameterization.

D. Yes, because the two methods have the same number of trainable parameters.

Correct answer: C

Explanation: A full update matrix can represent any matrix-shaped change, while a rank-$r$ product can represent only updates with rank at most $r$.

At sufficiently high rank, the factorization can represent any update matrix, but optimization through the factors is still parameterized differently. In the practical setting, LoRA is intended to approximate the useful update with far fewer trainable parameters.

## Section 6 — Understanding `AbstractLayer` and implementing LoRA

### Q41. What is the purpose of `AbstractLayer` in the notebook?

A. To replace the classifier with a convolutional layer.

B. To demonstrate how to freeze an original linear weight matrix and add a trainable residual update matrix.

C. To automatically select the optimal LoRA rank.

D. To calculate validation loss without running the model.

Correct answer: B

Explanation: `AbstractLayer` is an intermediate teaching implementation. It keeps the original weight matrix frozen and adds a full-sized trainable `dW` matrix. This is mathematically analogous to dense fine-tuning of the weight update, but it is not parameter-efficient because `dW` has the same size as the original matrix.

### Q42. Why does the `AbstractLayer` initialization create `dW` using `torch.zeros_like(self.W)`?

A. To create a matrix with the same shape as the original weight and initialize the residual update to zero.

B. To copy the original weights into the residual matrix.

C. To make the original weights trainable.

D. To set the layer's bias to zero permanently.

Correct answer: A

Explanation: `torch.zeros_like(self.W)` produces a zero tensor with the same dimensions and device-compatible properties as the original weight. Wrapping it in `nn.Parameter` makes the update trainable.

### Q43. Why is `AbstractLayer` not a parameter-efficient solution?

A. It has no trainable parameters.

B. It removes the original model's pretrained weights.

C. It cannot be used with a fully connected layer.

D. Its full-sized `dW` matrix contains as many parameters as the original weight matrix.

Correct answer: D

Explanation: Although the original weights are frozen, the entire residual matrix is trainable. LoRA replaces this large residual with two smaller factor matrices, reducing the number of trainable parameters.

### Q44. Which implementation correctly defines the two main LoRA parameter matrices for an original linear layer?

A. `A = nn.Parameter(torch.zeros_like(original_layer.weight))` and `B = nn.Parameter(torch.ones_like(original_layer.weight))`

B. `A = nn.Parameter(torch.zeros(r, r))` and `B = nn.Parameter(torch.zeros(r, r))`

C. `A = nn.Parameter(torch.randn(r, original_layer.in_features))` and `B = nn.Parameter(torch.zeros(original_layer.out_features, r))`

D. `A = nn.Parameter(torch.randn(original_layer.out_features, original_layer.in_features))` and `B = nn.Parameter(torch.zeros(1))`

Correct answer: C

Explanation: For an input vector of dimension $d_{\text{in}}$, the adapter first projects into an $r$-dimensional representation and then projects into $d_{\text{out}}$ dimensions.

The shapes are:

python

```
A.shape == (r, in_features)
B.shape == (out_features, r)
```

Thus, the update is $BA$, with shape $(d_{\text{out}},d_{\text{in}})$.

### Q45. Which forward-pass implementation correctly applies the LoRA update without explicitly constructing the full matrix $BA$?

A. `return original_layer(x) + F.linear(x, A)`

B. `return F.linear(F.linear(x, A), B) + original_layer(x) - original_layer(x)`

C. `return A @ B @ x`

D. `return original_layer(x) + F.linear(F.linear(x, A), B)`

Correct answer: D

Explanation: The efficient computation is

python

```
base = original_layer(x)
update = F.linear(F.linear(x, A), B)
return base + update
```

This computes the adapter contribution through two smaller linear transformations without explicitly materializing the full update matrix.

The base layer's bias is included once, through `original_layer(x)`.

### Q46. What bug exists in the supplied `LoraLayer` skeleton?

A. It refers to `verbose` without receiving or defining it, and the initialization and forward-pass logic are unfinished.

B. It uses a convolutional layer rather than a linear layer.

C. It initializes the classifier with 100 output classes.

D. It always freezes the LoRA matrices.

Correct answer: A

Explanation: The skeleton contains:

python

```
self.verbose = verbose
```

but `verbose` is not defined in the constructor's arguments or local scope. This raises a `NameError` when the line executes.

The placeholders `...` also mean the matrices, freezing logic, and forward pass still need to be implemented. A correct implementation should explicitly freeze the original weight and define $A$, $B$, and the adapter forward computation.

### Q47. Which parameters should remain trainable in the notebook's intended LoRA fine-tuning setup?

A. Every parameter in the model.

B. Only the original convolutional weights.

C. The $A$ and $B$ matrices for `fc1`–`fc4`, plus the target classifier's weights and bias.

D. Only the pretrained source classifier.

Correct answer: C

Explanation: The convolutional layers are frozen, and the original weight matrices in the adapted fully connected layers are also frozen. The LoRA factors are trainable, while the target classifier remains trainable because it must learn to distinguish the five target classes.

The biases of the original fully connected layers can remain frozen, as suggested by the notebook's adapter design.

### Q48. What is the purpose of using `strict=False` when loading the pretrained state dictionary into the target model?

A. To disable all parameter loading.

B. To permit missing or unexpected keys, such as the target classifier's new weights, while loading compatible parameters.

C. To make all pretrained weights trainable.

D. To allow tensors of any shape to overwrite one another.

Correct answer: B

Explanation: The notebook removes the source classifier's weight and bias from the checkpoint, then loads the remaining compatible parameters into the target model.

With `strict=False`, PyTorch permits missing keys for the newly initialized target classifier. It does not make incompatible tensor shapes load successfully.

## Section 7 — Parameter efficiency, checkpoints, and optimization

### Q49. For a linear layer with 1024 input features, 512 output features, and LoRA rank $r=8$, how many trainable adapter weights are required, excluding bias?

A. 4,096

B. 8,192

C. 12,288

D. 12,292

Correct answer: C

Explanation: The adapter contains

$$
N_{\text{LoRA}}=r(d_{\text{in}}+d_{\text{out}}).
$$

Substituting the values gives

$$
N_{\text{LoRA}}=8(1024+512)=12,288.
$$

The full weight matrix contains $1024\times512=524,288$ weights, so the adapter has substantially fewer trainable weights.

### Q50. What is the main reason LoRA can reduce optimizer-state memory?

A. It deletes the model's forward-pass activations.

B. It guarantees that the optimizer uses no gradients.

C. It stores all model weights in lower precision automatically.

D. Optimizer state is generally maintained for trainable parameters, so training fewer parameters reduces the amount of optimizer state.

Correct answer: D

Explanation: AdamW maintains state such as first and second moment estimates for trainable parameters. Reducing the number of trainable parameters can therefore reduce optimizer-state memory.

However, the frozen base model weights must still be stored for inference and training, and gradient and activation memory do not disappear entirely.

### Q51. Which statement best explains why a LoRA model's complete `state_dict` may not be dramatically smaller than the baseline model's `state_dict`?

A. The full state dictionary still includes the frozen base weights, in addition to the adapter parameters.

B. LoRA duplicates every training example.

C. LoRA cannot save tensors.

D. The state dictionary contains only optimizer moments.

Correct answer: A

Explanation: Freezing a parameter does not remove it from the model. A normal model state dictionary includes the original weights and the learned adapter matrices, as well as other model parameters.

To save only the adapters, an explicit adapter-only checkpoint format must be implemented.

### Q52. Why might LoRA be slower than dense fine-tuning during a forward or backward pass in this particular notebook?

A. Frozen parameters cannot be used for matrix multiplication.

B. The adapter introduces additional matrix multiplications, and the model is relatively small, so adapter overhead may outweigh some savings.

C. LoRA always runs on the CPU.

D. LoRA necessarily doubles the number of convolutional layers.

Correct answer: B

Explanation: LoRA reduces the number of trainable parameters, but the forward pass still computes the base transformation and the adapter contribution.

For a small model, extra operations and framework overhead can offset savings from computing fewer parameter gradients. The actual timing must be measured on the target hardware.

### Q53. What does the notebook's `get_state_dict_size()` function measure?

A. The number of bytes occupied by the checkpoint file on disk.

B. The number of training examples in memory.

C. The total number of scalar elements in the supplied state dictionary.

D. The number of trainable parameters only.

Correct answer: C

Explanation: The function sums `values.numel()` across the state dictionary. This counts tensor elements, not bytes.

To estimate storage in bytes, each tensor's element count must be multiplied by its element size, and serialization overhead may also affect the actual file size.

### Q54. What is a limitation of `get_opt_state_dict_size()`?

A. It cannot inspect any optimizer state.

B. It always returns the exact file size in megabytes.

C. It measures the number of training batches.

D. It counts tensor elements in the optimizer's state, but not the complete serialized optimizer file size or all associated metadata.

Correct answer: D

Explanation: The function counts elements inside each tensor in `state_dict()['state']`. It does not directly report bytes, account for all serialization overhead, or measure total GPU memory usage.

It is useful for comparing the scale of tensor-based optimizer state, provided the same counting convention is used for both models.

## Section 8 — Results, debugging, and experimental interpretation

### Q55. What should you conclude if LoRA achieves a validation accuracy within a few percentage points of the dense fine-tuning baseline while using far fewer trainable parameters?

A. LoRA has demonstrated a useful parameter-efficiency trade-off for this task.

B. LoRA must have trained every parameter.

C. The validation set was unnecessary.

D. The baseline implementation is automatically incorrect.

Correct answer: A

Explanation: Comparable validation performance with fewer trainable parameters is the main intended outcome. It indicates that the low-rank update can capture much of the adaptation needed for the target domain.

The conclusion should still consider runtime, memory use, training stability, and experimental variability.

### Q56. Why should the best validation accuracy be distinguished from the final validation accuracy?

A. They are always identical.

B. The best validation accuracy measures the highest value observed across epochs, whereas the final accuracy measures performance at the last epoch.

C. The best validation accuracy is measured only on the training set.

D. The final validation accuracy is always higher.

Correct answer: B

Explanation: The notebook's `best_val_acc` property computes the maximum of all recorded validation accuracies.

A model can reach its best validation performance before the last epoch and subsequently overfit or fluctuate. Reporting both metrics provides a clearer picture of the training trajectory.

### Q57. What should you do if the assertion that the original LoRA weight is frozen fails?

A. Remove the assertion without investigating.

B. Set every model parameter to zero.

C. Check that the original layer's weight has `requires_grad=False` and verify that the optimizer and adapter replacement are configured correctly.

D. Replace the target dataset with CIFAR-10.

Correct answer: C

Explanation: The assertion verifies a central LoRA requirement. If the original weight remains trainable, the model may perform dense fine-tuning as well as adapter learning, undermining the intended parameter-efficiency comparison.

Inspecting named parameters and their `requires_grad` flags helps identify the problem.

### Q58. What is a potential problem with the notebook's random train-validation split?

A. Random splits always leak validation labels into training.

B. Random splitting makes classification impossible.

C. A validation set cannot be used with CIFAR-100.

D. The split is not explicitly stratified, so class proportions may vary between training and validation; reproducibility also depends on the random-number state.

Correct answer: D

Explanation: A random split does not guarantee equal class proportions. For small or imbalanced datasets, stratified splitting may produce a more representative validation set.

The notebook sets a random seed before splitting, which helps reproducibility, but reproducibility still depends on the complete random-number state and execution order.

## Section 9 — Advanced concepts and extension exercises

### Q59. After training LoRA, how can its adapter be merged into the original linear layer?

A. Replace the original weight with $W+BA$, or $W+\frac{\alpha}{r}BA$ when using that scaling convention, and preserve the bias.

B. Replace the original weight with $A+B$.

C. Delete the original weight and retain only the bias.

D. Replace the original weight with $W-BAW$.

Correct answer: A

Explanation: Because the adapter contributes an additive weight update, it can be merged into the base matrix:

$$
W_{\text{merged}}=W+BA.
$$

If scaling is used, the corresponding scaled update must be merged. The resulting linear layer can use its ordinary weight matrix for inference without separately computing the adapter branch.

The merged and unmerged models should be checked for numerically equivalent outputs.

### Q60. Which experimental conclusion is most scientifically defensible after completing the notebook?

A. LoRA is always faster, more accurate, and smaller than every dense fine-tuning method.

B. LoRA's usefulness should be judged using measured trainable parameter counts, validation performance, runtime, memory usage, checkpoint sizes, and rank experiments under controlled conditions.

C. A single successful training run proves that rank never matters.

D. A smaller adapter automatically guarantees a smaller complete model checkpoint and faster inference.

Correct answer: B

Explanation: LoRA is a parameter-efficient adaptation technique, not a universal guarantee of faster execution or better accuracy. Its performance depends on the task, architecture, rank, optimizer, hardware, and implementation.

A complete report should distinguish the number of trainable parameters from the total model parameters, measure rather than assume speed and memory benefits, and explain any observed accuracy differences. It should also document the implementation, training curves, rank experiments, debugging issues, and limitations.

## Final exam revision checklist

Before submitting Practical 4, make sure you can explain the following without referring to the notebook:

- Transfer learning: why CIFAR-10 pretraining provides reusable features for the five-class CIFAR-100 fish task.
- Dataset preparation: why labels need remapping and how normalization and DataLoaders work.
- Architecture: the roles of the convolutional layers, adaptive pooling, MLP, and classifier head.
- Dense fine-tuning: which parameters are frozen and which are trainable.
- LoRA mathematics: why $W_{\text{effective}}=W+BA$, how the factor dimensions work, and how rank affects capacity.
- Initialization: why $B=0$ and a randomly initialized $A$ produce a zero initial update.
- Implementation: how to freeze the original weights, define adapter matrices, and compute the forward pass efficiently.
- Evaluation: how cross-entropy, top-1 accuracy, validation curves, and best-versus-final accuracy should be interpreted.
- Parameter efficiency: why fewer trainable parameters can reduce optimizer-state memory without eliminating the need to store the base model.
- Experimental reporting: how to compare accuracy, state dictionaries, optimizer state, runtime, GPU memory, and different ranks without inventing results.
- Extensions: how to merge adapters into base weights and save adapter-only checkpoints.

The central lesson: LoRA reduces the number of trainable parameters by constraining the learned weight update to a low-rank factorization. Its success is demonstrated not merely by having fewer parameters, but by retaining useful target-task performance and providing measurable practical benefits under a fair comparison.
