 # Advanced Machine Learning — Practical 4: LoRA
 
Absolutely. Below is a **comprehensive mastery MCQ set** covering the notebook from beginning to end: transfer learning, CIFAR-10/CIFAR-100, PyTorch code, the model architecture, training loop, dense fine-tuning, LoRA mathematics, implementation details, parameter counting, optimizer state, timing, rank experiments, and the optional extensions.

 I’ve deliberately included some **code-tracing and “what happens if…” questions**, because those are often the hardest questions in an advanced ML practical.


 ## Comprehensive MCQ Mastery Test

 **Format:** 60 questions\
 **Difficulty:** Beginner → Advanced\
 **Each question includes the correct answer and an explanation.**

---

 # Part 1 — Overall objective and transfer learning

 ### Q1. What is the main objective of this practical?

 A. Train a CNN from scratch on CIFAR-100\
 B. Compare different image augmentation methods\
 C. Perform parameter-efficient fine-tuning using LoRA\
 D. Replace convolutional layers with transformers

 **Answer: C**

 **Explanation:**\
 The practical starts with a pretrained model and investigates how to adapt it to a target dataset while updating far fewer parameters. The technique being investigated is **Low-Rank Adaptation (LoRA)**.

---

 ### Q2. What is the source domain in this practical?

 A. CIFAR-100\
 B. CIFAR-10\
 C. ImageNet\
 D. MNIST

 **Answer: B**

 **Explanation:**\
 The model is first pretrained on **CIFAR-10**. This gives the network a useful representation before it is adapted to the target task.

---

 ### Q3. What is the target domain?

 A. The entire CIFAR-10 dataset\
 B. ImageNet\
 C. A selected superclass from CIFAR-100\
 D. MNIST

 **Answer: C**

 **Explanation:**\
 The notebook selects one CIFAR-100 superclass, such as `"fish"` or `"vehicles"`, producing a small five-class target dataset.

---

 ### Q4. If:

```
TARGET_CATEGORY = 'fish'
```

 which classes are selected?

 A. fish, shark, whale, dolphin, seal\
 B. aquarium\_fish, flatfish, ray, shark, trout\
 C. trout, salmon, tuna, shark, whale\
 D. aquarium\_fish, dolphin, ray, whale, trout

 **Answer: B**

 **Explanation:**\
 The dictionary explicitly defines:

```
"fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"]
```

---

 ### Q5. Why is transfer learning useful here?

 A. CIFAR-100 contains no labels\
 B. The target dataset is relatively small, so pretrained representations can be reused\
 C. It eliminates the need for validation data\
 D. It guarantees 100% accuracy

 **Answer: B**

 **Explanation:**\
 The model has already learned useful visual representations from CIFAR-10. Rather than learning everything from random initialization on the smaller target dataset, we adapt the pretrained representation.

---

 ### Q6. What is the fundamental idea behind parameter-efficient fine-tuning?

 A. Increase the number of training examples\
 B. Update every model parameter more aggressively\
 C. Adapt a pretrained model while updating only a small number of parameters\
 D. Remove the pretrained model

 **Answer: C**

 **Explanation:**\
 PEFT methods such as LoRA try to preserve the pretrained model while learning only a relatively small task-specific update.

---

 # Part 2 — Dataset preparation

 ### Q7. What does this line do?

```
cifar10 = datasets.CIFAR10(root=data_root, download=False)
```

 A. Downloads CIFAR-100\
 B. Loads CIFAR-10 from the specified directory\
 C. Creates a neural network\
 D. Normalizes CIFAR-10

 **Answer: B**

 **Explanation:**\
 The notebook has already downloaded/extracted the data. `download=False` tells torchvision not to download it again.

---

 ### Q8. What does this expression produce?

```
cifar100.class_to_idx[name]
```

 A. The image corresponding to `name`\
 B. The integer class index corresponding to the class name\
 C. The number of samples in the class\
 D. A PyTorch model

 **Answer: B**

 **Explanation:**\
 `class_to_idx` maps class names to integer labels.

 For example, conceptually:

```
"shark" → 73
```

---

 ### Q9. Why is the target dataset's labels remapped?

 A. CIFAR-100 images must be resized\
 B. The selected classes may have original labels that are not 0–4\
 C. CrossEntropyLoss cannot use integers\
 D. PyTorch cannot handle CIFAR-100

 **Answer: B**

 **Explanation:**\
 Suppose the five selected CIFAR-100 classes originally have labels:

```
23, 44, 68, 73, 91
```

 For a five-class classifier, we want:

```
0, 1, 2, 3, 4
```

 Hence the `remap` dictionary.

---

 ### Q10. What does this create?

```
target_data = torch.utils.data.Subset(cifar100, subset_idxs)
```

 A. A new neural network\
 B. A subset containing only selected CIFAR-100 examples\
 C. A validation loader\
 D. A normalization transform

 **Answer: B**

 **Explanation:**\
 `Subset` creates a dataset view containing only the specified indices.

---

 ### Q11. What does `random_split` do here?

```
train_data_target, val_data_target = torch.utils.data.random_split(
    target_data, [1-val_frac, val_frac]
)
```

 A. Splits each image into pieces\
 B. Creates training and validation subsets\
 C. Splits labels from images\
 D. Splits CIFAR-100 into 100 classes

 **Answer: B**

 **Explanation:**\
 With `val_frac = 0.2`, approximately 80% of the target data becomes training data and 20% becomes validation data.

---

 ### Q12. Why is:

```
torch.manual_seed(0)
```

 used before splitting?

 A. To enable GPU execution\
 B. To make random operations reproducible\
 C. To normalize images\
 D. To initialize AdamW

 **Answer: B**

 **Explanation:**\
 Setting the random seed makes stochastic operations more reproducible, including the random split.

---

 # Part 3 — Image preprocessing and DataLoaders

 ### Q13. What does `transforms.ToTensor()` do?

 A. Converts an image into a PyTorch tensor\
 B. Converts a tensor into a PIL image\
 C. Performs classification\
 D. Calculates loss

 **Answer: A**

 **Explanation:**\
 It converts image data into tensor form suitable for PyTorch models.

---

 ### Q14. What is the purpose of:

```
transforms.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2))
```

 A. Randomly crops images\
 B. Normalizes each RGB channel\
 C. Converts RGB to grayscale\
 D. Changes the number of classes

 **Answer: B**

 **Explanation:**\
 The transform normalizes each channel using the specified means and standard deviations.

---

 ### Q15. Why does `collate_source()` use:

```
torch.stack(...)
```

 A. To combine individual image tensors into a batch tensor\
 B. To calculate gradients\
 C. To freeze weights\
 D. To perform pooling

 **Answer: A**

 **Explanation:**\
 Individual images have shape approximately:

 $$
3\times32\times32
$$

 and stacking them creates a batch:

 $$
B\times3\times32\times32
$$

---

 ### Q16. What is the purpose of:

```
label_tensor = torch.tensor(labels, dtype=torch.long)
```

 A. Convert labels into floating-point probabilities\
 B. Convert labels into integer tensors appropriate for classification loss\
 C. Normalize labels\
 D. One-hot encode labels

 **Answer: B**

 **Explanation:**\
 `CrossEntropyLoss` expects class indices represented as integer (`long`) tensors.

---

 ### Q17. Why are tensors moved to:

```
device = 'cuda'
```

 A. To make them larger\
 B. To execute computations on the GPU\
 C. To convert them to NumPy\
 D. To freeze them

 **Answer: B**

 **Explanation:**\
 If CUDA is available, moving the model and tensors to the GPU allows GPU acceleration.

---

 # Part 4 — Model architecture

 ### Q18. What is the purpose of:

```
nn.Conv2d(input_channels, 64, 3, padding=1)
```

 A. Fully connected classification\
 B. Extract spatial features using a convolution\
 C. Reduce the number of classes\
 D. Perform normalization

 **Answer: B**

 **Explanation:**\
 The first convolution extracts low-level spatial features such as edges and textures.

---

 ### Q19. What does:

```
nn.MaxPool2d(2)
```

 typically do?

 A. Doubles spatial dimensions\
 B. Halves spatial dimensions\
 C. Doubles channels\
 D. Removes all channels

 **Answer: B**

 **Explanation:**\
 A 2×2 max-pooling operation with default stride 2 reduces height and width by approximately a factor of two.

---

 ### Q20. What does:

```
nn.AdaptiveAvgPool2d((1,1))
```

 do?

 A. Produces exactly one spatial value per channel\
 B. Converts the image into 100 classes\
 C. Doubles image resolution\
 D. Removes all channels

 **Answer: A**

 **Explanation:**\
 If the input has shape:

```
B × 256 × H × W
```

 the output becomes:

```
B × 256 × 1 × 1
```

---

 ### Q21. After:

```
x = self.globalpool(x)
x = x.view(-1, self.conv2.out_channels)
```

 what is the feature dimension?

 A. 3\
 B. 64\
 C. 256\
 D. 1024

 **Answer: C**

 **Explanation:**\
 `conv2` outputs 256 channels, and global average pooling produces one value per channel.

 Therefore the flattened representation has 256 features.

---

 ### Q22. What is the architecture of `fc1`?

```
self.fc1 = nn.Linear(self.conv2.out_channels, 1024)
```

 A. 3 → 1024\
 B. 64 → 1024\
 C. 256 → 1024\
 D. 512 → 1024

 **Answer: C**

 **Explanation:**\
 `conv2.out_channels = 256`.

 Therefore:

 $$
fc1:256\rightarrow1024
$$

---

 ### Q23. What is the architecture of `fc2`?

 A. 256 → 1024\
 B. 1024 → 512\
 C. 512 → 1024\
 D. 512 → 256

 **Answer: B**

---

 ### Q24. What is the architecture of `fc3`?

 A. 512 → 512\
 B. 1024 → 512\
 C. 512 → 256\
 D. 256 → 512

 **Answer: A**

---

 ### Q25. What is the architecture of `fc4`?

 A. 512 → 1024\
 B. 1024 → 256\
 C. 512 → 256\
 D. 256 → 512

 **Answer: C**

---

 ### Q26. If the target task contains five classes, what is the classifier architecture?

 A. `Linear(10, 5)`\
 B. `Linear(256, 5)`\
 C. `Linear(512, 5)`\
 D. `Linear(1024, 5)`

 **Answer: B**

 **Explanation:**\
 `fc4` outputs 256 features, so the classifier maps:

 $$
256\rightarrow5
$$

---

 # Part 5 — Parameter counting

 ### Q27. What does:

```
param.numel()
```

 return?

 A. Number of layers\
 B. Number of elements in a tensor\
 C. Tensor dimensionality only\
 D. Learning rate

 **Answer: B**

 **Explanation:**\
 For a matrix of shape $512\times1024$:

 $$
512\times1024=524288
$$

 parameters.

---

 ### Q28. What does this calculate?

```
trainable_params = sum(
    param.numel()
    for param in model.parameters()
    if param.requires_grad
)
```

 A. Total FLOPs\
 B. Number of trainable parameters\
 C. Number of training samples\
 D. Number of classes

 **Answer: B**

---

 ### Q29. What happens to the trainable parameter count when:

```
model.conv1.requires_grad_(False)
```

 is executed?

 A. It increases\
 B. It decreases\
 C. It stays exactly the same\
 D. The model is deleted

 **Answer: B**

 **Explanation:**\
 The convolutional parameters remain in the model but are no longer trainable.

---

 # Part 6 — Training loop

 ### Q30. What loss function does the notebook use?

```
nn.CrossEntropyLoss()
```

 A. Mean squared error\
 B. Binary cross entropy\
 C. Multiclass cross entropy\
 D. Hinge loss

 **Answer: C**

---

 ### Q31. What does:

```
opt.zero_grad()
```

 do?

 A. Deletes model weights\
 B. Resets accumulated gradients\
 C. Resets optimizer learning rate\
 D. Freezes the model

 **Answer: B**

 **Explanation:**\
 PyTorch accumulates gradients by default, so they need to be cleared before each optimization step.

---

 ### Q32. What does:

```
batch_loss.backward()
```

 do?

 A. Performs the forward pass\
 B. Computes gradients using backpropagation\
 C. Updates weights directly\
 D. Evaluates validation accuracy

 **Answer: B**

---

 ### Q33. What does:

```
opt.step()
```

 do?

 A. Calculates the loss\
 B. Updates trainable parameters using the computed gradients\
 C. Creates the validation set\
 D. Freezes the model

 **Answer: B**

---

 ### Q34. What optimizer is used?

 A. SGD\
 B. RMSProp\
 C. AdamW\
 D. Adagrad

 **Answer: C**

 The notebook uses:

```
torch.optim.AdamW(...)
```

---

 ### Q35. What is the purpose of `weight_decay` in AdamW?

 A. Increase image resolution\
 B. Provide L2-style regularization\
 C. Increase the number of classes\
 D. Freeze parameters

 **Answer: B**

 **Explanation:**\
 The notebook calls it `l2_reg`, and passes it to AdamW as `weight_decay`.

---

 ### Q36. Why does the validation loop use:

```
with torch.no_grad():
```

 A. To disable all model computations\
 B. To avoid gradient tracking during evaluation\
 C. To freeze the dataset\
 D. To train faster using larger gradients

 **Answer: B**

 **Explanation:**\
 Validation doesn't require gradients. Disabling gradient tracking saves memory and computation.

---

 ### Q37. What does:

```
model.eval()
```

 do?

 A. Deletes the optimizer\
 B. Puts the model into evaluation mode\
 C. Freezes every parameter permanently\
 D. Starts training

 **Answer: B**

 **Explanation:**\
 It changes the behavior of layers such as dropout and batch normalization where relevant.

---

 ### Q38. What does:

```
model.train()
```

 do?

 A. Puts the model into training mode\
 B. Trains the model automatically\
 C. Initializes the weights\
 D. Changes the optimizer

 **Answer: A**

---

 # Part 7 — Accuracy and metrics

 ### Q39. What does this calculate?

```
pred.argmax(axis=1)
```

 A. The maximum logit for every class\
 B. The predicted class index for every sample\
 C. The loss\
 D. The gradient

 **Answer: B**

 **Explanation:**\
 The class with the highest output logit is treated as the prediction.

---

 ### Q40. What does:

```
top1_acc(pred, y)
```

 measure?

 A. Top-5 accuracy\
 B. Mean squared error\
 C. Fraction of examples whose highest-scoring class matches the label\
 D. Number of trainable parameters

 **Answer: C**

---

 ### Q41. Why does `CrossEntropyLoss` receive logits rather than softmax probabilities?

 A. CrossEntropyLoss internally handles the appropriate normalization\
 B. Softmax cannot be computed in PyTorch\
 C. Logits are always probabilities\
 D. It requires images instead

 **Answer: A**

 **Explanation:**\
 PyTorch's `CrossEntropyLoss` combines the appropriate log-softmax and negative log-likelihood behavior internally.

---

 # Part 8 — Pre-training

 ### Q42. Why is the classifier removed when transferring the CIFAR-10 model?

 A. The classifier is broken\
 B. CIFAR-10 has 10 classes while the selected target task has 5\
 C. The convolutional layers cannot be transferred\
 D. LoRA requires no classifier

 **Answer: B**

---

 ### Q43. What does:

```
del pt_state_dict['classifier.weight']
del pt_state_dict['classifier.bias']
```

 accomplish?

 A. Removes the pretrained classifier parameters from the checkpoint\
 B. Deletes the entire model\
 C. Deletes all convolutional parameters\
 D. Deletes the optimizer

 **Answer: A**

---

 ### Q44. Why is:

```
strict=False
```

 used with:

```
load_state_dict(...)
```

 ?

 A. To allow missing classifier parameters\
 B. To disable training\
 C. To ignore all errors\
 D. To create random images

 **Answer: A**

 **Explanation:**\
 The new target model has a different classifier, so its classifier parameters aren't present in the loaded state dictionary.

---

 # Part 9 — Dense fine-tuning

 ### Q45. Which parameters are trainable in the baseline model?

 A. Only convolutional parameters\
 B. Only classifier parameters\
 C. MLP and classifier parameters, with convolutional layers frozen\
 D. Nothing

 **Answer: C**

---

 ### Q46. What is the key characteristic of dense fine-tuning?

 A. It approximates weight updates using low-rank matrices\
 B. It directly updates the selected full weight matrices\
 C. It removes the pretrained weights\
 D. It trains only biases

 **Answer: B**

---

 ### Q47. If a weight matrix has 1,000,000 parameters and is trainable under dense fine-tuning, how many parameters can potentially receive gradient updates?

 A. 1\
 B. 10\
 C. 100,000\
 D. 1,000,000

 **Answer: D**

---

 # Part 10 — LoRA mathematics

 ### Q48. What is the fundamental LoRA equation?

 A. $z=Ax+Bx$\
 B. $z=Wx+BAx+b$\
 C. $z=WAx$\
 D. $z=W+B+A+x$

 **Answer: B**

---

 ### Q49. In LoRA, what happens to the original weight matrix $W$?

 A. It is deleted\
 B. It is randomly reinitialized\
 C. It is frozen\
 D. It is replaced by $A$

 **Answer: C**

---

 ### Q50. What does $BA$ represent?

 A. The classifier\
 B. An approximation to the weight update $\Delta W$\
 C. The bias\
 D. The loss function

 **Answer: B**

---

 ### Q51. Suppose:

 $$
W\in\mathbb{R}^{512\times1024}
$$

 and $r=8$.

 What are the dimensions of A and B?

 A. $A=512\times8,\ B=8\times1024$\
 B. $A=8\times1024,\ B=512\times8$\
 C. $A=1024\times8,\ B=8\times512$\
 D. Both are $8\times8$

 **Answer: B**

 **Explanation:**

 $$
A\in\mathbb{R}^{r\times d_{in}}
$$

 and:

 $$
B\in\mathbb{R}^{d_{out}\times r}
$$

 Therefore:

 $$
A=8\times1024
$$

 $$
B=512\times8
$$

---

 ### Q52. What is the shape of $BA$ in the previous example?

 A. $8\times8$\
 B. $1024\times512$\
 C. $512\times1024$\
 D. $1024\times1024$

 **Answer: C**

 Because:

 $$
(512\times8)(8\times1024)
=
512\times1024
$$

 which matches $W$.

---

 ### Q53. Why must $r$ be much smaller than the dimensions of $W$?

 A. To make $BA$ a scalar\
 B. To reduce the number of trainable parameters\
 C. To eliminate the classifier\
 D. To increase the image size

 **Answer: B**

---

 ### Q54. Which parameterization gives the lowest number of LoRA parameters?

 For a layer with $d_{in}=1000$, $d_{out}=500$:

 A. $r=1$\
 B. $r=10$\
 C. $r=100$\
 D. $r=500$

 **Answer: A**

 **Explanation:**

 LoRA parameter count is:

 $$
r d_{in}+d_{out}r
$$

 or:

 $$
r(d_{in}+d_{out})
$$

 So smaller $r$ means fewer parameters.

---

 # Part 11 — LoRA initialization

 ### Q55. Why is B typically initialized to zero?

 A. To ensure $BA=0$ initially\
 B. To make A zero\
 C. To remove the pretrained model\
 D. To increase the rank

 **Answer: A**

---

 ### Q56. Suppose:

 $$
B=0
$$

 What is $BA$?

 A. Identity matrix\
 B. Random matrix\
 C. Zero matrix\
 D. Undefined

 **Answer: C**

---

 ### Q57. Why is A typically randomly initialized rather than also initialized to zero?

 A. To introduce useful variation into the learning dynamics\
 B. Because A is not trainable\
 C. Because A must be square\
 D. To make $BA$ initially nonzero

 **Answer: A**

 **Explanation:**\
 With $B=0$, the product is still zero initially. A random initialization provides useful gradients/dynamics once B starts moving away from zero.

---

 ### Q58. If both A and B are initialized to zero, what is a potential problem?

 A. The initial LoRA update is nonzero\
 B. The adapter can have undesirable symmetry/gradient behavior\
 C. The pretrained weights disappear\
 D. The rank becomes infinite

 **Answer: B**

 **Explanation:**\
 The standard initialization avoids initializing both factors to zero because this can cause problematic learning dynamics.

---

 # Part 12 — Implementing `LoraLayer`

 ### Q59. Which implementation correctly defines A for a layer?

```
original_layer = nn.Linear(1024, 512)
r = 8
```

 A.

```
self.A = nn.Parameter(torch.randn(512, 8))
```

 B.

```
self.A = nn.Parameter(torch.randn(8, 1024))
```

 C.

```
self.A = nn.Parameter(torch.randn(1024, 512))
```

 D.

```
self.A = nn.Parameter(torch.randn(8, 512))
```

 **Answer: B**

 **Explanation:**

 $$
A=(r,d_{in})=(8,1024)
$$

---

 ### Q60. Which implementation correctly defines B?

 A.

```
self.B = nn.Parameter(torch.zeros(8, 1024))
```

 B.

```
self.B = nn.Parameter(torch.zeros(1024, 8))
```

 C.

```
self.B = nn.Parameter(torch.zeros(512, 8))
```

 D.

```
self.B = nn.Parameter(torch.zeros(512, 1024))
```

 **Answer: C**

 **Explanation:**

 $$
B=(d_{out},r)=(512,8)
$$

---

 # Part 13 — Forward pass code reasoning

 ### Q61. Which expression correctly represents the LoRA update?

 A.

```
x @ A @ B
```

 B.

```
x @ B @ A
```

 C.

```
A @ B @ x
```

 D.

```
W @ A @ B
```

 **Answer: A**

 **Explanation:**\
 If:

```
x: batch × in_features
A: r × in_features
```

 then because `x` is row-oriented in this implementation, you can calculate the equivalent transformation as:

```
x @ A.T
```

 followed by:

```
... @ B.T
```

 Mathematically this corresponds to:

 $$
BAx
$$

 The exact code must respect PyTorch's row-vector convention.

---

 ### Q62. Which is the safest direct implementation using PyTorch's `Linear`-style orientation?

 A.

```
z = self.original_layer(x) + x @ self.A.T @ self.B.T
```

 B.

```
z = self.original_layer(x) + x @ self.B.T @ self.A.T
```

 C.

```
z = self.original_layer(x) + self.A @ self.B @ x
```

 D.

```
z = self.original_layer(x) + self.B @ self.A @ x
```

 **Answer: A**

 **Explanation:**\
 For a batch of row vectors:

```
x              = [batch × in]
A.T            = [in × r]
x @ A.T        = [batch × r]

B.T            = [r × out]
(x @ A.T) @ B.T = [batch × out]
```

 This is equivalent to applying:

 $$
BA
$$

 to the input.

---

 ### Q63. Why shouldn't the LoRA forward pass simply replace `W` with `BA`?

 A. Because then the pretrained representation would be discarded\
 B. Because B cannot be multiplied by A\
 C. Because LoRA only works with biases\
 D. Because A and B must be frozen

 **Answer: A**

 The correct operation is:

 $$
Wx+BAx+b
$$

 not:

 $$
BAx+b
$$

 The original pretrained pathway remains active.

---

 ### Q64. What should the LoRA output equal immediately after initialization?

 A. Random output\
 B. Zero\
 C. The original layer output\
 D. The classifier output

 **Answer: C**

 **Explanation:**\
 Because:

 $$
B=0
$$

 we have:

 $$
BA=0
$$

 therefore:

 $$
Wx+BAx+b=Wx+b
$$

---

 # Part 14 — Freezing parameters

 ### Q65. Which parameter should definitely be frozen in the LoRA-adapted FC layers?

 A. A\
 B. B\
 C. Original W\
 D. Classifier bias

 **Answer: C**

---

 ### Q66. Which parameters should generally be trainable in the intended implementation?

 A. W only\
 B. A and B, plus the new classifier\
 C. Conv1 and Conv2 only\
 D. Nothing

 **Answer: B**

---

 ### Q67. If the original weight is accidentally left trainable, what happens?

 A. The experiment is no longer pure LoRA fine-tuning\
 B. The model becomes impossible to train\
 C. A and B disappear\
 D. The rank becomes zero

 **Answer: A**

 **Explanation:**\
 If W is trainable, you are simultaneously updating the full dense weight and LoRA adapters. That defeats the parameter-efficient comparison.

---

 # Part 15 — Parameter calculations

 ### Q68. A dense layer has:

 $$
d_{in}=1024,\quad d_{out}=512
$$

 How many weight parameters does it have?

 A. 1,536\
 B. 12,288\
 C. 524,288\
 D. 1,048,576

 **Answer: C**

 $$
1024\times512=524,288
$$

---

 ### Q69. For the same layer and $r=8$, how many LoRA parameters are there?

 A. 8,192\
 B. 12,288\
 C. 524,288\
 D. 536,576

 **Answer: B**

 $$
A=8\times1024=8192
$$

 $$
B=512\times8=4096
$$

 Total:

 $$
8192+4096=12,288
$$

---

 ### Q70. What percentage of the dense weight's parameters does this LoRA adapter use?

 A. Approximately 2.3%\
 B. Approximately 8%\
 C. Approximately 25%\
 D. Approximately 50%

 **Answer: A**

 $$
\frac{12,288}{524,288}\approx0.0234
$$

 So about **2.34%**.

---

 ### Q71. What is the general number of LoRA parameters for a weight matrix?

 A. $d_{in}d_{out}r$\
 B. $r(d_{in}+d_{out})$\
 C. $d_{in}+d_{out}+r$\
 D. $r^2$

 **Answer: B**

 Because:

 $$
A=rd_{in}
$$

 and:

 $$
B=d_{out}r
$$

 so:

 $$
N_{LoRA}=r(d_{in}+d_{out})
$$

---

 # Part 16 — Why LoRA can save memory

 ### Q72. Why does LoRA generally require less gradient memory?

 A. The input images become smaller\
 B. Fewer parameters require gradients\
 C. The model has no forward pass\
 D. The validation set is deleted

 **Answer: B**

---

 ### Q73. Why can LoRA substantially reduce optimizer memory?

 A. AdamW needs optimizer state primarily for trainable parameters\
 B. AdamW doesn't store any state\
 C. LoRA removes all parameters\
 D. The classifier disappears

 **Answer: A**

---

 ### Q74. Why might the full LoRA model `state_dict` still be large?

 A. It contains the frozen pretrained weights as well as adapters\
 B. LoRA duplicates every image\
 C. The optimizer is stored inside the model state dict\
 D. The classifier becomes larger

 **Answer: A**

 This is an especially important assignment insight.

---

 # Part 17 — Timing and computation

 ### Q75. Is LoRA guaranteed to make the forward pass faster?

 A. Yes\
 B. No\
 C. Only on CPU\
 D. Only with rank 1

 **Answer: B**

 **Explanation:**\
 LoRA adds an extra computational pathway:

 $$
Wx+BAx
$$

 so the forward pass may actually be slower.

---

 ### Q76. Why might LoRA be slower despite training fewer parameters?

 A. It performs additional matrix operations\
 B. It uses more training examples\
 C. It always doubles the image size\
 D. It disables GPU acceleration

 **Answer: A**

---

 ### Q77. What is the primary computational advantage of LoRA?

 A. It guarantees fewer forward FLOPs\
 B. It reduces the amount of parameter-gradient computation and optimizer state\
 C. It eliminates matrix multiplication\
 D. It eliminates the pretrained model

 **Answer: B**

---

 ### Q78. Why should CUDA timing often use synchronization?

 A. GPU operations can execute asynchronously relative to CPU code\
 B. CUDA cannot perform matrix multiplication\
 C. Synchronization changes the model's weights\
 D. It increases the batch size

 **Answer: A**

---

 # Part 18 — Rank experiments

 ### Q79. What happens to LoRA parameter count as rank $r$ increases?

 A. It decreases\
 B. It stays constant\
 C. It increases linearly with $r$\
 D. It becomes zero

 **Answer: C**

 Because:

 $$
N=r(d_{in}+d_{out})
$$

---

 ### Q80. Which statement about rank is most accurate?

 A. Larger rank always guarantees higher validation accuracy\
 B. Smaller rank always gives higher accuracy\
 C. Larger rank provides greater adaptation capacity but costs more parameters\
 D. Rank has no effect

 **Answer: C**

---

 ### Q81. Why might an extremely small rank perform poorly?

 A. The adapter may not have enough capacity to represent the required update\
 B. The model receives too many parameters\
 C. The target dataset disappears\
 D. The classifier has 100 classes

 **Answer: A**

---

 ### Q82. Why might increasing rank eventually provide diminishing returns?

 A. The target task may not require additional adaptation capacity\
 B. The model stops doing forward passes\
 C. CIFAR-100 becomes unlabeled\
 D. The convolutional layers become trainable

 **Answer: A**

---

 ### Q83. What is the most informative rank experiment?

 A. Compare only rank 1000\
 B. Compare several ranks while keeping other training settings consistent\
 C. Change rank, dataset, learning rate, optimizer, and epochs simultaneously\
 D. Only inspect training loss

 **Answer: B**

 **Explanation:**\
 To study the effect of rank, other experimental conditions should be held approximately constant.

---

 # Part 19 — Training curves

 ### Q84. What does increasing training accuracy while validation accuracy stagnates suggest?

 A. Possible overfitting\
 B. Guaranteed underfitting\
 C. No learning\
 D. Data loading failure

 **Answer: A**

---

 ### Q85. If both training and validation losses decrease steadily, what does that generally indicate?

 A. The model is learning useful patterns\
 B. The optimizer is broken\
 C. The labels are definitely wrong\
 D. The model cannot classify anything

 **Answer: A**

---

 ### Q86. If LoRA reaches nearly the same validation accuracy as dense fine-tuning with far fewer trainable parameters, what does this demonstrate?

 A. LoRA achieved parameter-efficient adaptation successfully\
 B. LoRA is useless\
 C. The target dataset was removed\
 D. Dense fine-tuning failed automatically

 **Answer: A**

---

 # Part 20 — Debugging questions

 ### Q87. Your LoRA model's initial predictions differ significantly from the pretrained model. What should you check first?

 A. Whether B was initialized to zero\
 B. Whether CIFAR-100 has 100 classes\
 C. Whether AdamW uses weight decay\
 D. Whether the plot title is correct

 **Answer: A**

---

 ### Q88. Your matrix multiplication fails with a shape error. Which issue is most likely?

 A. A or B has been given the wrong dimensions\
 B. CIFAR-10 has the wrong number of classes\
 C. The optimizer is incorrect\
 D. `torch.manual_seed()` wasn't called

 **Answer: A**

---

 ### Q89. For:

```
nn.Linear(256, 1024)
```

 which LoRA A shape is correct for rank 4?

 A. `(1024, 4)`\
 B. `(4, 256)`\
 C. `(256, 4)`\
 D. `(4, 1024)`

 **Answer: B**

---

 ### Q90. For the same layer, which B shape is correct?

 A. `(4, 256)`\
 B. `(256, 4)`\
 C. `(1024, 4)`\
 D. `(4, 1024)`

 **Answer: C**

---

 # Part 21 — State dictionaries

 ### Q91. What does:

```
model.state_dict()
```

 primarily contain?

 A. Model parameters and persistent buffers\
 B. Training images\
 C. Only gradients\
 D. Only optimizer statistics

 **Answer: A**

---

 ### Q92. What does:

```
optimizer.state_dict()
```

 contain?

 A. Only model architecture\
 B. Optimizer configuration and optimizer state\
 C. Images and labels\
 D. Only validation metrics

 **Answer: B**

---

 ### Q93. Why might the optimizer state be dramatically smaller for LoRA?

 A. Far fewer parameters are being optimized\
 B. LoRA doesn't use an optimizer\
 C. LoRA removes all model parameters\
 D. AdamW is disabled

 **Answer: A**

---

 # Part 22 — Optional merging

 ### Q94. After training, how can LoRA be merged into the original weights?

 A. $W_{merged}=W-BA$\
 B. $W_{merged}=W+BA$\
 C. $W_{merged}=BA-W$\
 D. $W_{merged}=A+B$

 **Answer: B**

---

 ### Q95. Why is merging useful for inference?

 A. It removes the need to separately compute the LoRA branch\
 B. It increases the rank\
 C. It deletes the classifier\
 D. It requires retraining

 **Answer: A**

 After merging:

 $$
W'=W+BA
$$

 and inference can simply use:

 $$
z=W'x+b
$$

---

 ### Q96. If the LoRA adapter is merged correctly, should the merged model's output match the unmerged LoRA model?

 A. Yes, up to numerical precision\
 B. No, never\
 C. Only if rank is zero\
 D. Only during training

 **Answer: A**

 **Explanation:**

 Unmerged:

 $$
Wx+BAx+b
$$

 Merged:

 $$
(W+BA)x+b
$$

 Mathematically they are equivalent.

---

 # Part 23 — Adapter-only saving

 ### Q97. Why might saving only A and B be useful?

 A. It creates a much smaller task-specific checkpoint\
 B. It deletes the pretrained model permanently\
 C. It increases training accuracy\
 D. It removes the need for labels

 **Answer: A**

---

 ### Q98. What would an adapter-only checkpoint typically contain?

 A. All CIFAR-100 images\
 B. Only A/B matrices for the adapted layers, possibly classifier parameters\
 C. Every frozen convolutional weight\
 D. The entire optimizer state

 **Answer: B**

---

 # Part 24 — Advanced conceptual questions

 ### Q99. Which statement best describes the assumption behind LoRA?

 A. All pretrained weights are useless\
 B. The required task-specific weight update can often be represented approximately in a low-dimensional subspace\
 C. Every target task requires a completely new model\
 D. Neural networks cannot transfer knowledge

 **Answer: B**

---

 ### Q100. Why does the low-rank constraint reduce parameter count?

 A. A full $d_{out}\times d_{in}$ matrix is represented through two smaller matrices involving rank $r$\
 B. It removes the input\
 C. It changes the number of classes\
 D. It eliminates the forward pass

 **Answer: A**

---

 # Part 25 — Code-tracing challenge

 ### Q101. Consider:

```
layer = nn.Linear(100, 50)
r = 5
```

 What are the shapes of W, A, and B?

 A. W `(100,50)`, A `(5,50)`, B `(100,5)`\
 B. W `(50,100)`, A `(5,100)`, B `(50,5)`\
 C. W `(50,100)`, A `(100,5)`, B `(5,50)`\
 D. All are `(5,5)`

 **Answer: B**

 **Explanation:**

 PyTorch stores:

 $$
W=(d_{out},d_{in})=(50,100)
$$

 LoRA:

 $$
A=(r,d_{in})=(5,100)
$$

 $$
B=(d_{out},r)=(50,5)
$$

---

 ### Q102. For the previous example, how many original weight parameters exist?

 A. 500\
 B. 5,000\
 C. 10,000\
 D. 50,000

 **Answer: B**

 $$
50\times100=5000
$$

---

 ### Q103. How many LoRA parameters are required with $r=5$?

 A. 250\
 B. 500\
 C. 750\
 D. 5,000

 **Answer: C**

 $$
5(100+50)=750
$$

---

 ### Q104. What fraction of the original weight parameter count is that?

 A. 5%\
 B. 10%\
 C. 15%\
 D. 50%

 **Answer: C**

 $$
\frac{750}{5000}=0.15
$$

 So 15%.

---

 # Part 26 — Understanding the complete experiment

 ### Q105. Which sequence correctly represents the notebook?

 A. LoRA → CIFAR-10 → dense fine-tuning\
 B. CIFAR-100 → random model → CIFAR-10\
 C. CIFAR-10 pretraining → dense target fine-tuning → LoRA target fine-tuning → comparison\
 D. CIFAR-100 pretraining → LoRA → CIFAR-10

 **Answer: C**

---

 ### Q106. Why should dense and LoRA experiments use the same target dataset?

 A. To make the comparison fair\
 B. Because LoRA only works on CIFAR-100\
 C. Because dense fine-tuning cannot use CIFAR-10\
 D. Because both datasets have identical labels

 **Answer: A**

---

 ### Q107. Why should the same training hyperparameters initially be used for baseline and LoRA?

 A. To provide a controlled comparison\
 B. Because LoRA cannot use different learning rates\
 C. Because the optimizer requires it\
 D. Because ranks become equal

 **Answer: A**

 **Explanation:**\
 The notebook explicitly says to fine-tune the LoRA model with the same hyperparameters as the baseline initially. Later you can investigate whether LoRA benefits from different settings.

---

 # Part 27 — Important “trap” questions

 ### Q108. Which statement is FALSE?

 A. LoRA freezes the original pretrained weight matrices\
 B. LoRA introduces trainable A and B matrices\
 C. LoRA always makes the forward pass faster\
 D. LoRA can substantially reduce trainable parameter count

 **Answer: C**

 **Explanation:**\
 LoRA can actually add computation to the forward pass.

---

 ### Q109. Which statement is FALSE?

 A. The LoRA update has the same shape as W\
 B. A and B individually are smaller than W\
 C. $BA$ can have the same shape as W\
 D. A and B must have the same shape as W

 **Answer: D**

 A and B are deliberately smaller.

---

 ### Q110. Which statement is FALSE?

 A. A zero B matrix makes the initial LoRA update zero\
 B. The original pretrained model can therefore be preserved initially\
 C. Both A and B should always be initialized to zero\
 D. A is typically randomly initialized

 **Answer: C**

---

 ### Q111. Which statement is FALSE?

 A. A smaller rank reduces adapter parameters\
 B. A larger rank increases adapter capacity\
 C. Increasing rank always guarantees better validation accuracy\
 D. Rank is a hyperparameter

 **Answer: C**

---

 # Part 28 — Master-level reasoning

 ### Q112. Suppose dense fine-tuning achieves 80% validation accuracy and LoRA achieves 79%. Which conclusion is most appropriate?

 A. LoRA failed because it isn't exactly 80%\
 B. LoRA may be a successful parameter-efficient alternative because the performance gap is small\
 C. Dense fine-tuning is always preferable\
 D. LoRA must have been implemented incorrectly

 **Answer: B**

 **Explanation:**\
 The assignment specifically says LoRA should ideally be within a few percentage points of the baseline.

---

 ### Q113. Suppose rank 1 achieves 72%, rank 4 achieves 78%, rank 8 achieves 80%, and dense achieves 79%. What is the most interesting conclusion?

 A. Rank doesn't matter\
 B. Rank 8 provides sufficient adaptation capacity and happens to outperform the dense baseline in this experiment\
 C. Rank 1 is optimal because it uses fewer parameters\
 D. Dense fine-tuning must be broken

 **Answer: B**

 **Explanation:**\
 The assignment specifically asks you to investigate whether a rank exists that can outperform the dense baseline.

---

 ### Q114. Why could LoRA outperform dense fine-tuning on validation accuracy despite using fewer trainable parameters?

 A. The low-rank constraint can act as a form of regularization\
 B. LoRA always has more parameters\
 C. Dense fine-tuning never learns\
 D. LoRA changes the target labels

 **Answer: A**

 **Explanation:**\
 Restricting the update to a lower-dimensional space can reduce unnecessary adaptation and potentially improve generalization on a small target dataset.

---

 ### Q115. What does the statement:

 > "The bias vector is already low-rank"

 mean in the context of the practical?

 A. Bias has only one dimension, so there is no need to create a low-rank factorization for it\
 B. Bias contains a matrix of rank zero\
 C. Bias should always be removed\
 D. Bias must be converted to a convolution

 **Answer: A**

 **Explanation:**\
 A bias is already a vector rather than a large matrix. The expensive parameter matrices are the primary target for LoRA.

---

 ### Q116. Why are the MLP layers particularly convenient for demonstrating LoRA in this practical?

 A. They use simple matrix multiplication with clearly defined weight matrices\
 B. They have no parameters\
 C. They cannot be frozen\
 D. They contain images

 **Answer: A**

 **Explanation:**\
 For a linear layer, the LoRA transformation is especially transparent:

 $$
Wx\rightarrow Wx+BAx
$$

---

 # Part 29 — Final exam-style questions

 ### Q117. Which best describes the relationship between dense fine-tuning and the LoRA abstraction?

 A. Dense fine-tuning changes W directly; LoRA freezes W and learns an approximation to its change\
 B. Dense fine-tuning and LoRA use completely unrelated objectives\
 C. LoRA removes W\
 D. Dense fine-tuning only updates biases

 **Answer: A**

---

 ### Q118. Which equation shows that the LoRA formulation is mathematically equivalent to adding a weight update?

 A.

 $$
Wx+BAx=(W+BA)x
$$

 B.

 $$
Wx+BAx=W(A+B)x
$$

 C.

 $$
Wx+BAx=(W+A+B)x
$$

 D.

 $$
Wx+BAx=WB+Ax
$$

 **Answer: A**

---

 ### Q119. If $r=d_{in}$ and the dimensions permit a sufficiently expressive factorization, what happens conceptually?

 A. The low-rank restriction becomes much less restrictive\
 B. The adapter has zero parameters\
 C. W becomes zero\
 D. The classifier disappears

 **Answer: A**

 **Explanation:**\
 As rank approaches the dimensions of the original matrix, the low-rank approximation becomes increasingly expressive. The parameter-efficiency advantage also decreases.

---

 ### Q120. What is the most complete description of why LoRA is useful?

 A. It guarantees better accuracy\
 B. It replaces all neural network layers\
 C. It allows task-specific adaptation using relatively few trainable parameters while preserving the pretrained model\
 D. It eliminates the need for training

 **Answer: C**

---

 # Final mastery checklist

 If you can answer these questions comfortably, you should understand essentially everything the practical is testing.

 You should be able to explain all of the following **without looking at the notebook**:

 ### Dataset

 - Why CIFAR-10 is the source domain
- Why a CIFAR-100 superclass is the target domain
- Why target labels need remapping
- Why the data is split into training/validation sets
- What normalization and batching do

 ### Model

 - Conv1 → pooling → Conv2 → global pooling
- Why the MLP layers are large
- The input/output dimensions of `fc1`–`fc4`
- Why the classifier has 10 outputs during pretraining and 5 during the target task

 ### Training

 - `forward()`
- `CrossEntropyLoss`
- `backward()`
- `zero_grad()`
- `optimizer.step()`
- `train()` vs `eval()`
- `torch.no_grad()`
- AdamW
- weight decay

 ### Transfer learning

 - What pretrained weights are reused
- Why the classifier is removed/reinitialized
- Why convolutional layers are frozen
- What dense fine-tuning means

 ### LoRA

 You should be able to derive:

 $$
\boxed{z=Wx+BAx+b}
$$

 and explain:

 $$
\boxed{\Delta W\approx BA}
$$

 with:

 $$
\boxed{A\in\mathbb{R}^{r\times d_{in}}}
$$

 $$
\boxed{B\in\mathbb{R}^{d_{out}\times r}}
$$

 and:

 $$
\boxed{N_{LoRA}=r(d_{in}+d_{out})}
$$

 ### Implementation

 You should know:

 - How to create A
- How to create B
- Why B starts at zero
- Why A is randomly initialized
- How to freeze W
- How to implement the forward pass
- Why matrix orientation matters
- How to test that the initial LoRA layer matches the original layer

 ### Evaluation

 You should know how to compare:

 - total parameters
- trainable parameters
- validation accuracy
- training curves
- state-dict size
- optimizer state size
- forward time
- backward time
- GPU memory
- different values of rank $r$

 ### Deep conceptual takeaway

 The entire practical can ultimately be reduced to this comparison:

 $$
\boxed{\text{Dense: } W\rightarrow W+\Delta W}
$$

 versus:

 $$
\boxed{\text{LoRA: } W\text{ frozen},\quad \Delta W\approx BA}
$$

 The **fundamental trade-off** is:

 > **LoRA sacrifices some flexibility and may add computation in the forward pass, in exchange for dramatically reducing the number of parameters that need to be trained and the associated gradient/optimizer memory.**

 That trade-off—not merely memorizing the `LoraLayer` code—is the central concept the assignment is testing.
