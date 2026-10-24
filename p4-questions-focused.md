Below is Chunk 1 only, covering Questions 1–30 and following the actual Practical 4 notebook. It is a standalone Markdown document ready to copy into `practical-4-lora-part-1.md`.

I’ll stop at Q30. I will not produce Chunk 2 until you have had a chance to copy and review this document.

Writing

# Advanced Machine Learning — Practical 4: Low-Rank Adaptation (LoRA)

## Part 1: Questions 1–30

**Suggested filename:** `practical-4-lora-part-1.md`

---

## Section 1 — Assignment Overview and Transfer Learning

### Q1. What is the objective of Practical 4?

The objective is to implement and evaluate **Low-Rank Adaptation (LoRA)**, a parameter-efficient fine-tuning technique, using PyTorch.

The notebook follows this workflow:

1. Pre-train a neural network on CIFAR-10.
2. Select a five-class subset of CIFAR-100 as the target domain.
3. Fine-tune the pre-trained network using dense fine-tuning as a baseline.
4. Replace selected fully connected layers with LoRA adapters.
5. Compare the two fine-tuning approaches in terms of trainable parameters, accuracy, training dynamics, memory consumption, and computational speed.

The central question is whether LoRA can achieve target-domain accuracy comparable to dense fine-tuning while substantially reducing the number of trainable parameters.

The notebook also asks us to investigate model and optimizer state sizes, test different LoRA ranks, and optionally merge the trained adapters into the original model.

### Q2. What is transfer learning, and why is it useful here?

**Transfer learning** uses knowledge acquired on one task or dataset to improve performance on another related task.

In this practical, the source dataset is CIFAR-10, while the target dataset is a selected superclass from CIFAR-100.

The model first learns visual features from the source dataset. These features can then be reused when adapting the network to the target classes.

For example, a network trained on CIFAR-10 may learn features such as edges, textures, shapes, and object parts. Some of these features remain useful when classifying fish from CIFAR-100.

Transfer learning is useful because the target dataset is relatively small, and learning an effective representation from scratch may require more data and training.

The practical compares two adaptation strategies:

- **Dense fine-tuning:** update the trainable weights of the selected layers.
- **LoRA fine-tuning:** freeze the original weights and learn low-rank updates to selected weight matrices.

### Q3. What are the source domain and target domain in the notebook?

The source domain is the original CIFAR-10 dataset:

- 10 classes.
- 50,000 training images in the original training split.
- 10,000 test images in the original test split.

The target domain is a selected group of five classes from CIFAR-100.

The notebook sets:

python

```
TARGET_CATEGORY = 'fish'
```

The selected classes are:

- `aquarium_fish`
- `flatfish`
- `ray`
- `shark`
- `trout`

These classes form the target classification task.

The original CIFAR-100 training split contains 50,000 images, with 500 images per class. Therefore, the selected five-class target subset contains 2,500 images before the notebook's train-validation split.

The notebook uses the dataset objects returned by `torchvision.datasets.CIFAR10` and `torchvision.datasets.CIFAR100` without explicitly requesting their test splits. Consequently, the validation data here comes from the original training datasets, not the official CIFAR test sets.

### Q4. What is the difference between pre-training, dense fine-tuning, and LoRA fine-tuning?

| Stage | Training data | Trainable parameters | Purpose |
| --- | --- | --- | --- |
| Pre-training | CIFAR-10 | All model parameters | Learn general visual representations |
| Dense fine-tuning baseline | Selected CIFAR-100 classes | MLP and classifier parameters | Adapt the representation to the target task |
| LoRA fine-tuning | Same target dataset | LoRA adapter matrices and classifier parameters | Adapt the model with fewer trainable parameters |

The convolutional layers are frozen in both target-domain experiments.

This is an important qualification: the baseline is called dense fine-tuning in the notebook, but it is **not full-network fine-tuning**. It updates the selected fully connected layers and the new classifier while keeping both convolutional layers frozen.

LoRA further reduces the trainable parameter count by freezing the original weights of the adapted fully connected layers.

### Q5. Why is parameter-efficient fine-tuning useful?

Fine-tuning a large pre-trained network can be expensive because it requires computing gradients and maintaining optimizer state for trainable parameters.

Parameter-efficient fine-tuning aims to reduce this burden by updating only a small set of additional parameters.

Potential benefits include:

- Fewer trainable parameters.
- Smaller optimizer states.
- Lower memory requirements for trainable gradients and optimizer data.
- More efficient storage of task-specific adapters.
- Easier management of multiple task-specific adaptations.

However, fewer trainable parameters do not automatically imply faster execution. LoRA introduces extra operations to compute the adapter contribution, so forward and backward execution times must be measured.

Similarly, the full LoRA model may still contain the original frozen weights. A reduction in trainable parameters is not necessarily the same as an equally large reduction in total model size.

---

## Section 2 — Dataset Preparation and Data Loading

### Q6. How does the notebook download and extract CIFAR-10 and CIFAR-100?

The notebook defines a helper function that downloads a compressed dataset archive if it does not already exist, then extracts it into the data directory.

python

```
def download_from_mirror(gdown_uri, dataset_name):
    tar_path = os.path.join(
        data_root,
        dataset_name + '.tar.gz'
    )

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if not os.path.exists(tar_path):
        gdown.download(
            id=gdown_uri,
            output=tar_path,
            quiet=False
        )
        print(f'Data tarball downloaded successfully: {tar_path}')
    else:
        print(f'Detected existing tarball: {tar_path}')

    data_dir = os.path.join(data_root, dataset_name)

    if not os.path.exists(data_dir):
        datasets.utils.extract_archive(
            tar_path,
            data_root
        )
        print(f'{dataset_name} extracted successfully.')
```

The root directory is:

python

```
data_root = 'data/'
```

The function avoids downloading an archive that already exists and avoids extracting it again when the expected extracted directory exists.

**Practical considerations:**

- The download IDs must refer to accessible archives.
- The runtime needs internet access when the files are not already present.
- The archive must have the expected directory structure for the subsequent `torchvision` dataset loader.
- The notebook uses `download=False` when constructing the datasets because it expects the data to have already been extracted.

If the download or extraction fails, check the file paths, archive contents, and permissions before training.

### Q7. How does the notebook create the five-class CIFAR-100 target subset?

The notebook defines a dictionary mapping each superclass name to five CIFAR-100 class names.

For the fish superclass:

python

```
TARGET_CATEGORY = 'fish'

superclass_names = cifar_superclasses[TARGET_CATEGORY]

superclass_idxs = set(
    cifar100.class_to_idx[name]
    for name in superclass_names
)

subset_idxs = [
    i for i, idx in enumerate(cifar100.targets)
    if idx in superclass_idxs
]

target_data = torch.utils.data.Subset(
    cifar100,
    subset_idxs
)
```

The `class_to_idx` dictionary converts class names into their original integer labels.

The code then identifies every example whose original label belongs to the selected five classes and creates a `Subset` containing those examples.

This selection does not modify the underlying images. It changes which examples are included in the target dataset.

The resulting target dataset contains five classes and, for the original CIFAR-100 training split, 2,500 examples.

### Q8. Why does the notebook remap the target labels?

CIFAR-100 originally assigns labels from 0 to 99. The target classifier, however, needs only five output classes.

The notebook creates a mapping:

python

```
target_data.remap = {
    cifar100.class_to_idx[name]: i
    for i, name in enumerate(superclass_names)
}

old_targets = [
    label for img, label in target_data
]

target_data.labels = [
    target_data.remap[label]
    for label in old_targets
]

target_data.classes = superclass_names
target_data.num_classes = len(superclass_names)
```

The resulting mapping for the fish superclass follows the order in the dictionary:

| Target label | Class |
| --- | --- |
| 0 | aquarium\_fish |
| 1 | flatfish |
| 2 | ray |
| 3 | shark |
| 4 | trout |

The classifier must produce five logits, and the target labels must lie between 0 and 4.

**Important implementation detail:** `target_data` is a `torch.utils.data.Subset`, so the notebook attaches custom attributes to it. The underlying CIFAR-100 dataset still returns the original labels. The collate function must therefore perform the remapping whenever it constructs a batch.

### Q9. How are the training and validation splits created?

The notebook sets:

python

```
val_frac = 0.2
batch_size = 256

torch.manual_seed(0)

train_data_source, val_data_source = (
    torch.utils.data.random_split(
        source_data,
        [1 - val_frac, val_frac]
    )
)

train_data_target, val_data_target = (
    torch.utils.data.random_split(
        target_data,
        [1 - val_frac, val_frac]
    )
)
```

The intended split is 80% training and 20% validation.

For CIFAR-10, this gives:

- 40,000 training examples.
- 10,000 validation examples.

For the selected CIFAR-100 superclass, this gives:

- 2,000 training examples.
- 500 validation examples.

**Potential compatibility issue:** In PyTorch versions where `random_split` requires integer lengths, passing `[0.8, 0.2]` will fail. A robust version explicitly computes integer lengths:

python

```
n_val_source = int(
    len(source_data) * val_frac
)

n_train_source = (
    len(source_data) - n_val_source
)

train_data_source, val_data_source = (
    torch.utils.data.random_split(
        source_data,
        [n_train_source, n_val_source],
        generator=torch.Generator().manual_seed(0)
    )
)

n_val_target = int(
    len(target_data) * val_frac
)

n_train_target = (
    len(target_data) - n_val_target
)

train_data_target, val_data_target = (
    torch.utils.data.random_split(
        target_data,
        [n_train_target, n_val_target],
        generator=torch.Generator().manual_seed(1)
    )
)
```

The explicit generators make the splits reproducible and independent of the global random-number-generator state.

### Q10. What do the preprocessing and collate functions do?

The notebook defines the following image transformation:

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

`ToTensor()` converts a PIL image into a floating-point tensor with channel-first shape:

$$
(C,H,W)
$$

For standard 8-bit RGB images, it also scales pixel values from the range $[0,255]$ to approximately $[0,1]$.

`Normalize()` applies the following transformation independently to each RGB channel:

$$
x'=\frac{x-\mu}{\sigma}
$$

Here, the mean is $0.5$ and the standard deviation is $0.2$ for every channel.

The custom collate functions then:

1. Collect images and labels from the batch.
2. Transform each image.
3. Stack the image tensors into a batch.
4. Convert labels into a `torch.long` tensor.
5. Move the batch to the selected device.

The source collate function uses the original CIFAR-10 labels. The target collate function remaps CIFAR-100 labels to the selected five-class range.

**Reproducibility note:** The same normalization is used for both datasets. This is consistent across source and target training, although the normalization constants are chosen by the notebook rather than estimated from the datasets.

---

## Section 3 — Model Architecture and Training Utilities

### Q11. What is the architecture of the `ConvMLP` model?

The notebook uses a small convolutional front end followed by a multilayer perceptron.

python

```
class ConvMLP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels=3
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            input_channels, 64, 3, padding=1
        )
        self.maxpool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(
            64, 256, 3, padding=1
        )
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)

        self.classifier = nn.Linear(
            256, num_classes
        )
```

The forward pass applies the convolutional layers and pooling operations, converts the resulting feature map into a vector, and feeds it through the fully connected layers.

For a CIFAR image with shape $3\times32\times32$:

| Operation | Output shape, excluding batch dimension |
| --- | --- |
| Input | $3\times32\times32$ |
| `conv1` \+ ReLU | $64\times32\times32$ |
| `maxpool` | $64\times16\times16$ |
| `conv2` \+ ReLU | $256\times16\times16$ |
| Global average pooling | $256\times1\times1$ |
| Flatten | $256$ |
| `fc1` | $1024$ |
| `fc2` | $512$ |
| `fc3` | $512$ |
| `fc4` | $256$ |
| `classifier` | 10 or 5 logits |

The architecture is deliberately chosen to make LoRA implementation straightforward for fully connected layers. LoRA can also be applied to convolutional weights, but this practical focuses on linear layers.

### Q12. How does the `ConvMLP` forward pass work?

The notebook defines:

python

```
def forward(self, x):
    x = self.maxpool(
        torch.relu(self.conv1(x))
    )

    x = self.globalpool(
        torch.relu(self.conv2(x))
    )

    x = x.view(-1, self.conv2.out_channels)

    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))

    x = self.classifier(x)

    return x
```

The network performs these operations:

1. The first convolution extracts local image features.
2. Max pooling reduces the spatial resolution.
3. The second convolution produces 256 feature maps.
4. Global average pooling reduces every feature map to one scalar.
5. The feature vector passes through four fully connected layers.
6. The classifier converts the final 256-dimensional representation into class logits.

ReLU introduces nonlinearity after each convolution and intermediate fully connected layer.

The final classifier has no softmax because `nn.CrossEntropyLoss` expects raw logits and internally combines log-softmax with negative log-likelihood loss.

### Q13. How many parameters does the model contain?

For a linear layer with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs, including its bias, the parameter count is:

$$
N_{\text{linear}}
=d_{\text{in}}d_{\text{out}}+d_{\text{out}}
$$

A convolutional layer with $C_{\text{in}}$ input channels, $C_{\text{out}}$ output channels, a $k\times k$ kernel, and a bias has:

$$
N_{\text{conv}}
=C_{\text{out}}C_{\text{in}}k^2+C_{\text{out}}
$$

For the architecture in the notebook, the parameter counts are:

| Layer | Parameters |
| --- | --- |
| `conv1` | 1,792 |
| `conv2` | 147,712 |
| `fc1` | 263,168 |
| `fc2` | 524,800 |
| `fc3` | 262,656 |
| `fc4` | 131,328 |
| Classifier, 10 classes | 2,570 |
| **Total, CIFAR-10 model** | **1,334,026** |
| Classifier, 5 classes | 1,285 |
| **Total, target model** | **1,332,741** |

These totals include biases.

The classifier has fewer parameters for the five-class target task because its output dimension decreases from 10 to 5.

The notebook's `num_parameters()` helper calculates the total and trainable parameter counts directly:

python

```
def num_parameters(model: nn.Module):
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_params, trainable_params
```

This function counts parameter elements, not bytes. The actual storage size depends on the data type and how the model is serialized.

### Q14. What does `requires_grad` mean, and how is it used in fine-tuning?

In PyTorch, a parameter with `requires_grad=True` participates in gradient computation during backpropagation.

A parameter with `requires_grad=False` is excluded from gradient computation.

The notebook freezes the convolutional layers in the baseline:

python

```
baseline_ft_model.conv1.requires_grad_(False)
baseline_ft_model.conv2.requires_grad_(False)
```

This prevents their weights and biases from being updated during target-domain training.

The fully connected layers and classifier remain trainable in the baseline.

For LoRA, the original weights of the adapted fully connected layers will also be frozen, while the low-rank adapter matrices remain trainable.

**Important distinction:** Freezing a parameter does not remove it from the model. Its values are still used during the forward pass and are generally still included in the model's `state_dict`.

### Q15. What is the role of `TrainingMetrics`?

`TrainingMetrics` stores the training and validation metrics needed to monitor learning.

Its principal attributes are:

python

```
self.train_loss = []
self.val_loss = []
self.train_acc = []
self.val_acc = []
```

The training metrics are logged at each batch. Validation metrics are logged at the end of each epoch.

The class also tracks the global training step and the step at which each validation run occurs. This allows it to calculate average training loss and accuracy over an epoch.

The `best_val_acc` property returns the maximum recorded validation accuracy:

python

```
@property
def best_val_acc(self):
    return np.max(self.val_acc)
```

The plotting method displays loss and accuracy curves for both training and validation, optionally overlaying the baseline results.

These curves are useful for comparing convergence speed, overfitting, and the effect of different LoRA ranks.

However, the best validation accuracy is not necessarily the final validation accuracy. Reports should distinguish the two.

### Q16. How does the training loop work?

The notebook's `train_model()` function uses cross-entropy loss and the AdamW optimizer.

Its central operations are:

python

```
loss_func = nn.CrossEntropyLoss()

opt = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=l2_reg
)
```

For each training batch, it:

1. Enters training mode with `model.train()`.
2. Retrieves the images and labels.
3. Clears the existing gradients using `opt.zero_grad()`.
4. Computes predictions with `model(x)`.
5. Computes the cross-entropy loss.
6. Calls `batch_loss.backward()` to calculate gradients.
7. Calls `opt.step()` to update trainable parameters.
8. Logs the batch loss and accuracy.

At the end of each epoch, the model is evaluated on the validation loader.

The same general training loop is used for dense fine-tuning and LoRA fine-tuning, making the experiments easier to compare.

One detail is that the optimizer is constructed from `model.parameters()`, which includes frozen parameters. PyTorch's optimizer does not update parameters whose gradients are absent, so the frozen weights remain unchanged. For clarity and potentially simpler optimizer state management, an implementation can instead pass only parameters with `requires_grad=True`.

### Q17. How are validation loss and accuracy calculated?

The notebook evaluates the model with:

python

```
model.eval()

with torch.no_grad():
    ...
```

`model.eval()` switches modules such as dropout and batch normalization into evaluation mode, where applicable. This particular architecture does not contain either of those layers.

`torch.no_grad()` disables gradient recording during evaluation, reducing unnecessary computation and memory use.

The classification accuracy is calculated as:

python

```
def top1_acc(pred, y):
    return (
        pred.argmax(axis=1) == y
    ).float().mean().item()
```

For each image, the predicted class is the index of the largest logit.

The accuracy is the proportion of predictions that match the ground-truth labels.

**Metric aggregation caveat:** The notebook averages batch-level validation losses and accuracies without weighting them by batch size. Because the final batch may be smaller than the others, this can give it equal weight to a full batch. A more exact dataset-level metric accumulates total loss and correct predictions weighted by the number of examples.

A robust accuracy calculation is:

$$
\text{Accuracy}
=\frac{\text{Number of correct predictions}}
{\text{Total number of examples}}
$$

For a five-class balanced target validation set, random uniform guessing has an expected accuracy of 20%. This is a theoretical reference, not a measured result from the notebook.

---

## Section 4 — Pre-Training and Dense Fine-Tuning Baseline

### Q18. How is the model pre-trained on CIFAR-10?

The notebook initializes a ten-class model and trains it on the CIFAR-10 training loader:

python

```
torch.manual_seed(1)

pt_model = ConvMLP(
    num_classes=source_data.num_classes
).to(device)

num_pt_epochs = 20
lr = 2e-3
l2_reg = 1e-3

pt_metrics = train_model(
    pt_model,
    train_loader_source,
    val_loader_source,
    num_pt_epochs,
    lr,
    l2_reg,
    title='Baseline Pre-training (CIFAR-10)'
)
```

The configuration uses:

- 20 epochs.
- Learning rate $2\times10^{-3}$.
- AdamW weight decay $10^{-3}$.
- Cross-entropy loss.
- A batch size of 256.

After training, the notebook prints the best validation accuracy and inspects some predictions.

The model's weights are saved as:

python

```
pt_save_name = 'pretrained_model.ckpt'

torch.save(
    pt_model.state_dict(),
    pt_save_name
)
```

The saved state dictionary contains the model's learned parameters and buffers, not the entire Python model definition.

**Execution requirement:** The notebook sets `device = 'cuda'`, so the runtime needs a compatible CUDA-enabled GPU. On a CPU-only machine, the device setting must be changed to `'cpu'`, although training will generally be slower.

### Q19. Why is the pre-trained classifier removed before fine-tuning?

The source classifier maps the 256-dimensional representation to 10 CIFAR-10 classes:

python

```
self.classifier = nn.Linear(256, 10)
```

The target task has five classes, so the target classifier must instead have shape:

$$
W_{\text{target}}\in\mathbb{R}^{5\times256}
$$

The notebook initializes a new five-class model and deletes the classifier parameters from the pre-trained state dictionary:

python

```
pt_state_dict = torch.load(pt_save_name)

del pt_state_dict['classifier.weight']
del pt_state_dict['classifier.bias']

baseline_ft_model.load_state_dict(
    pt_state_dict,
    strict=False
)
```

The feature-extraction layers retain their pre-trained values, while the target classifier keeps its newly initialized parameters.

The `strict=False` option permits the target model's missing classifier keys to remain initialized rather than requiring an exact match with the checkpoint.

This is an essential part of transfer learning: the learned representation is reused, but the task-specific prediction head is replaced.

**Security and compatibility note:** When loading checkpoints from a trusted source, use an explicit device mapping when necessary. In current PyTorch environments, consider:

python

```
pt_state_dict = torch.load(
    pt_save_name,
    map_location=device,
    weights_only=True
)
```

The `weights_only` argument is available in modern PyTorch releases. For older versions, check the installed API before using it.

### Q20. How is the dense fine-tuning baseline configured?

The baseline model is initialized with five output classes, the convolutional layers are frozen, and the pre-trained feature weights are loaded.

The notebook uses:

python

```
num_ft_epochs = 30
lr = 1e-4
l2_reg = 1e-2
```

It then calls:

python

```
baseline_ft_metrics = train_model(
    baseline_ft_model,
    train_loader_target,
    val_loader_target,
    num_ft_epochs,
    lr,
    l2_reg,
    title=(
        f'Baseline Dense Fine-tuning '
        f'({target_data.name})'
    )
)
```

The baseline trains:

- `fc1`
- `fc2`
- `fc3`
- `fc4`
- `classifier.weight`
- `classifier.bias`

The convolutional weights and biases are frozen.

This baseline is important because LoRA should be evaluated against the same target task, data split, model initialization, and training hyperparameters wherever possible.

A fair comparison also requires consistent measurement procedures and random seeds. The notebook initializes both target models with the same seed, but exact reproducibility can still depend on the environment, GPU kernels, data-loader behavior, and other sources of randomness.

---

## Section 5 — LoRA Theory and Mathematics

### Q21. What is Low-Rank Adaptation (LoRA)?

**LoRA is a parameter-efficient fine-tuning method that freezes an existing weight matrix and learns a low-rank update to it.**

Suppose a pre-trained linear layer has weight matrix $W$. During ordinary fine-tuning, its parameters are updated directly.

LoRA instead represents the adapted weight as:

$$
W'=W+\Delta W
$$

where $W$ is frozen and $\Delta W$ is learned through two smaller matrices.

For a linear layer, the forward pass becomes:

$$
y=xW^\top+x\Delta W^\top+b
$$

The original weight contributes the pre-trained representation, while the additional term allows the model to adapt to the target task.

The key idea is that the required change to the original weight matrix may be approximated adequately by a low-rank matrix.

This can dramatically reduce the number of parameters that need to be optimized.

### Q22. Why does LoRA use a low-rank factorization?

Consider a weight matrix:

$$
W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
$$

A full weight update would require:

$$
d_{\text{out}}d_{\text{in}}
$$

trainable parameters.

LoRA approximates the update using two matrices:

$$
B\in\mathbb{R}^{d_{\text{out}}\times r}
$$

$$
A\in\mathbb{R}^{r\times d_{\text{in}}}
$$

The update is:

$$
\Delta W=BA
$$

The number of trainable adapter parameters is:

$$
N_{\text{LoRA}}
=r d_{\text{out}}+r d_{\text{in}}
$$

or:

$$
\boxed{
N_{\text{LoRA}}
=r(d_{\text{in}}+d_{\text{out}})
}
$$

When $r$ is much smaller than both dimensions, this is much smaller than the full matrix.

For example, for a $1024\times512$ weight matrix and rank $r=8$:

$$
N_{\text{dense}}=1024\times512=524288
$$

$$
N_{\text{LoRA}}=8(1024+512)=12288
$$

The adapter has 12,288 trainable weight parameters instead of 524,288.

This is approximately a 97.66% reduction in the number of trainable weight parameters for that matrix, excluding any bias terms that may be trained separately.

This example is illustrative; it is not a measured result from a training run.

### Q23. What is the mathematical difference between dense fine-tuning, the abstract update, and LoRA?

The notebook introduces these three formulations.

**1\. Dense fine-tuning**

The weight matrix itself is updated:

$$
z=\operatorname{ReLU}(W'x+b)
$$

Here, $W'$ is learned through gradient descent.

**2\. Abstract weight update**

The original weight is frozen, and a full-size residual matrix is learned:

$$
z=\operatorname{ReLU}(Wx+\Delta Wx+b)
$$

Equivalently, using the notebook's row-vector convention for inputs:

$$
z=\operatorname{ReLU}(xW^\top+x\Delta W^\top+b)
$$

Since $W'=W+\Delta W$, the two formulations can represent the same adapted weight matrix.

**3\. LoRA**

The residual is factorized into two low-rank matrices:

$$
z=\operatorname{ReLU}(Wx+BAx+b)
$$

This expression uses a column-vector convention. For the notebook's PyTorch linear layers, which take row-oriented batches, the equivalent expression is:

$$
\boxed{
z=\operatorname{ReLU}
\left(xW^\top+x(BA)^\top+b\right)
}
$$

Here:

- $W$ is the frozen pre-trained weight matrix.
- $A$ and $B$ are trainable low-rank factors.
- $BA$ approximates the desired weight update.
- $b$ is the bias.

The distinction is that dense fine-tuning learns every selected weight directly, the abstract method learns a full-size update, and LoRA learns only the low-rank factors.

### Q24. What are the dimensions of the LoRA matrices, and why do they matter?

PyTorch stores the weight of an `nn.Linear(in_features, out_features)` layer with shape:

$$
W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
$$

For the low-rank update:

$$
A\in\mathbb{R}^{r\times d_{\text{in}}}
$$

$$
B\in\mathbb{R}^{d_{\text{out}}\times r}
$$

Therefore:

$$
BA\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
$$

The matrix product has the same dimensions as $W$, so it can be added to the original weight matrix.

For a batch of inputs:

$$
X\in\mathbb{R}^{n\times d_{\text{in}}}
$$

the adapter contribution is:

$$
X A^\top B^\top
\in\mathbb{R}^{n\times d_{\text{out}}}
$$

This is dimensionally identical to the ordinary linear output:

$$
XW^\top\in\mathbb{R}^{n\times d_{\text{out}}}
$$

In PyTorch, the adapter can therefore be implemented using two smaller linear transformations without explicitly constructing the full matrix $BA$ on every forward pass.

A dimension error in either matrix will cause the matrix multiplication to fail.

### Q25. What does the rank $r$ control?

The rank $r$ is the inner dimension of the adapter factorization.

For a layer with input dimension $d_{\text{in}}$ and output dimension $d_{\text{out}}$:

$$
N_{\text{LoRA}}=r(d_{\text{in}}+d_{\text{out}})
$$

Increasing $r$:

- Increases the number of trainable parameters linearly.
- Increases adapter computation and memory requirements.
- Allows a larger family of possible weight updates.
- May improve target-task accuracy, although improvement is not guaranteed.

Reducing $r$:

- Reduces the adapter's parameter count.
- Reduces the capacity of the learned update.
- May be sufficient when the target task requires only a small adjustment to the pre-trained representation.

The rank cannot exceed the smaller matrix dimension if the goal is a full-rank factorization, but the practical advantage of LoRA generally comes from using a much smaller rank.

The notebook invites experimentation with different ranks to find a useful trade-off between parameter efficiency and validation accuracy.

### Q26. Why should LoRA initially leave the pre-trained model's output unchanged?

The notebook recommends initializing $A$ with normally distributed values and $B$ with zeros.

For example:

python

```
nn.init.normal_(self.A, mean=0.0, std=0.02)
nn.init.zeros_(self.B)
```

The precise standard deviation is a design choice; the important property is that $B$ starts at zero.

Initially:

$$
B=0
$$

Therefore:

$$
\Delta W=BA=0
$$

The adapted layer initially computes:

$$
y=xW^\top+b
$$

which is exactly the output of the original linear layer.

This initialization is useful because the model begins with its pre-trained behavior intact rather than introducing an arbitrary perturbation before learning starts.

The adapter can then learn a residual update through backpropagation.

**Why not initialize both matrices to zero?** If both factors are zero, their gradients at initialization can fail to provide a useful learning signal to either factor. Initializing $A$ randomly and $B$ to zero allows $B$ to begin learning from the existing activations while preserving an initially zero update.

### Q27. How do the gradients flow through a LoRA layer?

Let the adapted weight be:

$$
W'=W+BA
$$

The original weight $W$ is frozen, while $A$ and $B$ are trainable.

During backpropagation, the gradient of the loss is propagated through the adapter operations to $A$ and $B$. The original weight is still used to compute the forward output, but its gradient is not calculated when `requires_grad=False`.

For a row-oriented input batch $X$, ignoring the bias:

$$
Y=XW^\top+XA^\top B^\top
$$

Let:

$$
G=\frac{\partial L}{\partial Y}
$$

Then the adapter gradients are:

$$
\frac{\partial L}{\partial B}
=G^\top XA^\top
$$

and:

$$
\frac{\partial L}{\partial A}
=B^\top G^\top X
$$

The dimensions are consistent with:

$$
B\in\mathbb{R}^{d_{\text{out}}\times r},
\qquad
A\in\mathbb{R}^{r\times d_{\text{in}}}
$$

At initialization, $B=0$, so the initial gradient of $A$ is zero. However, the gradient of $B$ can be nonzero because $A$ is randomly initialized. After $B$ begins changing, gradients can also flow into $A$.

This is one reason the standard initialization uses a random $A$ and zero-initialized $B$.

### Q28. Why is LoRA parameter-efficient but not necessarily faster?

LoRA reduces the number of trainable parameters in adapted layers, but its forward pass includes additional matrix operations.

For example, a dense linear layer computes:

$$
Y=XW^\top+b
$$

A LoRA layer computes:

$$
Y=XW^\top+XA^\top B^\top+b
$$

The additional adapter operations require computation. The implementation also needs to store the activations and intermediate values required to calculate gradients for $A$ and $B$.

Consequently, the practical speed depends on:

- Layer dimensions and rank.
- Batch size.
- Device and hardware utilization.
- Kernel-launch overhead.
- Whether the dense baseline's optimizer and gradient computations dominate runtime.
- Whether the adapter is merged into the original weights for inference.

LoRA often reduces optimizer-state memory because the optimizer maintains state for fewer trainable parameters. However, it does not automatically reduce all memory requirements by the same proportion, and the original frozen weights still occupy storage.

The notebook explicitly asks students to measure forward and backward execution times rather than assume that LoRA must be faster.

### Q29. What is the purpose of the notebook's `AbstractLayer`?

`AbstractLayer` provides an intermediate implementation between an ordinary linear layer and LoRA.

It wraps an existing `nn.Linear` layer, freezes its original weight, and creates a trainable update matrix with the same shape.

The key initialization is:

python

```
self.original_layer = original_layer

self.W = self.original_layer.weight
self.b = self.original_layer.bias

self.W.requires_grad = False

self.dW = nn.Parameter(
    torch.zeros_like(self.W),
    requires_grad=True
)
```

The forward pass is:

python

```
def forward(self, x):
    z_frozen = torch.matmul(x, self.W.T)
    z_update = torch.matmul(x, self.dW.T)

    z = z_frozen + z_update

    return z + self.b
```

This implements:

$$
y=xW^\top+x\Delta W^\top+b
$$

where $\Delta W$ is represented by `dW`.

Because `dW` is initialized to zero, the initial output should match the original linear layer, provided the same weight and bias are used.

The notebook tests this behavior using `torch.testing.assert_close`.

The key limitation is that `dW` is a full-size matrix. It has as many parameters as $W$, so this abstraction does not provide the parameter-count savings of LoRA.

Its purpose is conceptual: it demonstrates how the original weight can remain frozen while a separate additive update is learned.

### Q30. How should the `LoraLayer` implementation be designed before proceeding to training?

The notebook's implementation cell is intentionally incomplete. The following is a correct implementation consistent with the notebook's layer dimensions and initialization advice.

python

```
class LoraLayer(nn.Module):
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int,
        verbose: bool = False
    ):
        super().__init__()

        if r <= 0:
            raise ValueError("r must be positive")

        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.r = r

        # Freeze the original weight and bias.
        self.original_layer.requires_grad_(False)

        # A: (r, in_features)
        self.A = nn.Parameter(
            torch.empty(r, self.in_features)
        )

        # B: (out_features, r)
        self.B = nn.Parameter(
            torch.zeros(self.out_features, r)
        )

        # Random A and zero B give an initially zero update.
        nn.init.normal_(self.A, mean=0.0, std=0.02)

        if verbose:
            print(
                f"LoRA: {self.in_features} -> "
                f"{self.out_features}, rank={r}"
            )

    def forward(self, x):
        # Frozen original linear transformation.
        original_output = self.original_layer(x)

        # Low-rank update: x @ A.T @ B.T.
        adapter_output = (x @ self.A.T) @ self.B.T

        return original_output + adapter_output
```

This implementation deliberately does not include the optional $1/r$ scaling factor. That follows the notebook's observation that it can be omitted for this practical, although scaling is an important design choice in LoRA implementations.

If scaling is used, a common formulation is:

$$
\Delta W=\frac{\alpha}{r}BA
$$

where $\alpha$ is a scaling hyperparameter. In that case, the forward pass must multiply the adapter output by $\alpha/r$.

**Why this implementation works:**

1. The original linear layer retains its pre-trained weights and bias.
2. The original layer's parameters are frozen.
3. $A$ has shape $r\times d_{\text{in}}$.
4. $B$ has shape $d_{\text{out}}\times r$.
5. $B$ is initialized to zero, so the initial adapter output is zero.
6. Both $A$ and $B$ are registered as trainable parameters.
7. The forward pass adds the adapter output to the original linear output, including the original bias exactly once.

The original layer is stored as a child module, so its parameters remain part of the model's state dictionary even though they are frozen.

#### Replacing the fully connected layers

The notebook uses the following replacement pattern:

python

```
# Example rank; experiment with alternatives later.
r = 8

lora_model.fc1 = LoraLayer(lora_model.fc1, r)
lora_model.fc2 = LoraLayer(lora_model.fc2, r)
lora_model.fc3 = LoraLayer(lora_model.fc3, r)
lora_model.fc4 = LoraLayer(lora_model.fc4, r)
```

The rank $r=8$ is an illustrative starting choice, not a notebook-measured optimum.

The convolutional layers must remain frozen, as in the baseline. The target classifier remains trainable because it must learn to predict the five target classes.

#### Verify the initialization

Before training, verify that the LoRA model produces the same outputs as the corresponding model with the original linear layers.

For example, for a wrapped layer:

python

```
reference_layer = nn.Linear(512, 256).to(device)
lora_layer = LoraLayer(reference_layer, r=8).to(device)

x = torch.randn(4, 512, device=device)

torch.testing.assert_close(
    lora_layer(x),
    reference_layer(x)
)
```

This test checks the initial forward-pass equivalence. It does not test training or target-domain accuracy.

#### Count trainable parameters

After replacing the layers, use:

python

```
total_params, trainable_params = (
    num_parameters(lora_model)
)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)
print(
    "Trainable fraction:",
    trainable_params / total_params
)
```

For rank $r$, the four adapters introduce:

$$
N_{\text{adapter}}
=r[(256+1024)+(1024+512)+(512+512)+(512+256)]
$$

Therefore:

$$
\boxed{N_{\text{adapter}}=4608r}
$$

At $r=8$:

$$
N_{\text{adapter}}=36,864
$$

The five-class classifier contains:

$$
256\times5+5=1,285
$$

trainable parameters.

The expected total number of trainable parameters after adapting the four fully connected layers is therefore:

$$
\boxed{36,864+1,285=38,149}
$$

This calculation assumes that only the four adapter pairs and the target classifier are trainable, with both convolutional layers and all original weights and biases in the adapted fully connected layers frozen.

The total model parameter count will still include the frozen weights. The precise count should be confirmed with `num_parameters()` after executing the implementation.

#### Important notebook issue to fix

The provided starter cell contains:

python

```
self.verbose = verbose
```

but `verbose` is not defined in the supplied `LoraLayer.__init__` signature. Unless a global variable with that name happens to exist, this raises a `NameError`.

The implementation above fixes the problem by adding `verbose: bool = False` to the constructor.

The starter code also refers to `r` without defining its value in the supplied excerpt. Assign a positive integer before replacing the layers.

Finally, the model should be checked after replacement to ensure that only the intended parameters are trainable:

python

```
for name, param in lora_model.named_parameters():
    if param.requires_grad:
        print(name, tuple(param.shape))
```

The expected trainable parameters are the adapter matrices $A$ and $B$ for each of the four adapted layers, together with the target classifier's weight and bias.

**Transition to Part 2:** With the LoRA layer implemented and its initial behavior verified, the next stage is to train it on the target dataset, compare it with dense fine-tuning, measure state-dictionary and optimizer sizes, benchmark forward and backward execution, investigate alternative ranks, and complete the notebook's reporting and optional extension exercises.

---

Here is Chunk 2 only, continuing from Q31 through Q60. It follows the remaining sections of your actual notebook, including LoRA training, evaluation, parameter counts, benchmarking, debugging, reporting, and the optional extension exercises.

The document is standalone and ready to copy into `practical-4-lora-part-2.md`. As requested, I will stop at Q60.

Writing

# Advanced Machine Learning — Practical 4: Low-Rank Adaptation (LoRA)

## Part 2: Questions 31–60

**Suggested filename:** `practical-4-lora-part-2.md`

---

## Section 6 — Implementing and Training LoRA

### Q31. How should the pre-trained model be initialized before replacing its fully connected layers with LoRA adapters?

The notebook initializes the target model with five output classes and loads the pre-trained weights, excluding the original classifier.

python

```
torch.manual_seed(0)

lora_model = ConvMLP(
    num_classes=target_data.num_classes
).to(device)

lora_model.load_state_dict(
    pt_state_dict,
    strict=False
)
```

Here, `pt_state_dict` is the checkpoint dictionary from which the original classifier's weight and bias were removed.

The convolutional and fully connected layers therefore inherit the learned CIFAR-10 representations, while the new five-class classifier remains randomly initialized.

Next, the convolutional layers are frozen:

python

```
lora_model.conv1.requires_grad_(False)
lora_model.conv2.requires_grad_(False)
```

Finally, the fully connected layers are replaced with LoRA wrappers.

The order matters: LoRA adapters should wrap the pre-trained linear layers, not newly initialized linear layers. Otherwise, the adapters would not preserve the learned source-domain weights.

A crucial detail is that freezing the convolutional layers before wrapping the fully connected layers does not freeze the new adapter matrices. The adapters are separate trainable parameters.

### Q32. How do you replace all four fully connected layers with LoRA layers?

The notebook adapts `fc1`, `fc2`, `fc3`, and `fc4`:

python

```
r = 8

lora_model.fc1 = LoraLayer(lora_model.fc1, r)
lora_model.fc2 = LoraLayer(lora_model.fc2, r)
lora_model.fc3 = LoraLayer(lora_model.fc3, r)
lora_model.fc4 = LoraLayer(lora_model.fc4, r)
```

The rank is shared across all four layers in this example.

Their dimensions are:

| Layer | Input dimension | Output dimension | LoRA parameters at $r=8$ |
| --- | --- | --- | --- |
| `fc1` | 256 | 1024 | 10,240 |
| `fc2` | 1024 | 512 | 12,288 |
| `fc3` | 512 | 512 | 8,192 |
| `fc4` | 512 | 256 | 6,144 |
| **Total** | — | — | **36,864** |

Each adapter count is calculated using:

$$
N_{\text{LoRA}}=r(d_{\text{in}}+d_{\text{out}})
$$

The original linear layers remain inside their respective wrappers, so their frozen weights continue to contribute to the forward pass.

The classifier is not wrapped by LoRA in the notebook. Its weight and bias remain trainable because the output classes differ between CIFAR-10 and the selected CIFAR-100 task.

### Q33. How can you verify that the original weight matrices are frozen?

The notebook includes an assertion:

python

```
assert (
    lora_model.fc1.original_layer.weight.requires_grad
    == False
), "Make sure to freeze the original W matrices in each adapted layer"
```

This checks the first adapted layer.

A more complete check covers all four layers and their biases:

python

```
for layer_name in ["fc1", "fc2", "fc3", "fc4"]:
    layer = getattr(lora_model, layer_name)

    assert not layer.original_layer.weight.requires_grad
    assert not layer.original_layer.bias.requires_grad
    assert layer.A.requires_grad
    assert layer.B.requires_grad

print("LoRA freezing checks passed.")
```

This assumes the `LoraLayer` implementation uses the `A` and `B` attribute names from Part 1.

You should also inspect all trainable parameters:

python

```
for name, param in lora_model.named_parameters():
    if param.requires_grad:
        print(name, tuple(param.shape))
```

The expected trainable parameters are:

- The `A` and `B` matrices in `fc1`.
- The `A` and `B` matrices in `fc2`.
- The `A` and `B` matrices in `fc3`.
- The `A` and `B` matrices in `fc4`.
- The five-class classifier weight and bias.

The convolutional parameters and original fully connected weights and biases should be frozen.

### Q34. How many parameters should the LoRA model train?

For the four adapted layers, the total adapter parameter count is:

$$
N_{\text{adapter}}
=r[(256+1024)+(1024+512)+(512+512)+(512+256)]
$$

Simplifying:

$$
\boxed{N_{\text{adapter}}=4608r}
$$

For $r=8$:

$$
N_{\text{adapter}}=4608\times8=36,864
$$

The target classifier has 1,285 parameters:

$$
N_{\text{classifier}}=256\times5+5=1,285
$$

Therefore, the total expected number of trainable parameters is:

$$
\boxed{N_{\text{trainable}}=36,864+1,285=38,149}
$$

This assumes that the original weights and biases of the four adapted layers are frozen and that the convolutional layers are also frozen.

For comparison, the dense baseline trains all four original fully connected layers and the target classifier:

$$
\begin{aligned}
N_{\text{dense trainable}}
={}&263,168+524,800\\
&+262,656+131,328\\
&+1,285
\end{aligned}
$$

Thus:

$$
\boxed{N_{\text{dense trainable}}=1,183,237}
$$

At rank $r=8$, the number of trainable parameters is reduced by:

$$
1-\frac{38,149}{1,183,237}
\approx 96.78\%
$$

This is a theoretical count derived from the architecture, not a measured training result.

### Q35. Why does the total parameter count not decrease as much as the trainable parameter count?

LoRA freezes the original weights rather than removing them.

For example, `fc2` still contains its original $512\times1024$ weight matrix and 512-element bias. The LoRA wrapper also stores the two adapter matrices.

The total parameter count can therefore be larger than the original model's count because the adapters add parameters:

$$
N_{\text{total LoRA}}
=
N_{\text{original target model}}
+
N_{\text{adapter}}
$$

For the five-class model at $r=8$:

$$
N_{\text{total LoRA}}
=1,332,741+36,864
$$

$$
\boxed{N_{\text{total LoRA}}=1,369,605}
$$

This assumes the wrapper retains each original linear layer exactly once.

The important distinction is:

- **Total parameters:** all parameter elements stored by the model, including frozen parameters.
- **Trainable parameters:** parameter elements for which gradients and optimizer updates are enabled.

LoRA primarily reduces the latter.

Storage savings become more pronounced when saving only the adapter matrices instead of the entire model state dictionary.

---

## Section 7 — Training and Evaluating the LoRA Model

### Q36. How should the LoRA model be trained against the dense fine-tuning baseline?

The notebook trains LoRA using the same target data loaders and hyperparameters as the dense baseline:

python

```
lora_metrics = train_model(
    lora_model,
    train_loader_target,
    val_loader_target,
    num_ft_epochs,
    lr,
    l2_reg,
    title=f'LoRA Fine-tuning ({target_data.name})',
    baseline_metrics=baseline_ft_metrics
)
```

The shared configuration is:

| Hyperparameter | Value |
| --- | --- |
| Epochs | 30 |
| Learning rate | $10^{-4}$ |
| AdamW weight decay | $10^{-2}$ |
| Batch size | 256 |
| Target classes | 5 |

Using the same configuration makes the comparison easier to interpret.

However, the same learning rate does not guarantee that dense fine-tuning and LoRA have equally suitable optimization settings. The adapter parameterization changes the optimization problem, and LoRA may benefit from tuning its learning rate or rank.

The training loop returns a `TrainingMetrics` object containing training and validation loss and accuracy.

The notebook plots the LoRA curves against the baseline curves so that differences in convergence and final performance can be examined.

### Q37. How should the best validation accuracy of LoRA be compared with the baseline?

The notebook calculates:

python

```
print(
    f'Best LoRA-FT accuracy: '
    f'{lora_metrics.best_val_acc:.1%}'
)

diff = (
    lora_metrics.best_val_acc
    - baseline_ft_metrics.best_val_acc
)

if diff <= 0:
    print(
        f'  === {-diff:.1%} points below baseline ==='
    )
else:
    print(
        f'  +++ {diff:.1%} points ABOVE baseline!!! +++'
    )
```

The difference is:

$$
\Delta a=a_{\text{LoRA}}-a_{\text{dense}}
$$

where both accuracies are the best validation accuracies recorded during training.

Interpretation:

- $\Delta a>0$: LoRA achieved a higher best validation accuracy.
- $\Delta a<0$: LoRA achieved a lower best validation accuracy.
- $\Delta a=0$: both methods achieved the same best validation accuracy.

For example, if the baseline reaches 90% and LoRA reaches 88%, then:

$$
\Delta a=0.88-0.90=-0.02
$$

The difference is negative two percentage points.

**Important terminology:** A difference between two accuracies is usually reported in _percentage points_. A two-percentage-point difference is not the same as a 2% relative change.

No actual validation accuracy can be reported until the notebook has been executed.

### Q38. How should the training and validation curves be interpreted?

The notebook plots loss and accuracy across training steps and validation epochs.

Common patterns include:

| Pattern | Possible interpretation |
| --- | --- |
| Training loss decreases and validation loss decreases | Learning useful features |
| Training loss decreases but validation loss increases | Possible overfitting |
| Both losses remain high | Underfitting, optimization difficulty, or a bug |
| LoRA converges more slowly than dense fine-tuning | Adapter capacity or optimization may be limiting learning |
| LoRA reaches similar validation accuracy with fewer trainable parameters | Evidence of parameter-efficient adaptation |
| Training accuracy improves but validation accuracy stagnates | Possible overfitting or limited generalization |

The best validation accuracy is useful, but it should not be the only reported metric.

A good report should also consider:

- Final validation accuracy.
- Validation loss.
- The shape of the learning curves.
- Training duration.
- Parameter counts.
- Variation across repeated runs, if feasible.

One run is not enough to establish that LoRA is universally better or worse than dense fine-tuning.

### Q39. How can you inspect the target model's predictions?

The notebook selects a validation batch and passes the predictions to `inspect_batch()`:

python

```
target_x, target_y = next(
    iter(val_loader_target)
)

inspect_batch(
    target_x,
    target_y,
    predictions=lora_model(target_x),
    class_names=target_data.classes,
    title=f'LoRA-FT model on {target_data.name}'
)
```

The helper displays images, ground-truth labels, predicted labels, and correctness indicators.

The output logits are converted to predicted classes using `argmax`.

Visual inspection can reveal issues that a single accuracy score may hide, such as:

- Confusion between visually similar classes.
- Incorrect label remapping.
- Predictions concentrated on one class.
- Poor performance on particular images.
- Differences between the dense and LoRA models.

A correct-looking batch does not establish high overall accuracy. The validation metric must be calculated over the complete validation set.

**Implementation detail:** `inspect_batch()` assumes that labels and predictions are comparable class indices when it marks a prediction correct or incorrect. For this notebook, that is appropriate because the target labels have been remapped to the five-class range.

---

## Section 8 — Rank Selection and Optimization

### Q40. How should different LoRA ranks be tested?

The notebook explicitly recommends trying different values of $r$.

A simple experiment can test:

python

```
r_values = [1, 2, 4, 8, 16, 32]
```

For each rank:

1. Initialize a fresh target model.
2. Load the same pre-trained checkpoint.
3. Remove the original classifier weights from the checkpoint.
4. Freeze the convolutional layers.
5. Wrap the four fully connected layers with LoRA.
6. Verify that the original weights are frozen.
7. Train using the same data split and training configuration.
8. Record the best and final validation accuracy.
9. Record trainable parameter count and training duration.

Do not reuse a previously trained model when testing a new rank. Otherwise, the experiment may begin from already adapted parameters and cease to be a fair comparison.

A useful result table is:

| Rank $r$ | Trainable parameters | Best validation accuracy | Training time |
| --- | --- | --- | --- |
| 1 | 5,893 | Not measured | Not measured |
| 2 | 10,501 | Not measured | Not measured |
| 4 | 19,717 | Not measured | Not measured |
| 8 | 38,149 | Not measured | Not measured |
| 16 | 75,013 | Not measured | Not measured |
| 32 | 148,741 | Not measured | Not measured |

The parameter counts are calculated from the notebook's architecture and the assumption that only the four adapters and the five-class classifier are trainable.

The accuracy and training-time entries must be filled in from actual experiments.

### Q41. Can a larger LoRA rank improve validation accuracy?

Yes, it can, but there is no guarantee.

Increasing $r$ increases the dimensional capacity of the adapter update:

$$
N_{\text{adapter}}=r(d_{\text{in}}+d_{\text{out}})
$$

A larger rank allows a broader set of possible changes to the original weights. This can help when a small-rank update cannot adequately adapt the source representation to the target task.

However, larger ranks also introduce more trainable parameters and can increase the risk of overfitting on a small target dataset.

The optimal rank depends on:

- The relationship between source and target domains.
- The number of target examples.
- The architecture and selected layers.
- The learning rate and weight decay.
- The number of epochs.
- Initialization and random seed.

The objective is not simply to maximize rank. It is to identify a rank that achieves a useful balance between target performance and computational efficiency.

### Q42. Why might LoRA perform worse than dense fine-tuning?

Several explanations are possible.

**Limited adapter capacity:** A small rank may not represent the update needed for the target task.

**Optimization differences:** Dense fine-tuning and LoRA have different parameterizations. The same learning rate may not be optimal for both.

**Incorrect initialization:** If the adapter update is not initialized to zero, the network's initial behavior changes unexpectedly.

**Incorrect freezing:** If the original weights are not frozen, the experiment is no longer the intended LoRA comparison.

**Label errors:** If the CIFAR-100 labels are not remapped consistently, the model may be trained against incorrect target labels.

**Implementation bugs:** Incorrect matrix dimensions, duplicate biases, or incorrect transposes can make the adapter computation wrong.

**Overfitting:** A larger rank may fit the training set without improving validation performance.

A sensible debugging sequence is to verify the initial forward pass, inspect trainable parameter names, check that the adapter gradients are nonzero after optimization begins, and then inspect the training curves.

### Q43. What learning-rate and weight-decay considerations apply to LoRA?

The notebook uses:

python

```
lr = 1e-4
l2_reg = 1e-2
```

and applies AdamW to the model's parameters.

These settings provide a common starting point for comparing LoRA and dense fine-tuning, but they are not necessarily optimal for the adapter matrices.

A systematic experiment might compare a small grid of learning rates and ranks while holding the data split fixed.

For example:

- Rank: $r\in\{2,4,8,16\}$.
- Learning rate: $10^{-4}$, $3\times10^{-4}$, or $10^{-3}$.

These are candidate settings, not guaranteed optimal values.

Weight decay also requires care. In the notebook's original training loop, `AdamW` receives `model.parameters()`, including frozen parameters. Frozen parameters normally have no gradients and therefore are not updated. However, if the model contains unexpected trainable parameters, they may be affected by the optimizer.

A clearer optimizer construction is:

python

```
opt = torch.optim.AdamW(
    (
        p for p in model.parameters()
        if p.requires_grad
    ),
    lr=lr,
    weight_decay=l2_reg
)
```

This makes the intended set of optimized parameters explicit.

---

## Section 9 — Model Size, Optimizer State, and Runtime

### Q44. How does the notebook measure the model's state-dictionary size?

The notebook defines:

python

```
def get_state_dict_size(state_dict: dict):
    param_count = 0

    for key, values in state_dict.items():
        param_count += values.numel()

    return param_count
```

This function counts tensor elements in the state dictionary. It does not directly measure bytes on disk.

For example:

python

```
dense_state_count = get_state_dict_size(
    baseline_ft_model.state_dict()
)

lora_state_count = get_state_dict_size(
    lora_model.state_dict()
)
```

The state dictionary of a model usually contains its parameters and registered buffers, including frozen parameters.

Because LoRA retains the original weights and adds adapter matrices, its complete state dictionary may contain more tensor elements than the dense model's state dictionary.

To measure actual checkpoint size, serialize each state dictionary and inspect the resulting file size. Serialization overhead and tensor data types can affect the result.

A robust file-size comparison should save both checkpoints separately and compare their byte sizes rather than infer file size from trainable parameter counts.

### Q45. Why is the LoRA optimizer state often smaller than the dense optimizer state?

Optimizers such as AdamW maintain additional state for trainable parameters.

For each parameter that receives gradients, Adam-style optimizers typically maintain:

- A first-moment estimate.
- A second-moment estimate.
- A step counter or related metadata.

The exact representation depends on the optimizer and PyTorch version.

The dense baseline updates all four fully connected layers and the classifier. The LoRA model updates only the adapter matrices and the target classifier.

Therefore, the LoRA optimizer usually needs much less tensor-valued state.

The notebook defines:

python

```
def get_opt_state_dict_size(state_dict: dict):
    grad_count = 0

    for i, param in state_dict['state'].items():
        for key, values in param.items():
            grad_count += values.numel()

    return grad_count
```

Despite the function's name and local variable names, this counts tensor elements in the optimizer's stored state, not gradients themselves.

For a meaningful comparison, inspect optimizer state only after at least one optimizer step. Before training, AdamW may not yet have initialized its moment tensors.

### Q46. How can the state-dictionary and optimizer-state comparisons be implemented correctly?

Use separate optimizers for the dense and LoRA models and run at least one training step before inspecting optimizer state.

For example:

python

```
dense_optimizer = torch.optim.AdamW(
    (
        p for p in baseline_ft_model.parameters()
        if p.requires_grad
    ),
    lr=1e-4,
    weight_decay=1e-2
)

lora_optimizer = torch.optim.AdamW(
    (
        p for p in lora_model.parameters()
        if p.requires_grad
    ),
    lr=1e-4,
    weight_decay=1e-2
)
```

After training each model, measure the number of tensor elements in the model state dictionaries:

python

```
dense_model_elements = get_state_dict_size(
    baseline_ft_model.state_dict()
)

lora_model_elements = get_state_dict_size(
    lora_model.state_dict()
)
```

Then inspect optimizer state:

python

```
dense_optimizer_elements = get_opt_state_dict_size(
    dense_optimizer.state_dict()
)

lora_optimizer_elements = get_opt_state_dict_size(
    lora_optimizer.state_dict()
)
```

These optimizer counts are meaningful only if the optimizers have actually initialized their state.

For exact checkpoint size, use serialized files:

python

```
torch.save(
    baseline_ft_model.state_dict(),
    "dense_model_state.pt"
)

torch.save(
    lora_model.state_dict(),
    "lora_model_state.pt"
)
```

Then measure the file sizes using the operating system or Python's `os.path.getsize()`.

**Important:** The notebook's helper counts tensor elements, not bytes, and its optimizer helper excludes non-tensor metadata. Report these measurements accurately.

### Q47. How can forward-pass and backward-pass times be compared fairly?

The notebook asks students to compare the forward and backward computation times of dense fine-tuning and LoRA.

A fair benchmark should:

1. Use the same input batch and batch size.
2. Use the same device.
3. Use the same data type.
4. Warm up the models before timing.
5. Measure forward and backward passes separately.
6. Synchronize CUDA when timing GPU execution.
7. Repeat measurements and report an average or median.
8. Reset gradients consistently between iterations.

CUDA operations are asynchronous, so ordinary wall-clock timing without synchronization can underestimate the actual execution time.

A simple CUDA timing helper is:

python

```
import time
import numpy as np

def benchmark_forward_backward(
    model,
    x,
    y,
    repeats=50
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Warm-up
    for _ in range(10):
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()

    forward_times = []
    backward_times = []

    for _ in range(repeats):
        model.zero_grad(set_to_none=True)

        if x.is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        output = model(x)

        if x.is_cuda:
            torch.cuda.synchronize()

        forward_times.append(
            time.perf_counter() - start
        )

        loss = criterion(output, y)

        if x.is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        loss.backward()

        if x.is_cuda:
            torch.cuda.synchronize()

        backward_times.append(
            time.perf_counter() - start
        )

    return {
        "forward_ms": 1000 * np.median(forward_times),
        "backward_ms": 1000 * np.median(backward_times)
    }
```

Run the helper separately for the dense and LoRA models using the same batch.

This benchmark measures forward execution and backward execution, not the optimizer update or complete training step.

For CPU execution, CUDA synchronization is unnecessary. For very short operations, Python timing overhead may be significant, so use enough repetitions and report the measurement method.

### Q48. Why might the LoRA forward pass be slower even though it trains fewer parameters?

The LoRA forward pass adds operations to the original linear transformation.

A dense layer computes:

$$
Y=XW^\top+b
$$

The LoRA layer computes:

$$
Y=XW^\top+XA^\top B^\top+b
$$

The extra matrix multiplications may increase forward-pass time.

Backward execution is more nuanced. LoRA avoids computing gradients for the frozen original weights, which can reduce backward computation and gradient storage. However, it must still compute gradients for the adapter factors and propagate gradients through the adapter operations.

The measured result depends on:

- The size of the layer.
- The batch size.
- The rank.
- Hardware and software implementation.
- Kernel-launch overhead.
- Whether the computation is memory-bound or compute-bound.

For a relatively small network, the extra operations and kernel launches can outweigh the savings from freezing parameters.

For a large model with very large weight matrices, reducing trainable parameters can offer more substantial savings.

After training, merging the adapter into the original weights can remove the extra adapter operations from inference.

---

## Section 10 — Memory, Debugging, and Reproducibility

### Q49. How should GPU memory usage be measured in the notebook?

The notebook defines:

python

```
def check_memory():
    alloc, alloc_suffix = format_bytes(
        torch.cuda.memory_allocated(0)
    )

    reserved, reserved_suffix = format_bytes(
        torch.cuda.memory_reserved(0)
    )

    maxreserved, maxreserved_suffix = format_bytes(
        torch.cuda.max_memory_reserved(0)
    )

    print(f"Memory allocated: {alloc:.1f}{alloc_suffix}")
    print(f"Memory reserved: {reserved:.1f}{reserved_suffix}")
    print(
        f"Max memory reserved: "
        f"{maxreserved:.1f}{maxreserved_suffix}"
    )
```

The three measurements mean different things:

- **Allocated memory:** GPU memory currently occupied by tensors managed by PyTorch.
- **Reserved memory:** GPU memory held by PyTorch's caching allocator.
- **Maximum reserved memory:** the peak amount reserved since the peak-memory statistics were last reset.

The notebook correctly warns that these measurements reflect the entire session, not just one model.

For a fair baseline-versus-LoRA comparison:

1. Start from a clean runtime or carefully remove unrelated objects.
2. Load only the model and tensors needed for the experiment.
3. Reset peak-memory statistics.
4. Run a representative forward and backward pass.
5. Record peak allocated and reserved memory.
6. Repeat under the same batch size and device conditions.

For example:

python

```
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Run a representative training step here.

torch.cuda.synchronize()

print(
    "Peak allocated:",
    torch.cuda.max_memory_allocated()
)

print(
    "Peak reserved:",
    torch.cuda.max_memory_reserved()
)
```

`empty_cache()` releases unused cached blocks to the CUDA allocator, but it does not free memory held by live tensors.

### Q50. What common implementation errors should be checked when LoRA training fails?

The most important checks are:

**1\. Missing rank definition**

The starter notebook references `r` without assigning a value in the supplied implementation section.

Fix:

python

```
r = 8
```

**2\. Undefined****`verbose`****variable**

The starter constructor contains `self.verbose = verbose`, but `verbose` is absent from its argument list.

Add a constructor argument or remove the reference.

**3\. Incorrect adapter dimensions**

Use:

python

```
A.shape == (r, in_features)
B.shape == (out_features, r)
```

**4\. Original weights not frozen**

Check that the original linear layer has `requires_grad=False`.

**5\. Incorrect initialization**

Zero-initialize $B$ if the initial adapter update must be zero.

**6\. Duplicate bias**

The original linear layer already adds its bias. Do not add the bias a second time in the adapter path.

**7\. Incorrect target labels**

Ensure that all target labels are integers in the range 0–4 and that the class names match the mapping.

**8\. Incompatible device placement**

The input batch, original weights, and adapter parameters must be on compatible devices.

**9\. Incorrect classifier initialization**

The target classifier must output five logits rather than ten.

**10\. Accidental optimizer reuse**

Construct the optimizer for the correct model after its trainable parameters have been established. Replacing modules after constructing the optimizer can leave the optimizer referencing the old parameters.

### Q51. What bugs or inconsistencies in the original notebook should be flagged?

Several details deserve attention before reporting results.

**1\.****`random_split`****uses fractional lengths.**

The notebook passes `[1-val_frac, val_frac]` to `random_split`. Many PyTorch versions require integer lengths. Use explicitly calculated integer lengths, as shown in Q9.

**2\. The LoRA starter code references an undefined****`verbose`****variable.**

The constructor must accept `verbose` or stop using it.

**3\. The rank variable is not assigned in the provided LoRA implementation cell.**

Set a positive integer before constructing the adapters.

**4\. The parameter-reporting comment is misleading.**

The notebook comments that approximately 89% of parameters should be trainable after freezing the convolutional layers. The actual architecture gives approximately 88.9% of the five-class model's parameters as trainable at that stage, so the comment is a rounded estimate rather than an exact guarantee.

**5\. Validation metrics are averaged by batch rather than example count.**

The last, smaller validation batch receives the same weight as a full batch. Weighted aggregation is more accurate.

**6\. Validation loaders shuffle examples.**

Shuffling does not invalidate accuracy, but it makes batch-level visual comparisons less reproducible. For evaluation, `shuffle=False` is generally preferable.

**7\.****`format_bytes()`****uses a strict greater-than comparison.**

The loop uses `while size > power`. Exact powers of 1024 are not converted to the next unit, which can produce awkward formatting at boundaries.

**8\. The plotting code may calculate zero tick spacing.**

In `TrainingMetrics.plot()`, `num_expected_epochs // 10` can evaluate to zero when there are fewer than ten expected epochs. Passing zero as a tick stride causes an error. A robust implementation should clamp the stride to at least one.

**9\. The model checkpoint is loaded without an explicit****`map_location`****.**

This can cause device compatibility issues when moving between CUDA and CPU environments.

**10\. The notebook assumes CUDA availability.**

`device = 'cuda'` and the GPU memory functions require a CUDA-enabled environment. CPU fallback requires changing the device and guarding CUDA-specific operations.

These issues should be documented and corrected where necessary rather than silently ignored.

### Q52. How can reproducibility be improved when comparing dense fine-tuning with LoRA?

A reproducible experiment should control the sources of randomness and document the environment.

Recommended practices include:

- Fix the random seeds used for model initialization and dataset splitting.
- Use the same training and validation indices for both methods.
- Load the same pre-trained checkpoint.
- Use the same target classifier initialization where the comparison requires it.
- Keep batch size, epoch count, data preprocessing, and evaluation procedures consistent.
- Record the learning rate, weight decay, and LoRA rank.
- Record the PyTorch and CUDA versions, GPU model, and relevant hardware information.
- Save the best checkpoint based on validation performance.
- Repeat promising experiments with multiple seeds when possible.

For example:

python

```
import random
import numpy as np
import torch

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

For stricter reproducibility, deterministic algorithms can be enabled where supported, although this may reduce performance or cause some operations to raise errors.

Also distinguish **reproducibility** from **fairness**. Using the same seed does not automatically make two experiments equivalent if they use different data splits, initialization procedures, or optimization settings.

---

## Section 11 — Submission Report and Optional Extensions

### Q53. What should the final practical report contain?

The notebook's submission requirements ask for an explanation of the LoRA implementation, comparisons with dense fine-tuning, and discussion of experimental observations.

A useful report structure is:

1. **Objective:** Explain parameter-efficient fine-tuning and the purpose of the experiment.
2. **Dataset:** Describe CIFAR-10 pre-training and the selected CIFAR-100 superclass.
3. **Architecture:** Summarize `ConvMLP` and identify the frozen convolutional layers.
4. **Dense baseline:** Explain which parameters are updated and report the measured validation results.
5. **LoRA implementation:** Define the shapes and initialization of $A$ and $B$, the forward pass, and the trainable parameters.
6. **Training curves:** Compare dense and LoRA training and validation loss and accuracy.
7. **Parameter efficiency:** Compare total and trainable parameter counts.
8. **Checkpoint and optimizer sizes:** Report the actual measurements and explain the difference.
9. **Runtime and memory:** Compare forward, backward, and peak-memory measurements.
10. **Rank experiments:** Report the tested ranks and their outcomes.
11. **Debugging and limitations:** Describe any implementation issues and how they were resolved.
12. **Conclusion:** Summarize whether LoRA provided a useful trade-off between accuracy and computational efficiency.

Do not invent numerical results. Where experiments have not been run, label the values as pending or leave them out.

### Q54. How should results be presented in tables and plots?

A compact comparison table can summarize the main findings:

| Metric | Dense fine-tuning | LoRA |
| --- | --- | --- |
| Total model parameters | Measure | Measure |
| Trainable parameters | Measure | Measure |
| Best validation accuracy | Measure | Measure |
| Final validation accuracy | Measure | Measure |
| Training duration | Measure | Measure |
| Forward-pass time | Measure | Measure |
| Backward-pass time | Measure | Measure |
| Optimizer-state tensor count | Measure | Measure |
| Serialized model size | Measure | Measure |
| Peak GPU memory | Measure | Measure |

For rank experiments, use a separate table listing each rank, trainable parameter count, best validation accuracy, and runtime.

Recommended figures include:

- Training and validation loss.
- Training and validation accuracy.
- Validation accuracy against rank.
- Trainable parameter count against rank.
- Forward and backward time comparisons.

Use consistent axes and units. Report accuracy as either a fraction or percentage and clearly identify which representation is used.

The notebook's `TrainingMetrics.plot()` function already supports overlaying baseline curves, making it suitable for the main dense-versus-LoRA comparison.

### Q55. How can LoRA adapters be merged into the original weights?

After training, the learned adapter update can be combined with the original weight:

$$
W'=W+BA
$$

For the unscaled implementation in Part 1, the merged weight is:

python

```
with torch.no_grad():
    merged_weight = (
        lora_layer.original_layer.weight
        + lora_layer.B @ lora_layer.A
    )
```

If the implementation includes scaling $\alpha/r$, the merged weight must instead be:

$$
W'=W+\frac{\alpha}{r}BA
$$

To build a merged linear layer, create an ordinary `nn.Linear` layer with the same input and output dimensions, copy the merged weight and original bias, and use it in place of the wrapper.

python

```
def merge_lora_layer(lora_layer):
    original = lora_layer.original_layer

    merged = nn.Linear(
        original.in_features,
        original.out_features,
        bias=(original.bias is not None),
        device=original.weight.device,
        dtype=original.weight.dtype
    )

    with torch.no_grad():
        merged.weight.copy_(
            original.weight
            + lora_layer.B @ lora_layer.A
        )

        if original.bias is not None:
            merged.bias.copy_(original.bias)

    return merged
```

This function assumes the unscaled implementation from Part 1.

After replacing the LoRA wrappers with merged linear layers, compare the outputs:

python

```
lora_model.eval()
merged_model.eval()

with torch.no_grad():
    lora_output = lora_model(target_x)
    merged_output = merged_model(target_x)

torch.testing.assert_close(
    lora_output,
    merged_output,
    rtol=1e-4,
    atol=1e-5
)
```

Here, `merged_model` represents a copy of the model in which the four wrappers have been replaced by their merged linear layers.

Small floating-point differences may occur because the merged and unmerged implementations perform operations in a different order.

Merging removes the extra adapter operations at inference time. It does not, by itself, eliminate the need to store the resulting full weight matrices.

### Q56. How can only the LoRA adapter parameters be saved?

Instead of saving the complete model state dictionary, save the adapter matrices separately.

python

```
adapter_state = {}

for layer_name in ["fc1", "fc2", "fc3", "fc4"]:
    layer = getattr(lora_model, layer_name)

    adapter_state[f"{layer_name}.A"] = (
        layer.A.detach().cpu()
    )

    adapter_state[f"{layer_name}.B"] = (
        layer.B.detach().cpu()
    )

torch.save(
    adapter_state,
    "lora_adapters.pt"
)
```

The adapter checkpoint contains only the eight low-rank matrices.

To use it later, first initialize the same model architecture, load the pre-trained backbone, create the LoRA wrappers with the same rank, and load the saved adapter tensors.

For example:

python

```
adapter_state = torch.load(
    "lora_adapters.pt",
    map_location=device,
    weights_only=True
)

with torch.no_grad():
    for layer_name in ["fc1", "fc2", "fc3", "fc4"]:
        layer = getattr(lora_model, layer_name)

        layer.A.copy_(
            adapter_state[f"{layer_name}.A"]
        )

        layer.B.copy_(
            adapter_state[f"{layer_name}.B"]
        )
```

The example assumes the same architecture, rank, and layer names are used when loading.

**Important:** Saving only the adapters is insufficient to reconstruct a usable model unless the corresponding pre-trained backbone and target classifier are also available or separately saved.

For a practical deployment, document the base checkpoint identifier, rank, scaling convention, target class mapping, and any other configuration needed to reconstruct the adapted model.

### Q57. How does the LoRA rank affect adapter storage size?

For the four adapted layers, the total number of adapter parameters is:

$$
N_{\text{adapter}}=4608r
$$

If the adapter tensors use 32-bit floating-point values, each parameter occupies four bytes.

The approximate raw storage requirement is therefore:

$$
S_{\text{adapter}}=4\times4608r
$$

or:

$$
\boxed{S_{\text{adapter}}=18,432r
\text{ bytes}}
$$

For $r=8$:

$$
S_{\text{adapter}}=147,456
\text{ bytes}
$$

This is approximately 144 KiB of raw tensor data, excluding serialization metadata and any separately saved classifier parameters.

The target classifier adds 1,285 parameters, or another 5,140 bytes when stored as float32.

For comparison, the four dense weight matrices alone contain hundreds of thousands of parameters. Saving only the adapter tensors can therefore substantially reduce task-specific checkpoint size.

Actual serialized file sizes should still be measured because file formats, metadata, tensor types, and included model components affect the final result.

### Q58. How can the dense and LoRA models be compared in terms of GPU memory?

The notebook's optional extension asks students to compare GPU memory consumption.

The most useful measurements include:

- Peak allocated GPU memory during forward and backward execution.
- Peak reserved GPU memory.
- Memory used by the optimizer state.
- Memory required to hold the model parameters.
- Activation memory for a representative batch.

A robust procedure is:

1. Use a fresh runtime or isolate each model's measurement.
2. Load the model on the same GPU.
3. Use the same batch size and input shape.
4. Reset CUDA peak-memory statistics.
5. Run the forward pass and backward pass.
6. Synchronize CUDA.
7. Record the peak values.
8. Repeat for the other model under equivalent conditions.

Example:

python

```
def measure_peak_training_memory(model, x, y):
    model.train()
    model.zero_grad(set_to_none=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    loss_fn = nn.CrossEntropyLoss()

    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()

    torch.cuda.synchronize()

    return {
        "peak_allocated_bytes":
            torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes":
            torch.cuda.max_memory_reserved()
    }
```

This helper assumes the model and batch are already on CUDA.

The measurement includes the model, inputs, activations, gradients, and other live tensors in the process. It is not a direct measurement of adapter memory alone.

LoRA may reduce gradient and optimizer-state memory, but it still stores the original weights and computes activations. Its actual memory savings must be measured under the chosen workload.

### Q59. What are the main lessons and limitations of this LoRA practical?

The practical demonstrates that fine-tuning can be reformulated as learning a low-rank residual update to pre-trained weights.

Its main lessons are:

- Pre-trained representations can be reused for related target tasks.
- Freezing the original weights reduces the number of parameters requiring gradients.
- Low-rank factorization can reduce trainable parameter counts substantially.
- Zero-initializing one adapter factor preserves the original model's initial behavior.
- The rank controls the trade-off between parameter efficiency and adaptation capacity.
- Optimizer-state size, forward/backward speed, and GPU memory are separate quantities that should be measured independently.
- Merging adapters can eliminate the additional adapter operations during inference.
- Saving only the adapters can make task-specific checkpoints much smaller.

There are also limitations.

First, the experiment uses a relatively small custom network rather than a large transformer. Runtime behavior may differ considerably on modern language and vision models.

Second, only the fully connected layers receive LoRA adapters. Other architectures may adapt attention projections, convolutional kernels, or additional modules.

Third, a single validation split and training run cannot establish that one approach is universally superior.

Finally, a smaller trainable parameter count does not guarantee higher accuracy, faster training, lower total checkpoint size, or lower peak memory in every implementation.

The practical should therefore be treated as an experimental comparison, not proof that LoRA always outperforms dense fine-tuning.

### Q60. What should you remember for an exam, and what is the overall conclusion?

The most important concepts and equations are summarized below.

**1\. Standard linear layer**

$$
y=xW^\top+b
$$

**2\. Dense fine-tuning**

The selected weight matrix is updated directly:

$$
W'=W+\Delta W
$$

**3\. LoRA weight update**

$$
\boxed{\Delta W=BA}
$$

where:

$$
A\in\mathbb{R}^{r\times d_{\text{in}}}
$$

$$
B\in\mathbb{R}^{d_{\text{out}}\times r}
$$

**4\. LoRA forward pass**

$$
\boxed{y=xW^\top+x(BA)^\top+b}
$$

**5\. LoRA trainable parameter count**

$$
\boxed{N_{\text{LoRA}}=r(d_{\text{in}}+d_{\text{out}})}
$$

**6\. Parameter count for the four adapted layers**

$$
\boxed{N_{\text{adapter}}=4608r}
$$

**7\. Initial adapter behavior**

If $B=0$, then:

$$
BA=0
$$

and the adapter initially leaves the original layer's output unchanged.

**8\. Notebook-specific training strategy**

- Pre-train `ConvMLP` on CIFAR-10.
- Adapt to five selected CIFAR-100 classes.
- Freeze the convolutional layers.
- Compare dense fine-tuning against LoRA on the four fully connected layers.
- Keep the target classifier trainable.
- Compare accuracy, parameter counts, checkpoint size, optimizer state, runtime, and GPU memory.

**9\. Interpretation of experimental results**

LoRA is successful when it provides a useful balance between target-task performance and resource requirements. It does not need to be faster in every benchmark, and a larger rank does not guarantee higher validation accuracy.

**10\. Final conclusion**

LoRA replaces direct optimization of a large weight update with optimization of two smaller matrices whose product approximates that update. In this practical, the method provides a concrete way to investigate parameter-efficient transfer learning using a pre-trained convolutional-MLP model.

The final conclusion should be based on measured notebook results: report whether LoRA approaches the dense baseline's validation accuracy, how many trainable parameters it saves, and what happens to training time, optimizer-state size, checkpoint size, and peak GPU memory.

No numerical performance claim should be made until the experiments have actually been run.

---

**End of Part 2 — Questions 31–60.**
