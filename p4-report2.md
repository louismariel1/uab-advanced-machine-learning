# Part 2 — Data preparation and model architecture

These cells establish how the images are inspected, how the datasets are split and processed, and which neural network will be used to compare full fine-tuning with LoRA.

The most important result so far is that the baseline model contains 1,334,026 trainable parameters. This gives us a reference point for measuring how many parameters LoRA can save later.

## 1. Visualising the datasets

The function `inspect_data(data, layout=(3,5))` randomly selects 15 images and displays them with their class labels.

python
```
num_examples = np.prod(layout)
example_idxs = np.random.choice(
    range(len(data)), num_examples, replace=False
)
```

- `layout=(3,5)` creates a grid of 3 rows and 5 columns.
- `replace=False` ensures that the same dataset index is not selected twice.
- `imshow()` displays each image, while `set_title()` adds the class label.
- The function displays both CIFAR-10 and the selected CIFAR-100 vehicle subset.

### Why is this useful?

Visual inspection helps verify that the data have been loaded correctly and that the target subset contains the expected classes. It can also reveal unusual images or unexpected label mappings.

The images are displayed before normalisation, so they should retain their familiar visual appearance.

Limitation: inspecting 15 randomly selected images is a useful sanity check, but it does not establish dataset quality or demonstrate model performance. Also, the sampled images can differ between executions because this function does not explicitly seed NumPy's random generator.

No visualisation output has been supplied here, so we should not claim that specific image properties were observed.

## 2. Training and validation splits

The notebook divides each dataset into training and validation subsets using:

python
```
val_frac = 0.2
batch_size = 256
device = 'cuda'
```

The resulting splits are:

| Dataset | Training examples | Validation examples | Total |
| --- | --- | --- | --- |
| CIFAR-10 | 40,000 | 10,000 | 50,000 |
| CIFAR-100 vehicles | 2,000 | 500 | 2,500 |

### Target dataset split

Training

80% · 2,000 images

Validation

20% · 500 images

### Why use a validation set?

The training set is used to calculate gradients and update model parameters. The validation set provides an independent set of images for monitoring generalisation and comparing model configurations.

The target training set contains only 2,000 images. This makes the task suitable for investigating parameter-efficient fine-tuning, because adapting a model using a relatively small labelled dataset can be challenging.

The call to `torch.manual_seed(0)` helps make the PyTorch random operations reproducible, including the dataset splitting and later model initialisation, provided the same execution environment and random-number usage are maintained.

### An implementation detail worth noting

The split uses:

python
```
random_split(source_data, [1-val_frac, val_frac])
```

and the corresponding call for the target dataset.

In standard PyTorch, floating-point fractions passed to `random_split` are supported in current versions and are converted into integer lengths. This produces the reported 40,000/10,000 and 2,000/500 splits.

Also, `device = 'cuda'` explicitly selects a CUDA GPU. The notebook will fail at tensor transfer if CUDA is unavailable; the supplied output does not yet establish whether GPU execution was actually verified.

## 3. Image transformations and data loaders

Both datasets use the following preprocessing:

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

### What does this do?

Step 1 — Convert images to tensors.

`ToTensor()` converts each RGB image into a PyTorch tensor with channel-first shape:

$$
(3,32,32)
$$

For standard 8-bit RGB images, pixel values are converted from the range 0–255 to approximately 0–1.

Step 2 — Normalise the channels.

The transformation applies:

$$
x_{\mathrm{norm}}=\frac{x-0.5}{0.2}
$$

to each channel. Thus, an original pixel value of 0 becomes $-2.5$, while a value of 1 becomes $2.5$.

This centres the nominal pixel range around zero and scales it. The constants are fixed values rather than means and standard deviations estimated from the dataset.

### What do the collate functions do?

The two custom functions, `collate_source()` and `collate_target()`, perform three main operations:

1. Extract images and labels from each batch.
2. Transform the images and stack them into a batch tensor.
3. Convert labels to integer tensors and move both images and labels to the GPU.

The target collate function has an additional important operation:

python
```
labels = [
    target_data.remap[b[1]] for b in batch
]
```

This converts the original CIFAR-100 class indices into the five target labels, 0–4. Without this conversion, the target classifier's five outputs would not align with the original class indices.

The data loaders use batches of up to 256 images and shuffle their samples. The validation loaders are also shuffled, which does not prevent valid evaluation but is unnecessary when only aggregate validation metrics are required.

One detail to monitor: both datasets use the same normalisation constants. This keeps preprocessing consistent between source and target, which is useful for transfer learning.

## 4. Understanding the ConvMLP architecture

The notebook defines a custom neural network called `ConvMLP`.

Input image

RGB · 3 × 32 × 32

Convolution 1

3 → 64 channels · ReLU · MaxPool

Spatial size: 32 × 32 → 16 × 16

Convolution 2

64 → 256 channels · ReLU

Adaptive average pooling → 256 features

Fully connected MLP

256 → 1024 → 512 → 512 → 256

ReLU after each hidden layer

Classification layer

256 → 10 logits for CIFAR-10 pretraining

### Why use this architecture?

The convolutional layers extract spatial visual features, while the multilayer perceptron (MLP) processes those features to perform classification.

The second convolutional layer is followed by adaptive average pooling to reduce its output to a single spatial value per channel. The MLP therefore receives only 256 features per image, rather than a flattened representation of every spatial position.

The architecture is intentionally simple rather than state-of-the-art. The practical explicitly chooses a model with large fully connected layers because LoRA is straightforward to demonstrate on linear weight matrices.

Although LoRA is commonly associated with transformer models, its underlying idea is not limited to transformers.

## 5. Interpreting the parameter counts

The model summary provides the first quantitative result of the experiment.

Total parameters

# 1,334,026

Trainable parameters

## 1,334,026

Frozen parameters

## 0

100% of model parameters are initially trainable.

The table breaks down the parameters by layer:

| Layer | Parameters | Share of total (approx.) |
| --- | --- | --- |
| First convolution | 1,792 | 0.13% |
| Second convolution | 147,712 | 11.07% |
| Fully connected layer 1 | 263,168 | 19.73% |
| Fully connected layer 2 | 524,800 | 39.34% |
| Fully connected layer 3 | 262,656 | 19.69% |
| Fully connected layer 4 | 131,328 | 9.84% |
| Classifier | 2,570 | 0.19% |
| Total | 1,334,026 | 100% |

The second fully connected layer, `fc2`, has the largest number of parameters: 524,800. It accounts for approximately 39.3% of the model's total parameters.

The four hidden fully connected layers collectively contain 1,181,952 parameters, approximately 88.6% of the model total. This makes the MLP an especially relevant place to investigate low-rank adaptation.

### Why does the parameter count matter for LoRA?

A conventional linear layer computes:

$$
y=Wx+b
$$

where $W$ is its weight matrix and $b$ is its bias.

Under full fine-tuning, the weight matrix and bias can both be updated. LoRA instead freezes the original weight matrix and introduces a trainable update:

$$
W'=W+\Delta W
$$

where:

$$
\Delta W=BA
$$

For a weight matrix $W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$, a rank-$r$ adaptation uses:

$$
A\in\mathbb{R}^{r\times d_{\text{in}}},
\qquad
B\in\mathbb{R}^{d_{\text{out}}\times r}
$$

The number of trainable parameters for this adaptation is:

$$
r(d_{\text{in}}+d_{\text{out}})
$$

rather than $d_{\text{in}}d_{\text{out}}$ for the original weight matrix, excluding bias parameters from this comparison.

For example, `fc2` has a weight matrix of shape $512\times1024$, containing 524,288 weight parameters. A rank-8 adaptation would use only:

$$
8(512+1024)=12,288
$$

trainable adaptation parameters for that weight matrix, while freezing its original weights. That is about 2.34% of its original weight count. This is an illustrative calculation, not yet an observed notebook result; the actual savings depend on which layers LoRA is applied to and whether biases or other layers remain trainable.

## 6. Important distinction: this is not yet a trained model

The cell initialises `pt_model` and inspects its architecture:

python
```
pt_model = ConvMLP(
    num_classes=source_data.num_classes
).to(device)

pt_model.report_parameters()
torchinfo.summary(pt_model)
```

At this point, the network has randomly initialised weights. It has not yet been pretrained on CIFAR-10, so the parameter count says nothing about classification accuracy.

The next stages should train this model on CIFAR-10, then adapt it to the five-class target task. We will need those subsequent outputs before judging whether transfer learning or LoRA actually improves performance.

## 7. Interim conclusions for the final report

We can now record the following findings.

- The practical uses an 80/20 training-validation split for both the source and target datasets.
- The target task consists of five balanced vehicle classes and only 2,000 training examples.
- Images are converted to tensors and normalised, and target labels are remapped to the range 0–4.
- The model contains 1,334,026 parameters, all initially trainable.
- The fully connected layers account for most of the model's parameters, providing a natural setting for demonstrating LoRA.
- No training accuracy, validation accuracy, loss or measured computational savings have been reported yet.

One methodological point to watch in the coming cells is whether the target classifier is changed from 10 outputs to 5 outputs before fine-tuning. A five-class target task requires five output logits, and the pretrained feature layers must be handled consistently.

Paste the next cell and its output when you're ready. We'll continue the investigation and progressively assemble the evidence for the final report.
