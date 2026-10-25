Below is the GitHub Markdown (`.md`) version of the MCQ guide, with all 83 questions, correct answers, and explanations. I’ve preserved the question-focused structure and completed Q83, whose explanation was cut off in your supplied text.

You can copy this into a file such as `advanced_ml_practical_3_mcq.md`.

Writing

# Advanced Machine Learning Practical 3: Self-Supervised Learning — Comprehensive MCQ Guide

This question bank explains the assignment through multiple-choice questions (MCQs), with correct answers and detailed explanations. It covers the theory, the purpose of each notebook section, the PyTorch implementation, the experimental workflow, common coding errors, and the assessment requirements.

The questions follow a question-focused structure: each asks what a concept means, why a particular design choice was made, how the code works, or what would happen if something changed.

---

## Part 1 — Understanding the Assignment and Its Objective

### Q1. What is the primary objective of this practical?

A. To train a model using only CIFAR-10 labels.

B. To maximise validation accuracy on a small labelled target dataset by using representations learned from a larger unlabelled source dataset.

C. To generate new CIFAR-100 images.

D. To replace convolutional neural networks with transformers.

**Correct answer: B**

**Explanation:** The practical investigates whether self-supervised pre-training can improve performance on a downstream classification task when labelled target data is limited. The model first learns useful visual representations from unlabelled images, then those representations are adapted to the target classification task.

The final objective is to achieve better target validation accuracy than a model trained from scratch on the labelled target data alone.

### Q2. What does self-supervised learning mean in this assignment?

A. The model learns without using any loss function.

B. The model creates a learning signal from the structure or transformations of unlabelled data.

C. The model receives correct human annotations for every source image.

D. The model learns only from random noise.

**Correct answer: B**

**Explanation:** The source dataset has no usable class labels. Instead, the student designs a pretext task that creates training targets automatically. For example, an image can be rotated by 90°, 180°, or 270°, and the model is trained to predict which rotation was applied.

The images do not need human class annotations because the transformation itself provides the supervisory signal.

### Q3. What is the difference between a pretext task and a downstream task?

A. They are two names for the same task.

B. The pretext task evaluates final performance, while the downstream task generates artificial labels.

C. The pretext task teaches useful representations using automatically generated targets; the downstream task is the actual target problem.

D. The downstream task always uses unlabelled data.

**Correct answer: C**

**Explanation:** In this practical:

- **Pretext task:** A constructed task performed on unlabelled CIFAR-10 or STL-10 images.
- **Downstream task:** Classification of the selected labelled target classes.

The pretext task is a means to learn transferable features. Its accuracy is not the final objective.

### Q4. What is the purpose of the supervised baseline?

A. To provide a reference for measuring whether self-supervised pre-training helps.

B. To replace the need for validation data.

C. To train the pretext classifier.

D. To guarantee that fine-tuning achieves higher accuracy.

**Correct answer: A**

**Explanation:** The baseline is trained on the labelled target training set without self-supervised pre-training. Later, the fine-tuned model is evaluated on the same validation set.

If the fine-tuned model performs better, this provides evidence that pre-training helped under the chosen experimental setup. It does not guarantee that the improvement is caused solely by pre-training unless other factors are controlled.

### Q5. Which description best matches the source and target datasets in the CIFAR configuration?

A. Source: labelled CIFAR-100 subset; target: unlabelled CIFAR-10.

B. Source: unlabelled CIFAR-10; target: labelled subset of CIFAR-100.

C. Source and target: identical images with identical labels.

D. Source: validation images only; target: training images only.

**Correct answer: B**

**Explanation:** The notebook selects five related classes from CIFAR-100 to form the target classification problem. It uses CIFAR-10 as the larger source dataset and deliberately discards its labels.

The source and target datasets have different classes, but their images share general visual characteristics, such as edges, textures, shapes, and object parts. Those common visual patterns may transfer between datasets.

### Q6. Why can a model benefit from unlabelled source images belonging to different classes from the target classes?

A. All images from different datasets must have identical class labels.

B. The model can learn general visual features that are useful across multiple object categories.

C. The target classifier automatically converts every source image into a correct target label.

D. Different classes eliminate the need for a validation set.

**Correct answer: B**

**Explanation:** A convolutional network can learn reusable features such as edges, contours, textures, and shapes. These features are not exclusive to a particular class.

For example, edge detectors learned from vehicles may also help recognise animals or household objects. However, transfer is not guaranteed: the source and target distributions must share useful information.

### Q7. How does this practical differ from ordinary supervised transfer learning?

A. It does not use a neural network.

B. It pre-trains the backbone on an automatically constructed task rather than source class labels.

C. It never uses labelled data.

D. It does not require fine-tuning.

**Correct answer: B**

**Explanation:** In conventional supervised transfer learning, the source model learns from human-provided source labels. Here, the source class labels are unavailable or deliberately discarded. The model learns through a self-supervised objective before being fine-tuned using the labelled target dataset.

### Q8. How does self-supervised learning differ from the pseudo-labelling exercise mentioned in the notebook?

A. Self-supervised learning always requires fewer images.

B. Pseudo-labelling uses model-generated class predictions as training targets, whereas self-supervised learning constructs a pretext objective from the data itself.

C. Pseudo-labelling is completely unsupervised in every implementation.

D. Self-supervised learning cannot use image augmentations.

**Correct answer: B**

**Explanation:** Pseudo-labelling typically uses a model to assign predicted class labels to unlabelled examples and then trains on those predictions. Self-supervised learning instead uses signals such as applied rotations, crop relationships, or image-view similarity.

The distinction concerns how the training targets are obtained, not simply whether the dataset contains unlabelled images.

---

## Part 2 — Dataset Selection and Data Preparation

### Q9. What is the purpose of `base_data = 'cifar'`?

A. It chooses the target class.

B. It selects the CIFAR dataset configuration rather than STL-10.

C. It changes the model to a fully connected network.

D. It selects the GPU.

**Correct answer: B**

**Explanation:** The variable controls which dataset-loading branch executes:

python

```
base_data = 'cifar'
img_size = 96 if base_data == 'stl10' else 32
```

When `base_data` is `'cifar'`, the notebook uses CIFAR-100 for the labelled target subset and CIFAR-10 for the unlabelled source. When it is `'stl10'`, the notebook uses the labelled and unlabelled splits of STL-10.

The image size is set to 32 for CIFAR and 96 for the original STL-10 images.

### Q10. What does `TARGET_CATEGORY` control?

A. The source dataset's image size.

B. The GPU memory limit.

C. Which CIFAR-100 superclass group defines the target classification task.

D. The number of pretext epochs.

**Correct answer: C**

**Explanation:** The notebook groups CIFAR-100's fine-grained classes into 20 groups, each containing five classes. The student chooses one group, such as `"vehicles"` or `"flowers"`.

For example:

python

```
TARGET_CATEGORY = "vehicles"
```

The five target classes would be bicycle, bus, motorcycle, pickup truck, and train.

**Important:** The original notebook leaves `TARGET_CATEGORY =` incomplete. You must fill in a valid dictionary key before executing that cell.

### Q11. How many target classes are selected from CIFAR-100?

A. 2

B. 5

C. 10

D. 100

**Correct answer: B**

**Explanation:** Each entry in `cifar_superclasses` contains five CIFAR-100 classes. Selecting a category therefore creates a five-class classification problem.

This also means the target classifier's final output layer must have five outputs when using this configuration.

### Q12. Why does the notebook remap the original CIFAR-100 labels?

A. To make the images smaller.

B. To convert the selected five classes into contiguous labels from 0 to 4.

C. To remove duplicate images.

D. To make CIFAR-100 unlabelled.

**Correct answer: B**

**Explanation:** The original CIFAR-100 labels range from 0 to 99. A selected group of five classes might have original labels such as 0, 5, 10, 15, and 20, rather than consecutive integers.

The code creates a mapping:

python

```
class_remap = {
    cifar100.class_to_idx[name]: i
    for i, name in enumerate(superclass_names)
}
subset_labels = [class_remap[t] for t in subset_old_labels]
```

After remapping, the five target labels become 0, 1, 2, 3, 4, which are appropriate targets for a five-class classifier trained with `nn.CrossEntropyLoss()`.

### Q13. What is the purpose of `UnlabelledDataset`?

A. To store images together with ground-truth class labels.

B. To store images without exposing class labels.

C. To train a neural network automatically.

D. To normalise images on disk.

**Correct answer: B**

**Explanation:** The class stores only images and a dataset name.

python

```
class UnlabelledDataset(Dataset):
    def __init__(self, images, name):
        self.images = images
        self.name = name
```

Its `__getitem__()` method returns an image rather than an `(image, label)` pair. This is important because the pretext task must generate its own training targets.

### Q14. What is the main difference between `LabelledDataset` and `UnlabelledDataset`?

A. `LabelledDataset` stores image-target pairs, while `UnlabelledDataset` returns images only.

B. `LabelledDataset` supports only RGB images.

C. `UnlabelledDataset` cannot be used with a `DataLoader`.

D. They are identical except for their names.

**Correct answer: A**

**Explanation:** In `LabelledDataset`, the key operation is:

python

```
return img, target
```

In `UnlabelledDataset`, the key operation is:

python

```
return img
```

The labelled dataset also includes optional class names and methods for counting labels and splitting the dataset. Both classes inherit from PyTorch's `Dataset`, making them compatible with `DataLoader`.

### Q15. Why does `__len__()` matter in a custom PyTorch dataset?

A. It calculates the model's parameter count.

B. It tells PyTorch how many samples the dataset contains.

C. It computes cross-entropy loss.

D. It determines the number of output classes.

**Correct answer: B**

**Explanation:** PyTorch uses `__len__()` to determine the number of accessible samples. This information is needed for operations such as shuffling, batching, and iterating over a dataset.

The implementation is straightforward:

python

```
def __len__(self):
    return len(self.images)
```

### Q16. What does `__getitem__(self, i)` do?

A. Returns the dataset's complete contents.

B. Returns the sample at index `i` in the format defined by the dataset class.

C. Calculates the validation accuracy.

D. Loads the neural network weights.

**Correct answer: B**

**Explanation:** A `DataLoader` requests samples by index. The dataset's `__getitem__()` method retrieves the corresponding image and, when applicable, its target.

The notebook also converts NumPy arrays into PIL images where necessary, allowing torchvision transformations to process them.

### Q17. What is the purpose of `train_test_split(test_frac=0.1, seed=0)`?

A. To put 10% of the source images into the training set.

B. To reserve 10% of the labelled target dataset for validation.

C. To remove 10% of the model parameters.

D. To make the pretext task use 10 classes.

**Correct answer: B**

**Explanation:** The notebook performs the split on `target_data`, not on the source dataset:

python

```
val_frac = 0.1
target_data_train, target_data_val = target_data.train_test_split(
    val_frac, seed=0
)
```

The labelled target data is divided into 90% training and 10% validation data. The validation set is used to compare the baseline and fine-tuned models.

The seed makes the random split reproducible, provided the surrounding code and random-number handling remain consistent.

### Q18. Why should the validation set not be used for gradient updates?

A. It contains no images.

B. It should provide an independent evaluation of how well the model generalises to unseen target examples.

C. It must contain only unlabelled images.

D. It is used exclusively for data augmentation.

**Correct answer: B**

**Explanation:** Training on the validation labels would make the validation score less reliable as a measure of generalisation. The model should update its weights using the training set and use the validation set to assess performance and select reasonable hyperparameters.

Repeatedly tuning decisions against the same validation set can still cause indirect overfitting to that set, so a separate test set would be useful for a final unbiased evaluation if one is available.

### Q19. Why might a student choose STL-10 instead of CIFAR?

A. STL-10 has no unlabelled images.

B. STL-10 provides labelled and unlabelled splits, including 100,000 unlabelled images.

C. STL-10 has only grayscale images.

D. STL-10 always trains faster than CIFAR.

**Correct answer: B**

**Explanation:** STL-10 offers a large unlabelled dataset with a labelled training split, making it suitable for self-supervised learning. Its images are larger, however, so training may be slower and require more GPU memory.

The notebook optionally resizes STL-10 images to 64 × 64 to reduce computational cost.

### Q20. What does the following code do?

python

```
cifarx.targets = [-1] * len(cifarx.data)
```

A. It replaces every image with a black image.

B. It assigns a dummy value to the labels so they cannot serve as useful class annotations.

C. It changes every image into class −1 in the target task.

D. It deletes the images from memory.

**Correct answer: B**

**Explanation:** The code demonstrates the deliberate removal of useful CIFAR-10 class labels. The images remain available for self-supervised pre-training, but the original class targets are not used.

The subsequent `UnlabelledDataset` stores only the images, so the dummy values are not returned during data loading.

### Q21. What is a potential limitation of the notebook's `count_labels()` method?

A. It cannot work with any classification dataset.

B. Its implementation assumes that every class has at least one sample, because it indexes the array of observed counts by class index.

C. It always counts images twice.

D. It changes labels while counting them.

**Correct answer: B**

**Explanation:** `np.unique(..., return_counts=True)` returns counts only for labels actually present. If a class has zero samples, the returned count array may be shorter than the number of classes, making `label_counts[idx]` unsafe.

For this particular CIFAR-100 subset, all selected classes are normally represented, so the issue may not arise. A more robust implementation would initialise counts for every class and then fill in observed counts.

---

## Part 3 — Image Transformations and DataLoaders

### Q22. Why does the notebook use `transforms.ToTensor()`?

A. To convert a neural network into a tensor.

B. To convert an image into a PyTorch tensor suitable for model input.

C. To calculate the classification loss.

D. To remove the image's channels.

**Correct answer: B**

**Explanation:** PIL images and NumPy image arrays are not directly suitable as batches for a typical PyTorch convolutional network. `ToTensor()` converts an image into a tensor with channels first, typically shaped as `(C, H, W)`.

For an 8-bit RGB image, it also converts pixel values to floating point and scales them approximately from 0 to 1.

### Q23. What does `transforms.Normalize(mean=0.5, std=0.2)` do?

A. It divides every image's width by two.

B. It transforms each channel's pixel values using `(x - 0.5) / 0.2`.

C. It converts RGB images into grayscale.

D. It guarantees that every image has zero variance.

**Correct answer: B**

**Explanation:** Normalisation applies the transformation:

$$
x_{\text{norm}} = \frac{x-\mu}{\sigma}
$$

Here, the specified mean is 0.5 and standard deviation is 0.2 for each channel. If the input values are in the range `[0, 1]`, the approximate output range becomes −2.5 to 2.5.

The parameters are fixed values rather than statistics calculated from the dataset. The same preprocessing should be used consistently across baseline training, validation, pre-training, and fine-tuning whenever appropriate.

### Q24. What is the difference between `preprocess` and `preprocess_aug`?

A. `preprocess` uses only basic tensor conversion and normalisation; `preprocess_aug` additionally applies random image augmentations.

B. `preprocess` is for validation, while `preprocess_aug` is for calculating loss.

C. `preprocess_aug` removes all labels.

D. They always produce identical images.

**Correct answer: A**

**Explanation:** The standard pipeline is:

python

```
preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalise,
])
```

The augmented pipeline includes operations such as random resized cropping, horizontal flipping, colour jitter, and random erasing.

Augmentation creates varied training examples and can reduce overfitting. However, augmentations must be compatible with the chosen pretext task.

### Q25. Why is `RandomHorizontalFlip` potentially useful during training?

A. It doubles the number of classes.

B. It creates varied views of the same image and can encourage robustness to horizontal orientation changes.

C. It guarantees that all labels become correct.

D. It removes the need for normalisation.

**Correct answer: B**

**Explanation:** Horizontal flipping is a data augmentation technique. It can help a classifier learn features that are less dependent on whether an object appears on the left or right.

However, some tasks are sensitive to orientation. For example, transformations must be chosen carefully when the task involves predicting image rotations or other geometric changes.

### Q26. What is the purpose of `collate_labelled(batch, aug=False)`?

A. To train the model immediately without a loss function.

B. To convert a list of labelled samples into a batched image tensor and label tensor.

C. To download images from the internet.

D. To create a new neural network for every batch.

**Correct answer: B**

**Explanation:** A dataset generally returns individual samples. The `DataLoader` gathers them into a list, and the custom collate function:

- Extracts the images and labels.
- Applies the chosen image transformation.
- Stacks the image tensors into a batch.
- Converts labels to a `torch.long` tensor.
- Moves both tensors to the selected device.

The returned image batch normally has shape `(B, C, H, W)` and the label batch has shape `(B,)`.

### Q27. Why is `torch.long` used for the target labels?

A. Because `CrossEntropyLoss` expects class indices represented by integer class labels, normally of type `torch.long`.

B. Because all image tensors must be integers.

C. Because it converts class indices into probability distributions.

D. Because it is required by AdamW.

**Correct answer: A**

**Explanation:** For standard multiclass classification, `nn.CrossEntropyLoss()` expects model logits with shape `(B, C)` and integer target labels with shape `(B,)`. The target labels identify the correct class for each example.

The image tensors, by contrast, should generally be floating-point tensors.

### Q28. Why does the notebook use `torch.stack()` in the collate function?

A. To add a new batch dimension by combining individual image tensors of compatible shape.

B. To calculate the mean of all images.

C. To concatenate the class names into a string.

D. To shuffle the dataset.

**Correct answer: A**

**Explanation:** If each transformed image has shape `(3, 32, 32)`, stacking 128 images produces a tensor with shape `(128, 3, 32, 32)`.

The images must have compatible dimensions. Random crops should therefore output a consistent size.

### Q29. Why is `shuffle=True` normally used for the training DataLoader?

A. To randomly reorder training examples across epochs.

B. To change the true labels.

C. To randomly alter the validation set.

D. To guarantee a higher validation accuracy.

**Correct answer: A**

**Explanation:** Shuffling reduces dependence on the original order of samples and helps create varied minibatches. The notebook uses shuffling for the target training loader and the unlabelled source loader, but not for validation.

Shuffling alone does not guarantee improved performance.

### Q30. What is the purpose of `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`?

A. To choose the number of target classes.

B. To select a CUDA GPU when available, otherwise falling back to the CPU.

C. To determine whether the source images have labels.

D. To change the optimiser.

**Correct answer: B**

**Explanation:** This line makes the notebook usable on machines with and without compatible CUDA GPUs. The model and data must be placed on compatible devices during computation.

The notebook's custom collate functions move image and label batches to `device`, and the backbone's constructor also moves the model to the chosen device.

### Q31. Why are `inspect_dataset()` and `inspect_batch()` useful?

A. They replace the need for validation.

B. They allow visual checking of images, labels, predictions, and augmentation results.

C. They automatically correct all preprocessing errors.

D. They compute gradients.

**Correct answer: B**

**Explanation:** Visual inspection is an important debugging step. It helps reveal incorrect labels, unexpected image dimensions, distorted transformations, and preprocessing errors.

For example, before training a rotation prediction model, you should inspect images to verify that the rotation and its generated target label agree.

---

## Part 4 — Understanding the CNN Backbone and Classification Head

### Q32. What is the purpose of `ConvBackbone`?

A. To store the dataset labels.

B. To learn reusable visual features that can later be used by different task-specific heads.

C. To perform only data augmentation.

D. To calculate validation accuracy.

**Correct answer: B**

**Explanation:** The backbone is the feature extractor. It transforms an input image into a feature vector that captures useful information about the image.

The key architectural idea is modularity: the same backbone can be trained with a pretext head and later connected to a different head for the target classification task.

### Q33. What does `nn.Conv2d(3, 32, 3, padding=1)` mean?

A. A convolution from 32 input channels to 3 output channels using a 1 × 1 kernel.

B. A convolution from 3 input channels to 32 output channels using a 3 × 3 kernel with padding of 1.

C. A fully connected layer with 32 classes.

D. A pooling operation with a kernel size of 3.

**Correct answer: B**

**Explanation:** The arguments specify:

- `3`: number of input channels, corresponding to RGB.
- `32`: number of output channels, or learned feature maps.
- `3`: kernel size, meaning a 3 × 3 filter.
- `padding=1`: one pixel of padding around the image boundaries.

With stride 1, this padding preserves the spatial height and width of the feature maps.

### Q34. What is the purpose of `nn.MaxPool2d(2, 2)`?

A. To double the spatial dimensions.

B. To reduce the height and width of feature maps, typically by half.

C. To increase the number of classes.

D. To calculate a classification probability.

**Correct answer: B**

**Explanation:** Max pooling with a 2 × 2 window and stride 2 reduces each spatial dimension by approximately half. This lowers computational cost and increases the effective receptive field of later layers.

For CIFAR inputs of 32 × 32, two such pooling operations reduce the spatial dimensions to 8 × 8, assuming the convolutions preserve spatial size.

### Q35. Why does the backbone use `nn.AdaptiveAvgPool2d(1)`?

A. To convert the feature maps to a spatial size of 1 × 1 per channel.

B. To reduce the number of classes to one.

C. To resize the input image to 1 × 1 before the convolutions.

D. To normalise all model parameters.

**Correct answer: A**

**Explanation:** The preceding convolutional layers produce 256 feature channels. Adaptive average pooling reduces each channel's spatial map to a single average value.

For example, a tensor of shape `(B, 256, 8, 8)` becomes `(B, 256, 1, 1)`. After flattening, the result is `(B, 256)`.

This also makes the architecture less dependent on a specific spatial input size.

### Q36. What is the role of `nn.Dropout(0.3)` in the backbone?

A. It randomly removes 30% of the dataset.

B. During training, it randomly sets approximately 30% of the relevant activations to zero, helping reduce overfitting.

C. It permanently deletes 30% of the model's weights.

D. It removes 30% of the image channels.

**Correct answer: B**

**Explanation:** Dropout is a regularisation technique. During training, it randomly disables activations, encouraging the network not to rely too heavily on individual units.

It is disabled during normal evaluation when `model.eval()` is called.

### Q37. What does `backbone_output_size = 64` mean in this notebook?

A. The input image is 64 × 64.

B. The backbone's final feature vector contains 64 values per image.

C. The model contains exactly 64 parameters.

D. The model predicts 64 target classes.

**Correct answer: B**

**Explanation:** The final layer of the backbone is:

python

```
self.features = nn.Linear(128, num_features)
```

When `num_features=64`, it maps the preceding 128-dimensional representation to 64 output features. Thus, the backbone outputs a tensor with shape `(B, 64)`.

This dimension determines the expected input size of the classification head.

### Q38. What is the purpose of `ClassifierHead`?

A. To replace the backbone's convolutional layers.

B. To convert the backbone's feature representation into predictions for a particular task.

C. To split the dataset into training and validation sets.

D. To normalise the source images.

**Correct answer: B**

**Explanation:** The head maps the backbone's features to task-specific outputs:

python

```
self.projection = nn.Linear(input_size, projection_size)
self.classifier = nn.Linear(projection_size, num_classes)
```

For the target task, the output dimension is the number of target classes. For a rotation prediction pretext task, the output dimension would instead be the number of rotation categories.

### Q39. Why must the pretext and target classification heads generally be different?

A. They operate on different datasets, so they must have different backbone architectures.

B. They predict different targets, so their output dimensions and meanings may differ.

C. The target task cannot use linear layers.

D. The pretext head must contain no trainable parameters.

**Correct answer: B**

**Explanation:** A rotation prediction head might have four outputs, representing four rotation classes. A target classification head might have five outputs, representing the five selected CIFAR-100 classes.

The backbone learns potentially transferable features; the head maps those features to the particular prediction problem.

### Q40. What is the purpose of `nn.Sequential(baseline_backbone, baseline_target_task_head)`?

A. It combines the backbone and head so the output of the backbone is passed into the head.

B. It combines the source and target datasets.

C. It trains the backbone without a loss function.

D. It converts the model into a DataLoader.

**Correct answer: A**

**Explanation:** `nn.Sequential` chains modules in order. When the baseline model receives an image batch, the backbone produces feature vectors, and the target head converts them into class logits.

The complete flow is:

1. **Input image batch:** Shape `(B, 3, H, W)`.
2. **CNN backbone:** Extracts 64-dimensional features.
3. **Task-specific head:** Projects features and produces class logits.
4. **Loss and prediction:** Compared with the relevant targets.

### Q41. What is the difference between logits and probabilities?

A. Logits are raw model outputs; probabilities are values obtained by converting them into a normalised distribution, commonly with softmax.

B. Logits are the true labels.

C. Probabilities must always be integers.

D. Logits and probabilities are always identical.

**Correct answer: A**

**Explanation:** The classifier returns raw scores, called logits. `nn.CrossEntropyLoss()` accepts these logits directly and internally combines log-softmax with the negative log-likelihood loss.

You generally should not apply softmax before passing logits into `CrossEntropyLoss()`, because the loss function already handles this conversion in a numerically stable way.

---

## Part 5 — Training, Optimisation, and Evaluation

### Q42. What does `nn.CrossEntropyLoss()` measure in the supervised classifier?

A. The difference between the number of training and validation images.

B. How poorly the predicted class distribution matches the correct class labels.

C. The number of convolutional filters.

D. The number of correctly classified images.

**Correct answer: B**

**Explanation:** Cross-entropy penalises the model when it assigns insufficient probability to the correct class. Training aims to reduce this loss by adjusting the model parameters.

Lower loss often corresponds to better predictions, but loss and accuracy are not the same metric. A model can become more confident in its correct predictions and reduce loss without changing its accuracy.

### Q43. What is the purpose of `torch.optim.AdamW`?

A. It performs image augmentation.

B. It updates trainable parameters using gradients, with decoupled weight decay.

C. It computes class labels.

D. It divides the dataset into batches.

**Correct answer: B**

**Explanation:** AdamW is an optimisation algorithm that adapts parameter updates using gradient information and incorporates weight decay separately from the adaptive gradient update.

The notebook uses:

python

```
opt = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=l2_reg
)
```

Here, `lr` controls the learning rate and `l2_reg` is passed as the optimizer's weight-decay coefficient.

### Q44. Why does the training loop call `opt.zero_grad()` before backpropagation?

A. PyTorch automatically accumulates gradients, so previous gradients must be cleared before calculating the current batch's gradients.

B. It resets the model's predictions to zero.

C. It clears the training dataset.

D. It prevents the model from learning.

**Correct answer: A**

**Explanation:** By default, PyTorch accumulates gradients in parameter `.grad` attributes. If gradients are not cleared, gradients from multiple batches can unintentionally accumulate.

The usual training sequence is:

python

```
opt.zero_grad()
pred = model(x)
batch_loss = loss_func(pred, y)
batch_loss.backward()
opt.step()
```

Each step computes predictions, calculates loss, calculates gradients, and updates the parameters.

### Q45. What does `batch_loss.backward()` do?

A. It updates the model weights directly.

B. It computes gradients of the loss with respect to the trainable parameters.

C. It calculates validation accuracy.

D. It converts labels into one-hot vectors.

**Correct answer: B**

**Explanation:** Backpropagation uses automatic differentiation to calculate gradients. The optimiser then uses those gradients to update the parameters when `opt.step()` is called.

Without a differentiable loss and backpropagation, the standard gradient-based training loop would not learn from the current batch.

### Q46. What does `top1_acc(pred, y)` calculate?

A. The percentage of images for which the model's highest-scoring class matches the true label.

B. The mean training loss.

C. The number of model parameters.

D. The probability that the entire dataset is correctly classified.

**Correct answer: A**

**Explanation:** The function is:

python

```
def top1_acc(pred, y):
    return (pred.argmax(axis=1) == y).float().mean().item()
```

For every image, it selects the class with the highest logit, compares it with the correct label, converts the Boolean comparisons to floating-point values, and averages them.

The result is a fraction between 0 and 1. For example, 0.8 means 80% accuracy on the evaluated examples.

### Q47. Why does `evaluate_model()` use `model.eval()` and `torch.no_grad()`?

A. To increase the learning rate during validation.

B. To put the model into evaluation mode and disable gradient calculation.

C. To train the model twice per epoch.

D. To shuffle validation labels.

**Correct answer: B**

**Explanation:** `model.eval()` changes the behaviour of modules such as dropout and batch normalisation when they are present. `torch.no_grad()` prevents autograd from recording operations, reducing memory consumption during evaluation.

Neither operation updates model parameters. The validation data is used to assess performance, not to train the model.

### Q48. What does `metrics.best_val_acc` return?

A. The final training accuracy.

B. The highest validation accuracy recorded in the metrics object.

C. The average of training and validation losses.

D. The accuracy from the first epoch only.

**Correct answer: B**

**Explanation:** The property is implemented as:

python

```
@property
def best_val_acc(self):
    return np.max(self.val_acc)
```

It returns the maximum validation accuracy across the recorded epochs.

This differs from the final epoch's accuracy. A model can perform best at an earlier epoch and subsequently overfit.

### Q49. Why does the notebook record training and validation metrics separately?

A. Training metrics show optimisation performance, while validation metrics help assess generalisation to held-out examples.

B. The two metrics must always be identical.

C. Validation metrics are used only for image display.

D. Training metrics are never useful.

**Correct answer: A**

**Explanation:** Comparing the curves can reveal overfitting.

For example, training loss may continue to decrease while validation loss increases. This suggests the model is fitting the training examples increasingly well without improving its performance on unseen examples.

The `TrainingMetrics` class stores batch-level training measurements and epoch-level validation measurements, then plots them for inspection.

### Q50. What is the purpose of the baseline's `ft_l2_reg = 1e-1`?

A. To specify a validation fraction of 10%.

B. To set the weight-decay coefficient used by the optimiser.

C. To select one target class.

D. To specify 0.1 training epochs.

**Correct answer: B**

**Explanation:** The value `1e-1` equals 0.1. In the notebook it is passed to AdamW as `weight_decay`, which regularises the parameter updates.

The variable name `ft_l2_reg` is inherited from the fine-tuning terminology, but the same value is also used for baseline training. It should not be confused with the learning rate or the number of epochs.

### Q51. Why can the supervised baseline overfit quickly?

A. The labelled target dataset is relatively small compared with the model's capacity.

B. The model never receives labels.

C. The validation set is used as the training set.

D. Convolutional layers cannot learn visual features.

**Correct answer: A**

**Explanation:** A neural network with many trainable parameters can memorise a small training set, especially when trained for many epochs without sufficient regularisation or augmentation.

The notebook deliberately uses a small target subset to demonstrate why pre-training might be useful. Overfitting is a motivation for the experiment, not a guarantee that self-supervised learning will solve the problem.

### Q52. Is the baseline a true lower bound on achievable accuracy?

A. Yes, because no model can perform worse than a baseline.

B. No. It is a reference performance, not a mathematical lower bound on what other models can achieve.

C. Yes, because it uses cross-entropy.

D. No, because validation accuracy is meaningless.

**Correct answer: B**

**Explanation:** The notebook describes the baseline as a lower-bound reference in an informal experimental sense. A self-supervised model may perform worse than the baseline, especially if the pretext task is poorly designed, pre-training is insufficient, or the transferred features are not useful.

The baseline is the comparison point the student aims to beat.

---

## Part 6 — Designing and Implementing the Pretext Task

### Q53. What must the student implement in Task 1?

A. A new target dataset with more human annotations.

B. A self-supervised pretext task, including a way to generate its training inputs and targets, and a model trained on that task.

C. A replacement for the validation set.

D. Only a plot of the baseline results.

**Correct answer: B**

**Explanation:** The notebook leaves a placeholder:

python

```
### implement your pretext task here!
...
```

The student must choose a suitable task, generate the required targets from unlabelled images, create the appropriate data pipeline and model head, and train the backbone to solve the task.

Simply running the baseline training loop again does not satisfy the main objective.

### Q54. Which of the following is a valid self-supervised pretext task?

A. Predict the original CIFAR-10 class using its original class label.

B. Predict which of four rotations was applied to an unlabelled image.

C. Predict a random label unrelated to the image.

D. Predict the target class using validation labels during pre-training.

**Correct answer: B**

**Explanation:** A rotation prediction task constructs its targets automatically. For example, rotate an image by 0°, 90°, 180°, or 270°, and assign labels 0, 1, 2, or 3 respectively.

The model must learn image features that help distinguish these transformations. The other options either depend on unavailable source annotations, provide no meaningful learning signal, or misuse validation labels.

### Q55. How would you implement the four-way rotation task?

A. Randomly rotate each image but assign a fixed label of 0 to every image.

B. Rotate each image by one of four specified angles and use the applied angle's category as the label.

C. Rotate only the validation images.

D. Remove the image pixels and train on the rotation labels alone.

**Correct answer: B**

**Explanation:** Each transformed image must be paired with the correct rotation target. A simple implementation conceptually follows this procedure:

python

```
angles = [0, 90, 180, 270]

# For each source image:
# 1. Choose a rotation category from 0 to 3.
# 2. Rotate the image by the corresponding angle.
# 3. Apply compatible additional augmentations.
# 4. Return the transformed image and rotation category.
```

A dedicated collate function or dataset can perform this work. The pretext head would output four logits, and cross-entropy would train the model to predict the rotation category.

**Important:** Do not apply an additional random rotation after assigning the target unless you also update the label. Otherwise, the input and its target can become inconsistent.

### Q56. Why should the pretext task not be too easy?

A. An easy task always produces a larger loss.

B. A task can be solved using trivial cues without learning useful features for the downstream task.

C. An easy task cannot use neural networks.

D. A difficult task always guarantees better downstream accuracy.

**Correct answer: B**

**Explanation:** The purpose of pre-training is to learn representations that transfer, not merely to maximise pretext accuracy.

For example, a model may solve a simple image restoration task by learning low-level pixel statistics without learning meaningful object structure. A more informative task may require the model to understand shapes, object parts, or relationships between image regions.

However, making a task arbitrarily difficult is not the solution either: a task that cannot be optimised reliably may not produce useful features.

### Q57. What is contrastive learning trying to achieve?

A. To make representations of related image views more similar and distinguish them from unrelated views.

B. To make every image's feature vector identical.

C. To predict only the image's RGB values.

D. To eliminate all data augmentation.

**Correct answer: A**

**Explanation:** Contrastive learning constructs positive and negative relationships between examples.

For example, two augmented views of the same image may be treated as a positive pair, while views from different images are treated as negatives in a particular contrastive formulation.

The learning objective encourages representations to capture useful similarities rather than merely memorising exact pixels. The notebook suggests pairwise contrastive loss and triplet loss as simpler alternatives to a full SimCLR-style objective.

### Q58. What is the difference between a classification pretext task and a regression pretext task?

A. Classification predicts categories; regression predicts continuous values.

B. Classification always uses unlabelled data, whereas regression always uses labelled data.

C. Regression cannot be self-supervised.

D. There is no difference.

**Correct answer: A**

**Explanation:** Consider these examples:

- **Classification:** Predict one of four rotation categories.
- **Regression:** Predict the strength of a colour transformation or numerical crop parameters.

The loss function must match the target format. `CrossEntropyLoss` is appropriate for integer class targets, while a regression task might use `nn.MSELoss()` or `nn.L1Loss()`.

### Q59. Why must the student create a suitable collate function or dataset for the pretext task?

A. Because the ordinary unlabelled loader returns only images, while the pretext model generally needs images paired with automatically generated targets.

B. Because PyTorch cannot load unlabelled data.

C. Because the target labels must be copied into the source dataset.

D. Because collate functions automatically create a new backbone.

**Correct answer: A**

**Explanation:** The existing `source_unsup_loader` returns batches of images. For a rotation task, the training loop also needs rotation labels. For a contrastive task, it may need two views of each image.

A new data pipeline must therefore generate the inputs and targets required by the chosen objective.

### Q60. What is the role of the pretext classification head during self-supervised pre-training?

A. It produces the task-specific outputs needed to calculate the pretext loss.

B. It replaces the entire dataset.

C. It automatically transfers source class labels to the target dataset.

D. It is always retained unchanged for downstream classification.

**Correct answer: A**

**Explanation:** The pretext head might predict four rotation categories or some other task-specific targets. Its output dimension and loss depend on the selected task.

Once pre-training is complete, the student normally discards this head and attaches a new target head. The reusable component is the backbone.

### Q61. Why are augmentations important during self-supervised pre-training?

A. They create diverse image views that encourage robust features.

B. They always make the pretext task easier.

C. They automatically provide human annotations.

D. They guarantee that the pretext and target tasks are identical.

**Correct answer: A**

**Explanation:** Augmentations can teach the model to recognise useful structure despite changes in cropping, colour, or appearance.

However, the augmentations must not destroy the information needed for the task. For example, if the task is to predict rotation, applying a further random rotation without updating the target would corrupt the training signal.

### Q62. Why might the student use `nn.MSELoss()` instead of `nn.CrossEntropyLoss()`?

A. When the task predicts continuous quantities, such as crop coordinates or transformation intensity.

B. Whenever the model has convolutional layers.

C. Whenever the dataset contains RGB images.

D. Whenever the training set is small.

**Correct answer: A**

**Explanation:** Mean squared error measures the squared difference between predictions and continuous targets.

Cross-entropy, in contrast, is suitable for standard multiclass classification with integer class indices. Choosing the correct loss is part of designing the pretext task.

### Q63. Why does the notebook suggest inspecting transformed images before training?

A. To confirm that transformations work as intended and that generated targets match the actual transformations.

B. To increase the number of training epochs automatically.

C. To remove the need for a loss function.

D. To guarantee that the backbone will transfer successfully.

**Correct answer: A**

**Explanation:** A data pipeline may run without errors while still generating incorrect training pairs. For instance, an image may be rotated by 90° but accidentally assigned label 2 rather than label 1.

Visual inspection and simple assertions help catch these errors before expensive training begins.

### Q64. Does high pretext accuracy necessarily mean the backbone has learned excellent transferable features?

A. Yes, always.

B. No. High accuracy may indicate that the pretext task is easy or that it can be solved using features that do not help the downstream task.

C. Yes, provided the model uses AdamW.

D. No. Pretext accuracy should never improve.

**Correct answer: B**

**Explanation:** The pretext task is an intermediate objective. A model that achieves very high pretext accuracy using trivial cues may learn little that is useful for the target classification problem.

The student should consider pretext loss, convergence stability, the difficulty of the task, and—most importantly—the downstream validation performance.

---

## Part 7 — Transfer Learning and Fine-Tuning

### Q65. What does fine-tuning mean in this practical?

A. Training the model on the pretext task again without changing anything.

B. Adapting the pre-trained backbone to the labelled target classification task.

C. Deleting the backbone and training only on source images.

D. Evaluating the model without updating any parameters.

**Correct answer: B**

**Explanation:** After pre-training, the backbone contains features learned from the unlabelled source images. Fine-tuning adapts those features to the target classification problem using the target labels.

The student attaches a new classification head, then trains the resulting model on `target_train_loader` and evaluates it on `target_val_loader`.

### Q66. What should happen to the pretext head after pre-training?

A. It must always be retained without modification.

B. It is generally replaced with a new head designed for the target task.

C. It becomes the validation dataset.

D. It must be converted into a DataLoader.

**Correct answer: B**

**Explanation:** A pretext head predicts pretext targets, such as rotation categories. The target head predicts the selected target classes.

The output meanings differ, so the pretext head is usually discarded. The backbone is retained because its learned features may be reusable.

### Q67. How should the pre-trained backbone's weights be transferred?

A. Initialise a completely unrelated backbone and ignore the pre-trained weights.

B. Save the pre-trained backbone's `state_dict` and load it into a compatible backbone in the target model.

C. Copy the source image files into the model.

D. Copy the pretext validation accuracy into the target head.

**Correct answer: B**

**Explanation:** The notebook recommends saving and loading the backbone's state dictionary. For example:

python

```
torch.save(
    pretrained_backbone.state_dict(),
    "pretrained_backbone.pth"
)

target_backbone = initialise_backbone()
target_backbone.load_state_dict(
    torch.load("pretrained_backbone.pth", map_location=device)
)
```

This assumes that `pretrained_backbone` is the actual backbone trained on the pretext task and that the architecture is compatible.

The target model can then be constructed using `target_backbone` and a freshly initialised `ClassifierHead`.

### Q68. What is the difference between freezing and fine-tuning the backbone?

A. Freezing prevents the backbone's parameters from being updated; fine-tuning allows some or all of them to change.

B. Freezing means deleting the backbone.

C. Fine-tuning means disabling all gradients.

D. They are exactly the same operation.

**Correct answer: A**

**Explanation:** A frozen backbone acts as a fixed feature extractor. Only the target head, for example, may be trained.

When the backbone is fine-tuned, its parameters can adapt to the target dataset. Freezing may help when the target dataset is very small, while full fine-tuning may be more effective when the pre-trained features need substantial adaptation.

A typical freezing operation is:

python

```
for param in target_backbone.parameters():
    param.requires_grad = False
```

The optimiser should then be configured appropriately. If you later unfreeze the backbone, you should ensure that the optimiser includes its parameters.

### Q69. Why might a lower learning rate be useful during fine-tuning?

A. It prevents the target head from making predictions.

B. It can help preserve useful pre-trained features while making smaller adjustments for the target task.

C. It automatically increases the number of classes.

D. It removes the need for validation.

**Correct answer: B**

**Explanation:** Pre-trained parameters already contain potentially useful information. Large parameter updates may destroy that information, especially when the labelled target dataset is small.

A lower learning rate can make fine-tuning more gradual. The best value depends on the dataset, architecture, and pretext task.

### Q70. Why should baseline and fine-tuning comparisons use consistent experimental conditions?

A. To make the comparison more meaningful by reducing differences unrelated to pre-training.

B. To guarantee exactly the same accuracy.

C. To ensure the models have identical initial weights.

D. To eliminate all sources of experimental uncertainty.

**Correct answer: A**

**Explanation:** The comparison is most informative when both approaches use the same target training and validation splits, comparable model architectures, and similar training settings.

If the fine-tuned model uses strong augmentation and the baseline does not, any performance difference might partly reflect augmentation rather than pre-training.

The notebook explicitly recommends using the same hyperparameters for baseline and fine-tuning initially, although further controlled experiments may explore alternatives.

### Q71. What does this calculation measure?

python

```
improvement = ft_acc - baseline_acc
```

A. The relative percentage improvement divided by baseline accuracy.

B. The absolute difference in validation accuracy, expressed as a fraction of the full accuracy scale.

C. The training loss reduction.

D. The percentage of images removed from the dataset.

**Correct answer: B**

**Explanation:** Suppose the baseline accuracy is 0.60 and the fine-tuned accuracy is 0.72.

$$
\text{Improvement} = 0.72 - 0.60 = 0.12
$$

That is an improvement of **12 percentage points**.

The relative improvement is a different quantity:

$$
\frac{0.72-0.60}{0.60}\times100=20\%
$$

The notebook prints the absolute difference in percentage points, not the relative percentage improvement.

### Q72. What if the fine-tuned model performs worse than the baseline?

A. The experiment is invalid and must be deleted.

B. The result is still informative and should be reported honestly, with investigation of possible causes.

C. The baseline accuracy should be changed to make the fine-tuned model look better.

D. The validation labels should be used for pre-training.

**Correct answer: B**

**Explanation:** Self-supervised learning does not guarantee improvement. Possible causes include:

- A pretext task that teaches irrelevant features.
- Insufficient pre-training.
- Incorrect transformation-label pairs.
- Excessively strong or incompatible augmentations.
- Poor optimisation or overfitting.
- A source distribution that does not transfer well to the target.

A negative result can be valuable if the implementation is sound and the analysis explains what was learned.

---

## Part 8 — Code Comprehension and Debugging

### Q73. What is wrong with the notebook's final augmentation example as written?

python

```
from torchvision.transforms import v2

random_augment = v2.Compose([
    v2.RandomResizedCrop(size=img_dim, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    ...
])
```

A. `RandomHorizontalFlip` does not exist.

B. `img_dim` is not defined in the supplied notebook; the intended variable may be `img_size`.

C. `v2.Compose` cannot combine transformations.

D. Every torchvision transform requires a CUDA tensor.

**Correct answer: B**

**Explanation:** Earlier, the notebook defines:

python

```
img_size = 96 if base_data == 'stl10' else 32
```

But the later example refers to `img_dim`, which is undefined in the provided code. This can cause a `NameError` when the cell runs.

A likely correction is:

python

```
v2.RandomResizedCrop(size=img_size, scale=(0.8, 1.0))
```

There is also a pipeline detail to watch: v2 transformations can operate on supported image types, but the input type, transformation order, and normalisation must be compatible. Avoid applying `ToTensor()` twice.

### Q74. Why might a custom pretext training loop be necessary?

A. The ordinary supervised loop cannot train any model on images.

B. Some pretext tasks need multiple image views, feature comparisons, or losses that differ from standard classification.

C. PyTorch does not support backpropagation for pretext tasks.

D. All pretext tasks require a different programming language.

**Correct answer: B**

**Explanation:** A rotation prediction task can use a conventional supervised training loop because it produces one image and one integer target per example.

A contrastive task may need two augmented views of the same image, separate backbone passes, and a similarity-based loss. An image-to-image prediction task may need pixel-level outputs and a regression or reconstruction loss.

The training loop should match the inputs and objective of the chosen task.

### Q75. What is a potential problem with the notebook's validation loss aggregation?

A. The model cannot compute validation loss.

B. It averages batch losses equally, so batches of different sizes may receive equal weight.

C. It always produces a negative loss.

D. It includes training gradients by definition.

**Correct answer: B**

**Explanation:** The code appends each batch's mean loss and then calculates:

python

```
val_loss = np.mean(batch_val_losses)
```

If all validation batches have the same size, this is a reasonable approximation. However, the final batch may be smaller than the others. An unweighted mean gives that smaller batch the same influence as a full batch.

A more precise dataset-level mean loss weights each batch by its number of samples:

python

```
total_loss = 0.0
total_samples = 0

model.eval()
with torch.no_grad():
    for vx, vy in val_loader:
        vpred = model(vx)
        loss = loss_func(vpred, vy)

        total_loss += loss.item() * len(vy)
        total_samples += len(vy)

val_loss = total_loss / total_samples
```

The accuracy function in the notebook has a similar equal-batch-weighting limitation when the last batch is smaller.

### Q76. Why should the student be careful about how the random seed is set?

A. A seed guarantees identical results on every GPU and every software version.

B. Reproducibility depends on more than a single seed, including dataset splits, model initialisation, data-loader randomness, and nondeterministic operations.

C. Seeds prevent the model from learning.

D. Seeds are relevant only to the validation dataset.

**Correct answer: B**

**Explanation:** The notebook sets seeds in some places, such as the target split and the training function. However, complete reproducibility may require controlling Python, NumPy, and PyTorch random generators and, when necessary, CUDA and data-loader randomness.

A fixed seed makes comparisons more repeatable, but it does not automatically make every operation deterministic.

### Q77. What does `early_stopping=True` attempt to do in `train_supervised()`?

A. Stop training when the training loss reaches exactly zero.

B. Stop training when validation loss appears to be worsening according to the notebook's stopping rule.

C. Stop training after one batch.

D. Stop whenever validation accuracy increases.

**Correct answer: B**

**Explanation:** After several epochs, the code compares the most recent validation loss with the average of the preceding three validation losses:

python

```
recent_val_loss = np.mean(metrics.val_loss[-4:-1])

if metrics.val_loss[-1] > recent_val_loss:
    break
```

This is a simple stopping heuristic, not a robust patience-based early-stopping implementation. It may stop after a temporary increase, even if later epochs would improve performance. The provided baseline call also leaves early stopping disabled by default.

### Q78. Why is `model.train()` called at the beginning of each training epoch?

A. To switch modules such as dropout into training mode.

B. To initialise new model weights every epoch.

C. To remove the classification head.

D. To create a new dataset split.

**Correct answer: A**

**Explanation:** PyTorch modules can behave differently in training and evaluation modes. Dropout, for example, randomly masks activations during training but not during evaluation.

Calling `model.train()` ensures the model is in the correct mode before training batches are processed. It does not reinitialise the weights.

---

## Part 9 — Results, Interpretation, and Assessment

### Q79. Which result provides the strongest direct evidence that the practical's objective has been achieved?

A. The pretext task reaches 100% accuracy.

B. The fine-tuned model achieves higher target validation accuracy than the supervised baseline under a fair comparison.

C. The source dataset contains more images than the target dataset.

D. The backbone has more trainable parameters than the classification head.

**Correct answer: B**

**Explanation:** The ultimate objective is improved downstream performance. Pretext loss and accuracy help assess whether pre-training is working, but they are not substitutes for target validation results.

A fair comparison requires consistent data splits and appropriately controlled training conditions. Repeated runs may be helpful because neural network results can vary with random initialisation and data order.

### Q80. What should the final assessment report contain?

A. Only the code for downloading the dataset.

B. A description and motivation for the pretext task, its implementation, plots comparing validation results, and a discussion of issues and insights.

C. Only the highest pretext accuracy.

D. Only a screenshot of the training loss.

**Correct answer: B**

**Explanation:** The assessment specifically requests three elements:

1. **Pretext task:** Explain the task, why you selected it, and how you generated its training inputs and targets.
2. **Results:** Provide plots or other clear comparisons of the supervised baseline and the self-supervised, fine-tuned model.
3. **Reflection:** Discuss problems encountered, how they were resolved, and what you learned about pretext-task design and transfer learning.

The report should explain not just what happened, but why the observed results are plausible.

---

## Part 10 — Putting the Complete Assignment Together

### Q81. Which sequence correctly describes the complete experimental workflow?

A. Fine-tune → remove labels from target data → train baseline → pre-train.

B. Create target split → train supervised baseline → construct pretext task → pre-train backbone → replace head → fine-tune → compare results.

C. Train on validation labels → pre-train target head → discard all model weights.

D. Train the pretext head on the target validation set → evaluate on source labels.

**Correct answer: B**

**Explanation:** The order matters because the baseline must be established before the final comparison, and the backbone must learn from the pretext task before its weights can be transferred.

The notebook's workflow can be summarised as follows:

1. **Prepare datasets:** Small labelled target set and large unlabelled source set.
2. **Baseline:** Train a randomly initialised model on target labels.
3. **Pre-training:** Learn features from unlabelled images using a pretext task.
4. **Transfer the backbone:** Discard the pretext head and attach a new target head.
5. **Fine-tune on target labels:** Train using the target training set.
6. **Compare validation performance:** Baseline accuracy versus post-SSL fine-tuning accuracy.

### Q82. Which implementation choice is most important when transferring a pre-trained backbone?

A. The target model must use the same pretext head.

B. The backbone architecture and parameter shapes must be compatible with the saved weights.

C. The source and target datasets must have identical class labels.

D. The target dataset must be unlabelled.

**Correct answer: B**

**Explanation:** A state dictionary contains learned parameter values associated with particular model layers. Loading it into a compatible backbone allows the learned parameters to be reused.

The target head can differ because it has a different purpose and potentially a different output dimension. The source and target classes also do not need to be identical, which is central to the transfer-learning setup.

### Q83. Why should the target validation set remain separate from the source pretext training process?

A. Because validation images cannot be processed by a CNN.

B. Because it helps preserve a clean evaluation of the downstream task and avoids using target validation information to guide representation learning.

C. Because source images must have the same labels as the target validation images.

D. Because the validation set must contain only images from the source dataset.

**Correct answer: B**

**Explanation:** The target validation set should be reserved for assessing how well the model generalises to unseen target examples. Using its images or labels to design, train, or repeatedly optimise the pretext learning process can introduce information leakage and make the final validation score less independent.

The self-supervised backbone should be pre-trained using the designated unlabelled source training data. The labelled target training set is then used for fine-tuning, while the target validation set is used for evaluation and model selection.

This separation makes the baseline-versus-fine-tuned comparison more meaningful. If a separate test set is available, it should be reserved for the final evaluation after model and hyperparameter decisions have been completed.

---

## Final Revision Checklist

Before submitting the practical, ensure that you can explain the following:

- [ ]The difference between self-supervised learning, supervised learning, and pseudo-labelling.
- [ ]The roles of the source dataset, target dataset, pretext task, and downstream task.
- [ ]How labelled and unlabelled PyTorch datasets differ.
- [ ]How image transformations, normalisation, augmentation, and DataLoaders work.
- [ ]The purpose of the CNN backbone and task-specific classification heads.
- [ ]How `CrossEntropyLoss`, AdamW, backpropagation, and accuracy calculations work.
- [ ]How to design and implement a valid self-supervised pretext task.
- [ ]How to save and transfer backbone weights and replace the pretext head.
- [ ]How to compare baseline and fine-tuned validation performance fairly.
- [ ]How to identify coding errors, validation leakage, overfitting, and reproducibility problems.
- [ ]How to interpret negative results and discuss limitations honestly.
- [ ]How to present pretext-task motivation, experimental results, plots, and lessons learned in the final report.

**Key takeaway:** The practical is not simply about training a neural network. It is about designing a useful learning signal from unlabelled data, transferring the resulting representation to a labelled target task, and evaluating whether that transfer improves downstream performance under a fair experimental comparison.
