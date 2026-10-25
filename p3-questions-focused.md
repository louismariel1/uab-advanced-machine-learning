# Practical 3 — Self-Supervised Learning: Assignment Explained

## 1. What is the main objective of this assignment?

Question: What are you trying to achieve in this notebook?

The goal is to improve the classification accuracy of a neural network trained on a small labelled dataset by first teaching it useful visual features using a much larger unlabelled dataset.

This technique is called self-supervised learning (SSL). Instead of relying on manually supplied labels during pre-training, you design a task that generates its own training targets from the data.

The notebook follows three main stages:

Stage 1 — Supervised baseline

Train a model using the small labelled target dataset.

Output: `baseline_model` and `baseline_metrics`.

Stage 2 — Self-supervised pre-training

Train the backbone on unlabelled source images using your own pretext task.

Output: a backbone with learned visual representations.

Stage 3 — Supervised fine-tuning

Attach a new classification head and train the model on the labelled target dataset.

Output: `ft_metrics` and the final validation accuracy.

The main comparison is:

$$
\text{Supervised baseline}
\quad \text{vs.} \quad
\text{SSL pre-trained + fine-tuned model}
$$

The objective is to determine whether pre-training on unlabelled images helps the model perform better on the downstream classification task.

Important: Improvement is the intended outcome, not a guaranteed result. It depends on the pretext task, the data, the training procedure, and the quality of the learned representations.

## 2. What is self-supervised learning, and how does it differ from other learning paradigms?

Question: What does self-supervised learning mean?

Self-supervised learning creates a training signal from the structure or content of unlabelled data.

For example, suppose you take an image of a car, rotate it by 90 degrees, and ask a neural network to predict the rotation angle. You know which rotation you applied, so you can generate the correct target automatically without knowing the image's original class.

The network might learn edges, shapes, object parts, and spatial relationships while solving this task.

Question: How does SSL differ from supervised, semi-supervised, and transfer learning?

| Learning approach | Training information | Main purpose |
| --- | --- | --- |
| Supervised learning | Images and true labels | Learn the target classification task directly |
| Semi-supervised learning | A mixture of labelled and unlabelled data, often for the same task | Improve target-task learning using additional unlabelled examples |
| Self-supervised learning | Unlabelled images plus automatically generated training targets | Learn reusable representations |
| Transfer learning | A model or its weights learned from another task or dataset | Reuse learned knowledge on a new task |

Self-supervised learning and transfer learning are complementary rather than mutually exclusive. In this practical, SSL is the pre-training method, and transfer learning is the process of reusing the resulting backbone for the target task.

Question: Why can knowledge transfer between datasets with different classes?

The source and target datasets do not need to share the same class labels to share useful visual features.

For example, an unlabelled image dataset might help a network learn:

- Edges, corners, and textures.
- Curved and straight boundaries.
- Shapes and object parts.
- Spatial relationships and visual patterns.

These features can be useful for classifying different objects later, even if the source images do not contain the target classes.

The notebook therefore aims to learn general visual representations, rather than memorising the class labels of the source dataset.

## 3. What are the source dataset and target dataset?

Question: Which datasets does the notebook use?

The notebook offers two configurations. The active setting is:

python

```
base_data = 'cifar'
```

This selects CIFAR-100 as the target dataset and CIFAR-10 as the source dataset.

Medium

Target dataset — CIFAR-100 subset

LabelledThe notebook selects one superclass containing five CIFAR-100 classes. These labelled examples are split into training and validation subsets for the downstream classification task.

CS440 Lectures

Source dataset — CIFAR-10

UnlabelledThe notebook uses the images but replaces the labels with `-1`. Your pretext task must generate the targets needed for pre-training.

The target subset selection is incomplete in the supplied notebook:

python

```
TARGET_CATEGORY =
```

You must choose one of the keys in `cifar_superclasses`, such as:

python

```
TARGET_CATEGORY = "vehicles"
```

This selects the five classes bicycle, bus, motorcycle, pickup truck, and train.

The code then remaps their original CIFAR-100 class indices to five consecutive labels, `0` through `4`.

Question: What is the alternative STL-10 configuration?

If you change `base_data` to `'stl10'`, the notebook uses:

- The labelled training split of STL-10 as the target dataset.
- The unlabelled split of STL-10 as the source dataset.
- A resize to 64 × 64 pixels when `RESIZE_STL = True`.

STL-10 includes 100,000 unlabelled images, making it a more substantial pre-training dataset, but it requires more storage and computation.

The two configurations demonstrate the same general method, but the active CIFAR configuration is the one you should explain and run unless you deliberately choose STL-10.

## 4. How is the target dataset prepared?

Question: How does the notebook split the labelled data?

The notebook uses:

python

```
val_frac = 0.1

target_data_train, target_data_val = (
    target_data.train_test_split(val_frac, seed=0)
)
```

This reserves 10% of the selected target subset for validation and uses the remaining 90% for training.

For example, if the selected subset contains 2,500 images, the split would contain approximately 2,250 training images and 250 validation images.

The actual numbers depend on the selected dataset and its size.

The fixed seed makes the split reproducible, provided the rest of the execution is consistent.

Question: What are `LabelledDataset` and `UnlabelledDataset` responsible for?

- `LabelledDataset` stores images, their targets, the dataset name, and optional class names.
- `UnlabelledDataset` stores images without labels.
- Both classes convert NumPy image arrays into PIL images when retrieving individual examples.
- `__len__()` returns the number of samples.
- `__getitem__()` defines what a data loader receives for each sample.

These wrappers allow the notebook to use a consistent interface for both kinds of data.

Question: How are images converted into tensors?

The standard preprocessing pipeline is:

python

```
normalise = transforms.Normalize(
    mean=0.5,
    std=0.2,
)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalise,
])
```

`ToTensor()` converts an image to a channel-first tensor and typically scales pixel values from the range 0–255 to 0–1.

Normalization then applies:

$$
x_{\text{norm}} = \frac{x-0.5}{0.2}
$$

For an original scaled pixel value of $0.5$, the normalized value is zero.

Question: Why does the notebook use data augmentation?

Augmentation creates modified versions of training images to improve robustness and reduce overfitting.

The notebook includes transformations such as:

- Random resized crops.
- Horizontal flips.
- Colour jitter.
- Random erasing.
- Random grayscale conversion in the later example.

These operations can encourage the backbone to learn features that remain useful under changes in appearance.

However, augmentations must be compatible with the pretext task. For example, randomly rotating an image after assigning a rotation-prediction label could invalidate that label.

## 5. What are the data loaders and collate functions doing?

Question: Why are custom `collate_fn` functions needed?

A data loader collects individual dataset samples into batches. The custom collate functions also apply image transformations and move the resulting tensors to the selected device.

The notebook defines:

| Function | Input | Output |
| --- | --- | --- |
| `collate_labelled` | Labelled images and targets | Image tensor and integer-label tensor |
| `collate_labelled_aug` | Labelled images and targets | Augmented image tensor and integer-label tensor |
| `collate_unlabelled` | Unlabelled images | Image tensor only |
| `collate_unlabelled_aug` | Unlabelled images | Augmented image tensor only |

For example:

python

```
target_train_loader = DataLoader(
    target_data_train,
    batch_size=batch_size,
    collate_fn=collate_labelled,
    shuffle=True,
)
```

The batch size is 128 for CIFAR and 32 for STL-10.

The notebook also chooses a CUDA GPU when available and otherwise uses the CPU.

Question: Why do we need a separate pretext data loader?

The existing `source_unsup_loader` returns batches of unlabelled images. That is sufficient for a pretext task that generates targets inside the training loop, but it does not automatically create the required labels.

Depending on your chosen task, you must either:

- Generate the pretext targets inside a custom collate function.
- Generate the targets during training.
- Create a new dataset that explicitly stores the transformed images and their targets.

The loader must return exactly what your pretext training loop expects.

## 6. What are the backbone and classification head?

Question: Why does the notebook separate the model into two parts?

The notebook defines a feature-extracting backbone and a task-specific classification head. This modular design makes it possible to reuse the backbone for a different task without reusing the original output layer.

Input image

RGB image: 3 × H × W

`ConvBackbone`

Convolutional layers → pooling → fully connected layers

Output: 64 features

Pretext head

Predict the rotation, crop, or other pretext target

Target head

Predict the five target classes

The backbone is shared; the task-specific head changes.

Question: What does `ConvBackbone` do?

The backbone contains:

- Two convolutional layers with 32 and 64 output channels, each followed by ReLU and max pooling.
- Two additional convolutional layers with 128 and 256 output channels.
- Adaptive average pooling to reduce the spatial dimensions to $1 \times 1$.
- Fully connected layers of sizes 256 and 128, with dropout.
- A final feature projection from 128 dimensions to `num_features`.

The notebook sets:

python

```
backbone_output_size = 64
```

Therefore, `ConvBackbone` returns a 64-dimensional feature vector for each image.

These features are the representations that you hope will transfer from the pretext task to the target classification task.

Question: What does `ClassifierHead` do?

The target classification head is defined as:

python

```
class ClassifierHead(nn.Module):
    def __init__(self, input_size, projection_size, num_classes):
        super().__init__()
        self.projection = nn.Linear(input_size, projection_size)
        self.classifier = nn.Linear(projection_size, num_classes)

    def forward(self, x):
        x = self.projection(x)
        x = self.classifier(F.relu(x))
        return x
```

It maps the backbone's 64 features through a 32-dimensional projection to the number of target classes.

If the target contains five classes, its output shape is:

$$
(B,64)\rightarrow(B,32)\rightarrow(B,5)
$$

where $B$ is the batch size.

Question: Why can't you keep the pretext head for fine-tuning?

Because the pretext and downstream tasks have different outputs.

For example, a rotation-prediction head has four output classes, corresponding to rotations of 0°, 90°, 180°, and 270°. The target head instead has five outputs for the selected CIFAR-100 subset.

The correct procedure is to retain the trained backbone and create a new target classification head.

## 7. How does the supervised baseline work?

Question: Why must you train a baseline before self-supervised learning?

The baseline establishes how accurately the model can perform when it learns only from the small labelled target dataset.

Without this comparison, you cannot determine whether self-supervised pre-training provides any benefit.

The notebook initializes a fresh backbone and classification head and trains them together.

The principal hyperparameters are:

| Hyperparameter | Notebook value |
| --- | --- |
| Fine-tuning/baseline epochs | 30 |
| Learning rate | 0.001 |
| Weight decay (`ft_l2_reg`) | 0.1 |
| Batch size for CIFAR | 128 |
| Validation fraction | 10% |

Question: How does `train_supervised()` train the model?

For each training batch, it:

1. Retrieves the images and their correct class labels.
2. Passes the images through the backbone and classification head.
3. Calculates cross-entropy loss.
4. Backpropagates the loss.
5. Updates the model parameters using AdamW.
6. Records the batch loss and accuracy.

At the end of every epoch, `evaluate_model()` measures validation loss and accuracy with gradients disabled.

The best validation accuracy is stored in:

python

```
baseline_acc = baseline_metrics.best_val_acc
```

Question: Why might the baseline overfit?

The labelled target dataset is relatively small compared with the model's capacity. The model can start memorizing the training examples without learning features that generalize well to unseen images.

Signs of overfitting include training accuracy continuing to rise while validation accuracy stops improving or validation loss increases.

The baseline is therefore an essential benchmark, not necessarily a model with optimal performance.

Question: What does `TrainingMetrics` measure?

The helper class records:

- Training loss and accuracy for every batch.
- Validation loss and accuracy for every epoch.
- The best validation accuracy and lowest validation loss.
- Optional unsupervised losses and reference curves for plotting.

The plots help you compare the baseline and fine-tuned model's learning behaviour, not just their final scores.

## 8. Task 1 — What is a pretext task, and how should you design one?

Question: What exactly must you implement?

This is the central assignment task. You must invent a learning problem that uses unlabelled source images to generate training targets automatically.

A good pretext task should be learnable, but sufficiently challenging that solving it encourages the backbone to discover features that are useful for the downstream task.

The notebook suggests several possible approaches.

Representation Learning Through Self-Prediction Task Optimization - Thalles' blog

Option A — Rotation prediction

Apply one of four rotations to each image and predict which rotation was applied.

Recommended for a first implementation

It is relatively simple and can use ordinary classification loss.

Efficient Image Segmentation Using PyTorch: Part 1 | by Dhruv Matani | TDS Archive | Medium

Option B — Predict augmentation strength

Apply a controlled colour transformation and predict its intensity.

This is a regression task, so you would normally use a regression loss such as mean squared error.

Self-Supervised Learning

Option C — Jigsaw puzzle

Split an image into patches, shuffle them, and predict their original arrangement.

This can encourage spatial reasoning but requires more complex data preparation.

A Survey on Contrastive Self-Supervised Learning

Option D — Contrastive learning

Learn to distinguish two views of the same image from views of different images.

This can produce useful representations, but requires careful pair construction and an appropriate contrastive loss.

Question: Why is rotation prediction a good starting point?

It generates labels automatically and fits the existing supervised training loop.

For every source image $x$, randomly select a rotation $r$ from:

$$
r\in\{0,1,2,3\}
$$

The transformed image is $x_r$, and the target label is $r$.

The pretext model learns:

$$
f(x_r)\rightarrow \hat r
$$

where $\hat r$ is the predicted rotation class.

The loss is ordinary cross-entropy:

$$
L_{\text{pretext}}
=-\frac{1}{B}\sum_{i=1}^{B}
\log p_\theta(r_i\mid x_{r_i})
$$

Question: Why shouldn't the pretext task be too easy?

A task that can be solved using superficial visual cues may not encourage learning useful semantic representations.

For example, simply reconstructing a slightly corrupted image might be possible using low-level image features, without learning much about object structure.

However, very difficult tasks can also fail to converge. You need a balance between learning difficulty and useful representation learning.

Also, a high pretext accuracy does not automatically mean that the learned representations are good for the target task. Downstream validation accuracy is the final test.

## 9. How do you implement Task 1 using rotation prediction?

The following is a practical implementation that fits the notebook's existing `ConvBackbone`, `ClassifierHead`, `DataLoader`, and `train_supervised()` functions.

### Question: What transformations and labels should you generate?

First, define an augmentation pipeline that does not itself randomly rotate the images. The rotation applied afterward determines the target label.

python

```
from torchvision import transforms
from torch.utils.data import Subset

pretext_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        size=img_size,
        scale=(0.8, 1.0)
    ),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3
    ),
    transforms.ToTensor(),
    normalise,
])
```

This applies a random crop and colour variation before converting each image into a normalized tensor.

Now create a collate function that rotates each image and generates its label.

python

```
def collate_rotation(batch):
    images = [
        item[0] if isinstance(item, tuple) else item
        for item in batch
    ]

    rotated_images = []
    rotation_labels = []

    for img in images:
        x = pretext_transform(img)

        # Select 0, 90, 180, or 270 degrees
        r = torch.randint(0, 4, (1,)).item()

        # Rotate the tensor by r quarter-turns
        x_rotated = torch.rot90(x, r, dims=(1, 2))

        rotated_images.append(x_rotated)
        rotation_labels.append(r)

    x_batch = torch.stack(rotated_images).to(device)
    y_batch = torch.tensor(
        rotation_labels, dtype=torch.long
    ).to(device)

    return x_batch, y_batch
```

The four labels represent:

| Label | Rotation |
| --- | --- |
| 0 | 0° |
| 1 | 90° counterclockwise |
| 2 | 180° |
| 3 | 270° counterclockwise |

The precise clockwise or counterclockwise convention is not important as long as the transformation and its label agree.

Question: How do you create the pretext training and validation loaders?

It is good practice to reserve a portion of the source images for pretext validation. This lets you monitor whether the model learns the rotation task on images it did not train on.

python

```
from torch.utils.data import DataLoader, Subset

n_source = len(source_data)
n_pretext_val = int(0.1 * n_source)

generator = torch.Generator().manual_seed(0)
indices = torch.randperm(
    n_source, generator=generator
).tolist()

val_indices = indices[:n_pretext_val]
train_indices = indices[n_pretext_val:]

pretext_train_data = Subset(source_data, train_indices)
pretext_val_data = Subset(source_data, val_indices)

pretext_train_loader = DataLoader(
    pretext_train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_rotation
)

pretext_val_loader = DataLoader(
    pretext_val_data,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_rotation
)
```

This splits the source images into training and validation sets. The rotation labels are generated on the fly, so no manual annotations are needed.

One limitation is that the validation rotations and crops are randomly generated each time. This makes the validation metrics somewhat noisy. For a more rigorous experiment, use a fixed transformation and fixed rotation labels for the pretext validation set.

### Question: How do you construct and train the pretext model?

The pretext head needs four output classes rather than the five classes of the target task.

python

```
pretext_backbone = initialise_backbone()

pretext_head = ClassifierHead(
    input_size=backbone_output_size,
    projection_size=32,
    num_classes=4
)

pretext_model = nn.Sequential(
    pretext_backbone,
    pretext_head
)
```

You can now reuse the existing training loop because the pretext task returns the same kind of batch as supervised classification: an image tensor and an integer target tensor.

python

```
pretext_epochs = 30
pretext_lr = 1e-3
pretext_l2_reg = 1e-4

pretext_metrics = train_supervised(
    model=pretext_model,
    train_loader=pretext_train_loader,
    val_loader=pretext_val_loader,
    num_epochs=pretext_epochs,
    lr=pretext_lr,
    l2_reg=pretext_l2_reg,
    title="Self-supervised rotation prediction",
    dynamic_plot=False,
    seed=0
)
```

These are suggested starting hyperparameters, not fixed assignment requirements. You can increase the number of epochs if the pretext loss has not converged.

Question: What has the network learned after this training?

The network has learned to predict image rotations. In doing so, its backbone may have learned useful information about object boundaries, shapes, and spatial structure.

It has not been trained to predict the target classes directly. That distinction is the central idea of self-supervised pre-training.

Question: How can you inspect the pretext results?

Examine the training curves and check whether:

- The pretext training loss generally decreases.
- The pretext validation loss and accuracy show learning rather than immediate memorization.
- The validation metrics are reasonably stable.
- The task is neither trivially easy nor impossible to learn.

Rotation prediction can have dataset-specific shortcuts. For example, some images have strong orientation cues, while others are ambiguous. Consequently, a successful rotation classifier is not sufficient evidence that downstream performance will improve.

### Question: How do you save the pre-trained backbone?

Save the backbone separately from the pretext head.

python

```
pretrained_backbone_path = "pretrained_backbone.pth"

torch.save(
    pretext_backbone.state_dict(),
    pretrained_backbone_path
)
```

Saving only the backbone makes it clear which parameters should be transferred to the downstream task.

The checkpoint stores model weights, not the architecture itself. You must recreate a compatible `ConvBackbone` before loading the state dictionary.

## 10. Task 2 — How do you fine-tune the self-supervised model?

Question: What changes during fine-tuning?

The pretext model has learned to predict rotations from the source images. Fine-tuning adapts its backbone to classify the target dataset.

The procedure is:

1. Create a new target backbone.
2. Load the pre-trained backbone weights.
3. Create a new classification head with `num_target_classes` outputs.
4. Train the resulting model using the labelled target training set.
5. Evaluate it on the same target validation set used for the baseline.

The target labels are now used explicitly.

Question: How do you create the fine-tuning model?

python

```
# Recreate the backbone and load the learned features
ft_backbone = initialise_backbone()

ft_backbone.load_state_dict(
    torch.load(
        pretrained_backbone_path,
        map_location=device,
        weights_only=True
    )
)

# Create a fresh head for the downstream classification task
ft_target_head = ClassifierHead(
    input_size=backbone_output_size,
    projection_size=32,
    num_classes=num_target_classes
)

ft_model = nn.Sequential(
    ft_backbone,
    ft_target_head
)
```

This example uses `weights_only=True`, supported by modern PyTorch versions. If your environment uses an older version that does not support this argument, omit it for this plain state-dictionary checkpoint.

The backbone and target head are initialized separately. The backbone receives the pre-trained weights, while the target head starts with fresh weights.

Question: Should you freeze the backbone or update it?

The simplest approach is to fine-tune the entire model, updating both the backbone and classification head.

Alternatively, you can freeze the backbone initially and train only the new head:

python

```
for param in ft_backbone.parameters():
    param.requires_grad = False
```

You would then need to ensure that the optimizer includes only trainable parameters, and optionally unfreeze the backbone for a later fine-tuning phase.

For this assignment, full fine-tuning is a reasonable starting point because the source and target image distributions are related but not identical.

Question: How do you run fine-tuning using the notebook's existing function?

The following fills in the unfinished `train_supervised(...)` call.

python

```
print(
    f"\nFine-tuning self-supervised model "
    f"on {target_data.name}:"
)

ft_metrics = train_supervised(
    model=ft_model,
    train_loader=target_train_loader,
    val_loader=target_val_loader,
    num_epochs=ft_num_epochs,
    lr=ft_lr,
    l2_reg=ft_l2_reg,
    title=f"Post-SSL fine-tuning ({target_data.name})",
    baseline_metrics=baseline_metrics,
    baseline_accs={
        "supervised baseline": baseline_metrics.best_val_acc
    },
    seed=0
)

ft_acc = ft_metrics.best_val_acc

print(
    f"Post-SSL fine-tuning validation accuracy: "
    f"{ft_acc:.1%}"
)
```

For a fair comparison, this uses the same target training data, validation data, number of epochs, learning rate, and weight decay as the baseline.

The assignment does allow experimenting with different fine-tuning hyperparameters, such as a lower learning rate. If you change them, document the changes and consider whether the comparison remains fair.

## 11. How do you evaluate whether self-supervised learning worked?

Question: Which results should you compare?

The most important comparison is between the best validation accuracy of the supervised baseline and the best validation accuracy of the fine-tuned model.

python

```
baseline_acc = baseline_metrics.best_val_acc
ft_acc = ft_metrics.best_val_acc

improvement = ft_acc - baseline_acc

print(f"Baseline accuracy: {baseline_acc:.2%}")
print(f"Fine-tuned accuracy: {ft_acc:.2%}")
print(f"Absolute improvement: {improvement * 100:.2f} percentage points")
```

For example, if the baseline achieves 40% and the fine-tuned model achieves 48%, the improvement is 8 percentage points, equivalent to a 20% relative increase in accuracy.

These are illustrative values, not results from your notebook.

Question: How should you interpret the outcomes?

| Result | Interpretation |
| --- | --- |
| Fine-tuned accuracy is higher than baseline | The pre-training procedure provided a useful benefit |
| Fine-tuned accuracy is approximately equal to baseline | The pretext task may not have learned transferable features, or the training setup may need improvement |
| Fine-tuned accuracy is lower than baseline | Pre-training may have introduced unhelpful representations or the fine-tuning procedure may need adjustment |
| Pretext accuracy is high but downstream accuracy does not improve | Solving the pretext task did not necessarily teach features useful for the target task |

Remember that the pretext accuracy and downstream accuracy measure different tasks. Your final success criterion is downstream validation performance.

Question: What should the plots show?

The notebook's `TrainingMetrics.plot()` function can display training and validation loss and accuracy.

For the assessment, include plots that make it possible to compare:

- The baseline model's validation accuracy over its training epochs.
- The fine-tuned model's validation accuracy over its training epochs.
- The best validation accuracy achieved by each model.

You can also show the pretext training curves to demonstrate that the self-supervised stage learned something.

The baseline and fine-tuned models can have different numbers of training steps on the horizontal axis, so identify the curves clearly rather than relying only on their positions.

## 12. What implementation issues should you watch out for?

Question: Which parts of the supplied notebook are incomplete or need attention?

There are several important details to check before running the practical.

- Choose a target category. The line `TARGET_CATEGORY =` is intentionally incomplete and must be assigned a valid key, such as `"vehicles"`.
- Complete both task placeholders. The pretext-training section and fine-tuning section contain `...`, which must be replaced with your implementation.
- Fix the undefined `img_dim` variable in the example augmentation code. The notebook defines `img_size`, not `img_dim`. Replace `size=img_dim` with `size=img_size` if you use that example.
- Keep the pretext labels consistent with the transformations. If you apply a random transformation that changes the property being predicted, the assigned label might no longer be correct. For rotation prediction, assign the rotation label after applying any other augmentations.
- Use the right output dimensions. Rotation prediction requires four classes, while the target head must have `num_target_classes` outputs.
- Transfer only the backbone weights. The pretext head is specific to the pretext task and should not replace the downstream classification head.
- Monitor overfitting. A small labelled target set can cause the baseline and fine-tuned models to overfit. Compare validation curves, and consider regularization or carefully selected augmentation.
- Keep the comparison fair. Use the same target training and validation splits for both models. If you use augmentation during fine-tuning, consider using the same augmentation for baseline training.

Question: Is the supplied rotation implementation the only acceptable solution?

No. Rotation prediction is a suggested, relatively straightforward implementation. The assignment explicitly allows other pretext tasks.

If you choose contrastive learning, patch prediction, or an image-to-image task, the data pipeline, model head, and possibly the loss function will need to change accordingly.

## 13. What should you write in the assessment?

The notebook asks for three main things: an explanation of the pretext task, plots comparing the results, and a short discussion of the implementation and what you learned.

Writing

# Practical 3: Self-Supervised Learning — Results and Discussion

## 1\. Pretext task and motivation

I implemented **\[name of pretext task\]** using the unlabelled source dataset. The objective was to learn visual representations that could be reused for the downstream classification task.

The pretext task works by **\[explain how the images are transformed and how the training targets are generated automatically\]**. I implemented it using **\[describe the dataset or collate function, model backbone, task-specific head, and loss function\]**.

I chose this task because **\[explain why solving it might encourage the model to learn useful visual features\]**.

## 2\. Results

The supervised baseline achieved a best validation accuracy of **\[X%\]**. After self-supervised pre-training and supervised fine-tuning, the model achieved a best validation accuracy of **\[Y%\]**.

The absolute improvement was **\[Y − X percentage points\]**.

The training curves show **\[describe the observed training and validation loss and accuracy trends\]**. These results suggest that self-supervised pre-training **\[improved / did not significantly change / reduced\]** downstream performance.

## 3\. Discussion

One implementation issue I encountered was **\[describe an actual issue\]**. I addressed it by **\[explain the solution\]**.

The practical demonstrated that unlabelled images can provide a useful learning signal when the pretext task encourages the extraction of transferable visual features. However, the effectiveness of this approach depends on the difficulty of the pretext task, the quality of the learned representations, and the fine-tuning procedure.

From this experiment, I learned that **\[describe an insight supported by your results\]**.

The main limitation of my experiment was **\[describe any limitation, such as the small target dataset, computational constraints, or the choice of pretext task\]**.

Fill in the template using your actual experimental results. Do not report a performance improvement unless the measured validation accuracy supports it.

## 14. What are the tasks you must complete?

Use this checklist to track the assignment.

## Assignment checklist

0 of 6 completed

[ ]Prepare the datasetsChoose a target category or STL-10, inspect the images, and create the target train/validation split.

[ ]Train the supervised baselineRun train\_supervised() and record baseline\_metrics.best\_val\_acc.

[ ]Design and implement the pretext taskGenerate pretext targets from unlabelled images and create the required data pipeline.

[ ]Pre-train the backboneTrain the pretext model, inspect its learning curves, and save the backbone weights.

[ ]Fine-tune on the target taskLoad the pre-trained backbone, attach a fresh target head, and train on the labelled target data.

[ ]Compare and report resultsPlot the validation curves, calculate the accuracy difference, and write the assessment discussion.

## 15\. What is the key idea to remember?

The essential idea is that you can learn useful features from unlabelled images by designing a task whose targets can be generated automatically.

In this practical, you first establish a supervised baseline, then pre-train a backbone on your chosen pretext task, and finally transfer that backbone to a labelled downstream classification problem.

Your job is not simply to make the pretext model accurate. It is to design a pretext task that encourages useful representations and then demonstrate, through a fair validation comparison, whether those representations improve downstream performance.
