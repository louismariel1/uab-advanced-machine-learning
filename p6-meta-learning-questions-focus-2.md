# Advanced Machine Learning – Practical 6: Meta-Learning

Complete notebook explanation – Chunk 1 and Chunk 2

## Introduction: What is this practical about?

This practical explores meta-learning, an approach to machine learning in which a model learns how to adapt to new tasks using only a small number of labelled examples.

The central idea is different from conventional supervised learning.

In ordinary machine learning, we train a model to recognise a fixed set of classes. In meta-learning, we train a model across many different classification tasks so that it can learn new classification problems quickly.

The notebook uses the CIFAR-100 image dataset and investigates three approaches:

1. Baseline: Conventional supervised learning followed by few-shot fine-tuning.
2. Reptile: Gradient-based meta-learning that learns a useful model initialisation.
3. ProtoNet (Prototypical Networks): Metric-based meta-learning that learns an embedding space in which new classes can be recognised using class prototypes.

The ultimate objective is to determine which approach generalises best to previously unseen few-shot classification tasks.

### Notebook structure

| Part | Subject | Purpose |
| --- | --- | --- |
| 1 | Introduction to meta-learning | Understand the problem and learning setting |
| 2 | Environment and helper functions | Prepare PyTorch and visualisation utilities |
| 3 | CIFAR-100 preparation | Separate meta-training and meta-testing classes |
| 4 | Few-shot task generation | Create support and query datasets |
| 5 | Neural network architecture | Implement the CNN backbone and classifier |
| 6 | Baseline training | Train a conventional supervised model |
| 7 | Baseline meta-evaluation | Evaluate transfer to unseen few-shot tasks |
| 8 | Reptile | Implement gradient-based meta-learning |
| 9 | ProtoNet Task 1 | Implement prototype-based classification |
| 10 | ProtoNet Task 2 | Implement episodic meta-training |
| 11 | Final evaluation | Compare the approaches on unseen tasks |

# Part 1 – Understanding the big picture

## Question 1. What problem is this practical trying to solve?

Imagine that you want to build an image classifier.

Normally, you would collect thousands of labelled images and train a neural network to distinguish between classes such as:

- Cats
- Dogs
- Cars
- Birds
- Flowers

The problem is that conventional supervised learning generally requires many labelled examples of each class.

But suppose you encounter a completely new classification problem:

- Distinguish between otters and crocodiles.
- You have six labelled images of otters.
- You have six labelled images of crocodiles.

Can your model learn to distinguish them?

This is the problem of few-shot learning.

The goal is to develop a model that can learn new classification tasks from very few examples.

This leads to the central idea of meta-learning:

> Instead of learning a particular task, the model learns how to learn new tasks.

## Question 2. What is meta-learning?

Meta-learning means learning across multiple learning tasks.

Let's distinguish two approaches.

Ordinary machine learning

1. Give the model a dataset.
2. Train the model to recognise the dataset's classes.
3. Evaluate it on new images from those classes.

Meta-learning

1. Give the model many small classification tasks.
2. Train it to become good at adapting to those tasks.
3. Evaluate it on entirely new tasks involving previously unseen classes.

The difference is important.

| Ordinary learning | Meta-learning |
| --- | --- |
| Learns specific classes | Learns across tasks |
| Training examples are individual images | Training episodes contain tasks |
| Optimises classification on training classes | Optimises generalisation to new tasks |
| Usually needs substantial labelled data | Designed for very limited labelled data |

A useful analogy:

- Ordinary learning teaches someone to solve a particular mathematics problem.
- Meta-learning teaches someone a strategy for solving unfamiliar mathematics problems quickly.

## Question 3. What are the three important terms: task, support set and query set?

These concepts are fundamental to understanding the entire notebook.

A task is a small classification problem.

For example, imagine a task containing three classes:

- Butterflies
- Mice
- Couches

The task is a 3-way classification problem.

The notebook divides every task into two datasets.

### Support set

The support set contains the labelled examples provided to the model for learning the current task.

The notebook defines:

python

```python
SUPPORT_SIZE = 6
```

This means six examples per class.

For a 3-way task:

$$
3 \times 6 = 18
$$

The model receives 18 labelled support images.

### Query set

The query set contains separate images used to test whether the model has learned the task.

The notebook defines:

python

```python
QUERY_SIZE = 6
```

For a 3-way task:

$$
3 \times 6 = 18
$$

The model must classify these 18 query images.

The key distinction:

| Support set | Query set |
| --- | --- |
| Used for task learning | Used for task evaluation |
| Contains labelled examples | Contains labelled examples for calculating evaluation loss and accuracy |
| Helps the model understand the new classes | Tests its ability to classify new examples |

The query labels should not be used to train the model directly. They provide the targets against which predictions are compared.

In ProtoNet, query labels are used to calculate the meta-training loss, allowing gradients to improve the shared feature extractor. However, they are not used to construct the class prototypes.

## Question 4. What does N-way classification mean?

N-way classification refers to the number of different classes in a task.

The notebook uses:

python

```python
N_choices = [3, 4, 5]
```

Therefore, each randomly generated task contains either:

- 3 classes (3-way classification).
- 4 classes (4-way classification).
- 5 classes (5-way classification).

The number of examples per class is fixed at six.

| Task | Support images | Query images |
| --- | --- | --- |
| 3-way, 6-shot | 18 | 18 |
| 4-way, 6-shot | 24 | 24 |
| 5-way, 6-shot | 30 | 30 |

Strictly speaking, this is a variable-way, six-shot learning setup.

## Question 5. What is an episode?

An episode is one complete few-shot learning experiment.

It involves:

1. Selecting a subset of classes.
2. Creating a support set.
3. Creating a query set.
4. Learning from the support set, or constructing prototypes from it.
5. Evaluating predictions on the query set.

For example:

Episode 1

- Classes: butterfly, mouse, couch.
- Support set: 18 images.
- Query set: 18 images.

Episode 2

- Classes: flower, bicycle, television.
- Support set: 18 images.
- Query set: 18 images.

Each episode represents a different classification problem.

This is important because meta-learning trains across episodes rather than simply processing random batches from one fixed classification dataset.

# Part 2 – Environment and helper functions

## Question 6. Why do we need PyTorch and the other imported libraries?

The notebook begins by installing and importing the necessary libraries.

python

```python
!pip install torchinfo

import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

from IPython.display import display, clear_output
from torchinfo import summary
```

Their purposes are:

| Library | Purpose |
| --- | --- |
| PyTorch | Building and training neural networks |
| `torch.nn` | Neural network layers and loss functions |
| `torch.nn.functional` | Operations such as normalisation and cross-entropy |
| Torchvision | CIFAR-100 and image transformations |
| NumPy | Random sampling and numerical operations |
| Matplotlib | Plotting images, metrics and accuracy |
| Pandas | Smoothing training curves |
| PIL | Image processing |
| tqdm | Progress bars |
| os | File and directory management |
| torchinfo | Displaying neural network architectures |

## Question 7. How does the notebook select the computing device?

python

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 256
```

This checks whether a CUDA-compatible GPU is available.

- If available, computations run on the GPU.
- Otherwise, computations run on the CPU.

The batch size is 256.

However, in the few-shot tasks, the support and query datasets contain at most 30 examples each. Consequently, each task can be processed in a single batch.

## Question 8. What is the purpose of the TrainingMetrics class?

The notebook defines a custom class:

python

```python
class TrainingMetrics:
```

Its purpose is to record and visualise training performance.

It tracks four important quantities:

python

```python
self.train_loss = []
self.val_loss = []
self.train_acc = []
self.val_acc = []
```

These represent:

- Training loss.
- Validation loss.
- Training accuracy.
- Validation accuracy.

It also tracks:

python

```python
self.step = 0
self.epoch = 0
self.epoch_steps = [0]
```

These variables help distinguish training iterations from completed epochs.

### What do the logging methods do?

`log_train()`

python

```python
def log_train(self, loss: float, acc: float):
    self.train_loss.append(loss)
    self.train_acc.append(acc)
    self.step += 1
```

Records the training loss and accuracy at every batch update.

`log_val()`

python

```python
def log_val(self, loss: float, acc: float):
    self.val_loss.append(loss)
    self.val_acc.append(acc)
```

Records validation performance at the end of an epoch.

`epoch_loss` and `epoch_acc`

These properties calculate average training performance over the previous epoch and retrieve the latest validation performance.

`best_val_acc`

python

```python
@property
def best_val_acc(self):
    return np.max(self.val_acc)
```

Returns the highest validation accuracy recorded so far.

### What does the plot() method do?

It creates two plots:

1. Loss against training progress.
2. Accuracy against training progress.

It can also display:

- Baseline accuracy.
- Previous training curves.
- Meta-training episode boundaries.
- Smoothed training curves.

This is useful because meta-learning experiments involve many episodes, and we need to monitor whether the model is improving.

An important detail is that when `meta=True`, the class changes the plot labels to distinguish meta-training query performance from meta-test query performance.

## Question 9. Why does the notebook contain several dataset inspection functions?

The notebook defines four important helper functions.

### `inspect_dataset()`

Displays randomly selected images from a dataset together with their class labels.

It helps verify that the dataset has been loaded and labelled correctly.

### `inspect_batch()`

Displays a batch of images and optionally shows:

- Ground-truth labels.
- Model predictions.
- Whether predictions are correct.

A green tick indicates a correct prediction, while a red cross indicates an incorrect prediction.

### `count_labels()`

Counts how many examples belong to each class.

This helps verify the distribution of the dataset.

### `inspect_fewshot()`

Displays the images belonging to a few-shot task.

It can show all support or query examples, or one representative example per class.

These functions are not training algorithms. They are diagnostic tools that help us understand and verify the data.

## Question 10. How are classification accuracy metrics calculated?

The notebook defines two functions.

### Top-1 accuracy

python

```python
def top1_acc(pred, y):
    return (
        pred.argmax(axis=1) == y
    ).float().mean().item()
```

Top-1 accuracy checks whether the highest-scoring predicted class is correct.

Mathematically:

$$
Accuracy =
\frac{\text{Correct predictions}}
{\text{Total predictions}}
$$

### Top-5 accuracy

python

```python
def top5_acc(pred, y):
    top5_idxs = torch.sort(
        pred, dim=1, descending=True
    ).indices[:, :5]

    correct = (
        top5_idxs == y.unsqueeze(1)
    ).any(dim=1)

    return correct.float().mean().item()
```

Top-5 accuracy checks whether the correct class appears among the five highest-scoring predictions.

Why does the notebook use both?

- Top-5 is useful for the baseline's 80-class classification problem.
- Top-1 is used for few-shot tasks with only 3–5 classes.

# Part 3 – Understanding the dataset

## Question 11. Why does the notebook use CIFAR-100?

The CIFAR-100 dataset contains 100 image classes, with 600 images per class in its standard training dataset.

Images are small RGB images with dimensions:

$$
32 \times 32 \times 3
$$

It is suitable because it provides many different classes from which we can construct many different classification tasks.

The notebook loads it using:

python

```python
cifar100 = datasets.CIFAR100(
    root=data_root,
    download=False
)
```

The notebook also assigns:

python

```python
cifar100.name = 'CIFAR-100'
cifar100.num_classes = len(cifar100.classes)
```

These are additional attributes used for identification and visualisation.

## Question 12. Why divide CIFAR-100 into CIFAR-80 and CIFAR-20?

This is one of the most important design decisions in the practical.

If the model sees all 100 classes during meta-training, it becomes difficult to demonstrate that it can generalise to completely unseen classes.

Therefore, the notebook separates the classes into:

# 80

Meta-training classes

Used to generate training tasks.

# 20

Meta-testing classes

Used to generate unseen evaluation tasks.

CIFAR-100 class split

80% meta-training20% meta-testing

The 20 meta-test classes belong to four reserved superclasses:

python

```python
meta_test_superclasses = (
    'electricals',
    'reptiles',
    'fruit and veg',
    'large buildings'
)
```

Each superclass contains five classes.

For example:

- Electricals: clock, keyboard, lamp, telephone, television.
- Reptiles: crocodile, dinosaur, lizard, snake, turtle.
- Fruit and vegetables: apple, mushroom, orange, pear, sweet pepper.
- Large buildings: bridge, castle, house, road, skyscraper.

The remaining 16 superclasses provide the 80 meta-training classes.

### Why is this separation important?

Suppose the model learns tasks involving bicycles, flowers, animals and fish.

During evaluation, it receives tasks involving keyboards, reptiles and fruit.

It has never encountered those classes during meta-training.

If it performs well, this provides evidence that it has learned transferable representations or adaptation strategies rather than simply memorising the original classes.

This is called class-disjoint evaluation.

### How does the code create these class groups?

python

```python
meta_train_superclasses = [
    cl for cl in superclasses.keys()
    if cl not in meta_test_superclasses
]
```

The notebook then loops through the superclass dictionary:

python

```python
all_classes = []
meta_train_classes = []
meta_test_classes = []

for superclass, classes in superclasses.items():

    all_classes.extend(classes)

    if superclass in meta_test_superclasses:
        meta_test_classes.extend(classes)

    if superclass in meta_train_superclasses:
        meta_train_classes.extend(classes)
```

The result is two separate class lists.

The important point is that the split is performed at the class level, not by randomly splitting individual images.

## Question 13. What is the purpose of downloading CIFAR-100 from a mirror?

The notebook defines:

python

```python
data_root = 'data2/'
```

It then defines:

python

```python
def download_from_mirror(gdown_uri, dataset_name):
```

This function:

1. Creates the data directory if necessary.
2. Checks whether the dataset archive already exists.
3. Downloads the archive if necessary.
4. Extracts the archive.
5. Avoids downloading and extracting the same dataset repeatedly.

The function is a data preparation utility. It does not affect the meta-learning algorithm itself.

# Part 4 – Few-shot task generation

## Question 14. How many examples are available in each task?

The notebook defines:

python

```python
SUPPORT_SIZE = 6
QUERY_SIZE = 6
```

Therefore, every class contributes:

- Six support examples.
- Six query examples.

The number of classes varies between three and five.

This gives a small classification dataset containing between 18 and 30 images in each set.

## Question 15. What is the purpose of the Task class?

The notebook defines:

python

```python
class Task:
```

This is a custom Python class representing one few-shot classification problem.

Its constructor receives:

python

```python
def __init__(
    self,
    support_set,
    query_set,
    name,
    is_meta_test=False
):
```

It stores:

python

```python
self.support = support_set
self.query = query_set
self.name = name
self.is_meta_test = is_meta_test
```

It also checks that the support and query datasets contain the same classes:

python

```python
assert set(self.support.classes) == \
       set(self.query.classes)
```

This is important because the query set must test the same classification problem as the support set.

Then it creates two DataLoaders:

python

```python
self.support_loader = DataLoader(
    support_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

self.query_loader = DataLoader(
    query_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)
```

These allow the model to retrieve support and query batches.

Because the tasks contain at most 30 examples in each set, a batch size of 256 ensures that the entire set fits into one batch.

The method:

python

```python
def inspect(self, summarise=False):
```

is used to visualise the task's images.

## Question 16. How does `create_task()` generate a random task?

This function is the heart of the notebook's data-generation process.

python

```python
def create_task(
    num_support=SUPPORT_SIZE,
    num_query=QUERY_SIZE,
    N_choices=[3, 4, 5],
    task_name=None,
    meta_test=False,
    within_superclass=False
):
```

Let's understand its execution step by step.

### Step 1: Choose the number of classes

python

```python
num_task_classes = np.random.choice(N_choices)
```

Randomly selects 3, 4 or 5 classes.

### Step 2: Select the appropriate class pool

python

```python
if meta_test:
    meta_classes = meta_test_classes
else:
    meta_classes = meta_train_classes
```

If `meta_test=True`, the function samples from the 20 reserved classes.

Otherwise, it samples from the 80 training classes.

### Step 3: Select random classes

python

```python
task_idxs = np.random.choice(
    range(len(meta_classes)),
    size=num_task_classes,
    replace=False
)
```

`replace=False` prevents selecting the same class twice in one task.

### Step 4: Select images from each class

python

```python
all_class_data_idxs = [
    idx
    for idx, label in enumerate(cifar100.targets)
    if label == class_num
]
```

This finds all CIFAR-100 examples belonging to a particular class.

Then:

python

```python
np.random.shuffle(all_class_data_idxs)
```

Randomises the image indices.

### Step 5: Divide images into support and query examples

python

```python
support_idxs = all_class_data_idxs[:num_support]
query_idxs = all_class_data_idxs[-num_query:]
```

The first six examples become support images.

The last six become query images.

Because the indices are shuffled first, these are random, non-overlapping selections.

### Step 6: Assign new task-specific labels

This is particularly important:

python

```python
support_labels.extend([c] * num_support)
query_labels.extend([c] * num_query)
```

The original CIFAR-100 labels are replaced by new labels specific to the current task.

For example:

| Original CIFAR class | New task label |
| --- | --- |
| Butterfly | 0 |
| Mouse | 1 |
| Couch | 2 |

The classifier therefore learns a new classification problem with outputs numbered from zero.

### Step 7: Convert images into tensors

python

```python
support_x = torch.stack([
    preprocess(img) for img in support_imgs
])

support_y = torch.tensor(
    support_labels
).long()
```

The same operation is applied to the query images.

Finally, the function returns a `Task` object containing both datasets.

## Question 17. What does image preprocessing do?

python

```python
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```

There are two transformations.

ToTensor()

Converts an image into a PyTorch tensor and scales pixel values from approximately 0–255 to 0–1.

Normalize()

Transforms the values using:

$$
x_{\text{normalised}}
=
\frac{x-0.5}{0.5}
$$

The resulting pixel range is approximately:

$$
[-1,1]
$$

This helps make the inputs more consistent for neural network training.

## Question 18. Why is data augmentation used?

The notebook defines:

python

```python
from torchvision.transforms import v2

augment = v2.Compose([
    v2.RandomCrop(
        32,
        padding=4,
        padding_mode='reflect'
    ),
    v2.RandomHorizontalFlip()
])
```

This creates variations of training images.

- Random cropping changes the image's spatial position.
- Horizontal flipping creates mirrored images.

The function:

python

```python
def augment_batch(x):
    return torch.stack([
        augment(img) for img in x
    ])
```

applies these transformations to a batch.

Why is this particularly useful?

Each task contains only six support examples per class. Without augmentation, the model may simply memorise those images.

Augmentation provides slightly different versions of the same examples, encouraging better generalisation.

# Part 5 – Understanding the neural network architecture

## Question 19. What is a backbone, and why do we need one?

Before understanding the training algorithms, we need to understand the neural network.

The notebook uses two components:

1. A backbone that extracts useful features from images.
2. A classifier head that uses those features to predict classes.

The backbone learns to represent an image numerically.

For example, an image of a butterfly might be represented by a vector containing information about:

- Shapes.
- Edges.
- Colours.
- Textures.
- More complex visual patterns.

The classifier then uses this representation to decide which class the image belongs to.

This separation is important because the backbone can be reused for different classification tasks.

## Question 20. How does the ConvBackbone architecture work?

The notebook implements a small CNN called Conv-4.

python

```python
conv_num_features = 128

class ConvBackbone(nn.Module):
```

It contains four convolutional layers.

Each layer produces 128 feature channels.

The first convolution is:

python

```python
self.conv1 = nn.Conv2d(
    in_channels,
    num_features,
    3,
    padding=1
)
```

The parameters mean:

- Input channels: 3 (RGB).
- Output channels: 128.
- Kernel size: 3 × 3.
- Padding: 1.

Each convolutional layer is followed by Group Normalisation and ReLU, with max pooling reducing the spatial dimensions.

### Understanding the architecture step by step

## Conv-4 architecture

Input image

# 32 × 32 × 3

RGB image

Convolution 1

## 16 × 16 × 128

Conv → GroupNorm → ReLU → MaxPool

Convolution 2

## 8 × 8 × 128

Conv → GroupNorm → ReLU → MaxPool

Convolution 3

## 4 × 4 × 128

Conv → GroupNorm → ReLU → MaxPool

Convolution 4

## 2 × 2 × 128

Conv → GroupNorm → MaxPool

Final embedding

# 512

Feature vector2 × 2 × 128 = 512

The code for the forward pass is:

python

```python
def forward(self, x):

    x = self.maxpool(
        self.relu(self.norm1(self.conv1(x)))
    )

    x = self.maxpool(
        self.relu(self.norm2(self.conv2(x)))
    )

    x = self.maxpool(
        self.relu(self.norm3(self.conv3(x)))
    )

    x = self.maxpool(
        self.norm4(self.conv4(x))
    )

    x = x.view(x.size(0), -1)

    return x
```

The final operation:

python

```python
x = x.view(x.size(0), -1)
```

flattens the feature maps into vectors.

Therefore:

$$
2 \times 2 \times 128 = 512
$$

Every input image becomes a 512-dimensional feature vector.

### What is GroupNorm?

python

```python
self.norm1 = nn.GroupNorm(
    num_groups=num_features//4,
    num_channels=num_features
)
```

With 128 channels:

$$
128/4 = 32 \text{ groups}
$$

Each group contains four channels.

GroupNorm normalises features within groups of channels. Unlike BatchNorm, it does not depend on batch-wide running statistics.

This is useful in few-shot learning, where batches can be very small.

### Why is this backbone useful for meta-learning?

It provides a feature extractor that can be adapted to different classification tasks.

The backbone is the part we want to make increasingly useful through meta-training.

## Question 21. What is a ClassifierHead?

The backbone generates a 512-dimensional vector.

But that vector alone does not directly tell us whether an image belongs to a particular class.

We therefore need a classifier.

python

```python
class ClassifierHead(nn.Module):
```

The main component is:

python

```python
self.fc = nn.Linear(
    input_size,
    num_classes
)
```

For a 3-way task:

$$
512 \rightarrow 3
$$

For an 80-way classification problem:

$$
512 \rightarrow 80
$$

The forward function is:

python

```python
def forward(self, x):

    x = self.relu(x)

    if self.normalise_features:
        x = F.normalize(x, dim=-1) * 10

    x = self.fc(x)

    return x
```

Let's understand its operations.

- ReLU introduces non-linearity and removes negative feature values.
- Optional feature normalisation scales the feature vectors to a consistent magnitude.
- The linear layer produces class logits.

### What are logits?

Logits are raw numerical scores for each class.

For example:

python

```python
tensor([[2.1, 0.5, 3.4]])
```

The model predicts the third class because it has the highest score.

Cross-entropy loss converts these scores into a classification loss during training.

### What is `zero_init=True`?

python

```python
if zero_init:
    nn.init.zeros_(self.fc.weight)
    nn.init.zeros_(self.fc.bias)
```

This initialises the classifier weights and biases to zero.

The intention is to give each new task classifier a neutral starting point.

Implementation observation: The class defines a dropout layer, but the forward method never calls `self.dropout(x)`. Therefore, the `dropout=True` option currently has no effect.

## Question 22. Why do we need separate task-specific classifier heads?

Imagine we have trained a backbone and now encounter a new task:

- Class 0: Crocodile.
- Class 1: Turtle.
- Class 2: Snake.

We attach a new classifier head with three output neurons.

For another task:

- Class 0: Keyboard.
- Class 1: Television.
- Class 2: Telephone.

We need a different classifier head, again with three output neurons.

The backbone can be reused, but the classifier head must correspond to the current task's classes.

This is why the notebook separates the backbone from the classifier.

# Part 6 – Understanding the baseline

## Question 23. Why do we need a baseline?

Before introducing meta-learning, we need something to compare it against.

Otherwise, how can we establish that meta-learning actually improves performance?

The notebook uses conventional supervised learning as its baseline.

The process is:

CIFAR-80

All 80 meta-training classes

Supervised CNN training

Learn an 80-class classification problem

Pre-trained backbone

Save the feature extractor weights

Few-shot evaluation

Adapt or apply the backbone to each CIFAR-20 task

Average meta-test query accuracy

Notice that the baseline is not evaluated on the original 80 classes alone.

It is evaluated on the same 15 few-shot meta-test tasks as the meta-learning algorithms.

That makes the comparison meaningful.

## Question 24. How is the baseline dataset constructed?

First
