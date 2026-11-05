# Advanced Machine Learning – Practical 6: Meta-Learning

Both notebook chunks receivedThe notebook is about learning how to learn: instead of training a model to recognise a fixed collection of image classes, we train a model to adapt quickly to new classification problems using only a few labelled examples.

There are three main approaches in this practical:

1. Baseline: conventional supervised learning followed by few-shot fine-tuning.
2. Reptile: gradient-based meta-learning, which learns a useful initialisation for new tasks.
3. ProtoNet: metric-based meta-learning, which learns an embedding space where new classes can be recognised by comparing their prototypes.

The ultimate objective is to determine which approach generalises best to previously unseen few-shot classification tasks.

# Part 1 – Understanding the big picture

## Question 1. What problem is this practical trying to solve?

Imagine that you want to build an image classifier.

Normally, you would collect thousands of labelled images and train a neural network to distinguish between classes such as:

- Cats
- Dogs
- Cars
- Birds
- Flowers

The problem is that conventional supervised learning generally requires many examples of each class.

But suppose you encounter a completely new classification problem:

- Distinguish between otters and crocodiles.
- You only have six labelled images of otters.
- You only have six labelled images of crocodiles.

Can your model learn to distinguish them?

This is the problem of few-shot learning.

A Multi-Layer Feature Fusion Method for Few-Shot Image Classification

A Comprehensive Survey of Few shot Learning: Evolution, Applications, Challenges, and Opportunities

Impact of base dataset design on few-shot image classification

The goal is to develop a model that can learn new classification tasks from very few examples.

This leads to the central idea of meta-learning:

Instead of learning a particular task, the model learns how to learn new tasks.

## Question 2. What is meta-learning?

Meta-learning means learning across multiple learning tasks.

Let's distinguish two approaches.

Ordinary machine learning

- Give the model a dataset.
- Train the model to recognise the dataset's classes.
- Evaluate it on new images from those classes.

Meta-learning

- Give the model many small classification tasks.
- Train it to become good at adapting to those tasks.
- Evaluate it on entirely new tasks involving previously unseen classes.

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

Chapter 03 | Sebastian Raschka, PhD

[sebastianraschka.com](<https://sebastianraschka.com/books/ml-q-and-ai-chapters/ch03/?utm_source=chatgpt.com>)

Support set

The support set contains the labelled examples provided to the model for learning the current task.

In this practical:

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

Query set

The query set contains separate images used to test whether the model has learned the task.

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
| Contains labelled examples | Contains labelled examples during training/evaluation |
| Helps the model understand the new classes | Tests its ability to classify new examples |

The query labels are not supposed to be used to train the model directly. They provide the target against which predictions are compared.

## Question 4. What does N-way classification mean?

N-way classification refers to the number of different classes in a task.

The notebook uses:

python

```python
N_choices = [3, 4, 5]
```

Therefore, each randomly generated task contains either:

- 3 classes (3-way classification)
- 4 classes (4-way classification)
- 5 classes (5-way classification)

The number of examples per class is fixed at six.

Consequently, the possible task sizes are:

| Task | Support images | Query images |
| --- | --- | --- |
| 3-way, 6-shot | 18 | 18 |
| 4-way, 6-shot | 24 | 24 |
| 5-way, 6-shot | 30 | 30 |

Strictly speaking, this is a variable-way, six-shot learning setup.

# Part 2 – Understanding the dataset

## Question 5. Why does the notebook use CIFAR-100?

Medium

The CIFAR-100 dataset contains 100 image classes, with 600 images per class in its standard training dataset.

Images are small RGB images with dimensions:

$$
32 \times 32 \times 3
$$

It is suitable because it provides many different classes from which we can construct many different classification tasks.

The notebook uses:

python

```python
cifar100 = datasets.CIFAR100(
    root=data_root,
    download=False
)
```

This loads the CIFAR-100 dataset.

The notebook also assigns:

python

```python
cifar100.name = 'CIFAR-100'
cifar100.num_classes = len(cifar100.classes)
```

These are additional attributes used for identification and visualisation.

## Question 6. Why divide CIFAR-100 into CIFAR-80 and CIFAR-20?

This is one of the most important design decisions in the practical.

If the model sees all 100 classes during meta-training, it becomes difficult to demonstrate that it can generalise to completely unseen classes.

Therefore, the notebook separates the classes into:

Meta-training

80 classes

Used to generate training tasks.

Meta-testing

20 classes

Used to generate unseen evaluation tasks.

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

Why is this separation important?

Suppose the model learns tasks involving bicycles, flowers, dogs and fish.

During evaluation, it receives tasks involving keyboards, reptiles and fruit.

It has never encountered those classes during meta-training.

If it performs well, this provides evidence that it has learned transferable representations or adaptation strategies rather than simply memorising the original classes.

This is called class-disjoint evaluation.

## Question 7. What is the purpose of the `Task` class?

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
```

python

```python
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

This is particularly useful for verifying that the randomly generated tasks contain the expected classes.

## Question 8. How does `create_task()` generate a random task?

This function is the heart of the notebook's data-generation process.

python

```python
def create_task(
    num_support=SUPPORT_SIZE,
    num_query=QUERY_SIZE,
    N_choices=[3,4,5],
    task_name=None,
    meta_test=False,
    within_superclass=False
):
```

Let's understand its execution step by step.

Step 1: Choose the number of classes.

python

```python
num_task_classes = np.random.choice(N_choices)
```

Randomly selects 3, 4 or 5 classes.

Step 2: Select the appropriate class pool.

python

```python
if meta_test:
    meta_classes = meta_test_classes
else:
    meta_classes = meta_train_classes
```

If `meta_test=True`, the function samples from the 20 reserved classes.

Otherwise, it samples from the 80 training classes.

Step 3: Select random classes.

python

```python
task_idxs = np.random.choice(
    range(len(meta_classes)),
    size=num_task_classes,
    replace=False
)
```

`replace=False` prevents selecting the same class twice in one task.

Step 4: Retrieve images from each selected class.

python

```python
all_class_data_idxs = [
    idx for idx, label in enumerate(cifar100.targets)
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

Step 5: Divide the images into support and query examples.

python

```python
support_idxs = all_class_data_idxs[:num_support]
query_idxs = all_class_data_idxs[-num_query:]
```

The first six examples become support images.

The last six become query images.

Because the indices are shuffled first, these are random, non-overlapping selections.

Step 6: Assign new task-specific labels.

This is particularly important:

python

```python
support_labels.extend([c] * num_support)
query_labels.extend([c] * num_query)
```

The original CIFAR-100 labels are replaced by new labels specific to the current task.

For example:

| Original CIFAR label | New task label |
| --- | --- |
| Butterfly | 0 |
| Mouse | 1 |
| Couch | 2 |

The classifier therefore learns a new classification problem with outputs numbered from zero.

Step 7: Convert images into tensors.

python

```python
support_x = torch.stack([
    preprocess(img) for img in support_imgs
])
```

The same operation is applied to the query images.

Finally, the function returns a `Task` object containing both datasets.

### Question 9. What does image preprocessing do?

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

1. `ToTensor()` converts an image into a PyTorch tensor and scales pixel values from approximately 0–255 to 0–1.
2. `Normalize()` transforms the values using:

$$
x_{\text{normalised}} =
\frac{x-0.5}{0.5}
$$

The resulting pixel range is approximately:

$$
[-1,1]
$$

This helps make the inputs more consistent for neural network training.

### Question 10. Why is data augmentation used?

The notebook defines:

python

```python
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

Important: The notebook applies augmentation during training, not during query-set evaluation.

# Part 3 – Understanding the neural network architecture

## Question 11. What is a backbone, and why do we need one?

Before understanding the training algorithms, we need to understand the neural network.

The notebook uses two components:

1. A backbone that extracts useful features from images.
2. A classifier head that uses those features to predict classes.

CNN 사전학습 모델을 이용한 이미지 분류

[yeong-jin-data-blog.tistory.com](<https://yeong-jin-data-blog.tistory.com/entry/%EC%82%AC%EC%A0%84%ED%95%99%EC%8A%B5-%EB%AA%A8%EB%8D%B8-%EC%82%AC%EC%9A%A9?utm_source=chatgpt.com>)

The backbone learns to represent an image numerically.

For example, an image of a butterfly might be represented by a vector containing information about:

- Shapes
- Edges
- Colours
- Textures
- More complex visual patterns

The classifier then uses this representation to decide which class the image belongs to.

This separation is important because the backbone can be reused for different classification tasks.

## Question 12. How does the `ConvBackbone` architecture work?

The notebook implements a small CNN called Conv-4.

python

```python
class ConvBackbone(nn.Module):
```

It contains four convolutional layers.

python

```python
self.conv1 = nn.Conv2d(
    in_channels, num_features, 3, padding=1
)
```

The parameters mean:

- Input channels: 3 (RGB).
- Output channels: 128.
- Kernel size: 3 × 3.
- Padding: 1.

The number of output channels is controlled by:

python

```python
conv_num_features = 128
```

Each convolutional layer is followed by Group Normalisation and ReLU, with max pooling reducing the spatial dimensions.

### Understanding the architecture step by step

## Conv-4 architecture

Input image# 32 × 32 × 3

RGB image

Convolution 1

# 16 × 16 × 128

Conv → GroupNorm → ReLU → MaxPool

Convolution 2

# 8 × 8 × 128

Conv → GroupNorm → ReLU → MaxPool

Convolution 3

# 4 × 4 × 128

Conv → GroupNorm → ReLU → MaxPool

Convolution 4

# 2 × 2 × 128

Conv → GroupNorm → MaxPool

Final embedding# 512

Feature vector

2 × 2 × 128 = 512

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

What is GroupNorm?

python

```python
self.norm1 = nn.GroupNorm(
    num_groups=num_features//4,
    num_channels=num_features
)
```

With 128 channels:

$$
128/4=32 \text{ groups}
$$

Each group contains four channels.

GroupNorm normalises features within groups of channels. Unlike BatchNorm, it does not depend on batch-wide running statistics.

This is useful in few-shot learning, where batches can be very small.

Why is this backbone useful for meta-learning?

It provides a feature extractor that can be adapted to different classification tasks.

The backbone is the part we want to make increasingly useful through meta-training.

## Question 13. What is a `ClassifierHead`?

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

Cross-entropy loss will convert these scores into a classification loss during training.

### What is `zero_init=True`?

python

```python
if zero_init:
    nn.init.zeros_(self.fc.weight)
    nn.init.zeros_(self.fc.bias)
```

This initialises the classifier weights and biases to zero.

The intention is to give each new task classifier a neutral starting point.

Small implementation detail: The class defines a dropout layer, but the forward method never calls `self.dropout(x)`. Therefore, the `dropout=True` option currently has no effect. This is worth noticing when inspecting the implementation.

## Question 14. Why do we need separate task-specific classifier heads?

Imagine we have trained a backbone and now encounter a new task:

- Class 0: Crocodile
- Class 1: Turtle
- Class 2: Snake

We attach a new classifier head with three output neurons.

For another task:

- Class 0: Keyboard
- Class 1: Television
- Class 2: Telephone

We need a different classifier head, again with three output neurons.

The backbone can be reused, but the classifier head must correspond to the current task's classes.

This is why the notebook separates the backbone from the classifier.

# Part 4 – Understanding the baseline

## Question 15. Why do we need a baseline?

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

Few-shot fine-tuning

Adapt to each CIFAR-20 task

Average meta-test query accuracy

Notice that the baseline is not evaluated on the original 80 classes alone.

It is evaluated on the same 15 few-shot meta-test tasks as the meta-learning algorithms.

That makes the comparison meaningful.

## Question 16. How is the baseline dataset constructed?

First, the notebook identifies the original CIFAR-100 labels belonging to the meta-training classes:

python

```python
baseline_metatrain_class_idxs = [
    cifar100.class_to_idx[cl_name]
    for cl_name in meta_train_classes
]
```

Then it selects all images belonging to those classes:

python

```python
baseline_metatrain_class_data_idxs = [
    i for i, label in enumerate(cifar100.targets)
    if label in baseline_metatrain_class_idxs
]
```

It converts the images into tensors and creates new labels numbered from 0 to 79.

python

```python
baseline_y_metatrain = torch.Tensor([
    baseline_metatrain_cl2idx[
        cifar100.classes[label]
    ]
    for img, label in baseline_metatrain_examples
]).long()
```

Finally:

python

```python
baseline_metatrain_data = TensorDataset(
    baseline_x_metatrain,
    baseline_y_metatrain
)
```

The dataset is divided into training and validation:

python

```python
val_frac = 0.4

baseline_train, baseline_val = \
    torch.utils.data.random_split(
        baseline_metatrain_data,
        [1-val_frac, val_frac]
    )
```

This produces a 60% training split and a 40% validation split.

## Question 17. How does `train_supervised()` train the baseline?

This is the conventional neural network training function.

It receives:

python

```python
def train_supervised(
    model,
    train_loader,
    val_loader,
    num_epochs,
    lr,
    l2_reg=0.0,
    ...
):
```

Its main responsibilities are:

1. Initialise an optimiser.
2. Loop through training epochs.
3. Perform forward propagation.
4. Calculate the loss.
5. Perform backpropagation.
6. Update the parameters.
7. Evaluate the model on validation data.
8. Record metrics and optionally plot them.

### What is AdamW?

python

```python
opt = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=l2_reg
)
```

AdamW is an optimisation algorithm that updates the network parameters using gradient information.

The learning rate controls the size of the parameter updates.

The weight decay provides regularisation to discourage excessively large weights.

The notebook uses:

python

```python
base_epochs = 10
base_lr = 1e-3
base_l2 = 1e-3
```

### How does one training step work?

The most important code is:

python

```python
opt.zero_grad()

pred = model(x)

batch_loss = loss_fn(pred, y)

batch_loss.backward()

opt.step()
```

Let's translate this.

| Code | Meaning |
| --- | --- |
| `opt.zero_grad()` | Clear previously accumulated gradients |
| `model(x)` | Forward pass |
| `loss_fn(pred, y)` | Calculate prediction error |
| `batch_loss.backward()` | Calculate gradients through backpropagation |
| `opt.step()` | Update model parameters |

This is the standard supervised learning process.

### What is CrossEntropyLoss?

python

```python
loss_fn = nn.CrossEntropyLoss()
```

Cross-entropy measures how well the model's predicted class scores match the correct labels.

Lower loss generally means better predictions.

Mathematically:

$$
L=-\log(p_{\text{correct}})
$$

where $p_{\text{correct}}$ is the predicted probability assigned to the correct class.

The notebook also uses:

python

```python
acc_fn = top5_acc
```

Why top-5 instead of top-1?

Because the baseline has 80 output classes.

Top-5 accuracy counts a prediction as correct if the true class appears among the five highest-scoring predictions.

The baseline's top-5 validation accuracy is useful for checking ordinary CIFAR-80 classification, but it is not the final metric of interest.

## Question 18. What is the purpose of `evaluate_model()`?

Training and evaluation are different.

The model should update its weights during training, but not during validation.

The notebook uses:

python

```python
model.eval()

with torch.no_grad():
```

These operations are important.

- `model.eval()` switches modules into evaluation mode.
- `torch.no_grad()` prevents gradient calculations, saving memory and computation.

The function calculates validation loss and accuracy and
