# Advanced Machine Learning Practical 3 — Comprehensive MCQ

 ## Answers Directly Below Each Question

 This updated version keeps the **150-question coverage** of your original bank, but places the **correct answer and a short explanation immediately after every question**. This makes it useful for active revision rather than requiring you to constantly jump between the questions and a separate answer key.

 I have also tightened a few explanations where the distinction is important—for example, **pretext accuracy vs downstream performance**, **logits vs probabilities**, and **what exactly is transferred during fine-tuning**. For the PyTorch classification points, `CrossEntropyLoss` is designed to operate on logits with class-index targets, with class-index targets represented as integer/long tensors.  PyTorch Documentation+1

---

 # Section A — Self-Supervised Learning Concepts

 ### 1\. What is the primary objective of the practical?

 A. Maximise training accuracy on CIFAR-10\
 B. Maximise validation accuracy on a small labelled target dataset using a large pre-training dataset\
 C. Minimise validation loss on an unlabelled dataset\
 D. Train a generative model on CIFAR-100

 **Answer: B — Maximise validation accuracy on the labelled target task.**

 **Why:** The practical investigates whether representation learning from unlabelled source data improves downstream target-task performance.

---

 ### 2\. What distinguishes the source dataset from the target dataset?

 A. The source dataset is small and labelled; the target is large and unlabelled\
 B. Both datasets are labelled and contain identical classes\
 C. The source dataset is large and unlabelled; the target is small and labelled\
 D. Both datasets are completely unlabelled

 **Answer: C — The source is large and unlabelled; the target is small and labelled.**

 **Why:** This is the central setup of the practical.

---

 ### 3\. What is the key idea behind self-supervised learning?

 A. Manually label every source image\
 B. Use the validation set as training data\
 C. Create useful training labels/signals automatically from unlabelled data\
 D. Remove the feature extractor before training

 **Answer: C — Create training signals automatically from the data itself.**

 **Why:** The labels for the pretext task are generated from the input rather than supplied by a human.

---

 ### 4\. The source data is primarily used for:

 A. Supervised classification\
 B. Self-supervised pre-training\
 C. Final validation only\
 D. Hyperparameter optimisation only

 **Answer: B — Self-supervised pre-training.**

 **Why:** The source labels are unavailable or deliberately ignored.

---

 ### 5\. How does this practical differ from pseudo-labeling?

 A. The source and target datasets are identical\
 B. The source and target represent different tasks/datasets and source labels are unavailable\
 C. No labelled data is used at all\
 D. The target dataset is larger than the source

 **Answer: B — Source labels are not being used as pseudo-labels.**

 **Why:** Instead, the practical constructs an artificial pretext task.

---

 ### 6\. Why can SSL transfer between datasets with different classes?

 A. The model memorises source labels\
 B. Different datasets can share useful visual/structural features\
 C. Target labels are copied from source labels\
 D. Output classes are forced to be identical

 **Answer: B — Visual representations can generalise across class boundaries.**

 **Why:** Edges, textures, shapes and object structures can be useful even when the exact classes differ.

---

 ### 7\. Which component is intended to be reused downstream?

 A. Source dataset\
 B. Pretext labels\
 C. Pretrained backbone/feature encoder\
 D. Validation loss

 **Answer: C — The pretrained backbone.**

---

 ### 8\. What happens to the pretext-task head during transfer?

 A. It must always be retained\
 B. It is replaced with a target-task-specific head\
 C. It becomes the validation set\
 D. It is deleted together with the backbone

 **Answer: B — It is replaced.**

 **Why:** The pretext head predicts the artificial task, whereas the new head predicts the target classes.

---

 ### 9\. What is the purpose of the supervised baseline?

 A. Provide an upper bound\
 B. Measure performance when learning only from the labelled target data\
 C. Train the pretext task\
 D. Replace validation

 **Answer: B — It provides the from-scratch reference performance.**

---

 ### 10\. If SSL works effectively, what outcome is hoped for?

 A. Lower target validation accuracy\
 B. Improved downstream performance compared with training from scratch\
 C. Perfect pretext accuracy\
 D. Zero training time

 **Answer: B — Improved downstream target performance.**

 **Key idea:** Downstream performance is the important experimental outcome, not simply pretext accuracy.

---

 # Section B — Dataset Design

 ### 11\. Which dataset is used as the target in the CIFAR configuration?

 A. CIFAR-10\
 B. CIFAR-100 subset\
 C. MNIST\
 D. ImageNet

 **Answer: B — CIFAR-100 subset.**

---

 ### 12\. Which dataset is used as the unlabelled source?

 A. CIFAR-10\
 B. CIFAR-100\
 C. STL-10 labelled portion\
 D. MNIST

 **Answer: A — CIFAR-10.**

---

 ### 13\. What happens to the CIFAR-10 labels?

 A. Converted into CIFAR-100 labels\
 B. Shuffled\
 C. Explicitly destroyed/ignored\
 D. Used as validation labels

 **Answer: C — They are deliberately ignored/destroyed.**

---

 ### 14\. How many classes does each CIFAR-100 superclass contain?

 A. 2\
 B. 5\
 C. 10\
 D. 20

 **Answer: B — 5.**

---

 ### 15\. Suppose `TARGET_CATEGORY = "vehicles"`. Which could be one of its subclasses?

 A. Dolphin\
 B. Bicycle\
 C. Tiger\
 D. Sunflower

 **Answer: B — Bicycle.**

---

 ### 16\. What does this expression do?

```
superclass_idxs = set(
    [cifar100.class_to_idx[n] for n in superclass_names]
)
```

 A. Converts images into tensors\
 B. Finds the original CIFAR-100 integer indices of selected classes\
 C. Creates a train/validation split\
 D. Counts images

 **Answer: B — It obtains the original integer indices for the selected classes.**

---

 ### 17\. What is the purpose of `subset_idxs`?

```
subset_idxs = [
    i for i, idx in enumerate(cifar100.targets)
    if idx in superclass_idxs
]
```

 A. Identifies images belonging to the selected classes\
 B. Randomly augments images\
 C. Creates the source dataset\
 D. Removes all labels

 **Answer: A — It identifies the examples belonging to the selected classes.**

---

 ### 18\. Why is `class_remap` needed?

 A. To change image resolution\
 B. To map original labels to consecutive labels `0...4`\
 C. To convert labels to strings\
 D. To normalise pixels

 **Answer: B — It creates clean consecutive class indices.**

---

 ### 19\. If selected original class indices are `[12, 37, 72, 81, 95]`, what does remapping conceptually produce?

 A. `[12, 37, 72, 81, 95]`\
 B. `[1, 2, 3, 4, 5]`\
 C. `[0, 1, 2, 3, 4]`\
 D. Random labels

 **Answer: C — `[0, 1, 2, 3, 4]`.**

---

 ### 20\. What does `target_data.num_classes` represent?

 A. Number of batches\
 B. Number of pixels\
 C. Number of target classes\
 D. Number of source images

 **Answer: C — Number of target classes.**

---

 ### 21\. What is special about STL-10?

 A. It contains no images\
 B. It has labelled and large unlabelled portions\
 C. It has one class\
 D. It is smaller and lower resolution

 **Answer: B — STL-10 provides both labelled and substantial unlabelled data.**

---

 ### 22\. What is the original resolution of STL-10 images?

 A. 28 × 28\
 B. 32 × 32\
 C. 64 × 64\
 D. 96 × 96

 **Answer: D — 96 × 96.**

---

 ### 23\. Approximately how many unlabelled images does STL-10 provide?

 A. 1,000\
 B. 10,000\
 C. 50,000\
 D. 100,000

 **Answer: D — Approximately 100,000.**

---

 ### 24\. Why might STL-10 take longer to train?

 A. It has no GPU support\
 B. It contains higher-resolution images and a large unlabelled dataset\
 C. It has one class\
 D. Labels must be entered manually

 **Answer: B — Higher resolution and more source images increase computational work.**

---

 # Section C — Custom Dataset Classes

 ### 25\. What does `UnlabelledDataset.__getitem__()` return?

 A. `(image, label)`\
 B. Only an image\
 C. Only a label\
 D. Model prediction

 **Answer: B — Only an image.**

---

 ### 26\. What does `LabelledDataset.__getitem__()` return?

 A. Only an image\
 B. Only a label\
 C. `(image, target)`\
 D. `(prediction, target)`

 **Answer: C — An image and its target.**

---

 ### 27\. Why does `UnlabelledDataset.__getitem__()` check whether the image is a PIL image?

 A. To convert images into labels\
 B. To ensure the image has an appropriate PIL representation when required\
 C. To resize every image\
 D. To normalise pixels

 **Answer: B — It ensures compatibility with PIL-based transformations.**

---

 ### 28\. What does `__len__()` return?

 A. Number of classes\
 B. Number of batches\
 C. Number of stored examples\
 D. Number of pixels

 **Answer: C — Number of examples in the dataset.**

---

 ### 29\. What does this assertion guarantee?

```
assert len(images) == len(targets)
```

 A. Every image has a corresponding target\
 B. Every class has equal examples\
 C. Images have equal dimensions\
 D. Train and validation are equal

 **Answer: A — The image and target arrays are aligned.**

---

 ### 30\. What does `classes` represent?

 A. Neural network layers\
 B. Optional human-readable class names\
 C. Optimisation parameters\
 D. Image dimensions

 **Answer: B — Human-readable class names.**

---

 ### 31\. What does `count_labels()` primarily do?

 A. Trains the model\
 B. Counts examples belonging to each class\
 C. Removes labels\
 D. Normalises class frequencies

 **Answer: B — It counts class occurrences.**

---

 ### 32\. What is the purpose of `seed` in `train_test_split()`?

 A. Change learning rate\
 B. Make the random split reproducible\
 C. Initialise the network architecture\
 D. Determine number of classes

 **Answer: B — It controls reproducibility.**

---

 ### 33\. If `test_frac=0.1` and there are 1,000 examples, how many are selected?

 A. 10\
 B. 50\
 C. 100\
 D. 900

 **Answer: C — 100.**

---

 ### 34\. What does `replace=False` mean?

```
np.random.choice(..., replace=False)
```

 A. Examples can be selected repeatedly\
 B. Examples cannot be selected more than once\
 C. Labels are replaced\
 D. Images are replaced with zeros

 **Answer: B — Sampling occurs without replacement.**

---

 # Section D — Data Splitting and DataLoaders

 ### 35\. What validation fraction is used?

 A. 0.01\
 B. 0.05\
 C. 0.10\
 D. 0.50

 **Answer: C — 10%.**

---

 ### 36\. Why is a validation set required?

 A. To train the model directly\
 B. To evaluate generalisation during training\
 C. To generate CIFAR-10 labels\
 D. To increase classes

 **Answer: B — It provides an independent measure of performance during training.**

---

 ### 37\. What does `shuffle=True` in the training DataLoader accomplish?

 A. Randomises training sample order between epochs\
 B. Deletes samples\
 C. Changes labels\
 D. Normalises images

 **Answer: A — It randomises the order in which training examples are presented.**

---

 ### 38\. Why does the validation loader use `shuffle=False`?

 A. Validation does not require random ordering\
 B. Validation images cannot be shuffled\
 C. It changes labels\
 D. It enables augmentation

 **Answer: A — Ordering is unnecessary for evaluation.**

---

 ### 39\. Why are custom `collate_fn` functions useful?

 A. Define network architecture\
 B. Control how individual examples are transformed and assembled into batches\
 C. Calculate accuracy only\
 D. Replace optimizer

 **Answer: B — They customise batch construction.**

---

 ### 40\. What does this line do?

```
image_tensor = torch.stack([transform(img) for img in images])
```

 A. Combines individually transformed images into a batch tensor\
 B. Converts labels to strings\
 C. Calculates loss\
 D. Performs backpropagation

 **Answer: A — Each image is transformed and the results are stacked into a batch.**

---

 ### 41\. What datatype is explicitly used for classification labels?

```
torch.tensor(labels, dtype=torch.long)
```

 A. `float32`\
 B. `int64` / `long`\
 C. `bool`\
 D. `uint8`

 **Answer: B — `torch.long` (`int64`).**

---

 ### 42\. Why is `torch.long` appropriate for `CrossEntropyLoss` targets?

 A. It represents integer class indices\
 B. It represents images\
 C. It performs augmentation\
 D. It represents probabilities

 **Answer: A — Class-index targets are expected as integer/long values.**  PyTorch Documentation+1

---

 ### 43\. What does `.to(device)` accomplish?

 A. Moves tensors to CPU only\
 B. Moves tensors to the selected computation device\
 C. Normalises tensors\
 D. Converts tensors to NumPy

 **Answer: B — It moves the tensor to the selected CPU/GPU device.**

---

 # Section E — Preprocessing and Augmentation

 ### 44\. Which device is selected here?

```
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

 A. Always CPU\
 B. Always GPU\
 C. GPU if CUDA is available, otherwise CPU\
 D. TPU

 **Answer: C — CUDA GPU when available, otherwise CPU.**

---

 ### 45\. What does `ToTensor()` generally do?

 A. Converts an image into a PyTorch tensor\
 B. Converts tensors into PIL images\
 C. Converts labels into strings\
 D. Computes gradients

 **Answer: A — It converts the image into tensor form.**

---

 ### 46\. What is the purpose of:

```
transforms.Normalize(mean=0.5, std=0.2)
```

 A. Standardise pixel values using the supplied mean and standard deviation\
 B. Resize images\
 C. Rotate images\
 D. Remove labels

 **Answer: A — It normalises the tensor values.**

---

 ### 47\. What does `RandomHorizontalFlip(p=0.5)` mean?

 A. Every image is flipped\
 B. No image is flipped\
 C. Each image has a 50% chance of being flipped\
 D. Exactly half the dataset is permanently flipped

 **Answer: C — The transform is independently applied with probability 0.5.**

---

 ### 48\. What does `RandomApply(..., p=0.5)` do?

 A. Applies the enclosed transformation with probability 0.5\
 B. Applies it twice\
 C. Applies it to half the pixels\
 D. Disables augmentation

 **Answer: A — The transform is randomly applied with the specified probability.**

---

 ### 49\. What is the purpose of `ColorJitter`?

 A. Modify brightness, contrast, saturation and/or hue\
 B. Change labels\
 C. Perform pooling\
 D. Calculate loss

 **Answer: A — It creates colour variations.**

---

 ### 50\. What does `RandomErasing` introduce?

 A. Randomly removes/changes a region of an image\
 B. Removes labels\
 C. Removes validation data\
 D. Deletes model weights

 **Answer: A — It randomly masks/erases image regions.**

---

 ### 51\. Why are random augmentations important in SSL?

 A. Guarantee 100% pretext accuracy\
 B. Improve diversity and robustness of representations\
 C. Remove the backbone\
 D. Make all tasks trivial

 **Answer: B — They expose the model to varied views of the data.**

---

 ### 52\. Why can augmentation be harmful to a pretext task?

 A. It can destroy or conflict with the signal being predicted\
 B. It always increases memory\
 C. It always reduces dataset size\
 D. It prevents PyTorch running

 **Answer: A — An augmentation can make the artificial target ambiguous.**

---

 ### 53\. Why is random rotation problematic for four-way rotation prediction?

 A. It increases resolution\
 B. It changes the rotation signal that defines the label\
 C. It prevents convolution\
 D. It removes colour

 **Answer: B — The model could see a different rotation from the one encoded by the target.**

---

 # Section F — Model Architecture

 ### 54\. What is the purpose of `ConvBackbone`?

 A. Produce reusable feature representations\
 B. Perform validation splitting\
 C. Generate dataset labels\
 D. Store metrics

 **Answer: A — It acts as the feature extractor.**

---

 ### 55\. What is the output channel count of:

```
nn.Conv2d(input_channels, 32, 3, padding=1)
```

 A. 3\
 B. 16\
 C. 32\
 D. 64

 **Answer: C — 32.**

---

 ### 56\. What is the output channel count of:

```
self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
```

 A. 64\
 B. 128\
 C. 256\
 D. 512

 **Answer: C — 256.**

---

 ### 57\. What does:

```
nn.MaxPool2d(2, 2)
```

 generally do to spatial dimensions?

 A. Doubles them\
 B. Halves them\
 C. Leaves them unchanged\
 D. Converts them to vectors

 **Answer: B — With kernel size 2 and stride 2, spatial dimensions are generally halved.**

---

 ### 58\. Which operation reduces the final spatial maps to `1 × 1`?

 A. `Linear`\
 B. `Dropout`\
 C. `AdaptiveAvgPool2d(1)`\
 D. `Conv2d`

 **Answer: C — `AdaptiveAvgPool2d(1)`.**

---

 ### 59\. Why is adaptive average pooling useful?

 A. Produces a fixed spatial output size before fully connected layers\
 B. Creates labels\
 C. Randomly crops images\
 D. Calculates accuracy

 **Answer: A — It produces a predictable `1 × 1` spatial representation.**

---

 ### 60\. What is the input size of:

```
self.fc1 = nn.Linear(256, 256)
```

 A. 32\
 B. 128\
 C. 256\
 D. 512

 **Answer: C — 256.**

---

 ### 61\. What is the output dimensionality of:

```
self.fc2 = nn.Linear(256, 128)
```

 A. 32\
 B. 64\
 C. 128\
 D. 256

 **Answer: C — 128.**

---

 ### 62\. What is the final backbone output size?

```
backbone_output_size = 64
```

 A. 32\
 B. 64\
 C. 128\
 D. 256

 **Answer: B — 64.**

---

 ### 63\. What is the purpose of dropout?

 A. Increase parameters\
 B. Regularise by randomly dropping activations during training\
 C. Convert images to tensors\
 D. Calculate accuracy

 **Answer: B — Dropout is a regularisation mechanism.**

---

 ### 64\. Why separate the backbone from the head?

 A. To reuse the learned representation for different tasks\
 B. To avoid PyTorch\
 C. To eliminate training\
 D. To ensure identical outputs

 **Answer: A — The same representation can feed different task-specific heads.**

---

 # Section G — Classification Head

 ### 65\. What is the input size of the target classification head?

 A. 3\
 B. 32\
 C. 64\
 D. 128

 **Answer: C — 64.**

---

 ### 66\. What is the projection size?

```
projection_size = 32
```

 A. 16\
 B. 32\
 C. 64\
 D. 128

 **Answer: B — 32.**

---

 ### 67\. What determines `num_classes`?

 A. Number of source images\
 B. Number of target classes\
 C. Number of convolutional layers\
 D. Batch size

 **Answer: B — The number of classes in the task being predicted.**

---

 ### 68\. What does the classification head return?

 A. Class logits\
 B. Class names\
 C. Normalised images\
 D. Feature maps only

 **Answer: A — Class logits.**

---

 ### 69\. What does this do?

```
x = self.classifier(F.relu(x))
```

 A. Applies ReLU before the final classifier\
 B. Applies softmax before the classifier\
 C. Converts logits into labels\
 D. Performs backpropagation

 **Answer: A — ReLU is applied before the final linear classification layer.**

---

 ### 70\. Why doesn't the final classifier explicitly apply softmax?

 A. `CrossEntropyLoss` operates directly on logits and incorporates the relevant log-softmax calculation\
 B. Softmax is never useful\
 C. Accuracy requires raw images\
 D. ReLU performs softmax

 **Answer: A — The model should normally provide logits directly to `CrossEntropyLoss`.**  PyTorch Documentation+1

---

 # Section H — Accuracy, Metrics and Visualisation

 ### 71\. What does `top1_acc()` calculate?

```
(pred.argmax(axis=1) == y).float().mean().item()
```

 A. Mean squared error\
 B. Percentage of examples whose highest-scoring class matches the target\
 C. Top-5 accuracy\
 D. Validation loss

 **Answer: B — Top-1 classification accuracy.**

---

 ### 72\. What does `argmax(axis=1)` select?

 A. Highest-scoring class for each example\
 B. Highest-loss image\
 C. Highest-accuracy batch\
 D. Lowest feature

 **Answer: A — The class with the largest logit for each example.**

---

 ### 73\. Why use `.item()`?

 A. Converts a single tensor value into a Python scalar\
 B. Calculates gradients\
 C. Moves to GPU\
 D. Reshapes the batch

 **Answer: A — It extracts a Python scalar from a one-element tensor.**

---

 ### 74\. What does `TrainingMetrics.log_train()` record?

 A. Training loss and accuracy measurements\
 B. Only final validation accuracy\
 C. Dataset labels\
 D. Architecture

 **Answer: A — Training metrics.**

---

 ### 75\. What does `TrainingMetrics.log_val()` record?

 A. Validation metrics\
 B. Every image\
 C. Optimizer gradients\
 D. Pretext labels

 **Answer: A — Validation loss/accuracy information.**

---

 ### 76\. What does `best_val_acc` return?

```
return np.max(self.val_acc)
```

 A. Lowest validation accuracy\
 B. Mean training accuracy\
 C. Highest recorded validation accuracy\
 D. Final training accuracy

 **Answer: C — Maximum validation accuracy.**

---

 ### 77\. What does `best_val_loss` return?

 A. Maximum validation loss\
 B. Minimum validation loss\
 C. Average training loss\
 D. First validation loss

 **Answer: B — Minimum recorded validation loss.**

---

 ### 78\. Why use an exponentially weighted mean for training curves?

 A. Smooth noisy batch-level measurements\
 B. Increase accuracy\
 C. Remove validation data\
 D. Calculate gradients

 **Answer: A — It makes noisy training curves easier to interpret.**

---

 ### 79\. What does `PercentFormatter(xmax=1.0)` accomplish?

 A. Displays `0.75` as `75%`\
 B. Converts accuracy into loss\
 C. Clips accuracy\
 D. Converts labels to strings

 **Answer: A — It changes the display format to percentages.**

---

 ### 80\. What is the purpose of `epoch_steps`?

 A. Record positions where validation metrics are logged\
 B. Store learning rates\
 C. Store class indices\
 D. Store image dimensions

 **Answer: A — It tracks the x-axis positions for validation measurements.**

---

 # Section I — Supervised Training Loop

 ### 81\. Which loss is used?

 A. `MSELoss`\
 B. `CrossEntropyLoss`\
 C. `L1Loss`\
 D. `BCELoss`

 **Answer: B — `nn.CrossEntropyLoss()`.**

---

 ### 82\. Which optimizer is used?

 A. SGD\
 B. RMSprop\
 C. AdamW\
 D. Adagrad

 **Answer: C — AdamW.**

---

 ### 83\. What does `l2_reg` control?

 A. Dropout probability\
 B. Weight decay/L2 regularisation strength\
 C. Number of classes\
 D. Batch size

 **Answer: B — It controls regularisation through weight decay.**

---

 ### 84\. What does `opt.zero_grad()` do?

 A. Clears gradients from the previous step\
 B. Resets model weights\
 C. Resets dataset\
 D. Permanently sets loss to zero

 **Answer: A — Gradients need to be cleared before the next backward pass.**

---

 ### 85\. What happens in:

```
pred = model(x)
```

 A. Backpropagation\
 B. Forward propagation\
 C. Dataset splitting\
 D. Optimizer reset

 **Answer: B — It performs the forward pass.**

---

 ### 86\. What does:

```
batch_loss.backward()
```

 do?

 A. Computes gradients of the loss with respect to trainable parameters\
 B. Directly updates parameters\
 C. Evaluates validation data\
 D. Clears gradients

 **Answer: A — It performs backpropagation.**

---

 ### 87\. What does:

```
opt.step()
```

 do?

 A. Computes validation accuracy\
 B. Updates model parameters using gradients\
 C. Clears the dataset\
 D. Creates a new model

 **Answer: B — It applies the optimizer update.**

---

 ### 88\. Why call `model.train()`?

 A. Enable training behaviour such as dropout\
 B. Create the dataset\
 C. Compute validation metrics\
 D. Freeze parameters

 **Answer: A — It switches the model to training mode.**

---

 ### 89\. Why call `model.eval()` during validation?

 A. Activate dropout\
 B. Switch to evaluation behaviour\
 C. Update weights\
 D. Shuffle validation data

 **Answer: B — It switches modules such as dropout to evaluation behaviour.**

---

 ### 90\. What is the purpose of:

```
with torch.no_grad():
```

 during evaluation?

 A. Disable gradient tracking\
 B. Enable dropout\
 C. Increase learning rate\
 D. Update parameters

 **Answer: A — It prevents unnecessary gradient computation.**

---

 ### 91\. When is validation performed?

 A. After every batch\
 B. After every epoch\
 C. Only before training\
 D. Only after training finishes

 **Answer: B — Validation is performed after each epoch in the supplied training structure.**

---

 ### 92\. What does `evaluate_model()` return?

 A. Training loss and accuracy\
 B. Validation loss and accuracy\
 C. Model weights\
 D. Dataset labels

 **Answer: B — Validation loss and validation accuracy.**

---

 ### 93\. How are validation losses aggregated?

```
val_loss = np.mean(batch_val_losses)
```

 A. Maximum\
 B. Minimum\
 C. Mean\
 D. Sum

 **Answer: C — Mean.**

---

 ### 94\. What does the `seed` parameter affect?

 A. Reproducibility of relevant random behaviour\
 B. Number of classes\
 C. Input image size\
 D. Loss function

 **Answer: A — It helps make stochastic behaviour reproducible.**

---

 # Section J — Early Stopping and Training Behaviour

 ### 95\. What is the purpose of early stopping?

 A. Stop training when validation behaviour indicates deterioration/convergence\
 B. Guarantee 100% training accuracy\
 C. Increase dataset size\
 D. Remove validation

 **Answer: A — It helps avoid unnecessary or deteriorating training.**

---

 ### 96\. In the supplied code, early stopping is only considered when:

 A. `e > 0`\
 B. `e > 1`\
 C. `e > 4`\
 D. `e > 20`

 **Answer: C — `e > 4`.**

---

 ### 97\. What does this calculate?

```
recent_val_loss = np.mean(metrics.val_loss[-4:-1])
```

 A. Mean of current and previous three losses\
 B. Mean of the three validation losses immediately before the current one\
 C. Minimum validation loss\
 D. Training loss

 **Answer: B — `[-4:-1]` excludes the most recent element.**

---

 ### 98\. When does the supplied early-stopping condition trigger?

```
metrics.val_loss[-1] > recent_val_loss
```

 A. Current validation loss exceeds the recent previous mean\
 B. Training accuracy reaches 100%\
 C. Validation loss reaches zero\
 D. Training loss exceeds one

 **Answer: A — The current validation loss is worse than the recent comparison average.**

---

 ### 99\. Why might the baseline overfit quickly?

 A. The labelled target dataset is relatively small\
 B. Dataset is infinite\
 C. Model has no parameters\
 D. Validation data is used for training

 **Answer: A — A relatively small labelled dataset can make overfitting easier.**

---

 ### 100\. What is the purpose of L2 regularisation?

 A. Encourage smaller weights and help reduce overfitting\
 B. Increase image resolution\
 C. Generate pseudo-labels\
 D. Increase classes

 **Answer: A — It penalises excessively large weights.**

---

 # Section K — Pretext Tasks

 ### 101\. Which is a valid pretext task?

 A. Predict a random number unrelated to the image\
 B. Predict the rotation applied to an image\
 C. Memorise the filename\
 D. Predict the batch index

 **Answer: B — Rotation prediction is a classic pretext task.**

---

 ### 102\. In four-way rotation prediction, which classes might be used?

 A. 0°, 90°, 180°, 270°\
 B. 0° and 360°\
 C. Every possible continuous angle\
 D. Horizontal and vertical flips only

 **Answer: A — Four discrete rotation classes.**

---

 ### 103\. Which task predicts how two patches are spatially related?

 A. Relative-position prediction\
 B. Standard classification\
 C. Random erasing\
 D. Batch normalisation

 **Answer: A — Relative-position prediction.**

---

 ### 104\. What is the jigsaw pretext task?

 A. Predict how shuffled image patches should be arranged\
 B. Predict the original class label\
 C. Remove noise\
 D. Predict batch size

 **Answer: A — The model solves a spatial arrangement problem.**

---

 ### 105\. A same-image/different-image task could ask the model to:

 A. Determine whether two augmented crops came from the same original image\
 B. Predict the filename\
 C. Predict the GPU\
 D. Predict learning rate

 **Answer: A — It learns relationships between different views of images.**

---

 ### 106\. Why might simple denoising be a poor pretext task?

 A. It may primarily teach low-level pixel features\
 B. Denoising cannot be implemented\
 C. It always requires labels\
 D. CNNs cannot perform denoising

 **Answer: A — The task may not encourage sufficiently semantic representations.**

---

 ### 107\. What characterises a good pretext task?

 A. It encourages representations useful for downstream learning\
 B. It must achieve 100% accuracy\
 C. It uses original class labels\
 D. It is unrelated to the images

 **Answer: A — Transferability is the key consideration.**

---

 ### 108\. Why can extremely high pretext accuracy be suspicious?

 A. The pretext task may be too easy\
 B. It proves the model is broken\
 C. Target labels must be wrong\
 D. GPU is too fast

 **Answer: A — Very easy tasks may not force useful representation learning.**

---

 ### 109\. What should ideally happen during useful pretext training?

 A. The model learns a meaningful representation and the pretext objective converges reasonably\
 B. The model learns nothing\
 C. Accuracy must be exactly 100%\
 D. Loss increases indefinitely

 **Answer: A — Successful training should produce a useful representation.**

---

 ### 110\. Why can augmentations help SSL?

 A. Increase variation and encourage robust representations\
 B. Eliminate labels in every ML problem\
 C. Guarantee transfer success\
 D. Make every image identical

 **Answer: A — They provide varied views of the underlying data.**

---

 # Section L — Contrastive and Representation Learning

 ### 111\. What is the basic intuition behind contrastive learning?

 A. Learn useful relationships between related and unrelated examples/views\
 B. Train only on class labels\
 C. Remove transformations\
 D. Predict filenames

 **Answer: A — Contrastive learning learns representation relationships between examples.**

---

 ### 112\. Which simpler alternative to full SimCLR is mentioned?

 A. Pairwise contrastive loss\
 B. Binary cross-entropy only\
 C. K-means\
 D. PCA

 **Answer: A — Pairwise contrastive loss.**

---

 ### 113\. Which related loss is also mentioned?

 A. Triplet loss\
 B. Poisson loss\
 C. Hinge regression only\
 D. Dice loss

 **Answer: A — Triplet loss.**

---

 ### 114\. Why is full SimCLR not recommended for a quick practical?

 A. It is more complex to implement\
 B. It cannot work on images\
 C. It requires CIFAR-100 labels\
 D. It doesn't use neural networks

 **Answer: A — It introduces considerably more implementation complexity.**

---

 ### 115\. What is a key idea in contrastive pretext learning?

 A. Learn relationships between examples/views rather than relying on human class labels\
 B. Every image receives a human label\
 C. Only the head is trained\
 D. Augmentation cannot be used

 **Answer: A — The relationships themselves provide the learning signal.**

---

 # Section M — Fine-Tuning and Transfer Learning

 ### 116\. What should be transferred after pre-training?

 A. The trained backbone\
 B. Pretext labels\
 C. Source DataLoader only\
 D. Validation metrics

 **Answer: A — The learned backbone parameters.**

---

 ### 117\. Why is the pretext head normally not transferred directly?

 A. Its outputs correspond to the pretext task rather than the target task\
 B. It has no parameters\
 C. It belongs to the dataset\
 D. It is always frozen

 **Answer: A — Its output space is task-specific.**

---

 ### 118\. What is the purpose of saving/loading a backbone `state_dict`?

 A. Transfer learned model parameters\
 B. Store images\
 C. Save labels as text\
 D. Change image resolution

 **Answer: A — It preserves and reloads learned parameters.**

---

 ### 119\. Which strategy may be useful during fine-tuning?

 A. Use a lower learning rate\
 B. Always use a larger learning rate\
 C. Delete the backbone\
 D. Remove all regularisation

 **Answer: A — A lower learning rate can help preserve useful pretrained features while adapting them.**

---

 ### 120\. Why might layer freezing be useful?

 A. Preserve useful pretrained representations while training selected layers\
 B. Increase classes\
 C. Label source data\
 D. Eliminate validation

 **Answer: A — Freezing limits which parameters can change.**

---

 ### 121\. For a fair comparison, what should be considered regarding augmentation?

 A. Use comparable augmentation for the baseline and fine-tuning experiments\
 B. Remove validation\
 C. Change target classes\
 D. Destroy baseline labels

 **Answer: A — Otherwise augmentation differences can confound the comparison.**

---

 ### 122\. Which metric is used to compare SSL with the baseline?

 A. Best validation accuracy\
 B. Number of filters\
 C. Number of source images\
 D. Batch size

 **Answer: A — Best downstream validation accuracy.**

---

 ### 123\. What does this represent?

```
improvement = ft_acc - baseline_acc
```

 A. Difference in validation accuracy\
 B. Difference in training loss\
 C. Difference in dataset size\
 D. Difference in class count

 **Answer: A — Downstream validation-accuracy improvement.**

---

 # Section N — Code Interpretation

 ### 124\. What does this do?

```
baseline_model = nn.Sequential(
    baseline_backbone,
    target_task_head
)
```

 A. Chains the backbone output into the classification head\
 B. Trains two independent models\
 C. Freezes the backbone\
 D. Deletes the head

 **Answer: A — The backbone feeds its features into the target head.**

---

 ### 125\. What is the expected output shape for batch size `B` and `C` classes?

 A. `(B, 3)`\
 B. `(B, 64)`\
 C. `(B, 32)`\
 D. `(B, C)`

 **Answer: D — `(B, C)`.**

 **Why:** Each example gets one logit per class.

---

 ### 126\. If `num_target_classes = 5` and batch size is 128, what is the output shape?

 A. `(5, 128)`\
 B. `(128, 5)`\
 C. `(128, 64)`\
 D. `(5, 64)`

 **Answer: B — `(128, 5)`.**

---

 ### 127\. Why can the same backbone be used for different downstream tasks?

 A. Its output is a feature representation rather than a task-specific prediction\
 B. It has no trainable parameters\
 C. It knows every class automatically\
 D. It only works with five classes

 **Answer: A — The head determines the task-specific output.**

---

 ### 128\. What does `summary()` from `torchinfo` provide?

 A. Model architecture/parameter summary\
 B. Dataset download\
 C. Confusion matrix\
 D. Pretext task

 **Answer: A — It summarises the model structure and parameters.**

---

 ### 129\. What does `inspect_dataset()` help verify?

 A. Dataset images and labels/classes when available\
 B. Optimizer gradients\
 C. GPU temperature\
 D. Parameter updates

 **Answer: A — It allows inspection of the dataset contents.**

---

 ### 130\. What is the purpose of `inspect_batch()`?

 A. Visually inspect transformed images and optionally labels/predictions\
 B. Train the model\
 C. Calculate learning rate\
 D. Split the dataset

 **Answer: A — It helps visually debug the data pipeline and predictions.**

---

 ### 131\. What does this calculate?

```
pred_idxs = predictions.argmax(dim=1)
```

 A. Highest-scoring predicted class for each sample\
 B. Lowest-scoring class\
 C. Mean class index\
 D. Random class index

 **Answer: A — It converts logits into predicted class indices.**

---

 ### 132\. What does this calculate?

```
F.softmax(predictions, dim=1).max(dim=1)[0]
```

 A. Highest predicted class probability for each example\
 B. Highest raw pixel value\
 C. Lowest class probability\
 D. Training loss

 **Answer: A — It finds the maximum probability after softmax.**

---

 ### 133\. Why use:

```
images[b].permute([1,2,0])
```

 A. Change `C×H×W` into `H×W×C`\
 B. Flatten image\
 C. Resize image\
 D. Rotate image

 **Answer: A — It changes the dimension order for display.**

---

 ### 134\. What does this do?

```
img = (img_p - img_p.min()) / (img_p.max() - img_p.min())
```

 A. Rescales values approximately into `[0,1]`\
 B. Calculates cross-entropy\
 C. Adds noise\
 D. Computes accuracy

 **Answer: A — It performs min-max rescaling for display.**

---

 # Section O — Practical Design and Debugging

 ### 135\. If a pretext task reaches extremely high accuracy almost immediately, what should you consider?

 A. Making the task harder or increasing suitable augmentation\
 B. Deleting the backbone\
 C. Increasing target classes\
 D. Removing training data

 **Answer: A — The task may be too easy.**

---

 ### 136\. Why might a decreasing pretext loss fail to improve downstream performance?

 A. The task may teach features that are not useful for the downstream task\
 B. Decreasing loss always guarantees transfer\
 C. Target labels must be identical\
 D. Validation accuracy is irrelevant

 **Answer: A — Pretext-task success does not guarantee transferable representations.**

---

 ### 137\. Why should a pretext task avoid being completely trivial?

 A. A trivial task may not require rich representations\
 B. Trivial tasks always cause NaNs\
 C. They cannot be implemented\
 D. They require labels

 **Answer: A — The task should encourage meaningful feature learning.**

---

 ### 138\. Which component is particularly important when implementing a custom pretext task?

 A. Data transformation and `collate_fn` that generate valid input-target pairs\
 B. Python version\
 C. Removing validation\
 D. Changing every convolution

 **Answer: A — The data pipeline defines the artificial learning problem.**

---

 ### 139\. If a pretext task requires two images as input, what may need to change?

 A. Model/training-loop input handling may need to process both images\
 B. Only dataset name\
 C. Only batch size\
 D. Nothing

 **Answer: A — The model must receive and combine/process both inputs appropriately.**

---

 ### 140\. Why might a custom pretext task require a different loss function?

 A. Some tasks are regression, reconstruction, similarity, etc. rather than class-index classification\
 B. CrossEntropyLoss works identically for every task\
 C. Optimizers require a new loss each epoch\
 D. DataLoaders choose the loss automatically

 **Answer: A — Loss must match the learning objective.**

---

 # Section P — Advanced Integrated Questions

 ### 141\. Consider:

 **Unlabelled source → pretext task → pretrained backbone → new target head → target fine-tuning**

 What is the purpose of the first three stages?

 A. Learn reusable representations without human-provided source labels\
 B. Directly solve the target classification problem\
 C. Increase target labels\
 D. Replace validation

 **Answer: A — They perform representation learning before the target task.**

---

 ### 142\. Why can source classes differ from target classes?

 A. Visual features such as shapes, textures and structures can transfer\
 B. Source labels automatically become target labels\
 C. Source labels are secretly retained\
 D. Classes must always match

 **Answer: A — Transfer operates through learned representations rather than matching class IDs.**

---

 ### 143\. Suppose the target task has five classes but the pretext task has four rotation classes. What is correct?

 A. Reuse the pretrained backbone but replace the four-class head\
 B. Pretext head must output five classes\
 C. Target must have four classes\
 D. Backbone cannot be reused

 **Answer: A — The backbone is reusable; the task-specific head changes.**

---

 ### 144\. Why is baseline validation accuracy important?

 A. It provides the reference point for judging downstream benefit\
 B. Determines source labels\
 C. Becomes the pretext target\
 D. Determines image resolution

 **Answer: A — Without the baseline, it is difficult to assess whether SSL helped.**

---

 ### 145\. Baseline = 62%; SSL fine-tuned = 68%. What is the absolute improvement?

 A. 4 percentage points\
 B. 6 percentage points\
 C. 10 percentage points\
 D. 110 percentage points

 **Answer: B — 6 percentage points.**

 **Calculation:** `68% − 62% = 6 percentage points`.

---

 ### 146\. Which comparison is most informative?

 A. Fine-tuned SSL target validation accuracy vs supervised-from-scratch target validation accuracy\
 B. Source size vs target size only\
 C. Number of imports\
 D. Number of notebook cells

 **Answer: A — This directly tests the practical's main hypothesis.**

---

 ### 147\. Why should baseline and fine-tuning conditions generally be comparable?

 A. To reduce confounding from changes unrelated to SSL\
 B. To ensure identical learned weights\
 C. To make datasets identical\
 D. To prevent validation

 **Answer: A — A fair comparison controls other factors where possible.**

---

 ### 148\. What does fine-tuning mean?

 A. Starting from pretrained parameters and adapting them to the target task\
 B. Training only on unlabelled data forever\
 C. Never changing weights\
 D. Training without a target head

 **Answer: A — Fine-tuning adapts pretrained parameters to the downstream task.**

---

 ### 149\. Which statement best describes augmentation?

 A. It can improve robustness but must not destroy the pretext signal\
 B. It always improves every model\
 C. It should always include the transformation being predicted\
 D. It replaces validation

 **Answer: A — Augmentation must be compatible with the learning objective.**

---

 ### 150\. What is the ultimate experimental question?

 A. Whether representations learned from unlabelled source data improve labelled downstream performance\
 B. Whether CIFAR-10 has more images than CIFAR-100\
 C. Whether AdamW is faster than SGD\
 D. Whether all pretext tasks reach 100% accuracy

 **Answer: A — This is the central research question of the practical.**

---

 # High-Yield Revision Sheet

 If you're preparing for an exam, these are the concepts I'd make sure you can explain **without looking at your notes**.

 ## 1\. The entire practical

 Memorise this:

```
UNLABELLED SOURCE DATA
        ↓
   PRETEXT TASK
        ↓
  PRETRAIN BACKBONE
        ↓
   DISCARD OLD HEAD
        ↓
  NEW TARGET HEAD
        ↓
   FINE-TUNE
        ↓
TARGET VALIDATION ACCURACY
        ↓
COMPARE WITH BASELINE
```

 The central comparison is:

```
CNN trained from scratch
             VS
SSL-pretrained CNN + fine-tuning
```

---

 ## 2\. Self-supervised learning

 The most important definition:

 > **Self-supervised learning creates the learning signal automatically from the data rather than relying on human-provided labels.**

 For rotation prediction:

```
Original image
     ↓
Rotate 90°
     ↓
Artificial label = 90°
```

 Therefore:

```
image → CNN → rotation prediction
```

 The rotation label does **not** come from a human annotator.

---

 ## 3\. Backbone vs head

 Think:

```
                 MODEL
                   │
          ┌────────┴────────┐
          ↓                 ↓
      BACKBONE             HEAD
          ↓                 ↓
      FEATURES          PREDICTION
```

 The **backbone** learns reusable features.

 The **head** performs the specific task.

 Therefore:

```
PRETEXT:
backbone → rotation head → 4 classes

TARGET:
same backbone → target head → 5 classes
```

---

 ## 4\. The most important experimental point

 Do **not** confuse:

```
high pretext accuracy
```

 with:

```
good transfer
```

 The practical ultimately cares about:

```
TARGET VALIDATION PERFORMANCE
```

 For example:

```
Pretext A:
95% pretext accuracy
45% target accuracy

Pretext B:
75% pretext accuracy
55% target accuracy
```

 The second representation transferred better to the target task.

---

 ## 5\. Four-way rotation

 A simple pretext task is:

```
0°
90°
180°
270°
```

 Random guessing gives approximately:

```
1 / 4 = 25%
```

 The model learns to predict which rotation was applied.

 But **do not randomly rotate the image again after creating the rotation label**, because that can destroy the relationship between the image and target.

---

 ## 6\. Training loop

 This is extremely high-yield:

```
opt.zero_grad()

pred = model(x)

loss = loss_func(pred, y)

loss.backward()

opt.step()
```

 Know exactly what each line does:

 | Code | Meaning |
| --- | --- |
| `zero_grad()` | Clear previous gradients |
| `model(x)` | Forward pass |
| `loss_func(...)` | Calculate loss |
| `backward()` | Calculate gradients |
| `step()` | Update weights |

---

 ## 7\. Training vs evaluation

 ### Training

```
model.train()
```

 Used for training behaviour such as dropout.

 ### Evaluation

```
model.eval()

with torch.no_grad():
    ...
```

 `eval()` changes evaluation-sensitive module behaviour, while `no_grad()` prevents gradient tracking.

---

 ## 8\. CrossEntropyLoss

 Remember:

```
model
  ↓
raw logits
  ↓
CrossEntropyLoss
```

 Do **not** normally do:

```
model
 ↓
softmax
 ↓
CrossEntropyLoss
```

 The model outputs logits and `CrossEntropyLoss` handles the appropriate normalization internally.  PyTorch Documentation+1

 For 5-class classification with batch size 128:

```
pred.shape = [128, 5]
target.shape = [128]
```

 The target contains integer class indices such as:

```
0
3
1
4
2
...
```

 and class-index targets are represented using `torch.long`.  PyTorch Documentation

---

 ## 9\. `argmax`

 Suppose:

```
logits =
[1.2, 0.4, 3.8, 0.7, 0.1]
```

 Then:

```
argmax()
```

 returns:

```
2
```

 because class 2 has the highest score.

 Therefore:

```
pred.argmax(dim=1)
```

 gives the predicted class for each example.

---

 ## 10\. Backbone dimensions

 Know this progression:

```
RGB
 ↓
Conv 3 → 32
 ↓
Conv 32 → 64
 ↓
Pool
 ↓
Conv 64 → 128
 ↓
Conv 128 → 256
 ↓
AdaptiveAvgPool → 1×1
 ↓
Flatten
 ↓
Linear 256 → 256
 ↓
Linear 256 → 128
 ↓
Linear 128 → 64
 ↓
64-dimensional representation
```

 The key final number:

```
BACKBONE OUTPUT = 64
```

---

 ## 11\. Classification head

 For the target task:

```
64 features
      ↓
Linear 64 → 32
      ↓
ReLU
      ↓
Linear 32 → 5
      ↓
5 logits
```

 The `5` comes from the **number of target classes**.

 If the target had 10 classes:

```
Linear(... → 10)
```

 would be required instead.

---

 ## 12\. DataLoader concepts

 Remember:

```
shuffle=True
```

 for training.

```
shuffle=False
```

 for validation.

 Why?

 Training benefits from randomised sample order.

 Validation simply needs consistent evaluation of the validation examples.

---

 ## 13\. `collate_fn`

 Think of:

```
individual dataset examples
          ↓
      collate_fn
          ↓
       batch
```

 A custom `collate_fn` is particularly useful in this practical because it can apply transformations and construct the tensors required by the model.

 For example:

```
raw PIL images
      ↓
transform each image
      ↓
torch.stack(...)
      ↓
batch tensor
```

---

 ## 14\. Augmentation

 Augmentation can help because:

```
same underlying image
       ↓
different views
       ↓
more robust representation
```

 Examples:

```
RandomResizedCrop
RandomHorizontalFlip
ColorJitter
RandomGrayscale
RandomErasing
```

 But remember:

 > **The augmentation must not destroy the pretext signal.**

 For rotation prediction, arbitrary additional rotations are therefore problematic.

---

 ## 15\. Transfer learning

 The transfer process is:

```
PRETEXT MODEL

backbone + pretext head
        ↓
      train
        ↓
pretrained backbone
```

 Then:

```
TARGET MODEL

pretrained backbone
        +
new target head
        ↓
fine-tune
```

 The pretext head normally does **not** transfer because it predicts the wrong task.

---

 ## 16\. Baseline

 The baseline is:

```
small labelled target dataset
              ↓
        randomly initialised
             CNN
              ↓
           training
              ↓
       validation accuracy
```

 SSL model:

```
large unlabelled source
          ↓
      pretext task
          ↓
   pretrained backbone
          ↓
      target head
          ↓
       fine-tuning
          ↓
   validation accuracy
```

 Then:

```
SSL accuracy - baseline accuracy
```

 gives the absolute improvement in **percentage points**.

---

 # The 10 Things I'd Memorise First

 If you're short on time, learn these ten extremely well:

 1. **SSL:** unlabelled data → automatically generated learning signal.
2. **Pretext task:** artificial task used to learn representations.
3. **Backbone:** feature extractor.
4. **Head:** task-specific predictor.
5. **Transfer:** keep backbone, replace head.
6. **Fine-tuning:** adapt pretrained parameters to the target task.
7. **Baseline:** same target task trained from scratch.
8. **Main metric:** downstream target validation accuracy.
9. **Training loop:** `zero_grad → forward → loss → backward → step`.
10. **Central question:** **Does self-supervised pre-training improve downstream performance compared with training from scratch?**

 That last question is the one that ties essentially the entire practical together.
