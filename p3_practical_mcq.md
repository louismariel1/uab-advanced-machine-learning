# Advanced Machine Learning Practical 3 — Comprehensive MCQ

 Below is a comprehensive multiple-choice question bank covering the **concepts, PyTorch code, dataset engineering, preprocessing, model architecture, training loops, self-supervised learning, pretext tasks, augmentation, transfer learning, and assessment requirements** in the notebook.

 There are **80 MCQs**, progressing from foundational concepts to code-level questions. Each question has **one best answer**.

---

 ## Section A — Self-Supervised Learning Concepts

 ### 1\. What is the primary objective of the practical?

 A. Maximise training accuracy on CIFAR-10\
 B. Maximise validation accuracy on a small labelled target dataset using a large pre-training dataset\
 C. Minimise validation loss on an unlabelled dataset\
 D. Train a generative model on CIFAR-100

---

 ### 2\. In this practical, what distinguishes the source dataset from the target dataset?

 A. The source dataset is small and labelled; the target is large and unlabelled\
 B. Both datasets are labelled and contain identical classes\
 C. The source dataset is large and unlabelled; the target is small and labelled\
 D. Both datasets are completely unlabelled

---

 ### 3\. What is the key idea behind self-supervised learning in this practical?

 A. Manually label every source image\
 B. Use the validation set as training data\
 C. Create useful training labels/signals automatically from unlabelled data\
 D. Remove the feature extractor before training

---

 ### 4\. The source data is described as being used for:

 A. Supervised classification\
 B. Unsupervised pre-training\
 C. Final validation only\
 D. Hyperparameter optimisation only

---

 ### 5\. How does this practical differ from the earlier semi-supervised pseudo-labeling exercise?

 A. The source and target datasets are identical\
 B. The datasets represent different domains/tasks and the source labels are unavailable\
 C. No labelled data is used at all\
 D. The target dataset is larger than the source dataset

---

 ### 6\. Why can self-supervised learning transfer between datasets with different classes?

 A. The model memorises the source labels\
 B. Related datasets can share useful visual/structural features even when their classes differ\
 C. The target labels are automatically copied from the source\
 D. The output classes are forced to be identical

---

 ### 7\. Which component is intended to be reused during downstream fine-tuning?

 A. Only the source dataset\
 B. Only the pretext-task labels\
 C. The pretrained backbone/feature encoder\
 D. The validation loss

---

 ### 8\. During transfer to the target task, what happens to the task-specific pretext head?

 A. It is necessarily retained unchanged\
 B. It is replaced by a target-task-specific head\
 C. It is converted into the validation set\
 D. It is deleted together with the backbone

---

 ### 9\. What is the purpose of the supervised baseline?

 A. To provide an upper bound that cannot be exceeded\
 B. To measure performance when learning only from the small labelled target dataset\
 C. To train the pretext task\
 D. To replace the validation dataset

---

 ### 10. If self-supervised pre-training is effective, what outcome is hoped for?

 A. Lower target validation accuracy\
 B. Improved downstream performance compared with supervised training from scratch\
 C. Perfect pretext-task accuracy\
 D. Zero training time

---

 ## Section B — Dataset Design

 ### 11. Which dataset is used as the target in the CIFAR configuration?

 A. CIFAR-10\
 B. CIFAR-100 subset\
 C. MNIST\
 D. ImageNet

---

 ### 12\. Which dataset is used as the unlabelled source in the CIFAR configuration?

 A. CIFAR-10\
 B. CIFAR-100\
 C. STL-10 labelled portion\
 D. MNIST

---

 ### 13\. What happens to the CIFAR-10 labels?

 A. They are converted into CIFAR-100 labels\
 B. They are shuffled\
 C. They are explicitly destroyed/ignored\
 D. They are used as validation labels

---

 ### 14\. How many classes does each CIFAR-100 superclass contain in the notebook?

 A. 2\
 B. 5\
 C. 10\
 D. 20

---

 ### 15\. Suppose `TARGET_CATEGORY = "vehicles"`. Which is one of its CIFAR-100 subclasses?

 A. dolphin\
 B. bicycle\
 C. tiger\
 D. sunflower

---

 ### 16\. What does this expression do?

```
superclass_idxs = set(
    [cifar100.class_to_idx[n] for n in superclass_names]
)
```

 A. Converts images into tensors\
 B. Finds the original CIFAR-100 integer indices of the selected classes\
 C. Creates a train/validation split\
 D. Counts the number of images

---

 ### 17\. What is the purpose of `subset_idxs`?

```
subset_idxs = [
    i for i, idx in enumerate(cifar100.targets)
    if idx in superclass_idxs
]
```

 A. It identifies images belonging to the selected five classes\
 B. It randomly augments images\
 C. It creates the source dataset\
 D. It removes all labels

---

 ### 18\. Why is `class_remap` needed?

 A. To change image resolution\
 B. To map original CIFAR-100 labels to clean consecutive labels from 0 to 4\
 C. To convert labels to strings\
 D. To normalise image pixels

---

 ### 19\. If the original selected CIFAR-100 classes have indices `[12, 37, 72, 81, 95]`, what does the remapping conceptually produce?

 A. `[12, 37, 72, 81, 95]`\
 B. `[1, 2, 3, 4, 5]` only\
 C. `[0, 1, 2, 3, 4]`\
 D. Random labels

---

 ### 20\. What does `target_data.num_classes` represent?

 A. Number of training batches\
 B. Number of pixels\
 C. Number of target classes\
 D. Number of source images

---

 ### 21\. What is special about STL-10 compared with CIFAR-10 in this practical?

 A. It contains no images\
 B. It has a labelled and a large unlabelled portion\
 C. It only contains one class\
 D. It is smaller and lower resolution

---

 ### 22\. What is the original resolution of STL-10 images?

 A. 28 × 28\
 B. 32 × 32\
 C. 64 × 64\
 D. 96 × 96

---

 ### 23\. Approximately how many unlabelled images does STL-10 provide?

 A. 1,000\
 B. 10,000\
 C. 50,000\
 D. 100,000

---

 ### 24\. Why might STL-10 take longer to train?

 A. It has no GPU support\
 B. Its images are higher resolution and its unlabelled dataset is large\
 C. It has only one class\
 D. It requires labels to be manually entered

---

 ## Section C — Custom Dataset Classes

 ### 25\. What does `UnlabelledDataset.__getitem__()` return?

 A. `(image, label)`\
 B. Only an image\
 C. Only a label\
 D. A model prediction

---

 ### 26\. What does `LabelledDataset.__getitem__()` return?

 A. Only an image\
 B. Only a label\
 C. `(image, target)`\
 D. `(prediction, target)`

---

 ### 27\. Why does `UnlabelledDataset.__getitem__()` contain this check?

```
if not isinstance(img, Image):
    img = PIL.Image.fromarray(img)
```

 A. To convert images to labels\
 B. To ensure images are represented as PIL images when necessary\
 C. To resize all images to 32 × 32\
 D. To normalise them

---

 ### 28\. What does `__len__()` return?

 A. Number of classes\
 B. Number of batches\
 C. Number of stored examples\
 D. Number of pixels

---

 ### 29\. What does this assertion guarantee?

```
assert len(images) == len(targets)
```

 A. Every image has a corresponding target\
 B. Every class has the same number of examples\
 C. Every image has the same dimensions\
 D. Training and validation sets are equal

---

 ### 30\. What does `classes` represent in `LabelledDataset`?

 A. Neural network layers\
 B. Optional human-readable class names\
 C. Optimisation parameters\
 D. Image dimensions

---

 ### 31\. What does `count_labels()` primarily do?

 A. Trains the model\
 B. Counts examples belonging to each class\
 C. Removes class labels\
 D. Normalises class frequencies

---

 ### 32\. What is the purpose of the `seed` argument in `train_test_split()`?

 A. To change the learning rate\
 B. To make the random split reproducible\
 C. To initialise the neural network architecture\
 D. To determine the number of classes

---

 ### 33\. If `test_frac=0.1` and there are 1,000 examples, how many examples are selected for the test/validation subset?

 A. 10\
 B. 50\
 C. 100\
 D. 900

---

 ### 34\. What does `replace=False` mean in:

```
np.random.choice(..., replace=False)
```

 A. Examples can be selected repeatedly\
 B. Examples cannot be selected more than once\
 C. Labels are replaced\
 D. Images are replaced with zeros

---

 ## Section D — Data Splitting and DataLoaders

 ### 35\. What validation fraction is used in the notebook?

 A. 0.01\
 B. 0.05\
 C. 0.10\
 D. 0.50

---

 ### 36\. Why is a validation set required?

 A. To train the model directly\
 B. To evaluate generalisation during training\
 C. To generate CIFAR-10 labels\
 D. To increase the number of classes

---

 ### 37\. What does `shuffle=True` in the training DataLoader accomplish?

 A. Randomises training sample order between epochs\
 B. Deletes samples\
 C. Changes labels\
 D. Normalises the images

---

 ### 38. Why does the validation loader use:

```
shuffle=False
```

 A. Validation should not need random ordering\
 B. Validation images cannot be shuffled technically\
 C. It changes the validation labels\
 D. It enables augmentation

---

 ### 39\. Why are custom `collate_fn` functions useful in this notebook?

 A. They define the neural network architecture\
 B. They control how raw dataset examples are transformed and assembled into batches\
 C. They calculate validation accuracy only\
 D. They replace the optimizer

---

 ### 40\. What does this line do?

```
image_tensor = torch.stack([transform(img) for img in images])
```

 A. Combines individually transformed images into a batch tensor\
 B. Converts labels to strings\
 C. Calculates loss\
 D. Performs backpropagation

---

 ### 41\. What datatype is explicitly used for classification labels?

```
torch.tensor(labels, dtype=torch.long)
```

 A. `torch.float32`\
 B. `torch.int64` / `torch.long`\
 C. `torch.bool`\
 D. `torch.uint8`

---

 ### 42\. Why is `torch.long` appropriate for `CrossEntropyLoss` targets?

 A. CrossEntropyLoss expects integer class indices\
 B. CrossEntropyLoss expects images\
 C. It automatically performs augmentation\
 D. It represents probabilities

---

 ### 43\. What is the purpose of:

```
.to(device)
```

 inside the collate functions?

 A. Moves tensors to CPU only\
 B. Moves tensors to the selected computation device\
 C. Normalises tensors\
 D. Converts tensors to NumPy arrays

---

 ## Section E — Preprocessing and Augmentation

 ### 44\. Which device is selected by this expression?

```
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

 A. Always CPU\
 B. Always GPU\
 C. GPU if CUDA is available, otherwise CPU\
 D. TPU if available

---

 ### 45\. What does `transforms.ToTensor()` generally do?

 A. Converts an image into a PyTorch tensor representation\
 B. Converts tensors into PIL images\
 C. Converts labels into strings\
 D. Computes gradients

---

 ### 46\. What is the purpose of:

```
transforms.Normalize(mean=0.5, std=0.2)
```

 A. Standardise/normalise tensor pixel values using the specified mean and standard deviation\
 B. Resize images\
 C. Rotate images\
 D. Remove labels

---

 ### 47\. What does `RandomHorizontalFlip(p=0.5)` mean?

 A. Every image is flipped\
 B. No image is flipped\
 C. Each image has a 50% probability of being horizontally flipped\
 D. Half the dataset is permanently flipped

---

 ### 48\. What does `RandomApply(..., p=0.5)` do?

 A. Applies the enclosed transformation with probability 0.5\
 B. Always applies the transformation twice\
 C. Applies the transformation to exactly half the pixels\
 D. Disables augmentation

---

 ### 49\. What is the purpose of `ColorJitter`?

 A. Modify image colour properties such as brightness, contrast, saturation, and hue\
 B. Change class labels\
 C. Perform pooling\
 D. Calculate classification loss

---

 ### 50\. What does `RandomErasing` introduce?

 A. Randomly removes/changes a region of an image\
 B. Removes the class label\
 C. Removes the validation set\
 D. Deletes model weights

---

 ### 51\. Why are random augmentations particularly important in self-supervised learning?

 A. They guarantee 100% pretext accuracy\
 B. They can improve diversity and robustness of learned representations\
 C. They remove the need for a backbone\
 D. They make all tasks trivial

---

 ### 52\. Why can an augmentation be harmful to a pretext task?

 A. It can destroy or conflict with the signal the model is supposed to predict\
 B. It always increases GPU memory\
 C. It always reduces the dataset size\
 D. It prevents PyTorch from running

---

 ### 53. For a four-way rotation prediction task, why would random rotation augmentation be problematic?

 A. It increases image resolution\
 B. It changes the rotation signal that constitutes the pretext label\
 C. It prevents convolution\
 D. It removes colour information

---

 ## Section F — Model Architecture

 ### 54\. What is the purpose of the `ConvBackbone`?

 A. Produce reusable feature representations\
 B. Perform validation splitting\
 C. Generate dataset labels\
 D. Store training metrics

---

 ### 55\. What is the output channel count of `conv1`?

```
nn.Conv2d(input_channels, 32, 3, padding=1)
```

 A. 3\
 B. 16\
 C. 32\
 D. 64

---

 ### 56\. What is the output channel count of `conv4`?

```
self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
```

 A. 64\
 B. 128\
 C. 256\
 D. 512

---

 ### 57\. What does:

```
nn.MaxPool2d(2, 2)
```

 generally do to spatial dimensions?

 A. Doubles them\
 B. Halves them when stride is 2\
 C. Leaves them unchanged\
 D. Converts them into vectors

---

 ### 58\. Which operation reduces the final spatial feature maps to `1 × 1`?

 A. `nn.Linear`\
 B. `nn.Dropout`\
 C. `nn.AdaptiveAvgPool2d(1)`\
 D. `nn.Conv2d`

---

 ### 59\. Why is adaptive average pooling useful here?

 A. It produces a fixed spatial output size before the fully connected layers\
 B. It creates labels\
 C. It randomly crops images\
 D. It computes accuracy

---

 ### 60\. What is the number of inputs to `fc1`?

```
self.fc1 = nn.Linear(256, 256)
```

 A. 32\
 B. 128\
 C. 256\
 D. 512

---

 ### 61\. What is the output dimensionality of `fc2`?

```
self.fc2 = nn.Linear(256, 128)
```

 A. 32\
 B. 64\
 C. 128\
 D. 256

---

 ### 62\. What is the final backbone output size in this notebook?

```
backbone_output_size = 64
```

 A. 32\
 B. 64\
 C. 128\
 D. 256

---

 ### 63\. What is the purpose of dropout in the backbone?

 A. Increase the number of parameters\
 B. Provide regularisation by randomly dropping activations during training\
 C. Convert images to tensors\
 D. Compute accuracy

---

 ### 64\. Why is the backbone separated from the classification head?

 A. To allow the learned representation to be reused for different downstream tasks\
 B. To avoid using PyTorch\
 C. To eliminate the need for training\
 D. To ensure all tasks have identical output classes

---

 ## Section G — Classification Head

 ### 65\. What is the input size of the target classification head?

 A. 3\
 B. 32\
 C. 64\
 D. 128

---

 ### 66\. What is the projection size of the target classification head?

```
projection_size = 32
```

 A. 16\
 B. 32\
 C. 64\
 D. 128

---

 ### 67\. What determines `num_classes` in the target classification head?

 A. Number of source images\
 B. Number of target classes\
 C. Number of convolutional layers\
 D. Batch size

---

 ### 68\. What does the classification head return?

 A. Class logits\
 B. Class names\
 C. Normalised images\
 D. Feature maps only

---

 ### 69\. Why does the code use:

```
x = self.classifier(F.relu(x))
```

 A. It applies ReLU before the final classifier\
 B. It applies softmax before the final classifier\
 C. It converts logits into labels\
 D. It performs backpropagation

---

 ### 70\. Why does the final classifier not explicitly apply softmax?

 A. `CrossEntropyLoss` expects logits and internally handles the appropriate normalization\
 B. Softmax is never useful\
 C. Accuracy requires raw images\
 D. ReLU automatically performs softmax

---

 ## Section H — Accuracy, Metrics, and Visualisation

 ### 71\. What does `top1_acc()` calculate?

```
(pred.argmax(axis=1) == y).float().mean().item()
```

 A. Mean squared error\
 B. Percentage of examples whose highest-logit class matches the target\
 C. Top-5 accuracy\
 D. Validation loss

---

 ### 72\. What does `argmax(axis=1)` select?

 A. The class with the highest predicted logit for each example\
 B. The image with the highest loss\
 C. The batch with the highest accuracy\
 D. The feature with the lowest value

---

 ### 73\. Why does `.item()` appear at the end of `top1_acc()`?

 A. To convert a single tensor value into a Python scalar\
 B. To calculate gradients\
 C. To move the model to GPU\
 D. To reshape the batch

---

 ### 74\. What does `TrainingMetrics.log_train()` record?

 A. Batch-level training loss and accuracy\
 B. Only final validation accuracy\
 C. Dataset labels\
 D. Model architecture

---

 ### 75\. What does `TrainingMetrics.log_val()` record?

 A. Validation metrics at the end of an epoch\
 B. Every individual image\
 C. Optimizer gradients\
 D. Pretext labels

---

 ### 76\. What does `best_val_acc` return?

```
return np.max(self.val_acc)
```

 A. Lowest validation accuracy\
 B. Mean training accuracy\
 C. Highest recorded validation accuracy\
 D. Final training accuracy

---

 ### 77\. What does `best_val_loss` return?

 A. Maximum validation loss\
 B. Minimum recorded validation loss\
 C. Average training loss\
 D. First validation loss

---

 ### 78\. Why does the plotting code use an exponentially weighted mean for training curves?

 A. To smooth noisy batch-level measurements\
 B. To increase the model's accuracy\
 C. To remove validation data\
 D. To calculate gradients

---

 ### 79\. What does this formatting accomplish?

```
acc_ax.yaxis.set_major_formatter(
    mpl.ticker.PercentFormatter(xmax=1.0)
)
```

 A. Converts accuracy values such as `0.75` to a percentage display such as `75%`\
 B. Converts accuracy to a loss\
 C. Clips accuracy to zero\
 D. Converts labels into strings

---

 ### 80\. What is the purpose of `epoch_steps`?

 A. Record the training-step positions at which validation metrics are logged\
 B. Store optimizer learning rates\
 C. Store class indices\
 D. Store image dimensions

---

 # Section I — Supervised Training Loop

 ### 81\. Which loss function is used in `train_supervised()`?

 A. `nn.MSELoss()`\
 B. `nn.CrossEntropyLoss()`\
 C. `nn.L1Loss()`\
 D. `nn.BCELoss()`

---

 ### 82\. Which optimizer is used?

 A. SGD\
 B. RMSprop\
 C. AdamW\
 D. Adagrad

---

 ### 83\. What does `l2_reg` control?

 A. Dropout probability\
 B. Weight decay/L2 regularisation strength\
 C. Number of classes\
 D. Batch size

---

 ### 84\. What is the purpose of:

```
opt.zero_grad()
```

 A. Clear gradients from the previous optimisation step\
 B. Reset model weights\
 C. Reset the dataset\
 D. Set the loss to zero permanently

---

 ### 85\. What happens in:

```
pred = model(x)
```

 A. Backpropagation\
 B. Forward propagation\
 C. Dataset splitting\
 D. Optimizer reset

---

 ### 86\. What does:

```
batch_loss.backward()
```

 do?

 A. Computes gradients of the loss with respect to trainable parameters\
 B. Updates the parameters directly\
 C. Evaluates validation data\
 D. Clears gradients

---

 ### 87\. What does:

```
opt.step()
```

 do?

 A. Computes validation accuracy\
 B. Updates model parameters using the computed gradients\
 C. Clears the dataset\
 D. Creates a new model

---

 ### 88\. Why is `model.train()` called at the beginning of each training epoch?

 A. To enable training behaviour such as dropout\
 B. To create a training dataset\
 C. To compute validation metrics\
 D. To freeze all parameters

---

 ### 89\. Why is `model.eval()` called during validation?

 A. To activate training-time dropout\
 B. To switch the model into evaluation behaviour\
 C. To update model weights\
 D. To shuffle validation data

---

 ### 90\. What is the purpose of:

```
with torch.no_grad():
```

 during evaluation?

 A. Disable gradient tracking during inference\
 B. Enable dropout\
 C. Increase the learning rate\
 D. Update model parameters

---

 ### 91\. When is validation performed in `train_supervised()`?

 A. After every batch\
 B. After every epoch\
 C. Only before training\
 D. Only after training finishes

---

 ### 92\. What does `evaluate_model()` return?

 A. Training loss and training accuracy\
 B. Validation loss and validation accuracy\
 C. Model weights\
 D. Dataset labels

---

 ### 93\. How are batch validation losses aggregated?

```
val_loss = np.mean(batch_val_losses)
```

 A. Maximum\
 B. Minimum\
 C. Mean\
 D. Sum

---

 ### 94\. What does the `seed` parameter in `train_supervised()` affect?

 A. It can make PyTorch's random initialisation/relevant stochastic behaviour reproducible\
 B. It determines the number of classes\
 C. It changes the input image size\
 D. It changes the loss function

---

 ## Section J — Early Stopping and Training Behaviour

 ### 95\. What is the purpose of early stopping?

 A. Stop training when validation performance indicates convergence or deterioration\
 B. Guarantee 100% training accuracy\
 C. Increase the dataset size\
 D. Remove the validation set

---

 ### 96\. In the supplied code, early stopping is only considered when:

 A. `e > 0`\
 B. `e > 1`\
 C. `e > 4`\
 D. `e > 20`

---

 ### 97\. What does this calculate?

```
recent_val_loss = np.mean(metrics.val_loss[-4:-1])
```

 A. Mean of the current and previous three losses\
 B. Mean of the three validation losses immediately preceding the current loss\
 C. Minimum validation loss\
 D. Training loss

---

 ### 98\. According to the actual code, training stops when:

```
metrics.val_loss[-1] > recent_val_loss
```

 A. Current validation loss is greater than the mean of recent previous validation losses\
 B. Current training accuracy reaches 100%\
 C. Validation loss reaches exactly zero\
 D. Training loss becomes larger than one

---

 ### 99\. Why might the baseline model overfit quickly?

 A. The labelled target dataset is relatively small\
 B. The dataset is infinite\
 C. The model contains no trainable parameters\
 D. Validation data is used for training

---

 ### 100\. What is the purpose of L2 regularisation in this context?

 A. Encourage smaller model weights and help reduce overfitting\
 B. Increase image resolution\
 C. Generate pseudo-labels\
 D. Increase the number of classes

---

 # Section K — Pretext Tasks

 ### 101\. Which of the following is a valid pretext task suggested by the notebook?

 A. Predict a randomly generated number unrelated to the image\
 B. Predict the rotation angle applied to an image\
 C. Memorise the filename\
 D. Predict the batch index

---

 ### 102\. In four-way rotation prediction, a model might classify:

 A. 0°, 90°, 180°, and 270°\
 B. 0° and 360° only\
 C. Every possible angle continuously\
 D. Only horizontal and vertical flips

---

 ### 103\. Which task involves predicting how two patches are spatially related?

 A. Relative-position prediction\
 B. Standard classification\
 C. Random erasing\
 D. Batch normalization

---

 ### 104\. What is the jigsaw pretext task?

 A. Predict how shuffled image patches should be arranged\
 B. Predict the class of a CIFAR image using labels\
 C. Remove image noise\
 D. Predict batch size

---

 ### 105\. A same-image/different-image task could ask the model to:

 A. Predict whether two augmented crops originated from the same original image\
 B. Predict the original dataset filename\
 C. Predict the GPU being used\
 D. Predict the learning rate

---

 ### 106\. Why might simple denoising be a poor pretext task?

 A. The model may solve it using primarily low-level features rather than useful semantic representations\
 B. Denoising cannot be implemented in PyTorch\
 C. It always requires labels\
 D. It cannot use convolutional networks

---

 ### 107\. What is an important characteristic of a good pretext task?

 A. It should encourage representations useful for the downstream task\
 B. It must always achieve 100% accuracy\
 C. It must use the original class labels\
 D. It should be completely unrelated to the images

---

 ### 108\. Why is extremely high pretext-task accuracy potentially suspicious?

 A. It may indicate that the pretext task is too easy and is not forcing useful representations to be learned\
 B. It always proves the model is broken\
 C. It means the target labels are wrong\
 D. It means the GPU is too fast

---

 ### 109\. What should ideally happen during useful pretext training?

 A. The model learns a meaningful representation and the pretext loss converges reasonably\
 B. The model never learns anything\
 C. The pretext accuracy must be exactly 100%\
 D. The loss must increase indefinitely

---

 ### 110\. Why can augmentations help a self-supervised task?

 A. They increase variation and encourage more robust representations\
 B. They eliminate the need for labels in every possible ML task\
 C. They guarantee transfer learning success\
 D. They reduce every image to the same representation

---

 ## Section L — Contrastive and Representation Learning

 ### 111\. What is the basic intuition behind contrastive learning?

 A. Learn representations by encouraging related examples to have appropriate representations relative to unrelated examples\
 B. Train only on class labels\
 C. Remove all image transformations\
 D. Predict the dataset filename

---

 ### 112\. Which of the following does the notebook mention as a simpler alternative to a full SimCLR loss?

 A. Pairwise contrastive loss\
 B. Binary cross-entropy only\
 C. K-means\
 D. PCA

---

 ### 113\. Which related loss is also mentioned?

 A. Triplet loss\
 B. Poisson loss\
 C. Hinge regression only\
 D. Dice loss

---

 ### 114\. Why does the notebook not recommend implementing the full SimCLR approach for a quick practical?

 A. It is more complex to implement\
 B. It cannot work on images\
 C. It requires labelled CIFAR-100 classes\
 D. It does not use neural networks

---

 ### 115\. What is a key idea in contrastive-style pretext learning?

 A. The model learns relationships between examples or views rather than simply memorising class labels\
 B. Every image receives a human-provided label\
 C. Only the final classification head is trained\
 D. No augmentation can be used

---

 # Section M — Fine-Tuning and Transfer Learning

 ### 116\. After pre-training, which component should be transferred to the target model?

 A. The trained backbone\
 B. The pretext labels\
 C. The source DataLoader only\
 D. The validation metrics

---

 ### 117\. Why is the pretext classification head normally not transferred directly?

 A. Its outputs correspond to the pretext task rather than the target task\
 B. It has no parameters\
 C. It belongs to the dataset\
 D. It is always frozen

---

 ### 118\. What is the purpose of saving/loading a backbone `state_dict`?

 A. Transfer learned model parameters from pre-training to the downstream model\
 B. Store images\
 C. Save labels as text\
 D. Change image resolution

---

 ### 119\. Which training strategy may be useful after self-supervised pre-training?

 A. Fine-tuning with a lower learning rate\
 B. Always using a larger learning rate\
 C. Deleting the backbone\
 D. Removing regularisation in every case

---

 ### 120\. Why might layer freezing be useful?

 A. It can preserve useful pretrained representations while training only selected layers\
 B. It increases the number of classes\
 C. It converts an unlabelled dataset into a labelled one\
 D. It eliminates the need for validation

---

 ### 121\. For a fair comparison, if augmentation is used during fine-tuning, what should also be considered?

 A. Using comparable augmentation during baseline training\
 B. Removing the validation set\
 C. Changing the target classes\
 D. Destroying the baseline model's labels

---

 ### 122\. Which metric is used to compare the final SSL model against the baseline?

 A. Best validation accuracy\
 B. Number of convolutional filters\
 C. Number of source images\
 D. Training batch size

---

 ### 123\. What does this expression represent?

```
improvement = ft_acc - baseline_acc
```

 A. Difference in validation accuracy between fine-tuned SSL and baseline models\
 B. Difference in training loss\
 C. Difference in dataset size\
 D. Difference in number of classes

---

 # Section N — Code Interpretation

 ### 124\. What does this construct do?

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

---

 ### 125\. What is the expected output shape of the classification head for a batch of size `B` and `C` target classes?

 A. `(B, 3)`\
 B. `(B, 64)`\
 C. `(B, 32)`\
 D. `(B, C)`

---

 ### 126\. If `num_target_classes = 5`, what is the final output shape for a batch of 128 images?

 A. `(5, 128)`\
 B. `(128, 5)`\
 C. `(128, 64)`\
 D. `(5, 64)`

---

 ### 127\. Why can the same backbone architecture be used for different downstream classification tasks?

 A. Its output is a feature representation rather than a task-specific class prediction\
 B. It has no trainable parameters\
 C. It automatically knows every possible class\
 D. It only works with five classes

---

 ### 128\. What does `summary()` from `torchinfo` provide?

 A. A model architecture/parameter summary\
 B. A dataset download\
 C. A confusion matrix\
 D. A pretext task

---

 ### 129\. What does `inspect_dataset()` help verify?

 A. Dataset images and, when available, their labels/classes\
 B. Optimizer gradients\
 C. GPU temperature\
 D. Model parameter updates

---

 ### 130\. What is the purpose of `inspect_batch()`?

 A. Visually inspect transformed images and optionally labels/predictions\
 B. Train the model\
 C. Calculate the learning rate\
 D. Split the dataset

---

 ### 131\. When `predictions` is a 2-D tensor in `inspect_batch()`, what does the code use?

```
pred_idxs = predictions.argmax(dim=1)
```

 A. Highest-scoring predicted class for each sample\
 B. Lowest-scoring class\
 C. Mean class index\
 D. Random class index

---

 ### 132\. What does this calculate?

```
F.softmax(predictions, dim=1).max(dim=1)[0]
```

 A. The highest predicted class probability for each example\
 B. The highest raw pixel value\
 C. The lowest class probability\
 D. The model's training loss

---

 ### 133\. Why does `inspect_batch()` call:

```
images[b].permute([1,2,0])
```

 A. To change a tensor from channel-first `C×H×W` to image-display `H×W×C`\
 B. To flatten the image\
 C. To resize the image\
 D. To rotate the image

---

 ### 134\. What does this expression attempt to do?

```
img = (img_p - img_p.min()) / (img_p.max() - img_p.min())
```

 A. Rescale displayed image values into approximately `[0,1]`\
 B. Calculate cross-entropy\
 C. Add Gaussian noise\
 D. Compute accuracy

---

 # Section O — Practical Design and Debugging

 ### 135\. If a pretext task reaches extremely high accuracy almost immediately, what should you consider?

 A. Making the task harder or increasing augmentation strength\
 B. Immediately deleting the backbone\
 C. Increasing the target classes\
 D. Removing all training data

---

 ### 136\. Why might a pretext task fail to improve downstream performance even if its training loss decreases?

 A. The task may teach features that are not useful for the downstream task\
 B. Decreasing loss always guarantees transfer\
 C. The target task must have identical labels\
 D. Validation accuracy is irrelevant

---

 ### 137\. Why should a pretext task avoid being completely trivial?

 A. A trivial task may not require learning rich representations\
 B. Trivial tasks always cause NaNs\
 C. Trivial tasks cannot be implemented with PyTorch\
 D. Trivial tasks require labels

---

 ### 138\. Which component is particularly important when implementing a custom pretext task?

 A. The data transformation and `collate_fn` that generate valid input-target pairs\
 B. Changing Python's version\
 C. Removing the validation set\
 D. Changing every convolution layer

---

 ### 139\. If a pretext task requires two images as input, what may need to change?

 A. The training loop/model input handling may need to process both images and combine their features\
 B. Only the dataset name\
 C. Only the batch size\
 D. Nothing

---

 ### 140\. Why might a custom pretext task require a different loss function?

 A. Some tasks are regression, pixel prediction, or other objectives rather than integer-label classification\
 B. CrossEntropyLoss works identically for every possible task\
 C. Optimizers require a new loss every epoch\
 D. DataLoaders determine the loss automatically

---

 # Section P — Advanced Integrated Questions

 ### 141\. Consider this pipeline:

 **Unlabelled source → pretext task → pretrained backbone → new target head → target fine-tuning**

 What is the central purpose of the first three stages?

 A. Learn reusable representations without relying on human-provided source labels\
 B. Directly solve the target classification problem\
 C. Increase the number of target labels\
 D. Replace validation with training

---

 ### 142\. Why can a source dataset have classes that do not occur in the target dataset and still be useful?

 A. Visual features such as shapes, textures, edges, and structures can transfer across class boundaries\
 B. The model automatically changes source labels into target labels\
 C. The source labels are secretly retained\
 D. Classes must always be identical for transfer learning

---

 ### 143\. Suppose the target task has five classes but the pretext task has four classes representing rotations. Which statement is correct?

 A. The pretrained backbone can potentially be reused, but the four-class pretext head should be replaced\
 B. The pretext head must output five classes\
 C. The target dataset must be changed to four classes\
 D. The backbone cannot be reused

---

 ### 144\. Why is the validation accuracy of the baseline particularly important?

 A. It provides a reference point for determining whether pre-training helped downstream performance\
 B. It determines the source labels\
 C. It is used as the pretext target\
 D. It determines image resolution

---

 ### 145\. Suppose the baseline achieves 62% validation accuracy and the SSL fine-tuned model achieves 68%. What is the absolute improvement in percentage points?

 A. 4 percentage points\
 B. 6 percentage points\
 C. 10 percentage points\
 D. 110 percentage points

---

 ### 146\. Which comparison would be most informative for the practical's main objective?

 A. Pretrained/fine-tuned target validation accuracy versus supervised-from-scratch target validation accuracy\
 B. Source dataset size versus target dataset size only\
 C. Number of Python imports\
 D. Number of notebook cells

---

 ### 147\. Why should the baseline and fine-tuning hyperparameters generally be comparable when making a direct performance comparison?

 A. To reduce confounding from changes unrelated to self-supervised pre-training\
 B. To ensure both models have identical learned weights\
 C. To make the datasets identical\
 D. To prevent validation

---

 ### 148\. What does “fine-tuning” imply compared with training from scratch?

 A. Starting from pretrained parameters and adapting them to the target task\
 B. Training only on unlabelled data forever\
 C. Never changing any weights\
 D. Training without a target head

---

 ### 149\. Which statement best describes the role of augmentations in this practical?

 A. They can make self-supervised learning more robust, but must be chosen so they do not destroy the pretext signal\
 B. They are guaranteed to improve every model\
 C. They should always include the same transformation being predicted\
 D. They replace the need for a validation set

---

 ### 150\. What is the ultimate experimental question of the practical?

 A. Whether learned representations from an unlabelled source dataset improve performance on a labelled downstream task\
 B. Whether CIFAR-10 has more images than CIFAR-100\
 C. Whether AdamW is faster than SGD\
 D. Whether all pretext tasks reach 100% accuracy

---

 # Answer Key

 | Q | Ans | Q | Ans | Q | Ans | Q | Ans |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | B | 2 | C | 3 | C | 4 | B |
| 5 | B | 6 | B | 7 | C | 8 | B |
| 9 | B | 10 | B | 11 | B | 12 | A |
| 13 | C | 14 | B | 15 | B | 16 | B |
| 17 | A | 18 | B | 19 | C | 20 | C |
| 21 | B | 22 | D | 23 | D | 24 | B |
| 25 | B | 26 | C | 27 | B | 28 | C |
| 29 | A | 30 | B | 31 | B | 32 | B |
| 33 | C | 34 | B | 35 | C | 36 | B |
| 37 | A | 38 | A | 39 | B | 40 | A |
| 41 | B | 42 | A | 43 | B | 44 | C |
| 45 | A | 46 | A | 47 | C | 48 | A |
| 49 | A | 50 | A | 51 | B | 52 | A |
| 53 | B | 54 | A | 55 | C | 56 | C |
| 57 | B | 58 | C | 59 | A | 60 | C |
| 61 | C | 62 | B | 63 | B | 64 | A |
| 65 | C | 66 | B | 67 | B | 68 | A |
| 69 | A | 70 | A | 71 | B | 72 | A |
| 73 | A | 74 | A | 75 | A | 76 | C |
| 77 | B | 78 | A | 79 | A | 80 | A |
| 81 | B | 82 | C | 83 | B | 84 | A |
| 85 | B | 86 | A | 87 | B | 88 | A |
| 89 | B | 90 | A | 91 | B | 92 | B |
| 93 | C | 94 | A | 95 | A | 96 | C |
| 97 | B | 98 | A | 99 | A | 100 | A |
| 101 | B | 102 | A | 103 | A | 104 | A |
| 105 | A | 106 | A | 107 | A | 108 | A |
| 109 | A | 110 | A | 111 | A | 112 | A |
| 113 | A | 114 | A | 115 | A | 116 | A |
| 117 | A | 118 | A | 119 | A | 120 | A |
| 121 | A | 122 | A | 123 | A | 124 | A |
| 125 | D | 126 | B | 127 | A | 128 | A |
| 129 | A | 130 | A | 131 | A | 132 | A |
| 133 | A | 134 | A | 135 | A | 136 | A |
| 137 | A | 138 | A | 139 | A | 140 | A |
| 141 | A | 142 | A | 143 | A | 144 | A |
| 145 | B | 146 | A | 147 | A | 148 | A |
| 149 | A | 150 | A |  |  |  |  |

---

 # High-Yield Concepts to Memorise

 For an exam based on this practical, these are the areas most likely to generate conceptual or code-tracing questions:

 ### 1\. Self-supervised learning

 **Unlabelled source data → automatically constructed pretext signal → representation learning → transfer backbone → downstream target task.**

 The key distinction is that the source data does **not require human-provided labels**.

 ### 2\. Backbone vs. head

 The architecture is conceptually:

```
Input image
    ↓
ConvBackbone
    ↓
Feature representation
    ↓
Task-specific head
    ↓
Task prediction
```

 For the practical:

```
Source images
    ↓
Pretext head
    ↓
Pretext training
    ↓
Pretrained backbone
    ↓
NEW target classification head
    ↓
Fine-tuning
```

 The **backbone is reusable; the task-specific head generally changes**.

 ### 3\. Pretext task quality

 A good pretext task should:

 - be derived from unlabelled data;
- be sufficiently difficult;
- force the model to learn meaningful representations;
- produce features useful for the downstream task;
- avoid being solved using only trivial low-level cues;
- work well with suitable augmentations.

 Importantly, **very high pretext accuracy is not necessarily desirable** if it means the task is trivial.

 ### 4\. Augmentation

 Augmentations provide diversity and robustness, but they must **not destroy the pretext signal**.

 For example, if the task is:

```
Rotate image → predict rotation
```

 then randomly rotating the image again as an augmentation can make the target ambiguous.

 ### 5\. Training loop

 The core supervised loop is:

```
opt.zero_grad()
pred = model(x)
loss = loss_func(pred, y)
loss.backward()
opt.step()
```

 Remember:

 - `zero_grad()` → clear old gradients
- `model(x)` → forward pass
- `loss.backward()` → compute gradients
- `opt.step()` → update parameters

 ### 6\. Training vs evaluation

 Training:

```
model.train()
```

 Evaluation:

```
model.eval()

with torch.no_grad():
    ...
```

 `eval()` is particularly important for modules such as **Dropout**, whose behaviour differs between training and evaluation.

 ### 7\. CrossEntropyLoss

 The classifier outputs **raw logits**, not necessarily probabilities:

```
pred = model(x)
loss = nn.CrossEntropyLoss()(pred, y)
```

 The targets are integer class indices:

```
torch.long
```

 For a batch of 128 examples and 5 classes:

```
pred.shape = [128, 5]
y.shape    = [128]
```

 ### 8\. Accuracy

 The notebook's top-1 accuracy is essentially:

```
pred.argmax(dim=1)
```

 followed by comparison with the true labels.

 For example:

```
Predicted logits:
[1.2, 0.4, 3.8, 0.7, 0.1]

argmax → class 2
```

 ### 9\. Backbone architecture

 The important dimensional progression is:

```
RGB input
   ↓
Conv: 3 → 32
   ↓
Conv: 32 → 64
   ↓
MaxPool
   ↓
Conv: 64 → 128
   ↓
Conv: 128 → 256
   ↓
AdaptiveAvgPool → 1×1
   ↓
Flatten
   ↓
Linear: 256 → 256
   ↓
Linear: 256 → 128
   ↓
Linear: 128 → 64
   ↓
64-dimensional feature representation
```

 ### 10\. Final experimental comparison

 The central comparison is:

```
Supervised training from scratch
              VS
Self-supervised pre-training
              ↓
        Fine-tuning
```

 using the **target validation accuracy** as the primary downstream measure.

 The practical therefore tests whether representations learned from a **large unlabelled related dataset** can improve performance when only a **small amount of labelled target data** is available.
