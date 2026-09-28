 # Advanced Machine Learning — Practical 3

 ## Self-Supervised Learning: Comprehensive MCQ Study Guide

 **How to use this:** Try answering each question before looking at the answer. The explanations are deliberately detailed because they teach the underlying assignment, not just the correct option.

---

 # Part 1 — Big Picture

 ### 1\. What is the main objective of this practical?

 A. Maximise training accuracy on CIFAR-10\
 B. Maximise validation accuracy on a small labelled target dataset using help from a large dataset\
 C. Minimise the size of the CNN\
 D. Generate synthetic images

 **Answer: B**

 **Explanation:**\
 The assignment's main objective is to improve performance on a **small labelled downstream/target dataset** by using a **large source dataset for pre-training**.

 The key challenge is that the source dataset is **unlabelled**.

 The overall idea is:

```
Large unlabelled dataset
          ↓
Self-supervised pre-training
          ↓
Useful feature representation
          ↓
Small labelled dataset
          ↓
Fine-tuning
          ↓
Better validation accuracy
```

---

 ### 2\. What learning paradigm is the practical primarily about?

 A. Reinforcement learning\
 B. Supervised learning\
 C. Self-supervised learning\
 D. Online learning

 **Answer: C**

 **Explanation:**\
 The practical focuses on **self-supervised learning**, specifically using it as **unsupervised pre-training for transfer learning**.

 You create artificial labels from the data itself instead of receiving human-provided labels.

---

 ### 3\. Why is the source dataset called "unlabelled"?

 A. The images are corrupted\
 B. The labels are deliberately unavailable/destroyed\
 C. The dataset contains only one class\
 D. The labels are hidden from the CNN but available during training

 **Answer: B**

 **Explanation:**\
 For CIFAR-10, the notebook explicitly destroys the labels:

```
cifarx.targets = [-1]*len(cifarx.data)
```

 The model therefore only sees the images.

 This forces you to develop a **self-supervised learning signal**.

---

 ### 4\. What is the fundamental problem self-supervised learning is trying to solve here?

 A. How to train without GPUs\
 B. How to learn useful representations when human labels are unavailable\
 C. How to reduce image resolution\
 D. How to eliminate neural networks

 **Answer: B**

 **Explanation:**\
 The source dataset contains lots of images but no usable class labels.

 Self-supervised learning solves this by creating a **pretext task** whose targets can be generated automatically.

---

 # Part 2 — Target vs Source Dataset

 ### 5\. What is the "target task"?

 A. The artificial task used during pre-training\
 B. The final task you ultimately want the model to perform\
 C. Image downloading\
 D. Data augmentation

 **Answer: B**

 **Explanation:**\
 The target task is the actual downstream problem.

 For example, if you select the CIFAR-100 `vehicles` superclass, the target task is to classify images into:

```
bicycle
bus
motorcycle
pickup truck
train
```

 The final validation accuracy is measured on this task.

---

 ### 6\. Which dataset is labelled in the CIFAR version of the practical?

 A. CIFAR-10\
 B. CIFAR-100 subset\
 C. Both datasets\
 D. Neither dataset

 **Answer: B**

 **Explanation:**

 The setup is:

```
CIFAR-100 subset → labelled target dataset
CIFAR-10         → unlabelled source dataset
```

 The CIFAR-10 labels are deliberately discarded.

---

 ### 7\. Why doesn't the source dataset simply use the same classes as the target?

 A. The assignment wants to demonstrate transfer across related but different domains/tasks\
 B. CIFAR-10 contains no images\
 C. CIFAR-100 cannot be classified\
 D. It is impossible to use the same classes in machine learning

 **Answer: A**

 **Explanation:**\
 The interesting part is that the source and target tasks are different.

 For example:

```
Source:
airplane, bird, car, cat, ...

Target:
bicycle, bus, motorcycle, pickup truck, train
```

 Although the classes differ, the images share visual characteristics.

 The source task can therefore encourage the CNN to learn reusable features.

---

 ### 8\. What is the advantage of having related source and target distributions?

 A. The source labels automatically become target labels\
 B. Features learned from the source may transfer to the target\
 C. The target dataset becomes unnecessary\
 D. The CNN doesn't need training

 **Answer: B**

 **Explanation:**\
 This is the basis of transfer learning.

 Even if the classes are different, both datasets contain visual concepts such as:

 - edges
- shapes
- textures
- colours
- object parts
- spatial structure

 Those features can potentially be reused.

---

 ### 9\. In the CIFAR version, approximately how large is the source training dataset?

 A. 100 images\
 B. 500 images\
 C. 5,000 images\
 D. 50,000 images

 **Answer: D**

 **Explanation:**\
 CIFAR-10's training set contains 50,000 images.

 The assignment deliberately removes their labels.

---

 # Part 3 — Transfer Learning

 ### 10\. What is transfer learning in this practical?

 A. Moving a dataset between computers\
 B. Reusing learned model parameters/features from one task for another task\
 C. Transferring labels from CIFAR-10 to CIFAR-100\
 D. Copying Python files

 **Answer: B**

 **Explanation:**\
 The backbone learns useful representations during pre-training.

 You then transfer that backbone to the target task.

```
Pre-training:
image → backbone → pretext head

Fine-tuning:
image → same backbone → target head
```

---

 ### 11\. What part of the model is primarily intended to be transferred?

 A. Dataset\
 B. Validation labels\
 C. Backbone\
 D. Optimiser

 **Answer: C**

 **Explanation:**\
 The assignment explicitly says to transfer the **feature encoder backbone**.

 The pretext-specific head is discarded.

---

 ### 12\. Why is the pretext classification head discarded?

 A. It is broken\
 B. It was designed for the pretext task rather than the target task\
 C. It contains the dataset\
 D. It cannot contain weights

 **Answer: B**

 **Explanation:**\
 Suppose the pretext task is rotation prediction:

```
pretext head:
0°, 90°, 180°, 270°
```

 Your target task might have five completely different classes.

 Therefore the rotation head is no longer appropriate.

 You retain:

```
backbone → useful features
```

 and replace:

```
rotation head
```

 with:

```
target classification head
```

---

 # Part 4 — Baseline

 ### 13\. Why must you train a baseline before doing self-supervised learning?

 A. To make the notebook longer\
 B. To know how well supervised learning alone performs\
 C. To generate CIFAR-10 labels\
 D. To initialise the GPU

 **Answer: B**

 **Explanation:**\
 Without a baseline, you cannot determine whether SSL helped.

 Suppose:

```
Baseline = 60%
SSL + fine-tuning = 67%
```

 Then you have evidence that the SSL approach improved validation performance.

---

 ### 14\. What does the baseline model use?

 A. Only the unlabelled source data\
 B. Only the labelled target training data\
 C. Both datasets simultaneously\
 D. No training data

 **Answer: B**

 **Explanation:**\
 The baseline represents:

 > "What happens if I simply train the CNN on the small labelled target dataset?"

 This gives you the comparison point for SSL.

---

 ### 15\. Why does the baseline tend to overfit?

 A. The target dataset is relatively small\
 B. CIFAR images are too large\
 C. CrossEntropyLoss prevents learning\
 D. GPUs cause overfitting

 **Answer: A**

 **Explanation:**\
 The CNN has substantial capacity compared with the small target dataset.

 It can quickly memorise training examples rather than learning generalisable features.

 This is precisely the situation where pre-training can potentially help.

---

 # Part 5 — Self-Supervised Learning

 ### 16\. What is a pretext task?

 A. The final target classification problem\
 B. An artificially constructed task used to learn useful representations from unlabelled data\
 C. A validation procedure\
 D. A data downloading method

 **Answer: B**

 **Explanation:**\
 A pretext task creates a learning signal from the data itself.

 Examples include:

 - predicting image rotation
- predicting colour transformation intensity
- predicting crop parameters
- predicting relative patch positions
- solving a jigsaw puzzle
- determining whether two crops came from the same image

---

 ### 17. Why is a pretext task necessary?

 A. Neural networks always require human labels\
 B. It provides a training signal despite the source data having no labels\
 C. It reduces the number of images\
 D. It replaces the CNN

 **Answer: B**

 **Explanation:**\
 The source dataset has no human-provided labels.

 The pretext task generates artificial targets.

 For example:

```
Original image
      ↓
Rotate 180°
      ↓
Label = "180°"
```

 The label is known because **you created the transformation**.

---

 ### 18\. Which of the following is a valid pretext task suggested by the assignment?

 A. Predict four-way image rotation\
 B. Predict the original filename\
 C. Predict the computer's IP address\
 D. Predict the dataset download time

 **Answer: A**

 **Explanation:**\
 Rotation prediction is explicitly suggested.

 Other suggested tasks include:

 - colour jitter intensity regression
- crop parameter prediction
- relative position prediction
- jigsaw arrangement
- same-image/different-image prediction

---

 # Part 6 — Rotation Prediction Example

 ### 19\. Suppose you choose four-way rotation prediction. How many output classes does the pretext classifier need?

 A. 2\
 B. 3\
 C. 4\
 D. 5

 **Answer: C**

 **Explanation:**

 You can define:

```
0°   → class 0
90°  → class 1
180° → class 2
270° → class 3
```

 Therefore:

```
num_classes = 4
```

---

 ### 20\. Where does the label come from in rotation prediction?

 A. CIFAR-10's original label\
 B. A human annotator\
 C. The rotation operation you applied\
 D. The validation set

 **Answer: C**

 **Explanation:**\
 This is the defining property of the self-supervised task.

 If you apply a 90° rotation:

```
rotated_image = rotate(image, 90)
label = 1
```

 You automatically know the correct answer.

---

 ### 21\. If you rotate an image by 180°, what should the automatically generated label represent?

 A. The original CIFAR class\
 B. 180° rotation\
 C. The image index\
 D. The image colour

 **Answer: B**

 **Explanation:**\
 The label represents the artificial task—not the original semantic class.

---

 ### 22\. Which statement best describes the training example in rotation prediction?

 A. `(original image, animal class)`\
 B. `(rotated image, rotation angle)`\
 C. `(image filename, rotation angle)`\
 D. `(image, target dataset class)`

 **Answer: B**

 **Explanation:**

 For example:

```
image → rotate 90° → (rotated image, 90°)
```

 This creates a labelled example without requiring human annotation.

---

 # Part 7 — Why the Pretext Task Matters

 ### 23\. Why shouldn't the pretext task be extremely easy?

 A. Easy tasks always crash the GPU\
 B. The network may learn trivial features that don't transfer well\
 C. Easy tasks cannot use CrossEntropyLoss\
 D. The dataset becomes labelled

 **Answer: B**

 **Explanation:**\
 The objective isn't simply to achieve high pretext accuracy.

 The objective is to learn **useful representations**.

 For example, a task based only on a trivial colour difference might achieve 99% accuracy while teaching the backbone little about object structure.

---

 ### 24\. Which is the better goal during self-supervised pre-training?

 A. Maximum possible pretext accuracy regardless of what is learned\
 B. Learn representations useful for downstream tasks\
 C. Memorise the training images\
 D. Make the pretext dataset as small as possible

 **Answer: B**

 **Explanation:**\
 This is one of the most important conceptual points.

 You're not ultimately interested in:

```
"How accurately can I predict rotation?"
```

 You're interested in:

```
"Did predicting rotation cause the CNN to learn useful visual features?"
```

---

 ### 25\. Why might a denoising task be a poor pretext task?

 A. Denoising is impossible\
 B. The model may solve it using low-level features without learning useful semantic representations\
 C. It cannot use images\
 D. It requires labels from CIFAR-100

 **Answer: B**

 **Explanation:**\
 A model could learn simple pixel-level statistics to remove noise.

 But the target classification task may require:

```
object shape
object structure
semantic features
```

 Therefore a successful denoising model doesn't necessarily produce a useful transferable representation.

---

 # Part 8 — Data Augmentation

 ### 26\. Why does the assignment recommend random augmentation?

 A. To increase input diversity and robustness\
 B. To remove the target labels\
 C. To make the GPU slower\
 D. To replace pre-training

 **Answer: A**

 **Explanation:**\
 Augmentation creates varied versions of the same underlying image.

 Examples include:

```
RandomResizedCrop
RandomHorizontalFlip
ColorJitter
RandomGrayscale
RandomErasing
```

 This can encourage the model to learn features that are robust to irrelevant changes.

---

 ### 27\. Which augmentation could conflict with rotation prediction?

 A. Colour jitter\
 B. Random crop\
 C. Random rotation\
 D. Horizontal flip

 **Answer: C**

 **Explanation:**\
 Suppose you deliberately rotate an image 90°.

 Then you randomly rotate it again.

 The final image may no longer correspond to your intended label.

 For example:

```
intended:
90° → label 90°

random additional rotation:
90° + 180° = 270°
```

 Now your input corresponds to 270° but your label says 90°.

 That creates noisy/incorrect training targets.

---

 ### 28\. Which augmentation is generally compatible with rotation prediction?

 A. Another random rotation\
 B. Random crop\
 C. Randomly changing the rotation label\
 D. Replacing the image with another image

 **Answer: B**

 **Explanation:**\
 A crop doesn't directly destroy the rotation signal.

 Similarly, controlled colour jitter can be used.

---

 # Part 9 — The CNN Backbone

 ### 29\. What is the purpose of `ConvBackbone`?

 A. Download CIFAR\
 B. Extract features from images\
 C. Calculate validation accuracy\
 D. Generate labels

 **Answer: B**

 **Explanation:**\
 The backbone is the feature extractor.

 It contains:

```
Conv layers
↓
Pooling
↓
more Conv layers
↓
Adaptive average pooling
↓
Fully connected layers
↓
feature vector
```

---

 ### 30\. What is `backbone_output_size` set to in the notebook?

 A. 4\
 B. 5\
 C. 32\
 D. 64

 **Answer: D**

 **Explanation:**

 The notebook contains:

```
backbone_output_size = 64
```

 Therefore the backbone ultimately produces a 64-dimensional feature representation.

---

 ### 31\. Conceptually, what does the backbone produce?

 A. A dataset\
 B. A feature representation\
 C. A learning rate\
 D. A validation split

 **Answer: B**

 **Explanation:**

 Conceptually:

```
image
  ↓
backbone
  ↓
[64 learned features]
```

 These features are then passed to a task-specific head.

---

 # Part 10 — Classification Head

 ### 32\. What is the purpose of `ClassifierHead`?

 A. Convert learned features into predictions for a particular task\
 B. Load the dataset\
 C. Augment images\
 D. Split the dataset

 **Answer: A**

 **Explanation:**\
 The classification head takes the backbone's features and converts them into class logits.

 The notebook defines:

```
self.projection = nn.Linear(input_size, projection_size)
self.classifier = nn.Linear(projection_size, num_classes)
```

---

 ### 33\. Why is the classification head task-specific?

 A. Different tasks can have different numbers/types of outputs\
 B. Every task uses exactly the same labels\
 C. The backbone cannot produce features\
 D. The head contains the images

 **Answer: A**

 **Explanation:**

 Rotation:

```
4 outputs
```

 Target classification:

```
5 outputs
```

 Therefore the heads need to be different.

---

 ### 34\. If the target task has 5 classes, what should the final classifier layer output?

 A. 2 values\
 B. 4 values\
 C. 5 values\
 D. 64 values

 **Answer: C**

 **Explanation:**\
 Each output corresponds to one target class.

---

 # Part 11 — `nn.Sequential`

 ### 35\. What does this do?

```
baseline_model = nn.Sequential(
    baseline_backbone,
    target_task_head
)
```

 A. Trains the model\
 B. Chains the backbone and head together\
 C. Downloads the data\
 D. Splits the dataset

 **Answer: B**

 **Explanation:**

 It creates:

```
input
 ↓
baseline_backbone
 ↓
target_task_head
 ↓
prediction
```

---

 ### 36\. During fine-tuning, which architecture should you conceptually create?

 A. Target head → pretrained backbone\
 B. Pretrained backbone → target head\
 C. Two target heads\
 D. Dataset → head

 **Answer: B**

 **Explanation:**

```
image
 ↓
pretrained backbone
 ↓
target classification head
 ↓
target predictions
```

---

 # Part 12 — Dataset Classes

 ### 37\. What does `UnlabelledDataset` store?

 A. Images only\
 B. Images and class labels\
 C. Only labels\
 D. Predictions

 **Answer: A**

 **Explanation:**

```
self.images = images
self.name = name
```

 There is no target-label storage.

---

 ### 38\. What does `LabelledDataset` store?

 A. Only images\
 B. Images and targets\
 C. Only targets\
 D. Model weights

 **Answer: B**

 **Explanation:**

```
self.images = images
self.targets = targets
```

 It can also store class names.

---

 ### 39\. Why does `LabelledDataset` check that image and target lengths match?

 A. Every image needs a corresponding target\
 B. To increase GPU speed\
 C. To normalise images\
 D. To create augmentations

 **Answer: A**

 **Explanation:**

```
assert len(images) == len(targets)
```

 If there are 1,000 images, there must be 1,000 corresponding labels.

---

 # Part 13 — Train/Validation Split

 ### 40\. Why does the notebook create a validation split?

 A. To measure generalisation to unseen labelled examples\
 B. To create more training labels\
 C. To increase the source dataset\
 D. To pretrain the model

 **Answer: A**

 **Explanation:**\
 The target dataset is split into:

```
90% → training
10% → validation
```

 The model trains on the training portion and is evaluated on the validation portion.

---

 ### 41\. What is `val_frac = 0.1`?

 A. 1% validation\
 B. 10% validation\
 C. 50% validation\
 D. 90% validation

 **Answer: B**

 **Explanation:**

```
val_frac = 0.1
```

 means approximately 10% of the target data is held out for validation.

---

 ### 42\. Why is `seed=0` used in the split?

 A. To make the random split reproducible\
 B. To improve GPU performance\
 C. To change image colours\
 D. To create labels

 **Answer: A**

 **Explanation:**\
 Without a fixed seed, the random split can change between runs.

 Using a seed allows you to reproduce the same split.

---

 # Part 14 — DataLoaders and Collate Functions

 ### 43\. What is the purpose of a `DataLoader`?

 A. Efficiently provide batches of data during training\
 B. Build CNN layers\
 C. Calculate loss\
 D. Replace the dataset

 **Answer: A**

 **Explanation:**\
 A DataLoader takes your dataset and produces batches.

 For example:

```
Dataset
50,000 images
       ↓
DataLoader
       ↓
batch 1
batch 2
batch 3
...
```

---

 ### 44\. What is the purpose of `collate_labelled`?

 A. Turn a batch of individual examples into tensors suitable for the model\
 B. Train the CNN\
 C. Calculate validation accuracy\
 D. Download images

 **Answer: A**

 **Explanation:**\
 It takes individual:

```
(image, label)
```

 pairs and transforms them into:

```
image_tensor
label_tensor
```

 and moves them to the appropriate device.

---

 ### 45\. Why will you probably need a new collate function for your pretext task?

 A. The pretext task may require transforming images and generating artificial labels\
 B. DataLoaders cannot handle images\
 C. CIFAR has no labels\
 D. CNNs cannot use batches

 **Answer: A**

 **Explanation:**\
 For rotation prediction, your collate function could:

```
receive images
     ↓
randomly rotate each image
     ↓
create rotation labels
     ↓
return images + labels
```

 This is one of the main pieces you need to implement.

---

 # Part 15 — Loss Function

 ### 46\. What loss does the provided supervised training loop use?

 A. Mean squared error\
 B. CrossEntropyLoss\
 C. Triplet loss\
 D. KL divergence

 **Answer: B**

 **Explanation:**

```
loss_func = nn.CrossEntropyLoss()
```

 This is appropriate for standard multi-class classification.

---

 ### 47\. Can you use `CrossEntropyLoss` for the four-way rotation classification task?

 A. Yes\
 B. No\
 C. Only during validation\
 D. Only with CIFAR-100

 **Answer: A**

 **Explanation:**\
 Rotation prediction is a standard four-class classification problem.

 Your targets can be:

```
0
1
2
3
```

 and CrossEntropyLoss is appropriate.

---

 ### 48\. What if your pretext task is predicting a continuous colour-jitter intensity?

 A. You may need a regression loss instead\
 B. You must use CrossEntropyLoss\
 C. You cannot use neural networks\
 D. You must discard the backbone

 **Answer: A**

 **Explanation:**\
 The assignment explicitly points out that different pretext tasks may require different loss functions.

 For regression, you might use something like:

```
nn.MSELoss()
```

 rather than CrossEntropyLoss.

---

 # Part 16 — Optimisation

 ### 49\. Which optimiser does the provided training loop use?

 A. SGD only\
 B. AdamW\
 C. RMSProp\
 D. Adagrad

 **Answer: B**

 **Explanation:**

```
opt = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=l2_reg
)
```

---

 ### 50\. What does `lr` represent?

 A. Label ratio\
 B. Learning rate\
 C. Loss range\
 D. Layer radius

 **Answer: B**

 **Explanation:**\
 The learning rate controls the approximate size of parameter updates during optimisation.

---

 ### 51\. What does `l2_reg` control?

 A. Image resolution\
 B. Weight decay/L2 regularisation\
 C. Number of classes\
 D. Number of GPUs

 **Answer: B**

 **Explanation:**\
 The code passes:

```
weight_decay=l2_reg
```

 This provides regularisation intended to discourage excessively large weights.

---

 # Part 17 — Training Loop

 ### 52\. What happens during the forward pass?

 A. The model receives images and produces predictions\
 B. The model downloads images\
 C. The optimiser changes weights\
 D. The validation set is deleted

 **Answer: A**

 **Explanation:**

```
pred = model(x)
```

 This sends the batch through the network.

---

 ### 53\. What does `batch_loss.backward()` do?

 A. Calculates gradients using backpropagation\
 B. Updates weights directly\
 C. Deletes the loss\
 D. Performs validation

 **Answer: A**

 **Explanation:**\
 `backward()` computes gradients of the loss with respect to the trainable parameters.

 Then:

```
opt.step()
```

 uses those gradients to update the parameters.

---

 ### 54\. What does `opt.zero_grad()` do?

 A. Resets accumulated gradients before calculating new ones\
 B. Resets the model weights\
 C. Deletes the dataset\
 D. Sets accuracy to zero

 **Answer: A**

 **Explanation:**\
 PyTorch accumulates gradients by default.

 Therefore each training iteration generally starts with:

```
opt.zero_grad()
```

---

 ### 55\. What is the usual order of operations?

 A.

```
forward → loss → backward → optimizer step
```

 B.

```
optimizer step → backward → forward → loss
```

 C.

```
loss → optimizer step → dataset download
```

 D.

```
validation → backward → forward
```

 **Answer: A**

 **Explanation:**

```
opt.zero_grad()

pred = model(x)

batch_loss = loss_func(pred, y)

batch_loss.backward()

opt.step()
```

 This is the standard training pattern.

---

 # Part 18 — Evaluation

 ### 56\. Why does `evaluate_model()` use `torch.no_grad()`?

 A. To prevent gradient computation during evaluation\
 B. To train faster\
 C. To generate labels\
 D. To change the learning rate

 **Answer: A**

 **Explanation:**\
 During validation, you don't need gradients because you're not updating the model.

 This reduces unnecessary computation and memory usage.

---

 ### 57\. Why does the model call `model.eval()` during validation?

 A. To put layers such as dropout into evaluation behaviour\
 B. To delete the model\
 C. To train faster\
 D. To generate labels

 **Answer: A**

 **Explanation:**\
 The model behaves differently during training and evaluation for layers such as dropout.

 Training:

```
model.train()
```

 Validation:

```
model.eval()
```

---

 # Part 19 — Accuracy

 ### 58\. What does `top1_acc()` measure?

 A. Percentage of examples where the highest-logit class equals the target\
 B. Number of CNN layers\
 C. Validation loss\
 D. Number of images

 **Answer: A**

 **Explanation:**

```
pred.argmax(axis=1)
```

 selects the predicted class with the largest output score.

 Then it compares it with `y`.

---

 ### 59\. If 80 out of 100 validation images are classified correctly, what is the accuracy?

 A. 8%\
 B. 20%\
 C. 80%\
 D. 100%

 **Answer: C**

 **Explanation:**

```
80 / 100 = 0.80 = 80%
```

---

 # Part 20 — The Actual Assignment Implementation

 ### 60\. What is the first thing you need to decide for Task 1?

 A. Which GPU to buy\
 B. Which pretext task to implement\
 C. Which validation metric to delete\
 D. Which target labels to destroy

 **Answer: B**

 **Explanation:**\
 The assignment explicitly leaves the pretext task choice to you.

---

 ### 61\. After choosing a pretext task, what must you implement?

 A. A way to generate the corresponding inputs and labels\
 B. A new operating system\
 C. A new dataset from scratch\
 D. A new GPU

 **Answer: A**

 **Explanation:**\
 This is the main engineering challenge.

 For rotation:

```
source image
     ↓
random rotation
     ↓
rotated image + rotation label
```

---

 ### 62\. Where can the pretext transformations be applied?

 A. Only when downloading the dataset\
 B. Either by creating a new dataset or dynamically at batch time\
 C. Only after fine-tuning\
 D. Only during validation

 **Answer: B**

 **Explanation:**\
 The assignment gives both possibilities.

 You can:

 1. Create a new labelled dataset containing pretext examples, or
2. Generate transformed images and labels dynamically inside a collate function.

 Dynamic generation is often convenient because you get new transformations each epoch.

---

 ### 63\. What should the pretext model contain?

 A. A backbone and a task-specific pretext head\
 B. Only the target head\
 C. Only a DataLoader\
 D. Only a loss function

 **Answer: A**

 **Explanation:**

```
source image
    ↓
backbone
    ↓
pretext head
    ↓
pretext prediction
```

---

 ### 64\. What happens to the pretext head after pre-training?

 A. It becomes the target head automatically\
 B. It is generally discarded/replaced\
 C. It is copied into the validation dataset\
 D. It becomes the optimiser

 **Answer: B**

 **Explanation:**\
 The pretext head solves a temporary artificial task.

 You keep the learned backbone and attach a new target-specific head.

---

 # Part 21 — Fine-Tuning

 ### 65\. What is the goal of fine-tuning?

 A. Adapt the pretrained backbone to the actual target task\
 B. Train on the pretext task again\
 C. Remove all learned features\
 D. Destroy the source dataset

 **Answer: A**

 **Explanation:**\
 Fine-tuning adapts the pretrained representation to the target labels.

---

 ### 66\. What data is used for final fine-tuning?

 A. The unlabelled source dataset\
 B. The labelled target training dataset\
 C. The validation set\
 D. The original CIFAR-10 labels

 **Answer: B**

 **Explanation:**\
 The target dataset supplies the actual supervised labels.

---

 ### 67\. What data is used to calculate final validation accuracy?

 A. Target validation set\
 B. Source training set\
 C. Pretext training set\
 D. Target training set only

 **Answer: A**

 **Explanation:**\
 The notebook creates:

```
target_data_train
target_data_val
```

 The latter is used for validation.

---

 ### 68. Why might fine-tuning use a smaller learning rate than pre-training?

 A. The pretrained model already contains useful weights\
 B. Smaller learning rates always give better results\
 C. CNNs cannot use large learning rates\
 D. Validation requires small learning rates

 **Answer: A**

 **Explanation:**\
 The backbone has already learned useful features.

 A very large learning rate might destroy those useful representations.

 The assignment specifically suggests experimenting with a lower fine-tuning learning rate.

---

 ### 69. What is layer freezing?

 A. Preventing selected model parameters from being updated\
 B. Deleting layers\
 C. Turning images into ice\
 D. Freezing the GPU

 **Answer: A**

 **Explanation:**\
 You can potentially freeze early backbone layers while training later layers/head.

 Conceptually:

```
backbone:
[EARLY FEATURES] 🔒
[DEEP FEATURES]  train
[target head]     train
```

 This can preserve pretrained representations.

---

 # Part 22 — Comparing the Models

 ### 70\. What is the key result you need to compare?

 A. Dataset download time\
 B. Final/best target validation accuracy\
 C. Number of Python imports\
 D. Number of convolutional layers

 **Answer: B**

 **Explanation:**\
 The assignment is ultimately asking whether SSL improves downstream performance.

 Therefore the important comparison is:

```
Supervised baseline validation accuracy
vs
SSL + fine-tuned validation accuracy
```

---

 ### 71\. Suppose:

```
Baseline = 58%
SSL = 65%
```

 What is the improvement in percentage points?

 A. 7 percentage points\
 B. 7%\
 C. 12 percentage points\
 D. 65 percentage points

 **Answer: A**

 **Explanation:**

```
65% - 58% = 7 percentage points
```

---

 ### 72\. If SSL achieves lower validation accuracy than the baseline, what does that necessarily mean?

 A. The experiment is invalid\
 B. SSL always doesn't work\
 C. Your chosen pretext task may not have learned sufficiently useful transferable features\
 D. The source dataset must be labelled

 **Answer: C**

 **Explanation:**\
 A self-supervised approach is not guaranteed to improve performance.

 Possible causes include:

 - poor pretext task
- task too easy
- task too difficult
- insufficient pre-training
- inappropriate augmentation
- insufficient training
- optimisation problems
- poor transfer between source and target

 This can itself be an interesting finding to discuss.

---

 # Part 23 — Training Curves

 ### 73\. What does a decreasing training loss generally indicate?

 A. The model is becoming better at fitting its training objective\
 B. The dataset is getting smaller\
 C. The GPU is overheating\
 D. The validation set is increasing

 **Answer: A**

 **Explanation:**\
 The model is reducing its error on the training objective.

 However, low training loss alone doesn't prove good generalisation.

---

 ### 74\. What does a large gap between training and validation performance potentially indicate?

 A. Overfitting\
 B. Perfect generalisation\
 C. No learning\
 D. No dataset

 **Answer: A**

 **Explanation:**

 For example:

```
training accuracy = 99%
validation accuracy = 55%
```

 can indicate that the model has learned the training data too specifically.

---

 ### 75\. What does "stable convergence" mean in the context of the pretext task?

 A. The training process settles toward a relatively stable solution rather than behaving chaotically\
 B. Accuracy must reach exactly 100%\
 C. The model must stop after one epoch\
 D. The loss must become exactly zero

 **Answer: A**

 **Explanation:**\
 The assignment says the pretext model should converge stably.

 It does **not** require perfect accuracy.

---

 # Part 24 — CIFAR vs STL-10

 ### 76\. Which option is generally faster for this practical?

 A. CIFAR\
 B. STL-10\
 C. Both are guaranteed identical\
 D. Neither works

 **Answer: A**

 **Explanation:**\
 CIFAR images are 32×32.

 STL-10 images are 96×96, so they require considerably more computation unless resized.

---

 ### 77\. What is the resolution of original STL-10 images?

 A. 16×16\
 B. 32×32\
 C. 64×64\
 D. 96×96

 **Answer: D**

 **Explanation:**\
 STL-10 contains 96×96 images.

 The notebook optionally resizes them to:

```
(64, 64)
```

---

 ### 78\. How many unlabelled images does STL-10 provide?

 A. 1,000\
 B. 10,000\
 C. 50,000\
 D. 100,000

 **Answer: D**

 **Explanation:**\
 The assignment highlights STL-10's large unlabelled set of approximately **100,000 images**.

---

 # Part 25 — Understanding STL-10

 ### 79\. What is special about STL-10's unlabelled data?

 A. Every image is guaranteed to belong to one of the labelled classes\
 B. It contains images from a similar distribution but not necessarily the target classes\
 C. It contains only noise\
 D. It contains no images

 **Answer: B**

 **Explanation:**\
 This makes it an interesting semi-supervised/self-supervised setting.

 The unlabelled images can contain things such as:

```
houses
lizards
shopping carts
...
```

 while the labelled classes are different.

 The model can nevertheless learn useful visual representations.

---

 # Part 26 — Metrics Class

 ### 80\. What is the purpose of `TrainingMetrics`?

 A. Track training and validation statistics\
 B. Train the CNN by itself\
 C. Download CIFAR\
 D. Generate rotations

 **Answer: A**

 **Explanation:**\
 It stores:

```
train_loss
val_loss
train_acc
val_acc
unsup_loss
```

 and supports plotting.

---

 ### 81\. What does `best_val_acc` return?

 A. Highest recorded validation accuracy\
 B. Lowest training loss\
 C. First validation accuracy\
 D. Average training accuracy

 **Answer: A**

 **Explanation:**

```
return np.max(self.val_acc)
```

 Therefore it reports the maximum validation accuracy observed during training.

---

 ### 82\. What does `best_val_loss` return?

 A. Maximum validation loss\
 B. Minimum validation loss\
 C. Training accuracy\
 D. Pretext accuracy

 **Answer: B**

 **Explanation:**

```
return np.min(self.val_loss)
```

---

 # Part 27 — Visualisation

 ### 83\. Why does the notebook provide `inspect_dataset()`?

 A. To visually check that the dataset contains sensible images/labels\
 B. To train the CNN\
 C. To calculate gradients\
 D. To download the dataset

 **Answer: A**

 **Explanation:**\
 Visual inspection is extremely useful when building a self-supervised pipeline.

 You want to make sure your transformations are actually doing what you think.

---

 ### 84\. Why is visualising a pretext batch particularly useful?

 A. It can reveal incorrectly implemented transformations or labels\
 B. It replaces model training\
 C. It increases validation accuracy automatically\
 D. It removes the need for a DataLoader

 **Answer: A**

 **Explanation:**\
 Suppose you're doing rotation prediction.

 You should be able to look at a batch and verify:

```
image → clearly rotated 90° → label 90°
image → clearly rotated 180° → label 180°
...
```

 This can catch bugs before you spend hours training.

---

 # Part 28 — Common Implementation Mistakes

 ### 85\. You choose rotation prediction but randomly rotate the image again as augmentation. What is the main problem?

 A. The image becomes larger\
 B. The pretext label may no longer match the final image\
 C. CrossEntropyLoss stops working\
 D. The backbone disappears

 **Answer: B**

---

 ### 86\. You accidentally use the original CIFAR-10 class labels during pre-training. What happens conceptually?

 A. You are no longer following the intended unlabelled self-supervised setup\
 B. The model becomes unsupervised\
 C. The rotation task becomes better\
 D. Nothing changes

 **Answer: A**

 **Explanation:**\
 The point is specifically to learn from **unlabelled** source data.

 Using the original semantic labels would turn this into supervised learning.

---

 ### 87\. You transfer the pretext head instead of replacing it. What is the likely problem?

 A. Its outputs correspond to the wrong task\
 B. It automatically becomes perfect\
 C. It deletes the backbone\
 D. It improves all tasks

 **Answer: A**

 **Explanation:**\
 A rotation head outputs four rotation classes.

 Your target head needs to output the target classes.

---

 ### 88\. Your pretext task reaches 100% accuracy almost immediately. What should you investigate?

 A. Whether the task is too easy\
 B. Whether the GPU is broken\
 C. Whether the target labels disappeared\
 D. Whether CIFAR has images

 **Answer: A**

 **Explanation:**\
 Very high pretext accuracy isn't automatically desirable.

 The model might be solving the task using trivial cues.

 The assignment explicitly warns about this.

---

 ### 89\. Your pretext loss does not decrease at all. What should you investigate?

 A. Data/label generation, model architecture, learning rate and task difficulty\
 B. Only the target validation labels\
 C. The computer's wallpaper\
 D. Nothing

 **Answer: A**

 **Explanation:**\
 If the model cannot learn the pretext task, possible problems include:

 - incorrect labels
- broken transformations
- inappropriate loss
- learning rate
- model architecture
- task being too difficult
- preprocessing issues

---

 # Part 29 — Deep Conceptual Questions

 ### 90\. Why can self-supervised learning be considered "supervised" during training even though the original dataset is unlabelled?

 A. It secretly downloads labels\
 B. The model receives automatically generated targets\
 C. Human annotators label every image\
 D. There is no loss function

 **Answer: B**

 **Explanation:**\
 This distinction is important.

 The **original data is unlabelled**, but you construct artificial labels.

 So training looks like:

```
input + automatically generated target
             ↓
        supervised loss
```

 This is why it's called **self-supervised**.

---

 ### 91\. What is actually being learned during self-supervised pre-training?

 A. Only the artificial labels\
 B. Model parameters that produce useful representations for the pretext task\
 C. The target dataset labels\
 D. The validation accuracy

 **Answer: B**

 **Explanation:**\
 The pretext objective provides pressure for the backbone to learn representations that contain information useful for solving that task.

 The hope is that these representations also transfer to the target task.

---

 ### 92\. What is the key assumption behind self-supervised transfer learning in this practical?

 A. Features useful for the source/pretext task can also be useful for the target task\
 B. Source and target must have identical classes\
 C. Labels are unnecessary for the target task\
 D. The target dataset must be larger

 **Answer: A**

 **Explanation:**\
 This is the fundamental transfer-learning assumption.

 The tasks don't need to be identical.

 They need enough shared structure for learned representations to transfer.

---

 ### 93\. Why can a model trained on CIFAR-10 potentially help classify CIFAR-100 classes it has never seen?

 A. It memorises the CIFAR-100 labels\
 B. It can learn general visual features rather than specific class identities\
 C. CIFAR-10 and CIFAR-100 have identical classes\
 D. The model automatically receives CIFAR-100 labels

 **Answer: B**

 **Explanation:**\
 A CNN can learn features such as:

```
edges
curves
textures
shapes
spatial patterns
object parts
```

 These aren't necessarily tied to one specific class.

---

 # Part 30 — Assessment Questions

 ### 94\. What must you describe in the assessment?

 A. Only your GPU\
 B. Your pretext task, motivation and implementation\
 C. Every line of PyTorch source code\
 D. The history of CNNs

 **Answer: B**

---

 ### 95\. What should your results section contain?

 A. A plot or two comparing SSL fine-tuning with the supervised baseline\
 B. Only screenshots of the dataset\
 C. Only your code\
 D. Only training loss

 **Answer: A**

 **Explanation:**\
 The assignment explicitly asks for comparison with the supervised baseline.

---

 ### 96\. What should your discussion include?

 A. Problems encountered, solutions and insights\
 B. Only your final accuracy\
 C. Only the name of your dataset\
 D. Nothing beyond the code

 **Answer: A**

---

 # Part 31 — Scenario-Based MCQs

 These are closer to the kind of conceptual questions you might encounter in an exam.

 ### 97\. You have 50,000 unlabelled images and 2,000 labelled images. Which approach best matches this practical?

 A. Ignore the 50,000 images\
 B. Use the 50,000 images for self-supervised pre-training and the 2,000 labelled images for fine-tuning\
 C. Use the 2,000 images to label the 50,000 manually\
 D. Use only validation data

 **Answer: B**

---

 ### 98\. You train a rotation predictor successfully, then create a new target classifier. Which weights should you initialise from the rotation model?

 A. Only the final rotation classifier\
 B. The pretrained backbone\
 C. Only the optimiser\
 D. The validation predictions

 **Answer: B**

---

 ### 99\. You have:

```
Baseline = 61%
SSL = 61%
```

 What is the measured improvement?

 A. 0 percentage points\
 B. 1 percentage point\
 C. 61 percentage points\
 D. 100%

 **Answer: A**

 **Explanation:**\
 The SSL model did not improve over the baseline in this experiment.

 That doesn't necessarily mean the idea is useless; it means your particular implementation didn't produce an improvement.

---

 ### 100\. You have:

```
Baseline = 61%
SSL = 59%
```

 Which conclusion is most scientifically appropriate?

 A. Self-supervised learning never works\
 B. The experiment suggests this particular SSL setup did not outperform the baseline\
 C. The baseline must be wrong\
 D. The source dataset is useless for all tasks

 **Answer: B**

 **Explanation:**\
 Don't generalise beyond the experiment.

 Your evidence is only that:

 > Under this dataset, architecture, pretext task, training procedure and hyperparameters, SSL achieved lower validation accuracy.

 You should investigate why.

---

 ### 101\. You have:

```
Pretext training accuracy = 99%
Target validation accuracy = poor
```

 What is one plausible explanation?

 A. The pretext task may have been too easy and learned features that don't transfer well\
 B. High pretext accuracy guarantees good transfer\
 C. Validation accuracy doesn't matter\
 D. The target task must be identical

 **Answer: A**

---

 ### 102\. You have:

```
Pretext accuracy = 60%
Target accuracy improves substantially
```

 Is this necessarily contradictory?

 A. Yes, pretext accuracy must be above 90%\
 B. No, the pretext task can still encourage useful representations even without very high accuracy\
 C. Yes, the model must be discarded\
 D. No, because pretext accuracy is the final metric

 **Answer: B**

 **Explanation:**\
 This is explicitly mentioned in the assignment.

 A moderate pretext accuracy can be perfectly acceptable if the learned representation transfers well.

---

 # Part 32 — The Complete Pipeline

 ### 103\. Which sequence correctly describes the assignment?

 A.

```
Target → source → delete model → validation
```

 B.

```
Source images
→ pretext task
→ pre-train backbone
→ replace pretext head
→ target fine-tuning
→ validation
```

 C.

```
Validation
→ pre-training
→ source labels
```

 D.

```
Target labels
→ destroy labels
→ final model
```

 **Answer: B**

---

 ### 104\. Which sequence best represents the baseline?

 A.

```
Unlabelled source → pretext → target
```

 B.

```
Randomly initialised model → labelled target training data → target validation
```

 C.

```
Source → target labels → validation
```

 D.

```
Target validation → training → pretext
```

 **Answer: B**

---

 ### 105\. Which sequence best represents the SSL experiment?

 A.

```
Unlabelled source
    ↓
pretext task
    ↓
pretrained backbone
    ↓
target head
    ↓
labelled target data
    ↓
validation
```

 B.

```
Target validation
    ↓
source labels
    ↓
pretext
```

 C.

```
Target labels
    ↓
destroy them
    ↓
validation
```

 D.

```
GPU
    ↓
dataset
    ↓
accuracy
```

 **Answer: A**

---

 # Part 33 — Code Interpretation

 ### 106\. What does this line do?

```
baseline_backbone = initialise_backbone()
```

 A. Loads a pretrained backbone\
 B. Creates a new randomly initialised backbone\
 C. Loads CIFAR-10\
 D. Creates validation labels

 **Answer: B**

 **Explanation:**\
 `initialise_backbone()` creates a fresh `ConvBackbone`.

 That's what you want for the supervised baseline.

---

 ### 107\. What does this line do?

```
baseline_target_task_head = ClassifierHead(
    input_size=backbone_output_size,
    projection_size=32,
    num_classes=num_target_classes
)
```

 A. Creates a target classification head\
 B. Creates a rotation dataset\
 C. Creates the source dataset\
 D. Calculates accuracy

 **Answer: A**

---

 ### 108\. What does this line do?

```
baseline_model = nn.Sequential(
    baseline_backbone,
    baseline_target_task_head
)
```

 A. Combines backbone and target head\
 B. Combines two datasets\
 C. Combines training and validation\
 D. Combines source labels

 **Answer: A**

---

 ### 109\. Why is `num_target_classes` used for the final head?

 A. Because the final classifier must output one score per target class\
 B. Because the backbone always has that number of features\
 C. Because CIFAR-10 has that number of classes\
 D. Because validation requires it

 **Answer: A**

---

 # Part 34 — Important Distinctions

 ### 110\. What is the difference between a pretext task and a downstream task?

 A. They are identical\
 B. Pretext is an artificial training objective; downstream is the actual task you care about\
 C. Pretext uses validation data\
 D. Downstream uses no labels

 **Answer: B**

---

 ### 111\. What is the difference between the backbone and classification head?

 A. Backbone extracts representations; head maps them to task-specific outputs\
 B. Backbone stores labels; head stores images\
 C. They are exactly the same thing\
 D. Head performs data loading

 **Answer: A**

---

 ### 112\. What is the difference between pre-training and fine-tuning?

 A. Pre-training learns representations on the pretext task; fine-tuning adapts them to the target task\
 B. They are exactly identical\
 C. Pre-training only evaluates the model\
 D. Fine-tuning destroys the backbone

 **Answer: A**

---

 ### 113\. What is the difference between supervised and self-supervised learning here?

 A. Supervised learning uses human-provided target labels; self-supervised learning generates training targets from the data itself\
 B. Self-supervised learning never uses a loss\
 C. Supervised learning cannot use CNNs\
 D. There is no difference

 **Answer: A**

---

 # Part 35 — Final Mastery Questions

 ### 114\. Which statement best captures the central hypothesis of the assignment?

 A. Larger models always outperform smaller models\
 B. Learning useful representations from abundant unlabelled data can improve learning from a small labelled target dataset\
 C. Labels are never useful\
 D. CNNs should never be fine-tuned

 **Answer: B**

---

 ### 115\. What is ultimately being tested?

 A. Whether your computer can download CIFAR\
 B. Whether your chosen self-supervised representation improves downstream target performance\
 C. Whether rotation prediction reaches 100%\
 D. Whether the source and target classes are identical

 **Answer: B**

---

 ### 116\. Which result would provide evidence that your SSL approach helped?

 A.

```
Baseline: 70%
SSL: 60%
```

 B.

```
Baseline: 70%
SSL: 70%
```

 C.

```
Baseline: 70%
SSL: 76%
```

 D.

```
Baseline: unknown
SSL: 76%
```

 **Answer: C**

 **Explanation:**\
 The SSL model achieved higher target validation accuracy than the baseline.

---

 ### 117\. What is the most important design decision in Task 1?

 A. The name of the Python file\
 B. The choice and implementation of the pretext task\
 C. The colour of the plots\
 D. The number of Markdown cells

 **Answer: B**

---

 ### 118\. What makes a good pretext task?

 A. It is trivial and requires almost no learning\
 B. It forces the model to learn useful, transferable visual representations\
 C. It uses target labels directly\
 D. It has no measurable objective

 **Answer: B**

---

 ### 119\. Why does the assignment recommend developing on CIFAR before potentially trying STL-10?

 A. CIFAR is generally faster and easier to experiment with\
 B. STL-10 doesn't contain images\
 C. CIFAR has no labels\
 D. STL-10 cannot be used for self-supervised learning

 **Answer: A**

---

 ### 120\. If you had to explain the entire assignment in 20 seconds, which answer is best?

 A. "Train a CNN on CIFAR-100 and plot the loss."

 B. "Use a large unlabelled dataset to create an artificial pretext task, pre-train a CNN backbone on it, transfer that backbone to a small labelled target task, fine-tune it, and compare its validation accuracy against a model trained from scratch."

 C. "Train CIFAR-10 until it reaches 100%."

 D. "Use STL-10 instead of CIFAR-100."

 **Answer: B**

---

 # Final Mental Model

 If you remember only **one diagram**, remember this:

```
                 ┌──────────────────────────┐
                 │ LARGE SOURCE DATASET     │
                 │     NO LABELS            │
                 └────────────┬─────────────┘
                              │
                              ▼
                    Create PRETEXT TASK
                              │
                    e.g. rotation prediction
                              │
                              ▼
                    ┌─────────────────┐
                    │ CNN BACKBONE    │
                    │                 │
                    │ learns features │
                    └────────┬────────┘
                             │
                             ▼
                       PRETRAINED
                        BACKBONE
                             │
                    discard pretext head
                             │
                             ▼
                  attach TARGET HEAD
                             │
                             ▼
                 ┌────────────────────────┐
                 │ SMALL LABELLED TARGET  │
                 │        DATASET         │
                 └───────────┬────────────┘
                             │
                             ▼
                         FINE-TUNE
                             │
                             ▼
                    TARGET VALIDATION
                       ACCURACY
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Compare with        │
                  │ SUPERVISED BASELINE │
                  └─────────────────────┘
```

 ## The 10 things you absolutely need to know

 1. **Self-supervised learning** learns from unlabelled data by constructing its own training signal.
2. The **source dataset** is large and unlabelled.
3. The **target dataset** is smaller and labelled.
4. The **pretext task** is an artificial task created from the unlabelled data.
5. Rotation prediction is one possible pretext task.
6. The **backbone** learns reusable visual features.
7. The **pretext head** is specific to the artificial task and is discarded afterward.
8. A new **target head** is attached for the real classification task.
9. The pretrained model is **fine-tuned** using the small labelled target dataset.
10. The final question is: **does SSL + fine-tuning achieve better target validation accuracy than the supervised baseline?**

 ### One formula-like summary

```
SELF-SUPERVISED TRANSFER LEARNING

Unlabelled data
      +
Artificial pretext labels
      ↓
Pretext training
      ↓
Learned representation
      ↓
Transfer backbone
      ↓
Small labelled target data
      ↓
Fine-tuning
      ↓
Compare with baseline
```

 If you can comfortably answer **Questions 1–120**, you should have a strong conceptual understanding of essentially every important component of the pasted practical.
