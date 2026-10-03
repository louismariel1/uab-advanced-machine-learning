 ## 1\. Assignment summary

 ### Main objective

 Your goal is to **maximize validation accuracy on a small target dataset** and demonstrate that transfer learning can improve performance compared with training only on the target data.

 The practical has three main stages:

 1. **Baseline training**
   - Choose one CIFAR-100 superclass.
   - This gives you **5 target classes**.
   - Split the target data into training and validation sets.
   - Train `BasicCNN` from scratch.
   - Record the best validation accuracy.
2. **Transfer learning**
   - Use the full **CIFAR-10** dataset as the source domain.
   - Pre-train `BasicCNN` to learn general visual features.
   - Save the pretrained model.
3. **Fine-tuning**
   - Load the pretrained CNN.
   - Replace its CIFAR-10 classifier with a new classifier containing **5 output neurons** for the target classes.
   - Fine-tune the model on the small CIFAR-100 target dataset.
   - Compare its validation accuracy against the baseline.

 The expected result is roughly a **5–10 percentage-point improvement**, although the exact result depends on which target superclass you choose.

---

 ## 2\. Dataset structure

 ### Target domain

 The target comes from **CIFAR-100**, which contains:

 - 100 fine-grained classes
- 20 coarse superclasses
- 5 classes per superclass
- 32×32 RGB images
- 500 images per class

 You select one superclass, such as:

```
vehicles
├── bicycle
├── bus
├── motorcycle
├── pickup_truck
└── train
```

 For one superclass:

 - 5 classes
- 500 images/class
- 2,500 total images
- 400 training images/class
- 100 validation images/class

 So the target training set is deliberately small.

 ### Source domain

 The source is **CIFAR-10**:

 - 10 classes
- 50,000 training images
- 5,000 validation images after the 80/20 split
- same 32×32 RGB image format

 The important transfer-learning idea is:

 > **The source and target classes do not need to be identical for useful visual features to transfer.**

---

 # 3\. What the code is doing

 ## Data selection

 The code first maps the five classes in the chosen superclass to their original CIFAR-100 labels:

```
superclass_idxs = set([
    cifar100.class_to_idx[n]
    for n in superclass_names
])
```

 Then it finds all images belonging to those classes:

```
subset_idxs = [
    i for i, idx in enumerate(cifar100.targets)
    if idx in superclass_idxs
]
```

 A `Subset` is created:

```
target_data = torch.utils.data.Subset(cifar100, subset_idxs)
```

 Because the original CIFAR-100 labels might be, for example, `4, 8, 19, 31, 55`, they are remapped to:

```
0, 1, 2, 3, 4
```

 This is necessary because the target classifier has only five outputs.

---

 # 4\. Baseline CNN

 The model is:

```
Input: 3 × 32 × 32
        ↓
Conv2D: 3 → 32
        ↓
ReLU
        ↓
MaxPool
        ↓
Conv2D: 32 → 64
        ↓
ReLU
        ↓
MaxPool
        ↓
Conv2D: 64 → 64
        ↓
ReLU
        ↓
Flatten
        ↓
Linear: 4096 → 256
        ↓
ReLU
        ↓
Linear: 256 → 128
        ↓
ReLU
        ↓
Classifier: 128 → number of classes
```

 For the target task:

```
nn.Linear(128, 5)
```

 For CIFAR-10 pre-training:

```
nn.Linear(128, 10)
```

 ### Important architectural calculation

 The input is:

```
32 × 32
```

 The first max-pooling reduces it to:

```
16 × 16
```

 The second max-pooling reduces it to:

```
8 × 8
```

 There are 64 channels, so the flattened representation is:

```
64 × 8 × 8 = 4096
```

 Hence:

```
self.fc1 = nn.Linear(64 * 8 * 8, 256)
```

---

 # 5\. Baseline training

 The baseline uses:

 - Adam optimizer
- learning rate = `1e-3`
- weight decay = `1e-3`
- batch size = `128`
- 20 epochs
- cross-entropy loss

 The key training sequence is:

```
images
   ↓
forward pass
   ↓
predictions/logits
   ↓
cross-entropy loss
   ↓
backpropagation
   ↓
optimizer.step()
```

 Validation is performed after every epoch.

 The baseline's **best validation accuracy** is saved:

```
baseline_val_acc = np.max(baseline_metrics.val_acc)
```

 This becomes the benchmark that transfer learning must beat.

---

 # 6\. Why the baseline overfits

 The target dataset is small:

```
2,000 training images
```

 for five classes.

 A CNN has enough capacity to memorize aspects of the training set, while there aren't enough target examples to learn highly generalizable representations.

 Consequently, you may see:

```
training accuracy ↑↑
validation accuracy ↑ then plateaus/decreases
```

 This gap is evidence of **overfitting**.

 The practical specifically does **not** ask you to solve this primarily through architecture or regularization tuning. Instead, the intended solution is to exploit information from another dataset.

---

 # 7\. Pre-training

 You now create:

```
pt_model = BasicCNN(num_classes=10)
```

 The model is trained on CIFAR-10.

 The important conceptual point is that the model learns features such as:

 - edges
- textures
- shapes
- color patterns
- local structures
- increasingly complex visual representations

 These features can be useful even when the final target classes are completely different.

 After training:

```
torch.save(pt_model.state_dict(), 'pretrained_model.ckpt')
```

 Only the learned parameter values are saved.

---

 # 8\. Fine-tuning

 The pretrained model originally has:

```
128 → 10
```

 because CIFAR-10 has ten classes.

 But the target task has five classes.

 Therefore:

```
ft_model.classifier = nn.Linear(
    128,
    target_data.num_classes
)
```

 changes the final layer to:

```
128 → 5
```

 This is one of the **most important concepts in the entire practical**.

 You generally **cannot directly use the original CIFAR-10 classifier** for the target task because its output space is different.

 The earlier convolutional/feature-learning layers, however, can be reused.

 ### Fine-tuning learning rate

 The practical uses:

```
lr_ft = 1e-4
```

 rather than:

```
lr = 1e-3
```

 Why?

 The pretrained parameters already contain useful information. A smaller learning rate helps avoid destroying those useful representations through excessively large updates.

---

 # 9\. Baseline vs transfer learning

 Your final comparison is:

```
                 Target data
                     │
             ┌───────┴───────┐
             ↓               ↓
        Train from       Pretrained
          scratch          CNN
             │               │
             │          Replace classifier
             │               │
             │               ↓
             │          Fine-tune
             │               │
             └───────┬───────┘
                     ↓
             Compare validation
                 accuracy
```

 Your report should show plots comparing these two approaches.

 You also need a short explanation of:

 - what you implemented
- any problems encountered
- how you solved them
- what you learned about transfer learning

 And you must submit **both**:

 1. the report
2. the completed `.ipynb`

---

 # 10\. Concepts you should master for an exam

 The notebook tests much more than simply "what is transfer learning?"

 You should understand:

 - source vs target domain
- task transfer
- transfer learning
- pre-training
- fine-tuning
- feature reuse
- classifier replacement
- domain shift
- source/target class mismatch
- overfitting
- validation sets
- CNN architecture
- convolution
- pooling
- ReLU
- flattening
- logits
- cross-entropy
- `argmax`
- batch accuracy
- backpropagation
- Adam
- learning rate
- weight decay/L2 regularization
- train/eval modes
- `torch.no_grad()`
- `state_dict`
- model checkpoints
- `DataLoader`
- `collate_fn`
- normalization
- tensor dimensions
- label remapping
- training vs validation metrics
- epoch vs training step
- why pretraining can help despite different classes
- why the classifier must be changed
- why fine-tuning often uses a smaller learning rate
- freezing vs updating layers
- transferability of low-level vs high-level features

---

 # Comprehensive MCQ Bank

 Below is a deliberately comprehensive set. **Try answering each question before opening the explanation mentally**—the explanations are designed to teach the concept, not just tell you the letter.

 ## A. Assignment and transfer-learning concepts

 ### 1\. What is the primary objective of the practical?

 A. Maximize CIFAR-10 training accuracy\
 B. Minimize the number of CNN layers\
 C. Maximize validation accuracy on a small target dataset\
 D. Train a model without using validation data

 **Answer: C**

 **Explanation:** The practical's central objective is to maximize performance on the small **target** dataset, first with a baseline and then using transfer learning.

---

 ### 2\. Which dataset is the target domain?

 A. CIFAR-10\
 B. CIFAR-100 subset\
 C. ImageNet\
 D. MNIST

 **Answer: B**

 **Explanation:** The target consists of one selected superclass from CIFAR-100.

---

 ### 3\. Which dataset is the source domain?

 A. CIFAR-10\
 B. CIFAR-100\
 C. ImageNet\
 D. The validation subset

 **Answer: A**

 **Explanation:** CIFAR-10 provides the larger source dataset used for pre-training.

---

 ### 4\. Why is the target dataset intentionally small?

 A. To make the CNN unable to run\
 B. To simulate a realistic low-data target-domain scenario\
 C. To eliminate the need for validation\
 D. To make CIFAR-100 equivalent to CIFAR-10

 **Answer: B**

 **Explanation:** Transfer learning is particularly valuable when labelled target-domain data are scarce.

---

 ### 5\. What is the basic idea of transfer learning in this practical?

 A. Train two completely unrelated models\
 B. Train on source data, then reuse learned parameters on the target task\
 C. Remove the convolutional layers\
 D. Use only validation data

 **Answer: B**

 **Explanation:** The model first learns representations from CIFAR-10 and then those representations are adapted to the CIFAR-100 target task.

---

 ### 6\. Do the source and target datasets need exactly the same classes for transfer learning to work?

 A. Yes, always\
 B. No\
 C. Only if Adam is used\
 D. Only if the images are grayscale

 **Answer: B**

 **Explanation:** The practical deliberately demonstrates transfer between datasets with different class sets. Useful visual representations can transfer even when the semantic labels differ.

---

 ### 7\. What is "pre-training"?

 A. Training the model on the target validation set\
 B. Training a model on the source dataset before adapting it to the target task\
 C. Randomly initializing a classifier\
 D. Freezing every parameter permanently

 **Answer: B**

---

 ### 8\. What is fine-tuning?

 A. Training a model from scratch only\
 B. Adapting a pretrained model to the target task\
 C. Removing all learned parameters\
 D. Evaluating without gradients

 **Answer: B**

---

 ### 9\. Why can CIFAR-10 help with CIFAR-100 even though their classes differ?

 A. CIFAR-10 contains exactly the target labels\
 B. Both contain useful visual patterns and structures\
 C. CIFAR-10 has no labels\
 D. CIFAR-100 automatically copies CIFAR-10 weights

 **Answer: B**

 **Explanation:** Early and intermediate CNN features such as edges, textures, shapes and patterns can be useful across related image tasks.

---

 ### 10\. Which statement best describes the transfer-learning strategy?

 A. Transfer the labels from CIFAR-10\
 B. Transfer useful learned model representations/parameters\
 C. Transfer the CIFAR-10 validation images into CIFAR-100\
 D. Merge all classes into one class

 **Answer: B**

---

 ## B. Dataset and label handling

 ### 11\. How many fine-grained classes does CIFAR-100 contain?

 A. 10\
 B. 20\
 C. 50\
 D. 100

 **Answer: D**

---

 ### 12\. How many coarse superclasses are used in the notebook?

 A. 5\
 B. 10\
 C. 20\
 D. 100

 **Answer: C**

---

 ### 13\. How many classes are in each selected superclass?

 A. 2\
 B. 5\
 C. 10\
 D. 20

 **Answer: B**

---

 ### 14\. How many images are there per CIFAR-100 fine-grained class?

 A. 100\
 B. 250\
 C. 500\
 D. 1,000

 **Answer: C**

---

 ### 15\. If one superclass contains five classes, how many total images does its target subset contain?

 A. 500\
 B. 1,000\
 C. 2,000\
 D. 2,500

 **Answer: D**

 **Calculation:**

```
5 classes × 500 images = 2,500 images
```

---

 ### 16\. With an 80/20 split, how many target training images are available?

 A. 500\
 B. 1,000\
 C. 2,000\
 D. 2,500

 **Answer: C**

---

 ### 17\. How many target validation images are available?

 A. 100\
 B. 500\
 C. 1,000\
 D. 2,000

 **Answer: B**

---

 ### 18\. Why is label remapping necessary?

 A. The images must be converted to grayscale\
 B. The selected CIFAR-100 classes may have arbitrary original indices\
 C. PyTorch cannot process integer labels\
 D. Adam requires labels beginning at 1

 **Answer: B**

 **Explanation:** Suppose the selected classes originally have labels `1, 14, 25, 47, 83`. For a five-class classifier, it is convenient to represent them as `0,1,2,3,4`.

---

 ### 19\. What does this dictionary accomplish?

```
target_data.remap = {
    cifar100.class_to_idx[name]: i
    for i, name in enumerate(superclass_names)
}
```

 A. Maps target labels to new consecutive indices\
 B. Converts images to tensors\
 C. Normalizes images\
 D. Shuffles the dataset

 **Answer: A**

---

 ### 20\. Why must the target labels be integers?

 A. Cross-entropy classification expects class-index targets\
 B. CNNs cannot process strings as input images\
 C. Adam only accepts integers\
 D. Max pooling requires integer labels

 **Answer: A**

---

 ## C. Image preprocessing and DataLoader

 ### 21\. What does `transforms.ToTensor()` do?

 A. Converts an image into a PyTorch tensor\
 B. Converts the tensor into a NumPy array\
 C. Applies max pooling\
 D. Calculates the loss

 **Answer: A**

---

 ### 22\. What does this normalization approximately do?

```
transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
```

 A. Subtracts 0.5 and divides by 0.5 for each RGB channel\
 B. Multiplies every pixel by 0.5\
 C. Converts RGB to grayscale\
 D. Randomly changes image brightness

 **Answer: A**

 The transformation is approximately:

 $$
x' = \frac{x-0.5}{0.5}
$$

 per channel.

---

 ### 23\. Why are there three means and three standard deviations?

 A. There are three images per batch\
 B. RGB images have three channels\
 C. The CNN has three convolutional layers\
 D. There are three target classes

 **Answer: B**

---

 ### 24\. What is the purpose of `collate_target()`?

 A. Define the CNN\
 B. Convert a batch of dataset examples into tensors suitable for the model\
 C. Calculate validation accuracy\
 D. Save the model

 **Answer: B**

---

 ### 25\. What does `torch.stack()` do in the collate function?

 A. Combines individual image tensors into a batch tensor\
 B. Adds model layers\
 C. Calculates gradients\
 D. Sorts labels

 **Answer: A**

---

 ### 26\. What is the purpose of `DataLoader`?

 A. It defines convolution filters\
 B. It provides batches of data during training/evaluation\
 C. It computes cross-entropy\
 D. It replaces the optimizer

 **Answer: B**

---

 ### 27. Why is `shuffle=True` useful for the training loader?

 A. It changes the class labels\
 B. It randomizes the order of training examples between epochs\
 C. It changes image resolution\
 D. It freezes the CNN

 **Answer: B**

---

 ### 28\. Should validation data normally be used to update model weights?

 A. Yes\
 B. No\
 C. Only with Adam\
 D. Only after pre-training

 **Answer: B**

 **Explanation:** Validation data are used to measure generalization, not to perform gradient updates.

---

 ## D. CNN architecture

 ### 29\. What is the input shape of each CIFAR image?

 A. 1×28×28\
 B. 3×32×32\
 C. 3×224×224\
 D. 32×32×32

 **Answer: B**

---

 ### 30\. What does `input_channels=3` represent?

 A. Three CNN layers\
 B. Three RGB channels\
 C. Three classes\
 D. Three pooling operations

 **Answer: B**

---

 ### 31\. What does this layer do?

```
nn.Conv2d(3, 32, 3, padding=1)
```

 A. Converts 3 channels to 32 feature maps using 3×3 filters\
 B. Converts 32 channels to 3\
 C. Reduces image size to 3×3\
 D. Creates 32 classes

 **Answer: A**

---

 ### 32\. What is the purpose of convolutional layers?

 A. Learn spatial/visual features\
 B. Perform label remapping\
 C. Split data into training and validation sets\
 D. Calculate accuracy directly

 **Answer: A**

---

 ### 33\. Why is `padding=1` used with a 3×3 convolution?

 A. It approximately preserves spatial dimensions\
 B. It halves the image size\
 C. It doubles the channels\
 D. It removes the RGB channels

 **Answer: A**

---

 ### 34\. What does ReLU do?

 A. $x^2$\
 B. $\max(0,x)$\
 C. $1/x$\
 D. $\log(x)$

 **Answer: B**

---

 ### 35\. What is the purpose of max pooling?

 A. Increase the spatial resolution\
 B. Reduce spatial dimensions while retaining strong activations\
 C. Increase the number of classes\
 D. Normalize labels

 **Answer: B**

---

 ### 36\. After the first max pooling operation, a 32×32 image becomes:

 A. 31×31\
 B. 30×30\
 C. 16×16\
 D. 8×8

 **Answer: C**

---

 ### 37\. After the second max pooling operation, the spatial dimensions become:

 A. 4×4\
 B. 8×8\
 C. 16×16\
 D. 32×32

 **Answer: B**

---

 ### 38\. How many channels exist immediately before flattening?

 A. 3\
 B. 32\
 C. 64\
 D. 128

 **Answer: C**

---

 ### 39\. What is the size of the flattened feature vector?

 A. 64\
 B. 128\
 C. 512\
 D. 4096

 **Answer: D**

 Because:

 $$
64\times8\times8=4096
$$

---

 ### 40\. Why does the model contain this layer?

```
nn.Linear(64 * 8 * 8, 256)
```

 A. To transform the flattened convolutional representation into a 256-dimensional representation\
 B. To perform max pooling\
 C. To produce 64 classes\
 D. To normalize the input

 **Answer: A**

---

 ### 41\. What does the final classifier output represent?

 A. Image pixels\
 B. Class logits\
 C. Gradients\
 D. Feature-map dimensions

 **Answer: B**

---

 ### 42\. During baseline target training, how many outputs does the classifier have?

 A. 3\
 B. 5\
 C. 10\
 D. 100

 **Answer: B**

---

 ### 43\. During CIFAR-10 pre-training, how many outputs does the classifier have?

 A. 5\
 B. 10\
 C. 20\
 D. 100

 **Answer: B**

---

 ## E. Loss, logits, accuracy and optimization

 ### 44\. Which loss function is used?

 A. Mean squared error\
 B. Binary cross entropy\
 C. Cross entropy\
 D. Hinge loss

 **Answer: C**

---

 ### 45\. What does `CrossEntropyLoss` expect as model output?

 A. Raw logits\
 B. Class names\
 C. Already-argmaxed labels\
 D. Images

 **Answer: A**

---

 ### 46\. Why shouldn't you apply `softmax` before `CrossEntropyLoss` in this implementation?

 A. `CrossEntropyLoss` internally handles the appropriate log-softmax operation\
 B. Softmax prevents the CNN from training\
 C. Softmax changes RGB images\
 D. Adam cannot process probabilities

 **Answer: A**

---

 ### 47\. What does this expression calculate?

```
pred.argmax(axis=1)
```

 A. The largest input pixel\
 B. The predicted class index\
 C. The average loss\
 D. The batch size

 **Answer: B**

---

 ### 48\. If the model outputs:

```
[1.2, -0.4, 3.8, 0.7, 0.2]
```

 what class does `argmax` predict?

 A. Class 0\
 B. Class 1\
 C. Class 2\
 D. Class 3

 **Answer: C**

---

 ### 49\. What does this expression measure?

```
(pred.argmax(axis=1) == y).float().mean()
```

 A. Batch accuracy\
 B. Batch loss\
 C. Learning rate\
 D. Gradient magnitude

 **Answer: A**

---

 ### 50\. Why is `.item()` used here?

```
batch_loss.item()
```

 A. To convert a single-element tensor into a Python scalar\
 B. To calculate the gradient\
 C. To reshape the batch\
 D. To reset the optimizer

 **Answer: A**

---

 ### 51\. What is the purpose of:

```
opt.zero_grad()
```

 A. Delete the model\
 B. Clear gradients from the previous optimization step\
 C. Reset the learning rate\
 D. Clear validation data

 **Answer: B**

---

 ### 52\. What does:

```
batch_loss.backward()
```

 do?

 A. Performs backpropagation to compute gradients\
 B. Updates the parameters directly\
 C. Calculates validation accuracy\
 D. Saves the model

 **Answer: A**

---

 ### 53\. What actually updates the model parameters?

 A. `zero_grad()`\
 B. `backward()`\
 C. `opt.step()`\
 D. `argmax()`

 **Answer: C**

---

 ### 54\. What is the normal order during a training iteration?

 A. `step → backward → forward`\
 B. `forward → loss → backward → step`\
 C. `loss → step → forward`\
 D. `backward → forward → step`

 **Answer: B**

---

 ### 55\. What optimizer is used?

 A. SGD\
 B. Adam\
 C. RMSProp\
 D. Adagrad

 **Answer: B**

---

 ### 56\. What is the purpose of the learning rate?

 A. Determine the size of parameter updates\
 B. Determine the number of classes\
 C. Determine the batch size\
 D. Determine the image resolution

 **Answer: A**

---

 ### 57\. What does `weight_decay=1e-3` provide?

 A. L2-style regularization\
 B. Data augmentation\
 C. More training samples\
 D. Larger convolution kernels

 **Answer: A**

---

 ## F. Training and evaluation modes

 ### 58\. Why does the code call:

```
model.train()
```

 before training?

 A. To put the model into training mode\
 B. To start gradient descent automatically\
 C. To load training data\
 D. To save parameters

 **Answer: A**

---

 ### 59\. Why does the code call:

```
model.eval()
```

 during validation?

 A. To put the model into evaluation mode\
 B. To delete gradients\
 C. To reset the optimizer\
 D. To increase the learning rate

 **Answer: A**

---

 ### 60\. Why is `torch.no_grad()` used during validation?

 A. To prevent gradient computation when parameters aren't being updated\
 B. To make the model train faster through larger gradients\
 C. To increase the number of classes\
 D. To normalize the images

 **Answer: A**

---

 ### 61\. Which combination is appropriate for validation?

 A. `model.train()` \+ `backward()`\
 B. `model.eval()` \+ `torch.no_grad()`\
 C. `model.train()` \+ `optimizer.step()`\
 D. `model.eval()` \+ `optimizer.step()`

 **Answer: B**

---

 ### 62\. What is an epoch?

 A. One parameter update\
 B. One complete pass through the training dataset\
 C. One validation example\
 D. One convolution

 **Answer: B**

---

 ### 63\. What is a training step in this notebook?

 A. One batch update\
 B. One complete dataset pass\
 C. One validation epoch\
 D. One model save

 **Answer: A**

---

 ### 64\. With 2,000 training images and batch size 128, approximately how many batches are processed per epoch?

 A. 2\
 B. 8\
 C. 16\
 D. 128

 **Answer: B**

 **Explanation:** $2000/128\approx15.6$, so depending on the DataLoader's `drop_last` setting, the final partial batch means roughly **16 batches**.

---

 ## G. Metrics and visualization

 ### 65\. What does `TrainingMetrics.log_train()` record?

 A. Training loss and training accuracy\
 B. Validation loss only\
 C. Test accuracy only\
 D. Model parameters

 **Answer: A**

---

 ### 66\. How often is validation recorded?

 A. Every batch\
 B. Every epoch\
 C. Every 10 epochs\
 D. Only at the end

 **Answer: B**

---

 ### 67\. Why is validation normally calculated after an epoch rather than after every training batch?

 A. It provides a more useful epoch-level measure of generalization and avoids unnecessary computation\
 B. Validation requires gradients\
 C. Validation changes the model\
 D. Cross entropy only works once per epoch

 **Answer: A**

---

 ### 68\. What does the best validation accuracy represent?

 A. Maximum observed validation accuracy during training\
 B. Final training accuracy\
 C. Average training loss\
 D. Maximum batch size

 **Answer: A**

---

 ### 69\. If training accuracy is 98% but validation accuracy is 55%, what is the most likely issue?

 A. Underfitting\
 B. Overfitting\
 C. Missing labels\
 D. No convolution

 **Answer: B**

---

 ### 70\. If both training and validation accuracy remain very low, which is more suggestive?

 A. Severe overfitting\
 B. Underfitting or optimization problems\
 C. Perfect generalization\
 D. Data leakage

 **Answer: B**

---

 ## H. Pre-training

 ### 71\. Why does `pt_model` have 10 output classes?

 A. The target has 10 classes\
 B. CIFAR-10 has 10 classes\
 C. The CNN always requires 10 outputs\
 D. There are 10 convolution filters

 **Answer: B**

---

 ### 72\. What is the purpose of pre-training?

 A. Learn useful parameters from a larger source dataset\
 B. Directly maximize target validation accuracy without target data\
 C. Replace validation\
 D. Remove the classifier

 **Answer: A**

---

 ### 73\. Why is CIFAR-10 particularly convenient as the source here?

 A. It has the same image resolution and RGB structure as the target\
 B. It has exactly the same classes\
 C. It is smaller than the target\
 D. It contains no labels

 **Answer: A**

---

 ### 74\. Why is source pre-training likely to produce useful features?

 A. The network can learn general visual structures from many labelled examples\
 B. CIFAR-10 labels automatically become CIFAR-100 labels\
 C. The target data are copied into CIFAR-10\
 D. The classifier knows the target classes beforehand

 **Answer: A**

---

 ### 75\. What is saved by:

```
torch.save(pt_model.state_dict(), model_save_name)
```

 A. The model's parameter state dictionary\
 B. Only the training images\
 C. Only validation accuracy\
 D. The optimizer's gradients

 **Answer: A**

---

 ### 76\. Why does the practical recommend saving the pretrained model before fine-tuning?

 A. To preserve a clean pretrained model that can be loaded into a separate fine-tuning model\
 B. Because fine-tuning cannot happen without a file\
 C. Because PyTorch cannot train a model twice\
 D. To increase the batch size

 **Answer: A**

---

 ## I. Fine-tuning

 ### 77\. What is wrong with directly using the pretrained CIFAR-10 classifier for the target task?

 A. It has 10 outputs while the target has 5 classes\
 B. It has too few convolutional layers\
 C. It has no weights\
 D. It cannot process RGB images

 **Answer: A**

---

 ### 78\. What does this accomplish?

```
ft_model.classifier = nn.Linear(128, target_data.num_classes)
```

 A. Replaces the source classifier with a target-specific classifier\
 B. Deletes the convolutional layers\
 C. Doubles the number of channels\
 D. Freezes the model

 **Answer: A**

---

 ### 79\. After replacing the classifier, which part of the model initially contains transferred knowledge?

 A. The pretrained convolutional/feature layers\
 B. The newly initialized classifier\
 C. The validation loader\
 D. The labels

 **Answer: A**

---

 ### 80\. Why might the new classifier need substantial learning?

 A. It is newly initialized for the target classes\
 B. It already perfectly knows the target classes\
 C. It contains CIFAR-100 labels from pre-training\
 D. It is never trained

 **Answer: A**

---

 ### 81\. Why does fine-tuning use `1e-4` instead of `1e-3` in the notebook?

 A. A smaller learning rate can make smaller updates to useful pretrained features\
 B. CIFAR-100 requires exactly `1e-4`\
 C. Adam only works with `1e-4`\
 D. The classifier has five outputs

 **Answer: A**

---

 ### 82\. What would happen if you simply continued using the original 10-class classifier?

 A. The model would produce predictions for the wrong output space\
 B. It would automatically discover the five target classes\
 C. It would become a regression model\
 D. Nothing would change

 **Answer: A**

---

 ### 83\. In the provided fine-tuning code, are all parameters updated?

 A. Yes\
 B. No, all convolutional layers are frozen\
 C. Only the classifier is updated\
 D. Only the first convolution is updated

 **Answer: A**

 **Explanation:** No layers are explicitly frozen. The optimizer receives `ft_model.parameters()`, so the pretrained feature layers and new classifier can all be updated.

---

 ### 84\. What would "freezing" a layer mean?

 A. Preventing its parameters from being updated during fine-tuning\
 B. Deleting its parameters\
 C. Converting it to CPU\
 D. Increasing its learning rate

 **Answer: A**

---

 ### 85\. Why might freezing early layers sometimes be useful?

 A. Early features may already be useful and preserving them can reduce the number of parameters being adapted\
 B. Frozen layers automatically increase the dataset size\
 C. Frozen layers change five classes into ten\
 D. Freezing always guarantees higher accuracy

 **Answer: A**

---

 ## J. Transfer-learning reasoning

 ### 86\. Which features are generally more transferable across image tasks?

 A. Low-level features such as edges and textures\
 B. Exact final-layer class identities\
 C. Target labels\
 D. Dataset filenames

 **Answer: A**

---

 ### 87\. Which component is usually most task-specific?

 A. Final classifier\
 B. First convolution\
 C. Input RGB channels\
 D. Max pooling

 **Answer: A**

 **Explanation:** The final layer maps learned representations to specific class labels, so it must often be replaced for a new task.

---

 ### 88\. Why can transfer learning reduce overfitting?

 A. The model begins with useful representations learned from much more data\
 B. It removes the validation set\
 C. It makes the target dataset larger\
 D. It guarantees zero training error

 **Answer: A**

---

 ### 89\. What does "domain shift" refer to?

 A. Differences between source and target data distributions/tasks\
 B. Changing the learning rate\
 C. Shuffling batches\
 D. Replacing ReLU with pooling

 **Answer: A**

---

 ### 90\. If source and target domains are highly unrelated, what might happen?

 A. Transfer may provide little benefit or even hurt performance\
 B. Transfer must always improve performance\
 C. The classifier automatically becomes correct\
 D. Validation becomes unnecessary

 **Answer: A**

---

 ### 91\. Why might one target superclass benefit more from CIFAR-10 pre-training than another?

 A. Some target visual features may be more similar to features learned from CIFAR-10\
 B. CIFAR-10 changes depending on the target superclass\
 C. The optimizer changes its name\
 D. The target superclass changes the image resolution

 **Answer: A**

---

 ### 92\. What does "negative transfer" mean?

 A. Transfer learning harms target performance\
 B. Transfer improves performance\
 C. The learning rate is negative\
 D. Validation accuracy becomes exactly zero

 **Answer: A**

---

 ## K. Practical implementation/debugging

 ### 93\. Why would this assertion fail?

```
assert torch.cuda.is_available()
```

 A. No CUDA-capable GPU/runtime is available\
 B. CIFAR-100 has too many classes\
 C. The model has too many layers\
 D. The learning rate is too small

 **Answer: A**

---

 ### 94\. What is the purpose of:

```
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

 A. Select the computation device\
 B. Select the loss function\
 C. Select the target classes\
 D. Select the optimizer

 **Answer: A**

---

 ### 95\. Why must the model and tensors generally be on the same device?

 A. PyTorch operations generally require compatible devices\
 B. Otherwise labels change class\
 C. Otherwise convolution becomes pooling\
 D. Otherwise Adam stops existing

 **Answer: A**

---

 ### 96\. What is likely to happen if `x` is on GPU but the model is on CPU?

 A. A device mismatch error\
 B. Automatic perfect transfer\
 C. The model becomes frozen\
 D. Validation accuracy becomes 100%

 **Answer: A**

---

 ### 97\. Why does the code use:

```
.to(device)
```

 on the model?

 A. To move model parameters to the selected device\
 B. To normalize weights\
 C. To change the number of classes\
 D. To calculate gradients

 **Answer: A**

---

 ### 98\. What does this do?

```
ft_model.load_state_dict(
    torch.load('pretrained_model.ckpt')
)
```

 A. Loads the pretrained parameter values into the model\
 B. Loads the target images\
 C. Changes the learning rate\
 D. Freezes every layer

 **Answer: A**

---

 ### 99\. Why must the architecture match the checkpoint when loading a state dictionary?

 A. Parameter names/shapes need to correspond\
 B. PyTorch requires identical batch sizes\
 C. The datasets must have identical labels\
 D. The images must be identical

 **Answer: A**

---

 ### 100\. What would likely happen if you tried to load a 10-class checkpoint into a model whose classifier already has 5 outputs?

 A. There can be a size mismatch for the classifier parameters\
 B. It always works automatically\
 C. The model deletes the classifier\
 D. The CNN becomes a regression network

 **Answer: A**

 **Important:** This is precisely why the notebook first creates a model matching the source architecture, loads the checkpoint, and **then replaces the classifier**.

---

 # L. Higher-level exam questions

 These are the questions I would particularly expect you to be able to answer in an exam or viva.

 ### 101\. Why is it valid to replace only the final classifier after loading the pretrained model?

 A. The earlier layers learn representations that can be reused, while the final layer maps those representations to source-specific classes\
 B. Earlier layers never learn anything\
 C. The final classifier contains all the model's knowledge\
 D. The convolutional layers only store labels

 **Answer: A**

---

 ### 102\. What would be the main disadvantage of training the target CNN from scratch?

 A. It has to learn useful visual representations using very little target data\
 B. It cannot use cross entropy\
 C. It cannot use convolution\
 D. It cannot use validation

 **Answer: A**

---

 ### 103\. Why doesn't pre-training directly guarantee improved target accuracy?

 A. Source and target tasks can differ, causing weak or negative transfer\
 B. Pre-training always produces a worse model\
 C. CIFAR-10 has no images\
 D. Fine-tuning cannot modify weights

 **Answer: A**

---

 ### 104\. Which result provides the strongest evidence that transfer learning helped?

 A. Transfer model has higher target validation accuracy than the baseline\
 B. Transfer model has higher source training accuracy\
 C. Transfer model has lower source loss\
 D. Transfer model has more parameters

 **Answer: A**

---

 ### 105\. Suppose the baseline achieves 55% validation accuracy and fine-tuning achieves 63%. What is the absolute improvement?

 A. 5 percentage points\
 B. 8 percentage points\
 C. 10 percentage points\
 D. 14.5 percentage points

 **Answer: B**

 $$
63\%-55\%=8\text{ percentage points}
$$

---

 ### 106\. Is an 8 percentage-point improvement the same as an 8% relative improvement?

 A. Yes\
 B. No

 **Answer: B**

 The absolute improvement is **8 percentage points**.

 Relative improvement would be:

 $$
\frac{63-55}{55}\times100
\approx14.5\%
$$

---

 ### 107\. Which experiment would directly investigate whether freezing different layers affects transfer?

 A. Freeze different subsets of layers before fine-tuning and compare target validation accuracy\
 B. Change the dataset filenames\
 C. Remove validation data\
 D. Change the class names only

 **Answer: A**

---

 ### 108\. What does increasing network width generally mean?

 A. Increasing the number of feature channels/units\
 B. Increasing image height\
 C. Increasing the number of classes\
 D. Increasing batch size

 **Answer: A**

---

 ### 109\. What does increasing network depth mean?

 A. Adding more layers\
 B. Adding more images\
 C. Increasing image resolution\
 D. Increasing validation percentage

 **Answer: A**

---

 ### 110\. What is the purpose of the optional source-to-target ratio experiment?

 A. Investigate how the amount of source data affects transfer performance\
 B. Change the number of target classes\
 C. Test whether RGB works\
 D. Replace Adam

 **Answer: A**

---

 # M. Code-tracing questions

 ### 111\. What does this line do?

```
baseline_model = BasicCNN(num_classes=target_data.num_classes).to(device)
```

 A. Creates a CNN whose output dimension equals the number of target classes and moves it to the selected device\
 B. Loads CIFAR-10 weights\
 C. Freezes the classifier\
 D. Creates the validation set

 **Answer: A**

---

 ### 112\. What does this line do?

```
pred = baseline_model(x)
```

 A. Performs a forward pass\
 B. Performs backpropagation\
 C. Updates the parameters\
 D. Calculates accuracy automatically

 **Answer: A**

---

 ### 113\. What does this line do?

```
batch_loss = loss_func(pred, y)
```

 A. Calculates the classification loss\
 B. Updates model weights\
 C. Calculates gradients\
 D. Saves the model

 **Answer: A**

---

 ### 114\. What does this line do?

```
opt.step()
```

 A. Updates parameters using the computed gradients\
 B. Computes the loss\
 C. Switches to evaluation mode\
 D. Loads the checkpoint

 **Answer: A**

---

 ### 115\. Why is `opt.zero_grad()` called before the forward/backward cycle?

 A. PyTorch accumulates gradients, so old gradients should be cleared before calculating the new update\
 B. It clears the dataset\
 C. It resets the weights\
 D. It disables training

 **Answer: A**

---

 ### 116\. Why is validation performed under:

```
with torch.no_grad():
```

 A. There is no need to construct a gradient computation graph when only evaluating\
 B. It enables backpropagation\
 C. It increases the learning rate\
 D. It freezes the optimizer permanently

 **Answer: A**

---

 ### 117\. What is the purpose of:

```
np.max(ft_metrics.val_acc)
```

 A. Find the best validation accuracy observed during fine-tuning\
 B. Find maximum training loss\
 C. Find the largest image\
 D. Find the largest class label

 **Answer: A**

---

 ### 118\. Why does the notebook maintain separate metric objects?

```
baseline_metrics
pt_metrics
ft_metrics
```

 A. To independently track baseline, pre-training and fine-tuning performance\
 B. Because PyTorch requires three metric objects\
 C. To create three models\
 D. To create three datasets

 **Answer: A**

---

 ### 119\. What is the purpose of comparing the fine-tuning plot against `baseline_metrics`?

 A. Visually determine whether transfer learning improves target performance relative to training from scratch\
 B. Change the optimizer\
 C. Normalize the data\
 D. Increase the number of target classes

 **Answer: A**

---

 ### 120\. What is the central conclusion the practical is designed to demonstrate?

 A. A model pretrained on a larger, related source dataset can learn useful representations that improve performance on a small target task\
 B. Larger datasets always have exactly the same classes\
 C. CNNs cannot work on small datasets\
 D. Validation accuracy is irrelevant

 **Answer: A**

---

 # Final "master the notebook" checklist

 If you can answer these questions without looking at the notebook, you should be in very good shape.

 ### Data

 - Why is CIFAR-100 the target?
- Why select only one superclass?
- How many target classes are there?
- Why are labels remapped?
- What is the 80/20 split doing?
- Why is the target considered low-data?

 ### CNN

 - What is the input shape?
- What does each convolution do?
- Why use ReLU?
- Why use max pooling?
- Why does the flattened layer have 4096 inputs?
- What is the role of the fully connected layers?
- What does the classifier output?

 ### Training

 - What are logits?
- Why use cross entropy?
- What does `argmax` do?
- What is a batch?
- What is an epoch?
- What does `zero_grad()` do?
- What does `backward()` do?
- What does `step()` do?
- What does Adam do?
- What does weight decay do?

 ### Evaluation

 - Difference between training and validation?
- Why `eval()`?
- Why `no_grad()`?
- What does validation accuracy measure?
- How can you identify overfitting from the curves?

 ### Transfer learning

 - What is the source domain?
- What is the target domain?
- What is pre-training?
- What is fine-tuning?
- Why can features transfer across different classes?
- Why replace the classifier?
- Why does the new classifier have five outputs?
- Why use a lower learning rate?
- What happens if source and target domains are very different?
- What is negative transfer?
- What does freezing a layer mean?

 ### PyTorch implementation

 - What does `state_dict()` contain?
- Why save a checkpoint?
- Why load the checkpoint before replacing the classifier?
- Why must model and data be on the same device?
- What does `DataLoader` do?
- What does `collate_fn` do?
- Why normalize the images?
- Why maintain separate metric objects?

 ### Assignment/report

 Your final submission needs to demonstrate:

```
Baseline
   ↓
Target validation accuracy
   ↓
Pre-training on CIFAR-10
   ↓
Save checkpoint
   ↓
Replace 10-class classifier with 5-class classifier
   ↓
Fine-tune on target
   ↓
Target validation accuracy
   ↓
Compare baseline vs transfer learning
```

 The **single most important conceptual distinction** to remember is:

 > **Pre-training learns reusable representations from the source domain; fine-tuning adapts those representations to the target task.**

 And the **single most important implementation detail** is:

 > **The pretrained CIFAR-10 model has a 10-class output layer, so after loading its weights you replace that classifier with a new 5-class target classifier.**
