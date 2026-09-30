Absolutely. Below is a **presentation/viva-style narrative script** that walks through the notebook from beginning to end, connecting the **code → concept → experiment → interpretation**. It is written so you can essentially read it aloud while presenting the notebook.

 # Advanced Machine Learning Practical 3 — Full Narrative Script

 ## 1\. Introduction

 > “This practical is about **self-supervised learning and transfer learning**.
>
>  The central question is: **Can I use a large amount of unlabelled image data to learn useful visual representations, and then transfer those representations to a target classification problem where I only have a smaller labelled dataset?**
>
>  The practical compares two approaches.
>
>  The first is a conventional supervised baseline, where I train a CNN from scratch using the labelled target data.
>
>  The second uses self-supervised pre-training. I first take an unlabelled source dataset, create an artificial or pretext task from the images themselves, train a CNN to solve that task, keep the learned feature extractor, attach a new classification head, and fine-tune it on the labelled target dataset.
>
>  The final comparison is based primarily on **target validation accuracy**.”

 The overall experiment can therefore be represented as:

```
                BASELINE
                   │
       Small labelled target data
                   │
                   ▼
          CNN trained from scratch
                   │
                   ▼
          Target validation accuracy
                   │
                   │
                   │ compare
                   ▼
          SELF-SUPERVISED PIPELINE
                   │
          Large unlabelled data
                   │
                   ▼
             Pretext task
                   │
                   ▼
          Pretrained backbone
                   │
                   ▼
          New target classifier
                   │
                   ▼
               Fine-tuning
                   │
                   ▼
          Target validation accuracy
```

---

 # 2\. What is self-supervised learning?

 > “Before looking at the code, it is important to understand what self-supervised learning means.
>
>  In ordinary supervised learning, an image comes with a human-provided label. For example, an image might have the label ‘tiger’.
>
>  In self-supervised learning, the source images don't need human-provided labels. Instead, I construct a learning problem from the image itself.”

 For example:

```
Original image
      │
      ▼
Rotate by 90°
      │
      ▼
Rotated image + automatically known rotation label
```

 The label is known because **I performed the transformation**.

 So the important distinction is:

```
Supervised:
image → human label

Self-supervised:
image → automatically generated learning signal
```

 > “Although I call the overall approach self-supervised, once I have generated the artificial labels, I can often use ordinary supervised learning machinery such as cross-entropy loss and backpropagation.
>
>  The important difference is the **origin of the labels**, not necessarily the optimisation algorithm.”

---

 # 3\. The source and target datasets

 > “The notebook provides different dataset configurations, but the CIFAR configuration is the clearest way to understand the experiment.
>
>  The source dataset is CIFAR-10, which is used without its labels.
>
>  The target dataset is a selected subset of CIFAR-100.”

 So conceptually:

```
SOURCE
CIFAR-10
unlabelled
      │
      ▼
self-supervised pre-training

TARGET
CIFAR-100 subset
labelled
      │
      ▼
downstream classification
```

 > “The important point is that the source and target do not need to have identical classes.
>
>  CIFAR-10 may contain classes such as dogs, cats, horses and vehicles, while the target may contain completely different CIFAR-100 classes.
>
>  The reason transfer can still work is that CNNs can learn visual features that are useful across different classes, such as edges, textures, shapes, object parts and spatial structure.”

---

 # 4\. Constructing the target dataset

 > “The notebook then constructs the target task from CIFAR-100.”

 CIFAR-100 has groups of five related classes.

 For example:

```
large carnivores

bear
leopard
lion
tiger
wolf
```

 The notebook selects one such superclass and extracts the corresponding images.

 The original CIFAR-100 labels may not be:

```
0, 1, 2, 3, 4
```

 Instead, they are global CIFAR-100 class indices.

 Therefore the notebook creates a mapping:

```
original CIFAR-100 labels
          ↓
      class_remap
          ↓
0, 1, 2, 3, 4
```

 > “This is important because the classifier needs a clean set of consecutive class indices.”

 For example:

```
Original:

12 → bear
37 → leopard
72 → lion
81 → tiger
95 → wolf

Remapped:

0 → bear
1 → leopard
2 → lion
3 → tiger
4 → wolf
```

---

 # 5\. Creating the unlabelled source dataset

 > “The source dataset is treated differently because its labels aren't supposed to be used.”

 The notebook effectively removes or ignores the labels:

```
cifarx.targets = [-1] * len(cifarx.data)
```

 > “The purpose is to make sure that the pre-training stage does not accidentally use the original CIFAR-10 class labels.”

 The source therefore becomes conceptually:

```
image
image
image
image
...
```

 rather than:

```
image → dog
image → truck
image → airplane
```

 > “This is what allows us to construct our own self-supervised learning problem.”

---

 # 6\. Custom dataset classes

 The notebook defines dataset classes such as:

```
UnlabelledDataset
LabelledDataset
```

 > “These classes control what the DataLoader receives from the dataset.”

 For the unlabelled dataset:

```
__getitem__(index)
```

 returns essentially:

```
image
```

 For the labelled dataset, it returns:

```
image, target
```

 So:

```
UnlabelledDataset:

(image)

LabelledDataset:

(image, label)
```

 The notebook also checks things such as:

```
assert len(images) == len(targets)
```

 > “This ensures that every image has a corresponding target when working with labelled data.”

---

 # 7\. Train/validation split

 > “The target data is then divided into training and validation subsets.”

 For example, if the validation fraction is:

```
test_frac = 0.1
```

 then approximately 10% of the examples are used for validation.

 The purpose is to separate:

```
training data
```

 from:

```
validation data
```

 so that I can measure how well the model generalises to examples that were not used for parameter updates.

 The split uses a seed so that the experiment can be reproducible.

 > “Reproducibility is important because neural networks involve randomness in things such as data splitting, parameter initialisation and augmentation.”

---

 # 8\. DataLoaders

 The notebook then creates PyTorch `DataLoader`s.

 Conceptually:

```
Dataset
   ↓
DataLoader
   ↓
mini-batches
   ↓
model
```

 For training:

```
shuffle=True
```

 > “This randomises the order of training examples between epochs.”

 For validation:

```
shuffle=False
```

 > “There is normally no need to randomly reorder the validation examples.”

---

 # 9\. The collate function

 One particularly useful part of the notebook is the custom `collate_fn`.

 > “The collate function controls how individual dataset examples are assembled into a batch.”

 For example:

```
image_tensor = torch.stack(
    [transform(img) for img in images]
)
```

 This means:

```
individual PIL images
       ↓
transformation
       ↓
individual tensors
       ↓
torch.stack
       ↓
batch tensor
```

 If the batch size is 128, the final tensor might have a shape such as:

```
[128, 3, H, W]
```

 The labels are converted to:

```
torch.long
```

 because classification targets for `CrossEntropyLoss` are integer class indices.

---

 # 10\. Device selection

 The notebook uses:

```
torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

 > “This means that if a CUDA-compatible GPU is available, computation is performed on the GPU. Otherwise it falls back to the CPU.”

 Then tensors and models can be moved using:

```
.to(device)
```

 The important idea is that the model and input tensors need to be on compatible devices.

---

 # 11\. Image preprocessing

 The notebook uses image transformations such as:

```
ToTensor
Normalize
RandomResizedCrop
RandomHorizontalFlip
ColorJitter
RandomGrayscale
RandomErasing
```

 > “These transformations serve two different purposes: preprocessing and augmentation.”

 `ToTensor` converts an image into a PyTorch tensor representation.

 Normalisation changes the numerical scale of the image channels.

 Random augmentations create different versions of the same underlying image.

 For example:

```
original image
      │
 ┌────┼────────┐
 ▼    ▼        ▼
crop  flip   colour change
 │    │        │
 ▼    ▼        ▼
 A    B        C
```

 > “The goal is to encourage the model to learn robust visual representations rather than memorising individual pixel arrangements.”

---

 # 12\. A critical issue with augmentation

 > “However, augmentation has an important interaction with self-supervised learning.”

 Suppose my pretext task is:

```
predict the rotation
```

 I take:

```
original
   ↓
rotate 90°
   ↓
label = 90°
```

 If I then apply another random rotation as an augmentation, the input might no longer correspond to the label.

 For example:

```
intended rotation = 90°

additional random rotation = 180°

actual visible rotation = 270°
```

 But the label still says:

```
90°
```

 > “Therefore, augmentations must preserve the information required to solve the pretext task.”

 This is one of the most important practical considerations in designing the experiment.

---

 # 13\. The CNN backbone

 Now we reach the model.

 The notebook separates the network into:

```
BACKBONE + HEAD
```

 The backbone is:

```
ConvBackbone
```

 > “The backbone is responsible for extracting features from the image.”

 The architecture can be viewed approximately as:

```
RGB image
    ↓
Conv 3 → 32
    ↓
Conv 32 → 64
    ↓
Pooling
    ↓
Conv 64 → 128
    ↓
Conv 128 → 256
    ↓
Adaptive average pooling
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

 The key output is:

```
64 features
```

 So conceptually:

```
image
   ↓
CNN
   ↓
[feature 1, feature 2, ..., feature 64]
```

 > “These 64 values are not directly class labels. They are a learned representation of the image.”

---

 # 14\. Why use a backbone?

 > “The reason for separating the backbone from the classification head is transfer learning.”

 The architecture is:

```
                MODEL
                  │
         ┌────────┴────────┐
         │                 │
      BACKBONE            HEAD
         │                 │
     features          prediction
```

 The head depends on the task.

 For example:

```
Rotation task:

Backbone → 64 features → 4-class head
```

 But later:

```
Target task:

Backbone → 64 features → 5-class head
```

 > “The backbone can therefore be reused while the task-specific head changes.”

---

 # 15\. Adaptive average pooling

 The backbone contains:

```
nn.AdaptiveAvgPool2d(1)
```

 > “This reduces the spatial feature maps to a fixed 1-by-1 spatial representation per channel.”

 So instead of ending with something like:

```
256 × H × W
```

 we obtain:

```
256 × 1 × 1
```

 which can then be flattened and passed through fully connected layers.

 > “This gives the fully connected part a fixed input dimensionality.”

---

 # 16\. Dropout

 The backbone also uses dropout.

 > “Dropout is a regularisation technique. During training, some activations are randomly removed.
>
>  This can reduce the tendency of the model to rely too heavily on particular features and can help reduce overfitting.”

 This also explains why the notebook distinguishes:

```
model.train()
```

 from:

```
model.eval()
```

 because dropout behaves differently during training and evaluation.

---

 # 17\. The classification head

 The classification head is task-specific.

 For example:

```
ClassifierHead(
    input_size=64,
    projection_size=32,
    num_classes=5
)
```

 Conceptually:

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

 > “The final five values are the logits corresponding to the five target classes.”

 The head therefore converts:

```
generic visual representation
```

 into:

```
task-specific prediction
```

---

 # 18\. Why no softmax at the end?

 The final layer produces raw logits.

 The code does not need:

```
softmax(...)
```

 before passing them to:

```
CrossEntropyLoss
```

 > “CrossEntropyLoss is designed to receive raw logits and internally incorporates the appropriate normalisation.”

 The target labels are integer class indices:

```
0
1
2
3
4
```

 and are stored using:

```
torch.long
```

---

 # 19\. The supervised training loop

 The notebook defines a training function such as:

```
train_supervised()
```

 The central loop is:

```
opt.zero_grad()

pred = model(x)

loss = loss_func(pred, y)

loss.backward()

opt.step()
```

 This is worth understanding line by line.

 ### `opt.zero_grad()`

 > “PyTorch accumulates gradients by default, so I clear the gradients from the previous optimisation step.”

 ### `pred = model(x)`

 > “This performs the forward pass.”

 ### `loss = loss_func(pred, y)`

 > “This measures how different the predictions are from the target labels.”

 ### `loss.backward()`

 > “This calculates gradients of the loss with respect to the trainable parameters.”

 ### `opt.step()`

 > “The optimiser then uses those gradients to update the parameters.”

 So:

```
zero_grad
    ↓
forward
    ↓
loss
    ↓
backward
    ↓
parameter update
```

---

 # 20\. AdamW

 The notebook uses:

```
AdamW
```

 as the optimiser.

 > “AdamW is an adaptive gradient-based optimisation algorithm. The W refers to decoupled weight decay, which provides a form of regularisation.”

 The notebook also exposes:

```
l2_reg
```

 or weight decay.

 > “This can discourage excessively large weights and help reduce overfitting.”

---

 # 21\. The baseline experiment

 Now we get to the first actual experiment.

 > “Before testing self-supervised learning, I need a baseline.”

 The baseline starts with randomly initialised parameters:

```
baseline_backbone = initialise_backbone()
```

 Then attaches the target classification head:

```
Random backbone
      +
Target classification head
      ↓
Baseline model
```

 It is trained only on the labelled target dataset.

 Conceptually:

```
CIFAR-100 target training data
            ↓
      random CNN
            ↓
          train
            ↓
   target validation set
            ↓
    baseline accuracy
```

 > “This tells me how well the architecture performs without any self-supervised pre-training.”

 This number is extremely important because it becomes the reference point for the later experiment.

---

 # 22\. Training and validation metrics

 The notebook tracks things such as:

```
training loss
training accuracy
validation loss
validation accuracy
```

 The `TrainingMetrics` class stores these measurements.

 `best_val_acc` represents the highest recorded validation accuracy.

 `best_val_loss` represents the minimum recorded validation loss.

 The training curves can then be plotted.

 > “The plots allow me to see whether the model is learning, overfitting, or converging.”

---

 # 23\. Accuracy calculation

 The notebook calculates top-1 accuracy using the predicted class:

```
pred.argmax(axis=1)
```

 Suppose the model outputs:

```
[1.2, 0.4, 3.8, 0.7, 0.1]
```

 The largest value is:

```
3.8
```

 which corresponds to:

```
class 2
```

 So:

```
argmax
```

 returns the predicted class.

 Accuracy then checks:

```
predicted class == true class
```

 and calculates the proportion of correct predictions.

---

 # 24\. Evaluation mode

 During validation, the notebook uses:

```
model.eval()
```

 and:

```
with torch.no_grad():
```

 > “`eval()` switches modules such as dropout into evaluation behaviour.
>
>  `torch.no_grad()` disables gradient tracking because I am evaluating the model rather than updating its parameters.”

 This reduces unnecessary computation and memory usage.

---

 # 25\. Early stopping

 The notebook also contains early stopping logic.

 > “The purpose of early stopping is to avoid continuing training once validation performance indicates that the model may be starting to deteriorate.”

 The code examines recent validation losses.

 For example:

```
recent_val_loss = np.mean(metrics.val_loss[-4:-1])
```

 This takes the three validation losses immediately preceding the current one.

 Then the code checks whether the latest validation loss has increased relative to this recent average.

 > “The exact stopping rule is defined by the notebook's implementation, so when explaining the experiment I should describe the actual code rather than assuming a generic early-stopping algorithm.”

---

 # 26\. Now the main part: designing the pretext task

 > “After establishing the baseline, the main creative component of the practical is to design a pretext task.”

 A pretext task is an artificial task constructed from the unlabelled images.

 The ideal task should:

```
use unlabelled data
       ↓
require meaningful visual reasoning
       ↓
learn useful representations
       ↓
transfer to downstream classification
```

 The important point is:

 > “I don't ultimately care about the pretext task for its own sake. I care about whether solving it produces useful features.”

---

 # 27\. Rotation prediction

 A simple example is four-way rotation prediction.

 The four classes are:

```
0°
90°
180°
270°
```

 The procedure is:

```
unlabelled image
       ↓
choose rotation
       ↓
rotate image
       ↓
rotation becomes artificial label
       ↓
CNN
       ↓
predict rotation
```

 For example:

```
rotated_image = rotate(image, 90)
label = 1
```

 where:

```
0 → 0°
1 → 90°
2 → 180°
3 → 270°
```

 > “The important point is that no human had to label the image as a dog, cat, truck or anything else. The only label required is the transformation I applied.”

---

 # 28\. Why might rotation prediction learn useful features?

 > “To determine the orientation of an object, the network may need to recognise shapes, object parts and spatial relationships.”

 For example, it may learn that:

```
eyes are above a mouth
legs are below a body
certain shapes have a natural orientation
```

 Therefore:

```
rotation prediction
       ↓
spatial/semantic features
       ↓
potentially useful representation
```

 > “This is the hypothesis being tested by the practical.”

---

 # 29\. Other possible pretext tasks

 The notebook also discusses alternatives.

 One possibility is **relative position prediction**.

 For example:

```
┌─────┬─────┐
│  A  │  B  │
└─────┴─────┘
```

 The model could predict the spatial relationship between two patches.

 Another is a **jigsaw task**:

```
original image
      ↓
split into patches
      ↓
shuffle patches
      ↓
predict arrangement
```

 Another possibility is a same-image/different-image task:

```
crop A ──┐
         ├→ same or different?
crop B ──┘
```

 > “The common principle is that the artificial task should encourage the model to learn useful relationships in the visual data.”

---

 # 30\. What makes a good pretext task?

 > “A good pretext task isn't simply one that is easy to solve.”

 Imagine:

```
Pretext A:
95% accuracy
downstream accuracy = 45%

Pretext B:
70% accuracy
downstream accuracy = 55%
```

 The second task has produced the more useful representation for the target task.

 Therefore:

```
pretext accuracy
       ≠
ultimate objective
```

 The actual objective is:

```
downstream validation performance
```

 > “If the pretext task is too trivial, the network may solve it using simple low-level cues without learning representations that transfer well.”

---

 # 31\. Pretext model

 Once the pretext task has been defined, the model becomes:

```
ConvBackbone
     ↓
64-dimensional features
     ↓
Pretext head
     ↓
Pretext prediction
```

 For rotation prediction:

```
64 features
     ↓
rotation head
     ↓
4 logits
```

 > “The same backbone architecture is used, but the head now corresponds to the pretext task rather than the final target task.”

---

 # 32\. Pretext training

 The source images are passed through the pretext data pipeline.

 Conceptually:

```
CIFAR-10 images
       ↓
construct artificial task
       ↓
image + artificial label
       ↓
pretext model
       ↓
loss
       ↓
backpropagation
       ↓
update backbone
```

 > “The backbone learns parameters that allow it to solve the pretext problem.”

 The pretext task can therefore use the same general training machinery:

```
CrossEntropyLoss
AdamW
backpropagation
```

 provided the task is a classification problem.

---

 # 33\. What should we monitor during pretext training?

 We can monitor:

```
pretext training loss
pretext validation loss
pretext training accuracy
pretext validation accuracy
```

 If there are four equally likely rotation classes, random guessing would give roughly:

```
25%
```

 accuracy.

 > “Therefore, substantially better than random performance indicates that the model is learning something about the artificial task.”

 But again:

 > “High pretext accuracy by itself doesn't prove that the representation will transfer well.”

---

 # 34\. Saving the pretrained backbone

 Once pre-training is complete:

```
Pretext model
      │
      ├── backbone
      │
      └── pretext head
```

 We keep:

```
backbone
```

 and discard the:

```
pretext head
```

 The backbone parameters can be saved using its `state_dict`.

 Conceptually:

```
torch.save(
    pretext_backbone.state_dict(),
    "pretrained_backbone.pt"
)
```

 > “The state dictionary contains the learned parameter values.”

---

 # 35\. Transfer learning

 Now we create a new target model.

```
Pretrained backbone
        +
New target classification head
```

 The pretext head is not reused because its outputs correspond to:

```
0°
90°
180°
270°
```

 while the target task may require:

```
bear
leopard
lion
tiger
wolf
```

 So:

```
Pretext:

Backbone → 4 classes

Target:

Backbone → 5 classes
```

 The backbone is reusable because it represents general visual features.

---

 # 36\. Fine-tuning

 > “The transferred model is then fine-tuned on the labelled target dataset.”

 The pipeline is:

```
CIFAR-100 target image
          ↓
pretrained backbone
          ↓
64 features
          ↓
new target head
          ↓
target class prediction
```

 This is different from training from scratch because the backbone no longer starts from random weights.

 It starts from:

```
features learned from source data
```

 and adapts them to:

```
target classification
```

---

 # 37\. Learning rate during fine-tuning

 > “Because the backbone has already learned useful features, I may use a smaller learning rate during fine-tuning.”

 For example:

```
pre-training:
higher learning rate

fine-tuning:
potentially lower learning rate
```

 The reasoning is that a very aggressive update could destroy useful pretrained representations.

 Another possible strategy is freezing some layers:

```
early backbone layers → frozen
later layers → trainable
target head → trainable
```

 > “Freezing can preserve some pretrained features while allowing later layers and the target head to adapt.”

---

 # 38\. Fair comparison

 This is an important experimental-design issue.

 We now have:

 ### Baseline

```
Random backbone
      ↓
Target head
      ↓
Target training
```

 and:

 ### SSL

```
Pretrained backbone
      ↓
Target head
      ↓
Target fine-tuning
```

 > “To interpret the comparison fairly, I want the experimental conditions to be as comparable as possible.”

 For example:

```
same target dataset
same validation set
same model architecture
comparable augmentation
comparable optimisation setup
```

 The main intended difference should be:

```
random initialisation
          vs
self-supervised pre-training
```

---

 # 39\. The final experiment

 At the end, I obtain something like:

```
Baseline validation accuracy:       XX.X%

SSL fine-tuned validation accuracy: YY.Y%
```

 The improvement is:

```
improvement = ft_acc - baseline_acc
```

 For example, if:

```
baseline = 62%
SSL = 68%
```

 then:

```
improvement = 6 percentage points
```

 > “This is the key downstream result because it directly measures whether self-supervised pre-training helped the target classification task.”

---

 # 40\. Interpreting possible outcomes

 There are several possible outcomes.

 ### Case 1: SSL improves downstream performance

```
baseline = 50%
SSL      = 58%
```

 > “This would provide evidence that the self-supervised pre-training learned representations that were useful for the target task.”

 ### Case 2: SSL performs similarly

```
baseline = 50%
SSL      = 51%
```

 > “The representation may provide little additional benefit under the particular experimental conditions.”

 ### Case 3: SSL performs worse

```
baseline = 50%
SSL      = 45%
```

 > “This could indicate that the pretext task was poorly aligned with the downstream task, that fine-tuning was not effective, that the source and target data were not sufficiently compatible, or that optimisation and augmentation choices need adjustment.”

 > “The important point is that I should analyse the evidence rather than assuming that self-supervised learning must improve performance.”

---

 # 41\. Pretext accuracy versus downstream accuracy

 This distinction is particularly important for explaining the results.

 Suppose:

```
Pretext accuracy = 95%
```

 but:

```
Target accuracy = 45%
```

 That does not necessarily mean the pretext learning was useful.

 Conversely:

```
Pretext accuracy = 70%
```

 and:

```
Target accuracy = 55%
```

 could represent a more useful learned representation.

 The relationship is:

```
Pretext task
      ↓
Representation learning
      ↓
Transfer
      ↓
Downstream performance
```

 > “Therefore, downstream performance is the final test of whether the learned representation was useful for this practical.”

---

 # 42\. Visualising the experiments

 The notebook's metric system allows us to plot learning curves.

 For example:

```
Loss
 │
 │\
 │ \
 │  \____
 │       \____
 └────────────── Epoch
```

 and:

```
Accuracy
 │
 │       ______
 │     /
 │   /
 │__/
 └────────────── Epoch
```

 These plots help answer questions such as:

 - Is the model learning?
- Is validation performance improving?
- Is the model overfitting?
- Did the pretext task converge?
- Does fine-tuning continue to improve?

---

 # 43\. What overfitting looks like

 Suppose:

```
training accuracy ↑↑↑
validation accuracy ↑ then ↓
```

 while training loss continues decreasing.

 > “This can indicate overfitting.”

 The model is becoming increasingly good at the training data but its performance on unseen validation examples is deteriorating.

 Possible responses include:

```
stronger augmentation
regularisation
dropout
weight decay
early stopping
freezing layers
smaller model
```

 The appropriate intervention depends on the observed behaviour.

---

 # 44\. Why the practical uses a small target dataset

 > “The reason the experiment is interesting is that labelled data can be expensive, while unlabelled images can be much easier to obtain.”

 The hypothetical situation is:

```
Lots of unlabelled images
        +
Small amount of labelled data
```

 Self-supervised learning asks:

```
Can I extract useful information from the unlabelled images first?
```

 Then:

```
transfer learned representation
        ↓
use small labelled dataset more effectively
```

 This is the broader motivation behind representation learning.

---

 # 45\. Why source and target classes don't have to match

 > “Another important concept is that the source and target tasks can be different.”

 For example:

```
Source:
dogs, cats, cars, trucks...

Target:
bear, lion, tiger...
```

 The model isn't transferring:

```
dog → tiger
```

 Instead, it is transferring lower-level or intermediate representations.

 For example:

```
edges
textures
shapes
spatial relationships
object structures
```

 Those can potentially be useful for many visual tasks.

---

 # 46\. Contrastive learning

 The notebook also introduces the broader concept of contrastive learning.

 > “Instead of asking the network to predict an explicit transformation such as rotation, contrastive learning can learn representations by comparing different examples or views.”

 For example:

```
same image
   │
 ┌─┴─┐
 ▼   ▼
view A  view B
   \   /
    \ /
 representation
```

 The general idea is to encourage related views to have compatible representations while distinguishing unrelated examples.

 The notebook mentions simpler approaches such as:

```
pairwise contrastive loss
triplet loss
```

 rather than requiring a full implementation of something like SimCLR.

---

 # 47\. Why not immediately implement full SimCLR?

 > “A full contrastive learning system such as SimCLR introduces additional complexity.”

 It can involve:

```
two augmented views
projection heads
positive pairs
negative examples
similarity calculations
temperature parameters
contrastive loss
```

 For a practical focused on understanding the principles, a simpler pretext task can make the experimental pipeline easier to implement and analyse.

---

 # 48\. The three stages to remember

 If I need to remember the whole practical under exam conditions, I can reduce it to three words:

 ## Pretext

```
Unlabelled data
      ↓
Artificial task
      ↓
Learn representation
```

 ## Transfer

```
Pretrained backbone
      ↓
Discard pretext head
      ↓
Attach target head
```

 ## Fine-tune

```
Small labelled target data
      ↓
Adapt model
      ↓
Measure validation performance
```

 So:

```
PRETEXT → TRANSFER → FINE-TUNE
```

---

 # 49\. What the code is really doing

 At a high level, almost the entire notebook can be understood as four major pieces.

 ### Dataset engineering

```
source_data
target_data
splits
DataLoaders
transforms
```

 ### Model engineering

```
ConvBackbone
ClassifierHead
Sequential model
```

 ### Optimisation

```
CrossEntropyLoss
AdamW
backward()
step()
```

 ### Experimentation

```
baseline
pre-training
transfer
fine-tuning
comparison
```

 > “So the notebook isn't just teaching me how to train a CNN. It is teaching me how to construct and evaluate a representation-learning experiment.”

---

 # 50\. How I would explain the baseline in a viva

 If asked:

 > **“Why do you need the baseline?”**

 I would answer:

 > “The baseline tells us how well the target model performs when trained from scratch using only the labelled target data. Without it, we wouldn't know whether self-supervised pre-training actually provided a benefit.”

---

 # 51\. How I would explain the backbone

 If asked:

 > **“What is the backbone?”**

 I would answer:

 > “The backbone is the feature extractor. It transforms an input image into a learned feature representation. It is separated from the task-specific classification head so that the representation can be reused for another task.”

---

 # 52\. How I would explain the pretext task

 If asked:

 > **“What is a pretext task?”**

 I would answer:

 > “It is an artificial learning problem constructed from unlabelled data. The labels or targets are generated automatically from transformations or relationships within the data, allowing the network to learn representations without human-provided labels.”

---

 # 53\. How I would explain why the pretext head is discarded

 If asked:

 > **“Why don't you transfer the entire pretext model?”**

 I would answer:

 > “Because the pretext head is specific to the artificial task. For example, a rotation head predicts four rotation classes, while my downstream classifier predicts target object classes. The backbone contains the reusable representation, while the head is task-specific.”

---

 # 54\. How I would explain fine-tuning

 If asked:

 > **“What is fine-tuning?”**

 I would answer:

 > “Fine-tuning means starting from pretrained parameters and continuing training on the labelled target task so that the learned representation can adapt to the new objective.”

---

 # 55\. How I would explain why SSL might help

 If asked:

 > **“Why should this work?”**

 I would answer:

 > “The hypothesis is that the source images contain visual structure that is also useful for the target task. By solving a self-supervised pretext task, the CNN can learn general visual features before seeing the limited target labels. Those pretrained features may make learning the target task more effective.”

---

 # 56\. How I would explain failure

 If asked:

 > **“What if self-supervised learning doesn't improve the result?”**

 I would answer:

 > “That would not necessarily mean the whole approach is invalid. It could indicate that the chosen pretext task did not produce representations aligned with the downstream task, that the augmentations removed useful information, that the source and target domains were not sufficiently related, or that the fine-tuning strategy needs adjustment. The experiment is testing that hypothesis empirically.”

---

 # 57\. The complete notebook in one mental model

 The entire notebook can finally be reduced to this:

```
                    DATA
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   SOURCE DATA               TARGET DATA
   unlabelled                  labelled
        │                         │
        │                         │
        ▼                         │
   PRETEXT TASK                   │
        │                         │
        ▼                         │
   PRETEXT MODEL                  │
        │                         │
        ▼                         │
   LEARNED BACKBONE               │
        │                         │
        │                         │
        └──────────┐              │
                   ▼              ▼
             TARGET MODEL
                   │
          ┌────────┴────────┐
          │                 │
     pretrained          new target
      backbone              head
          │                 │
          └────────┬────────┘
                   ▼
               FINE-TUNE
                   │
                   ▼
          VALIDATION ACCURACY
                   │
                   ▼
             COMPARE WITH
                BASELINE
```

---

 # 58\. The final explanation I would give to the lecturer

 > “The purpose of this practical is to investigate whether self-supervised pre-training can improve downstream image classification when labelled target data is limited.
>
>  I first construct a small labelled target task from CIFAR-100 and establish a supervised baseline by training the CNN from scratch.
>
>  I then treat CIFAR-10 as an unlabelled source dataset and construct a pretext task from the images themselves. For example, with rotation prediction, I transform each image by one of four rotations and use the known transformation as the artificial label.
>
>  I train a CNN consisting of a reusable backbone and a pretext-specific head. During this stage, the backbone learns visual representations from the unlabelled images.
>
>  After pre-training, I discard the pretext head and retain the pretrained backbone. I then attach a new classification head corresponding to the target classes and fine-tune the resulting model using the labelled target dataset.
>
>  Finally, I compare the target validation accuracy of the fine-tuned model against the baseline trained from scratch.
>
>  The key question is therefore not simply whether the pretext task achieves high accuracy. The important question is whether the representation learned from the unlabelled source data improves downstream target performance.”

---

 # 59\. The 10 things I would memorise before the practical/viva

 If you don't have time to memorise the entire notebook, make sure you can explain these ten things:

 1. **What is self-supervised learning?**\
    Learning from unlabelled data using automatically generated targets.
2. **What is a pretext task?**\
    An artificial task constructed from the data to encourage representation learning.
3. **What is the backbone?**\
    The reusable feature extractor.
4. **What is the head?**\
    The task-specific prediction component.
5. **Why discard the pretext head?**\
    Its outputs correspond to the artificial pretext task, not the target task.
6. **Why keep the backbone?**\
    It contains the learned visual representation.
7. **What is fine-tuning?**\
    Adapting pretrained parameters to the labelled target task.
8. **What is the baseline?**\
    The same target model trained from scratch without self-supervised pre-training.
9. **What is the key metric?**\
    Target validation accuracy.
10. **What is the ultimate experiment?**

```
TRAIN FROM SCRATCH
        VS
SELF-SUPERVISED PRE-TRAIN
        +
FINE-TUNING
```

 with the comparison made on the **same downstream target task**.

 ### The single sentence to remember

 > **“I use an artificial pretext task to learn a reusable visual representation from unlabelled source images, transfer the pretrained backbone to a labelled target task, fine-tune it, and compare its downstream validation accuracy against an equivalent model trained from scratch.”**
