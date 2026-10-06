# Practical 3 - Efficient Learning - Practical

Absolutely. This assignment is essentially asking you to **invent a small self-supervised learning problem, train a CNN on it using unlabeled images, then reuse the learned CNN features for a different classification problem**.

 The most important thing to understand is the overall pipeline:

 > **Unlabelled images → invent a pretext task → pre-train feature extractor → throw away pretext head → attach classification head → fine-tune on small labelled dataset → beat baseline**

 I'll break the notebook down from the beginning and then explain exactly what you need to implement.

 ## 1\. What is the assignment trying to teach?

 The central idea is **self-supervised learning (SSL)**.

 Normally, suppose you have images:

```
image → label
cat   → cat
dog   → dog
car   → car
```

 You train a classifier directly:

```
image → CNN → classifier → class
```

 But labelled data can be expensive.

 Self-supervised learning says:

 > "What if I can create a learning problem from the images themselves, without human-provided labels?"

 For example, take an image:

```
original image
      ↓
rotate 90°
      ↓
rotated image + label "90°"
```

 You didn't need a human to tell you the label. **You created the label yourself.**

 The model learns to predict:

```
0°, 90°, 180°, 270°
```

 While doing that, the CNN hopefully learns useful visual features such as:

 - edges
- shapes
- textures
- object parts
- spatial structure
- semantic information

 Then you throw away the rotation classifier and reuse the CNN.

---

 # 2\. The big picture

 Your assignment has **two datasets and two tasks**.

 ### Dataset A: Source / pre-training dataset

 For CIFAR, this is:

```
CIFAR-10
50,000 images
NO LABELS
```

 The notebook deliberately destroys the labels:

```
cifarx.targets = [-1] * len(cifarx.data)
```

 So your model sees only:

```
image
image
image
image
...
```

 You have to invent labels.

---

 ### Dataset B: Target / downstream dataset

 This is a small subset of CIFAR-100.

 For example, if you choose:

```
TARGET_CATEGORY = "vehicles"
```

 then the five classes are:

```
bicycle
bus
motorcycle
pickup_truck
train
```

 These **do have labels**.

 Your final objective is:

 > classify these five classes as accurately as possible.

---

 # 3\. Why use different datasets?

 This is important.

 The model is pre-trained on:

```
CIFAR-10
```

 but ultimately tested on:

```
CIFAR-100 subset
```

 And the classes don't overlap.

 For example:

```
CIFAR-10:
airplane
bird
cat
deer
dog
frog
horse
ship
truck
automobile
```

 Target might be:

```
CIFAR-100:
bicycle
bus
motorcycle
pickup truck
train
```

 So you **cannot simply learn "cat = class 3" during pretraining**.

 Instead, you're trying to learn general visual knowledge:

```
edges
shapes
textures
object structure
spatial relationships
...
```

 which can transfer to the target problem.

 That's the important idea behind the assignment.

---

 # 4\. What exactly are you comparing?

 You need to create two models.

 ## Model 1 — Baseline

 Train directly on the small labelled target dataset:

```
CIFAR-100 subset
       ↓
     CNN
       ↓
classifier
       ↓
5 classes
```

 This tells you:

 > "How well can I do without self-supervised learning?"

 This is your baseline.

---

 ## Model 2 — Self-supervised model

 First train the CNN using unlabeled CIFAR-10:

```
CIFAR-10 images
      ↓
pretext task
      ↓
CNN backbone
      ↓
pretext classifier
```

 Then throw away the pretext classifier:

```
CNN backbone
      ↓
NEW classifier
      ↓
CIFAR-100 target classes
```

 Then fine-tune.

 Finally compare:

```
                    Validation accuracy
Baseline            40%
SSL + fine-tuning   48%
```

 If SSL works, you should get an improvement.

---

 # 5\. The most important concept: backbone vs head

 The notebook deliberately separates the network into two parts.

 ## Backbone

 This is:

```
ConvBackbone
```

 Its job is to turn an image into useful features.

 For example:

```
image
  ↓
convolution layers
  ↓
feature extraction
  ↓
64-dimensional vector
```

 The output is:

```
backbone_output_size = 64
```

 So:

```
image → backbone → [64 features]
```

---

 ## Head

 The head converts those features into something specific to a task.

 For the target classification task:

```
64 features
    ↓
32-dimensional projection
    ↓
5 class predictions
```

 That's:

```
ClassifierHead(
    input_size=64,
    projection_size=32,
    num_classes=5
)
```

 The key idea is:

 > **The backbone is reusable. The head is task-specific.**

 This is exactly what you're going to exploit.

---

 # 6\. Understanding the baseline

 The notebook creates:

```
baseline_backbone = initialise_backbone()
```

 Then:

```
baseline_target_task_head = ClassifierHead(...)
```

 Then combines them:

```
baseline_model = nn.Sequential(
    baseline_backbone,
    baseline_target_task_head
)
```

 So conceptually:

```
                   BASELINE

CIFAR-100 image
       │
       ▼
┌───────────────┐
│ ConvBackbone  │
│               │
│ feature       │
│ extraction    │
└───────┬───────┘
        │
        │ 64 features
        ▼
┌───────────────┐
│ Target Head   │
└───────┬───────┘
        │
        ▼
  5 class logits
```

 Then:

```
baseline_metrics = train_supervised(...)
```

 trains **the entire thing from scratch**.

 You record:

```
baseline_acc = baseline_metrics.best_val_acc
```

 This is your benchmark.

---

 # 7\. What is the pretext task?

 This is **the main thing you have to invent**.

 The assignment gives you several possibilities.

 For example:

 ### Option A — Rotation prediction

 Take an image:

```
       🐶
```

 Randomly rotate it:

```
90°
180°
270°
0°
```

 Then tell the model which rotation you applied.

 The labels are generated automatically:

```
0°   → 0
90°  → 1
180° → 2
270° → 3
```

 So the model learns:

```
rotated image → rotation angle
```

 This is called the **pretext task**.

---

 # 8\. Why is rotation a useful pretext task?

 Imagine the model needs to identify whether an image was rotated 90°.

 It can't necessarily solve that by looking at a single pixel.

 It needs to learn things about:

 - object orientation
- shapes
- edges
- spatial structure
- object parts

 Those are potentially useful for the downstream classification task.

 The important distinction is:

 ### Bad pretext task

 Something like:

```
blur image → reconstruct original image
```

 might mainly teach:

```
low-level pixel statistics
```

 That might not produce useful semantic features.

 ### Better pretext task

 Something like:

```
image → determine rotation
```

 forces the model to understand more of the image structure.

---

 # 9\. What does "self-supervised" actually mean here?

 This is worth remembering for your report.

 You're technically using labels, but **you created the labels automatically**.

 Suppose you start with:

```
image
```

 You apply:

```
rotate(image, 90)
```

 You know that you applied 90°.

 Therefore you can create:

```
(rotated_image, 90)
```

 The human never labelled the image.

 That's why it's called **self-supervised**.

 The supervision comes from the transformation itself.

---

 # 10\. What does your implementation need to do?

 The assignment gives you this:

```
### implement your pretext task here!

...
```

 That's the main section you need to fill.

 You need to solve approximately five things.

 ### Step 1 — Create pretext examples

 Given:

```
unlabelled image
```

 create:

```
transformed image + automatically generated label
```

 For rotation:

```
original
   ↓
random rotation
   ↓
(rotated image, rotation_label)
```

---

 ### Step 2 — Create a DataLoader

 You currently have:

```
source_unsup_loader
```

 which returns only:

```
images
```

 But your pretext model needs:

```
images + pretext labels
```

 Therefore you need a new `collate_fn` or dataset.

 For example:

```
source image
     ↓
randomly choose rotation
     ↓
rotate image
     ↓
return image, rotation_label
```

 Your DataLoader then returns:

```
x, y
```

 just like ordinary supervised learning.

 This is why the assignment says:

 > "You'll probably need to create a new collate\_fn for a new data loader."

---

 # 11\. Then create a pretext model

 Suppose you choose rotation.

 You need:

```
ConvBackbone
     ↓
rotation head
     ↓
4 outputs
```

 Why 4?

 Because:

```
0°
90°
180°
270°
```

 So:

```
pretext_head = ClassifierHead(
    input_size=64,
    projection_size=32,
    num_classes=4
)
```

 Then:

```
pretext_model = nn.Sequential(
    pretext_backbone,
    pretext_head
)
```

---

 # 12\. Train the pretext model

 Now you can use something very similar to:

```
train_supervised(...)
```

 because your pretext problem has now become an ordinary classification problem.

 The model sees:

```
rotated image
```

 and predicts:

```
0 / 90 / 180 / 270
```

 Loss:

```
nn.CrossEntropyLoss()
```

 So conceptually:

```
              PRETEXT TRAINING

CIFAR-10 image
      ↓
random rotation
      ↓
rotated image
      ↓
┌───────────────┐
│   Backbone    │
└───────┬───────┘
        │
        ▼
┌────────────────┐
│ Rotation Head  │
└───────┬────────┘
        │
        ▼
  0/90/180/270
```

---

 # 13\. What does the model learn?

 This is the key conceptual part.

 Initially:

```
backbone
```

 has random weights.

 After pre-training:

```
backbone
```

 has learned useful visual representations.

 For example, it might learn filters detecting:

```
edges
corners
textures
shapes
object boundaries
spatial arrangements
```

 You don't actually care that much about the rotation classifier itself.

 You care about:

 > **the features inside the backbone.**

---

 # 14\. Then comes the transfer

 Suppose your pretext model is:

```
pretext_model = nn.Sequential(
    pretext_backbone,
    pretext_head
)
```

 After training, you keep:

```
pretext_backbone
```

 and throw away:

```
pretext_head
```

 Then create:

```
target_head = ClassifierHead(
    input_size=64,
    projection_size=32,
    num_classes=5
)
```

 And combine:

```
finetune_model = nn.Sequential(
    pretext_backbone,
    target_head
)
```

 Now:

```
CIFAR-100 target image
        ↓
pre-trained backbone
        ↓
64 useful features
        ↓
new target classifier
        ↓
target class
```

 That's the entire experiment.

---

 # 15\. Why replace the head?

 Imagine your pretext task is:

```
rotation prediction
```

 Its output is:

```
4 classes
```

 But your target task is:

```
5 CIFAR-100 classes
```

 So the old head makes no sense.

 The old head answers:

 > "Was this image rotated 0°, 90°, 180°, or 270°?"

 The new head answers:

 > "Is this a bicycle, bus, motorcycle, pickup truck, or train?"

 Therefore:

```
OLD HEAD
rotation → 4 outputs
```

 is discarded.

```
NEW HEAD
target classification → 5 outputs
```

 is created.

 But:

```
BACKBONE
```

 is kept.

---

 # 16\. Fine-tuning vs freezing

 The assignment says **fine-tune**.

 That generally means the backbone is allowed to continue learning.

 So initially:

```
pre-trained backbone
        ↓
target head
```

 Then train on your labelled target data.

 Both can update:

```
backbone ← updated
head     ← updated
```

 You can optionally experiment with freezing the backbone:

```
for param in backbone.parameters():
    param.requires_grad = False
```

 Then only the new head learns.

 But I would first implement the straightforward version:

 > **pre-train → transfer backbone → fine-tune entire model**

 because that is the core requirement.

---

 # 17\. What does "fine-tuning" actually accomplish?

 Suppose pretraining learned:

```
edges
textures
shapes
object orientation
```

 Now your small labelled dataset teaches the model:

```
these features → bicycle
these features → bus
these features → train
...
```

 So you're not starting from random weights.

 You're starting from:

```
useful visual representation
```

 which should make learning easier.

---

 # 18\. The complete pipeline

 Here's the entire assignment in one diagram:

```
                    SELF-SUPERVISED LEARNING
                    ========================

             UNLABELLED SOURCE DATA
                    CIFAR-10
                       │
                       │
                       ▼
               Create pretext labels
                       │
                       │
               e.g. rotate image
                       │
                       ▼
              ┌─────────────────┐
              │  Pretext Model  │
              │                 │
              │   Backbone      │
              │       ↓         │
              │  Rotation Head  │
              └────────┬────────┘
                       │
                       │ train
                       ▼
              PRE-TRAINED BACKBONE
                       │
                       │
                       │ discard pretext head
                       │
                       ▼
             ┌───────────────────┐
             │ PRE-TRAINED       │
             │ BACKBONE          │
             └─────────┬─────────┘
                       │
                       │ attach new head
                       ▼
             ┌───────────────────┐
             │ TARGET CLASSIFIER │
             └─────────┬─────────┘
                       │
                       ▼
              CIFAR-100 TARGET DATA
                    labelled
                       │
                       │ fine-tune
                       ▼
                FINAL MODEL
                       │
                       ▼
              VALIDATION ACCURACY
                       │
                       ▼
             Compare with baseline
```

---

 # 19\. What the existing code is already doing for you

 A lot of the difficult infrastructure has already been provided.

 You don't need to rewrite everything.

 ### Already provided:

 - dataset classes
- dataset downloading
- train/validation split
- preprocessing
- augmentation
- DataLoaders
- CNN backbone
- classification head
- training loop
- evaluation
- accuracy calculation
- metric tracking
- plotting
- baseline training

 ### Your main job:

 **Design and implement the self-supervised pretext task.**

 Then:

 **transfer the trained backbone and fine-tune it.**

---

 # 20\. One issue in the notebook you should notice

 This line appears:

```
random_augment = v2.Compose([
    v2.RandomResizedCrop(size=img_dim, scale=(0.8, 1.0)),
```

 But earlier the notebook uses:

```
img_size
```

 not:

```
img_dim
```

 So this will likely produce:

```
NameError: name 'img_dim' is not defined
```

 You probably want:

```
v2.RandomResizedCrop(size=img_size, scale=(0.8, 1.0))
```

 Also, be careful with augmentation ordering and tensor/PIL expectations when using `torchvision.transforms.v2`.

---

 # 21\. What pretext task should you choose?

 If your goal is:

 > **Get the assignment working without making your life unnecessarily difficult**

 I'd recommend **rotation prediction**.

 It's conceptually simple and fits perfectly into the existing classification training code.

 ### Pretext task

 Randomly rotate every image into one of four orientations:

```
             0°
             ↑

90° ← image → 270°

             ↓
            180°
```

 Label:

```
0°   → 0
90°  → 1
180° → 2
270° → 3
```

 Model:

```
CIFAR-10
   ↓
rotation
   ↓
CNN backbone
   ↓
4-class head
```

 Then transfer the backbone.

---

 # 22\. Why augmentation matters

 Suppose you rotate an image and that's the only transformation.

 The model might find shortcuts.

 The assignment therefore suggests adding random augmentations.

 For example:

```
original image
      ↓
rotate 90°
      ↓
random crop
      ↓
horizontal flip
      ↓
colour jitter
      ↓
model
```

 But there's an important warning.

 If your pretext task is rotation prediction, **don't randomly rotate the image again**.

 Otherwise you're creating contradictory information.

 For example:

```
You say:
"This is a 90° image."

But then your augmentation randomly rotates it another 180°.

The model actually sees:
270°
```

 That's bad.

---

 # 23\. How do you know whether the pretext task is good?

 This is one of the most important parts of the assignment.

 You should **not** simply say:

 > "My pretext accuracy reached 99%, therefore my pretext task is excellent."

 The assignment explicitly warns against this.

 If the task is ridiculously easy:

```
pretext accuracy = 99%
```

 the model may not have learned useful representations.

 What really matters is:

```
Does the pre-trained backbone improve target-task accuracy?
```

 So your final metric is:

```
SSL + fine-tuning accuracy
```

 versus:

```
baseline accuracy
```

---

 # 24\. Example of what your results might look like

 Imagine your baseline gives:

```
Baseline:
validation accuracy = 42%
```

 After SSL:

```
Pretext accuracy = 63%
```

 Then fine-tuning:

```
SSL + fine-tuning:
validation accuracy = 49%
```

 Your conclusion would be:

```
Baseline:              42%
Self-supervised model: 49%

Improvement:            +7 percentage points
```

 That's a successful result.

 Notice that the pretext accuracy itself isn't the final objective.

---

 # 25\. What should you write for the assessment?

 The assignment gives you three main things to report.

 ## A. Describe your pretext task

 Explain:

 1. What transformation you use.
2. How you generate labels.
3. Why the task should produce useful features.
4. What augmentations you use.
5. How you train it.

 For rotation, your explanation would essentially be:

 > I used a four-way rotation prediction task. Each unlabeled source image was randomly assigned one of four rotations: 0°, 90°, 180°, or 270°. The rotation angle provides an automatically generated label, allowing the backbone to be trained without human annotations. The motivation is that predicting orientation requires the network to learn spatial and semantic image features rather than simply memorising pixel-level statistics.

---

 # 26\. B. Show results

 You should compare:

```
Baseline
vs
SSL + fine-tuning
```

 Ideally show training curves.

 For example:

```
Validation accuracy

60% |                    SSL
    |                   /---
50% |             -----/
    |            /
40% |-----------/--------- Baseline
    |
30% |
    +-------------------------
       training steps
```

 The notebook's `TrainingMetrics.plot()` is already designed to help you do this.

---

 # 27\. C. Discuss problems and insights

 This can be quite simple.

 You could discuss things such as:

 - the pretext task initially converged slowly
- stronger augmentation improved transfer performance
- very weak augmentation made the pretext task too easy
- the baseline overfit quickly because the labelled target dataset is small
- the pre-trained backbone gave better validation accuracy
- fine-tuning worked better than training from scratch
- changing the learning rate affected stability

 The point is to show that you actually investigated the behaviour of the system rather than just running it.

---

 # 28\. What you should NOT do

 There are a few common misunderstandings.

 ### Don't train the pretext task on CIFAR-100 labels

 The whole point is:

```
CIFAR-10 = unlabeled
```

 You create your own labels.

---

 ### Don't use the pretext classifier during final evaluation

 You don't want:

```
rotation head → CIFAR-100
```

 You want:

```
pretrained backbone → new target head
```

---

 ### Don't compare an unfair baseline

 If you use heavy augmentation for SSL fine-tuning but no augmentation for your baseline, you should be careful about claiming that the improvement is purely due to SSL.

 The assignment explicitly mentions keeping comparisons fair.

---

 # 29\. A good mental model

 Think of the assignment as teaching the CNN two different languages.

 ### Stage 1

 You teach the CNN:

 > "Understand images."

 using an artificial game:

 > "Tell me how I transformed this image."

 That's pretraining.

 ### Stage 2

 You teach it:

 > "Now use what you've learned to distinguish these five classes."

 That's fine-tuning.

 So:

```
PRETEXT TASK
"What happened to this image?"

             ↓
       useful features

             ↓

TARGET TASK
"What object is this?"
```

---

 # 30\. The minimum implementation you need

 If you choose rotation prediction, your implementation needs roughly these components:

```
1. Rotation transform
        ↓
2. Automatically generate rotation labels
        ↓
3. Pretext DataLoader
        ↓
4. Pretext backbone
        ↓
5. 4-class rotation head
        ↓
6. Train pretext model
        ↓
7. Save/copy backbone weights
        ↓
8. Create new target classification head
        ↓
9. Attach pretrained backbone
        ↓
10. Fine-tune on CIFAR-100 subset
        ↓
11. Compare against baseline
        ↓
12. Plot/report results
```

---

 ## 31\. One more important detail: `TARGET_CATEGORY`

 You need to fill this:

```
TARGET_CATEGORY =
```

 For example:

```
TARGET_CATEGORY = "vehicles"
```

 That gives:

```
bicycle
bus
motorcycle
pickup_truck
train
```

 You can choose any of the listed superclasses:

```
aquatic mammals
fish
flowers
food containers
fruit and veg
electricals
furniture
insects
large carnivores
large buildings
natural scenes
large herbivores
medium mammals
invertebrates
people
reptiles
small mammals
trees
vehicles
weird vehicles
```

 For a first implementation, I'd choose **vehicles**, **flowers**, or **fish**, because the classes are intuitive when inspecting the images.

---

 # 32\. In one sentence: what are you being graded on?

 The assignment is essentially asking:

 > **Can you design a meaningful artificial learning task from unlabeled images, use it to learn a reusable CNN representation, transfer that representation to a small labelled classification problem, and demonstrate that it improves over training from scratch?**

 That's the core of the whole notebook.

 If you want, I can next take **the exact notebook you pasted and walk you through implementing Task 1 (rotation prediction) line-by-line**, including the `Dataset/DataLoader`, pretext model, training, backbone transfer, and fine-tuning code.
