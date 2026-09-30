
 # Summary of Practical 3: Pretext Learning 
 > **Unlabelled images → invent a learning task → pre-train feature extractor → transfer feature extractor → train classifier on small labelled dataset → compare against training from scratch.**

 Yes — **conceptually, this is a very good summary of the notebook and the practical**. The overall pipeline and the distinction between pretext learning, transfer, and downstream evaluation are clear and correct.

1. **“Self-supervised learning” vs. “pretext learning”**\
    Your terminology is reasonable. The practical specifically frames the exercise around a _pretext task_. Calling it self-supervised learning is appropriate because the artificial targets are generated from the input data rather than supplied by humans.
2. **The “10,000 labelled / 50,000 unlabelled” numbers**\
    If those numbers are illustrative rather than exact notebook dataset sizes, make that explicit. Your later description correctly focuses on CIFAR-10 as the source and a selected CIFAR-100 subset as the target.
3. **“Small labelled target dataset”**\
    This is an important qualification. The target isn't necessarily a naturally small dataset; the practical constructs a restricted target problem/subset. I'd phrase it as **“the labelled target subset provided/constructed by the notebook”** unless the notebook explicitly specifies a particular sample count.
4. **Rotation prediction isn't necessarily the required solution**\
    Your summary generally makes this clear, but the later sections sometimes sound like rotation prediction is what the assignment _expects_. If the notebook says you must **design your own pretext task**, rotation is an example/recommendation, not necessarily the mandated task.
5. **“The network can't simply memorize arbitrary pixel values”**\
    I'd soften this. A sufficiently flexible network can exploit shortcuts. The important point is that a _well-designed_ pretext task encourages useful representations; it does not guarantee them.
6. **Fine-tuning vs. feature extraction**\
    Your summary correctly describes fine-tuning as the main transfer stage. But if the notebook allows freezing the backbone, distinguish:
   - **feature extraction:** freeze pretrained backbone, train new head;
   - **fine-tuning:** update some/all pretrained backbone weights as well.
7. **The strongest conclusion is downstream performance**\
    This part of your summary is especially good. Pretext accuracy is an intermediate diagnostic; the meaningful comparison for the practical is the target-task performance of the pretrained/transfer model versus the scratch baseline.
8. **“Everything else is mostly scaffolding”**\
    I'd change this slightly. The pretext dataset/task is indeed the main piece you have to design, but **the experimental comparison, hyperparameters, augmentation choices, transfer strategy, plots, and interpretation are also part of the practical work**.

 ### One thing I'd change in your final sentence

 Instead of:

 > **The key piece that needs to be implemented is the pretext task. Everything else in the notebook is mostly scaffolding...**

 I'd use:

 > **The main piece you need to design and implement is the pretext task. The rest of the notebook provides the training, transfer-learning, evaluation, and comparison framework needed to test whether the representation learned from that task actually improves the downstream target classification problem.**

 That is more precise and avoids understating the experimental component.

 ### Overall

 Your mental model is essentially:

```
UNLABELLED SOURCE DATA
        ↓
design pretext task
        ↓
pretext training
        ↓
PRETRAINED BACKBONE
        ↓
discard pretext head
        ↓
attach target-task head
        ↓
fine-tune on labelled target data
        ↓
target validation performance
        ↓
compare with identical architecture trained from scratch
```

 **That is the core idea of the practical.** The most important conceptual point you've captured correctly is that **the purpose of the pretext task is not to maximize pretext accuracy for its own sake; it is to learn a representation that transfers effectively to the downstream task.**
 
 
 ## 1\. What problem are you solving?

 Imagine you have:

 - **10,000 labelled images** for your actual task.
- **50,000 unlabelled images** from a related domain.

 Normally, you might train a CNN directly on the 10,000 labelled images.

 But the assignment asks:

 > Can we use the 50,000 images **without their labels** to learn useful visual representations first?

 That's **self-supervised learning (SSL)**.

 The important distinction is:

 ### Normal supervised learning

```
image ──→ CNN ──→ class prediction
                  ↑
             real label
```

 For example:

```
image of a tiger → CNN → "tiger"
                       ↑
                    label
```

 ### Self-supervised learning

 There are no human-provided labels.

 Instead, **you construct a task where the labels can be generated automatically from the image itself**.

 For example:

```
original image
     ↓
rotate by 90°
     ↓
"90°" ← automatically generated label
```

 Then the network learns to predict the rotation.

 The hope is that, while solving this artificial problem, the CNN learns useful things like:

 - edges
- textures
- shapes
- object parts
- spatial structure
- semantic visual patterns

 Those features can then be reused for the real classification problem.

---

 # 2\. The entire assignment in one diagram

 This is probably the most important diagram to understand:

```
                    ┌─────────────────────────┐
                    │ Large unlabelled data   │
                    │      CIFAR-10           │
                    └────────────┬────────────┘
                                 │
                                 │ create artificial task
                                 ▼
                    ┌─────────────────────────┐
                    │    PRETEXT TASK         │
                    │                         │
                    │ e.g. predict rotation  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     PRE-TRAIN CNN       │
                    │                         │
                    │     Backbone            │
                    └────────────┬────────────┘
                                 │
                         learned features
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Throw away pretext head │
                    │ Keep the backbone       │
                    └────────────┬────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │ Small labelled target dataset      │
              │       CIFAR-100 subset             │
              └────────────────┬───────────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ New classification head │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Fine-tuning         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Validation accuracy  │
                    └─────────────────────────┘
```

 Then you compare that against:

```
Small labelled dataset
        │
        ▼
CNN from scratch
        │
        ▼
Baseline accuracy
```

 Your objective is **not simply to get high pretext accuracy**.

 Your real objective is:

 > **Get better target-task validation accuracy after self-supervised pre-training than you get by training from scratch.**

---

 # 3\. What are the two datasets?

 The notebook gives you two choices.

 ## Option A — CIFAR

 This is probably the easier choice.

 ### Target dataset

 A subset of **CIFAR-100**.

 The notebook groups CIFAR-100 classes into categories such as:

```
"aquatic mammals"
"fish"
"flowers"
"food containers"
"fruit and veg"
"electricals"
"furniture"
"insects"
...
```

 Each category contains five CIFAR-100 classes.

 For example:

```
TARGET_CATEGORY = "large carnivores"
```

 would give you:

```
bear
leopard
lion
tiger
wolf
```

 So your actual target task becomes:

```
image → one of 5 classes
```

 For example:

```
                  ┌→ bear
image → CNN ──────┼→ leopard
                  ├→ lion
                  ├→ tiger
                  └→ wolf
```

---

 ## Source dataset

 CIFAR-10.

 But here's the important part:

```
cifarx.targets = [-1]*len(cifarx.data)
```

 The labels are deliberately destroyed.

 So you're only allowed to see:

```
image
image
image
image
...
```

 not:

```
image → airplane
image → dog
image → truck
```

 That's what makes the pre-training self-supervised.

---

 # 4\. Why can CIFAR-10 help CIFAR-100?

 This is the key intuition.

 Suppose your target classes are:

```
bear
leopard
lion
tiger
wolf
```

 CIFAR-10 doesn't contain those classes.

 But CIFAR-10 contains things like:

```
dog
cat
horse
bird
car
airplane
...
```

 The exact classes are different.

 However, images still contain common visual concepts:

```
edges
textures
fur
eyes
legs
backgrounds
shapes
curves
object boundaries
etc.
```

 So the CNN can learn general visual representations from CIFAR-10 and potentially reuse them when learning the CIFAR-100 task.

 That's the transfer-learning idea.

---

 # 5\. What is the "backbone"?

 This is one of the most important concepts in the notebook.

 Your model is divided into two parts:

```
                 MODEL
                   │
          ┌────────┴────────┐
          │                 │
       BACKBONE           HEAD
          │                 │
   extracts features    performs task
```

 The backbone is your CNN:

```
class ConvBackbone(nn.Module):
```

 It contains:

```
Conv → Conv → Pool
              ↓
           Conv → Conv
              ↓
        Global average pool
              ↓
             FC
              ↓
             FC
              ↓
          features
```

 Eventually it outputs:

```
backbone_output_size = 64
```

 So an image becomes a 64-dimensional feature representation.

 Conceptually:

```
image
  ↓
CNN backbone
  ↓
[0.13, 0.82, 0.04, ..., 0.71]
       64 features
```

 Those 64 numbers represent what the CNN has learned about the image.

---

 # 6\. What is the classification head?

 The head takes those features and performs a particular task.

 For your target task:

```
ClassifierHead(
    input_size=64,
    projection_size=32,
    num_classes=5
)
```

 So:

```
64 features
     ↓
Linear(64 → 32)
     ↓
ReLU
     ↓
Linear(32 → 5)
     ↓
5 class logits
```

 The crucial idea is that **the head is task-specific**.

 For example, suppose your self-supervised task is rotation prediction.

 You might have:

```
BACKBONE
   ↓
64 features
   ↓
PRETEXT HEAD
   ↓
4 outputs

0°, 90°, 180°, 270°
```

 Later, you throw away that head.

 You keep:

```
BACKBONE
   ↓
64 useful features
```

 and attach a new head:

```
BACKBONE
   ↓
64 features
   ↓
TARGET HEAD
   ↓
5 classes
```

 That's the central mechanism of the practical.

---

 # 7\. What is the baseline?

 Before doing any self-supervised learning, you need to establish:

 > How well can the model perform using only the labelled target data?

 That's your baseline.

 The notebook does:

```
baseline_backbone = initialise_backbone()
```

 and:

```
baseline_target_task_head = ClassifierHead(...)
```

 Then:

```
baseline_model = nn.Sequential(
    baseline_backbone,
    baseline_target_task_head
)
```

 You train it using:

```
train_supervised(...)
```

 on:

```
CIFAR-100 target training set
```

 and evaluate it on:

```
CIFAR-100 target validation set
```

 Suppose you get:

```
Baseline validation accuracy = 47%
```

 Then your goal is to beat that using SSL.

 For example:

```
Baseline:       47%
After SSL:      55%
```

 The important result is the **difference in downstream validation performance**.

---

 # 8\. What exactly is your assignment?

 The actual coding assignment starts here:

 > **Task 1: Pretext Training**

 You have to design your own self-supervised task.

 This is the main creative part.

 The notebook gives you several possibilities.

---

 # 9\. Example: rotation prediction

 This is probably one of the easiest pretext tasks to understand and implement.

 Take an unlabelled image:

```
        original
           ↓
       ┌───────┐
       │       │
       │  DOG  │
       │       │
       └───────┘
```

 Randomly rotate it by one of:

```
0°
90°
180°
270°
```

 You automatically know the answer because **you performed the rotation**.

 Therefore:

```
rotated image → rotation classifier → 0/90/180/270
```

 No human labels were required.

 For example:

```
original = image

rotated = rotate(original, 90)

x = rotated
y = 1
```

 where:

```
0 → 0°
1 → 90°
2 → 180°
3 → 270°
```

 Now you've converted an unlabelled dataset:

```
image
image
image
image
```

 into a self-supervised dataset:

```
rotated image → rotation label
rotated image → rotation label
rotated image → rotation label
```

 The labels are **synthetic**, but valid for the artificial task.

---

 # 10\. Why would rotation prediction teach useful features?

 This is the important conceptual question you'll probably want to explain in your report.

 Suppose the network has to determine whether an image is upside down.

 It may need to learn:

 - object orientation
- shape
- spatial structure
- object parts
- relative positioning
- semantic cues

 Therefore the network can't simply memorize arbitrary pixel values.

 It has to develop some understanding of the image.

 The hope is:

```
rotation learning
       ↓
useful visual representation
       ↓
transfer
       ↓
target classification
```

---

 # 11. Why shouldn't the pretext task be too easy?

 This is a subtle but important point in the assignment.

 Suppose you create this task:

 > "Predict whether I added one tiny amount of Gaussian noise."

 The network might solve this using extremely simple low-level features.

 For example:

```
noise detection
     ↓
pixel statistics
     ↓
solution
```

 The CNN doesn't necessarily need to understand objects.

 That's bad for transfer.

 You want something more like:

```
pretext task
      ↓
requires understanding shapes / structure
      ↓
learn useful representation
      ↓
better downstream performance
```

 So the question isn't:

 > "Can I create a task that the CNN can solve?"

 Almost certainly yes.

 It's:

 > **"Can I create a task whose solution requires the CNN to learn features useful for my eventual target task?"**

 That's the heart of the assignment.

---

 # 12\. Data augmentation

 You'll notice a lot of augmentation code:

```
RandomResizedCrop
RandomHorizontalFlip
ColorJitter
RandomGrayscale
RandomErasing
```

 These create variations of the same image.

 For example:

```
             original
                │
       ┌────────┼─────────┐
       ↓        ↓         ↓
     crop     flip      colour
       │        │         │
       ▼        ▼         ▼
   image A   image B   image C
```

 This prevents the model from simply memorising exact images.

 It encourages more robust features.

 However, the notebook makes an important warning:

 > **The augmentation must not destroy the pretext signal.**

 For example, if your task is:

```
predict rotation
```

 you should **not** also randomly rotate the image.

 Otherwise:

```
you rotate image 90°
        ↓
augmentation randomly rotates another 180°
        ↓
actual rotation = 270°
```

 Now your label says 90° but the model sees 270°.

 That's contradictory training data.

---

 # 13\. What do you actually need to code?

 There are essentially **two pieces of coding** you need to complete.

 ## Part 1 — Pretext dataset

 You need to turn:

```
source_data
```

 into something like:

```
image → artificial label
```

 You can do this either beforehand or dynamically.

 For example:

```
source_data
    │
    ▼
randomly rotate
    │
    ▼
rotated image + rotation label
```

 Then make a `DataLoader`.

---

 ## Part 2 — Pretext model

 You reuse:

```
ConvBackbone
```

 but attach a head appropriate to your pretext task.

 For rotation:

```
Backbone
   ↓
64 features
   ↓
ClassifierHead
   ↓
4 outputs
```

 because there are four possible rotations.

---

 # 14\. Your training loop

 You can actually reuse most of:

```
train_supervised()
```

 because from the neural network's perspective, your pretext task is supervised.

 This sounds contradictory:

 > "We're doing self-supervised learning, but you're using supervised training?"

 That's perfectly normal.

 The **source data has no human labels**, but you manufacture labels from the data.

 So after constructing:

```
image → artificial label
```

 you can use ordinary supervised learning machinery:

```
CrossEntropyLoss
      ↓
backpropagation
      ↓
AdamW
      ↓
CNN
```

 The distinction is where the labels came from.

---

 # 15\. Pre-training

 Suppose you choose rotation prediction.

 You create:

```
pretext_backbone = initialise_backbone()
```

 Then:

```
pretext backbone
       +
rotation classification head
```

 Train it on the entire unlabelled source dataset.

 Conceptually:

```
CIFAR-10 images
      ↓
random rotations
      ↓
rotation labels
      ↓
CNN
      ↓
learn representation
```

 You monitor:

```
training loss
validation loss
training accuracy
validation accuracy
```

 If the rotation task is:

```
0°, 90°, 180°, 270°
```

 random guessing is approximately:

```
25%
```

 So you would expect the model to learn something above that.

 But, importantly, **very high pretext accuracy isn't automatically better**.

 The actual question is whether the learned backbone transfers well.

---

 # 16\. Saving the backbone

 Once pre-training is finished, you want:

```
PRETRAINED BACKBONE
```

 but you don't care about:

```
PRETEXT HEAD
```

 For example:

```
torch.save(
    pretext_backbone.state_dict(),
    "pretrained_backbone.pt"
)
```

 Then later:

```
target_backbone = initialise_backbone()

target_backbone.load_state_dict(
    torch.load("pretrained_backbone.pt")
)
```

 Now you've transferred the learned representation.

---

 # 17\. Fine-tuning

 Now return to your actual task.

 Suppose your target dataset contains:

```
bear
leopard
lion
tiger
wolf
```

 Create:

```
pretrained backbone
       +
new 5-class classification head
```

 So:

```
CIFAR-100 target image
       ↓
PRETRAINED BACKBONE
       ↓
64 features
       ↓
NEW CLASSIFICATION HEAD
       ↓
bear/leopard/lion/tiger/wolf
```

 Then train on your small labelled dataset.

 This is the **fine-tuning stage**.

---

 # 18\. Why is it called fine-tuning?

 Because the backbone already has useful knowledge.

 Instead of starting with random weights:

```
random CNN
    ↓
learn everything from small dataset
```

 you start with:

```
CNN already trained on lots of images
    ↓
adapt it to target task
```

 You can use a smaller learning rate because you don't necessarily want to destroy the useful features it already learned.

 For example:

```
ft_lr = 1e-4
```

 instead of:

```
ft_lr = 1e-3
```

 You can also experiment with freezing some layers:

```
early layers → frozen
later layers → trainable
classification head → trainable
```

---

 # 19\. The experiment you are ultimately performing

 You are comparing two approaches.

 ### Experiment A — Baseline

```
CIFAR-100 labelled data
          ↓
      random CNN
          ↓
      train CNN
          ↓
 validation accuracy
```

 ### Experiment B — Self-supervised

```
CIFAR-10 unlabelled
          ↓
     pretext task
          ↓
   pre-trained CNN
          ↓
      transfer
          ↓
CIFAR-100 labelled data
          ↓
       fine-tune
          ↓
 validation accuracy
```

 Then compare:

```
                 Validation accuracy

Baseline             XX.X%
SSL + fine-tuning    YY.Y%
```

 That's essentially the entire experiment.

---

 # 20\. What the assignment expects you to hand in

 The assessment section is quite short.

 You need three things.

 ## A. Describe your pretext task

 Explain:

 1. What task did you choose?
2. Why did you choose it?
3. How did you generate the artificial labels?
4. What architecture did you use?
5. What augmentations did you use?

 For example:

 > I used a four-way rotation prediction task. Each unlabelled image was randomly rotated by 0°, 90°, 180° or 270°, with the rotation angle serving as the automatically generated target. A CNN backbone was trained with a four-class classification head. The pretext task was selected because rotation prediction requires the model to learn spatial and semantic properties of objects rather than relying solely on pixel-level reconstruction.

 That's the kind of explanation they're looking for.

---

 # 21\. Show your results

 You need a plot or two.

 A useful plot would show:

```
Validation accuracy
      │
 60%  │                 ───── SSL
      │             ───
 50%  │─────── baseline
      │
 40%  │
      └────────────────────────
                 epochs
```

 You can also show:

 - pretext training loss
- pretext validation accuracy
- downstream training/validation accuracy
- baseline vs SSL validation accuracy

 The notebook's `TrainingMetrics.plot()` is already designed to help with this.

---

 # 22\. Discuss what went wrong / what you learned

 This part doesn't require some dramatic failure.

 You could discuss things like:

 - the pretext task initially converged too quickly
- the model overfit
- stronger augmentation improved transfer
- pretext accuracy increased but downstream accuracy didn't
- changing the learning rate improved fine-tuning
- freezing the backbone hurt/helped
- the pretext task was too easy
- the pretext task didn't capture sufficiently semantic features

 This is actually a very important part of the assignment.

---

 # 23\. A particularly important distinction

 Don't make this mistake:

 > "My pretext accuracy is 90%, therefore my SSL model is good."

 Not necessarily.

 Imagine:

```
Pretext task A
accuracy = 95%
downstream accuracy = 45%
```

 versus:

```
Pretext task B
accuracy = 70%
downstream accuracy = 55%
```

 For this assignment, **B has produced the more useful representation**, because the objective is downstream performance.

 The pretext task is merely a mechanism for learning features.

 Think of it as:

```
             PRETEXT TASK
                  │
                  ▼
          FEATURE LEARNING
                  │
                  ▼
        DOWNSTREAM PERFORMANCE
                  ↑
             what matters
```

---

 # 24\. What I'd recommend you implement

 For a first attempt, I would keep the project relatively simple.

 A sensible approach is:

 ### Pretext task: 4-way rotation prediction

 Use:

```
0°
90°
180°
270°
```

 Then:

```
Unlabelled image
      ↓
Random crop/colour augmentation
      ↓
Randomly rotate
      ↓
Rotation label
      ↓
ConvBackbone
      ↓
4-class head
```

 Then transfer:

```
trained ConvBackbone
      ↓
new 5-class target head
      ↓
fine-tune
```

 This is much simpler than implementing something like full SimCLR.

---

 # 25\. One issue in the provided notebook

 There is a small apparent inconsistency in the supplied example:

```
v2.RandomResizedCrop(size=img_dim, ...)
```

 but earlier the notebook defines:

```
img_size
```

 not `img_dim`.

 So if you run that cell exactly as provided, you may get:

```
NameError: name 'img_dim' is not defined
```

 You'd likely want:

```
v2.RandomResizedCrop(size=img_size, scale=(0.8, 1.0))
```

 Also, the line:

```
TARGET_CATEGORY =
```

 is intentionally left for you to fill in. For example:

```
TARGET_CATEGORY = "large carnivores"
```

---

 # 26\. What each important notebook variable means

 Here's a useful cheat sheet:

 | Variable | Meaning |
| --- | --- |
| `source_data` | Large unlabelled dataset used for SSL |
| `target_data` | Small labelled dataset for the real task |
| `target_data_train` | Target training split |
| `target_data_val` | Target validation split |
| `source_unsup_loader` | DataLoader for unlabelled images |
| `ConvBackbone` | Feature extractor |
| `ClassifierHead` | Task-specific prediction layer |
| `baseline_model` | CNN trained from scratch |
| `baseline_metrics` | Performance of baseline |
| `baseline_acc` | Best baseline validation accuracy |
| `pretext task` | Artificial task created from unlabelled data |
| `pretext model` | Backbone + pretext-specific head |
| `ft_metrics` | Fine-tuning results |
| `ft_acc` | Best post-SSL validation accuracy |

---

 # 27\. The conceptual difference between the three stages

 You can remember the assignment with these three words:

 ### 1\. Pretext

 > **Learn features without real labels.**

```
unlabelled images
       ↓
artificial task
       ↓
learn representation
```

 ### 2\. Transfer

 > **Keep the representation.**

```
pretrained backbone
       ↓
discard old head
       ↓
attach new head
```

 ### 3\. Fine-tune

 > **Adapt those features to the real task.**

```
small labelled target dataset
       ↓
fine-tune
       ↓
validation accuracy
```

---

 # 28\. In one sentence

 If you need to explain the assignment to your lecturer:

 > **The practical investigates whether a CNN can learn transferable visual representations from a large unlabelled source dataset using a self-supervised pretext task, and whether those representations improve classification accuracy on a small labelled target dataset compared with training the same architecture from scratch.**

 And your experimental pipeline is simply:

```
              BASELINE
                 │
       small labelled dataset
                 ↓
          CNN from scratch
                 ↓
          baseline accuracy
                 │
                 │ compare
                 ▼
             SELF-SUPERVISED
                 │
       large unlabelled dataset
                 ↓
          pretext task
                 ↓
       pretrained backbone
                 ↓
       new target classifier
                 ↓
            fine-tuning
                 ↓
          final accuracy
```

 **The key piece that needs to be implemented is the pretext task.** Everything else in the notebook is mostly scaffolding to let measure whether the idea actually improves downstream performance.
