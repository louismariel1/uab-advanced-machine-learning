# Consistency Regularization Notebook — Complete Explanation

 This notebook implements **semi-supervised learning using consistency regularization**, specifically a simplified version of **Unsupervised Data Augmentation (UDA)**.

 The central idea is:

 > **Use the small amount of labelled data to teach the model what the classes mean, and use the large amount of unlabelled data to teach the model that small, class-preserving changes to an image should not change its prediction.**

 The notebook progresses through:

 1. Creating a partially labelled CIFAR-10 dataset.
2. Training a supervised baseline.
3. Creating augmented versions of unlabelled images.
4. Defining a KL-divergence consistency loss.
5. Combining supervised and unsupervised losses.
6. Training with UDA.
7. Improving UDA using confidence filtering.
8. Experimenting with `lambda_u` and warmup.
9. Comparing everything against the baseline.

---

 # 1\. What problem is the notebook solving?

 Imagine we have a dataset containing many images, but only a small proportion have labels.

 For example:

```
50,000 images
     |
     +-- 5,000 labelled
     |
     +-- 45,000 unlabelled
```

 Traditional supervised learning would throw away the information contained in the 45,000 unlabelled examples.

 Semi-supervised learning asks:

 > **Can we use the unlabelled examples to improve the model?**

 That is exactly what this practical investigates.

 The dataset is CIFAR-10.

 There are ten classes:

```
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

 The notebook artificially destroys 90% of the training labels.

 Therefore, approximately:

```
10% labelled
90% unlabelled
```

 The validation set remains fully labelled because we need labels to measure performance.

---

 # 2\. The overall architecture

 The entire practical can be understood as this pipeline:

```
                    CIFAR-10
                       |
              ---------------------
              |                   |
         labelled              unlabelled
              |                   |
              |             ----------------
              |             |              |
              |           clean         augmented
              |             |              |
              |             |              |
              v             v              v
          supervised     model          model
          prediction       |              |
              |             |              |
              v             -------->     |
       Cross Entropy              KL divergence
              |                         |
              -----------+--------------
                         |
                         v
                  total training loss
                         |
                         v
                    backpropagation
                         |
                         v
                       model
```

 There are therefore **two learning signals**:

 ### Supervised signal

```
labelled image
      |
      v
    model
      |
      v
prediction
      |
      v
compare with true label
      |
      v
Cross Entropy Loss
```

 ### Unsupervised signal

```
unlabelled image
       |
       +----------> clean image ----------> model
       |                                      |
       |                                      v
       |                              prediction P
       |
       +----------> augmented image -----> model
                                              |
                                              v
                                       prediction Q

                              P should ≈ Q

                              KL(P || Q)
```

 The two losses are then combined.

---

 # 3\. Why does consistency regularization work?

 This is the most important conceptual idea in the notebook.

 Suppose we have an image of a dog.

 The original image might be:

```
       DOG
        |
        v
      Model
        |
        v
dog = 0.92
cat = 0.05
horse = 0.02
...
```

 Now horizontally flip it and change its colour slightly:

```
       AUGMENTED DOG
             |
             v
           Model
             |
             v
dog = 0.90
cat = 0.06
horse = 0.02
...
```

 We want these predictions to be similar.

 Why?

 Because the augmentation should not change the semantic class.

 A horizontal flip does not turn a dog into a cat.

 Therefore:

 > **If two versions of an image represent the same underlying object/class, the model should make similar predictions for both.**

 This is called the **consistency assumption**.

---

 # 4\. The semi-supervised assumption

 The method relies on an important assumption:

 > **Small or appropriate perturbations of an example should not change its class.**

 Mathematically, if `x` is an image and `T(x)` is a class-preserving transformation:

```
y(x) = y(T(x))
```

 Therefore, ideally:

```
f(x) ≈ f(T(x))
```

 where `f` is the model.

 This provides a training signal even though we do not know the true label.

 That is the clever part.

 We don't know:

```
"This image is a dog."
```

 But we can still say:

```
"The model's prediction for the clean image
should be approximately the same as its
prediction for the augmented image."
```

 That gives us an **unsupervised learning objective**.

---

 # 5\. Why the validation set remains labelled

 The notebook creates:

```
VAL_FRAC = 0.1
```

 So 10% of CIFAR-10 is held out for validation.

 The remaining 90% becomes the training pool.

 Then 90% of the training labels are destroyed.

 This creates:

```
Original CIFAR-10
       |
       +----------------+
       |                |
    Training         Validation
       |                |
       |              labels kept
       |
       +----------------+
       |                |
   10% labelled      90% unlabelled
```

 The validation set **must retain labels** because otherwise we couldn't calculate validation accuracy.

 The unlabelled training set does not expose its labels to the learning algorithm.

---

 # 6\. What does `label = -1` mean?

 The notebook uses:

```
label = -1
```

 to represent an unknown label.

 So:

```
0 → airplane
1 → automobile
...
9 → truck
-1 → unknown
```

 The `-1` is not a real CIFAR-10 class.

 It is simply a marker saying:

 > "We have an image, but we do not know its label."

---

 # 7\. Why split the training data into `trainL` and `trainU`?

 After destroying the labels, the notebook creates:

```
data_trainL
```

 and

```
data_trainU
```

 They represent:

```
trainL = labelled training data
trainU = unlabelled training data
```

 This makes the later training loop much easier.

 The supervised path consumes:

```
(x, y)
```

 while the unsupervised path consumes:

```
x
```

 or later:

```
(x_clean, x_aug)
```

---

 # 8\. Why is normalization important?

 The notebook defines:

```
normalise = transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2023, 0.1994, 0.2010)
)
```

 These are approximately the CIFAR-10 per-channel means and standard deviations.

 Normalization changes the input from raw pixel values into a standardized representation.

 The important practical point comes later:

 > **Augmentation must happen before normalization.**

 Why?

 The torchvision augmentation pipeline expects normal image values approximately in:

```
[0, 1]
```

 After normalization, values are no longer restricted to `[0,1]`.

 Therefore:

```
raw image
   ↓
augmentation
   ↓
normalization
   ↓
model
```

 not:

```
raw image
   ↓
normalization
   ↓
augmentation
```

 This is an important implementation detail.

---

 # 9\. The BasicCNN

 The model is deliberately simple.

 It contains:

```
Conv2D
 ↓
ReLU
 ↓
MaxPool
 ↓
Conv2D
 ↓
ReLU
 ↓
MaxPool
 ↓
Conv2D
 ↓
ReLU
 ↓
Conv2D
 ↓
ReLU
 ↓
Fully connected
 ↓
Dropout
 ↓
Fully connected
 ↓
Dropout
 ↓
Classifier
```

 The final layer is:

```
self.classifier = nn.Linear(128, num_classes)
```

 Since CIFAR-10 has ten classes:

```
128 → 10
```

 The model therefore outputs **10 logits**.

---

 # 10\. What is a logit?

 The final model output is not directly a probability.

 For example:

```
[-1.2, 3.7, 0.2, ...]
```

 These are logits.

 Softmax converts them into probabilities:

```
P(class | x)
```

 For example:

```
airplane   0.01
car        0.02
bird       0.03
cat        0.04
dog        0.87
...
```

 The largest probability determines the predicted class.

---

 # 11\. Why train a baseline first?

 Before using semi-supervised learning, we need to answer:

 > **How well can we do using only the labelled data?**

 This is our baseline.

 The notebook trains:

```
baseline_model = BasicCNN(num_classes=10)
```

 using only:

```
trainL_loader
```

 It does **not** use the unlabelled images.

 The loss is ordinary cross entropy:

```
loss_func = nn.CrossEntropyLoss()
```

 So:

```
labelled image
      ↓
    model
      ↓
    logits
      ↓
Cross Entropy
      ↓
backpropagation
```

 The notebook expects roughly:

```
50–55% validation accuracy
```

 although the exact result depends on training.

---

 # 12\. Why calculate the full-supervision result?

 The notebook optionally trains another model using all the labels.

 This is not available in the actual semi-supervised problem.

 It is simply an **upper-bound reference**.

 We therefore have three important reference points:

```
                     performance
                         ↑

       Full labels      │       ← hypothetical upper reference
                         │
                         │
       UDA              │       ← desired improvement
                         │
       Partial labels  │       ← supervised baseline
                         │
                         └──────────────────
```

 This helps us understand whether semi-supervised learning is closing some of the gap created by missing labels.

---

 # 13\. The central idea: UDA

 The notebook now implements a simplified form of:

 **Unsupervised Data Augmentation (UDA).**

 The key idea is:

 > Apply a strong, class-preserving augmentation to unlabelled data and force the model to make similar predictions on the clean and augmented versions.

 Suppose:

```
x = clean image
T(x) = augmented image
```

 Then we want:

```
f(x) ≈ f(T(x))
```

---

 # 14\. The complete mathematical objective

 The notebook gives the objective:

 $$
L =
L_{sup}
+
\lambda_u L_{unsup}
$$

 where:

 $$
L_{sup}
=
CE(y, f(x))
$$

 and:

 $$
L_{unsup}
=
KL(P \| Q)
$$

 where:

```
P = prediction on clean image
Q = prediction on augmented image
```

 Therefore:

 $$
L =
CE(y,f(x))
+
\lambda_u
KL(P\|Q)
$$

 This equation is the **most important equation in the practical**.

---

 # 15\. What does `lambda_u` do?

 `lambda_u` controls the importance of the unsupervised loss.

 For example:

```
lambda_u = 1
```

 means:

```
total loss =
supervised loss + 1 × unsupervised loss
```

 If:

```
lambda_u = 3
```

 then:

```
total loss =
supervised loss + 3 × unsupervised loss
```

 Therefore increasing `lambda_u` makes consistency regularization more influential.

 But bigger is **not automatically better**.

 If `lambda_u` is too large, the model may strongly enforce consistency around predictions that are themselves incorrect.

 This is one of the central challenges of consistency regularization.

---

 # 16\. Task 1 — Data augmentation

 The notebook defines:

```
augmentation = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),
])
```

 There are two types of transformation.

 ### Geometric transformation

```
RandomHorizontalFlip()
```

 This changes the spatial arrangement.

 ### Photometric transformation

```
ColorJitter(...)
```

 This changes image appearance:

```
brightness
contrast
saturation
hue
```

 The key requirement is:

 > The augmentation should substantially change the pixels without changing the semantic class.

---

 # 17\. Why not use arbitrary transformations?

 Because consistency regularization assumes:

```
x and T(x) have the same class.
```

 Suppose the image is:

```
dog
```

 and our augmentation is so aggressive that the image becomes unrecognizable.

 Then we can no longer assume:

```
class(x) = class(T(x))
```

 The consistency objective becomes harmful.

 This is why the notebook warns against excessive:

 - blurring
- pixel noise
- overly destructive transformations

 especially because CIFAR-10 images are already only `32 × 32` pixels.

---

 # 18\. Why create both clean and augmented images?

 The custom collate function returns:

```
return clean_image_tensor, aug_image_tensor
```

 So one batch looks like:

```
(clean images, augmented images)
```

 For example:

```
clean:
x1 x2 x3 x4 ...

augmented:
T(x1) T(x2) T(x3) T(x4) ...
```

 The model can then calculate:

```
model(x)
```

 and:

```
model(T(x))
```

 and compare the predictions.

---

 # 19\. Why augmentation happens inside the collate function

 This is a practical design decision.

 The original images are loaded first.

 Then:

```
augmented_images = [augmentation(img) for img in images]
```

 Then both versions are normalized.

 This ensures the same original image produces:

```
clean version
+
random augmented version
```

 within the same batch.

---

 # 20\. Task 2 — The consistency loss

 This is the most technically important part.

 The model produces:

```
clean_logits
```

 and:

```
aug_logits
```

 We need to compare them.

 The notebook uses KL divergence.

 The conceptual objective is:

 $$
KL(P\|Q)
$$

 where:

```
P = clean prediction
Q = augmented prediction
```

---

 # 21\. Why use KL divergence?

 Cross entropy normally compares:

```
prediction vs true class
```

 But we don't know the true class for the unlabelled image.

 Instead, we have a probability distribution from another prediction:

```
P = [0.05, 0.02, 0.87, ...]
```

 We want the augmented prediction:

```
Q
```

 to resemble `P`.

 KL divergence measures how different two probability distributions are.

 Thus:

```
clean prediction
       ↓
   pseudo-target
       ↓
     KL
       ↑
augmented prediction
```

---

 # 22\. The clean prediction acts as a pseudo-label

 This is a crucial conceptual point.

 Suppose the model predicts:

```
clean image:

dog       0.90
cat       0.05
horse     0.02
...
```

 We don't know whether the image really is a dog.

 But the model is saying:

 > "I currently believe this is probably a dog."

 That prediction becomes a **soft pseudo-label**.

 Notice that it is _soft_.

 We don't convert it to:

```
dog = 1
everything else = 0
```

 Instead we retain the entire probability distribution:

```
dog       0.90
cat       0.05
horse     0.02
...
```

 This contains more information.

---

 # 23\. Why must the clean prediction be detached?

 This is one of the most important exam questions from the practical.

 The code uses:

```
clean_probs = F.softmax(
    clean_logits.detach(),
    dim=1
)
```

 Why?

 Because the clean prediction is being treated as the **target**.

 We want:

```
clean prediction → fixed target
augmented prediction → learns to match target
```

 If we don't detach it, gradients can flow through both branches.

 Then the model can reduce the KL divergence by changing:

```
the target
```

 as well as:

```
the prediction
```

 That creates an unstable objective.

 Conceptually:

```
WITHOUT detach:

clean model output  <---- gradients
       |
       v
      KL
       ^
       |
augmented output <---- gradients
```

 With detach:

```
clean model output
       |
       X  gradient blocked
       |
       v
    pseudo-target
       |
       v
      KL
       ^
       |
augmented output <---- gradients
```

 Therefore:

 > **Detach the clean prediction so it acts as a fixed pseudo-label during this loss calculation.**

---

 # 24\. Why is the direction of KL important?

 The mathematical objective is:

 $$
KL(P\|Q)
$$

 where:

```
P = clean prediction
Q = augmented prediction
```

 We want:

```
augmented prediction ≈ clean prediction
```

 so the clean distribution is the target.

 PyTorch's implementation is slightly unintuitive.

 The notebook explains that:

```
F.kl_div(input, target)
```

 expects:

```
input = log(Q)
target = P
```

 and therefore:

```
F.kl_div(
    aug_log_probs,
    clean_probs,
    reduction='batchmean'
)
```

 implements:

 $$
KL(P\|Q)
$$

 even though the arguments appear in the opposite order from the mathematical notation.

 This is a classic implementation trap.

---

 # 25\. Why use `log_softmax`?

 The model produces logits:

```
aug_logits
```

 but KL divergence expects log-probabilities for its input.

 Therefore:

```
aug_log_probs = F.log_softmax(
    aug_logits,
    dim=1
)
```

 converts:

```
logits
```

 into:

```
log probabilities
```

 The clean branch uses:

```
clean_probs = F.softmax(
    clean_logits.detach(),
    dim=1
)
```

 So the two branches are:

```
clean logits
    ↓
detach
    ↓
softmax
    ↓
P
```

 and:

```
augmented logits
    ↓
log_softmax
    ↓
log Q
```

 Then:

```
F.kl_div(log_Q, P)
```

 calculates:

 $$
KL(P\|Q)
$$

---

 # 26\. Why use `batchmean`?

 The notebook explicitly recommends:

```
reduction='batchmean'
```

 This makes the KL loss behave appropriately as an average over the batch.

 It avoids unwanted dependence of the loss magnitude on batch size.

 That matters particularly because the notebook discusses using different batch sizes for labelled and unlabelled data.

---

 # 27\. A subtle bug in the notebook's sanity check

 There is a small typo here:

```
assert unsup_loss.shape == (), f"divergence loss should be a scalar, but has shape: {unsup_loss_term.shape}"
```

 The variable:

```
unsup_loss_term
```

 is not defined.

 It should refer to:

```
unsup_loss
```

 For example:

```
assert unsup_loss.shape == (), \
    f"divergence loss should be a scalar, but has shape: {unsup_loss.shape}"
```

 The assertion itself is checking something sensible: KL loss should be a scalar.

---

 # 28\. Task 3 — Complete UDA training

 Now everything is combined.

 Each training iteration receives:

```
labelled batch
+
unlabelled clean/augmented batch
```

 Suppose:

```
labelled:
x, y

unlabelled:
u, T(u)
```

 The model calculates:

```
model(x)
model(u)
model(T(u))
```

 Then:

```
supervised loss = CE(model(x), y)

unsupervised loss = KL(model(u) || model(T(u)))
```

 Finally:

 $$
L =
L_{sup} + \lambda_uL_{unsup}
$$

---

 # 29\. Step-by-step through `train_uda`

 ## Step 1 — Create supervised loss

```
loss_func = nn.CrossEntropyLoss()
```

 This handles the labelled examples.

---

 ## Step 2 — Create optimizer

```
opt = torch.optim.AdamW(...)
```

 The optimizer updates model parameters.

---

 ## Step 3 — Make the unlabelled loader infinite

 The code uses:

```
trainU_iter = iter(itertools.cycle(trainU_loader))
```

 Why?

 Because the labelled and unlabelled datasets may have different numbers of batches.

 Suppose:

```
trainL = 8 batches
trainU = 70 batches
```

 If we simply zip them:

```
zip(trainL_loader, trainU_loader)
```

 we would stop after only 8 batches.

 Instead, the unlabelled loader cycles:

```
U1
U2
U3
...
U70
U1
U2
...
```

 while the labelled loader determines how many training iterations happen in an epoch.

---

 # 30\. Why the labelled loader controls the epoch?

 The training loop says:

```
for batch in trainL_loader:
```

 Therefore one epoch contains one iteration for every labelled batch.

 Each labelled batch is paired with an unlabelled batch.

 This is a practical way of ensuring that every supervised iteration gets an additional unsupervised learning signal.

---

 # 31\. Warmup epochs

 The code contains:

```
if e < warmup_epochs:
    current_lambda_u = 0.0
else:
    current_lambda_u = lambda_u
```

 Suppose:

```
warmup_epochs = 3
lambda_u = 2
```

 Then:

```
epoch 0 → λ = 0
epoch 1 → λ = 0
epoch 2 → λ = 0
epoch 3 → λ = 2
epoch 4 → λ = 2
...
```

 During warmup:

```
supervised learning only
```

 After warmup:

```
supervised + consistency learning
```

---

 # 32\. Why is warmup useful?

 Early in training, the model's predictions on unlabelled images are poor.

 Suppose the model sees:

```
dog image
```

 but initially predicts:

```
truck = 0.70
dog = 0.05
```

 If this bad prediction becomes the pseudo-label, consistency training may reinforce the wrong answer.

 This gives the fundamental problem:

 > **The unsupervised signal is only useful if the model's predictions are reasonably meaningful.**

 Warmup allows the supervised data to establish a reasonable classifier first.

---

 # 33\. The training iteration

 Inside the loop:

```
pred = model(x)
```

 gives the supervised prediction.

 Then:

```
supervised_loss = loss_func(pred, y)
```

 calculates:

 $$
L_{sup}
$$

 Next:

```
clean_logits = model(ux_clean)
aug_logits = model(ux_aug)
```

 gives the two unlabelled predictions.

 Then:

```
unsupervised_loss = unsup_loss_fn(
    clean_logits,
    aug_logits
)
```

 calculates:

 $$
L_{unsup}
$$

 Finally:

```
total_loss = supervised_loss + current_lambda_u * unsupervised_loss
```

 implements:

 $$
\boxed{
L = L_{sup} + \lambda_u L_{unsup}
}
$$

---

 # 34\. Backpropagation

 Then:

```
total_loss.backward()
opt.step()
```

 The model therefore learns from **both sources of information**.

 The supervised loss tells it:

 > "This labelled image belongs to class X."

 The consistency loss tells it:

 > "Your prediction should be stable when this unlabelled image is perturbed."

 Together:

```
labelled data
     ↓
semantic supervision
     ↓
model
     ↑
prediction stability
     ↑
unlabelled data
```

---

 # 35\. An important metric detail

 The code records:

```
metrics.log_train(
    loss=supervised_loss.item(),
    acc=batch_acc,
    unsup_loss=unsupervised_loss.item()
)
```

 Notice that `loss=` receives:

```
supervised_loss
```

 rather than:

```
total_loss
```

 So the plotted "training loss" is really the supervised loss.

 The unsupervised loss is tracked separately.

 This is useful for interpretation, but you must not confuse:

```
tracked supervised loss
```

 with:

```
actual optimization objective
```

 The optimizer minimizes:

 $$
L_{total}
=
L_{sup}
+
\lambda_uL_{unsup}
$$

---

 # 36\. Confidence filtering

 The notebook then explores an important extension.

 The problem is that not every pseudo-label is reliable.

 Suppose the clean model predicts:

```
dog = 0.97
```

 This is reasonably trustworthy.

 But suppose it predicts:

```
dog = 0.21
cat = 0.19
horse = 0.18
...
```

 That is highly uncertain.

 Using the second prediction as a pseudo-label can inject noise.

 Therefore the notebook introduces:

```
confidence_threshold = 0.8
```

---

 # 37\. How confidence is calculated

 First:

```
clean_probs = F.softmax(
    clean_logits.detach(),
    dim=1
)
```

 Then:

```
confidence = clean_probs.max(dim=1).values
```

 For each image:

 $$
confidence =
\max_k P(y=k|x)
$$

 For example:

```
Prediction:

dog       0.91
cat       0.04
horse     0.02
...
```

 gives:

```
confidence = 0.91
```

---

 # 38\. Creating the confidence mask

 The code uses:

```
mask = confidence >= confidence_threshold
```

 With:

```
confidence_threshold = 0.8
```

 we obtain something like:

```
confidence:

0.95 → True
0.87 → True
0.63 → False
0.92 → True
0.41 → False
```

 Only:

```
True
True
False
True
False
```

 examples contribute to the consistency loss.

---

 # 39\. Why confidence filtering can help

 The idea is:

 > **Only trust pseudo-labels when the model is sufficiently confident.**

 Therefore:

```
all unlabelled examples
        |
        v
 confidence filter
        |
        +---- low confidence → ignore
        |
        +---- high confidence → consistency loss
```

 This can reduce the amount of noisy unsupervised supervision.

 But there is a trade-off.

 If the threshold is too high:

```
very few examples
```

 will contribute.

 Then the unlabelled dataset provides little useful training signal.

---

 # 40\. What happens if no example passes the threshold?

 The code checks:

```
if mask.any():
```

 If no example is sufficiently confident:

```
unsupervised_loss = torch.tensor(
    0.0,
    device=device
)
```

 Thus:

```
unsupervised contribution = 0
```

 for that batch.

 The supervised loss still trains the model.

---

 # 41\. The final experiment

 The notebook performs a small hyperparameter sweep.

 It tests:

```
lambda_u   warmup
------------------
1          0
2          0
3          0
4          0
3          3
2          5
3          5
```

 This is essentially asking:

 > How strongly should we trust consistency regularization, and should we delay it?

 Each experiment uses:

```
experiment_model = BasicCNN(num_classes=10)
```

 which is important.

---

 # 42\. Why use a fresh model for every experiment?

 Suppose experiment 1 produces:

```
80% accuracy
```

 and then experiment 2 continues training that same model.

 We could no longer fairly compare the two hyperparameter settings.

 Experiment 2 would have inherited information from experiment 1.

 Instead:

```
Experiment 1 → fresh model
Experiment 2 → fresh model
Experiment 3 → fresh model
...
```

 This gives a fairer comparison.

---

 # 43\. The two experimental modes

 The notebook has:

```
USE_CONFIDENCE_THRESHOLD = False
```

 This gives:

```
standard UDA
```

 Then it can be changed to:

```
USE_CONFIDENCE_THRESHOLD = True
```

 giving:

```
UDA + confidence filtering
```

 with:

```
CONFIDENCE_THRESHOLD = 0.8
```

 So the experiments compare:

```
Standard UDA
        vs
Confidence-filtered UDA
```

 under the same hyperparameter combinations.

---

 # 44\. What result are we looking for?

 The key comparison is:

```
UDA validation accuracy
        -
baseline validation accuracy
```

 The notebook expects at least approximately:

```
+5 percentage points
```

 if consistency regularization has been implemented effectively.

 For example:

```
Baseline = 52%
UDA      = 58%
```

 would mean:

```
+6 percentage points
```

 of improvement.

 This is **not** the same as a 6% relative improvement.

---

 # 45\. Percentage points versus percent improvement

 This is worth remembering.

 If:

```
baseline = 50%
UDA = 55%
```

 then the improvement is:

```
5 percentage points
```

 Relative improvement is:

 $$
\frac{55-50}{50}=10\%
$$

 So:

```
+5 percentage points
```

 is:

```
+10% relative improvement
```

 The notebook correctly reports:

```
improvement = val_acc - partial_label_val_acc
```

 which is a difference in accuracy, i.e. percentage points when expressed as percentages.

---

 # 46\. Why the unlabelled data can improve classification

 This is the deeper machine-learning intuition.

 Imagine the labelled examples are sparse.

 They provide information such as:

```
"This is a dog."
"This is a truck."
"This is a cat."
```

 But the unlabelled dataset gives us information about the structure of the input space.

 Consistency regularization says:

```
If two inputs are nearby under a meaningful transformation,
their predictions should remain similar.
```

 This encourages smoother decision boundaries.

 Instead of allowing the model to make unstable predictions:

```
dog → dog
tiny perturbation → cat
tiny perturbation → dog
tiny perturbation → horse
```

 we encourage:

```
dog → dog
augmentation → dog
augmentation → dog
augmentation → dog
```

 The resulting classifier should be more robust.

---

 # 47\. The decision-boundary interpretation

 This is an excellent way to understand consistency regularization.

 Imagine a simplified 2D input space.

 Without consistency regularization:

```
       class A       |       class B
                     |
                     |
                     |
```

 The model might place the decision boundary in an unstable location.

 With consistency regularization, we encourage the prediction to remain stable around each data point.

 Conceptually:

```
       A A A A       |       B B B B
       A A A A       |       B B B B
       A A A A       |       B B B B
```

 The model is encouraged not to put a decision boundary through regions where perturbations of the same example should have the same prediction.

 This is closely related to the **cluster assumption** and smoothness assumptions used in semi-supervised learning.

---

 # 48\. The most important conceptual distinction

 Do not confuse **consistency regularization** with ordinary supervised augmentation.

 In ordinary supervised augmentation:

```
image + known label
        |
        v
augment image
        |
        v
CE against same known label
```

 For example:

```
augmented dog → dog
```

 because we know the label.

 In consistency regularization:

```
unlabelled image
        |
        +---- clean → prediction P
        |
        +---- augmented → prediction Q

             force P ≈ Q
```

 We don't know the actual class.

 The **model's own prediction** provides the target.

---

 # 49\. Why this is semi-supervised rather than self-supervised learning

 The training objective still includes:

```
labelled examples
```

 and their ground-truth labels.

 Therefore this is:

 > **semi-supervised learning**

 rather than purely unsupervised/self-supervised learning.

 The two components are:

```
supervised learning
+
unsupervised consistency learning
```

---

 # 50\. Why the clean prediction is a pseudo-label rather than a true label

 A true label is supplied by the dataset:

```
y = dog
```

 A pseudo-label is generated by the model:

```
model(x) → dog = 0.91
```

 Therefore pseudo-labels can be wrong.

 This explains why:

 - warmup can help,
- confidence filtering can help,
- `lambda_u` needs tuning.

---

 # 51\. The role of each major hyperparameter

 ## `lr`

```
lr = 1e-3
```

 Controls the optimizer's step size.

 Too large:

```
unstable training
```

 Too small:

```
slow learning
```

---

 ## `l2_reg`

```
l2_reg = 1e-3
```

 Controls weight decay.

 It regularizes model parameters and can reduce overfitting.

---

 ## `lambda_u`

 Controls:

```
strength of consistency loss
```

 Higher. The clean prediction is converted to a probability distribution and detached from the computation graph so that it acts as a fixed soft pseudo-label. The augmented prediction is converted to log-probabilities. KL divergence is then calculated between the clean-label. We want the augmented prediction to move toward the clean prediction, rather than allowing gradient updates to modify both the prediction and:

```
stronger use of unlabelled data
```

 but potentially:

```
more influence from incorrect pseudo-labels
```

---

 ## `warmup_epochs`

 Controls how long the model trains without consistency regularization.

 Higher:

```
more time to establish useful supervised predictions
```

 but:

```
less time exploiting unlabelled data
```

---

 ## `confidence_threshold`

 Controls how confident the model must be before an unlabelled example contributes to the consistency loss.

 Higher:

```
cleaner pseudo-labels
```

 but:

```
fewer training examples
```

---

 # 52\. Why there is a fundamental trade-off in confidence threshold

 Imagine 1,000 unlabelled examples.

 At threshold `0.5`:

```
900 examples accepted
```

 but some predictions are unreliable.

 At threshold `0.8`:

```
500 examples accepted
```

 and they are probably more reliable.

 At threshold `0.99`:

```
50 examples accepted
```

 but now we are throwing away most of the unlabelled data.

 Therefore:

```
low threshold
    ↓
more data
but more noise

high threshold
    ↓
less data
but less noise
```

 This is a classic precision-versus-coverage trade-off.

---

 # 53\. Why warmup and confidence filtering solve related problems

 Both mechanisms address the same fundamental issue:

 > **Early or uncertain model predictions may be bad pseudo-labels.**

 Warmup says:

```
Don't use pseudo-labels yet.
```

 Confidence filtering says:

```
Use pseudo-labels only when confident.
```

 They can therefore be complementary.

---

 # 54\. The entire algorithm in pseudocode

 The whole practical can be reduced to:

```
Create labelled dataset L
Create unlabelled dataset U

Train supervised baseline on L

For each training iteration:

    Get labelled batch (x, y)

    Get unlabelled batch u

    Create augmented version:
        u_aug = T(u)

    Supervised prediction:
        p = model(x)

    Supervised loss:
        L_sup = CE(p, y)

    Clean prediction:
        q = model(u)

    Augmented prediction:
        r = model(u_aug)

    Detach q

    Convert:
        q → probabilities
        r → log probabilities

    Consistency loss:
        L_unsup = KL(q || r)

    Optionally:
        keep only high-confidence q

    Total loss:
        L = L_sup + λ L_unsup

    Backpropagate L

    Update model

Validate on labelled validation data
```

 That is the entire practical.

---

 # 55\. What is actually learned from the unlabelled data?

 This is an excellent conceptual exam question.

 The model does **not** directly learn:

```
"This unlabelled image is a dog."
```

 Instead it learns:

```
"The prediction should be invariant to this
class-preserving transformation."
```

 Repeated over many unlabelled images, this provides a powerful regularization signal.

---

 # 56\. Why does the augmentation need to be class-preserving?

 Because the entire objective assumes:

 $$
y(x) = y(T(x))
$$

 If that assumption is false, the method is actively teaching the model something incorrect.

 For example, imagine an augmentation that transforms:

```
dog → something visually resembling a cat
```

 Then forcing:

```
prediction(dog) = prediction(transformed image)
```

 would be inappropriate.

 Therefore:

 > **The quality of the augmentation is critical to consistency regularization.**

---

 # 57\. Why UDA can fail

 The notebook explicitly gives several debugging clues.

 ### Problem 1 — No `detach`

 If you don't detach the clean target:

```
both sides can move
```

 and the pseudo-label itself can change.

---

 ### Problem 2 — Wrong KL direction

 If you reverse the target/prediction relationship incorrectly, you are optimizing a different objective.

 Remember:

```
mathematical:

KL(P || Q)

P = clean target
Q = augmented prediction
```

 PyTorch:

```
F.kl_div(
    log_Q,
    P
)
```

---

 ### Problem 3 — `lambda_u` too large

 The model may trust noisy consistency information too much.

---

 ### Problem 4 — Augmentation too strong

 The clean and augmented image may no longer have equivalent semantics.

---

 ### Problem 5 — Model predictions initially poor

 Bad pseudo-labels create bad consistency targets.

 Possible solution:

```
warmup
```

 or:

```
confidence filtering
```

---

 # 58\. What does "consistency" actually mean?

 It means:

 $$
f(x) \approx f(T(x))
$$

 It does **not** necessarily mean:

```
the pixels should look similar.
```

 The pixels may change considerably.

 Instead:

 > **The semantic prediction should remain stable.**

 This distinction is crucial.

---

 # 59\. The practical's learning hierarchy

 The notebook is actually teaching several increasingly sophisticated ideas:

```
Level 1
Supervised learning
       ↓
learn from labelled examples

Level 2
Semi-supervised learning
       ↓
use unlabelled examples

Level 3
Consistency regularization
       ↓
same semantic input → same prediction

Level 4
UDA
       ↓
strong augmentation + consistency

Level 5
Pseudo-label confidence
       ↓
trust reliable predictions more

Level 6
Hyperparameter tuning
       ↓
balance supervised and unsupervised signals
```

---

 # 60\. Important code-to-concept mapping

 | Code | Meaning |
| --- | --- |
| `trainL_loader` | labelled training data |
| `trainU_loader` | unlabelled training data |
| `val_loader` | labelled validation data |
| `augmentation` | class-preserving perturbation |
| `ux_clean` | clean unlabelled images |
| `ux_aug` | augmented unlabelled images |
| `clean_logits` | model prediction on clean image |
| `aug_logits` | model prediction on augmented image |
| `clean_probs` | pseudo-label distribution |
| `.detach()` | prevents target gradients |
| `F.softmax()` | logits → probabilities |
| `F.log_softmax()` | logits → log probabilities |
| `F.kl_div()` | consistency loss |
| `lambda_u` | unsupervised-loss weight |
| `warmup_epochs` | delay consistency training |
| `confidence_threshold` | reject uncertain pseudo-labels |
| `mask` | selects reliable examples |
| `total_loss` | actual optimization objective |

---

 # 61\. The three most important equations

 If you remember only three equations, remember these.

 ### Supervised loss

 $$
\boxed{
L_{sup}=CE(y,f(x))
}
$$

 ### Consistency loss

 $$
\boxed{
L_{unsup}=KL(P_{clean}\|P_{aug})
}
$$

 with:

 $$
P_{clean} = softmax(f(x))
$$

 and:

 $$
P_{aug}=softmax(f(T(x)))
$$

 ### Total loss

 $$
\boxed{
L=L_{sup}+\lambda_uL_{unsup}
}
$$

 Everything else in the practical supports these equations.

---

 # 62\. The most important implementation rule

 If you get an exam question asking:

 > "How is the unsupervised consistency loss implemented?"

 A strong answer would be:

 > The unlabelled image is passed through the model twice: once in its clean form and once after a class-preserving augmentation. The clean prediction is converted to a probability distribution and detached from the computation graph so that it acts as a fixed soft pseudo-label. The augmented prediction is converted to log-probabilities. KL divergence is then calculated between the clean probability distribution and augmented prediction, using the clean prediction as the target. This consistency loss is weighted by `lambda_u` and added to the supervised cross-entropy loss.

 That captures essentially the entire practical.

---

 # 63\. A second important exam answer

 ### Question

 **Why is the clean prediction detached?**

 ### Answer

 Because it acts as a pseudo-label. We want the augmented prediction to move toward the clean prediction, rather than allowing gradient updates to modify both the prediction and its target. Detaching prevents gradients from flowing through the clean branch during the consistency-loss calculation, producing a more stable training objective.

---

 # 64\. A third important exam answer

 ### Question

 **Why can consistency regularization exploit unlabelled data?**

 ### Answer

 Because it does not require the true class label. Instead, it uses the assumption that a class-preserving transformation should not change the semantic prediction. The model is therefore trained to produce similar predictions for clean and augmented versions of the same unlabelled example.

---

 # 65\. A fourth important exam answer

 ### Question

 **Why can confidence filtering improve UDA?**

 ### Answer

 The model's pseudo-labels can be incorrect, especially early in training. Confidence filtering only applies the consistency loss to examples for which the clean prediction has sufficiently high maximum probability. This reduces the influence of unreliable pseudo-labels, although an excessively high threshold may discard too much useful unlabelled data.

---

 # 66\. The deepest idea behind the practical

 The practical is really about this principle:

 > **Unlabelled data tells us something about where the model should behave smoothly, even when it does not tell us exactly what the correct class is.**

 The labelled examples tell us:

```
where the classes are.
```

 The unlabelled examples tell us:

```
how the decision function should behave around real data.
```

 Consistency regularization combines both.

---

 # 67\. Final mental model

 When you look at this notebook, think:

```
                    SEMI-SUPERVISED LEARNING
                              |
                +-------------+-------------+
                |                           |
          LABELLED DATA              UNLABELLED DATA
                |                           |
          true labels                 no true labels
                |                           |
          Cross Entropy             create augmentation
                |                           |
                |                    clean prediction
                |                           |
                |                    augmented prediction
                |                           |
                |                       KL divergence
                |                           |
                +-------------+-------------+
                              |
                              v
                     TOTAL LOSS
                              |
                 CE + lambda × KL
                              |
                              v
                        BACKPROPAGATE
                              |
                              v
                           MODEL
```

 And the single sentence to remember is:

 > **Consistency regularization trains a model to fit the labelled examples while producing stable predictions for different class-preserving views of unlabelled examples.**

 That is the conceptual heart of this entire notebook.
