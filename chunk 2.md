## Lecture 3 — Chunk 2: Self-Supervised Learning & Data Augmentation

This section continues the idea of **self-supervised learning (SSL)** and focuses heavily on **pretext tasks, shortcuts/spurious signals, and data augmentation**.

### 1\. Self-supervised learning as transfer learning

Self-supervised learning usually isn't the final goal.

Instead, we use it to learn a good **feature extractor / encoder** from large amounts of unlabeled data:

**Unlabeled data → self-supervised pre-training → useful representation → supervised fine-tuning → downstream task**

So SSL can be viewed as a special form of **transfer learning**:

- **Pre-training:** self-supervised, no human labels.
- **Fine-tuning:** supervised, using labels for the actual task.

The important idea is that the model learns **general-purpose features** during pre-training that can later be reused.

---

## 2\. How do we evaluate self-supervised learning?

A key problem:

> A model doing well on the pretext task does **not necessarily mean** it learned useful representations.

For example, suppose the pretext task is to reconstruct a missing part of an image. The model might become extremely good at reconstruction while learning mostly **low-level details** that are useless for classification.

Therefore:

**Don't primarily evaluate SSL by pretext-task performance.**

Instead:

1. Pre-train the model using the self-supervised task.
2. Transfer the learned representation to a downstream supervised task.
3. Evaluate performance on that downstream task.

If the representation is genuinely useful, it should help with tasks that weren't directly part of the pretext objective.

---

# 3\. Pretext tasks

A **pretext task** is an artificially constructed task where the training signal comes from the data itself rather than human-provided labels.

The goal isn't necessarily to solve the pretext task for its own sake.

The goal is:

> **Force the model to learn useful representations while solving the pretext task.**

Several examples are discussed.

### Image colorisation

Give the model a grayscale image and ask it to predict the missing colors.

The model needs to understand things about the image to make sensible predictions.

For example:

**grayscale image → model → predicted colors**

Potentially useful knowledge includes recognizing objects, materials, environments, etc.

---

### Image inpainting

Remove part of an image and ask the model to reconstruct the missing region.

For example:

**original:** 🏠🌳🚗 **corrupted:** 🏠⬜🚗 **prediction:** reconstruct the missing region.

This can teach useful visual representations.

However, there is a problem:

### Generative tasks can waste computation

A model may spend a lot of capacity learning to reproduce **low-level visual details** rather than learning high-level semantic information.

For example, accurately reconstructing exact textures or pixel values isn't necessarily useful for recognizing objects.

So sometimes we'd rather design a task that directly encourages the desired high-level features.

---

# 4\. Spatial-context tasks

One example is a **jigsaw-puzzle-style task**.

Parts of an image are shuffled, and the model must determine their correct spatial arrangement.

This can encourage learning:

- **Geometry**
- **Compositionality**
- **World knowledge**
- Spatial relationships

The important principle is:

> Instead of asking the model to reconstruct pixels, design a task that forces it to understand the relationships we care about.

---

# 5\. The major danger: shortcut learning

This is one of the **most important concepts in this section**.

A good pretext task should force the model to learn useful features.

But there may be a much easier solution:

> The model finds a **shortcut** that solves the pretext task without learning the intended concept.

These unwanted clues are called **spurious signals**.

### Example: image patches

Suppose we shuffle image patches and ask the model to put them back into the correct positions.

We want it to understand:

> "This patch belongs next to that patch because of the visual content."

But the model might discover another clue.

For example, the edges of the patches could contain subtle information about their original position.

Then the model can solve the task without learning meaningful visual structure.

---

## 6\. Chromatic aberration: a particularly sneaky shortcut

The lecture gives an even more surprising example.

The model discovered **chromatic aberration**—tiny differences between color channels caused by the camera lens.

Humans might not notice it.

But a CNN can detect these tiny statistical patterns.

Therefore:

**Intended solution:**

> Understand image content → determine spatial relationships.

**Actual shortcut:**

> Detect tiny camera artifacts → determine spatial relationships.

This is a classic example of why designing SSL objectives is difficult.

### The general lesson

We want:

> **High-level features**

and not:

> **Low-level shortcuts**

In other words:

**Good pretext task = difficult to solve without learning the desired representation.**

---

# 7\. Data augmentation

Data augmentation is extremely important in self-supervised learning.

In ordinary supervised learning, augmentation can:

- compensate for small datasets,
- act as regularization,
- encourage representation stability.

In self-supervised learning, augmentation has an additional role:

> It can help the model learn **robust and invariant features**.

---

## 8\. What makes a good augmentation?

A useful transformation should generally preserve the **semantics** we care about.

For example:

🐱 → darker 🐱

Still a cat.

🐱 → flipped 🐱

Probably still a cat.

🐱 → cropped 🐱

Probably still a cat.

So transformations can modify the raw pixels while preserving the underlying meaning.

The important distinction is:

**Pixel-level information changes, but semantic identity stays the same.**

---

# 9\. Common data augmentations

### Geometric transformations

Examples include:

- Rotation
- Translation
- Cropping
- Flipping
- Scaling

For classification, these are often straightforward.

For **object detection** or **segmentation**, however, we must also transform the spatial labels.

For example, if we flip an image horizontally, the object's:

- bounding box
- segmentation mask

must also be flipped.

---

### Colour distortion

Examples include:

- Hue changes
- Saturation changes
- Brightness changes
- Colour jitter
- Converting to grayscale

These encourage the model not to rely too heavily on exact color values.

---

### Noise and occlusion

Examples include:

- Gaussian noise
- Image filters
- Random erasing
- Occlusion

These make the input harder while encouraging the model to use robust features.

---

# 10\. Cutout / Random Erasing

**Cutout** randomly removes a region of the image.

Why can this be surprisingly effective?

Suppose a model normally recognizes a dog using one particular feature.

If we add mild Gaussian noise, the model might still use that same feature.

But if we **erase the feature completely**, the model has to find another useful feature.

Therefore:

> **Cutout encourages the model to rely on diverse sets of features.**

This improves robustness.

A useful intuition:

**Without Cutout:**

> "I recognize a dog because I see this one feature."

**With Cutout:**

> "That feature might disappear, so I need several independent clues."

---

# 11\. Combining augmentations

We don't have to apply only one transformation.

We can **chain multiple random augmentations** together.

For example:

**Original image**

→ random crop → horizontal flip → color jitter → grayscale → random erasing

This produces much more varied training examples and can encourage stronger representations.

---

# 12\. Label-preserving transformations

The fundamental idea behind conventional augmentation is:

> **Change the input without changing its label.**

If:

\[ (x,y) \]

is a training example, then an augmentation (T) should ideally produce:

\[ (T(x),y) \]

where the label remains valid.

For example:

\[ \\text{cat image} \\rightarrow \\text{flipped cat image} \]

Both have label:

\[ y=\\text{cat} \]

---

# 13\. What about transformations that change the label?

This leads to **label-mixing augmentations**.

Sometimes we deliberately change the image **and also change the label accordingly**.

The major examples here are:

- **MixUp**
- **CutMix**

---

## 14\. MixUp

MixUp takes two training examples and linearly interpolates them.

Suppose we have:

\[ (x_1,y_1\) \]

and

\[ (x_2,y_2\) \]

We create:

\[ x'=\\lambda x_1+(1-\\lambda)x_2 \]

and simultaneously:

\[ y'=\\lambda y_1+(1-\\lambda)y_2 \]

So both the **image and label** are mixed.

For example:

**Cat + Dog → image containing a mixture of cat and dog**

with something like:

\[ y' = 0.7(\\text{cat})+0.3(\\text{dog}) \]

---

## 15\. Why does MixUp help?

The lecture highlights several ideas.

### Label smoothing

It prevents the model from being excessively confident.

### Smooth feature representations

Mixed examples encourage the model to behave smoothly between different training examples.

### Local linearity

The model learns approximately smooth/linear behavior between nearby representations.

### Feature reuse

Features useful for multiple classes can be reused efficiently.

The lecture notes that the effectiveness of MixUp is somewhat surprising, but empirically it works very well.

---

# 16\. CutMix

CutMix combines ideas from **Cutout + MixUp**.

Instead of blending the entire images, we:

1. Take a region from image A.
2. Paste it into image B.
3. Adjust the label according to how much of each image remains.

Conceptually:

**Image A + patch from Image B → mixed image**

The label is also mixed according to the area occupied by each image.

For example, if 75% of the resulting image comes from A and 25% from B:

\[ y'=0.75y_A+0.25y_B \]

The lecture notes that this can work **extremely well**.

---

# 17\. Augmentation in self-supervised learning

Here's an important distinction.

In supervised learning, we can say:

> "This transformation preserves the label."

But in self-supervised learning:

> **There may be no labels at all.**

Therefore, we don't literally have a label-preserving transformation.

Instead, we can think about:

> **Semantics-preserving transformations.**

The purpose is to transform the input in a way that helps create a useful **proxy objective**.

So:

**Supervised augmentation**

\[ \\text{preserve label} \]

while SSL augmentation is more like:

\[ \\text{preserve useful semantics} \]

and use the transformation to construct a training signal.

---

# 18\. AutoAugment

A natural question is:

> How do we know which augmentations to use?

**AutoAugment** attempts to learn the augmentation policy automatically.

It uses **reinforcement learning** to search for a good sequence of transformations for a particular dataset.

Conceptually:

\[ \\text{Dataset} \\rightarrow \\text{search augmentation policies} \\rightarrow \\text{best policy} \]

The problem is that this search can be **very computationally expensive**.

And if every dataset needs its own search, it's impractical.

---

# 19\. RandAugment

**RandAugment** simplifies this idea.

Instead of learning a complicated policy separately for every dataset, it uses a generic strategy:

> Randomly select a small number of transformations from a predefined list.

Two important parameters are:

### (N): number of transformations

Typically around **2–3**.

### (M): magnitude

Controls how strongly the transformations are applied.

So conceptually:

\[ \\text{Image} \\xrightarrow\[\\text{magnitude }M\]{N\\text{ random transforms}} \\text{Augmented image} \]

The advantage is that RandAugment is much simpler and cheaper than running reinforcement learning to find a custom policy.

---

# 20\. Weak vs strong augmentation

The lecture makes an important distinction.

### Weak augmentation

Primarily useful as **regularization**.

It can help prevent overfitting and can be useful when we need relatively stable targets, such as in some semi-supervised methods.

### Strong augmentation

Encourages the model to learn **invariances**.

This is especially useful for self-supervised consistency-based approaches.

In other words:

**Weak augmentation → "Don't overfit."**

**Strong augmentation → "Learn what information doesn't matter."**

---

# 21\. Temporal context as a pretext task

Self-supervised learning isn't limited to static images.

For videos, we can exploit **time**.

For example:

> Predict the next frame.

This can force the model to learn:

- World knowledge
- Physics
- Temporal relationships
- Temporal awareness

The intuition is powerful:

If you want to predict what happens next, you need some understanding of **what is happening now**.

---

# 22\. Language models and self-supervision

Language is particularly well suited to self-supervised learning.

Why?

Because:

> **Unlabeled text is extremely easy to obtain, while labeled text is expensive.**

Imagine trying to create the equivalent of a huge supervised dataset for a language model.

You would need humans to provide high-quality answers to enormous numbers of complex questions across virtually every area of knowledge.

That is extremely expensive.

Instead, we can learn from the text itself.

This is one of the fundamental ideas behind modern language models.

---

# 23\. Masked word prediction

A simple language-model pretext task is:

> Hide a word and ask the model to predict it.

For example:

> "I put my hat on my \_."

The model should predict:

\[ \\boxed{\\text{head}} \]

Another example:

> "I opened the door and got in my \_."

Possible prediction:

\[ \\boxed{\\text{truck}} \]

The important point is that **the text itself provides the training signal**.

There is no human annotator saying:

> "The answer to this example is 'head'."

The sentence already contains the answer.

That's the essence of **self-supervision**.

---

# ⭐ Key concepts to remember

For an exam, I would focus particularly on these:

| Concept | Core idea |
| --- | --- |
| **Self-supervised learning** | Learn representations without human labels |
| **Transfer learning** | SSL pre-training → supervised downstream task |
| **Pretext task** | Artificial task whose solution provides a learning signal |
| **Downstream evaluation** | Test whether learned representations help another task |
| **Spurious signal** | Unwanted shortcut that solves the pretext task |
| **Shortcut learning** | Model solves the task without learning intended semantics |
| **Data augmentation** | Modify inputs to improve robustness/representation learning |
| **Semantic preservation** | Augmentation should ideally preserve relevant meaning |
| **Cutout** | Randomly erase regions to encourage diverse features |
| **MixUp** | Linearly mix images **and labels** |
| **CutMix** | Paste a region from one image into another and mix labels |
| **AutoAugment** | Uses RL to learn augmentation policies |
| **RandAugment** | Randomly applies a small number of transforms with controlled magnitude |
| **Weak augmentation** | Mainly regularization |
| **Strong augmentation** | Encourages invariance |
| **Temporal pretext tasks** | Learn from temporal relationships, e.g. predicting future frames |
| **Masked word prediction** | Predict missing words using surrounding context |

### The big picture

The whole chunk can be reduced to one central principle:

> **A self-supervised task is only useful if solving it forces the model to learn representations that transfer to something we actually care about.**

That leads to the main design challenge:

\[ \\boxed{\\text{Design a task that is difficult to solve using shortcuts}} \]

and instead:

\[ \\boxed{\\text{forces useful high-level representations}} \]

This idea—**avoiding shortcuts while designing a useful pretext objective**—is one of the most important takeaways from this part of the lecture.
