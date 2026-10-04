# Curriculum Learning & Semi-Supervised Learning — Exam-Ready Notes

 ## 1\. Curriculum Learning

 ### Core idea

 **Curriculum learning** trains a model by controlling the **order and difficulty of training examples**.

 Instead of presenting all training examples uniformly, we can start with easier examples and progressively introduce harder ones.

 > **Key intuition:** Learn simple patterns first, then use them to learn more difficult patterns.

 ### Why use curriculum learning?

 It can:

 - Improve optimisation and training stability.
- Speed up learning.
- Help the model discover useful representations.
- Reduce the effect of noisy or difficult examples early in training.
- Potentially improve final generalisation.

 ### Important distinction

 Curriculum learning is about **how training data are presented**.

 It does **not necessarily change the model architecture or objective function**.

---

 # 2\. Semi-Supervised Learning

 ## Motivation

 Suppose we have:

 - A small labelled dataset

 $$
\mathcal{D}_L = \{(x_i,y_i)\}
$$

 - A much larger unlabelled dataset

 $$
\mathcal{D}_U = \{x_j\}
$$

 The problem is that obtaining labels can be expensive, while collecting unlabelled data is often cheap.

 **Semi-supervised learning (SSL)** attempts to use **both labelled and unlabelled data**.

---

 ## 3\. Standard Supervised Learning

 For labelled examples, the usual objective is:

 $$
\mathcal{L}_{sup} = \mathrm{CE}\left(y_i,f_\theta(x_i)\right)
$$

 where:

 - $x\_i$ = input
- $y\_i$ = true label
- $f\_\\theta(x\_i)$ = model prediction
- $\\mathrm{CE}$ = cross-entropy loss

 The model learns directly from the known labels.

---

 # 4\. Self-Training / Pseudo-Labeling

 ## Basic idea

 Use the model itself to generate labels for unlabelled examples.

 For an unlabelled example $x\_j$:

 $$
\hat y_j = \arg\max_y f_\theta(x_j)
$$

 The predicted label $\\hat y\_j$ becomes a **pseudo-label**.

 We then train using both:

 $$
\mathcal{L} = \underbrace{\mathrm{CE}(y_i,f_\theta(x_i)) }_{\text{supervised}} + \underbrace{\mathrm{CE}(\hat(y_j),f_\theta(x_j)}_{\text{unsupervised}}
$$

 ### Intuition

 The model says:

 > "I think this unlabelled example belongs to class $k$."

 Then the model is trained to become more consistent with that prediction.

---

 ## Problem: Confirmation Bias

 Naive self-training can be **fragile**.

 If the model makes a wrong prediction:

 $$
x_j \rightarrow \text{wrong pseudo-label}
$$

 that wrong label is then used as training data.

 The model can reinforce its own mistakes:

 $$
\text{wrong prediction}
\rightarrow
\text{wrong pseudo-label}
\rightarrow
\text{training on wrong label}
\rightarrow
\text{more confident wrong prediction}
$$

 This is called **confirmation bias**.

 Another failure mode is **mode collapse**, where the model may repeatedly assign examples to the same or a small number of classes.

 Therefore, effective SSL methods need a **stability mechanism**.

---

 # 5\. Co-Training

 ## Core idea

 Instead of having one model teach itself, use **two different models**.

 Each model sees a different view of the data and therefore makes partially independent errors.

 Conceptually:

 $$
\text{Model A}
\longrightarrow
\text{pseudo-labels}
\longrightarrow
\text{Model B}
$$

 and

 $$
\text{Model B}
\longrightarrow
\text{pseudo-labels}
\longrightarrow
\text{Model A}
$$

 ### Why does this help?

 If the models make independent errors, one model is less likely to reinforce exactly the same mistakes as the other.

 > **Exam point:** Co-training provides stability by using **multiple models/views that make different errors**.

---

 # 6\. Consistency Regularisation

 ## Core idea

 A model should produce **similar predictions for different perturbations or augmentations of the same input**.

 For an input $x$ and a perturbed version $x+\\eta$:

 $$
f_\theta(x)
\approx
f_\theta(x+\eta)
$$

 Therefore, we introduce an additional **consistency loss**.

 A generic objective is:

 $$
\mathcal{L}
=
\mathcal{L}_{sup}
+
\lambda \mathcal{L}_{cons}
$$

 where:

 - $\\mathcal{L}\_{sup}$ = supervised loss
- $\\mathcal{L}\_{cons}$ = consistency loss
- $\\lambda$ = weight controlling the importance of the unsupervised term

 ### Key intuition

 The model should be **invariant to small, meaningful changes in the input**.

 For example:

 $$
\text{image}
\rightarrow
\begin{cases}
\text{original image}\\
\text{augmented image}
\end{cases}
$$

 Both should produce approximately the same prediction.

---

 # 7\. $\\Pi$-Model

 The **$\\Pi$-model** is an early implementation of consistency regularisation.

 The same input is passed through the model with different perturbations/noise.

 The model is encouraged to produce consistent predictions.

 Conceptually:

 $$
f_\theta(x+\eta_1)
\approx
f_\theta(x+\eta_2)
$$

 The total loss contains:

 $$
\mathcal{L}
=
\mathcal{L}_{sup}
+
\lambda\mathcal{L}_{cons}
$$

 The important idea is not the exact architecture, but:

 > **Different noisy/augmented versions of the same input should have consistent predictions.**

---

 # 8\. UDA — Unsupervised Data Augmentation

 **Unsupervised Data Augmentation (UDA)** applies consistency training using strong data augmentation.

 For an unlabelled example:

 1. Obtain the prediction for the original input.
2. Apply a strong augmentation.
3. Predict again.
4. Force the predictions to agree.

 Conceptually:

 $$
f_\theta(x)
\approx
f_\theta(\mathrm{Aug}(x))
$$

 For labelled examples, ordinary supervised cross-entropy is sufficient:

 $$
\mathcal{L}_{labelled}
=
\mathrm{CE}(y,f_\theta(x))
$$

 For unlabelled examples, the consistency loss is applied when the model's original prediction is sufficiently confident.

 A simplified form is:

 $$
\mathcal{L}_{unlabelled}
=
\mathrm{CE}
\left(
f_\theta(x),
f_\theta(\mathrm{Aug}(x))
\right)
$$

 ### Why use a confidence threshold?

 If the original prediction is unreliable, forcing the augmented example to match it can propagate errors.

 Therefore:

 $$
\text{use consistency loss only if confidence is high}
$$

 This reduces confirmation bias.

---

 # 9\. Scaling Consistency Regularisation

 Instead of using only one augmentation, we can apply many augmentations within a batch.

 The general idea becomes:

 $$
f_\theta(x)
\approx
f_\theta(\mathrm{Aug}_1(x))
\approx
f_\theta(\mathrm{Aug}_2(x))
\approx \cdots
$$

 This encourages the model to learn representations that are robust to many perturbations.

 This idea leads naturally toward **self-supervised learning**.

---

 # 10\. Model Stability

 A central problem in semi-supervised learning is:

 > **How do we prevent the model from reinforcing its own errors?**

 Three important approaches are:

 | Method | Stability mechanism |
| --- | --- |
| Self-training | Uses pseudo-labels |
| Co-training | Two models/views teach each other |
| Consistency regularisation | Forces robustness to perturbations |
| Temporal ensembling | Averages predictions over time |
| Mean Teacher | Averages model parameters over time |

The general theme is:

 $$
\boxed{\text{stability} \Rightarrow \text{less confirmation bias}}
$$

---

 # 11\. Temporal Ensembling

 ## Problem with ordinary consistency training

 Early in training, both:

 - the prediction used as the target, and
- the current prediction

 may be unreliable.

 If the target itself is noisy, consistency training can reinforce bad predictions.

 ### Solution

 Use an **ensemble of previous predictions** as the target.

 Let $\\tilde z\_i^t$ be the smoothed prediction for example $i$ at time $t$.

 Update it using an exponential moving average (EMA):

 $$
\tilde z_i^t
=
\alpha\tilde z_i^{t-1}
+
(1-\alpha)z_i^t
$$

 where:

 - $z\_i^t$ = current prediction
- $\\tilde z\_i^{t-1}$ = previous averaged prediction
- $\\alpha$ controls how much history is retained

 The consistency target therefore incorporates predictions from previous training iterations.

---

 # 12\. Exponential Moving Average (EMA)

 The EMA equation is:

 $$
\boxed{
\tilde z_t
=
\alpha\tilde z_{t-1}
+
(1-\alpha)z_t
}
$$

 If $\\alpha$ is large, the history has greater influence.

 If $\\alpha$ is small, the current prediction has greater influence.

 ### Why is EMA useful?

 It smooths noisy updates.

 Instead of:

 $$
z_t
$$

 being used directly, we use:

 $$
\tilde z_t
$$

 which changes more gradually.

 Therefore, EMA provides a kind of **friction** against sudden noisy changes.

---

 # 13\. Why Temporal Ensembling Works

 Suppose predictions fluctuate:

 $$
z_1,\ z_2,\ z_3,\ldots
$$

 The EMA produces a smoother sequence:

 $$
\tilde z_1,\ \tilde z_2,\ \tilde z_3,\ldots
$$

 Thus:

 - Noise is reduced.
- The target becomes more stable.
- The model is less likely to chase its own short-term prediction errors.

 The same general EMA mechanism is also used in optimisation algorithms such as **Adam**.

---

 # 14\. Problems with Temporal Ensembling

 Temporal ensembling has two major limitations.

 ### 1\. Memory/storage

 We need to maintain an averaged prediction for **every training example**:

 $$
\tilde z_1,\tilde z_2,\ldots,\tilde z_N
$$

 This becomes expensive for large datasets.

 ### 2\. Slow updates

 The prediction for each example is updated only when that example is encountered.

 In the original formulation, this can effectively mean updates only once per epoch.

 Therefore, the stabilising target can **lag behind** the current model.

---

 # 15\. Mean Teacher

 The **Mean Teacher** method addresses the limitations of temporal ensembling.

 Instead of storing an EMA prediction for every example, maintain an **EMA of the model parameters**.

 There are two models:

 - **Student:** the model being trained normally.
- **Teacher:** an EMA version of the student.

 Let the student parameters be:

 $$
\theta
$$

 and the teacher parameters be:

 $$
\theta'
$$

 The teacher is updated using:

 $$
\boxed{
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta
}
$$

 Thus the teacher is an exponential moving average of the student.

---

 # 16\. Mean Teacher Objective

 The student and teacher receive perturbed versions of the same unlabelled input.

 The objective can be written as:

 $$
\mathcal{L}
=
\underbrace{
\mathrm{CE}
\left(
y_i,f_\theta(x_i)
\right)
}_{\text{supervised loss}}
+
\underbrace{
\mathrm{MSE}
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right)
}_{\text{unsupervised consistency loss}}
$$

 where:

 - $\\theta$ = student parameters
- $\\theta'$ = teacher parameters
- $\\eta$ = student-side noise/augmentation
- $\\eta'$ = teacher-side noise/augmentation

 The teacher provides a more stable target because its parameters are averaged over the student's history.

---

 # 17\. Why Mean Teacher Is More Stable

 The teacher changes more slowly than the student:

 $$
\theta'
\approx
\text{average of previous student parameters}
$$

 Therefore:

 $$
\text{student prediction}
\rightarrow
\text{stable teacher target}
$$

 rather than:

 $$
\text{student prediction}
\rightarrow
\text{equally noisy prediction target}
$$

 This reduces the risk of confirmation bias.

 ### Main advantage over temporal ensembling

 Temporal ensembling stores:

 $$
\boxed{\text{EMA prediction for every example}}
$$

 Mean Teacher stores:

 $$
\boxed{\text{EMA model parameters}}
$$

 The teacher can therefore produce predictions for any example and updates after every training step/batch.

 This scales much better to large datasets.

---

 # 18\. Timeline of Semi-Supervised Methods

 Assume:

 - labelled data $(x\_i,y\_i)$
- unlabelled data $x\_j$

 ## 18.1 Online Self-Training

 Generate a pseudo-label:

 $$
\hat y_j
=
\arg\max_y f_\theta(x_j)
$$

 Then:

 $$
\boxed{
\mathcal{L}
=
\mathrm{CE}(y_i,f_\theta(x_i))
+
\mathrm{CE}(\hat y_j,f_\theta(x_j))
}
$$

 **Key idea:** the model teaches itself.

---

 ## 18.2 Consistency Regularisation / UDA

 Add noise or augmentation:

 $$
x_j'
=
x_j+\eta
$$

 Then encourage:

 $$
f_\theta(x_j)
\approx
f_\theta(x_j')
$$

 A simplified objective:

 $$
\boxed{
\mathcal{L}
=
\mathrm{CE}(y_i,f_\theta(x_i))
+
\mathrm{CE}
\left(
f_\theta(x_j),
f_\theta(x_j+\eta)
\right)
}
$$

 **Key idea:** the model should be invariant to perturbations.

---

 ## 18.3 Temporal Ensembling

 Use a historical EMA target:

 $$
\boxed{
\mathcal{L}
=
\mathrm{CE}(y_i,f_\theta(x_i))
+
\mathrm{MSE}
\left(
f_\theta(x_j+\eta),
\tilde z_j
\right)
}
$$

 where:

 $$
\tilde z_j
=
\alpha\tilde z_j^{\,old}
+
(1-\alpha)z_j
$$

 **Key idea:** smooth the target using previous predictions.

---

 ## 18.4 Mean Teacher

 Maintain an EMA teacher model:

 $$
\theta'
=
\mathrm{EMA}(\theta)
$$

 and use:

 $$
\boxed{
\mathcal{L}
=
\mathrm{CE}(y_i,f_\theta(x_i))
+
\mathrm{MSE}
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right)
}
$$

 **Key idea:** use a slowly changing teacher model as the consistency target.

---

 # 19\. Comparison of the Main Methods

 | Method | What provides the target? | Main idea | Main weakness |
| --- | --- | --- | --- |
| Self-training | Current model | Pseudo-label unlabelled data | Confirmation bias |
| Co-training | Another model | Different views teach each other | Requires useful independent views/models |
| $\\Pi$-model | Another noisy prediction | Prediction consistency | Target can be unstable |
| UDA | Prediction on original input | Strong augmentation consistency | Requires suitable augmentations |
| Temporal Ensembling | EMA of previous predictions | Smooth target over time | Stores prediction per example; slower updates |
| Mean Teacher | EMA teacher model | Stable model-generated target | Requires maintaining teacher model |

---

 # 20\. The Central Concept: Stability

 This entire progression can be understood as a search for a **better target** for unlabelled data.

 ### Self-training

 $$
\text{current model}
\rightarrow
\text{pseudo-label}
$$

 Problem: target can be wrong.

 ### Consistency regularisation

 $$
\text{prediction on }x
\rightarrow
\text{target for prediction on }x+\eta
$$

 Problem: both predictions can be unstable.

 ### Temporal ensembling

 $$
\text{historical predictions}
\rightarrow
\text{EMA target}
$$

 Better, but requires storing targets for every example.

 ### Mean Teacher

 $$
\text{EMA model}
\rightarrow
\text{stable target}
$$

 This gives a scalable and effective consistency-based approach.

---

 # 21\. Active Learning

 **Active learning** is related to, but different from, semi-supervised learning.

 ### Semi-supervised learning

 We have:

 - labelled examples
- unlabelled examples

 and try to exploit the unlabelled examples **without necessarily obtaining new labels**.

 ### Active learning

 The algorithm **strategically selects informative unlabelled examples** and sends them for human labelling.

 The loop is:

 $$
\text{unlabelled data}
\rightarrow
\text{select informative samples}
\rightarrow
\text{human labels them}
\rightarrow
\text{retrain}
$$

 ### Key distinction

 > **Semi-supervised learning:** make better use of labels we already have plus unlabelled data.

 > **Active learning:** decide **which new examples should be labelled**.

---

 # 22\. Curriculum Learning vs Semi-Supervised Learning vs Active Learning

 | Technique | Main question |
| --- | --- |
| Curriculum learning | **In what order should we train on the data?** |
| Semi-supervised learning | **How can we use unlabelled data?** |
| Active learning | **Which examples should we ask humans to label?** |

These techniques can also be combined.

---

 # 23\. High-Yield Exam Concepts

 ## Know these definitions

 ### Curriculum learning

 Training strategy that controls the order/difficulty of examples, often progressing from easier to harder examples.

 ### Semi-supervised learning

 Learning from a combination of labelled and unlabelled data.

 ### Self-training

 The model generates pseudo-labels for unlabelled examples and trains on them.

 ### Confirmation bias

 The model reinforces its own incorrect predictions through self-generated training targets.

 ### Co-training

 Multiple models/views teach each other using their predictions on unlabelled data.

 ### Consistency regularisation

 Force predictions to remain similar under perturbations or augmentations of the same input.

 ### Temporal ensembling

 Use an EMA of previous predictions as a more stable consistency target.

 ### Mean Teacher

 Use an EMA of model parameters to create a slowly changing teacher model.

 ### Active learning

 Select informative unlabelled examples to obtain new human labels.

---

 # 24\. Essential Equations to Memorise

 ## Supervised learning

 $$
\boxed{
\mathcal{L}_{sup}
=
\mathrm{CE}(y_i,f_\theta(x_i))
}
$$

 ## Self-training

 $$
\boxed{
\hat y_j
=
\arg\max_y f_\theta(x_j)
}
$$

 and:

 $$
\boxed{
\mathcal{L}
=
\mathcal{L}_{sup}
+
\mathrm{CE}(\hat y_j,f_\theta(x_j))
}
$$

 ## Consistency regularisation

 $$
\boxed{
f_\theta(x)
\approx
f_\theta(x+\eta)
}
$$

 ## Temporal ensembling

 $$
\boxed{
\tilde z_t
=
\alpha\tilde z_{t-1}
+
(1-\alpha)z_t
}
$$

 ## Mean Teacher

 $$
\boxed{
\theta'
=
\alpha\theta'
+
(1-\alpha)\theta
}
$$

 and:

 $$
\boxed{
\mathcal{L}
=
\mathrm{CE}(y_i,f_\theta(x_i))
+
\mathrm{MSE}
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right)
}
$$

---

 # 25\. Exam-Style "Explain the Difference" Answers

 ### Self-training vs consistency regularisation

 **Self-training** creates a pseudo-label from the model's prediction and treats it as a target.

 **Consistency regularisation** does not necessarily require a hard class label. Instead, it forces predictions for different perturbations of the same input to agree.

---

 ### Temporal ensembling vs Mean Teacher

 Both use **EMA smoothing** to obtain a more stable target.

 Temporal ensembling maintains an EMA prediction **for every training example**.

 Mean Teacher maintains an EMA of the **model parameters**, allowing the teacher to generate stable predictions for examples as they are encountered.

 Therefore Mean Teacher scales better to large datasets.

---

 ### Self-training vs Co-training

 Self-training uses **one model to teach itself**.

 Co-training uses **multiple models/views to teach each other**, ideally exploiting independent errors.

---

 ### Semi-supervised vs active learning

 Semi-supervised learning tries to exploit existing unlabelled data.

 Active learning chooses which unlabelled examples should receive **new human labels**.

---

 # 26\. Common Exam Traps

 ### Trap 1: "Pseudo-labels are always correct."

 False.

 Pseudo-labels can be wrong, causing confirmation bias.

---

 ### Trap 2: "Consistency means the model should always predict the same output."

 Not exactly.

 It means the prediction should be stable under **appropriate perturbations/augmentations that should not change the semantic class**.

---

 ### Trap 3: "Temporal ensembling averages model weights."

 False.

 Temporal ensembling averages **predictions**.

 Mean Teacher averages **model parameters**.

---

 ### Trap 4: "Mean Teacher stores a target for every training example."

 False.

 That is the key issue with temporal ensembling.

 Mean Teacher stores an EMA **teacher model**.

---

 ### Trap 5: "The teacher is trained using the supervised loss."

 The **student** is directly optimised using gradient descent.

 The teacher is updated through EMA:

 $$
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta
$$

---

 ### Trap 6: "Active learning is just another form of semi-supervised learning."

 Not exactly.

 Active learning actively obtains **new labels**, whereas semi-supervised learning exploits existing unlabelled data.

---

 # 27. One-Minute Revision Summary

 If you only have one minute before the exam, remember:

 $$
\boxed{
\text{SSL}
=
\text{labelled data}
+
\text{unlabelled data}
}
$$

 ### Self-training

 > **Model labels its own data.**

 $$
x
\rightarrow
\hat y
\rightarrow
\text{train}
$$

 Problem:

 $$
\boxed{\text{confirmation bias}}
$$

 ### Co-training

 > **Models teach each other.**

 $$
A\leftrightarrow B
$$

 ### Consistency regularisation

 > **Same input, different perturbation → same prediction.**

 $$
f(x)\approx f(x+\eta)
$$

 ### Temporal ensembling

 > **Average predictions over time.**

 $$
\tilde z_t
=
\alpha\tilde z_{t-1}
+
(1-\alpha)z_t
$$

 Problem:

 $$
\boxed{\text{storage + stale targets}}
$$

 ### Mean Teacher

 > **Average the model instead of its predictions.**

 $$
\theta'
=
\alpha\theta'
+
(1-\alpha)\theta
$$

 Then:

 $$
\boxed{
\text{Student}
\rightarrow
\text{Teacher target}
}
$$

 ### Active learning

 > **Choose which examples should get human labels.**

---

 # 28\. Final Conceptual Picture

 The progression is:

 $$
\boxed{
\text{Self-training}
\rightarrow
\text{Consistency}
\rightarrow
\text{Temporal Ensembling}
\rightarrow
\text{Mean Teacher}
}
$$

 with the central problem being:

 $$
\boxed{
\text{How can we use unlabelled data without reinforcing errors?}
}
$$

 The answer becomes increasingly sophisticated:

 $$
\begin{aligned}
\text{Self-training}
&:\quad \text{use predictions as labels}\\
\text{Consistency}
&:\quad \text{use invariance to perturbations}\\
\text{Temporal Ensembling}
&:\quad \text{smooth predictions over time}\\
\text{Mean Teacher}
&:\quad \text{smooth model parameters over time}
\end{aligned}
$$

 The overarching principle is:

 $$
\boxed{
\text{Better stability}
\Rightarrow
\text{more reliable unsupervised targets}
\Rightarrow
\text{better semi-supervised learning}
}
$$

 This is ready to save directly as something like `curriculum-and-semi-supervised-learning.md`. GitHub will render the `$...$` and `$$...$$` math delimiters as LaTeX math in supported Markdown rendering.
