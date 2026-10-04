# Curriculum Learning & Semi-Supervised Learning — Exam-Ready Notes

---

 # 1\. Big Picture

 This lecture covers two related ideas for improving learning when training data are imperfect or expensive to label:

 ### Curriculum Learning

 **Idea:** Control the order in which training examples are presented.

 > Start with easier / higher-quality examples and gradually introduce harder / noisier examples.

 The motivation is that neural networks can learn more effectively when the training problem is structured rather than presented as a completely random mixture of examples.

 ### Semi-Supervised Learning

 **Idea:** Train using a combination of:

 - a small amount of **labelled data**: $(x_i,y_i)$
- a large amount of **unlabelled data**: $x_j$

 The goal is to extract useful information from the unlabelled examples instead of throwing them away.

---

 # 2\. Why Semi-Supervised Learning?

 Suppose we have:

 $$
\mathcal D_L=\{(x_i,y_i)\}
$$

 for labelled data and

 $$
\mathcal D_U=\{x_j\}
$$

 for unlabelled data.

 Ordinary supervised learning minimises:

 $$
\mathcal L_{\text{sup}}
=
\operatorname{CE}(y_i,f_\theta(x_i))
$$

 Only labelled examples contribute directly to the loss.

 But often:

 $$
|\mathcal D_U| \gg |\mathcal D_L|
$$

 because collecting inputs is cheap while obtaining reliable labels is expensive.

 Semi-supervised learning attempts to exploit the structure of $\mathcal D_U$.

 A generic objective is:

 $$
\boxed{
\mathcal L
=
\mathcal L_{\text{sup}}
+
\lambda\mathcal L_{\text{unsup}}
}
$$

 where:

 - $\mathcal L_{\text{sup}}$: ordinary supervised loss
- $\mathcal L_{\text{unsup}}$: constraint derived from unlabelled data
- $\lambda$: controls the importance of the unsupervised objective.

---

 # 3\. Central Problem: Confirmation Bias

 The main difficulty is that the model has to generate information about data for which we do not know the correct answer.

 For example:

 $$
x_j
\rightarrow f_\theta(x_j)
\rightarrow \hat y_j
$$

 We might treat:

 $$
\hat y_j = \arg\max_y f_\theta(x_j)_y
$$

 as a pseudo-label.

 But what if the model is wrong?

 Then we train on its own mistake.

 The process can become:

 $$
\boxed{
\text{wrong prediction}
\rightarrow
\text{pseudo-label}
\rightarrow
\text{training}
\rightarrow
\text{more confident wrong prediction}
}
$$

 This is **confirmation bias**.

 It can eventually cause **mode collapse**, where the model becomes biased toward a small number of classes or predictions.

 Therefore, effective semi-supervised learning needs a **stability mechanism**.

---

 # 4\. Self-Training / Pseudo-Labeling

 The simplest approach is **self-training**.

 ### Procedure

 1. Train the model on labelled data.
2. Run the model on unlabelled data.
3. Use its predictions as pseudo-labels.
4. Train using both real and pseudo-labels.
5. Repeat.

 For labelled data:

 $$
\mathcal L_L
=
\operatorname{CE}(y_i,f_\theta(x_i))
$$

 For an unlabelled example $x_j$:

 $$
\hat y_j
=
\arg\max_y f_\theta(x_j)_y
$$

 and:

 $$
\mathcal L_U
=
\operatorname{CE}(\hat y_j,f_\theta(x_j))
$$

 Thus:

 $$
\boxed{
\mathcal L
=
\operatorname{CE}(y_i,f_\theta(x_i))
+
\operatorname{CE}(\hat y_j,f_\theta(x_j))
}
$$

 The first term is the **supervised loss**.

 The second term is the **unsupervised/pseudo-label loss**.

---

 ## Online Self-Training

 The model can generate pseudo-labels during training rather than creating a fixed pseudo-labelled dataset.

 Conceptually:

 $$
\boxed{
\hat y_j = \arg\max f_\theta(x_j)
}
$$

 and then:

 $$
\mathcal L
=
\operatorname{CE}(y_i,f_\theta(x_i))
+
\operatorname{CE}(\hat y_j,f_\theta(x_j))
$$

 ### Main weakness

 The model trains on its own predictions.

 Therefore:

 > **Self-training is simple but fragile because prediction errors can reinforce themselves.**

---

 # 5\. Co-Training

 A solution to the stability problem is **co-training**.

 Instead of having one model teach itself, use two different models.

 The models see different **views** of the data and ideally make somewhat independent errors.

 For example:

 $$
x
\rightarrow
\begin{cases}
f_{\theta_1}(x^{(1)})\\
f_{\theta_2}(x^{(2)})
\end{cases}
$$

 One model can provide pseudo-labels for the other.

 ### Key idea

 $$
\boxed{
\text{Model A teaches Model B}
}
$$

 and:

 $$
\boxed{
\text{Model B teaches Model A}
}
$$

 rather than:

 $$
\text{Model A teaches itself}
$$

 ### Why does this help?

 If the models make independent errors, one model's mistakes are less likely to be identical to the other's.

 This provides a **stability mechanism** against confirmation bias.

---

 # 6\. Consistency Regularisation

 A more general and very important idea is **consistency regularisation**.

 ### Core principle

 If two inputs are different versions of the **same underlying example**, the model should make the same prediction for both.

 Suppose:

 $$
x
$$

 is the original input and:

 $$
x' = \operatorname{augment}(x)
$$

 is an augmented version.

 Then we want:

 $$
\boxed{
f_\theta(x)
\approx
f_\theta(x')
}
$$

 The model should be **invariant to the augmentation/noise**.

---

 ## Consistency Loss

 For example:

 $$
\mathcal L_{\text{cons}}
=
\operatorname{CE}
\left(
f_\theta(x),
f_\theta(x')
\right)
$$

 or:

 $$
\mathcal L_{\text{cons}}
=
\operatorname{MSE}
\left(
f_\theta(x),
f_\theta(x')
\right)
$$

 depending on the method.

 The overall objective becomes:

 $$
\boxed{
\mathcal L
=
\mathcal L_{\text{sup}}
+
\lambda\mathcal L_{\text{cons}}
}
$$

 ### Intuition

 The model is told:

 > "I don't know the true label for this example, but I do know that changing it slightly shouldn't change its prediction."

 This extracts useful information from unlabelled data without explicitly assigning a hard label.

---

 # 7\. Π-Model

 An early implementation of consistency regularisation is the **Π-model**.

 The same model receives different noisy versions of the same input.

 For example:

 $$
x+\eta_1
$$

 and:

 $$
x+\eta_2
$$

 The model should produce similar predictions:

 $$
f_\theta(x+\eta_1)
\approx
f_\theta(x+\eta_2)
$$

 The consistency loss encourages:

 $$
\boxed{
\mathcal L_{\text{cons}}
=
\operatorname{MSE}
\left(
f_\theta(x+\eta_1),
f_\theta(x+\eta_2)
\right)
}
$$

 The supervised loss is still used whenever a label is available.

---

 # 8\. UDA — Unsupervised Data Augmentation

 **UDA** = **Unsupervised Data Augmentation**.

 The idea is to compare predictions for:

 - the original/unmodified input
- a strongly augmented version.

 Conceptually:

 $$
x_j
\rightarrow f_\theta(x_j)
$$

 and:

 $$
\operatorname{Augment}(x_j)
\rightarrow
f_\theta(\operatorname{Augment}(x_j))
$$

 Then enforce:

 $$
\boxed{
f_\theta(x_j)
\approx
f_\theta(\operatorname{Augment}(x_j))
}
$$

---

 ## Confidence Thresholding

 A major detail in UDA is that consistency is only enforced when the prediction on the original example is sufficiently confident.

 For example:

 $$
\max_y f_\theta(x_j)_y > \tau
$$

 where $\tau$ is a confidence threshold.

 Only then do we apply the unsupervised consistency loss.

 ### Why?

 Because an uncertain prediction is a poor target.

 If:

 $$
f_\theta(x)=[0.34,0.33,0.33]
$$

 then forcing the augmented example to reproduce that prediction is not particularly useful.

 But if:

 $$
f_\theta(x)=[0.98,0.01,0.01]
$$

 then the prediction provides a much stronger target.

---

 # 9\. Scaling Consistency Regularisation

 Instead of applying consistency to only one augmentation, we can generate many augmentations within a batch.

 The model is encouraged to produce consistent predictions across all versions.

 Conceptually:

 $$
x
\rightarrow
\{x^{(1)},x^{(2)},...,x^{(K)}\}
$$

 and:

 $$
f(x^{(1)})
\approx
f(x^{(2)})
\approx
\cdots
\approx
f(x^{(K)})
$$

 This provides a powerful constraint on the representation learned by the model.

---

 # 10\. Model Stability

 This is one of the most important conceptual themes of the lecture.

 ### Naive self-training

 $$
\boxed{\text{fragile}}
$$

 because the model can reinforce its own mistakes.

 ### Co-training

 $$
\boxed{\text{two models provide mutual supervision}}
$$

 and can reduce correlated errors.

 ### Consistency regularisation

 $$
\boxed{\text{prediction should be stable under perturbations}}
$$

 This provides resistance against unstable predictions.

 ### Overall principle

 > Semi-supervised learning works better when there is some mechanism preventing noisy predictions from becoming self-reinforcing targets.

---

 # 11\. Temporal Ensembling

 Consistency regularisation has another problem.

 Early in training, both the prediction and the target can be unreliable.

 If we simply compare:

 $$
f_\theta(x)
$$

 with another prediction from the current model, we may be forcing the model to match its own noisy predictions.

 ### Solution

 Use an **ensemble of previous predictions** as a more stable target.

 For example, maintain:

 $$
\tilde z_i^t
$$

 for example $i$.

 Update it using an exponential moving average:

 $$
\boxed{
\tilde z_i^t
=
\alpha\tilde z_i^{t-1}
+
(1-\alpha)z_i^t
}
$$

 where:

 - $z_i^t$: current prediction
- $\tilde z_i^{t-1}$: previous averaged prediction
- $\alpha$: smoothing parameter.

 The lecture notes give approximately:

 $$
\alpha\approx0.6
$$

 for the cited implementation.

---

 # 12\. Why EMA?

 EMA = **Exponential Moving Average**.

 The newest prediction contributes:

 $$
1-\alpha
$$

 while previous predictions continue to influence the target.

 Repeated expansion gives:

 $$
\tilde z_t
=
(1-\alpha)z_t
+
\alpha(1-\alpha)z_{t-1}
+
\alpha^2(1-\alpha)z_{t-2}
+\cdots
$$

 Therefore, older predictions receive exponentially decreasing weights.

 Hence:

 $$
\boxed{\text{EMA = exponentially weighted moving average}}
$$

---

 # 13\. Why Temporal Ensembling Helps

 Suppose predictions fluctuate:

 $$
0.9,\;0.6,\;0.85,\;0.7,\;0.9,\ldots
$$

 Instead of trusting the latest noisy prediction completely, EMA smooths the sequence.

 This gives:

 - less sensitivity to noise,
- more stable targets,
- resistance to sudden bad updates,
- a kind of "friction" against rapid changes.

 ### Key intuition

 > The teacher target should not change too quickly.

---

 # 14\. Connection to Adam

 The same basic EMA mechanism appears in optimisers such as **Adam**.

 The reason is similar:

 > Smooth noisy quantities over time instead of reacting completely to every individual update.

 So when you see:

 $$
m_t
=
\beta m_{t-1}
+
(1-\beta)x_t
$$

 think:

 $$
\boxed{\text{EMA / smoothing}}
$$

---

 # 15\. Problem with Temporal Ensembling

 Temporal ensembling has two important weaknesses.

 ## 1\. Storage

 We need to maintain an averaged prediction:

 $$
\tilde z_i
$$

 for **every training example**.

 If the dataset contains millions of examples, this becomes expensive.

 ## 2\. Slow updates

 The averaged prediction is updated only when the corresponding example is encountered.

 The lecture highlights that it effectively updates once per epoch.

 Therefore:

 > The target can lag behind the current state of the model.

 This motivates **Mean Teacher**.

---

 # 16\. Mean Teacher

 Mean Teacher is one of the most important methods in this lecture.

 Instead of maintaining an EMA prediction for every individual example, maintain an **EMA of the model parameters**.

 There are two models:

 ### Student

 The model being trained:

 $$
f_\theta
$$

 ### Teacher

 An exponential moving average of the student:

 $$
f_{\theta'}
$$

 where:

 $$
\boxed{
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta
}
$$

 The teacher is therefore a smoothed version of the student.

---

 # 17\. Student–Teacher Architecture

 The student and teacher receive perturbed versions of the same unlabelled example:

 $$
x+\eta
$$

 and:

 $$
x+\eta'
$$

 Then:

 $$
f_\theta(x+\eta)
$$

 is encouraged to match:

 $$
f_{\theta'}(x+\eta')
$$

 The teacher provides the target.

---

 # 18\. Mean Teacher Loss

 For labelled data:

 $$
\mathcal L_{\text{sup}}
=
\operatorname{CE}
\left(
y_i,
f_\theta(x_i)
\right)
$$

 For unlabelled data:

 $$
\mathcal L_{\text{unsup}}
=
\operatorname{MSE}
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right)
$$

 Therefore:

 $$
\boxed{
\mathcal L
=
\operatorname{CE}
\left(
y_i,f_\theta(x_i)
\right)
+
\lambda
\operatorname{MSE}
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right)
}
$$

 and:

 $$
\boxed{
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta
}
$$

---

 # 19\. Why Mean Teacher Is Better Than Temporal Ensembling

 ### Temporal Ensembling

 Stores:

 $$
\tilde z_i
$$

 for every example.

 ### Mean Teacher

 Stores:

 $$
\theta'
$$

 one averaged model.

 Therefore:

 | Method | What is averaged? | Main issue |
| --- | --- | --- |
| Temporal Ensembling | Predictions for every example | Memory + slow updates |
| Mean Teacher | Model parameters | Much more scalable |

The teacher can generate a prediction for every example whenever needed.

 Thus:

 $$
\boxed{
\text{Mean Teacher scales much better}
}
$$

 and can work effectively at large dataset scales such as ImageNet.

---

 # 20\. Why Does the Teacher Stabilise Training?

 The teacher is not updated using ordinary backpropagation.

 Instead:

 $$
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta
$$

 Therefore, it changes more slowly than the student.

 The student may make a noisy update:

 $$
\theta_t
$$

 but the teacher only partially incorporates it.

 So:

 $$
\boxed{
\text{Teacher = slowly moving, stable target}
}
$$

 This prevents the target from changing as rapidly as the student.

---

 # 21\. Timeline of Semi-Supervised Learning Methods

 The lecture presents the methods as a progression.

 ## Stage 1 — Self-Training

 Use the model's own prediction as a pseudo-label:

 $$
\boxed{
\hat y_j=\arg\max f(x_j)
}
$$

 Loss:

 $$
\mathcal L
=
\operatorname{CE}(y_i,f(x_i))
+
\operatorname{CE}(\hat y_j,f(x_j))
$$

 ### Problem

 Confirmation bias.

---

 ## Stage 2 — Consistency Regularisation / UDA

 Instead of trusting a hard pseudo-label, require predictions to remain stable under noise/augmentation:

 $$
\boxed{
f(x_j)
\approx
f(x_j+\eta)
}
$$

 Loss:

 $$
\mathcal L
=
\operatorname{CE}(y_i,f(x_i))
+
\operatorname{CE}
\left(
f(x_j),
f(x_j+\eta)
\right)
$$

 ### Improvement

 Uses the assumption that small perturbations should not change the correct prediction.

---

 ## Stage 3 — Temporal Ensembling

 Instead of using one potentially noisy prediction as the target, use an EMA of previous predictions:

 $$
\boxed{
\mathcal L
=
\operatorname{CE}(y_i,f(x_i))
+
\operatorname{MSE}
\left(
f(x_j+\eta),
\tilde z_j
\right)
}
$$

 where:

 $$
\tilde z_j
=
\operatorname{EMA}
\left(
\text{previous predictions}
\right)
$$

 ### Improvement

 More stable targets.

 ### Problem

 Need to store predictions for every example.

---

 ## Stage 4 — Mean Teacher

 Replace the EMA of every example's prediction with an EMA of the model itself:

 $$
\boxed{
f_{\theta'}
=
\operatorname{EMA}(f_\theta)
}
$$

 and:

 $$
\boxed{
\mathcal L
=
\operatorname{CE}(y_i,f_\theta(x_i))
+
\operatorname{MSE}
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right)
}
$$

 ### Improvement

 Stable targets without storing per-example predictions.

---

 # 22\. The Key Progression to Memorise

 This is probably the single most useful sequence to know for an exam:

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

 Think of it as:

 ### Self-training

 > "Trust my prediction."

 ↓

 ### Consistency

 > "At least make predictions stable under perturbations."

 ↓

 ### Temporal Ensembling

 > "Don't trust one prediction; average predictions over time."

 ↓

 ### Mean Teacher

 > "Don't store an average prediction for every example; average the model itself."

---

 # 23\. Curriculum Learning vs Semi-Supervised Learning

 These concepts should not be confused.

 ### Curriculum Learning

 Uses information about:

 - sample difficulty,
- label quality,
- ordering.

 Goal:

 $$
\boxed{
\text{Improve training dynamics}
}
$$

 Example:

 $$
\text{easy examples}
\rightarrow
\text{hard examples}
$$

---

 ### Semi-Supervised Learning

 Uses:

 - labelled examples,
- unlabelled examples,
- similarity/structure between them.

 Goal:

 $$
\boxed{
\text{Learn useful representations from labelled + unlabelled data}
}
$$

---

 # 24\. Active Learning

 Active learning is related but different.

 Instead of simply using all available unlabelled data, the model **strategically selects informative examples** and asks for their labels.

 Pipeline:

 $$
\text{Unlabelled data}
\rightarrow
\text{select informative samples}
\rightarrow
\text{human labelling}
\rightarrow
\text{add to training data}
$$

 The key idea is:

 > Don't label everything. Label the examples that are expected to be most useful.

---

 ## Difference from Semi-Supervised Learning

 ### Semi-supervised learning

 Uses unlabelled data **without necessarily obtaining labels**.

 $$
\boxed{
\text{Use unlabelled examples directly}
}
$$

 ### Active learning

 Selects examples and obtains **new labels**.

 $$
\boxed{
\text{Choose which examples humans should label}
}
$$

---

 # 25\. Comparison Table

 | Method | Main idea | Target | Main weakness/problem |
| --- | --- | --- | --- |
| Self-training | Use own predictions as labels | Model's prediction | Confirmation bias |
| Co-training | Two models teach each other | Other model | Requires useful independent views |
| Consistency regularisation | Same input under perturbations should give same output | Another prediction | Targets can be unreliable |
| Π-model | Two noisy versions of same input | Other noisy prediction | Early predictions may be unstable |
| UDA | Strong augmentation should preserve prediction | Prediction on original input | Requires useful augmentations/confidence |
| Temporal Ensembling | EMA of previous predictions | Historical predictions | Memory + stale targets |
| Mean Teacher | EMA of model parameters | Teacher model | More complex than basic self-training |
| Active Learning | Select samples to label | Human-provided labels | Requires additional labelling |

---

 # 26\. Important Assumption Behind Consistency Learning

 Consistency regularisation relies on an important assumption:

 > **Perturbations/augmentations should preserve the semantic class.**

 If:

 $$
x'=\operatorname{Augment}(x)
$$

 then we assume:

 $$
y(x')=y(x)
$$

 Therefore:

 $$
f(x')\approx f(x)
$$

 ### Important exam point

 Not every augmentation is valid.

 If an augmentation changes the semantic meaning of an example, forcing consistency can be harmful.

---

 # 27\. Confidence Thresholding

 Confidence thresholding is another important concept.

 Suppose:

 $$
p=f(x)
$$

 and:

 $$
c=\max_k p_k
$$

 Only use the prediction if:

 $$
\boxed{
c>\tau
}
$$

 where $\tau$ is a threshold.

 ### Why?

 High-confidence predictions are more likely to be correct.

 Thus:

 $$
\text{confidence filtering}
\rightarrow
\text{fewer incorrect pseudo-targets}
\rightarrow
\text{less confirmation bias}
$$

---

 # 28\. Supervised vs Unsupervised Loss

 It is useful to be able to identify the two components immediately.

 ### Supervised

 Requires a ground-truth label:

 $$
\boxed{
\mathcal L_{\text{sup}}
=
\operatorname{CE}(y,f(x))
}
$$

 ### Unsupervised

 Does not require a ground-truth label.

 Examples:

 $$
\boxed{
\operatorname{CE}(\hat y,f(x))
}
$$

 or:

 $$
\boxed{
\operatorname{MSE}(f(x),f(x+\eta))
}
$$

 or:

 $$
\boxed{
\operatorname{MSE}(f_\theta(x+\eta),f_{\theta'}(x+\eta'))
}
$$

---

 # 29\. Cross-Entropy vs MSE in These Methods

 ### Cross-Entropy

 Often used when comparing a prediction to a target distribution/pseudo-label:

 $$
\operatorname{CE}(p,q)
$$

 ### MSE

 Often used for consistency between two prediction vectors:

 $$
\operatorname{MSE}(p,q)
=
\frac{1}{K}
\sum_{k=1}^{K}(p_k-q_k)^2
$$

 The important thing is not to memorise that one loss is universally required.

 Rather:

 > The unsupervised objective measures how different the model's prediction is from the desired stable target.

---

 # 30\. What the Teacher Actually Does

 A common exam trap is to think that the teacher is a second independently trained network.

 It is **not**.

 The teacher is an EMA of the student:

 $$
\boxed{
\theta'_t
=
\alpha\theta'_{t-1}
+
(1-\alpha)\theta_t
}
$$

 So the teacher is essentially an **ensemble of historical versions of the student**.

 This gives the teacher smoother predictions.

---

 # 31\. Teacher vs Student

 | Student | Teacher |
| --- | --- |
| Parameters $\theta$ | Parameters $\theta'$ |
| Directly trained with gradient descent | Updated using EMA |
| Changes quickly | Changes slowly |
| Learns from labels + consistency target | Provides consistency target |
| Noisy/current model | Smoothed/historical model |

The teacher therefore acts as a **stable role model** for the student.

---

 # 32\. Why EMA Creates Stability

 Suppose the student parameters jump:

 $$
\theta_{t-1}=1
$$

 and:

 $$
\theta_t=2
$$

 with:

 $$
\alpha=0.9
$$

 then:

 $$
\theta'_t
=
0.9\theta'_{t-1}+0.1(2)
$$

 The teacher moves only a little toward the student's new value.

 Thus:

 $$
\boxed{
\text{student changes quickly, teacher changes slowly}
}
$$

 This is the central stabilising mechanism.

---

 # 33\. Exam-Style Derivation: Mean Teacher

 Given:

 - labelled example $(x_i,y_i)$
- unlabelled example $x_j$
- student $f_\theta$
- teacher $f_{\theta'}$
- perturbations $\eta,\eta'$

 ### Supervised term

 $$
\mathcal L_{\text{sup}}
=
\operatorname{CE}
(y_i,f_\theta(x_i))
$$

 ### Consistency term

 $$
\mathcal L_{\text{cons}}
=
\operatorname{MSE}
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right)
$$

 ### Total

 $$
\boxed{
\mathcal L
=
\mathcal L_{\text{sup}}
+
\lambda\mathcal L_{\text{cons}}
}
$$

 ### Teacher update

 $$
\boxed{
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta
}
$$

 This set of equations is highly worth memorising.

---

 # 34\. Exam-Style Derivation: Temporal Ensembling

 For an unlabelled example $x_j$:

 Current prediction:

 $$
z_j^t=f_\theta(x_j+\eta)
$$

 Historical target:

 $$
\tilde z_j^t
=
\alpha\tilde z_j^{t-1}
+
(1-\alpha)z_j^t
$$

 Consistency loss:

 $$
\boxed{
\mathcal L_{\text{cons}}
=
\operatorname{MSE}
\left(
f_\theta(x_j+\eta),
\tilde z_j
\right)
}
$$

 The key distinction:

 $$
\boxed{
\text{Temporal Ensembling averages predictions}
}
$$

 whereas:

 $$
\boxed{
\text{Mean Teacher averages parameters}
}
$$

---

 # 35\. Common Exam Questions

 ## Q1. Why is naive self-training unstable?

 Because the model generates pseudo-labels using its own predictions. Incorrect predictions can therefore become training targets, producing confirmation bias and potentially mode collapse.

---

 ## Q2. What is consistency regularisation?

 It forces a model to produce similar predictions for different perturbed/augmented versions of the same input:

 $$
f(x)\approx f(\operatorname{Augment}(x))
$$

 This allows unlabelled data to contribute a training signal.

---

 ## Q3. What is the purpose of data augmentation in consistency training?

 The augmentation generates a different version of the same semantic example. The model is encouraged to learn representations that are invariant to changes that should not unlabelled data were treated as pseudo-labels. This suffers from confirmation bias because incorrect predictions can reinforce themselves. Consistency regularisation instead requires predictions to remain stable under perturbations or augmentations of the same input. Temporal ensembling further stabilises this process by using an exponential moving average of previous predictions as the target. However, storing a prediction for every example alter the label.

---

 ## Q4. Why use confidence thresholds in UDA?

 Because low-confidence predictions are unreliable targets. Only sufficiently confident predictions are used for the consistency objective, reducing confirmation bias.

---

 ## Q5. What is temporal ensembling?

 It maintains an EMA of previous predictions for each training example and uses that averaged prediction as a more stable target.

 $$
\tilde z_t
=
\alpha\tilde z_{t-1}
+
(1-\alpha)z_t
$$

---

 ## Q6. What is the main weakness of temporal ensembling?

 It requires storing an averaged prediction for every example and updates those predictions relatively infrequently, causing memory and lag problems.

---

 ## Q7. How does Mean Teacher solve this?

 Instead of maintaining an EMA prediction for every example, it maintains an EMA of the model parameters.

 $$
\theta'
=
\operatorname{EMA}(\theta)
$$

 The teacher can then generate stable targets whenever needed.

---

 ## Q8. Why is the teacher more stable than the student?

 Because its parameters are updated using an EMA, so individual noisy gradient updates have only a small effect.

---

 ## Q9. What is the difference between co-training and self-training?

 Self-training uses one model's predictions as its own pseudo-labels.

 Co-training uses multiple models, ideally with different views of the data, so that they can provide supervision for each other.

---

 ## Q10. What is active learning?

 Active learning selects the most informative unlabelled examples and obtains labels for them, rather than trying to exploit all unlabelled data without additional annotation.

---

 # 36\. Likely "Compare These" Question

 ### Self-Training vs Consistency Regularisation

 **Self-training:**

 $$
\hat y=\arg\max f(x)
$$

 Then train toward the hard pseudo-label.

 **Consistency:**

 $$
f(x)\approx f(x+\eta)
$$

 The target can be another prediction rather than a hard class.

 ### Main difference

 Self-training says:

 > "My prediction is the label."

 Consistency says:

 > "My prediction should remain stable when the input is perturbed."

---

 # 37\. Likely "Compare These" Question

 ### Temporal Ensembling vs Mean Teacher

 Both use EMA to stabilise the target.

 But:

 $$
\boxed{
\text{Temporal Ensembling: EMA of predictions}
}
$$

 while:

 $$
\boxed{
\text{Mean Teacher: EMA of model parameters}
}
$$

 Temporal ensembling:

 - prediction stored per example,
- memory scales with dataset size,
- targets can become stale.

 Mean Teacher:

 - one teacher model,
- better scalability,
- teacher generates fresh predictions.

---

 # 38\. Likely "Explain the Evolution" Question

 A strong answer would be:

 > Early semi-supervised methods used self-training, where a model's own predictions on unlabelled data were treated as pseudo-labels. This suffers from confirmation bias because incorrect predictions can reinforce themselves. Consistency regularisation instead requires predictions to remain stable under perturbations or augmentations of the same input. Temporal ensembling further stabilises this process by using an exponential moving average of previous predictions as the target. However, storing a prediction for every example scales poorly and updates can become stale. Mean Teacher addresses this by maintaining an exponential moving average of the model parameters, producing a stable teacher model that generates targets for the student.

 That paragraph essentially captures the whole methodological progression.

---

 # 39\. Core Equations to Memorise

 ### Supervised learning

 $$
\boxed{
\mathcal L_{\text{sup}}
=
\operatorname{CE}(y,f_\theta(x))
}
$$

 ### Self-training

 $$
\boxed{
\hat y=\arg\max f_\theta(x)
}
$$

 $$
\boxed{
\mathcal L_U
=
\operatorname{CE}(\hat y,f_\theta(x))
}
$$

 ### Consistency

 $$
\boxed{
\mathcal L_U
=
\operatorname{MSE}
\left(
f_\theta(x),
f_\theta(x+\eta)
\right)
}
$$

 ### Temporal Ensembling

 $$
\boxed{
\tilde z_t
=
\alpha\tilde z_{t-1}
+
(1-\alpha)z_t
}
$$

 ### Mean Teacher

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
\mathcal L_U
=
\operatorname{MSE}
\left(
f_\theta(x+\eta),
f_{\theta'}(x+\eta')
\right)
}
$$

 ### Generic semi-supervised objective

 $$
\boxed{
\mathcal L
=
\mathcal L_{\text{sup}}
+
\lambda\mathcal L_U
}
$$

---

 # 40\. One-Minute Revision Sheet

 If you have almost no time before the exam, remember this:

 ### Curriculum Learning

 $$
\boxed{\text{Control sample difficulty/order}}
$$

 Easy → hard.

---

 ### Semi-Supervised Learning

 $$
\boxed{\text{Use labelled + unlabelled data}}
$$

---

 ### Self-Training

 $$
\boxed{\text{Model labels its own data}}
$$

 Problem:

 $$
\boxed{\text{confirmation bias}}
$$

---

 ### Co-Training

 $$
\boxed{\text{Models teach each other}}
$$

 Different views → potentially independent errors.

---

 ### Consistency Regularisation

 $$
\boxed{
f(x)\approx f(\operatorname{Augment}(x))
}
$$

 Same example → same prediction.

---

 ### UDA

 Strong augmentation + consistency.

 Use confidence thresholding to avoid unreliable targets.

---

 ### Temporal Ensembling

 $$
\boxed{
\text{EMA of predictions}
}
$$

 More stable target.

 Problem:

 $$
\boxed{\text{memory + stale updates}}
$$

---

 ### Mean Teacher

 $$
\boxed{
\text{EMA of model parameters}
}
$$

 Student = trained normally.

 Teacher = EMA of student.

 Teacher provides stable targets.

---

 ### Active Learning

 $$
\boxed{
\text{Select informative samples to label}
}
$$

---

 # 41\. Conceptual Map

```
                 SEMI-SUPERVISED LEARNING
                          |
              +-----------+-----------+
              |                       |
          Labelled                 Unlabelled
          data                    data
              |                       |
              |              Need a training signal
              |                       |
              |         +-------------+-------------+
              |         |                           |
              |    Self-training              Consistency
              |         |                    regularisation
              |         |                           |
              |    pseudo-labels               perturbations
              |         |                           |
              |    confirmation              stable predictions
              |       bias                         |
              |                                     |
              |                         +-----------+-----------+
              |                         |                       |
              |                  Temporal Ensembling       Mean Teacher
              |                         |                       |
              |                  EMA predictions         EMA parameters
              |                         |                       |
              |                  per-example memory       teacher model
              |                         |                       |
              +-------------------------+-----------------------+
                                        |
                                  Stable learning
```

---

 # 42\. The Three Stability Mechanisms

 A particularly useful way to organise the lecture is:

 ### 1\. Multiple models

 **Co-training**

 $$
\text{Model A}\leftrightarrow\text{Model B}
$$

 Different models/views reduce correlated errors.

 ### 2\. Multiple perturbations

 **Consistency regularisation**

 $$
x\rightarrow x+\eta
$$

 Predictions should remain consistent.

 ### 3\. Multiple points in time

 **Temporal Ensembling / Mean Teacher**

 $$
t-2,t-1,t
$$

 Use historical information to produce a stable target.

 This gives a beautiful conceptual summary:

 $$
\boxed{
\text{Stability can come from diversity, perturbations, or time}
}
$$

---

 # 43\. High-Yield Definitions

 **Curriculum learning:** Training strategy that controls the order/difficulty of examples, typically moving from easier to harder examples.

 **Semi-supervised learning:** Learning from both labelled and unlabelled data.

 **Pseudo-label:** A label generated by a model rather than provided by a human.

 **Self-training:** Using a model's predictions on unlabelled data as pseudo-labels.

 **Co-training:** Using multiple models/views to provide supervision for each other.

 **Consistency regularisation:** Encouraging similar predictions for perturbed versions of the same input.

 **UDA:** Consistency training using data augmentation, particularly strong augmentation of unlabelled examples.

 **Temporal ensembling:** Using an EMA of historical predictions as a stable target.

 **EMA:** Exponential moving average; a weighted average that gives more weight to recent values while retaining historical information.

 **Mean Teacher:** A student-teacher semi-supervised method where the teacher is an EMA of the student's parameters.

 **Active learning:** Selecting informative unlabelled examples for human annotation.

 **Confirmation bias:** The tendency of self-training to reinforce the model's existing incorrect predictions.

 **Mode collapse:** A failure mode where the model becomes concentrated on a limited set of predictions/classes.

---

 # 44\. Final Exam Strategy

 If asked to **explain the lecture**, structure your answer around the problem and successive solutions:

 1. **Problem:** Labels are expensive, but unlabelled data are abundant.
2. **Self-training:** Generate pseudo-labels.
3. **Problem:** Confirmation bias.
4. **Co-training:** Use different models/views.
5. **Consistency:** Require invariance to perturbations.
6. **UDA:** Use strong augmentation and confidence filtering.
7. **Problem:** Current predictions can still be noisy.
8. **Temporal ensembling:** Average predictions over time.
9. **Problem:** Per-example storage and stale updates.
10. **Mean Teacher:** Average model parameters instead.
11. **Active learning:** Alternatively, select the most useful unlabelled samples for actual annotation.

 The central story is:

 $$
\boxed{
\text{Exploit unlabelled data}
\rightarrow
\text{stabilise the learning signal}
\rightarrow
\text{produce better representations}
}
$$

 And the most important progression is:

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

 **If you remember only one sentence:**

 > Semi-supervised learning uses unlabelled data by constructing an additional training signal, and the major challenge is making that signal stable enough that the model does not simply reinforce its own mistakes.

 This is the consolidated version of the four chunks, with the slide duplication/LaTeX extraction artefacts removed and the equations reconstructed into a consistent notation.
