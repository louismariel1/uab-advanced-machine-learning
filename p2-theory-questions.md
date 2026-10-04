# Curriculum Learning & Semi-Supervised Learning — Exam-Ready Q&A Notes

 > **How to use these notes:** Cover the answers and try to answer each question yourself. The questions are ordered to build the concepts from fundamentals → methods → equations → comparisons → exam-level reasoning.

---

 # Part I — Curriculum Learning

 ## 1\. What is curriculum learning?

 **Answer:**\
 Curriculum learning is a training strategy where examples are presented to a model in a deliberate order, typically from **easy to difficult**.

 The basic idea is:

 > Start with examples that are easy for the model to learn, then gradually introduce harder examples.

 Instead of training on the entire dataset with a random sampling strategy, curriculum learning changes the **training distribution over time**.

---

 ## 2\. What is the intuition behind curriculum learning?

 **Answer:**

 Humans often learn progressively:

 1. Learn simple concepts.
2. Build a foundation.
3. Introduce more difficult examples.
4. Eventually solve complex problems.

 Curriculum learning applies a similar idea to machine learning.

 If the model first learns simple patterns, these patterns can provide a useful foundation for learning harder examples.

---

 ## 3\. What is the central assumption behind curriculum learning?

 **Answer:**

 The central assumption is that **the order in which training examples are presented can affect optimisation and generalisation**.

 An appropriate curriculum may:

 - make optimisation easier,
- provide better initial representations,
- reduce the difficulty of early training,
- improve convergence,
- and sometimes improve final generalisation.

---

 ## 4\. What does "easy" mean in curriculum learning?

 **Answer:**

 "Easy" is task-dependent.

 An example might be considered easy if:

 - it has a clear label,
- it contains little noise,
- it has a simple structure,
- the model already predicts it correctly,
- or it has a low training loss.

 Difficulty can therefore be defined using either:

 - **human/domain knowledge**, or
- **information obtained from the model itself**.

---

 ## 5\. What is a hand-designed curriculum?

 **Answer:**

 A hand-designed curriculum uses prior knowledge to determine the order of examples.

 For example, suppose we are training an image classifier.

 We could begin with:

 - large, clear images,
- simple backgrounds,
- unambiguous examples,

 and later introduce:

 - occluded images,
- noisy images,
- unusual viewpoints,
- difficult classes.

 The curriculum is designed before or during training using domain knowledge.

---

 ## 6\. What is self-paced learning?

 **Answer:**

 Self-paced learning lets the **model determine which examples are currently easy enough to learn**.

 A common strategy is:

 1. Initially select easy examples.
2. Train the model.
3. Re-evaluate example difficulty.
4. Gradually include harder examples.

 Thus, unlike a fixed curriculum, the training process can adapt the curriculum based on the model's current state.

---

 ## 7\. How can training loss be used to estimate difficulty?

 **Answer:**

 A simple approach is to use the model's loss on an example.

 For example:

 $$
\text{difficulty}(x_i) \approx L(x_i)
$$

 A low-loss example is treated as easy, while a high-loss example is treated as difficult.

 However, high loss does not necessarily mean that an example is intrinsically difficult.

 It could also indicate:

 - label noise,
- an incorrect label,
- an unusual sample,
- or that the model has not yet learned the relevant feature.

---

 ## 8\. Why can curriculum learning help optimisation?

 **Answer:**

 Early in training, the model may struggle to learn from a highly heterogeneous dataset.

 Starting with easier examples can provide useful gradients and representations.

 The training process can therefore look like:

 $$
\text{simple patterns}
\rightarrow
\text{useful representations}
\rightarrow
\text{more difficult patterns}
$$

 This can make the optimisation trajectory easier to navigate.

---

 ## 9\. Is curriculum learning guaranteed to improve performance?

 **Answer:**

 No.

 Curriculum learning is a **training strategy**, not a guarantee of better performance.

 A poorly designed curriculum can:

 - introduce bias,
- remove useful examples for too long,
- reinforce incorrect assumptions about difficulty,
- or cause the model to overfit easy examples.

 Therefore, the curriculum itself must be evaluated.

---

 ## 10\. What is the difference between curriculum learning and random sampling?

 **Answer:**

 | Random sampling | Curriculum learning |
| --- | --- |
| Examples are sampled without deliberately controlling difficulty | Examples are deliberately ordered or weighted |
| Training distribution is relatively stable | Training distribution changes over time |
| No explicit notion of difficulty | Difficulty is used to control training |
| Simple baseline | Structured training strategy |

---

 # Part II — Semi-Supervised Learning

 ## 11\. What is semi-supervised learning?

 **Answer:**

 Semi-supervised learning uses both:

 - a relatively small amount of **labelled data**, and
- a larger amount of **unlabelled data**.

 We can write the datasets as:

 $$
D_L = \{(x_i,y_i)\}
$$

 for labelled data, and

 $$
D_U = \{x_j\}
$$

 for unlabelled data.

 The goal is to use the unlabelled data to improve the model beyond what could be achieved using only the labelled examples.

---

 ## 12\. Why is semi-supervised learning useful?

 **Answer:**

 Obtaining labels can be expensive.

 For example:

 - medical images may require expert annotation,
- speech data may require transcription,
- scientific data may require specialist knowledge.

 However, collecting raw unlabelled data can be relatively cheap.

 Semi-supervised learning attempts to exploit this large supply of unlabelled data.

---

 ## 13\. What is the key assumption behind semi-supervised learning?

 **Answer:**

 The key assumption is that **unlabelled examples contain useful information about the structure of the data distribution**.

 The model therefore tries to learn from both:

 $$
\text{label information}
+
\text{data structure}
$$

---

 ## 14\. What is the general objective in semi-supervised learning?

 **Answer:**

 The model typically has two components:

 1. A supervised loss using labelled examples.
2. An unsupervised loss using unlabelled examples.

 A generic objective is:

 $$
L
=
L_{\text{sup}}
+
\lambda L_{\text{unsup}}
$$

 where:

 - $L\_{\\text{sup}}$ is the supervised loss,
- $L\_{\\text{unsup}}$ is the unsupervised loss,
- $\\lambda$ controls the importance of the unsupervised objective.

---

 ## 15\. What is the supervised loss?

 **Answer:**

 For labelled data, a typical classification loss is cross-entropy:

 $$
L_{\text{sup}}
=
\mathrm{CE}(y_i,f_\theta(x_i))
$$

 where:

 - $x\_i$ is the input,
- $y\_i$ is its true label,
- $f\_\\theta$ is the model.

 This is standard supervised learning.

---

 # Part III — Self-Training

 ## 16\. What is self-training?

 **Answer:**

 Self-training uses the model's own predictions on unlabelled examples as **pseudo-labels**.

 The basic process is:

 1. Train on labelled data.
2. Predict labels for unlabelled examples.
3. Treat sufficiently confident predictions as pseudo-labels.
4. Train on those pseudo-labelled examples.
5. Repeat.

---

 ## 17\. What is a pseudo-label?

 **Answer:**

 A pseudo-label is a label generated by the model rather than supplied by a human.

 For an unlabelled example $x\_j$:

 $$
\hat{y}_j
=
\arg\max_y f_\theta(x_j)_y
$$

 The predicted class with the highest probability becomes the pseudo-label.

---

 ## 18\. What is the self-training objective?

 **Answer:**

 The loss can be written approximately as:

 $$
L
=
\mathrm{CE}(y_i,f_\theta(x_i))
+
\mathrm{CE}(\hat{y}_j,f_\theta(x_j))
$$

 The first term is the supervised loss.

 The second term is the pseudo-label or unsupervised loss.

---

 ## 19\. Why is naive self-training dangerous?

 **Answer:**

 Because the model can make mistakes and then **train on its own mistakes**.

 This creates a feedback loop:

 $$
\text{wrong prediction}
\rightarrow
\text{wrong pseudo-label}
\rightarrow
\text{training on wrong label}
\rightarrow
\text{stronger wrong prediction}
$$

 This is called **confirmation bias**.

---

 ## 20\. What is mode collapse in this context?

 **Answer:**

 Mode collapse refers to a situation where the model increasingly predicts a limited set of classes or patterns, rather than maintaining useful diversity.

 If pseudo-labels are biased, self-training can reinforce that bias.

 Therefore, semi-supervised learning needs mechanisms that make the training process more stable.

---

 # Part IV — Co-Training

 ## 21\. What is co-training?

 **Answer:**

 Co-training uses **two models** that teach each other.

 The important idea is that the models should make sufficiently **independent errors**.

 For example:

 $$
f_{\theta_1}(x)
\qquad
f_{\theta_2}(x)
$$

 Each model can generate pseudo-labels for the other.

---

 ## 22\. Why can co-training be more stable than naive self-training?

 **Answer:**

 In self-training, a model teaches itself.

 In co-training:

 $$
\text{Model 1}
\rightarrow
\text{Model 2}
$$

 and

 $$
\text{Model 2}
\rightarrow
\text{Model 1}
$$

 If the models use different views or representations of the data, their errors may be less correlated.

 This reduces the chance that one model simply reinforces its own mistakes.

---

 ## 23\. What is the key requirement for effective co-training?

 **Answer:**

 The models should ideally have **different but complementary views of the data** and make relatively independent errors.

 If both models make exactly the same mistakes, co-training provides little benefit.

---

 # Part V — Consistency Regularisation

 ## 24\. What is consistency regularisation?

 **Answer:**

 Consistency regularisation encourages a model to produce similar predictions for different perturbed versions of the same input.

 The basic principle is:

 > If two inputs represent the same underlying example, their predictions should be similar.

 For example:

 $$
x
\quad\text{and}\quad
x+\eta
$$

 should produce similar predictions.

---

 ## 25\. Why is consistency regularisation useful for unlabelled data?

 **Answer:**

 An unlabelled example has no ground-truth label.

 However, we can still require:

 $$
f_\theta(x)
\approx
f_\theta(x+\eta)
$$

 where $\\eta$ represents a perturbation or augmentation.

 Thus, we can obtain a training signal without knowing the true class.

---

 ## 26\. What is the generic consistency loss?

 **Answer:**

 A generic form is:

 $$
L_{\text{cons}}
=
D
\left(
f_\theta(x),
f_\theta(x+\eta)
\right)
$$

 where $D$ measures the difference between the two predictions.

 Possible choices include:

 - mean squared error,
- KL divergence,
- cross-entropy,
- other distribution-matching losses.

---

 ## 27\. What is the main idea of the Pi-model?

 **Answer:**

 The Pi-model is an early implementation of consistency regularisation.

 The model receives different perturbations of the same example and is encouraged to produce consistent predictions.

 Conceptually:

 $$
x+\eta_1
\rightarrow
f_\theta(x+\eta_1)
$$

 and

 $$
x+\eta_2
\rightarrow
f_\theta(x+\eta_2)
$$

 should produce similar outputs.

---

 ## 28\. What happens to labelled and unlabelled examples in consistency training?

 **Answer:**

 For labelled examples, we can use the normal supervised loss:

 $$
L_{\text{sup}}
=
\mathrm{CE}(y,f_\theta(x))
$$

 For unlabelled examples, we use a consistency loss:

 $$
L_{\text{unsup}}
=
D(f_\theta(x+\eta_1),f_\theta(x+\eta_2))
$$

 The overall objective becomes:

 $$
L
=
L_{\text{sup}}
+
\lambda L_{\text{unsup}}
$$

---

 # Part VI — UDA

 ## 29\. What is UDA?

 **Answer:**

 UDA stands for **Unsupervised Data Augmentation**.

 The idea is to compare predictions on:

 - an original example, and
- a heavily augmented version of that example.

 The model should be consistent under the augmentation.

---

 ## 30\. Why does UDA use strong augmentation?

 **Answer:**

 Strong augmentation creates a more challenging version of the same underlying example.

 The model is encouraged to learn features that are invariant to irrelevant changes.

 For example:

 $$
x
\rightarrow
\text{strong augmentation}
\rightarrow
x'
$$

 and we want:

 $$
f_\theta(x)
\approx
f_\theta(x')
$$

---

 ## 31\. Why should UDA avoid trusting uncertain predictions?

 **Answer:**

 Suppose the model predicts an unlabelled example with very low confidence.

 Using that prediction as a target could introduce noise.

 Therefore, UDA can apply the consistency loss only when the original prediction is sufficiently confident.

 This gives:

 $$
\text{high confidence}
\Rightarrow
\text{use consistency target}
$$

 while uncertain predictions may be ignored.

---

 # Part VII — Model Stability

 ## 32\. Why is stability important in semi-supervised learning?

 **Answer:**

 Semi-supervised methods often use predictions generated by models as training targets.

 If those predictions are unstable or incorrect, the model can reinforce its own errors.

 Therefore, successful semi-supervised learning usually needs a mechanism that reduces **confirmation bias**.

---

 ## 33\. What are the three major stability ideas discussed in the lecture?

 **Answer:**

 1. **Co-training**\
    Use different models that ideally make independent errors.
2. **Consistency regularisation**\
    Require predictions to be stable under perturbations.
3. **Temporal/model averaging**\
    Use averaged predictions or an averaged model to create more stable targets.

---

 # Part VIII — Temporal Ensembling

 ## 34\. What is temporal ensembling?

 **Answer:**

 Temporal ensembling creates a more stable target by averaging predictions for an example over previous training iterations.

 Instead of trusting only the current prediction:

 $$
z_i^{(t)}
$$

 we maintain an averaged prediction:

 $$
\tilde{z}_i^{(t)}
$$

---

 ## 35\. What update rule is used for temporal ensembling?

 **Answer:**

 The lecture gives an exponential moving average:

 $$
\tilde{z}_i^{(t)}
=
\alpha \tilde{z}_i^{(t-1)}
+
(1-\alpha)z_i^{(t)}
$$

 where $\\alpha$ controls how strongly previous predictions are retained.

 The paper used approximately:

 $$
\alpha \approx 0.6
$$

---

 ## 36\. Why is this called an exponential moving average?

 **Answer:**

 Repeatedly expanding the update gives:

 $$
\tilde{z}^{(t)}
=
\alpha^t\tilde{z}^{(0)}
+
(1-\alpha)
\sum_{k=1}^{t}
\alpha^{t-k}z^{(k)}
$$

 Therefore, older predictions receive exponentially decreasing weights.

 Recent predictions have greater influence than very old predictions.

---

 ## 37\. Why does temporal ensembling improve stability?

 **Answer:**

 Individual predictions can be noisy.

 Averaging predictions over time smooths this noise.

 Therefore:

 $$
\text{noisy predictions}
\rightarrow
\text{EMA}
\rightarrow
\text{more stable target}
$$

 The temporal lag also prevents the target from changing too rapidly.

 This creates a form of resistance to noisy updates.

---

 ## 38\. Where else is the exponential moving average idea used?

 **Answer:**

 A similar mechanism appears in optimisation algorithms such as **Adam**, where moving averages of gradient-related quantities are maintained.

 The common idea is:

 > Smooth noisy quantities over time instead of reacting completely to each individual update.

---

 # Part IX — Problems with Temporal Ensembling

 ## 39\. What are the main disadvantages of temporal ensembling?

 **Answer:**

 There are two major problems.

 ### 1\. Storage

 We need to store a moving-average prediction for every training example:

 $$
\tilde{z}_1,\tilde{z}_2,\ldots,\tilde{z}_N
$$

 This becomes expensive for large datasets.

 ### 2\. Slow updates

 The prediction for an example may only be updated when that example is encountered again.

 Therefore, the target can lag behind the current state of the model.

---

 # Part X — Mean Teacher

 ## 40\. What is the Mean Teacher algorithm?

 **Answer:**

 Mean Teacher replaces the per-example moving-average predictions of temporal ensembling with a **moving-average model**.

 Instead of storing:

 $$
\tilde{z}_i
$$

 for every example, we maintain a second model whose parameters are an exponential moving average of the student's parameters.

---

 ## 41\. What are the student and teacher models?

 **Answer:**

 The **student** is the model being directly trained.

 Its parameters are:

 $$
\theta
$$

 The **teacher** is the EMA version of the student.

 Its parameters are:

 $$
\theta'
$$

 The teacher generates the target that the student is trained to match.

---

 ## 42\. How are the teacher parameters updated?

 **Answer:**

 The teacher parameters are updated using an exponential moving average:

 $$
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta
$$

 Thus, the teacher changes more slowly than the student.

---

 ## 43\. Why is the teacher more stable than the student?

 **Answer:**

 The student changes after every gradient update.

 The teacher averages many previous versions of the student.

 Therefore:

 $$
\text{student}
=
\text{fast-changing model}
$$

 while:

 $$
\text{teacher}
=
\text{smoothed historical model}
$$

 This produces a more stable target.

---

 ## 44\. What is the Mean Teacher consistency loss?

 **Answer:**

 A simplified form is:

 $$
L_{\text{unsup}}
=
\mathrm{MSE}
\left(
f_\theta(x+\eta),
f_{\theta'}(x+\eta')
\right)
$$

 where:

 - $\\theta$ are the student's parameters,
- $\\theta'$ are the teacher's parameters,
- $\\eta$ and $\\eta'$ are perturbations.

 The student is trained to agree with the teacher.

---

 ## 45\. What is the full Mean Teacher objective?

 **Answer:**

 The objective can be written as:

 $$
L
=
\mathrm{CE}(y_i,f_\theta(x_i))
+
\lambda
\mathrm{MSE}
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right)
$$

 The first term is the supervised loss.

 The second term is the unsupervised consistency loss.

---

 ## 46\. Why does Mean Teacher scale better than temporal ensembling?

 **Answer:**

 Temporal ensembling stores an EMA prediction for every training example.

 Mean Teacher stores only:

 - the student model,
- the teacher model.

 Therefore, it avoids maintaining a separate prediction vector for every example.

 This makes it much more suitable for large datasets.

---

 ## 47\. Why does the Mean Teacher target update more frequently?

 **Answer:**

 The teacher parameters are updated after training steps.

 Therefore, the teacher can change after every batch rather than waiting for an individual example to be encountered again.

 This makes the target more responsive while still being smoother than the student.

---

 # Part XI — Comparing the Main Methods

 ## 48\. What is the evolution from self-training to Mean Teacher?

 **Answer:**

 The main progression is:

 $$
\text{Self-training}
\rightarrow
\text{Consistency regularisation}
\rightarrow
\text{Temporal ensembling}
\rightarrow
\text{Mean Teacher}
$$

 Each stage addresses weaknesses in the previous approach.

---

 ## 49\. How does self-training work?

 **Answer:**

 The model predicts a pseudo-label for an unlabelled example:

 $$
\hat{y}
=
\arg\max_y f_\theta(x)
$$

 The model then trains on that prediction.

 **Main problem:** confirmation bias.

---

 ## 50\. How does consistency regularisation improve self-training?

 **Answer:**

 Rather than requiring the model to generate a discrete pseudo-label, consistency regularisation requires predictions to remain stable under perturbations:

 $$
f_\theta(x)
\approx
f_\theta(x+\eta)
$$

 This avoids requiring a ground-truth label.

 **Main idea:** learn invariance to perturbations.

---

 ## 51\. How does temporal ensembling improve consistency regularisation?

 **Answer:**

 Instead of using a potentially unreliable current prediction as the target, temporal ensembling uses an EMA of previous predictions:

 $$
\tilde{z}^{(t)}
=
\alpha\tilde{z}^{(t-1)}
+
(1-\alpha)z^{(t)}
$$

 This produces a smoother target.

---

 ## 52\. How does Mean Teacher improve temporal ensembling?

 **Answer:**

 Temporal ensembling stores an averaged prediction for every example.

 Mean Teacher instead stores an averaged **model**:

 $$
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta
$$

 The teacher can then generate predictions for arbitrary examples.

 Thus, Mean Teacher:

 - avoids per-example prediction storage,
- updates continuously,
- scales better to large datasets.

---

 # Part XII — The Timeline You Should Know

 ## 53\. What is the timeline of the semi-supervised methods?

 **Answer:**

 ### Online self-training

 Use the model's own predictions:

 $$
\hat{y}_j
=
\arg\max_y f_\theta(x_j)
$$

 and train on the pseudo-label.

 ### Consistency regularisation

 Require predictions to agree under perturbations:

 $$
f_\theta(x_j)
\approx
f_\theta(x_j+\eta)
$$

 ### Temporal ensembling

 Compare the current prediction against an EMA of previous predictions:

 $$
L_{\text{unsup}}
=
\mathrm{MSE}
\left(
f_\theta(x_j+\eta),
\tilde{z}_j
\right)
$$

 ### Mean Teacher

 Compare the student prediction against the teacher prediction:

 $$
L_{\text{unsup}}
=
\mathrm{MSE}
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right)
$$

 with:

 $$
\theta'
=
\mathrm{EMA}(\theta)
$$

---

 # Part XIII — High-Value Comparison Questions

 ## 54\. What is the difference between pseudo-labeling and consistency regularisation?

 **Answer:**

 **Pseudo-labeling:**

 $$
x
\rightarrow
\hat{y}
\rightarrow
\text{train against } \hat{y}
$$

 The model creates a discrete target.

 **Consistency regularisation:**

 $$
x,\ x'
\rightarrow
f(x),f(x')
\rightarrow
\text{make predictions agree}
$$

 The target is another model prediction rather than necessarily a hard class label.

---

 ## 55\. What is the difference between temporal ensembling and Mean Teacher?

 **Answer:**

 | Temporal Ensembling | Mean Teacher |
| --- | --- |
| Averages predictions | Averages model parameters |
| Stores an EMA prediction for each example | Stores an EMA teacher model |
| Can require large memory for large datasets | More scalable |
| Example-specific targets | Teacher can predict any example |
| Updates when examples are revisited | Teacher updates after training steps |

---

 ## 56\. What is the difference between self-training and co-training?

 **Answer:**

 | Self-training | Co-training |
| --- | --- |
| One model teaches itself | Two models teach each other |
| Errors can reinforce themselves | Different models can reduce correlated errors |
| Vulnerable to confirmation bias | Can be more stable if views are sufficiently independent |

---

 ## 57\. What is the difference between curriculum learning and semi-supervised learning?

 **Answer:**

 **Curriculum learning** focuses on:

 > **Which examples should the model learn from, and in what order?**

 **Semi-supervised learning** focuses on:

 > **How can we exploit unlabelled examples in addition to labelled examples?**

 They address different problems, although they can be combined.

---

 # Part XIV — Active Learning

 ## 58\. What is active learning?

 **Answer:**

 Active learning strategically selects the **most informative unlabelled examples** and asks for their labels.

 The process is:

 $$
\text{unlabelled pool}
\rightarrow
\text{select informative samples}
\rightarrow
\text{human labelling}
\rightarrow
\text{train}
$$

---

 ## 59\. How is active learning different from semi-supervised learning?

 **Answer:**

 Semi-supervised learning tries to exploit unlabelled data **without necessarily obtaining labels**.

 Active learning chooses which examples should be **labelled next**.

 Therefore:

 - Semi-supervised learning: **use unlabelled data**
- Active learning: **choose what to label**

---

 ## 60\. How is active learning related to curriculum learning?

 **Answer:**

 Both involve strategically selecting training examples.

 Curriculum learning controls the **order/difficulty** of training examples.

 Active learning controls **which examples are worth labelling**.

---

 # Part XV — Exam-Level Reasoning

 ## 61\. Why is confirmation bias particularly dangerous in semi-supervised learning?

 **Answer:**

 Because the model's predictions become part of its own training signal.

 A small error can therefore become self-reinforcing:

 $$
\text{prediction error}
\rightarrow
\text{pseudo-label}
\rightarrow
\text{training signal}
\rightarrow
\text{stronger error}
$$

 This is why stability mechanisms are important.

---

 ## 62\. Why does consistency regularisation help reduce confirmation bias?

 **Answer:**

 It does not require the model to commit to an absolute class label for every unlabelled example.

 Instead, it asks the model to maintain a stable prediction under small or meaningful perturbations.

 This encourages:

 $$
\text{prediction stability}
$$

 rather than simply:

 $$
\text{prediction confidence}
$$

---

 ## 63\. Why might confidence thresholding help pseudo-labeling?

 **Answer:**

 Low-confidence predictions are more likely to be incorrect.

 Therefore, only using predictions satisfying:

 $$
\max_y f_\theta(x) > \tau
$$

 for some threshold $\\tau$ can reduce noisy pseudo-labels.

 The trade-off is that a high threshold may discard too many useful examples.

---

 ## 64\. Why is an EMA useful for machine learning models?

 **Answer:**

 Gradient updates are noisy.

 An EMA smooths parameter changes:

 $$
\theta'_t
=
\alpha\theta'_{t-1}
+
(1-\alpha)\theta_t
$$

 Therefore, the averaged model is less sensitive to individual noisy updates.

---

 ## 65\. What happens when $\\alpha$ is close to 1?

 **Answer:**

 The EMA places more weight on the previous value.

 For example:

 $$
\alpha \rightarrow 1
$$

 means:

 - very slow adaptation,
- strong smoothing,
- more historical information,
- but potentially greater lag.

---

 ## 66\. What happens when $\\alpha$ is small?

 **Answer:**

 The EMA follows the current model more closely.

 Thus:

 $$
\alpha \rightarrow 0
$$

 means less smoothing and faster adaptation.

 There is therefore a trade-off:

 $$
\text{large }\alpha
\Rightarrow
\text{stable but slow}
$$

 $$
\text{small }\alpha
\Rightarrow
\text{responsive but noisy}
$$

---

 # Part XVI — Formula Recognition Questions

 ## 67\. If you see this equation, what method is it describing?

 $$
\hat{y}_j
=
\arg\max_y f_\theta(x_j)
$$

 **Answer:**

 **Self-training / pseudo-labeling.**

 The model chooses its most likely class as a pseudo-label.

---

 ## 68\. If you see this equation, what method is it describing?

 $$
\tilde{z}^{(t)}
=
\alpha\tilde{z}^{(t-1)}
+
(1-\alpha)z^{(t)}
$$

 **Answer:**

 **Temporal ensembling / exponential moving average.**

 The prediction is averaged over time.

---

 ## 69\. If you see this equation, what method is it describing?

 $$
\theta'
=
\alpha\theta'
+
(1-\alpha)\theta
$$

 **Answer:**

 **Mean Teacher.**

 The teacher parameters are an EMA of the student parameters.

---

 ## 70\. If you see this equation, what method is it describing?

 $$
L_{\text{unsup}}
=
D(f_\theta(x),f_\theta(x+\eta))
$$

 **Answer:**

 **Consistency regularisation.**

 The model is encouraged to make similar predictions for perturbed versions of the same input.

---

 ## 71\. If you see this equation, what method is it describing?

 $$
L
=
\mathrm{CE}(y,f_\theta(x))
+
\lambda
\mathrm{MSE}
\left(
f_\theta(x+\eta),
f_{\theta'}(x+\eta')
\right)
$$

 **Answer:**

 **Mean Teacher.**

 The first term is supervised classification loss.

 The second term is student-teacher consistency loss.

---

 # Part XVII — Common Exam Traps

 ## 72\. Is consistency regularisation the same as pseudo-labeling?

 **Answer:**

 No.

 Pseudo-labeling explicitly generates a class target:

 $$
\hat{y}
$$

 Consistency regularisation instead requires two predictions to agree:

 $$
f(x)
\approx
f(x')
$$

 They are related but conceptually different.

---

 ## 73\. Is the Mean Teacher teacher model trained by gradient descent?

 **Answer:**

 Not directly.

 The student is trained using gradient descent.

 The teacher is updated using an EMA of the student's parameters:

 $$
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta
$$

 This distinction is important.

---

 ## 74\. Is the teacher always better than the student?

 **Answer:**

 Not necessarily in an absolute sense.

 The teacher is intended to provide a **more stable target** because it averages the student's historical parameters.

 Its main advantage is stability rather than being independently trained to minimise a separate supervised objective.

---

 ## 75\. Why not simply use the current student prediction as the target?

 **Answer:**

 Because then the target can change rapidly and may contain significant noise.

 Using the student to generate its own target provides little stabilisation.

 The teacher solves this by creating a slower-moving target.

---

 # Part XVIII — One-Minute Revision

 ## 76\. What are the most important ideas to remember?

 **Answer:**

 ### Curriculum learning

 $$
\boxed{\text{Easy examples} \rightarrow \text{Hard examples}}
$$

 Controls the **training order**.

 ### Semi-supervised learning

 $$
\boxed{\text{Labelled data}+\text{Unlabelled data}}
$$

 Uses unlabelled examples to improve learning.

 ### Self-training

 $$
\boxed{\text{Model predicts pseudo-labels}}
$$

 Main danger: **confirmation bias**.

 ### Co-training

 $$
\boxed{\text{Two models teach each other}}
$$

 Requires sufficiently different/error-independent views.

 ### Consistency regularisation

 $$
\boxed{f(x)\approx f(x+\eta)}
$$

 The model should be robust to perturbations.

 ### Temporal ensembling

 $$
\boxed{\text{EMA of predictions}}
$$

 Smooths noisy predictions over time.

 ### Mean Teacher

 $$
\boxed{\text{EMA of model parameters}}
$$

 Teacher provides a stable target for the student.

 ### Active learning

 $$
\boxed{\text{Select informative samples to label}}
$$

 Optimises **which data should receive labels**.

---

 # Part XIX — The Core Story of the Lecture

 ## 77\. What is the overall conceptual story?

 **Answer:**

 The lecture can be remembered as a progression toward **stable learning from limited labels**.

 ### Step 1 — Self-training

 Use the model's own predictions.

 **Problem:** the model can reinforce its mistakes.

 ### Step 2 — Consistency regularisation

 Instead of simply trusting pseudo-labels, require predictions to be stable under perturbations.

 **Problem:** both predictions can still be unreliable early in training.

 ### Step 3 — Temporal ensembling

 Average predictions over time.

 **Benefit:** smoother and more reliable targets.

 **Problem:** storing predictions for every example does not scale well.

 ### Step 4 — Mean Teacher

 Average the model itself rather than every individual prediction.

 **Benefit:** a stable teacher can generate targets for any example and scales much better.

 The conceptual progression is therefore:

 $$
\boxed{
\text{Self-training}
\rightarrow
\text{Consistency}
\rightarrow
\text{Temporal averaging}
\rightarrow
\text{Mean Teacher}
}
$$

---

 # Part XX — Final Exam Checklist

 Before the exam, make sure you can answer all of these without looking at the notes:

 - What is curriculum learning?
- Why might easy-to-hard training help?
- What is self-paced learning?
- What is semi-supervised learning?
- What are labelled and unlabelled datasets?
- What is pseudo-labeling?
- What is confirmation bias?
- What is self-training?
- Why can self-training collapse?
- What is co-training?
- Why should co-training models make independent errors?
- What is consistency regularisation?
- Why does consistency regularisation work without labels?
- What is the Pi-model?
- What is UDA?
- Why use strong augmentation?
- Why use confidence thresholds?
- What is temporal ensembling?
- What is an exponential moving average?
- What does $\\alpha$ control?
- Why does EMA improve stability?
- What are the scalability problems of temporal ensembling?
- What is Mean Teacher?
- What is the difference between student and teacher?
- How is the teacher updated?
- Why is the teacher more stable?
- Why does Mean Teacher scale better?
- What is active learning?
- How does active learning differ from semi-supervised learning?
- How does curriculum learning differ from semi-supervised learning?
- Can you identify self-training, consistency regularisation, temporal ensembling, and Mean Teacher from their equations?
- Can you explain the progression from self-training to Mean Teacher?

---

 # Ultra-Short Memory Map

 $$
\boxed{
\begin{array}{c}
\text{Curriculum Learning}\\
\downarrow\\
\text{Control example difficulty/order}
\end{array}
}
$$

 $$
\boxed{
\begin{array}{c}
\text{Self-training}\\
\downarrow\\
\text{Use model's predictions as labels}\\
\downarrow\\
\text{Confirmation bias}
\end{array}
}
$$

 $$
\boxed{
\begin{array}{c}
\text{Consistency}\\
\downarrow\\
f(x)\approx f(x+\eta)
\end{array}
}
$$

 $$
\boxed{
\begin{array}{c}
\text{Temporal Ensembling}\\
\downarrow\\
\text{EMA of predictions}
\end{array}
}
$$

 $$
\boxed{
\begin{array}{c}
\text{Mean Teacher}\\
\downarrow\\
\text{EMA of model parameters}\\
\downarrow\\
\text{Stable teacher targets}
\end{array}
}
$$

 **The single most important conceptual chain:**

 $$
\boxed{
\text{Limited labels}
\rightarrow
\text{exploit unlabelled data}
\rightarrow
\text{stability is essential}
\rightarrow
\text{EMA provides stability}
\rightarrow
\text{Mean Teacher}
}
$$

 This version deliberately avoids `\operatorname` and uses GitHub-friendly constructs such as `\mathrm{CE}`, `\mathrm{MSE}`, `\arg\max`, subscripts, superscripts, and standard mathematical operators.
