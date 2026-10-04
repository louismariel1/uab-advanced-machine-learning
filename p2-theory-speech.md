# Curriculum Learning & Semi-Supervised Learning — Complete Teaching Script

 Below is a **spoken-word lecture script** designed for an audio-only class. It assumes students cannot see your slides, so every important diagram, equation, and conceptual transition is described aloud.

---

 # 1\. Opening: What Are We Trying to Solve?

 Hello everyone.

 Today we are going to study **Curriculum Learning and Semi-Supervised Learning**.

 These two topics may initially look quite different.

 Curriculum learning asks:

 > **In what order should we show training examples to a machine-learning model?**

 Semi-supervised learning asks:

 > **How can we learn when only some of our training examples have labels?**

 But there is a common idea underneath both.

 In ordinary supervised learning, we tend to assume that our training data are already available in a convenient form: the examples are labelled, and we simply train on them.

 In practice, however, neither assumption is necessarily true.

 We might have millions of examples but labels for only a small fraction.

 Or perhaps all our examples are labelled, but some are much harder, noisier, or less informative than others.

 So today's lecture is really about **making better use of the data we have**.

 We will begin with curriculum learning.

 Then we will move to semi-supervised learning.

 Within semi-supervised learning, we will progressively develop a sequence of increasingly sophisticated ideas:

 - self-training,
- pseudo-labels,
- co-training,
- consistency regularisation,
- temporal ensembling,
- and finally Mean Teacher.

 At each stage, I want you to keep one question in mind:

 > **What problem does this method solve, and what new problem does it introduce?**

 That question is particularly useful for understanding how these methods evolved historically.

---

 # 2\. Curriculum Learning

 Let's start with curriculum learning.

 The central idea is extremely simple.

 Imagine that you are teaching a student mathematics.

 Would you normally begin with the hardest problems in the textbook?

 Probably not.

 You might start with relatively easy examples.

 Once the student understands those, you introduce progressively more difficult problems.

 Eventually, the student can tackle problems that would have been overwhelming at the beginning.

 Curriculum learning applies the same basic idea to machine learning.

 Instead of presenting training examples in an arbitrary order, we deliberately organise them according to their **difficulty**.

 We begin with easier examples and gradually introduce spend a lot of its early training capacity trying to deal with examples that it is simply not ready to understand learn a representation that is very good for the easy subset but poorly suited to the overall task harder ones.

 So the basic principle is:

 > **Easy examples first, difficult examples later.**

 This is called a **curriculum**.

---

 # 3\. Why Might Curriculum Learning Help?

 Let's think about what happens during ordinary neural-network training.

 Suppose our training set contains three kinds of examples.

 First, very easy examples.

 Second, moderately difficult examples.

 Third, extremely difficult or ambiguous examples.

 At the beginning of training, the model knows very little.

 If we immediately expose it to the difficult examples, their gradients may be noisy or confusing.

 The model may spend a lot of its early training capacity trying to deal with examples that it is simply not ready to understand.

 Curriculum learning proposes a different strategy.

 First teach the model the relatively simple structure in the data.

 Then, once the model has learned useful representations, introduce increasingly difficult examples.

 The intuition is similar to human education:

 > **Build competence before demanding sophistication.**

---

 # 4\. What Does "Difficulty" Mean?

 Now we have an important question.

 What exactly makes an example difficult?

 There is no universal answer.

 Difficulty can be defined in several ways.

 For example, an image may be difficult because:

 - the object is partially hidden,
- the image is noisy,
- the object is very small,
- the lighting is unusual,
- or the image contains several visually similar classes.

 In natural-language processing, a sentence might be difficult because:

 - it contains unusual vocabulary,
- its syntax is complicated,
- or its meaning is ambiguous.

 Difficulty can therefore be based on domain knowledge.

 But we can also define difficulty using the model itself.

 For example, perhaps we measure the model's confidence.

 If the model consistently predicts an example correctly with high confidence, we might consider that example relatively easy.

 If the model repeatedly makes mistakes on it, we might consider it difficult.

 So curriculum learning does not necessarily require a perfect, human-defined difficulty score.

 The curriculum can potentially be **model-driven**.

---

 # 5\. Curriculum Learning Versus Random Sampling

 Let's compare curriculum learning with standard training.

 In ordinary stochastic gradient descent, we generally sample mini-batches from the training data.

 The order is often effectively random.

 Every mini-batch may contain a mixture of easy and difficult examples.

 Curriculum learning introduces structure into this ordering.

 A simple curriculum might look like this:

 At the beginning:

 > 80 percent easy examples, 20 percent difficult examples.

 Later:

 > 50 percent easy, 50 percent difficult.

 Eventually:

 > mostly difficult examples.

 The exact schedule depends on the problem.

 The key point is that **training difficulty changes over time**.

---

 # 6\. Does Curriculum Learning Always Help?

 No.

 This is an important exam point.

 Curriculum learning is not a universal guarantee of better performance.

 A poorly designed curriculum can actually make learning worse.

 For example, imagine that the "easy" examples are systematically different from the examples the model will encounter at test time.

 The model could learn a representation that is very good for the easy subset but poorly suited to the overall task.

 So curriculum design involves a trade-off.

 We want examples to become more difficult over time, but we also want the early examples to teach information that remains useful later.

 There is therefore a broader question:

 > **How should we choose the curriculum?**

 Possible approaches include:

 - manually defining difficulty,
- using metadata,
- using model confidence,
- using loss,
- or learning the curriculum itself.

---

 # 7\. A Useful Mental Model for Curriculum Learning

 Think of curriculum learning as controlling the **training distribution over time**.

 At time zero, we train on distribution $P_0(x)$.

 Later, we train on $P_1(x)$.

 Eventually, we approach the full training distribution $P(x)$.

 So instead of saying:

 > "Train on the entire dataset immediately."

 we say:

 > "Start with a manageable subset and gradually expose the model to the full complexity of the problem."

 This is the key conceptual picture.

---

 # 8\. Transition to Semi-Supervised Learning

 Now let's move to our second major topic:

 **semi-supervised learning**.

 Here the problem is different.

 Suppose we have a small labelled dataset:

 $$
D_L = \{(x_i,y_i)\}
$$

 where $x_i$ is an input and $y_i$ is its label.

 But we also have a much larger unlabelled dataset:

 $$
D_U = \{x_j\}.
$$

 The challenge is:

 > How can we use the unlabelled examples to improve our model?

 This is extremely important in real-world machine learning because obtaining labels is often expensive.

 Imagine medical images.

 Collecting another million images might be relatively easy.

 Having a medical expert label all one million images is much harder.

 So we may have:

 > a small amount of labelled data,

 and

 > a huge amount of unlabelled data.

 Semi-supervised learning tries to exploit both.

---

 # 9\. The Basic Semi-Supervised Objective

 In ordinary supervised learning, our loss might simply be:

 $$
L_{\text{sup}}
=
CE(y_i,f(x_i)),
$$

 where $CE$ represents cross-entropy and $f(x_i)$ is the model's prediction.

 Semi-supervised learning introduces an additional loss:

 $$
L
=
L_{\text{sup}}
+
\lambda L_{\text{unsup}}.
$$

 Here:

 - $L_{\text{sup}}$ uses genuinely labelled data,
- $L_{\text{unsup}}$ uses unlabelled data,
- and $\lambda$ controls how strongly the unsupervised objective influences training.

 This equation is one of the most important equations in the entire lecture.

 Nearly everything we discuss from this point onward can be understood as different ways of constructing

 $$
L_{\text{unsup}}.
$$

---

 # 10\. The First Idea: Self-Training

 The simplest approach is called **self-training**.

 Suppose our model sees an unlabelled example $x_j$.

 It produces a prediction.

 For example, suppose the model is classifying photographs as either:

 - cat,
- dog,
- or horse.

 The model sees an unlabelled image and predicts:

 > Cat: 0.92\
>  Dog: 0.05\
>  Horse: 0.03

 We can take the model's most likely prediction:

 $$
\hat y_j
=
\arg\max_y f(x_j).
$$

 We call this predicted label a **pseudo-label**.

 We then pretend that the pseudo-label is a real label and train on it.

 The resulting objective is approximately:

 $$
L
=
CE(y_i,f(x_i))
+
CE(\hat y_j,f(x_j)).
$$

 The first term uses real labels.

 The second uses labels generated by the model itself.

---

 # 11\. Why Is Self-Training Attractive?

 The attraction is obvious.

 We have thousands or millions of unlabelled examples.

 Instead of ignoring them, we allow the model to label them automatically.

 This effectively converts unlabelled data into additional training data.

 But there is a major problem.

 Where did the pseudo-label come from?

 From the model itself.

 So what happens if the model is wrong?

 Suppose the image is actually a dog.

 But the model predicts:

 > Cat.

 We now treat "cat" as the truth.

 The model trains on its own mistake.

 This can reinforce the error.

 And that leads to a fundamental problem:

 > **confirmation bias.**

---

 # 12\. Confirmation Bias in Self-Training

 Let's make this concrete.

 Suppose the model initially has a slight tendency to confuse cats and foxes.

 It incorrectly labels some foxes as cats.

 Those examples are then added to the training signal with the incorrect label "cat".

 The model becomes even more confident that those visual features correspond to cats.

 It then makes more similar mistakes.

 Those mistakes generate more pseudo-labels.

 And those pseudo-labels reinforce the same mistake.

 So we can get a feedback loop:

 > model makes an error\
>  → error becomes a pseudo-label\
>  → model trains on the error\
>  → model becomes more confident in the error\
>  → more errors are generated.

 This is why naive self-training can be **fragile**.

---

 # 13\. Confidence Thresholding

 One obvious solution is:

 > Don't trust every prediction.

 Instead, only use pseudo-labels when the model is sufficiently confident.

 For example, suppose we choose a threshold of 0.95.

 If the model predicts:

 > Cat: 0.97

 we use the pseudo-label.

 But if it predicts:

 > Cat: 0.61\
>  Dog: 0.39

 we ignore the example.

 The idea is that high-confidence predictions are more likely to be correct.

 So the unsupervised loss becomes something like:

 $$
L_{\text{unsup}}
=
\mathbf{1}[p_{\max}>\tau]
CE(\hat y_j,f(x_j)),
$$

 where:

 - $p_{\max}$ is the model's maximum predicted probability,
- $\tau$ is the confidence threshold,
- and the indicator determines whether the example contributes to the loss.

 This is a simple but important stabilisation mechanism.

---

 # 14\. Online Self-Training

 There is an important distinction.

 In **offline self-training**, we might:

 1. train a model,
2. generate pseudo-labels for the unlabelled dataset,
3. add those pseudo-labelled examples,
4. retrain.

 In **online self-training**, pseudo-labels can be generated during training.

 The model produces a prediction for an unlabelled example and immediately uses that prediction as a training target.

 This can make the process more dynamic.

 However, it also makes stability more important because the model is constantly generating the targets that influence its own future updates.

---

 # 15\. The Need for Stability

 At this point we can identify the central challenge of semi-supervised learning.

 We want to exploit unlabelled data.

 But our model's predictions are not necessarily reliable.

 So we need some mechanism that prevents the model from simply reinforcing its own mistakes.

 The lecture calls this **model stability**.

 There are several ways to achieve it.

 One is **co-training**.

 Another is **consistency regularisation**.

 Let's look at co-training first.

---

 # 16\. Co-Training

 The key idea behind co-training is:

 > Use two different models, ideally based on different views of the data.

 Imagine that every example can be described using two independent feature sets.

 For a webpage, for example, we might have:

 - the text on the page,
- and the links pointing to or from the page.

 We train two models:

 $$
f_1(x^{(1)})
$$

 and

 $$
f_2(x^{(2)}).
$$

 Each model sees a different view.

 Model 1 generates confident pseudo-labels for some unlabelled examples.

 Those only the model's current prediction as the target. Build the target from predictions made at previous points in predictions as pseudo-labels. If those predictions are incorrect, the errors become part of the training signal. This can create confirmation sufficiently independent errors, one model's confident predictions can provide useful pseudo-labels for the other. This reduces the dependence on a single model pseudo-labels can be used to train Model 2.

 Model 2 does the same for Model 1.

 So instead of:

 > one model teaching itself,

 we have:

 > two models teaching each other.

---

 # 17\. Why Does Co-Training Help?

 The critical assumption is that the two models make **different errors**.

 If both models always make exactly the same mistakes, then co-training provides little benefit.

 But if their errors are relatively independent, one model can provide useful information to the other.

 For example:

 Model A might be excellent at identifying objects based on visual shape.

 Model B might be better at identifying objects based on texture.

 If Model A is highly confident about an example, it can provide a pseudo-label to Model B.

 The hope is that the second model can learn from information that was not already available to it.

 This is the key intuition behind co-training.

---

 # 18\. Limitations of Co-Training

 Co-training depends on a strong assumption:

 > the data must have useful, sufficiently distinct views.

 That is not always available.

 For many modern datasets, there may not be two natural feature sets that satisfy this assumption.

 So we need another way to create diversity.

 This leads us to **consistency regularisation**.

---

 # 19\. Consistency Regularisation

 Consistency regularisation takes a very elegant approach.

 Instead of asking:

 > "What is the correct label for this unlabelled example?"

 we ask:

 > **"Should the model make the same prediction if we slightly perturb the example?"**

 Usually the answer is yes.

 Suppose we have an image $x$.

 We add some noise or apply an augmentation to obtain:

 $$
x' = x+\eta.
$$

 If the perturbation does not fundamentally change the meaning of the input, then the model should produce approximately the same prediction:

 $$
f(x)
\approx
f(x+\eta).
$$

 This gives us an unsupervised learning signal without needing the true label.

---

 # 20\. Why Does This Make Sense?

 Consider an image of a dog.

 Suppose we:

 - slightly crop it,
- change the brightness,
- add a small amount of noise,
- flip it,
- or make another transformation that preserves its semantic identity.

 We still expect the answer to be:

 > dog.

 So a good classifier should be **invariant to reasonable perturbations**.

 Consistency regularisation turns this intuition into a loss function.

 For example:

 $$
L_{\text{unsup}}
=
CE(f(x),f(x+\eta)).
$$

 Depending on the method, we might use cross-entropy, mean squared error, KL divergence, or another distance between predictions.

 The central principle remains the same:

 > **Different reasonable versions of the same input should produce consistent predictions.**

---

 # 21\. The Π-Model

 An early implementation of this idea is the **Π-model**, associated with Laine and Aila.

 The basic mechanism is straightforward.

 Take the same input.

 Create two stochastic versions.

 For example:

 $$
x+\eta_1
$$

 and

 $$
x+\eta_2.
$$

 Pass both through the model.

 Then encourage their predictions to agree.

 So we can think of the loss as:

 $$
L_{\text{unsup}}
=
D\left(
f(x+\eta_1),
f(x+\eta_2)
\right),
$$

 where $D$ measures disagreement.

 For labelled examples, we still use the normal supervised loss.

 For unlabelled examples, we use the consistency loss.

 Thus the overall training objective combines both.

---

 # 22\. Why Is Consistency Better Than Naive Self-Training?

 Notice something subtle.

 Self-training says:

 > "The model's prediction is the target."

 Consistency regularisation says:

 > "The model should behave similarly under perturbations."

 That is a weaker and often safer assumption.

 We don't need to claim:

 > "The model knows the correct class."

 We only require:

 > "Small changes that preserve the meaning of the input should not cause the prediction to change dramatically."

 This gives us useful information without requiring a fully trustworthy pseudo-label.

---

 # 23\. Unsupervised Data Augmentation — UDA

 A stronger version is **Unsupervised Data Augmentation**, or UDA.

 The basic idea is to take:

 - the original example,
- and a strongly augmented version.

 For a labelled example, we can simply use the true label.

 For an unlabelled example, we encourage the prediction on the original example to agree with the prediction on the augmented example.

 Conceptually:

 $$
x
\rightarrow
f(x)
$$

 and

 $$
x+\eta
\rightarrow
f(x+\eta).
$$

 Then we minimise their disagreement.

 The important point is that UDA uses **strong augmentations**.

 The model therefore has to learn predictions that remain stable even under substantial but semantically valid transformations.

---

 # 24\. Why Use Confidence in UDA?

 There is still a problem.

 Suppose the model is completely uncertain about an unlabelled example.

 Maybe its prediction is:

 > Cat: 0.34\
>  Dog: 0.33\
>  Fox: 0.33.

 Should we force the augmented version to reproduce this uncertain prediction?

 Possibly not.

 The target itself may be unreliable.

 UDA therefore uses a confidence-based strategy.

 For an unlabelled example, we use the consistency loss when the prediction on the original example is sufficiently confident.

 In other words:

 > **Only trust the model as a teacher when it is sufficiently confident.**

 This is another form of stabilisation.

---

 # 25\. A General View of Consistency Regularisation

 Let's step back.

 We have now seen several methods:

 - self-training,
- co-training,
- the Π-model,
- UDA.

 They look different, but they share a common structure.

 We always have labelled examples contributing a supervised loss:

 $$
L_{\text{sup}}.
$$

 And we have unlabelled examples contributing some form of constraint:

 $$
L_{\text{unsup}}.
$$

 So generally:

 $$
L
=
L_{\text{sup}}
+
\lambda L_{\text{unsup}}.
$$

 The question becomes:

 > What should $L_{\text{unsup}}$ be?

 For self-training:

 $$
L_{\text{unsup}}
=
CE(\hat y,f(x)).
$$

 For consistency regularisation:

 $$
L_{\text{unsup}}
=
D(f(x),f(x+\eta)).
$$

 The fundamental objective is therefore not changing.

 We are changing the way in which the unlabelled data provide information.

---

 # 26\. Why Consistency Regularisation Works

 There is a deeper intuition here.

 Suppose the input space contains many points representing essentially the same underlying object.

 For example, these might be:

 - the same image with slightly different lighting,
- the same sentence with harmless perturbations,
- or different views of the same object.

 We want the classifier's decision function to vary smoothly within these regions.

 In other words:

 > Nearby or semantically equivalent inputs should have similar predictions.

 This encourages the model to learn a smoother decision boundary.

 That is one reason consistency-based methods can exploit unlabelled data effectively.

---

 # 27\. The Remaining Problem: Which Prediction Should We Trust?

 However, consistency regularisation has a weakness.

 Imagine that the model sees an unlabelled example early in training.

 The model predicts incorrectly.

 We then compare one noisy prediction with another noisy prediction.

 Both may be unreliable.

 The consistency loss can still encourage the model to agree with itself.

 So we have a problem:

 > **The target and the prediction may both be bad.**

 This motivates another stabilisation mechanism.

 We want a target that is more stable than the current prediction.

 This leads to **temporal ensembling**.

---

 # 28\. Temporal Ensembling

 The key idea is:

 > Don't use only the model's current prediction as the target. Build the target from predictions made at previous points in training.

 Suppose example $i$ has current prediction:

 $$
z_i^t.
$$

 Instead of using only this prediction, we maintain an averaged prediction:

 $$
\tilde z_i^t.
$$

 The update is:

 $$
\tilde z_i^t
=
\alpha \tilde z_i^{t-1}
+
(1-\alpha)z_i^t.
$$

 Here $\alpha$ controls how much we trust the previous history.

 If $\alpha$ is large, the target changes slowly.

 If $\alpha$ is smaller, it responds more quickly to new predictions.

---

 # 29\. Understanding the EMA

 This equation is an **exponential moving average**, or EMA.

 Let's make the intuition concrete.

 Suppose the previous averaged prediction is:

 $$
0.8
$$

 and the new prediction is:

 $$
0.6.
$$

 Suppose:

 $$
\alpha=0.9.
$$

 Then the new target is:

 $$
0.9(0.8)+0.1(0.6)=0.78.
$$

 Notice what happened.

 The new prediction moved from 0.8 to 0.6, but the averaged target only moved from 0.8 to 0.78.

 The historical predictions act like a stabilising force.

 This is exactly why the method is useful.

---

 # 30\. Why Is the Moving Average "Exponential"?

 Let's expand the recursion.

 We have:

 $$
\tilde z_t
=
\alpha \tilde z_{t-1}
+
(1-\alpha)z_t.
$$

 Substitute the previous expression:

 $$
\tilde z_t
=
\alpha
\left[
\alpha\tilde z_{t-2}
+
(1-\alpha)z_{t-1}
\right]
+
(1-\alpha)z_t.
$$

 Therefore:

 $$
\tilde z_t
=
\alpha^2\tilde z_{t-2}
+
\alpha(1-\alpha)z_{t-1}
+
(1-\alpha)z_t.
$$

 Continuing this process shows that older predictions receive weights proportional to:

 $$
\alpha^2,\alpha^3,\alpha^4,\ldots
$$

 Hence the name:

 > **exponential moving average.**

 Recent predictions receive more weight, but older predictions continue to influence the target.

---

 # 31\. Why Does Temporal Ensembling Stabilise Learning?

 Imagine that the model's predictions fluctuate:

 > 0.60, 0.85, 0.55, 0.90, 0.65.

 If we use the current prediction directly, the target is constantly jumping around.

 The EMA smooths these fluctuations.

 So temporal ensembling provides a form of **friction**.

 It prevents the training target from changing too rapidly.

 This reduces the effect of noisy predictions.

 The lecture also points out that the same basic EMA mechanism appears in optimisers such as Adam.

 The broader principle is:

 > **Averaging noisy quantities over time can produce a more stable estimate.**

---

 # 32\. Temporal Ensembling Loss

 We can now construct the semi-supervised objective.

 For labelled data:

 $$
L_{\text{sup}}
=
CE(y_i,f(x_i)).
$$

 For unlabelled data, instead of comparing the current prediction with another instantaneous prediction, we compare it with the temporally averaged target:

 $$
L_{\text{unsup}}
=
MSE(f(x_j+\eta),\tilde z_j).
$$

 So:

 $$
L
=
CE(y_i,f(x_i))
+
MSE(f(x_j+\eta),\tilde z_j).
$$

 The target $\tilde z_j$ represents the model's historical belief about that example.

---

 # 33\. The Problem with Temporal Ensembling

 Temporal ensembling is more stable.

 But it introduces another problem.

 We have to store an averaged prediction for **every unlabelled training example**.

 Imagine a dataset with ten million examples.

 For every example, we need to maintain something like:

 $$
\tilde z_1,\tilde z_2,\ldots,\tilde z_{10\,000\,000}.
$$

 This can require substantial memory.

 There is another issue.

 The averaged prediction is typically updated only when the corresponding example is processed.

 So the target can lag behind the current model.

 The larger the dataset becomes, the more problematic this can be.

 We therefore want the benefits of temporal averaging without maintaining a separate prediction for every training example.

 And this leads us to one of the most important methods in the lecture:

 > **Mean Teacher.**

---

 # 34\. Mean Teacher

 Mean Teacher makes a beautiful conceptual change.

 Temporal ensembling stores an EMA of the **predictions**.

 Mean Teacher instead stores an EMA of the **model parameters**.

 Imagine that our student model has parameters:

 $$
\theta.
$$

 We create a second model whose parameters are:

 $$
\theta'.
$$

 The teacher parameters are updated using an EMA:

 $$
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta.
$$

 So the teacher is effectively a smoothed historical version of the student.

---

 # 35\. Student and Teacher

 We now have two models.

 The **student** is the model being trained directly by gradient descent.

 The **teacher** is not updated by ordinary backpropagation.

 Instead, the teacher parameters are updated using the EMA of the student parameters.

 So:

 > Student learns from labelled data and consistency targets.

 > Teacher provides stable targets for the student.

 This gives us a very useful interpretation.

 The teacher is essentially an ensemble of previous versions of the student.

---

 # 36\. Mean Teacher Objective

 For a labelled example $(x_i,y_i)$, we use the normal supervised loss:

 $$
CE(y_i,f_\theta(x_i)).
$$

 For an unlabelled example $x_j$, we can perturb the student and teacher inputs independently.

 The consistency term can be written:

 $$
MSE
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right).
$$

 Therefore the complete objective is:

 $$
L
=
CE(y_i,f_\theta(x_i))
+
MSE
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right).
$$

 And the teacher parameters are updated using:

 $$
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta.
$$

 This is the central mathematical structure of Mean Teacher.

---

 # 37\. Why Is Mean Teacher Better Than Temporal Ensembling?

 Let's compare them.

 Temporal ensembling:

 > Maintain an EMA prediction for every example.

 Mean Teacher:

 > Maintain one EMA model.

 That is a major scalability improvement.

 The teacher can generate a target for any example immediately.

 We do not need a stored prediction associated with that particular example.

 Furthermore, the teacher is updated after every training step or batch.

 So unlike temporal ensembling, the teacher does not need to wait until an example is revisited in a future epoch.

 This makes Mean Teacher much more suitable for large datasets.

---

 # 38\. A Useful Interpretation of Mean Teacher

 There is an elegant way to understand Mean Teacher.

 Suppose the student has changed slightly during every training step:

 $$
\theta_0,\theta_1,\theta_2,\theta_3,\ldots
$$

 The teacher is approximately averaging these historical models:

 $$
\theta'
\approx
\text{EMA}(\theta_0,\theta_1,\theta_2,\ldots).
$$

 So the teacher represents a **smoothed model trajectory**.

 Instead of trusting the latest model, we trust a temporally averaged model.

 This reduces the noise caused by individual parameter updates.

---

 # 39\. The Evolution of the Methods

 Let's now pause and reconstruct the historical progression.

 This is important because the methods are much easier to remember if you understand why one leads to the next.

 We begin with:

 ### Self-training

 The model generates pseudo-labels for unlabelled examples.

 Problem:

 > The model can reinforce its own mistakes.

 Then:

 ### Co-training

 Two models teach one another.

 Problem:

 > We need useful, sufficiently independent views of the data.

 Then:

 ### Consistency regularisation

 We require predictions to remain stable under perturbations.

 Problem:

 > The target prediction may still be unreliable.

 Then:

 ### Temporal ensembling

 We average predictions over time.

 Problem:

 > We must store a target for every example, and targets can lag behind.

 Then:

 ### Mean Teacher

 We average the model itself rather than storing predictions for every example.

 This gives us:

 > a stable, continuously updated teacher.

 This progression is one of the most important conceptual stories in this lecture.

---

 # 40\. The Big Picture

 Let's now put everything into one framework.

 We have labelled data:

 $$
(x_i,y_i)
$$

 and unlabelled data:

 $$
x_j.
$$

 For the labelled examples, we know exactly what to do:

 $$
L_{\text{sup}}
=
CE(y_i,f(x_i)).
$$

 The interesting question is:

 > What should we do with $x_j$?

 Different methods answer this differently.

 Self-training:

 > Generate a pseudo-label.

 Co-training:

 > Ask another model to provide the pseudo-label.

 Consistency regularisation:

 > Ask the model to be invariant to perturbations.

 Temporal ensembling:

 > Compare the current prediction with a historical average.

 Mean Teacher:

 > Compare the student with a historical average of the model itself.

---

 # 41\. The Role of Stability

 There is an even deeper unifying principle.

 Semi-supervised learning has a dangerous feedback loop.

 We use the model's predictions to create additional training information.

 But those predictions may be wrong.

 Therefore:

 > **We need a mechanism that prevents errors from reinforcing themselves too strongly.**

 Different algorithms solve this in different ways.

 Self-training with confidence thresholds says:

 > Only trust sufficiently confident predictions.

 Co-training says:

 > Use another model as the source of supervision.

 Consistency regularisation says:

 > Use invariance under perturbations rather than assuming the exact class label.

 Temporal ensembling says:

 > Average predictions over time.

 Mean Teacher says:

 > Use an averaged model as a stable teacher.

 This is the conceptual thread connecting the entire topic.

---

 # 42\. Consistency as a General Principle

 The lecture makes an important observation:

 > Consistency regularisation is not tied to one particular semi-supervised algorithm.

 It is a general principle.

 Suppose we have some transformation $T$ that should preserve the meaning of an example.

 Then we want:

 $$
f(x)
\approx
f(T(x)).
$$

 The transformation could represent:

 - noise,
- augmentation,
- cropping,
- changes in appearance,
- or other perturbations.

 The general assumption is:

 > **Semantic-preserving transformations should not change the prediction.**

 This idea eventually connects naturally to **self-supervised learning**.

 In self-supervised learning, we will push this idea much further by constructing learning objectives directly from relationships between different views of unlabelled data.

 So semi-supervised consistency training is an important bridge toward self-supervised learning.

---

 # 43\. Curriculum Learning and Semi-Supervised Learning Together

 Let's return briefly to curriculum learning.

 At first, the two topics might seem unrelated.

 Curriculum learning says:

 > Control **which examples** the model learns from and when.

 Semi-supervised learning says:

 > Exploit **unlabelled examples** by constructing additional learning signals.

 But these ideas can interact.

 For example, suppose we are doing self-training.

 We could initially trust only very high-confidence pseudo-labels.

 As the model improves, we could gradually lower the confidence threshold.

 This creates a curriculum:

 > easy, highly reliable pseudo-labels first;

 then:

 > progressively more uncertain examples later.

 So the concepts can reinforce one another.

---

 # 44\. Active Learning

 There is another related technique that we should distinguish from semi-supervised learning.

 It is called **active learning**.

 Semi-supervised learning says:

 > "We have unlabelled data. How can we learn from them without labels?"

 Active learning takes a different approach.

 It says:

 > "We have unlabelled data. Which examples should we ask a human to label?"

 Imagine we have one million unlabelled images.

 We probably cannot afford to manually label all one million.

 An active-learning system might identify the examples that would provide the most useful information.

 Those examples are sent to a human annotator.

 The newly labelled examples are then added to the training set.

 So the loop is:

 > train model\
>  → identify informative examples\
>  → obtain labels\
>  → retrain model\
>  → repeat.

---

 # 45\. Semi-Supervised Learning Versus Active Learning

 The distinction is worth remembering.

 **Semi-supervised learning:**

 > Make use of unlabelled examples without necessarily obtaining new labels.

 **Active learning:**

 > Select which unlabelled examples should receive labels.

 So semi-supervised learning tries to extract value from the existing unlabelled data.

 Active learning spends the limited labelling budget strategically.

 They can also be combined.

 For example, a system might:

 - use semi-supervised learning to exploit most unlabelled examples,
- while using active learning to decide which additional examples should be manually labelled.

---

 # 46\. An End-to-End Example

 Let's put everything together using a concrete example.

 Suppose we are building a system that classifies X-ray images.

 We have:

 > 10,000 labelled images

 and:

 > 1,000,000 unlabelled images.

 A fully supervised system can train only on the 10,000 labelled images.

 But we want to exploit the million unlabelled images.

 We begin with self-training.

 The model predicts labels for the unlabelled images.

 We keep only highly confident predictions.

 But we notice that some systematic errors are being reinforced.

 We could introduce a second model and use co-training if we have genuinely different views.

 Alternatively, we can use consistency regularisation.

 For each unlabelled X-ray, we create two perturbed versions.

 We require the predictions to remain consistent.

 But early in training the predictions are unstable.

 So we introduce temporal averaging.

 However, maintaining a prediction for every one of our million images becomes inconvenient.

 So instead we use Mean Teacher.

 The student learns from:

 - real labels,
- plus consistency with the teacher.

 The teacher is an EMA of the student.

 Now we have a stable target that improves throughout training.

 That is the entire story in one example.

---

 # 47\. Exam Question: Why Can Naive Self-Training Fail?

 If I asked you this in an exam, a strong answer would be:

 Naive self-training can fail because the model uses its own predictions as pseudo-labels. If those predictions are incorrect, the errors become part of the training signal. This can create confirmation bias, where incorrect predictions are repeatedly reinforced and the model may collapse toward incorrect or overly confident predictions.

 The key phrase to remember is:

 > **self-reinforcing errors.**

---

 # 48\. Exam Question: Why Does Co-Training Help?

 A good answer is:

 Co-training uses two models trained on different views of the data. If the models make sufficiently independent errors, one model's confident predictions can provide useful pseudo-labels for the other. This reduces the dependence on a single model's potentially self-reinforcing mistakes.

 The key phrase is:

 > **independent or complementary views.**

---

 # 49\. Exam Question: What Is Consistency Regularisation?

 A good answer is:

 Consistency regularisation adds an unsupervised loss that encourages a model to produce similar predictions for different perturbed or augmented versions of the same unlabelled input.

 Mathematically:

 $$
L_{\text{unsup}}
=
D(f(x),f(T(x))),
$$

 where $T(x)$ is a perturbation or augmentation that should preserve the semantic identity of $x$.

 The key phrase is:

 > **same meaning, same prediction.**

---

 # 50\. Exam Question: What Is the Π-Model?

 The Π-model is an early consistency-regularisation method.

 The same input is subjected to different stochastic perturbations, and the model is trained to make consistent predictions for the resulting versions.

 So the model effectively learns:

 $$
f(x+\eta_1)
\approx
f(x+\eta_2).
$$

 The supervised loss is applied to labelled examples, while the consistency loss provides the learning signal for unlabelled examples.

---

 # 51\. Exam Question: What Is UDA?

 UDA stands for **Unsupervised Data Augmentation**.

 It applies a weak or original version and a strongly augmented version of an unlabelled example and encourages their predictions to agree.

 For unlabelled data, the consistency loss is typically used only when the model's original prediction is sufficiently confident.

 The important ideas are:

 - strong augmentation,
- consistency,
- and confidence-based filtering.

---

 # 52\. Exam Question: What Is Temporal Ensembling?

 Temporal ensembling maintains an exponentially averaged prediction for each training example.

 The update is:

 $$
\tilde z_i^t
=
\alpha\tilde z_i^{t-1}
+
(1-\alpha)z_i^t.
$$

 The averaged prediction becomes a more stable target for consistency training.

 It reduces the effect of noisy predictions by incorporating information from previous training iterations.

---

 # 53\. Exam Question: Why Is Temporal Ensembling an EMA?

 Because the current target is a weighted combination of:

 - the previous averaged target,
- and the current prediction.

 Repeated substitution gives exponentially decreasing weights to older predictions.

 Recent predictions have more influence, while older predictions still contribute.

 Hence:

 > **exponential moving average.**

---

 # 54\. Exam Question: What Is the Main Weakness of Temporal Ensembling?

 There are two important weaknesses.

 First:

 > We need to store a moving-average prediction for every example.

 This scales poorly for very large datasets.

 Second:

 > Each example's averaged target is updated only when that example is processed.

 Therefore the target can lag behind the current model.

 These limitations motivate Mean Teacher.

---

 # 55\. Exam Question: What Is Mean Teacher?

 Mean Teacher maintains two models:

 - a student,
- and a teacher.

 The student is updated using gradient descent.

 The teacher is updated as an EMA of the student parameters:

 $$
\theta'
=
\alpha\theta'
+
(1-\alpha)\theta.
$$

 The student is trained to agree with the teacher on unlabelled examples.

 So the teacher provides a stable consistency target.

---

 # 56\. Exam Question: Why Is Mean Teacher More Scalable?

 Because we no longer need to store a separate averaged prediction for every example.

 Instead, we maintain a single teacher model.

 The teacher can generate a prediction for any example when needed.

 Furthermore, the teacher parameters can be updated after every batch.

 Therefore the target can evolve continuously with training.

---

 # 57\. Exam Question: Compare Temporal Ensembling and Mean Teacher

 Here is the essential comparison.

 Temporal ensembling averages:

 > **predictions for each example.**

 Mean Teacher averages:

 > **model parameters.**

 Temporal ensembling therefore requires:

 > one stored target per example.

 Mean Teacher requires:

 > one additional model.

 Temporal ensembling can lag because an example's prediction is updated only when the example is revisited.

 Mean Teacher can update the teacher after every student update.

 So Mean Teacher is generally much better suited to large-scale datasets.

---

 # 58\. Exam Question: What Is Model Stability?

 Model stability refers to mechanisms that prevent unreliable predictions from causing unstable or self-reinforcing learning in semi-supervised training.

 Examples include:

 - confidence thresholds,
- independent models in co-training,
- consistency constraints,
- temporal averaging,
- and EMA teacher models.

 The broader problem is:

 > **confirmation bias from noisy pseudo-supervision.**

---

 # 59\. Exam Question: What Is the Overall Semi-Supervised Objective?

 The general structure is:

 $$
L
=
L_{\text{sup}}
+
\lambda L_{\text{unsup}}.
$$

 The supervised component might be:

 $$
L_{\text{sup}}
=
CE(y_i,f(x_i)).
$$

 The exact form of $L_{\text{unsup}}$ depends on the method.

 For self-training:

 $$
L_{\text{unsup}}
=
CE(\hat y_j,f(x_j)).
$$

 For consistency regularisation:

 $$
L_{\text{unsup}}
=
D(f(x_j),f(x_j+\eta)).
$$

 For temporal ensembling:

 $$
L_{\text{unsup}}
=
MSE(f(x_j+\eta),\tilde z_j).
$$

 For Mean Teacher:

 $$
L_{\text{unsup}}
=
MSE
\left(
f_\theta(x_j+\eta),
f_{\theta'}(x_j+\eta')
\right).
$$

 So the evolution is largely an evolution in the definition of the unsupervised target.

---

 # 60\. The Full Timeline to Memorise

 If you remember only one conceptual diagram from this lecture, remember this sequence:

 **Self-training**

 Model generates pseudo-labels.

 ↓

 Problem: confirmation bias.

 ↓

 **Co-training**

 Two models teach each other.

 ↓

 Problem: requires suitable independent views.

 ↓

 **Consistency regularisation**

 Same input under perturbations should give the same prediction.

 ↓

 Problem: the target can still be unreliable.

 ↓

 **Temporal ensembling**

 Average predictions over time.

 ↓

 Problem: storage and lag.

 ↓

 **Mean Teacher**

 Average the model parameters instead.

 ↓

 Result:

 **stable teacher + student consistency training.**

 That is the intellectual progression.

---

 # 61\. Connecting This Back to Curriculum Learning

 Let's return to our first topic.

 Curriculum learning controls the **difficulty or ordering of examples**.

 Semi-supervised learning controls how we extract information from **labelled and unlabelled examples**.

 Both are examples of a broader philosophy:

 > Don't simply throw all available data into a learning algorithm and hope for the best.

 Instead, think carefully about:

 - what information the model receives,
- when it receives it,
- how reliable that information is,
- and how the training process can be stabilised.

 This is an important mindset for advanced machine learning.

---

 # 62\. A Practical Design Recipe

 Suppose you are given a new semi-supervised learning problem in an exam or research project.

 A sensible thought process is:

 First, identify the labelled and unlabelled datasets.

 Second, establish a strong supervised baseline.

 Third, ask whether the model can generate useful pseudo-labels.

 If yes, consider self-training.

 Then ask:

 > How can I prevent incorrect pseudo-labels from being reinforced?

 Possible answers include:

 - confidence thresholds,
- data augmentation,
- consistency regularisation,
- multiple models,
- or EMA teachers.

 Then ask:

 > Is the target itself stable?

 If not, temporal averaging or Mean Teacher may help.

 This gives you a systematic way to reason about method selection.

---

 # 63\. Common Mistakes

 Let's finish with several common misconceptions.

 ### Mistake 1: "Semi-supervised learning means no labels."

 No.

 Semi-supervised learning explicitly assumes that **some labelled data exist** alongside unlabelled data.

 ### Mistake 2: "Pseudo-labels are ground truth."

 No.

 Pseudo-labels are model predictions.

 They are potentially useful but potentially wrong.

 ### Mistake 3: "Consistency means every input should receive the same prediction."

 No.

 It means different **valid perturbations of the same input** should receive consistent predictions.

 Different unrelated inputs should obviously be allowed to have different predictions.

 ### Mistake 4: "The teacher in Mean Teacher is trained using backpropagation."

 Not directly.

 The teacher is updated using an EMA of the student's parameters.

 ### Mistake 5: "Temporal ensembling and Mean Teacher are identical."

 They use the same stabilising idea—EMA—but apply it to different quantities.

 Temporal ensembling averages:

 > predictions.

 Mean Teacher averages:

 > parameters.

---

 # 64\. Final Summary

 Let's bring the entire lecture together.

 Curriculum learning asks:

 > **In what order should the model encounter training examples?**

 Its central intuition is:

 > start with easier examples and gradually increase difficulty.

 Semi-supervised learning asks:

 > **How can we learn from a mixture of labelled and unlabelled data?**

 The general objective is:

 $$
L
=
L_{\text{sup}}
+
\lambda L_{\text{unsup}}.
$$

 Self-training creates pseudo-labels from the model's own predictions.

 Its main weakness is confirmation bias.

 Co-training uses multiple models and relies on their errors being sufficiently independent.

 Consistency regularisation avoids requiring explicit labels by requiring predictions to remain stable under perturbations.

 The Π-model is an early implementation of this idea.

 UDA strengthens the idea using strong data augmentation and confidence filtering.

 Temporal ensembling improves stability by maintaining an exponential moving average of predictions over time.

 Its weakness is that we need to maintain a prediction for every example and updatesTherefore, design the learning process so that useful signals are amplified while unreliable signals can lag.

 Mean Teacher solves this by maintaining an EMA of the model itself.

 The student is trained by gradient descent.

 The teacher is an EMA of the student.

 The student learns to match the teacher on unlabelled data.

 And the central idea behind the entire progression is:

 > **Unlabelled data are useful, but only if we can extract a reliable learning signal from them without allowing the model's own mistakes to become self-reinforcing.**

 That is the key lesson.

---

 # 65\. Final Mental Checklist

 Before finishing, I want you to be able to answer these questions without looking at your notes.

 What is curriculum learning?

 Why might easy examples be useful early in training?

 What does "difficulty" mean?

 What is semi-supervised learning?

 Why are unlabelled examples valuable?

 What is a pseudo-label?

 Why does self-training suffer from confirmation bias?

 How does confidence thresholding help?

 What is co-training?

 Why does co-training require different views of the data?

 What is consistency regularisation?

 Why should predictions be invariant to semantic-preserving perturbations?

 What is the Π-model?

 What is UDA?

 Why does UDA use strong augmentation?

 Why might confidence filtering be necessary?

 What is temporal ensembling?

 What is an exponential moving average?

 Why does EMA stabilise training?

 What are the limitations of temporal ensembling?

 What is Mean Teacher?

 How is the teacher updated?

 What is the difference between temporal ensembling and Mean Teacher?

 And finally:

 > **Why is stability so important in semi-supervised learning?**

 If you can answer those questions clearly, you understand the core material of this lecture.

 The overarching story is simple:

 **Use the data you have.\
 Introduce useful information gradually.\
 Exploit unlabelled data.\
 But never forget that a model's own predictions can be wrong.\
 Therefore, design the learning process so that useful signals are amplified while unreliable signals are stabilised.**

 That is the central idea connecting curriculum learning and modern semi-supervised learning.
