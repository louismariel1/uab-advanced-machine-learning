# Advanced Machine Learning Tutorial: Contrastive Learning, Student–Teacher Models, Ensembles, and Boosting

We can extend the original cat-versus-dog example to explain four important learning strategies:

1. Contrastive learning: Learn which images are similar and which are different.
2. Student–teacher learning: Transfer knowledge from a stronger or more established model to another model.
3. Ensemble learning: Combine several models to make a better prediction.
4. Boosting: Train models sequentially so that later models correct mistakes made by earlier ones.

The important distinction is that these methods address different questions. Contrastive learning focuses on what representations a model should learn. Student–teacher learning focuses on how knowledge can be transferred. Ensembles focus on how multiple models can work together. Boosting focuses on how a sequence of models can progressively improve predictions.

We will use the same practical example throughout: a system that recognizes cats and dogs in photographs. This keeps the explanations connected to the backbone-and-head architecture from the original tutorial.

# 1. Contrastive learning: Learning what looks similar

## 1.1 The main idea

Imagine you have 100,000 animal photographs but no labels telling you which contain cats and which contain dogs.

A conventional supervised model needs labeled examples to learn the distinction directly. Contrastive learning offers another approach: teach the model to recognize relationships between examples.

5 Reasons Pawever Plus Helps Senior Cats

How Caring for Senior Dogs Impacted One Foster Caregiver | ASPCA

Punya Hewan Peliharaan? Hindari Tanaman Ini di Rumah

Suppose we show the model two photographs of the same cat, taken from different angles. We want their learned representations to be similar. We also show it a cat and a dog, and encourage their representations to be different.

The model learns a representation in which related images are close together and unrelated images are farther apart.

This is called contrastive learning.

## 1.2 How it works with a backbone and a projection head

The architecture can be represented as follows:

Input image

Photograph of a cat

Backbone

Extracts visual features

Projection head

Maps features into a contrastive embedding space

Embedding vector

$$
\mathbf{z}=[0.82,\;0.51,\;-0.26]
$$

A numerical representation of the image

There is an important difference from the original tutorial:

- In ordinary classification, the head predicts a class such as cat or dog.
- In contrastive learning, the projection head produces an embedding designed to support comparisons between images.

The backbone still extracts features, but the training objective changes.

## 1.3 A small numerical example

For simplicity, suppose the model represents each image using only two numbers:

$$
\mathbf{z}=[z_1,z_2]
$$

These two numbers are illustrative coordinates in an embedding space, not necessarily interpretable visual features.

Consider three images:

| Image | Embedding |
| --- | --- |
| Cat A | $[0.9,0.1]$ |
| Cat B | $[0.8,0.2]$ |
| Dog A | $[-0.8,0.1]$ |

Cat A and Cat B have similar embeddings. Dog A has a different embedding.

We can calculate their Euclidean distances.

For the two cats:

$$
d(A,B)=\sqrt{(0.9-0.8)^2+(0.1-0.2)^2}
$$

$$
d(A,B)=\sqrt{0.02}\approx0.141
$$

For Cat A and Dog A:

$$
d(A,D)=\sqrt{(0.9+0.8)^2+(0.1-0.1)^2}
$$

$$
d(A,D)=1.7
$$

The two cat images are much closer than the cat and dog images.

The training process attempts to produce this kind of structure in the embedding space.

### Illustrative embedding space

Conceptual illustration, not a plot of the exact numerical embeddings above.

### What does the training actually optimize?

A contrastive loss typically rewards similarity between a positive pair and penalizes insufficient separation from negative examples.

One common formulation is the InfoNCE loss:

$$
\mathcal{L}_i =
-\log
\frac{
\exp(\operatorname{sim}(z_i,z_i^+)/\tau)
}{
\sum_j \exp(\operatorname{sim}(z_i,z_j)/\tau)
}
$$

Here:

- $z_i$ is the embedding of an image.
- $z_i^+$ is the embedding of a related image, such as another augmented view of the same photograph.
- $\operatorname{sim}$ measures similarity, often cosine similarity.
- $\tau$ is a temperature parameter.
- The denominator includes competing examples.

Minimizing the loss encourages the model to give the positive pair a high similarity relative to the competing examples.

In methods such as SimCLR, positive pairs can be created by taking one photograph and applying two different transformations, such as random crops, color changes, or flips.

Important nuance: Contrastive learning does not automatically mean that all cats will be close together and all dogs will be far away. In self-supervised learning, positive pairs may be two views of the same image, without any knowledge of the animal's class. The model learns whatever useful similarities the training objective and data encourage.

## 1.4 How do we use the learned backbone?

After contrastive pretraining, we can discard the projection head and reuse the backbone.

Contrastively pretrained backbone

New cat-versus-dog classification head

Prediction: Cat — 95%

We then train the classification head using a smaller labeled dataset, optionally fine-tuning the backbone.

This combines self-supervised or weakly supervised representation learning with conventional supervised classification.

### When is contrastive learning useful?

- When there are many unlabeled images but relatively few labeled ones.
- When learning image or text representations that can be reused across tasks.
- In image retrieval, where we want to find images similar to a query.
- In multimodal learning, where models learn aligned representations of images and text, as in CLIP-style approaches.

The key takeaway: Contrastive learning teaches a model how to represent relationships between examples, rather than directly teaching it a class label for every training example.

# 2. Student–teacher learning: Transferring knowledge from one model to another

## 2.1 The main idea

Suppose we have already trained an excellent cat-versus-dog classifier. It uses a large backbone and achieves high accuracy, but it is expensive to run on a mobile phone.

We want a smaller model that can make predictions quickly.

Instead of training the small model from scratch using only the correct class labels, we let it learn from the predictions of the larger model.

- Teacher: The established model that provides guidance.
- Student: The model that learns from the teacher's outputs.
- Distillation: The process of transferring knowledge from teacher to student.

Kimi K2.6: Running 1 Trillion Parameters Locally

### Teacher

- Larger model
- Often more accurate
- More computationally expensive

Core ML – inference on iOS

### Student

- Smaller model
- Faster inference
- Learns from teacher outputs

The teacher and student do not need to have the same architecture. For example, the teacher might use a large Vision Transformer while the student uses a lightweight convolutional neural network.

## 2.2 The basic architecture

Teacher model

Large backbone + classification head

Teacher's predictions

Cat: 80% · Dog: 20%

Student model

Learns to reproduce the teacher's output distribution

Student's predictions

Cat: 78% · Dog: 22%

The student learns by comparing its predictions with the teacher's predictions. During ordinary distillation, the teacher's weights are fixed, and the student is trained to minimize a distillation loss.

## 2.3 A small numerical example

Suppose the teacher examines a photograph and produces the following probabilities:

| Class | Teacher |
| --- | --- |
| Cat | 80% |
| Dog | 20% |

The student initially produces:

| Class | Student |
| --- | --- |
| Cat | 55% |
| Dog | 45% |

The student should adjust its parameters to bring its predictions closer to the teacher's.

For a simple illustration, we can compare the distributions using the Kullback–Leibler (KL) divergence:

$$
D_{\mathrm{KL}}(P\|Q)
=\sum_c P(c)\log\frac{P(c)}{Q(c)}
$$

Here, $P$ is the teacher's probability distribution and $Q$ is the student's distribution.

For the numbers above:

$$
D_{\mathrm{KL}}
=0.8\log\frac{0.8}{0.55}
+0.2\log\frac{0.2}{0.45}
$$

$$
D_{\mathrm{KL}}\approx0.129
$$

This positive value indicates a difference between the distributions. Training the student to reduce this divergence brings its output distribution closer to the teacher's.

The exact distillation loss used in practice can include a temperature parameter, a scaling factor, and an additional loss based on the ground-truth labels.

### Why not simply teach the student that the answer is "cat"?

Because the teacher's output contains more information than the final class.

Suppose the teacher predicts:

- Cat: 80%
- Dog: 20%

This suggests that the image has features that the teacher considers more consistent with a cat, while still having some resemblance to a dog.

Compare that with a different image for which the teacher predicts:

- Cat: 99%
- Dog: 1%

Both images receive the same class label, but the teacher expresses different levels of uncertainty.

These probability distributions provide what is often called soft-target information. In multiclass tasks, the relative probabilities assigned to alternative classes can convey additional useful information.

However, soft targets are not guaranteed to be better than ground-truth labels. The teacher can be wrong, and its knowledge is limited by its training and architecture.

## 2.4 Where do the backbone and head fit in?

There are several possibilities.

Approach A: Distill predictions

The student has its own backbone and classification head. It learns to reproduce the teacher's predictions.

Approach B: Distill intermediate features

The student also learns to match selected representations from the teacher's backbone. Because the models may have different architectures, a projection layer may be needed to align feature dimensions.

Approach C: Distill both

The student learns from the teacher's output probabilities and intermediate features, potentially alongside the true labels.

This gives a general training objective such as:

$$
\mathcal{L}_{\text{student}}
=
\alpha\mathcal{L}_{\text{labels}}
+\beta\mathcal{L}_{\text{distillation}}
+\gamma\mathcal{L}_{\text{features}}
$$

The coefficients $\alpha,\beta,\gamma$ control the relative importance of the losses. Some terms can be omitted depending on the method.

## 2.5 What about self-supervised student–teacher learning?

Not every student–teacher system is designed to compress a large model into a smaller one.

In methods such as BYOL and DINO, teacher–student architectures can help learn useful representations without relying on conventional class labels.

A common pattern is:

1. Create different augmented views of an image.
2. Process one view through the student network and another through the teacher network.
3. Train the student to match a target derived from the teacher's representation.
4. Update the teacher using a method such as an exponential moving average of the student's weights.

The teacher can therefore evolve during training rather than remaining fixed.

This is different from conventional knowledge distillation, where a pretrained teacher is typically held fixed.

### When is student–teacher learning useful?

- Deploying a large model's capabilities in a smaller model.
- Reducing inference cost and latency.
- Transferring useful representations between architectures.
- Learning representations from unlabeled images in self-supervised training.

The key takeaway: Student–teacher learning transfers information from one model to another. The transferred information can be predictions, intermediate features, or learned representations.

# 3. Ensemble learning: Combining multiple models

## 3.1 The main idea

Now suppose we have three cat-versus-dog classifiers.

- Model A is particularly good at recognizing facial features.
- Model B is good at recognizing body shape.
- Model C performs well on different lighting conditions.

Rather than selecting only one model, we combine their predictions.

This is called ensemble learning.

The intuition is similar to asking several independent experts for an opinion and combining their judgments. The method works best when the models have useful predictive ability and make at least some different errors.

Same input image

Photograph of an animal

Model A

Cat: 90%

Model B

Cat: 60%

Model C

Cat: 80%

Ensemble aggregation

Combines the models' predictions

Final prediction: Cat

Average cat probability: 76.7%

## 3.2 A small numerical example

Suppose three models predict the following probabilities for the same photograph.

| Model | Cat | Dog |
| --- | --- | --- |
| A | 90% | 10% |
| B | 60% | 40% |
| C | 80% | 20% |

A simple ensemble takes the average of the probabilities:

$$
P(\text{cat})=\frac{0.90+0.60+0.80}{3}
$$

$$
P(\text{cat})=0.767
$$

Similarly,

$$
P(\text{dog})=\frac{0.10+0.40+0.20}{3}
=0.233
$$

The ensemble therefore predicts:

- Cat: 76.7%
- Dog: 23.3%

The class with the higher average probability is cat.

Notice that the ensemble's probability is lower than Model A's 90%. This is not a problem: combining models does not necessarily increase confidence. The objective is to improve predictive reliability, not to maximize the confidence score.

These probabilities are illustrative. Averaging probabilities does not guarantee better accuracy or perfectly calibrated confidence.

## 3.3 Different ways to combine models

There are several common ensemble strategies.

### A. Hard voting

Each model votes for one class.

| Model | Vote |
| --- | --- |
| A | Cat |
| B | Dog |
| C | Cat |

Cat receives two votes and dog receives one, so the ensemble predicts cat.

Hard voting ignores how confident each model is.

### B. Soft voting

Each model produces class probabilities, which are averaged.

This is the numerical example above. Soft voting uses more information than hard voting, although the models' probability estimates should ideally be reasonably calibrated.

### C. Weighted averaging

Some models are more reliable than others. We can assign different weights:

$$
P(y\mid x)=\sum_{m=1}^{M}w_mP_m(y\mid x)
$$

where the weights satisfy

$$
\sum_{m=1}^{M}w_m=1
$$

For example, suppose Model A receives weight 0.5, Model B receives weight 0.2, and Model C receives weight 0.3.

Then:

$$
P(\text{cat})=
0.5(0.90)+0.2(0.60)+0.3(0.80)
$$

$$
P(\text{cat})=0.81
$$

The weighted ensemble predicts an 81% cat probability.

The weights should be determined using an appropriate validation procedure rather than arbitrarily adjusted to improve performance on the test set.

## 3.4 What happens to the backbone and head?

An ensemble can combine models in different ways.

Independent-model ensemble

Each model has its own backbone and head. Their predictions are combined at the end.

Shared-backbone, multiple-head system

A common backbone extracts features, and several heads make different predictions. Their outputs are combined. This is a multi-head architecture, although it is not necessarily equivalent to an ensemble of independent models.

Heterogeneous ensemble

Different models may use different architectures, such as a ResNet, a Vision Transformer, and a smaller convolutional network.

In the conventional independent-model ensemble, all models are usually trained before their predictions are combined.

## 3.5 Why can an ensemble outperform an individual model?

Imagine the following errors:

- Model A misclassifies a cat in dim lighting.
- Model B recognizes the cat correctly.
- Model C also recognizes the cat correctly.

The combined prediction can correct Model A's error.

But if all three models make the same mistake, combining their predictions may not help.

This leads to an important principle: Model diversity matters.

Ensembles tend to benefit when their members make partly different errors. Simply duplicating the same model with identical predictions usually provides little benefit.

Common ensemble methods include:

- Bagging: Train models on different bootstrap samples of the training data. Random forests are a classic example.
- Voting and averaging: Combine predictions from multiple trained models.
- Stacking: Train a separate model, called a meta-model, to combine predictions from base models.

### When is ensemble learning useful?

- When predictive performance is more important than inference cost.
- When several models have complementary strengths.
- In high-stakes or challenging prediction tasks where validation demonstrates a meaningful benefit.
- In competitions and production systems that can afford multiple model evaluations.

The key takeaway: Ensemble learning combines predictions from multiple models. Its success depends on the quality of the models and how differently they make mistakes.

# 4. Boosting: Learning from previous mistakes

## 4.1 The main idea

Boosting is a particular family of ensemble methods.

Instead of training several models independently and combining their predictions afterward, boosting trains models sequentially. Each new model is trained to address weaknesses in the current ensemble.

Return to the cat-versus-dog example.

Suppose our first model is good at recognizing typical cats and dogs but struggles with unusual images, such as a small dog photographed from a distance.

We then train a second model that focuses more on the examples the first model handles poorly. A third model can address remaining weaknesses.

The final prediction combines the contributions of the models.

Training data

Labeled cat and dog photographs

Model 1

Learns the initial classification patterns

Identify weaknesses

Find or emphasize poorly predicted examples

Model 2

Learns additional patterns

Repeat the process

Train further models as needed

Final boosted ensemble

Combines the models' contributions

The exact procedure depends on the boosting algorithm. AdaBoost adjusts example weights, while gradient boosting fits new models to gradients of a loss function. Neither method is simply a matter of telling a model which examples it got wrong.

## 4.2 A small numerical example using AdaBoost

AdaBoost is a useful way to understand the central intuition behind boosting.

Suppose we have 10 training photographs:

- 6 cats
- 4 dogs

Initially, every image receives equal weight:

$$
w_i=\frac{1}{10}=0.1
$$

We train a simple classifier, called a weak learner, which correctly classifies 8 images and misclassifies 2.

Its weighted error is initially:

$$
\epsilon_1=\frac{2}{10}=0.2
$$

AdaBoost assigns the learner a contribution weight:

$$
\alpha_1=\frac{1}{2}
\ln\left(\frac{1-\epsilon_1}{\epsilon_1}\right)
$$

Substituting the error:

$$
\alpha_1=\frac{1}{2}\ln\left(\frac{0.8}{0.2}\right)
$$

$$
\alpha_1=\frac{1}{2}\ln(4)\approx0.693
$$

A lower error gives the learner a larger contribution, assuming its error is below 0.5 in this binary setting.

### Step 1: Increase the importance of the misclassified examples

The two incorrectly classified images receive greater relative weight. The correctly classified examples receive less relative weight.

After the standard AdaBoost update and normalization, each of the two incorrectly classified examples has weight 0.25, while each of the eight correctly classified examples has weight 0.0625.

Check the total:

$$
2(0.25)+8(0.0625)=1
$$

The weights sum to one.

The next learner therefore faces a training distribution that emphasizes the two difficult examples.

### Step 2: Train a second learner

Suppose the second learner correctly classifies both previously misclassified images and makes mistakes on two other examples.

Its weighted error is:

$$
\epsilon_2=2(0.0625)=0.125
$$

Its contribution becomes:

$$
\alpha_2=\frac{1}{2}
\ln\left(\frac{1-0.125}{0.125}\right)
$$

$$
\alpha_2=\frac{1}{2}\ln(7)\approx0.973
$$

The second learner has a larger contribution weight than the first because it achieves a lower weighted error.

The example weights are updated again, and training continues.

These calculations illustrate the standard binary AdaBoost update with normalized example weights and a weak learner whose weighted error is below 0.5.

## 4.3 How does boosting make the final prediction?

In binary AdaBoost, each learner predicts a class encoded as either $-1$ or $+1$.

The final ensemble calculates a weighted sum:

$$
F(x)=\sum_{m=1}^{M}\alpha_m h_m(x)
$$

where:

- $h_m(x)$ is the prediction of learner $m$.
- $\alpha_m$ is that learner's contribution weight.
- $M$ is the number of learners.

The final prediction is determined by the sign of the sum:

$$
\hat y=\operatorname{sign}(F(x))
$$

A learner with a larger contribution weight has more influence on the final decision.

This differs from a simple majority vote, in which each model has equal influence.

## 4.4 Gradient boosting: The connection to modern machine learning

AdaBoost is only one boosting algorithm. Another important family is gradient boosting.

Suppose we are training a cat-versus-dog classifier using a differentiable loss function. We can represent the current model's output as $F(x)$.

Gradient boosting proceeds by:

1. Starting with an initial prediction.
2. Computing the direction in which the predictions should change to reduce the loss.
3. Training a new learner to approximate that direction.
4. Adding the new learner's contribution to the existing ensemble.
5. Repeating the process.

A simplified update is:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

Here, $h_m(x)$ is the new learner's contribution, and $\eta$ is the learning rate.

Unlike the original AdaBoost example, gradient boosting does not necessarily increase weights on misclassified examples explicitly. It uses the gradient of the chosen loss function to determine what the next learner should improve.

Popular gradient-boosting implementations include XGBoost, LightGBM, and CatBoost. They are particularly effective for many structured or tabular datasets, although their capabilities are not limited to that setting.

## 4.5 What about the backbone and head?

Boosting is most commonly associated with decision trees, often small trees called weak learners. In that case, there may be no neural-network backbone/head separation at all.

For example:

- Learner 1 may use rules about ears and face shape.
- Learner 2 may capture additional patterns related to body proportions.
- Learner 3 may refine decisions for difficult examples.

The final model combines their outputs.

Boosting can also be applied using other differentiable or trainable learners, including neural networks, although the details vary by algorithm.

A crucial distinction is that boosting does not ordinarily train one shared neural-network backbone and replace its head repeatedly. Instead, it adds successive learners to an existing predictive model.

### When is boosting useful?

- Classification and regression on structured data.
- Problems where successive refinement of predictions works well.
- Tasks where nonlinear relationships and feature interactions matter.
- Situations where gradient-boosted trees provide a strong performance baseline.

The key takeaway: Boosting builds an ensemble sequentially, with each new learner guided by the errors or loss remaining in the current model.

# 5. Compare all four learning strategies

Now we can put the four approaches side by side using our cat-versus-dog classifier.

| Learning type | Main question | What happens during training? | What is combined or learned? |
| --- | --- | --- | --- |
| Contrastive learning | Which examples should have similar representations? | Learns from positive and negative pairs, or related contrastive objectives | Useful embeddings |
| Student–teacher learning | How can one model learn from another? | A student matches teacher predictions, features, or representations | Knowledge transferred to the student |
| Ensemble learning | Can multiple models make a better prediction together? | Multiple models are trained and their outputs are aggregated | Predictions from multiple models |
| Boosting | Can we improve a model by addressing its remaining errors? | Learners are added sequentially, guided by errors or loss | A weighted or additive predictive model |

## 5.1 Compare their architectures

## Contrastive learning

Learns representations through relationships between examples.

Images and augmented views

Backbone + projection head

Contrastive loss

## Student–teacher learning

Transfers knowledge between models.

Teacher

Student

Teacher targets guide student training

## Ensemble learning

Combines the predictions of multiple models.

Model A

Model B

Model C

Aggregation → prediction

## Boosting

Adds learners sequentially to improve the current model.

Learner 1

Learner 2 → addresses remaining loss

Learner 3 → refines the ensemble

Combined prediction

# 6. How these approaches relate to each other

These four methods are not mutually exclusive. They can be combined in the same machine-learning pipeline.

Consider a real-world cat-versus-dog recognition system.

1. Unlabeled animal images

A large collection of photographs

2. Contrastive pretraining

Learn reusable visual representations

3. Train a teacher model

Fine-tune the backbone and classification head on labeled images

4. Distill into a student

Transfer useful predictive behavior to a smaller model

5. Deploy or ensemble

Use the student alone, or combine multiple models if justified

Boosting is another possible route, especially for tabular features or a separate classification stage. However, it is not automatically the next step after distillation.

The right combination depends on the task, the available labels, the computational budget, and the performance requirements.

## 6.1 Three important distinctions

First: contrastive learning and supervised classification have different objectives. Contrastive learning shapes a representation space. Classification directly optimizes predictions for target classes. Contrastive pretraining can make subsequent classification more effective.

Second: student–teacher learning and ensembles have different deployment goals. Distillation often transfers knowledge into one deployable student. An ensemble usually retains multiple predictive models, which can improve performance but increase inference cost.

Third: boosting is a particular kind of ensemble, not a synonym for ensemble learning. A conventional ensemble can train its models independently. Boosting adds learners sequentially, using the current model's errors or loss to guide further learning.

# 7. Check your understanding

## Mini-quiz

0 of 4 answered

1. You have 200,000 unlabeled animal images and want to learn reusable visual representations before labeling a small dataset. Which method is the best fit?

Contrastive learning

Hard voting

Boosting

2. A large model is accurate but too expensive to run on a phone. You want a smaller model to imitate its predictions. Which method fits?

Contrastive learning only

Student–teacher learning

Independent-model ensemble

3. Three independently trained models make different errors. You average their cat and dog probabilities. What are you using?

Ensemble learning

Contrastive loss

Feature distillation

4. A new weak learner is trained after examining the remaining loss from an existing model. Which approach does this describe?

Ordinary soft voting

Transfer learning only

Boosting

 Check answers

# Final takeaway

The backbone-and-head example gives us a starting point, but advanced machine learning introduces several different ways to train and combine models:

- Contrastive learning: Learn meaningful representations by comparing examples.
- Student–teacher learning: Learn from another model's predictions, features, or representations.
- Ensemble learning: Combine the predictions of multiple models.
- Boosting: Build a stronger predictive model by adding learners sequentially.

A useful way to remember the differences is to ask four questions:

1. What should the model learn to represent?
2. From whom or what should the model learn?
3. How can multiple models contribute to one prediction?
4. How can successive learners improve an existing model?

These questions describe different dimensions of machine learning, and understanding them makes it easier to choose the right method for a particular problem.
