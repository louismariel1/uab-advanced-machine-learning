# Advanced Machine Learning — MCQ Practice Bank

Topic: Few-Shot Learning and Meta-Learning Based on: Practice Exam 6 Format: Multiple-choice questions with answers and explanations Difficulty: Intermediate to advanced

This practice bank covers the key concepts in your exam, including episodic evaluation, supervised baselines, Reptile, Prototypical Networks (ProtoNet), mathematical calculations, tensor shapes, debugging, and experimental design.

Unlike a simple revision sheet, these questions test both conceptual understanding and your ability to apply the concepts in exam-style scenarios.

## Part A — Few-Shot Learning and Episodic Evaluation

### Question 1. What is the primary goal of few-shot learning?

Which statement is correct?

- A. To maximise training accuracy on a fixed set of classes.
- B. To recognise new classes using only a small number of labelled examples per class.
- C. To eliminate the need for labelled data completely.
- D. To train a separate deep neural network from scratch for every new class.

Correct answer: B

Explanation:

Few-shot learning aims to generalise to new classification tasks using very few labelled examples. It may use previously learned representations, an adaptable parameter initialisation, or a similarity-based classifier.

- A is incorrect because high training accuracy does not guarantee generalisation to unseen classes.
- C describes neither few-shot learning nor its general objective; few-shot methods still use a small labelled support set.
- D is incorrect because training an entire network from scratch for every new class is not required and may be impractical.

Exam tip: Focus on learning how to generalise from limited examples, not memorising the original training classes.

### Question 2. Support set versus query set

In a five-way, three-shot classification episode, which statements are correct? Select all that apply.

- A. The support set contains three labelled examples per class.
- B. The query set is used to measure how well the task-specific classifier generalises.
- C. The query set must contain exactly the same images as the support set.
- D. During ProtoNet meta-training, the query loss can update the backbone.

Correct answers: A, B and D

Explanation:

- A is correct: three-shot means three labelled support examples per class.
- B is correct: query examples evaluate classification on additional examples from the episode.
- C is incorrect: support and query examples are distinct samples in the standard episodic setup.
- D is correct: query cross-entropy propagates gradients through the query embeddings and the support-derived prototypes, updating the backbone.

The support set constructs or adapts the classifier, while the query set measures its performance.

### Question 3. Calculating the size of an episode

A six-way, four-shot task has eight query examples per class. How many examples are in the episode altogether?

- A. 24
- B. 48
- C. 72
- D. 96

Correct answer: C — 72

Explanation:

The support count is

$$
N_{\text{support}}=6\times4=24.
$$

The query count is

$$
N_{\text{query}}=6\times8=48.
$$

Therefore,

$$
N_{\text{total}}=24+48=\boxed{72}.
$$

Remember that ways refers to the number of classes, while shots refers to the number of labelled support examples per class.

### Question 4. Meta-training, meta-validation and meta-testing

Which procedure provides the most reliable estimate of generalisation to unseen classes?

- A. Use meta-test accuracy to select the best checkpoint after every epoch.
- B. Use meta-training accuracy to select the checkpoint and report it as test performance.
- C. Select the checkpoint using meta-validation tasks, then evaluate it on reserved meta-test tasks.
- D. Use the same episodes for training, validation and testing to ensure a fair comparison.

Correct answer: C

Explanation:

Each split serves a different purpose:

- Meta-training: Learns model parameters from sampled training tasks.
- Meta-validation: Guides hyperparameter tuning, checkpoint selection and other model-selection decisions.
- Meta-testing: Estimates final generalisation performance on reserved tasks.

Repeatedly selecting checkpoints based on test accuracy leaks information from the test set into model selection. This can cause overfitting to the test tasks and produce an overly optimistic estimate of performance.

Exam tip: The test set is for final evaluation, not repeated model selection.

## Part B — Supervised Baselines and Evaluation Metrics

### Question 5. Understanding the supervised baseline

A baseline model is trained on the meta-training classes. During few-shot evaluation, its backbone is frozen and a new classification head is trained on the support set. Which statements are correct? Select all that apply.

- A. The backbone extracts features from images.
- B. The original classification head is automatically suitable for unseen classes.
- C. The new classification head learns to distinguish the classes in the current episode.
- D. The experiment tests whether supervised pre-training produces transferable representations.

Correct answers: A, C and D

Explanation:

The backbone converts each image into a feature vector. The new head learns a mapping from those features to the classes in the current few-shot task.

The original head was trained for the original class vocabulary, so it is not automatically suitable for unseen classes. Freezing the backbone isolates the usefulness of its learned features: if the new head performs well, the representations are likely transferable.

This is an important baseline because it helps determine whether episodic meta-learning offers an advantage over ordinary supervised representation learning.

### Question 6. Top-1 accuracy versus top-5 accuracy

A model classifies 100 query images. The correct class has the highest predicted score for 72 images and appears among the five highest-scoring classes for 91 images.

What are the top-1 and top-5 accuracies, respectively?

- A. 72% and 91%
- B. 91% and 72%
- C. 72% and 72%
- D. 91% and 100%

Correct answer: A — 72% and 91%

Explanation:

Top-1 accuracy is the proportion of examples whose highest-scoring predicted class is correct:

$$
\text{Top-1 accuracy}=\frac{72}{100}\times100=72\%.
$$

Top-5 accuracy is the proportion for which the correct class appears anywhere among the five highest-scoring predictions:

$$
\text{Top-5 accuracy}=\frac{91}{100}\times100=91\%.
$$

Top-5 accuracy is usually at least as high as top-1 accuracy because it allows more than one candidate prediction to count as correct.

### Question 7. Cross-entropy loss

A classifier assigns probabilities of 0.80, 0.15 and 0.05 to three classes. The true class is the second class.

What is the cross-entropy loss for this example, using the natural logarithm?

- A. $-\ln(0.80)\approx0.223$
- B. $-\ln(0.15)\approx1.897$
- C. $-\ln(0.05)\approx2.996$
- D. $-\ln(1)=0$

Correct answer: B — approximately 1.897

Explanation:

For a single example, categorical cross-entropy is

$$
L=-\ln(p_{\text{true}}).
$$

The true class is the second class, whose predicted probability is $0.15$. Therefore,

$$
L=-\ln(0.15)\approx\boxed{1.897}.
$$

Cross-entropy penalises the model for assigning low probability to the correct class. If the model assigns a probability close to one to the true class, the loss approaches zero.

Important distinction: Accuracy measures whether predictions are correct; cross-entropy also measures how much probability the model assigns to the correct answer.

## Part C — Reptile and Optimisation

### Question 8. Calculating a Reptile update

Reptile uses the update rule

$$
\theta_{\text{new}}=\theta+\epsilon(\theta'-\theta).
$$

Suppose $\theta=4$, $\theta'=10$ and $\epsilon=0.3$. What is the updated parameter?

- A. 4.3
- B. 5.8
- C. 7.0
- D. 10.0

Correct answer: B — 5.8

Explanation:

Substitute the values:

$$
\begin{aligned}
\theta_{\text{new}}
&=4+0.3(10-4)\\
&=4+0.3(6)\\
&=4+1.8\\
&=\boxed{5.8}.
\end{aligned}
$$

The update moves the meta-parameter 30% of the way from 4 toward the task-adapted parameter 10.

### Question 9. Interpreting the Reptile meta-step size

Which statements about the outer-loop meta-step size $\epsilon$ are correct? Select all that apply.

- A. If $\epsilon=0$, the meta-parameter does not change.
- B. If $\epsilon=1$, the updated parameter equals $\theta'$.
- C. The outer-loop meta-step size must equal the inner-loop learning rate.
- D. For $0<\epsilon<1$, the update moves partway toward $\theta'$.

Correct answers: A, B and D

Explanation:

The Reptile update is a weighted movement toward the adapted parameters.

- A is correct because the update becomes $\theta_{\text{new}}=\theta$.
- B is correct because $\theta+(\theta'-\theta)=\theta'$.
- C is incorrect because the inner-loop learning rate controls task adaptation, while $\epsilon$ controls the outer meta-update.
- D is correct because the update interpolates between the original and adapted parameters.

These two learning rates control different optimisation processes and should be treated as separate hyperparameters.

### Question 10. What does Reptile learn?

Which statement best describes Reptile?

- A. It learns fixed class prototypes and never changes model parameters during training.
- B. It learns an initialisation that can adapt effectively to new tasks through gradient-based fine-tuning.
- C. It learns only a fixed linear classifier for the original training classes.
- D. It uses the query set exclusively to construct mean class prototypes.

Correct answer: B

Explanation:

Reptile is a gradient-based meta-learning algorithm. It repeatedly samples tasks, fine-tunes the model on each task, and moves the meta-initialisation toward the task-adapted parameters.

The objective is to find an initialisation that can adapt effectively to new tasks with relatively little task-specific training.

Options A and D confuse Reptile with prototype-based classification, while C describes a fixed supervised classifier rather than a meta-learned initialisation.

### Question 11. Inner-loop versus outer-loop optimisation

A Reptile experiment uses an inner-loop learning rate of $0.01$ and an outer-loop meta-step size of $0.2$. Which interpretation is correct?

- A. Both values control the same gradient-descent step.
- B. The inner-loop rate controls task-specific fine-tuning; the outer-loop rate controls the movement of the meta-initialisation.
- C. The outer-loop rate must be smaller than the inner-loop rate.
- D. The inner-loop rate determines the number of classes in an episode.

Correct answer: B

Explanation:

The inner loop adapts the model to an individual task using gradient-based optimisation. The outer loop updates the shared meta-initialisation based on the adapted parameters.

There is no general requirement that the outer-loop rate must be smaller than the inner-loop rate. Their appropriate values depend on the model, task distribution and experimental setup.

## Part D — Prototypical Networks (ProtoNet)

### Question 12. Calculating a class prototype

A class has three support embeddings:

$$
z_1=(2,1),\quad z_2=(4,3),\quad z_3=(3,5).
$$

What is the class prototype?

- A. $(3,3)$
- B. $(9,9)$
- C. $(3,2)$
- D. $(2,3)$

Correct answer: A — $(3,3)$

Explanation:

ProtoNet calculates the coordinate-wise mean:

$$
c=\frac{z_1+z_2+z_3}{3}.
$$

Thus,

$$
c=\left(
\frac{2+4+3}{3},
\frac{1+3+5}{3}
\right)
=\boxed{(3,3)}.
$$

The prototype summarises the support embeddings for that class. A new query is classified by comparing its embedding with the prototypes of all episode classes.

### Question 13. Nearest-prototype classification

A query embedding has the following Euclidean distances to three class prototypes:

$$
d=[2,5,1].
$$

Which class should ProtoNet predict?

- A. Class 1
- B. Class 2
- C. Class 3
- D. The class with the largest distance

Correct answer: C — Class 3

Explanation:

ProtoNet predicts the class with the smallest distance to the query embedding.

The distances are:

- Class 1: 2
- Class 2: 5
- Class 3: 1

Since $1$ is the smallest distance, the third prototype is the nearest.

ProtoNet commonly converts distances into logits by negating them. This makes the nearest prototype receive the largest logit.

### Question 14. Why use negative distances as logits?

Suppose ProtoNet calculates Euclidean distances and uses

$$
z_k=-d_k.
$$

Why is the negative sign necessary in this formulation?

- A. To ensure that all logits are positive.
- B. Because larger distances indicate better class matches.
- C. To make smaller distances correspond to larger logits.
- D. To eliminate the need for a softmax function.

Correct answer: C

Explanation:

Classification logits are interpreted such that a larger logit indicates stronger preference for a class. However, distances work in the opposite direction: smaller distances indicate more similar embeddings.

Negating the distances reverses the ranking. For example, distances $[1,3,2]$ become logits $[-1,-3,-2]$, so the nearest prototype has the largest logit.

Negative logits do not have to be positive, and the softmax function can still be applied normally.

### Question 15. Tensor shapes in ProtoNet

A five-way task contains 25 support examples, 75 query examples and a 64-dimensional embedding. Which statements are correct? Select all that apply.

- A. Support embeddings have shape `(25, 64)`.
- B. Prototypes have shape `(5, 64)`.
- C. Query logits have shape `(5, 75)`.
- D. Query labels have shape `(75,)`.
- E. Query logits have shape `(75, 5)`.

Correct answers: A, B, D and E

Explanation:

The expected shapes are:

| Tensor | Shape |
| --- | --- |
| Support embeddings | `(25, 64)` |
| Prototypes | `(5, 64)` |
| Query embeddings | `(75, 64)` |
| Query logits | `(75, 5)` |
| Query labels | `(75,)` |

There is one prototype per class, and each prototype has 64 features. Each of the 75 queries receives five class scores, giving a logits matrix of shape `(75, 5)`.

The first dimension of the logits represents query examples; the second represents classes. This ordering is important when computing cross-entropy loss.

### Question 16. Temperature scaling

A query has distances

$$
d=[1,3,2]
$$

to three prototypes. ProtoNet calculates logits using

$$
z_k=-\frac{d_k}{T}.
$$

If $T=2$, which logits are correct?

- A. $[0.5,1.5,1.0]$
- B. $[-2,-6,-4]$
- C. $[-0.5,-1.5,-1.0]$
- D. $[-1,-3,-2]$

Correct answer: C

Explanation:

Divide each negative distance by the temperature:

$$
z=\left[-\frac12,-\frac32,-\frac22\right].
$$

Therefore,

$$
\boxed{z=[-0.5,-1.5,-1.0]}.
$$

Temperature affects the sharpness of the probability distribution after softmax. Increasing a positive temperature makes the logits less extreme, while decreasing it makes them more extreme.

Because $T>0$, dividing all negative distances by $T$ preserves the ranking of the logits. The nearest prototype remains the predicted class, assuming the embeddings and distances do not change.

### Question 17. ProtoNet meta-training

Which sequence correctly describes one ProtoNet meta-training episode?

- A. Sample a task → train a new linear head → discard the backbone → evaluate on support examples.
- B. Sample a task → compute support embeddings and prototypes → compute query logits → calculate query loss → backpropagate and update the backbone.
- C. Sample a task → compute prototypes from query labels → evaluate on support examples → update only the prototypes.
- D. Train on all query labels first → construct prototypes → calculate support loss.

Correct answer: B

Explanation:

A standard ProtoNet episode proceeds as follows:

1. Sample classes and support/query examples.
2. Pass the support images through the backbone.
3. Compute one mean embedding per class.
4. Embed the query images and calculate their distances to the prototypes.
5. Convert distances to logits and calculate cross-entropy using the query labels.
6. Backpropagate the loss and update the backbone parameters.

The query loss encourages the learned embedding space to support correct classification of examples that were not used to construct the prototypes.

### Question 18. Why calculate loss on query examples?

Why does standard ProtoNet meta-training calculate the classification loss on query examples rather than relying on support examples alone?

- A. Query examples are always easier to classify than support examples.
- B. Support examples do not have labels.
- C. Query loss evaluates whether support-derived prototypes generalise to additional examples and trains the embedding space accordingly.
- D. Query loss prevents the backbone from receiving gradients.

Correct answer: C

Explanation:

Support examples construct the prototypes, while query examples test whether those prototypes represent the classes effectively.

The query loss provides a learning signal that encourages the backbone to produce embeddings that generalise beyond the examples used to calculate the prototypes.

Gradients can flow through both the query embeddings and the support-derived prototypes. Consequently, the backbone learns from the complete episodic classification process.

## Part E — Comparing Methods and Diagnosing Errors

### Question 19. Baseline versus Reptile versus ProtoNet

Which comparison is correct for the practical setup described in the exam?

- A. The baseline freezes its backbone and trains a new head; Reptile fine-tunes parameters; ProtoNet calculates support prototypes without test-time gradient updates.
- B. The baseline uses prototypes; Reptile uses a fixed original head; ProtoNet fine-tunes all parameters during testing.
- C. All three methods train a new linear head during every test episode.
- D. Reptile and ProtoNet use identical adaptation procedures because both are meta-learning methods.

Correct answer: A

Explanation:

The three methods differ in how they use the support set.

| Method | What it learns | Adaptation during testing |
| --- | --- | --- |
| Supervised baseline | Transferable features and an original classifier | Train a new head with the backbone frozen |
| Reptile | An adaptable parameter initialisation | Fine-tune model parameters |
| ProtoNet | An embedding space suitable for prototype classification | Compute mean support prototypes |

Reptile and ProtoNet are both meta-learning methods, but they use different mechanisms. Reptile relies on gradient-based adaptation; ProtoNet relies on distances in a learned embedding space.

### Question 20. Debugging invalid class labels

A five-way ProtoNet produces logits of shape `(75, 5)`. However, the query labels are

`[12, 27, 41, 58, 93, ...]`

and cross-entropy raises an error. What is the most likely cause and the correct fix?

- A. The backbone embedding dimension is too large; reduce it to five.
- B. The query set has too many examples; reduce it to five.
- C. The labels are global dataset class IDs; consistently remap them to task-local indices from 0 to 4.
- D. Cross-entropy cannot be used with ProtoNet.

Correct answer: C

Explanation:

The five columns of the logits correspond to the five classes in the current episode. For standard integer-target cross-entropy, each target must therefore be an index in the range 0–4.

Global dataset IDs such as 12 and 93 do not directly correspond to these logit columns.

For example, a task could use this mapping:

| Global class ID | Task-local index |
| --- | --- |
| 12 | 0 |
| 27 | 1 |
| 41 | 2 |
| 58 | 3 |
| 93 | 4 |

Apply the same mapping consistently to support labels and query labels, and ensure that the prototype row order matches the task-local class indices.

The problem is not the number of queries or the embedding dimension. Cross-entropy is an appropriate loss for ProtoNet.

### Question 21. Detecting test-set leakage

A researcher evaluates ProtoNet on meta-test tasks after every epoch and selects the checkpoint with the highest meta-test accuracy. Why is this method problematic?

- A. Meta-test accuracy cannot be calculated for ProtoNet.
- B. Repeated checkpoint selection uses test information for model selection and can produce an optimistically biased performance estimate.
- C. ProtoNet requires the test set to contain the original training classes.
- D. Checkpoint selection should be based exclusively on training loss.

Correct answer: B

Explanation:

The meta-test set should provide an independent estimate of final generalisation. Repeatedly selecting the best checkpoint based on test accuracy indirectly adapts the model-selection process to the test tasks.

A better procedure is to:

1. Train the model on meta-training tasks.
2. Evaluate checkpoints and tune hyperparameters on meta-validation tasks.
3. Select the final configuration using validation results.
4. Evaluate that selected configuration on reserved meta-test tasks.

This helps preserve the independence of the final test estimate.

### Question 22. Designing a fair experiment

You want to compare a supervised baseline, Reptile and ProtoNet on an unseen-class few-shot classification benchmark. Which experimental design is strongest?

- A. Train each model on different class splits and compare their training accuracy.
- B. Select the best model using test accuracy and report only its highest score.
- C. Use comparable held-out evaluation tasks, a clearly defined few-shot protocol, validation-based model selection, and query accuracy on reserved test tasks.
- D. Evaluate each method on whichever classes are easiest for that method.

Correct answer: C

Explanation:

A fair comparison requires a consistent evaluation protocol. The methods should be evaluated under comparable conditions, including the same definition of the few-shot task and appropriately separated class splits.

A strong experimental design should:

- Use disjoint meta-training, meta-validation and meta-test classes when evaluating generalisation to unseen classes.
- Specify the number of ways, shots and query examples.
- Use validation tasks for checkpoint and hyperparameter selection.
- Report query performance on reserved test tasks.
- Apply consistent evaluation procedures across methods.

Where appropriate, also report variability across sampled episodes or multiple random seeds. This helps determine whether an observed performance difference is robust rather than an artefact of a particular sample of tasks.

## Part F — Advanced Application Questions

These final questions combine multiple concepts and are useful for preparing for more challenging exam questions.

### Question 23. Gradient flow through prototypes

During ProtoNet meta-training, the query loss depends on the query embeddings and on prototypes calculated from support embeddings. Which statement is correct?

- A. Gradients flow only through query embeddings because support embeddings are not part of the loss.
- B. Gradients can flow through both query embeddings and support-derived prototypes, updating the shared backbone.
- C. Gradients update only the class labels.
- D. ProtoNet cannot use gradient-based optimisation because it has no learned linear classification head.

Correct answer: B

Explanation:

The prototype is calculated from the backbone outputs of support examples:

$$
c_k=\frac{1}{|S_k|}\sum_{x_i\in S_k}f_\theta(x_i).
$$

The query loss depends on the distances between query embeddings and these prototypes. During training, backpropagation can therefore propagate through both the query branch and the support-prototype branch.

Although ProtoNet does not require a learned linear classification head, its backbone is trainable. The query loss updates the embedding function so that future episodes become easier to classify.

Key distinction: No test-time gradient updates does not mean no gradient-based training.

### Question 24. Interpreting few-shot generalisation results

A supervised baseline achieves 98% accuracy on its original training classes but 65% average query accuracy on held-out five-way tasks. ProtoNet achieves 72% average query accuracy on the same held-out evaluation tasks.

Which conclusion is best supported?

- A. The baseline is necessarily better because its training accuracy is higher.
- B. ProtoNet performs better on the evaluated few-shot tasks, but further evidence may be needed to establish the robustness of the difference.
- C. ProtoNet must have data leakage because its test accuracy exceeds 70%.
- D. Training accuracy is the only valid metric for comparing few-shot methods.

Correct answer: B

Explanation:

The most relevant comparison is performance on the held-out few-shot tasks, not accuracy on the original training classes.

ProtoNet's 72% query accuracy exceeds the baseline's 65% by seven percentage points on the stated evaluation. This supports the conclusion that ProtoNet performs better under that protocol.

However, to judge the reliability of the difference, examine variability across episodes, random seeds and other relevant experimental conditions. The result alone does not establish statistical significance or guarantee superior performance on every task distribution.

### Question 25. Choosing the appropriate adaptation mechanism

You have only five labelled examples per new class at test time. You want to minimise test-time computation and avoid gradient updates. Which method most directly satisfies this requirement in the practical setup?

- A. Reptile, because it requires gradient-based fine-tuning on the support set.
- B. The supervised baseline, because its new classification head requires no training.
- C. ProtoNet, because it constructs prototypes from support embeddings and classifies queries using distances.
- D. All three methods, because they have identical test-time procedures.

Correct answer: C

Explanation:

ProtoNet requires only a forward pass through the backbone, calculation of the class prototypes and distance-based classification of the query examples. It does not require test-time gradient updates.

By contrast, Reptile performs support-set fine-tuning, while the specified supervised baseline trains a fresh classification head for each task.

ProtoNet is therefore the most direct choice when avoiding test-time optimisation is a key requirement, although the best overall method still depends on accuracy, latency, memory and the characteristics of the task distribution.

## Final revision: the concepts you should know by heart

| Concept | Essential fact |
| --- | --- |
| Few-shot learning | Recognise new classes from few labelled examples |
| Episode | A sampled task with support and query sets |
| Support set | Constructs prototypes or supports task adaptation |
| Query set | Evaluates classification and supplies the episodic training loss |
| Meta-validation | Selects checkpoints and hyperparameters |
| Meta-testing | Estimates final generalisation |
| Supervised baseline | Tests transferability of supervised features |
| Reptile | Learns an initialisation for rapid gradient-based adaptation |
| Reptile update | $\theta_{\text{new}}=\theta+\epsilon(\theta'-\theta)$ |
| ProtoNet | Learns an embedding space for prototype classification |
| Prototype | Mean of the support embeddings for a class |
| ProtoNet logits | Negative distances, optionally scaled by temperature |
| ProtoNet loss | Query-set cross-entropy |
| Label debugging | Map global class IDs to consistent task-local indices |
| Tensor shapes | For $Q$ queries and $N$ classes, logits have shape `(Q, N)` |
| Experimental design | Select using validation tasks; reserve test tasks for final evaluation |

Suggested study method: First answer all 25 questions without looking at the explanations. Then revisit every incorrect answer and explain, in your own words, why the correct option is right and each distractor is wrong. Pay particular attention to the Reptile update, prototype calculations, tensor shapes and label remapping, because these concepts require both conceptual understanding and accurate application.
