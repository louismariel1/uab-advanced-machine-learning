# Advanced Machine Learning — Interactive Practice Exam 6

Topic: Few-Shot Learning and Meta-Learning Based on: Practical 6 and the official Advanced Machine Learning exam format.

This practice exam follows the real exam's emphasis on conceptual understanding, algorithm intuition, mathematical reasoning, and interpreting implementation details rather than writing code from scratch.

- All questions will be visible on one page.
- The exam mixes multiple-choice questions and free-written answers.
- Answer every question before submitting.
- When you submit, you'll receive an estimated score, question-by-question feedback, and model answers with explanations.
- The questions cover the supervised baseline, support/query sets, Reptile, Prototypical Networks, episodic training, evaluation, and implementation debugging.

The practice exam is worth 35 points, like the collection of past exam questions in your official guide. The automatic grading of written answers is an approximation; use the model answers and explanations to assess technical accuracy.

# Practice exam

10 questions · 35 points · All questions shown

Multiple choice

## 8 points

4 questions

Written answers

## 27 points

6 questions

Instructions

- Answer the four multiple-choice questions by selecting one option.
- Write your own explanations for Questions 5–10. Include equations or pseudocode where appropriate.
- The written-answer scoring is a rough automated keyword-based estimate, not a substitute for human marking.
- Submit only when you are ready to see your score and the model answers.

## Section A — Multiple choice

2 points per question · Select one answer each.

1. What is the main objective of few-shot learning? (2 points)

A. Learn a fixed classifier on many labelled examples from the same classes.

B. Improve training accuracy by increasing the number of output neurons.

c C. Recognise new classes using only a few labelled examples per class.

D. Avoid using labelled examples altogether.

Correct · 2/2

Model answer: C. Recognise new classes using only a few labelled examples per class.

Few-shot learning evaluates adaptation or classification on new tasks with only a few labelled support examples.

2. In episodic few-shot learning, what is the main distinction between the support and query sets? (2 points)

A. Both sets are used only to update the backbone.

b B. Support examples define or adapt the classifier; query examples measure performance.

C. The query set defines prototypes and the support set calculates test accuracy.

D. Support and query must contain exactly the same images.

Correct · 2/2

Model answer: B. Support examples define or adapt the classifier; query examples measure performance.

The support set defines the prototypes; the query set evaluates classification and supplies the training loss.

3. Which description best characterises the Reptile meta-update? (2 points)

a A. Fine-tune on a task, then move the initial parameters toward the adapted parameters.

B. Average predictions from many unrelated models.

C. Calculate class prototypes without training a backbone.

D. Freeze every parameter and optimise only a fixed classifier.

Correct · 2/2

Model answer: A. Fine-tune on a task, then move the initial parameters toward the adapted parameters.

Reptile moves its meta-initialisation toward parameters obtained after task-specific fine-tuning.

4. How does a Prototypical Network convert distances into classification logits? (2 points)

A. Use positive distances because larger logits should indicate greater distance.

B. Use cosine learning-rate scheduling instead of a distance metric.

C. Train a separate permanent output neuron for every possible test class.

d D. Negate the distances, so the nearest prototype has the largest logit.

Correct · 2/2

Model answer: D. Negate the distances, so the nearest prototype has the largest logit.

ProtoNet uses negative distances as logits. Smaller distances become larger logits, making the nearest prototype the preferred class.

## Section B — Free-written answers

Answer in your own words. Explain why, not just what.

5. Prototypical Networks: constructing prototypes (3 points)

Explain how ProtoNet constructs a prototype for each class from the support set. If three support embeddings are (1, 2), (2, 3), and (3, 1), calculate their prototype.

how ProtoNet constructs a prototype = ProtoNet calculates the mean of the support embedding belonging to the class. (1, 2), (2, 3), and (3, 1), prototype c = ((1+2+3)/3, (2+3+1)/3)=(2, 2)

Estimated score

### 3/3

Model answer

Pass support images through the backbone to obtain embeddings. For each class, average the embeddings of its labelled support examples. Each mean vector is that class's prototype.A prototype is a class representative in embedding space, not a learned output neuron. For the numerical example, the prototype is ((1+2+3)/3, (2+3+1)/3) = (2, 2).

6. Reptile: explain the update rule (4 points)

Given $\theta \leftarrow \theta+\epsilon(\theta_i'-\theta)$, explain the meaning of each term and what happens when $\epsilon=1$.

θ = original metamodel parameters before task specific adaptation θ´ = corresponding parameter after inner-loop fine-tuning on a sample task ϵ = outer loop meta-step size. ϵ=1, θ = θ´ -\> no change in the original metamodel parameters

Estimated score

### 3/4

Model answer

Let θ be the initial meta-parameters and θ′ the parameters after fine-tuning on a sampled task. Reptile updates θ ← θ + ε(θ′ − θ), where ε controls how far the meta-parameters move toward the task-adapted parameters.Reptile does not simply replace the initialisation with the adapted model unless ε = 1. When ε = 1, the update sets θ equal to θ′ for that task.

7. ProtoNet meta-training algorithm (4 points)

Describe one training episode, from sampling the task to updating the model. Explain which set supplies the loss.

One training episode of ProtoNet; 1\. Sample a task from the meta-training classes. 2\. Pass the support images through the backbone. 3\. Calculate the mean embedding for each support class to construct the prototypes. 4\. Pass the query images through the same backbone. 5\. Calculate distances from query embeddings to the prototypes and convert them into logits. 6\. Calculate cross-entropy loss using the query labels. 7\. Backpropagate the loss and update the backbone parameters.

Estimated score

### 4/4

Model answer

Sample an episode, obtain its support and query sets, construct prototypes from the support embeddings, predict query logits, calculate cross-entropy against query labels, backpropagate the query loss, and update the backbone with the optimiser.The query loss trains the embedding backbone through both query embeddings and the support-derived prototypes.

8. ProtoNet: distances, logits, and temperature (5 points)

A query has distances \[1, 3, 2\] to three prototypes. Write its logits when temperature $T=2$. Explain why negative distances are used and what temperature changes.

logits = \[-0.5, -1.5, -1\] . why negative distances are used: Ordinary classifiers produce logits where larger values indicate a stronger support for a class. Distances have the opposite interpretation. smaller values indicate a better match. The closest prototype has the largest logit. Temperature control how sharply distinguish between the prototypes.

Estimated score

### 4/5

Model answer

For query embedding z and prototype cₖ, calculate Euclidean distance d(z,cₖ). The logit is −d(z,cₖ)/T, with positive temperature T. Negation ensures the closest prototype has the largest logit. Temperature changes the logit scale and probability sharpness, but not the nearest prototype when T is positive.Cross-entropy accepts logits directly; a separate softmax is not required. For the given distances, the logits are \[−0.5, −1.5, −1.0\]. The first prototype remains the predicted class.

9. Meta-testing, evaluation, and leakage (5 points)

Explain how the practical evaluates few-shot generalisation to unseen classes. Identify the main evaluation metric and explain why repeatedly selecting checkpoints using meta-test performance is problematic.

The function meta\_evaluate\_protonet() evaluates the model on the meta-test tasks without updating its weights. For each task, it: Obtains the support examples and their labels. Obtains the query examples and their labels. Constructs prototypes from the support set. Predicts the query labels. Calculates query cross-entropy loss and top-1 accuracy. Averages the results across the meta-test tasks.

Estimated score

### 3/5

Model answer

Evaluate on tasks containing classes held out from meta-training. For each task, use the support set to construct prototypes (ProtoNet) or fine-tune parameters (Reptile), then measure query accuracy. Average accuracy across tasks. Ideally, use meta-validation tasks for checkpoint selection and reserve meta-test tasks for final reporting to avoid test-set leakage.Repeatedly selecting checkpoints based on meta-test performance leaks information from the evaluation set into model selection.

10. Debugging and implementation reasoning (6 points)

ProtoNet training produces poor accuracy, and some queries are assigned to the wrong classes. Describe at least three implementation checks. Include expected tensor shapes for prototypes and query logits in a five-way task with 64-dimensional embeddings and 75 queries.

1\. Label encoding problem verification: The global labels must be remapped consistently to local labels. 2\. Risk of repeated meta-test checkpoint selection: Repeatedly selecting checkpoints based on meta-test accuracy leaks test information into model selection. 3\. Better procedure: Use separate meta-validation tasks to monitor performance, select checkpoints, and tune hyperparameters.

Estimated score

### 2/6

Model answer

Check that support labels and query labels use consistent task-local class indices; embeddings have shape (batch size, embedding dimension); prototypes have shape (number of classes, embedding dimension); and logits have shape (number of queries, number of classes). During training, clear gradients, calculate query cross-entropy, backpropagate, and step the optimiser. Check train/eval mode and verify that the correct model variable and checkpoint are used.Shape errors, mismatched label encodings, missing gradient updates, incorrect model variables, and inconsistent checkpoints can all cause poor results or runtime errors. The prototypes should have shape (5, 64), and the query logits should have shape (75, 5). Query labels should have shape (75,).

# Your exam results

Estimated total score

# 27.0 / 35

## 77.1%

Multiple choice: 8/8 · Written answers: 19.0/27

Good start — review missed concepts

## Multiple-choice answer review

Q1 — Correct

Your answer: C · Correct answer: C

Few-shot learning evaluates adaptation or classification on new tasks with only a few labelled support examples.

Q2 — Correct

Your answer: B · Correct answer: B

The support set defines the prototypes; the query set evaluates classification and supplies the training loss.

Q3 — Correct

Your answer: A · Correct answer: A

Reptile moves its meta-initialisation toward parameters obtained after task-specific fine-tuning.

Q4 — Correct

Your answer: D · Correct answer: D

ProtoNet uses negative distances as logits. Smaller distances become larger logits, making the nearest prototype the preferred class.

## Revision priorities

- Reptile: Understand the inner-loop adaptation and outer-loop interpolation.
- ProtoNet: Practise prototype calculation, tensor shapes, negative-distance logits, and query-loss backpropagation.
- Evaluation: Explain support/query separation, held-out classes, and meta-test leakage.
- Implementation: Check label encoding, gradient updates, model modes, and checkpoint variables.
   Retake exam

Important grading note: Multiple-choice answers are graded exactly. Written answers receive an estimated score based on matching key concepts and terms, so a correct answer phrased differently may be under-scored. The model answers provide the definitive revision reference.
