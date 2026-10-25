# Baseline vs Reptile vs ProtoNet — how they fit together

The key idea is that all three methods address the same overall problem: classifying new image categories from only a few labelled examples. They differ in how they learn useful features and how they use the few examples when faced with a new task.

- Baseline: learns useful image features through ordinary supervised training, then trains a new classifier for each task.
- Reptile: learns an initialisation that can be adapted quickly to new tasks through gradient-based fine-tuning.
- ProtoNet: learns an embedding space where a new class can be represented by the average embedding of its few labelled examples.

## 1. Which exam questions relate to each method?

The interactive exam contains questions that directly test one method and others that compare the methods or test concepts shared by all three.

| Exam question | Main method or concept |
| --- | --- |
| Q1. Objective of few-shot learning | All three |
| Q2. Support set vs query set | All three |
| Q3. Reptile update rule | Reptile |
| Q4. Negative distances as logits | ProtoNet |
| Q5. Constructing class prototypes | ProtoNet |
| Q6. Reptile equation and step size | Reptile |
| Q7. ProtoNet meta-training episode | ProtoNet |
| Q8. Distances, logits and temperature | ProtoNet |
| Q9. Meta-testing and evaluation leakage | All three |
| Q10. Debugging and tensor shapes | Mainly ProtoNet, with some general training concepts |

The baseline is covered most directly by the comparison and evaluation concepts in Q1, Q2 and Q9, and by the practical's separate baseline-training and fine-tuning procedures. None of the ten questions specifically asks you to explain the baseline's full training pipeline, so that is a useful extra topic to revise.

## 2. How each method works

Convolutional Neural Network for CIFAR-10 | Kaggle

## A. Baseline — supervised pre-training

Learn useful featuresThe model trains on labelled images from the meta-training classes, using a backbone and a classification head.

At test time

Freeze the backbone and train a new classifier head using the few labelled support examples.

Requires: labelled training data, a backbone, a classifier head and task-specific head training.

Does not require: episodic meta-training or gradient updates to the backbone during the practical's specified baseline evaluation.

Artificial Neural Networks: A Complete Beginner's Guide - Wasil Zafar

## B. Reptile — optimisation-based meta-learning

Learn to adapt quicklyRepeatedly sample tasks, fine-tune the backbone on each task, then move the meta-initialisation towards the adapted parameters.

At test time

Start from the learned initialisation and fine-tune on the new task's support set.

Requires: sampled training tasks, a trainable initialisation, inner-loop optimisation and a meta-step size.

Key operation: $\theta \leftarrow \theta+\epsilon(\theta_i'-\theta)$.

Prototypical Network — 原型網絡 - Sharon Peng - Medium

## C. ProtoNet — metric-based meta-learning

Learn to compare examplesTrain the backbone episodically so that query images can be classified using the mean embeddings of labelled support examples.

At test time

Calculate prototypes from the support set and classify queries by distance. No gradient-based fine-tuning is needed.

Requires: a backbone, support labels, prototype calculation, a distance metric and query-loss optimisation during meta-training.

Key operation: $z_k=-d(f_\theta(x),c_k)/T$.

These are three distinct approaches to few-shot learning: the baseline transfers supervised features, Reptile learns an adaptable initialisation, and ProtoNet learns a metric space for prototype-based classification.

OpenAI

+2

## 3. What does each method use or require?

| Component | Baseline | Reptile | ProtoNet |
| --- | --- | --- | --- |
| Convolutional backbone | Yes | Yes | Yes |
| Original fixed-class head during pre-training | Yes | No, in this practical's backbone-only meta-training | No |
| Support set | Yes | Yes | Yes |
| Query set | For evaluation | For evaluation/monitoring | For training loss and evaluation |
| Episodic meta-training | No | Yes | Yes |
| Fine-tunes backbone on each task | No, in the specified evaluation | Yes | No |
| Creates a new classifier head | Yes | Yes, during task adaptation | No |
| Calculates class prototypes | No | No | Yes |
| Uses gradients at meta-test time | For the new head | For task adaptation | No |
| Main learning objective | Supervised classification loss | Fast adaptation across tasks | Query classification loss using prototypes |

One important distinction: all three use a backbone to extract image features, but they train that backbone differently. The baseline learns features for its original classes; Reptile optimises its initial parameters for subsequent fine-tuning; ProtoNet optimises features for distance-based classification.

## 4. How are they related during the practical?

Think of the practical as a controlled comparison.

CIFAR image data

Split into meta-training, validation and test classes

Baseline

Supervised training

Reptile

Task adaptation

ProtoNet

Prototype learning

Evaluate on the same held-out few-shot tasks

Use each task's support examples, predict its query examples, and average query accuracy across tasks.

The baseline acts as the reference point. Reptile asks whether learning a good initialisation for adaptation improves performance. ProtoNet asks whether learning a good embedding space for prototype classification works better.

The comparison is meaningful when the methods use consistent task splits, evaluation episodes and metrics.

## 5. What should you remember for the exam?

Use this short decision rule:

- If the question mentions a backbone trained on many classes followed by a newly trained classifier head, think baseline.
- If it mentions fine-tuning on a task and moving the initial parameters towards the adapted parameters, think Reptile.
- If it mentions averaging support embeddings and classifying queries using negative Euclidean distances, think ProtoNet.
- If it mentions support/query sets, episodes, held-out classes or average query accuracy, it may apply to all three.

The three methods do not depend on one another in the sense that Reptile must be trained before ProtoNet. They are alternative approaches that can be trained and evaluated separately. A pre-trained baseline backbone can optionally initialise another method, but that is an experimental choice, not a requirement of Reptile or ProtoNet.
